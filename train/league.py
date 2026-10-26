"""Self-play league (M4): a pool of frozen policy snapshots + the FSMs, opponents sampled by PFSP.

PolicyPool keeps every snapshot's weights stacked in [capacity, ...] tensors and evaluates all envs in one
batched pass (each env gathers its opponent's weights), so the cost does not grow with the pool size.
League tracks the learner's win rate against every opponent (EMA over finished episodes) and samples
opponents with prioritized fictitious self-play: weight ∝ (1 − win_rate)^p, i.e. mostly the ones we lose to.
"""

from __future__ import annotations

import torch

from train.models import Actor


class PolicyPool:
    def __init__(self, capacity: int, obs_dim: int, hidden: list[int], device: str | torch.device, act_dim: int = 2):
        self.capacity, self.device = capacity, device
        sizes = [obs_dim, *hidden, act_dim]
        self.W = [torch.zeros(capacity, a, b, device=device) for a, b in zip(sizes[:-1], sizes[1:])]
        self.B = [torch.zeros(capacity, b, device=device) for b in sizes[1:]]
        self.mean = torch.zeros(capacity, obs_dim, device=device)
        self.std = torch.ones(capacity, obs_dim, device=device)
        self.clip = 10.0
        self.names: list[str] = []
        self.size = 0

    def put(self, slot: int, actor: Actor, name: str) -> None:
        linears = [m for m in actor.net if isinstance(m, torch.nn.Linear)]
        with torch.no_grad():
            for W, B, lin in zip(self.W, self.B, linears):
                W[slot] = lin.weight.T
                B[slot] = lin.bias
            self.mean[slot] = actor.norm.mean
            self.std[slot] = actor.norm.std()
        if slot < len(self.names):
            self.names[slot] = name
        else:
            self.names.append(name)
        self.size = len(self.names)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
        """obs [N,D], slot [N] (long) → deterministic actions [N,2] (same math as Actor.forward)."""
        x = ((obs - self.mean[slot]) / self.std[slot]).clamp(-self.clip, self.clip)
        for i, (W, B) in enumerate(zip(self.W, self.B)):
            x = torch.bmm(x.unsqueeze(1), W[slot]).squeeze(1) + B[slot]
            if i < len(self.W) - 1:
                x = torch.relu(x)
        return torch.tanh(x)


class League:
    """Opponent bookkeeping for the trainer. Opponent indices follow SumoTask.opp_names."""

    def __init__(self, opp_names: list[str], pool: PolicyPool, cfg: dict, device: str | torch.device):
        self.names, self.pool, self.cfg = opp_names, pool, cfg
        self.n_fixed = len(opp_names) - pool.capacity  # stationary + FSM variants
        self.win = torch.full((len(opp_names),), 0.5, device=device)  # EMA win rate of the learner vs each
        self.games = torch.zeros(len(opp_names), device=device)
        self.next_slot = 0

    def record(self, opp: torch.Tensor, won: torch.Tensor, drew: torch.Tensor) -> None:
        """Finished episodes: opponent index, win flag, draw flag (a draw counts as half)."""
        score = won.float() + 0.5 * drew.float()
        total = torch.zeros_like(self.win).index_add_(0, opp, score)
        count = torch.zeros_like(self.win).index_add_(0, opp, torch.ones_like(score))
        seen = count > 0
        batch_rate = total / count.clamp(min=1)
        alpha = 1 - (1 - self.cfg["win_ema"]) ** count  # one EMA step per episode
        self.win = torch.where(seen, self.win + alpha * (batch_rate - self.win), self.win)
        self.games += count

    def add_snapshot(self, actor: Actor, name: str) -> str:
        """Freeze the current actor into the pool; when full, overwrite the oldest non-anchor slot."""
        anchors = self.cfg["anchors"]
        cap = self.pool.capacity
        if self.pool.size < cap:
            slot = self.pool.size
        else:
            slot = anchors + (self.next_slot % (cap - anchors))
            self.next_slot += 1
        self.pool.put(slot, actor, name)
        idx = self.n_fixed + slot
        self.win[idx] = 0.5
        self.games[idx] = 0
        return f"pool{slot}"

    def state(self) -> dict:
        p = self.pool
        return {"W": p.W, "B": p.B, "mean": p.mean, "std": p.std, "names": list(p.names),
                "win": self.win, "games": self.games, "next_slot": self.next_slot}

    def load_state(self, s: dict) -> None:
        p = self.pool
        for dst, src in zip(p.W + p.B + [p.mean, p.std], s["W"] + s["B"] + [s["mean"], s["std"]]):
            dst.copy_(src)
        p.names = list(s["names"])
        p.size = len(p.names)
        self.win.copy_(s["win"])
        self.games.copy_(s["games"])
        self.next_slot = s["next_slot"]

    def weights(self) -> torch.Tensor:
        """PFSP over pool policies, a fixed share for the FSM/stationary set (so old skills are not forgotten)."""
        hard = (1 - self.win).clamp(min=0).pow(self.cfg["pfsp_power"]) + self.cfg["pfsp_floor"]
        w = torch.zeros_like(self.win)
        fixed = slice(0, self.n_fixed)
        pool = slice(self.n_fixed, self.n_fixed + self.pool.size)
        share = self.cfg["fixed_share"] if self.pool.size else 1.0
        w[fixed] = share * hard[fixed] / hard[fixed].sum()
        if self.pool.size:
            w[pool] = (1 - share) * hard[pool] / hard[pool].sum()
        return w
