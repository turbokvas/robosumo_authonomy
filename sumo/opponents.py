"""Hard-coded sumo FSM, vectorized over envs. Drives one robot slot from that robot's own sensors.

Classic logic for 3 front E18 + 4 corner line sensors:
    ESCAPE (highest priority): front line → back up, then spin away from the side that saw the line;
                               rear line → short forward burst. Push-through: while the centre E18 sees the
                               opponent, a front line means we are pushing it out, so keep going
                               (all variants except cautious_edge_waiter).
    ATTACK: centre E18 only → slight arc (attack_arc, side flips on every re-attack) so contact comes at an angle
            and the wedge gets under; side E18 → steer towards it; lost → keep turning
            towards the last seen side for `memory_s`, then SEARCH. Seeing it longer than attack_timeout
            (a clinch: no encoders, so time is all the MCU has) → short back-off and turn, re-attack at an angle
            (not for cautious_edge_waiter).
    SEARCH, per variant:
        aggressive            spin towards the last seen side, every burst_period a forward burst
        cautious_edge_waiter  the same logic, all speeds scaled down by cautious_scale (slow, waits for contact)
        spinner               spins one way all match, attacks on sight
        random_walker         random forward-biased wheel commands, ignores the E18 (only escapes edges)
PWM convention: [left, right] in [-1, 1]; spinning left = (−s, +s).
"""

from __future__ import annotations

from typing import Any

import torch

from sumo.config import get, rand_ranges

SEARCH, ATTACK, ESC_BACK, ESC_TURN, ESC_FWD, REPO_BACK, REPO_TURN = range(7)
PARAM_PREFIX = "opponent.fsm."


class FSM:
    def __init__(
        self,
        spec: dict[str, Any],
        n_envs: int,
        device: str | torch.device,
        variant: str | None = None,
        generator: torch.Generator | None = None,
    ):
        """variant=None: every match draws a variant uniformly from spec opponent.variants."""
        self.variants: list[str] = get(spec, "opponent.variants")
        self.fixed_variant = None if variant is None else self.variants.index(variant)
        self.weights: torch.Tensor | None = None  # sampling weights over variants on reset (None = uniform)
        self.ranges = {k[len(PARAM_PREFIX):]: v for k, v in rand_ranges(spec).items() if k.startswith(PARAM_PREFIX)}
        self.n, self.device, self.gen = n_envs, device, generator
        z = lambda dtype=torch.float32: torch.zeros(n_envs, device=device, dtype=dtype)  # noqa: E731
        self.variant = z(torch.long)
        self.prm = {k: z() for k in self.ranges}
        self.mode = z(torch.long)
        self.timer = z()
        self.esc_dir = z()
        self.last_dir = z()
        self.spin_dir = z()
        self.arc_dir = z()
        self.clock = z()          # time since reset, drives bursts
        self.engaged = z()        # continuous time with the opponent in sight
        self.wander = torch.zeros(n_envs, 2, device=device)
        self.wander_timer = z()
        self.reset()

    def _rand(self, *shape: int) -> torch.Tensor:
        return torch.rand(*shape, device=self.device, generator=self.gen)

    def _sign(self, n: int) -> torch.Tensor:
        return torch.where(self._rand(n) < 0.5, -1.0, 1.0)

    def reset(self, env_mask: torch.Tensor | None = None) -> None:
        if env_mask is None:
            env_mask = torch.ones(self.n, dtype=torch.bool, device=self.device)
        self.reset_idx(env_mask.nonzero().squeeze(-1))

    def reset_idx(self, idx: torch.Tensor) -> None:
        """New match for envs `idx` (long): fresh variant (unless fixed), parameters and state. No host syncs."""
        k = idx.numel()
        if k == 0:
            return
        if self.fixed_variant is None:
            w = torch.ones(len(self.variants), device=self.device) if self.weights is None else self.weights
            self.variant.index_copy_(0, idx, torch.multinomial(w, k, replacement=True, generator=self.gen))
        else:
            self.variant.index_fill_(0, idx, self.fixed_variant)
        for name, (lo, hi) in self.ranges.items():
            self.prm[name].index_copy_(0, idx, lo + (hi - lo) * self._rand(k))
        for t in (self.mode, self.timer, self.clock, self.engaged, self.wander_timer):
            t.index_fill_(0, idx, SEARCH if t is self.mode else 0)
        for t in (self.esc_dir, self.last_dir, self.spin_dir, self.arc_dir):
            t.index_copy_(0, idx, self._sign(k))

    def is_variant(self, name: str) -> torch.Tensor:
        return self.variant == self.variants.index(name)

    def act(self, ir: torch.Tensor, line: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """ir [N,3] bool (L, C, R), line [N,4] bool (FL, FR, RL, RR), dt [N] control period → pwm [N,2]."""
        P = self.prm
        L, C, R = ir.unbind(-1)
        FL, FR, RL, RR = line.unbind(-1)
        front_line, rear_line = FL | FR, RL | RR
        walker = self.is_variant("random_walker")
        cautious = self.is_variant("cautious_edge_waiter")
        scale = torch.where(cautious, P["cautious_scale"], torch.ones_like(dt))
        front_line = front_line & ~(C & ~cautious & ~walker)  # push-through
        self.clock += dt
        self.timer -= dt

        # --- escape / reposition transitions (line sensors win over everything)
        escaping = (self.mode == ESC_BACK) | (self.mode == ESC_TURN) | (self.mode == ESC_FWD)
        expired = self.timer <= 0
        mode = self.mode.clone()
        timer = self.timer.clone()
        to_turn = (mode == ESC_BACK) & (expired | rear_line)
        mode = torch.where(to_turn, ESC_TURN, mode)
        timer = torch.where(to_turn, P["escape_turn_s"], timer)
        repo_turn = (mode == REPO_BACK) & (expired | rear_line)
        mode = torch.where(repo_turn, REPO_TURN, mode)
        timer = torch.where(repo_turn, P["reposition_turn_s"], timer)
        done = ((mode == ESC_TURN) & ~to_turn & expired) | ((mode == ESC_FWD) & expired)
        done |= (mode == REPO_TURN) & ~repo_turn & expired
        mode = torch.where(done, SEARCH, mode)
        start_fwd = rear_line & ~front_line & ~escaping
        mode = torch.where(start_fwd, ESC_FWD, mode)
        timer = torch.where(start_fwd, P["escape_back_s"], timer)
        away = torch.where(FL & ~FR, -1.0, torch.where(FR & ~FL, 1.0, self.esc_dir))  # FL seen → spin right
        self.esc_dir = torch.where(front_line, away, self.esc_dir)
        mode = torch.where(front_line, ESC_BACK, mode)
        timer = torch.where(front_line, P["escape_back_s"], timer)

        # --- attack / memory (not for the random walker)
        free = mode < ESC_BACK
        sees = (L | C | R) & ~walker
        self.engaged = torch.where(sees & (mode == ATTACK), self.engaged + dt, torch.zeros_like(self.engaged))
        clinch = free & (self.engaged > P["attack_timeout_s"]) & ~cautious
        self.esc_dir = torch.where(clinch, self._sign(self.n), self.esc_dir)
        self.arc_dir = torch.where(clinch, -self.esc_dir, self.arc_dir)  # turned away, arc back in from that side
        mode = torch.where(clinch, REPO_BACK, mode)
        timer = torch.where(clinch, P["reposition_back_s"], timer)
        self.engaged = torch.where(clinch, torch.zeros_like(self.engaged), self.engaged)
        free = mode < ESC_BACK
        side = torch.where(L & ~R, 1.0, torch.where(R & ~L, -1.0, self.last_dir))
        self.last_dir = torch.where(free & sees, side, self.last_dir)
        mode = torch.where(free & sees, ATTACK, mode)
        timer = torch.where(free & sees, P["memory_s"], timer)
        lost = free & ~sees & (mode == ATTACK) & (timer <= 0)
        mode = torch.where(lost, SEARCH, mode)
        self.mode, self.timer = mode, timer

        # --- commands
        atk = P["attack_speed"] * scale
        inner = P["steer_inner"]
        es = P["escape_speed"]
        spin = P["search_spin"] * scale

        def pair(l: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
            return torch.stack([l, r], -1)

        def spin_to(direction: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
            return pair(-direction * speed, direction * speed)

        half_inner = (1 + inner) / 2
        arc = 1 - P["attack_arc"]
        arc_cmd = torch.where((self.arc_dir > 0).unsqueeze(-1), pair(atk * arc, atk), pair(atk, atk * arc))
        attack_cmd = torch.where(
            C.unsqueeze(-1),
            torch.where(L.unsqueeze(-1) & ~R.unsqueeze(-1), pair(atk * half_inner, atk),
                        torch.where(R.unsqueeze(-1) & ~L.unsqueeze(-1), pair(atk, atk * half_inner), arc_cmd)),
            torch.where(L.unsqueeze(-1), pair(atk * inner, atk),
                        torch.where(R.unsqueeze(-1), pair(atk, atk * inner), spin_to(self.last_dir, spin))),
        )

        burst_phase = torch.remainder(self.clock, P["burst_period_s"]) < P["burst_s"]
        search_cmd = torch.where(
            (burst_phase & (self.is_variant("aggressive") | self.is_variant("cautious_edge_waiter"))).unsqueeze(-1),
            pair(atk, atk),
            spin_to(torch.where(self.is_variant("spinner"), self.spin_dir, self.last_dir), spin),
        )
        self.wander_timer -= dt
        new_wander = self.wander_timer <= 0
        fresh = self._rand(self.n, 2) * 1.3 - 0.3  # forward-biased, [-0.3, 1]
        self.wander = torch.where(new_wander.unsqueeze(-1), fresh, self.wander)
        self.wander_timer = torch.where(new_wander, P["wander_period_s"], self.wander_timer)
        search_cmd = torch.where(walker.unsqueeze(-1), self.wander, search_cmd)

        m = self.mode.unsqueeze(-1)
        return torch.where(
            (m == ESC_BACK) | (m == REPO_BACK), pair(-es, -es),
            torch.where((m == ESC_TURN) | (m == REPO_TURN), spin_to(self.esc_dir, es),
                        torch.where(m == ESC_FWD, pair(es, es),
                                    torch.where(m == ATTACK, attack_cmd, search_cmd))),
        )
