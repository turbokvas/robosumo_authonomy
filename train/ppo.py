"""PPO with an asymmetric actor-critic, CleanRL style: one file, everything on the GPU.

    uv run python -m train.ppo --config configs/m3.yaml

Robot 0 learns; robot 1 is an FSM variant or a stationary target, sampled per episode from the current curriculum
stage. Each checkpoint stores actor/critic/optimizer, both configs and the git hash.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from sumo.config import SPEC_PATH, get, load_spec
from sumo.env import SumoTask
from train.eval import evaluate, video
from train.models import Actor, Critic

ROOT = Path(__file__).resolve().parent.parent


def git_hash() -> str:
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=ROOT) != 0
        return h + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class WinWindow:
    """Ring buffer of the last `size` episode outcomes (1 = win) on the device."""

    def __init__(self, size: int, device: str):
        self.buf = torch.zeros(size, device=device)
        self.ptr = 0
        self.filled = 0

    def add(self, wins: torch.Tensor) -> None:
        n = wins.numel()
        if n == 0:
            return
        idx = (self.ptr + torch.arange(n, device=wins.device)) % self.buf.numel()
        self.buf[idx] = wins.float()
        self.ptr = (self.ptr + n) % self.buf.numel()
        self.filled = min(self.filled + n, self.buf.numel())

    def rate(self) -> float | None:
        return self.buf.mean().item() if self.filled == self.buf.numel() else None

    def clear(self) -> None:
        self.ptr = self.filled = 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--total-steps", type=int, default=None, help="override ppo.total_steps (smoke tests)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    spec = load_spec()
    pc, rc, cc, ec = cfg["ppo"], cfg["reward"], cfg["curriculum"], cfg["eval"]
    total = args.total_steps or pc["total_steps"]
    dev = args.device
    torch.manual_seed(cfg["seed"])

    run_dir = ROOT / "runs" / f"{cfg['run_name']}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True)
    shutil.copy(args.config, run_dir / "config.yaml")
    shutil.copy(SPEC_PATH, run_dir / "robot_spec.yaml")
    commit = git_hash()
    (run_dir / "git_hash.txt").write_text(commit + "\n")
    writer = SummaryWriter(str(run_dir))
    print(f"run: {run_dir}  (git {commit})")

    n, T = cfg["env"]["num_envs"], pc["rollout"]
    task = SumoTask(spec, cfg, n, dev, cfg["seed"])
    stages = cc["stages"]
    stage = 0
    task.set_opponents(stages[stage]["opponents"])
    task._new_episodes(torch.ones(n, dtype=torch.bool, device=dev))
    window = WinWindow(cc["window"], dev)

    obs_dim, cobs_dim = task.obs().shape[1], task.critic_obs().shape[1]
    actor = Actor(obs_dim, get(spec, "controller.policy.hidden"), init_log_std=pc["init_log_std"]).to(dev)
    critic = Critic(cobs_dim, pc["critic_hidden"]).to(dev)
    opt = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=pc["lr"], eps=1e-5)
    print(f"actor obs {obs_dim}, critic obs {cobs_dim}, actor params {sum(p.numel() for p in actor.parameters()):,}")

    buf = {
        "obs": torch.zeros(T, n, obs_dim, device=dev),
        "cobs": torch.zeros(T, n, cobs_dim, device=dev),
        "act": torch.zeros(T, n, 2, device=dev),
        "logp": torch.zeros(T, n, device=dev),
        "val": torch.zeros(T, n, device=dev),
        "rew": torch.zeros(T, n, device=dev),
        "done": torch.zeros(T, n, device=dev),
    }
    n_opp = len(task.opp_names)
    stats = torch.zeros(n_opp, 3, device=dev)  # per opponent: wins, losses, draws since last log
    ep_len_sum = torch.zeros((), device=dev)

    def save(tag: str) -> Path:
        path = run_dir / f"ckpt_{tag}.pt"
        torch.save({
            "actor": actor.state_dict(), "critic": critic.state_dict(), "opt": opt.state_dict(),
            "step": step, "stage": stage, "obs_dim": obs_dim, "cobs_dim": cobs_dim,
            "cfg": cfg, "spec": spec, "git": commit,
        }, path)
        shutil.copy(path, run_dir / "ckpt_latest.pt")
        return path

    step, next_eval, update = 0, ec["every_steps"], 0
    t_start = time.perf_counter()
    while step < total:
        task.shaping_scale = max(0.0, 1.0 - step / rc["shaping_anneal_steps"])
        t0 = time.perf_counter()
        with torch.no_grad():
            for t in range(T):
                obs, cobs = task.obs(), task.critic_obs()
                d = actor.dist(obs)
                a = d.sample()
                buf["obs"][t], buf["cobs"][t], buf["act"][t] = obs, cobs, a
                buf["logp"][t] = d.log_prob(a).sum(-1)
                buf["val"][t] = critic(cobs)
                r, done, info = task.step(a)
                buf["rew"][t], buf["done"][t] = r, done.float()
                if done.any():
                    w, o = info["winner"][done], info["opp_type"][done]
                    outcome = torch.where(w == 0, 0, torch.where(w == 1, 1, 2))
                    stats.index_put_((o, outcome), torch.ones_like(o, dtype=torch.float), accumulate=True)
                    ep_len_sum += info["ep_len"][done].sum()
                    window.add(w == 0)
            last_val = critic(task.critic_obs())

            adv = torch.zeros_like(buf["rew"])
            gae = torch.zeros(n, device=dev)
            for t in reversed(range(T)):
                next_val = last_val if t == T - 1 else buf["val"][t + 1]
                nonterm = 1 - buf["done"][t]
                delta = buf["rew"][t] + pc["gamma"] * next_val * nonterm - buf["val"][t]
                gae = delta + pc["gamma"] * pc["gae_lambda"] * nonterm * gae
                adv[t] = gae
            ret = adv + buf["val"]
        rollout_s = time.perf_counter() - t0

        flat = {k: v.flatten(0, 1) for k, v in buf.items()}
        flat_adv, flat_ret = adv.flatten(), ret.flatten()
        batch = T * n
        mb = batch // pc["minibatches"]
        for _ in range(pc["epochs"]):
            perm = torch.randperm(batch, device=dev)
            for i in range(pc["minibatches"]):
                idx = perm[i * mb:(i + 1) * mb]
                d = actor.dist(flat["obs"][idx])
                logp = d.log_prob(flat["act"][idx]).sum(-1)
                ratio = (logp - flat["logp"][idx]).exp()
                a_mb = flat_adv[idx]
                a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                pg = torch.max(-a_mb * ratio, -a_mb * ratio.clamp(1 - pc["clip"], 1 + pc["clip"])).mean()
                v_loss = 0.5 * (critic(flat["cobs"][idx]) - flat_ret[idx]).pow(2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pg + pc["vf_coef"] * v_loss - pc["ent_coef"] * ent
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([*actor.parameters(), *critic.parameters()], pc["max_grad_norm"])
                opt.step()
        actor.norm.update(flat["obs"])
        critic.norm.update(flat["cobs"])
        step += batch
        update += 1

        # curriculum
        rate = window.rate()
        if rate is not None and rate >= cc["promote_win_rate"] and stage < len(stages) - 1:
            stage += 1
            task.set_opponents(stages[stage]["opponents"])
            window.clear()
            print(f"[{step:,}] curriculum → {stages[stage]['name']} (window win rate {rate:.1%})")
            save(f"stage{stage}_{stages[stage]['name']}")

        if update % 10 == 0:
            sps = step / (time.perf_counter() - t_start)
            writer.add_scalar("perf/sps", sps, step)
            writer.add_scalar("perf/rollout_s", rollout_s, step)
            writer.add_scalar("loss/policy", pg.item(), step)
            writer.add_scalar("loss/value", v_loss.item(), step)
            writer.add_scalar("loss/entropy", ent.item(), step)
            writer.add_scalar("policy/std", actor.log_std.exp().mean().item(), step)
            writer.add_scalar("train/shaping_scale", task.shaping_scale, step)
            writer.add_scalar("train/stage", stage, step)
            writer.add_scalar("train/reward_per_step", buf["rew"].mean().item(), step)
            eps = stats.sum().item()
            if eps:
                writer.add_scalar("train/episode_len_s", ep_len_sum.item() / eps / 100, step)
                parts = []
                for i, name in enumerate(task.opp_names):
                    tot = stats[i].sum().item()
                    if tot:
                        wr = stats[i, 0].item() / tot
                        writer.add_scalar(f"train_win/{name}", wr, step)
                        parts.append(f"{name[:10]} {wr:4.0%}")
                print(f"[{step:>13,}] {sps:,.0f} sps  stage {stages[stage]['name']}  std {actor.log_std.exp().mean():.2f}  "
                      f"shaping {task.shaping_scale:.2f}  " + "  ".join(parts), flush=True)
            stats.zero_()
            ep_len_sum.zero_()

        if step >= next_eval or step >= total:
            next_eval += ec["every_steps"]
            actor.eval()
            res = {}
            for name in task.opp_names:
                r = evaluate(actor, spec, cfg, name, ec["matches"], dev)
                res[name] = r["win"]
                writer.add_scalar(f"eval_win/{name}", r["win"], step)
                writer.add_scalar(f"eval_draw/{name}", r["draw"], step)
            path = save(f"{step // 1_000_000:05d}M")
            print(f"[{step:,}] eval " + "  ".join(f"{k[:10]} {v:.0%}" for k, v in res.items()) + f"  → {path.name}", flush=True)
            try:
                video(actor, spec, cfg, ["aggressive", "spinner", "random_walker"], ec["video_matches"] // 3, dev,
                      run_dir / f"video_{step // 1_000_000:05d}M.mp4")
            except FileNotFoundError:
                print("ffmpeg not on PATH, skipping video")
            actor.train()

    writer.close()


if __name__ == "__main__":
    main()
