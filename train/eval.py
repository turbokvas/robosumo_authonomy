"""Evaluate a policy against every opponent type: one match per env, deterministic actions (mean), optional video.

    uv run python -m train.eval --ckpt runs/<run>/ckpt_latest.pt [--matches 2048] [--video 6]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from sumo import physics
from sumo.config import get, load_spec
from sumo.env import DRAW, SumoTask
from sumo.render import Frame, write_video
from train.models import Actor


@torch.no_grad()
def evaluate(
    actor: Actor,
    spec: dict[str, Any],
    cfg: dict[str, Any],
    opponent: str,
    matches: int,
    device: str,
    seed: int = 12345,
    record: int = 0,
) -> dict[str, Any]:
    """Win/loss/draw rates of `actor` (robot 0) vs `opponent` over `matches` first episodes."""
    task = SumoTask(spec, cfg, matches, device, seed)
    task.set_opponents({opponent: 1.0})
    task._new_episodes(torch.ones(matches, dtype=torch.bool, device=device))
    result = torch.full((matches,), DRAW, dtype=torch.long, device=device)
    finished = torch.zeros(matches, dtype=torch.bool, device=device)
    frames: list[list[Frame]] = [[] for _ in range(record)]
    env = task.env
    k = 0
    while not finished.all():
        if record and k % 2 == 0:
            ctr = physics.centers(env.state, env.p)[:record].cpu().numpy()
            th = env.state.theta[:record].cpu().numpy().copy()
            out = physics.com_out_of_ring(env.state, env.p)[:record].cpu().numpy()
            lift = env.state.lift[:record].cpu().numpy().copy()
            t = env.t[:record].cpu().numpy()
            fin = finished[:record].cpu()
            for e in range(record):
                if not fin[e]:
                    frames[e].append(Frame(float(t[e]), ctr[e], th[e], out[e], lift[e]))
        _, done, info = task.step(actor(task.obs()))
        new = done & ~finished
        result = torch.where(new, info["winner"], result)
        finished |= done
        k += 1
    return {
        "win": (result == 0).float().mean().item(),
        "loss": (result == 1).float().mean().item(),
        "draw": (result == DRAW).float().mean().item(),
        "frames": frames,
        "ring_radius": env.p.ring_radius[:record].cpu().tolist(),
        "frame_dt": 2 * float(env.ctrl_dt[:record].mean()) if record else 0.0,
    }


def video(actor: Actor, spec: dict, cfg: dict, opponents: list[str], per_opponent: int, device: str, path: Path) -> Path:
    eps, titles, radii, dt = [], [], [], 0.02
    for opp in opponents:
        r = evaluate(actor, spec, cfg, opp, per_opponent, device, seed=777, record=per_opponent)
        eps += r["frames"]
        radii += r["ring_radius"]
        titles += [f"blue policy vs orange {opp}"] * per_opponent
        dt = r["frame_dt"]
    half = tuple(x / 2 for x in get(spec, "robot.footprint"))
    return write_video(eps, titles, radii, get(spec, "arena.edge_line_width"), half, path, round(1 / dt))


def load_actor(ckpt_path: str | Path, device: str) -> tuple[Actor, dict, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    spec, cfg = ckpt["spec"], ckpt["cfg"]
    actor = Actor(ckpt["obs_dim"], get(spec, "controller.policy.hidden")).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor, spec, cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--matches", type=int, default=2048)
    ap.add_argument("--video", type=int, default=0, help="matches per opponent type to record")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    actor, spec, cfg = load_actor(args.ckpt, args.device)
    names = ["stationary", *get(spec, "opponent.variants")]
    for opp in names:
        r = evaluate(actor, spec, cfg, opp, args.matches, args.device)
        print(f"{opp:>22}: win {r['win']:6.1%}  loss {r['loss']:6.1%}  draw {r['draw']:6.1%}")
    if args.video:
        path = Path(args.ckpt).with_suffix(".mp4")
        print(video(actor, spec, cfg, ["aggressive", "spinner", "random_walker"], args.video, args.device, path))


if __name__ == "__main__":
    main()
