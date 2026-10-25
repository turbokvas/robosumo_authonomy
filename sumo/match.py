"""FSM vs FSM matches: win rates and videos.

    uv run python -m sumo.match --a aggressive --b random_walker --n 4096 --video 6
    uv run python -m sumo.match --matrix --n 2048
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sumo import physics
from sumo.config import get, load_spec
from sumo.env import DRAW, SumoEnv
from sumo.opponents import FSM
from sumo.render import VIDEO_DIR, Frame, write_video


def run_matches(
    spec: dict,
    variant_a: str | None,
    variant_b: str | None,
    n: int,
    device: str = "cuda",
    seed: int = 0,
    record: int = 0,
    frame_every: int = 2,
    randomize: bool = True,
) -> dict:
    """One match per env, robot 0 = FSM a, robot 1 = FSM b. Returns win/draw rates (+ frames of the first `record` envs)."""
    env = SumoEnv(spec, n, device=device, seed=seed, randomize=randomize)
    fsms = [FSM(spec, n, device, variant_a, env.gen), FSM(spec, n, device, variant_b, env.gen)]
    readings = env.readings()
    finished = torch.zeros(n, dtype=torch.bool, device=device)
    result = torch.full((n,), DRAW, dtype=torch.long, device=device)
    frames: list[list[Frame]] = [[] for _ in range(record)]
    k = 0
    while not finished.all():
        if record and k % frame_every == 0:
            ctr = physics.centers(env.state, env.p)[:record].cpu().numpy()
            th = env.state.theta[:record].cpu().numpy().copy()
            out = physics.com_out_of_ring(env.state, env.p)[:record].cpu().numpy()
            lift = env.state.lift[:record].cpu().numpy().copy()
            t = env.t[:record].cpu().numpy()
            for e in range(record):
                if not frames[e] or not finished[e]:
                    frames[e].append(Frame(float(t[e]), ctr[e], th[e], out[e], lift[e]))
        dt = env.ctrl_dt
        action = torch.stack([f.act(readings["ir"][:, i], readings["line"][:, i], dt) for i, f in enumerate(fsms)], 1)
        readings, done, winner = env.step(action)
        new = done & ~finished
        result = torch.where(new, winner, result)
        finished |= done
        k += 1
    return {
        "a_wins": (result == 0).float().mean().item(),
        "b_wins": (result == 1).float().mean().item(),
        "draws": (result == DRAW).float().mean().item(),
        "result": result.cpu(),
        "frames": frames,
        "ring_radius": env.p.ring_radius[:record].cpu().tolist(),
        "frame_dt": float(env.ctrl_dt[:record].mean()) * frame_every,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="aggressive")
    ap.add_argument("--b", default="random_walker")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--video", type=int, default=0, help="record the first K matches into one video")
    ap.add_argument("--matrix", action="store_true", help="all variants against each other")
    ap.add_argument("--nominal", action="store_true", help="no domain randomization (FSM params still random)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    spec = load_spec()

    if args.matrix:
        variants = get(spec, "opponent.variants")
        print(f"{'a \\ b':>22}" + "".join(f"{v[:12]:>14}" for v in variants) + "   (a wins / draws)")
        for a in variants:
            row = []
            for b in variants:
                r = run_matches(spec, a, b, args.n, args.device, randomize=not args.nominal)
                row.append(f"{r['a_wins']:6.0%} /{r['draws']:4.0%}")
            print(f"{a:>22}" + "".join(f"{c:>14}" for c in row))
        return

    r = run_matches(spec, args.a, args.b, args.n, args.device, record=args.video, randomize=not args.nominal)
    print(f"{args.a} vs {args.b}: a wins {r['a_wins']:.1%}, b wins {r['b_wins']:.1%}, draws {r['draws']:.1%}")
    if args.video:
        half = tuple(x / 2 for x in get(spec, "robot.footprint"))
        path = VIDEO_DIR / f"m2_{args.a}_vs_{args.b}.mp4"
        titles = [f"#{i}: blue {args.a} vs orange {args.b}" for i in range(args.video)]
        fps = round(1 / r["frame_dt"])
        write_video(r["frames"], titles, r["ring_radius"], get(spec, "arena.edge_line_width"), half, path, fps)
        print(path)


if __name__ == "__main__":
    main()
