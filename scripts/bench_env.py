"""Throughput of the full env step (physics substeps + sensors), with and without FSM brains.

    uv run python scripts/bench_env.py --n 4096
"""

import argparse
import time

import torch

from sumo.config import load_spec
from sumo.env import SumoEnv
from sumo.opponents import FSM


def bench(env: SumoEnv, fsm: FSM | None, steps: int) -> float:
    action = torch.zeros(env.n, 2, 2, device=env.device)
    r = env.readings()
    for k in range(steps + 20):
        if k == 20:
            torch.cuda.synchronize()
            t = time.perf_counter()
        if fsm is not None:
            action[:, 1] = fsm.act(r["ir"][:, 1], r["line"][:, 1], env.ctrl_dt)
        r, done, _ = env.step(action)
        env.reset(done)
    torch.cuda.synchronize()
    return env.n * steps / (time.perf_counter() - t)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()
    spec = load_spec()
    env = SumoEnv(spec, args.n, "cuda", seed=0)
    print(f"env only : {bench(env, None, args.steps):>12,.0f} env-steps/s")
    fsm = FSM(spec, args.n, "cuda", None, env.gen)
    print(f"env + FSM: {bench(env, fsm, args.steps):>12,.0f} env-steps/s")
