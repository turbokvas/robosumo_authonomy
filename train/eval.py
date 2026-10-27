"""Evaluate policies: win rates vs every opponent type, round-robin matrix + Elo, videos.

One match per env, deterministic actions (mean).

    uv run python -m train.eval --ckpt runs/<run>/ckpt_latest.pt [--matches 2048] [--video 6]
    uv run python -m train.eval --matrix results/m3/ckpt_00500M.pt runs/<run>/ckpt_00500M.pt ... [--matches 512]
    uv run python -m train.eval --ckpt NEW.pt --regression configs/regression.yaml [--baseline OLD.pt]
    ... --set robot.drive.wheel_diameter=0.08 --set robot.motors.strength=1.2   # evaluate on changed physics
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

from sumo import interface as iface
from sumo import physics
from sumo.config import apply_overrides, get, load_spec
from sumo.env import DRAW, SumoTask
from sumo.match import run_matches
from sumo.render import Frame, write_video
from train.league import PolicyPool
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
    pool: Any = None,
    sensors: str = "installed",
) -> dict[str, Any]:
    """Win/loss/draw rates of `actor` (robot 0) vs `opponent` over `matches` first episodes.

    opponent: "stationary", an FSM variant, or "pool<i>" (a frozen policy in `pool`, train.league.PolicyPool).
    """
    task = SumoTask(spec, cfg, matches, device, seed, pool=pool, sensors=sensors)
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


@torch.no_grad()
def _free_gpu() -> None:
    """Hand torch's cached blocks back before/after building an eval env next to the training one."""
    gc.collect()
    torch.cuda.empty_cache()


def evaluate_many(
    actor: Actor,
    spec: dict[str, Any],
    cfg: dict[str, Any],
    opponents: list[str],
    matches: int,
    device: str,
    pool: PolicyPool | None = None,
    seed: int = 4242,
    sensors: str = "installed",
    max_envs: int = 2048,
) -> dict[str, dict[str, float]]:
    """`matches` first episodes against each of `opponents` → {name: {win, loss, draw}}. Opponents are batched
    together up to max_envs worlds: an eval env sits next to the training env on an 8 GB card (M5 ran out of
    memory at a 9216-world 3D eval)."""
    per = max(1, max_envs // matches)
    if len(opponents) > per:
        out = {}
        for i in range(0, len(opponents), per):
            out.update(evaluate_many(actor, spec, cfg, opponents[i:i + per], matches, device, pool, seed + i, sensors,
                                     max_envs))
        return out
    n = matches * len(opponents)
    # a second (3D) env next to the training one: hand cached blocks back first, or the two allocators (torch,
    # Warp) can run an 8 GB card out of memory (M5 died this way at its 400M eval)
    _free_gpu()
    task = SumoTask(spec, cfg, n, device, seed, pool=pool, sensors=sensors)
    types = torch.tensor([task.opp_names.index(o) for o in opponents], device=device).repeat_interleave(matches)
    task.opp_type.copy_(types)
    task.fsm.variant.copy_((types - 1).clamp(0, task.n_fsm - 1))
    result = torch.full((n,), DRAW, dtype=torch.long, device=device)
    finished = torch.zeros(n, dtype=torch.bool, device=device)
    while not finished.all():
        _, done, info = task.step(actor(task.obs()))
        result = torch.where(done & ~finished, info["winner"], result)
        finished |= done
    out = {}
    for i, name in enumerate(opponents):
        r = result[i * matches:(i + 1) * matches]
        out[name] = {"win": (r == 0).float().mean().item(), "loss": (r == 1).float().mean().item(),
                     "draw": (r == DRAW).float().mean().item()}
    del task
    _free_gpu()
    return out


def elo(names: list[str], score: dict[tuple[str, str], float], anchor: list[str], iters: int = 3000) -> dict[str, float]:
    """Bradley-Terry fit of Elo ratings to pairwise scores (win + draw/2); mean of `anchor` entrants pinned to 1000."""
    idx = {n: i for i, n in enumerate(names)}
    pairs = torch.tensor([[idx[a], idx[b]] for a, b in score])
    s = torch.tensor(list(score.values())).clamp(1e-3, 1 - 1e-3)
    r = torch.zeros(len(names), requires_grad=True)
    opt = torch.optim.Adam([r], lr=0.05)
    for _ in range(iters):
        p = torch.sigmoid(r[pairs[:, 0]] - r[pairs[:, 1]])
        loss = -(s * p.log() + (1 - s) * (1 - p).log()).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    ratings = r.detach() * 400 / math.log(10)
    ratings += 1000 - ratings[[idx[a] for a in anchor]].mean()
    return {n: ratings[idx[n]].item() for n in names}


def regression(ckpt: str, suite_path: str, device: str, spec: dict, baseline: str | None = None) -> dict[str, Any]:
    """Fixed opponents × fixed seeds (configs/regression.yaml). With a baseline: the candidate must not lose
    ground against any opponent the baseline already beat (win ≥ beaten_at), beyond `tolerance`."""
    suite = yaml.safe_load(Path(suite_path).read_text(encoding="utf-8"))
    anchors = suite.get("anchor_checkpoints", [])
    opponents = suite["opponents"] + [f"pool{i}" for i in range(len(anchors))]
    pool = PolicyPool(max(len(anchors), 1), iface.OBS_DIM, list(iface.HIDDEN), device)
    for i, a in enumerate(anchors):
        pool.put(i, load_actor(a, device, spec)[0], a)
    names = suite["opponents"] + anchors

    def run(path: str) -> dict[str, float]:
        actor, _, cfg = load_actor(path, device, spec)
        tot = {n: 0.0 for n in names}
        for seed in suite["seeds"]:
            res = evaluate_many(actor, spec, cfg, opponents, suite["matches"], device, pool=pool, seed=seed,
                                sensors=suite.get("sensors", "installed"))
            for name, key in zip(names, opponents):
                tot[name] += res[key]["win"] / len(suite["seeds"])
        return tot

    cand = run(ckpt)
    out: dict[str, Any] = {"candidate": cand}
    if baseline:
        base = run(baseline)
        drops = {n: base[n] - cand[n] for n in names if base[n] >= suite["beaten_at"] and base[n] - cand[n] > suite["tolerance"]}
        out.update(baseline=base, regressions=drops, passed=not drops)
    return out


def extreme_eval(ckpt: str, suite_path: str, device: str, sim: str | None = None) -> dict[str, Any]:
    """Win rates on nominal hardware and on each extreme scenario (configs/eval_extreme.yaml), randomization off."""
    suite = yaml.safe_load(Path(suite_path).read_text(encoding="utf-8"))
    base = load_spec()
    out: dict[str, Any] = {}
    for name, overrides in suite["scenarios"].items():
        spec = apply_overrides(base, overrides)
        actor, spec, cfg = load_actor(ckpt, device, spec)
        cfg = {**cfg, "env": {**cfg["env"], "randomize": False, **({"sim": sim} if sim else {})}}
        res = evaluate_many(actor, spec, cfg, suite["opponents"], suite["matches"], device)
        out[name] = {"per_opponent": res, "win": sum(r["win"] for r in res.values()) / len(res)}
    nominal = out["nominal"]["win"]
    for name, r in out.items():
        r["drop"] = (nominal - r["win"]) / max(nominal, 1e-6)  # relative to nominal
    return out


def round_robin(ckpts: list[str], matches: int, device: str, spec: dict | None = None) -> dict[str, Any]:
    """All checkpoints vs each other, vs the FSMs and the stationary target; FSMs vs each other; Elo over all."""
    actors = [load_actor(c, device, spec) for c in ckpts]
    spec, cfg = actors[0][1], actors[0][2]
    names = [f"{Path(c).parent.name[-15:]}/{Path(c).stem.replace('ckpt_', '')}" for c in ckpts]
    variants = get(spec, "opponent.variants")
    pool = PolicyPool(len(actors), iface.OBS_DIM, list(iface.HIDDEN), device)
    for i, (a, _, _) in enumerate(actors):
        pool.put(i, a, names[i])
    score: dict[tuple[str, str], float] = {}
    table: dict[str, dict[str, dict[str, float]]] = {}
    opponents = ["stationary", *variants, *[f"pool{j}" for j in range(len(actors))]]
    for i, (a, _, _) in enumerate(actors):
        res = evaluate_many(a, spec, cfg, opponents, matches, device, pool=pool)
        res = {(names[int(k[4:])] if k.startswith("pool") else k): v for k, v in res.items()}
        table[names[i]] = res
        for opp, v in res.items():
            if opp != "stationary" and opp != names[i]:
                score[(names[i], opp)] = v["win"] + 0.5 * v["draw"]
    for a in variants:
        for b in variants:
            if a != b:
                r = run_matches(spec, a, b, matches, device, randomize=cfg["env"]["randomize"])
                score[(a, b)] = r["a_wins"] + 0.5 * r["draws"]
    ratings = elo(names + variants, score, anchor=variants)
    return {"names": names, "table": table, "elo": ratings}


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


def load_actor(ckpt_path: str | Path, device: str, spec: dict | None = None) -> tuple[Actor, dict, dict]:
    """Actor + the run's training config. The spec is the current robot_spec.yaml (it must match the current sim
    code) unless one is passed (e.g. with overrides); the checkpoint's own copy documents what it was trained on."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    iface.check_checkpoint(ckpt, ckpt_path)
    spec, cfg = spec or load_spec(), ckpt["cfg"]
    actor = Actor(iface.OBS_DIM, list(iface.HIDDEN)).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor, spec, cfg


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles/pipes default to cp1251
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--matrix", nargs="+", default=None, help="checkpoints for a round-robin + Elo")
    ap.add_argument("--out", default=None, help="write the round-robin / regression result as JSON")
    ap.add_argument("--regression", default=None, help="regression suite yaml (fixed opponents and seeds)")
    ap.add_argument("--extreme", default=None, help="nominal vs extreme hardware scenarios (configs/eval_extreme.yaml)")
    ap.add_argument("--baseline", default=None, help="with --regression: checkpoint the candidate must not fall behind")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="spec override (repeatable)")
    ap.add_argument("--sim", default=None, choices=["2d", "3d"], help="override the simulator (default: as trained)")
    ap.add_argument("--sensors", default="installed", choices=["installed", "all", "none"],
                    help="reserved sensor groups present in eval matches")
    ap.add_argument("--matches", type=int, default=2048)
    ap.add_argument("--video", type=int, default=0, help="matches per opponent type to record")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.extreme:
        ex = extreme_eval(args.ckpt, args.extreme, args.device, args.sim)
        opps = list(next(iter(ex.values()))["per_opponent"])
        print(f"{'scenario':>14} {'mean win':>9} {'drop':>7}  " + "".join(f"{o[:10]:>11}" for o in opps))
        for name, r in ex.items():
            flag = "" if r["drop"] <= 0.15 else "  > 15%"
            print(f"{name:>14} {r['win']:9.1%} {r['drop']:7.1%}  "
                  + "".join(f"{r['per_opponent'][o]['win']:11.0%}" for o in opps) + flag)
        if args.out:
            Path(args.out).write_text(json.dumps(ex, indent=2), encoding="utf-8")
        return
    spec = apply_overrides(load_spec(), args.set)
    if args.set:
        print("spec overrides:", ", ".join(args.set))
    if args.regression:
        rg = regression(args.ckpt, args.regression, args.device, spec, args.baseline)
        for name, w in rg["candidate"].items():
            base = f"   baseline {rg['baseline'][name]:6.1%}" if "baseline" in rg else ""
            print(f"{name[-30:]:>30}: win {w:6.1%}{base}")
        if "passed" in rg:
            print("REGRESSION PASSED" if rg["passed"] else f"REGRESSION FAILED: {rg['regressions']}")
        if args.out:
            Path(args.out).write_text(json.dumps(rg, indent=2), encoding="utf-8")
        return
    if args.matrix:
        rr = round_robin(args.matrix, args.matches, args.device, spec)
        cols = list(next(iter(rr["table"].values())).keys())
        print(f"{'win% (row vs col)':>28}" + "".join(f"{c[-12:]:>13}" for c in cols))
        for name, row in rr["table"].items():
            print(f"{name:>28}" + "".join(f"{row[c]['win']:13.0%}" for c in cols))
        print("Elo (FSM mean = 1000):")
        for name, e in sorted(rr["elo"].items(), key=lambda kv: -kv[1]):
            print(f"{name:>28}  {e:7.0f}")
        if args.out:
            Path(args.out).write_text(json.dumps(rr, indent=2), encoding="utf-8")
        return
    if not args.ckpt:
        ap.error("--ckpt or --matrix is required")
    actor, spec, cfg = load_actor(args.ckpt, args.device, spec)
    if args.sim:
        cfg = {**cfg, "env": {**cfg["env"], "sim": args.sim}}
    names = ["stationary", *get(spec, "opponent.variants")]
    for opp in names:
        r = evaluate(actor, spec, cfg, opp, args.matches, args.device, sensors=args.sensors)
        print(f"{opp:>22}: win {r['win']:6.1%}  loss {r['loss']:6.1%}  draw {r['draw']:6.1%}")
    if args.video:
        path = Path(args.ckpt).with_suffix(".mp4")
        print(video(actor, spec, cfg, ["aggressive", "spinner", "random_walker"], args.video, args.device, path))


if __name__ == "__main__":
    main()
