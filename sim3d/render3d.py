"""3D match videos: record qpos of the first worlds from SumoEnv3D, replay them in CPU MuJoCo, encode with ffmpeg.

    uv run python -m sim3d.render3d --ckpt results/m4/ckpt_02000M.pt --opponent stationary --matches 4 \
        --out runs/videos/m4_in_3d_stationary.mp4
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import mujoco
import numpy as np
import torch

from sim3d.model import scene_xml
from sumo.config import load_spec
from sumo.env import SumoTask
from train.eval import load_actor


def record(ckpt: str, opponent: str, matches: int, seconds: float, seed: int, collision: bool = False) -> tuple:
    actor, spec, cfg = load_actor(ckpt, "cuda")
    cfg = {**cfg, "env": {**cfg["env"], "sim": "3d"}}
    task = SumoTask(spec, cfg, matches, "cuda", seed, sensors="installed")
    task.set_opponents({opponent: 1.0})
    task._new_episodes(torch.ones(matches, dtype=torch.bool, device="cuda"))
    env = task.env
    frames, finished = [], torch.zeros(matches, dtype=torch.bool, device="cuda")
    with torch.no_grad():
        for _ in range(round(seconds * 100)):
            frames.append(env.qpos.cpu().numpy().copy())
            _, done, _ = task.step(actor(task.obs()))
            finished |= done
            if finished.all():
                break
    return spec, np.stack(frames)  # [T, N, nq]


def render(spec: dict, qpos: np.ndarray, out: Path, size=(360, 480), collision: bool = False) -> Path:
    m = mujoco.MjModel.from_xml_string(scene_xml(spec))
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, size[0], size[1])
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 0 if collision else 1, 1, 1 if collision else 0, 0, 0]
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (0, 0, 0.02)
    cam.distance, cam.elevation, cam.azimuth = 1.9, -40, 90
    T, N, _ = qpos.shape
    cols = min(N, 2)
    rows = -(-N // cols)
    H, W = size[0] * rows, size[1] * cols
    out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", "50", "-i", "-", "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    for t in range(0, T, 2):
        canvas = np.zeros((H, W, 3), np.uint8)
        for e in range(N):
            d.qpos[:] = qpos[t, e]
            mujoco.mj_forward(m, d)
            r.update_scene(d, cam, opt)
            y, x = divmod(e, cols)
            canvas[y * size[0]:(y + 1) * size[0], x * size[1]:(x + 1) * size[1]] = r.render()
        ff.stdin.write(canvas.tobytes())
    ff.stdin.close()
    ff.wait()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--opponent", default="aggressive")
    ap.add_argument("--matches", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--collision", action="store_true", help="draw the collision pieces instead of the CAD mesh")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    spec, q = record(a.ckpt, a.opponent, a.matches, a.seconds, a.seed)
    print(render(load_spec(), q, a.out, collision=a.collision))


if __name__ == "__main__":
    main()
