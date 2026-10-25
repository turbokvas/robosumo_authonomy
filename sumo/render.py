"""Match videos: matplotlib frames → mp4 via ffmpeg.

`python -m sumo.render --demo` writes the M1 sanity videos (driving, collisions) to runs/videos/.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402

from sumo import physics  # noqa: E402
from sumo.config import get, load_spec, sample_randomization  # noqa: E402

ROBOT_COLORS = ("tab:blue", "tab:orange")
VIDEO_DIR = Path(__file__).resolve().parent.parent / "runs" / "videos"


@dataclass
class Frame:
    t: float
    centers: np.ndarray  # [R,2]
    theta: np.ndarray    # [R]
    out: np.ndarray      # [R] bool


def robot_outline(center: np.ndarray, theta: float, half: tuple[float, float], wedge: float) -> np.ndarray:
    """Footprint polygon with the front edge drawn as a wedge tip, world coords [K,2]."""
    hx, hy = half
    body = np.array([[-hx, -hy], [hx - wedge, -hy], [hx, 0.0], [hx - wedge, hy], [-hx, hy]])
    c, s = math.cos(theta), math.sin(theta)
    return body @ np.array([[c, s], [-s, c]]) + center


def write_video(
    episodes: list[list[Frame]],
    titles: list[str],
    ring_radius: list[float],
    edge_width: float,
    half: tuple[float, float],
    path: Path,
    fps: int,
    draw_robots: tuple[int, ...] = (0, 1),
) -> Path:
    """One subplot per episode, all played in sync."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = min(len(episodes), 2)
    rows = math.ceil(len(episodes) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
    fig.patch.set_facecolor("0.35")
    artists = []
    for ax, title, r in zip(axes.flat, titles, ring_radius):
        ax.set_aspect("equal")
        lim = r + 0.15
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_facecolor("0.35")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.add_patch(Circle((0, 0), r, color="black", zorder=0))  # edge band
        ax.add_patch(Circle((0, 0), r - edge_width, color="white", zorder=0))  # white centre
        ax.set_title(title, color="white", fontsize=10)
        polys = [ax.add_patch(Polygon(np.zeros((5, 2)), closed=True, fc=ROBOT_COLORS[i], ec="k", zorder=2)) for i in draw_robots]
        clock = ax.text(-lim + 0.03, lim - 0.1, "", color="white", fontsize=9)
        artists.append((polys, clock))
    for ax in list(axes.flat)[len(episodes):]:
        ax.axis("off")

    n_frames = max(len(e) for e in episodes)
    writer = FFMpegWriter(fps=fps, bitrate=2400)
    with writer.saving(fig, str(path), dpi=100):
        for k in range(n_frames):
            for ep, (polys, clock) in zip(episodes, artists):
                f = ep[min(k, len(ep) - 1)]
                for poly, i in zip(polys, draw_robots):
                    poly.set_xy(robot_outline(f.centers[i], float(f.theta[i]), half, wedge=0.04))
                    poly.set_facecolor("red" if f.out[i] else ROBOT_COLORS[i])
                clock.set_text(f"t = {f.t:5.2f} s")
            writer.grab_frame()
    plt.close(fig)
    return path


Command = Callable[[float], list[list[float]]]  # t → [[l0, r0], [l1, r1]]


def simulate(
    spec: dict,
    starts: list[tuple[tuple[float, float], float, tuple[float, float], float]],
    commands: list[Command],
    seconds: float,
    frame_dt: float,
    device: str = "cpu",
) -> tuple[list[list[Frame]], list[float]]:
    """Run scripted scenarios (one env each) with nominal parameters and record frames."""
    n = len(starts)
    rand = sample_randomization(spec, n, device=device, randomize=False)
    p = physics.make_params(spec, rand)
    s = physics.zeros_state(n, device)
    center = torch.tensor([[a, b] for a, _, b, _ in starts], device=device, dtype=torch.float32)
    theta = torch.tensor([[ta, tb] for _, ta, _, tb in starts], device=device, dtype=torch.float32)
    physics.set_pose(s, p, center, theta)

    episodes: list[list[Frame]] = [[] for _ in range(n)]
    steps_per_frame = round(frame_dt / p.dt)
    for k in range(round(seconds / p.dt) + 1):
        t = k * p.dt
        if k % steps_per_frame == 0:
            ctr = physics.centers(s, p).cpu().numpy()
            th = s.theta.cpu().numpy()
            out = physics.com_out_of_ring(s, p).cpu().numpy()
            for e in range(n):
                episodes[e].append(Frame(t, ctr[e], th[e], out[e]))
        pwm = torch.tensor([c(t) for c in commands], device=device, dtype=torch.float32)
        physics.step(s, p, pwm)
    return episodes, (p.ring_radius.cpu().tolist())


FAR = (10.0, 10.0)  # parks the unused robot outside the view


def demo(spec: dict, out_dir: Path) -> list[Path]:
    half = tuple(x / 2 for x in get(spec, "robot.footprint"))
    edge = get(spec, "arena.edge_line_width")
    fps = 50
    frame_dt = 1 / fps

    def ramp(t: float) -> float:
        return min(1.0, t / 3.0)

    drive = [
        ("full throttle straight", ((-0.6, 0.0), 0.0, FAR, 0.0), lambda t: [[1, 1], [0, 0]]),
        ("arc 1.0 / 0.6", ((0.0, -0.6), 0.0, FAR, 0.0), lambda t: [[0.6, 1.0], [0, 0]]),
        ("spin in place, then go", ((0.0, 0.0), 0.0, FAR, 0.0),
         lambda t: [[-1, 1], [0, 0]] if t < 1.5 else [[1, 1], [0, 0]]),
        ("slow ramp 0 → 1 (deadband)", ((-0.6, 0.0), 0.0, FAR, 0.0), lambda t: [[ramp(t), ramp(t)], [0, 0]]),
    ]
    collide = [
        ("head-on, both full", ((-0.4, 0.0), 0.0, (0.4, 0.0), math.pi), lambda t: [[1, 1], [1, 1]]),
        ("offset head-on (8 cm)", ((-0.4, 0.04), 0.0, (0.4, -0.04), math.pi), lambda t: [[1, 1], [1, 1]]),
        ("T-bone, opponent idle", ((-0.5, 0.0), 0.0, (0.1, 0.0), math.pi / 2), lambda t: [[1, 1], [0, 0]]),
        ("push from behind, opponent idle", ((-0.6, 0.0), 0.0, (-0.25, 0.0), 0.0), lambda t: [[1, 1], [0, 0]]),
    ]
    paths = []
    for name, scen, seconds, robots in (("m1_drive", drive, 4.0, (0,)), ("m1_collisions", collide, 5.0, (0, 1))):
        eps, radii = simulate(spec, [s for _, s, _ in scen], [c for _, _, c in scen], seconds, frame_dt)
        paths.append(write_video(eps, [t for t, _, _ in scen], radii, edge, half, out_dir / f"{name}.mp4", fps, robots))
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="render the M1 physics sanity videos")
    ap.add_argument("--out", type=Path, default=VIDEO_DIR)
    args = ap.parse_args()
    if args.demo:
        for p in demo(load_spec(), args.out):
            print(p)
