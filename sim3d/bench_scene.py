"""3D feasibility check (spike before M6): two primitive robots on a raised dohyo, batched in MuJoCo Warp.

Geometry and masses come from robot_spec.yaml; the body is a box, the front wedge a thin sloped box, wheels are
cylinders on hinge joints with velocity actuators (torque-limited). Not the final model: the Fusion meshes replace
the primitives later. Measures env-steps/s for N parallel worlds (one env-step = one 10 ms control period).

    uv run python -m sim3d.bench_scene --worlds 1024 4096 8192
"""

from __future__ import annotations

import argparse
import math
import time

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from sumo.config import get, load_spec


def robot_xml(name: str, x: float, yaw_deg: float, spec: dict, rgba: str) -> str:
    L, W = get(spec, "robot.footprint")
    H = get(spec, "robot.height")
    r = get(spec, "robot.drive.wheel_diameter") / 2
    track = get(spec, "robot.drive.track_width")
    axle_x = get(spec, "robot.drive.axle_x")
    mass = get(spec, "robot.mass")
    wedge_len = 0.06
    wedge_angle = 20.0  # deg from the ground; placeholder until the Fusion mesh
    body_mass = mass - 2 * 0.1
    z0 = r  # axle height
    return f"""
    <body name="{name}" pos="{x} 0 {z0 + 0.005}" euler="0 0 {yaw_deg}">
      <freejoint/>
      <geom name="{name}_body" type="box" size="{(L - wedge_len) / 2} {W / 2} {H / 2 - 0.01}"
            pos="{-wedge_len / 2} 0 {H / 2 - z0 + 0.005}" mass="{body_mass}" rgba="{rgba}"/>
      <geom name="{name}_wedge" type="box" size="{wedge_len / 2} {W / 2} 0.004"
            pos="{L / 2 - wedge_len / 2} 0 {-z0 + 0.012}" euler="0 {wedge_angle} 0" mass="0.05"
            friction="{get(spec, 'robot.front.friction_mu')} 0.005 0.0001" rgba="{rgba}"/>
      <body name="{name}_wl" pos="{axle_x} {track / 2} 0">
        <joint name="{name}_jl" type="hinge" axis="0 1 0" damping="0.01"/>
        <geom type="cylinder" size="{r} {0.025}" euler="90 0 0" mass="0.1"
              friction="{get(spec, 'robot.drive.tire_mu')} 0.005 0.0001"/>
      </body>
      <body name="{name}_wr" pos="{axle_x} {-track / 2} 0">
        <joint name="{name}_jr" type="hinge" axis="0 1 0" damping="0.01"/>
        <geom type="cylinder" size="{r} {0.025}" euler="90 0 0" mass="0.1"
              friction="{get(spec, 'robot.drive.tire_mu')} 0.005 0.0001"/>
      </body>
    </body>"""


def scene_xml(spec: dict, start: float = 0.3) -> str:
    R = get(spec, "arena.ring_diameter") / 2
    dt = get(spec, "controller.physics_dt_s")
    est = get(spec, "robot.motors.est_16v")
    tau = est["stall_torque_nm"]
    kv = tau / (est["no_load_rpm"] * 2 * math.pi / 60)  # torque per rad/s: stall torque at zero speed
    acts = "".join(
        f'<velocity name="{n}_{s}" joint="{n}_j{s}" kv="{kv}" forcerange="{-tau} {tau}" ctrlrange="-30 30"/>'
        for n in ("a", "b") for s in ("l", "r")
    )
    START = start
    return f"""
<mujoco model="sumo_spike">
  <option timestep="{dt}" integrator="implicitfast" cone="pyramidal"/>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 0.1" pos="0 0 -0.05" rgba=".3 .3 .3 1"/>
    <geom name="dohyo" type="cylinder" size="{R} 0.025" pos="0 0 -0.025" rgba="1 1 1 1" friction="1 0.005 0.0001"/>
    {robot_xml("a", -START, 0, spec, "0.2 0.4 0.9 1")}
    {robot_xml("b", START, 180, spec, "0.9 0.5 0.1 1")}
  </worldbody>
  <actuator>{acts}</actuator>
</mujoco>"""


def bench(m: mujoco.MjModel, nworld: int, substeps: int, control_steps: int) -> float:
    mm = mjw.put_model(m)
    d = mjw.make_data(m, nworld=nworld, nconmax=48, njmax=192)  # per world
    rng = np.random.default_rng(0)
    ctrl = wp.array(rng.uniform(5, 30, size=(nworld, m.nu)).astype(np.float32), dtype=float)
    wp.copy(d.ctrl, ctrl)
    for _ in range(substeps):  # compile + warm up
        mjw.step(mm, d)
    with wp.ScopedCapture() as cap:
        for _ in range(substeps):
            mjw.step(mm, d)
    wp.synchronize()
    t = time.perf_counter()
    for _ in range(control_steps):
        wp.capture_launch(cap.graph)
    wp.synchronize()
    rate = nworld * control_steps / (time.perf_counter() - t)
    q = d.qpos.numpy()
    print(f"  sanity: robot a z={q[0, 2]:.3f} m, finite={np.isfinite(q).all()}, contacts/world≈{d.nacon.numpy()[0] / nworld:.1f}")
    return rate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, nargs="+", default=[1024, 4096])
    ap.add_argument("--steps", type=int, default=50, help="control steps to time")
    ap.add_argument("--start", type=float, default=0.3, help="robot start offset from centre (0.11 = in contact)")
    args = ap.parse_args()
    spec = load_spec()
    m = mujoco.MjModel.from_xml_string(scene_xml(spec, args.start))
    substeps = round(1 / (get(spec, "controller.control_rate_hz") * get(spec, "controller.physics_dt_s")))
    print(f"model: nq={m.nq} nv={m.nv} nu={m.nu} ngeom={m.ngeom}, {substeps} physics steps per control step")
    for n in args.worlds:
        rate = bench(m, n, substeps, args.steps)
        print(f"{n:>6} worlds: {rate:>10,.0f} env-steps/s  (2D sim with sensors: ~430k)", flush=True)


if __name__ == "__main__":
    main()
