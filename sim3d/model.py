"""MJCF of the dohyo and two robots, built from the CAD assets (sim3d/build_assets.py) and robot_spec.yaml.

Geometry (meshes, axle position, wheel size) comes from sim3d/assets/generated/geometry.yaml, i.e. from the CAD;
mass, centre of mass and every physical coefficient from the spec (nominal values here; SumoEnv3D overwrites them
per world for randomization). Contacts use primitives only (analytic in MuJoCo, see build_assets): the body is a
main box, a lip box and sloped plates along the scoop profile; wheels are cylinders that touch only the ring;
the ring is a plane by default (`sim.sim3d.platform`), losing is decided by the CoM rule as in 2D.
Collision bits: ring contype 1/conaffinity 1, bodies 3/2 (ring and each other), wheels 1/0 (ring only).
Each wheel has a motor actuator, force = τ_stall·ctrl − (τ_stall/ω₀)·ω (the DC motor torque-speed line; ctrl = drive
fraction · V/V_nom), and the gearbox's reflected inertia as joint armature. The worm gear's self-locking is joint
damping that SumoEnv3D switches on per world while the motor is stopped (integrated implicitly, so it can be stiff;
as a stiff actuator it made the solver diverge). Deadband, lag and the firmware layer run in SumoEnv3D.

    uv run python -m sim3d.model --render runs/videos/sim3d_robot.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco
import numpy as np
import yaml

from sumo.config import get, load_spec

GEN = Path(__file__).resolve().parent / "assets" / "generated"


def load_geometry() -> dict:
    return yaml.safe_load((GEN / "geometry.yaml").read_text(encoding="utf-8"))


def assets_xml() -> str:
    cols = sorted(GEN.glob("body_col_*.stl"))
    meshes = [f'<mesh name="body_vis" file="{(GEN / "body.stl").as_posix()}"/>',
              f'<mesh name="wheel_vis" file="{(GEN / "wheel.stl").as_posix()}"/>']
    meshes += [f'<mesh name="{c.stem}" file="{c.as_posix()}"/>' for c in cols]
    return "\n    ".join(meshes)


def robot_xml(name: str, pos: tuple[float, float], yaw: float, spec: dict, geo: dict, rgba: str) -> str:
    mass = get(spec, "robot.mass")
    # robot.mass is the whole robot; it is split between body and wheels by their CAD volumes (uniform density)
    wheel_mass = mass * geo["wheel_volume_m3"] / (geo["body_volume_m3"] + 2 * geo["wheel_volume_m3"])
    com_x = geo["axle_x"] + get(spec, "robot.com_x_from_axle")
    com_z = get(spec, "sim.sim3d.com_height")
    armature = get(spec, "robot.motors.reflected_inertia_kgm2")
    L, W, H = geo["body_length"], geo["body_width"], geo["body_height"]
    mb = mass - 2 * wheel_mass
    ixx, iyy, izz = mb * (W**2 + H**2) / 12, mb * (L**2 + H**2) / 12, mb * (L**2 + W**2) / 12
    r, w = geo["wheel_radius"], geo["wheel_width"]
    ax, az, half = geo["axle_x"], geo["axle_z"], geo["track_width"] / 2
    tire = get(spec, "robot.drive.tire_mu")
    scoop = get(spec, "robot.front.friction_mu")
    cols = "".join(
        f'<geom name="{name}_{b["name"]}" type="box" size="{b["size"][0]} {b["size"][1]} {b["size"][2]}" '
        f'pos="{b["pos"][0]} {b["pos"][1]} {b["pos"][2]}" euler="0 {b["pitch_deg"]} 0" contype="3" conaffinity="2" '
        f'friction="{scoop} 0.005 0.0001" rgba="{rgba}" group="3"/>'
        for b in geo["collision_boxes"]
    )
    wheels = "".join(f"""
      <body name="{name}_w{s}" pos="{ax} {sign * half} {az}">
        <joint name="{name}_j{s}" type="hinge" axis="0 1 0" damping="0.0" armature="{armature}"/>
        <geom type="cylinder" size="{r} {w / 2}" euler="90 0 0" mass="{wheel_mass}" friction="{tire} 0.005 0.0001"
              rgba="0.1 0.1 0.1 1" group="3" contype="1" conaffinity="0"/>
      </body>""" for s, sign in (("l", 1), ("r", -1)))
    return f"""
    <body name="{name}" pos="{pos[0]} {pos[1]} 0.002" euler="0 0 {math.degrees(yaw)}">
      <freejoint/>
      <inertial pos="{com_x} 0 {com_z}" mass="{mb}" diaginertia="{ixx} {iyy} {izz}"/>
      <geom type="mesh" mesh="body_vis" contype="0" conaffinity="0" rgba="{rgba}" group="1"/>
      <!-- E18 beam target: the body's bounding box, no contacts, hidden group 4 (rays skip alpha-0 geoms) -->
      <geom name="{name}_raybox" type="box" size="{L / 2} {W / 2} {H / 2}" pos="0 0 {H / 2}" group="4"
            contype="0" conaffinity="0" rgba="0 0 0 0.01"/>
      {cols}{wheels}
    </body>"""


def scene_xml(spec: dict, starts: tuple = ((-0.3, 0.0, 0.0), (0.3, 0.0, math.pi))) -> str:
    """Actuators: a_motor_l, a_motor_r, b_motor_l, b_motor_r."""
    geo = load_geometry()
    # the platform is built at the largest randomized radius (keeps the broadphase bounds conservative);
    # SumoEnv3D shrinks it per world
    R = max(get(spec, "arena.ring_diameter_rand")) / 2
    est = get(spec, "robot.motors.est_16v")
    tau = est["stall_torque_nm"]
    b = tau / (est["no_load_rpm"] * 2 * math.pi / 60)
    acts = "".join(f'<general name="{n}_motor_{s}" joint="{n}_j{s}" gaintype="fixed" gainprm="{tau}" '
                   f'biastype="affine" biasprm="0 0 {-b}" ctrlrange="-1.2 1.2" ctrllimited="true"/>'
                   for n in ("a", "b") for s in ("l", "r"))
    (xa, ya, ta), (xb, yb, tb) = starts
    if get(spec, "sim.sim3d.platform") == "plane":
        platform_xml = ('<geom name="dohyo" type="plane" size="3 3 0.1" pos="0 0 0" rgba="0.95 0.95 0.95 1" '
                        'contype="1" conaffinity="1"/>')
    else:
        platform_xml = (f'<geom name="dohyo" type="cylinder" size="{R} 0.025" pos="0 0 -0.025" '
                        f'rgba="0.95 0.95 0.95 1" contype="1" conaffinity="1"/>')
    return f"""
<mujoco model="sumo">
  <compiler meshdir="." />
  <option timestep="{get(spec, 'sim.sim3d.physics_dt_s')}" integrator="implicitfast" cone="pyramidal"
          iterations="{get(spec, 'sim.sim3d.solver_iterations')}" ls_iterations="{get(spec, 'sim.sim3d.solver_ls_iterations')}"/>
  <visual><global offwidth="1600" offheight="900"/><headlight ambient="0.5 0.5 0.5" diffuse="0.6 0.6 0.6"/></visual>
  <default><geom solref="{get(spec, 'sim.sim3d.contact_timeconst')} 1"/></default>
  <asset>
    {assets_xml()}
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="3 3 0.1" pos="0 0 -0.05" rgba="0.25 0.25 0.25 1" contype="1" conaffinity="1"/>
    {platform_xml}
    <geom name="edge" type="cylinder" size="{R} 0.0002" pos="0 0 0.0002" rgba="0.05 0.05 0.05 1" contype="0" conaffinity="0"/>
    <geom name="centre" type="cylinder" size="{R - get(spec, 'arena.edge_line_width')} 0.0003" pos="0 0 0.0003"
          rgba="0.95 0.95 0.95 1" contype="0" conaffinity="0"/>
    {robot_xml("a", (xa, ya), ta, spec, geo, "0.2 0.4 0.9 1")}
    {robot_xml("b", (xb, yb), tb, spec, geo, "0.9 0.5 0.1 1")}
  </worldbody>
  <actuator>{acts}</actuator>
</mujoco>"""


def render(path: Path, seconds: float = 0.0, show_collision: bool = False) -> None:
    spec = load_spec()
    m = mujoco.MjModel.from_xml_string(scene_xml(spec))
    d = mujoco.MjData(m)
    d.ctrl[:] = 1.0  # full drive
    for _ in range(round(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 450, 800)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 1, 1, int(show_collision), 0, 0]
    if show_collision:
        opt.geomgroup[1] = 0
    imgs = []
    for az, el, dist, look in ((150, -20, 0.55, (-0.3, 0, 0.04)), (90, -5, 0.5, (-0.3, 0, 0.04)), (120, -35, 1.6, (0, 0, 0))):
        cam = mujoco.MjvCamera()
        cam.lookat[:], cam.azimuth, cam.elevation, cam.distance = look, az, el, dist
        r.update_scene(d, cam, opt)
        imgs.append(r.render())
    from PIL import Image

    Image.fromarray(np.concatenate(imgs, 1)).save(path)
    print(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=0.0)
    ap.add_argument("--collision", action="store_true", help="show the convex collision pieces instead of the mesh")
    a = ap.parse_args()
    render(a.render, a.seconds, a.collision)
