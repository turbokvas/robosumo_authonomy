"""3D env (MuJoCo Warp) invariants. Same spirit as test_physics: RL exploits sim bugs, so pin the basics down."""

import math

import pytest
import torch

pytest.importorskip("mujoco_warp")
if not torch.cuda.is_available():
    pytest.skip("SumoEnv3D needs CUDA", allow_module_level=True)

from sim3d.env3d import SumoEnv3D  # noqa: E402
from sumo import physics  # noqa: E402
from sumo.config import get, load_spec  # noqa: E402

DEV = "cuda"


@pytest.fixture(scope="module")
def spec():
    return load_spec()


def place(env, c0, th0, c1, th1):
    """Put both robots at given centres/headings in every world, at rest."""
    n = env.n
    for r, (c, th) in enumerate(((c0, th0), (c1, th1))):
        a = env.free_qadr[r]
        env.qpos[:, a:a + 2] = torch.tensor(c, device=DEV)
        env.qpos[:, a + 2] = 0.002
        env.qpos[:, a + 3] = math.cos(th / 2)
        env.qpos[:, a + 4:a + 6] = 0
        env.qpos[:, a + 6] = math.sin(th / 2)
    env.qvel.zero_()
    env.state.u_lag.zero_()
    env.action.zero_()
    import mujoco_warp as mjw

    env._to_warp()
    mjw.kinematics(env.mm, env.d)
    env._to_torch()
    env._refresh_state(None)
    env.rest_z[:] = env.xipos[:, env.body_id, 2]
    env.t.zero_()


def run(env, action, seconds):
    out = None
    for _ in range(round(seconds * 100)):
        out = env.step(action)
    return out


def test_interface_matches_2d(spec):
    env = SumoEnv3D(spec, 8, DEV, seed=0)
    r = env.readings()
    assert {k: tuple(v.shape[1:]) for k, v in r.items()} == {
        "ir": (2, 3), "ir_side": (2, 2), "line": (2, 4), "battery": (2,), "wheel_speed": (2, 2), "gyro_z": (2,),
        "accel_xy": (2, 2)}
    assert not physics.com_out_of_ring(env.state, env.p).any()


def test_idle_robot_stands_still(spec):
    env = SumoEnv3D(spec, 4, DEV, seed=0, randomize=False)
    place(env, (-0.3, 0.0), 0.0, (0.3, 0.0), math.pi)
    p0 = env.state.pos.clone()
    run(env, torch.zeros(4, 2, 2, device=DEV), 1.0)
    assert (env.state.pos - p0).norm(dim=-1).max() < 1e-3


def test_straight_speed_near_free_speed(spec):
    env = SumoEnv3D(spec, 4, DEV, seed=0, randomize=False)
    place(env, (-0.6, 0.0), 0.0, (0.7, 0.7), 0.0)
    a = torch.zeros(4, 2, 2, device=DEV)
    a[:, 0] = 1.0
    run(env, a, 0.8)
    v = env.state.vel[:, 0, 0]
    rpm = get(spec, "robot.motors.est_16v.no_load_rpm")
    v_free = rpm * 2 * math.pi / 60 * get(spec, "robot.drive.wheel_diameter") / 2
    assert torch.all((v > 0.75 * v_free) & (v < 1.05 * v_free))
    assert env.state.pos[:, 0, 1].abs().max() < 0.01  # straight


def test_beams_see_opponent_ahead(spec):
    env = SumoEnv3D(spec, 2, DEV, seed=0, randomize=False)
    place(env, (-0.3, 0.0), 0.0, (0.3, 0.0), math.pi)
    g = env.sensors.geometry3d()
    assert g[:, :, 1].all()  # centre E18 of both robots
    place(env, (-0.3, 0.0), math.pi, (0.3, 0.0), 0.0)  # back to back
    assert not env.sensors.geometry3d()[:, :, :3].any()


def test_no_divergence_with_random_commands(spec):
    """Regression: locking a spinning wheel instantly (worm model) threw robots into the air."""
    env = SumoEnv3D(spec, 256, DEV, seed=0)
    g = torch.Generator(DEV).manual_seed(0)
    prev = env.xipos[:, env.body_id, 2].clone()
    jumps = 0
    for _ in range(300):
        _, done, _ = env.step(torch.rand(256, 2, 2, device=DEV, generator=g) * 2 - 1)
        z = env.xipos[:, env.body_id, 2]
        jumps += int((((z - prev) > 0.015).any(-1) & ~done).sum())
        env.reset(done)
        prev = env.xipos[:, env.body_id, 2].clone()
    assert jumps == 0
    assert int(env.n_diverged.sum()) == 0


def test_worm_holds_stopped_wheels_against_a_push(spec):
    env = SumoEnv3D(spec, 2, DEV, seed=0, randomize=False)
    place(env, (-0.6, 0.0), 0.0, (-0.25, 0.0), 0.0)  # robot 0 pushes the idle robot 1 from behind
    a = torch.zeros(2, 2, 2, device=DEV)
    a[:, 0] = 1.0
    run(env, a, 1.0)
    assert env.state.wheel_w[:, 1].abs().max() < 1.0  # pushed robot's wheels stay (nearly) locked


@pytest.mark.parametrize("blunt", [0, 1])
def test_head_on_blunter_blade_rides_up(spec, blunt):
    """Head-on, both pushing: the blunter blade edge (higher axis) rides up the other's scoop, whichever slot it is
    in. Regression: a blunt plate end face at the tip lifted the pusher (and slot 0) regardless of the blades."""
    env = SumoEnv3D(spec, 2, DEV, seed=0, randomize=False)
    env.p.clearance[:] = 0.0
    env.p.clearance[:, blunt] = 0.002
    env._apply_params(torch.arange(2, device=DEV))
    half = env.half[0]
    place(env, (-half - 0.01, 0.0), 0.0, (half + 0.01, 0.0), math.pi)
    run(env, torch.ones(2, 2, 2, device=DEV), 1.0)
    tilt = torch.rad2deg(env.state.wedge)
    assert (tilt[:, blunt] > tilt[:, 1 - blunt] + 2.0).all(), tilt


def test_side_push_beats_head_on(spec):
    """A locked opponent is pushed far when hit from the side, much less when hit head-on (it lifts the pusher)."""
    env = SumoEnv3D(spec, 2, DEV, seed=0, randomize=False)
    a = env.free_qadr[1]
    place(env, (-0.35, 0.0), 0.0, (0.0, 0.0), 0.0)
    env.qpos[0, a + 3], env.qpos[0, a + 6] = math.cos(math.pi / 4), math.sin(math.pi / 4)  # world 0: side
    env.qpos[1, a + 3], env.qpos[1, a + 6] = 0.0, 1.0  # world 1: head-on
    import mujoco_warp as mjw

    env._to_warp()
    mjw.kinematics(env.mm, env.d)
    env._to_torch()
    env._refresh_state(None)
    x0 = env.state.pos[:, 1, 0].clone()
    act = torch.zeros(2, 2, 2, device=DEV)
    act[:, 0] = 1.0
    run(env, act, 2.0)
    moved = env.state.pos[:, 1, 0] - x0
    assert moved[0] > 0.2 and moved[0] > 2 * moved[1], moved


def test_sliding_tires_grip_less(spec):
    """Rolling tires keep (nearly) the static μ, a locked wheel dragged sideways or a spinning wheel gets μ·slide_ratio."""
    env = SumoEnv3D(spec, 2, DEV, seed=0, randomize=False)
    place(env, (-0.6, 0.0), 0.0, (0.6, 0.6), 0.0)
    act = torch.zeros(2, 2, 2, device=DEV)
    act[:, 0] = 1.0
    run(env, act, 0.8)  # robot 0 at free speed, robot 1 idle
    k = get(spec, "robot.drive.tire_slide_ratio")
    mu = get(spec, "robot.drive.tire_mu")
    rolling = env.m_friction[:, env.wheel_geom_lr[0][0], 0]
    assert (rolling > 0.95 * mu).all(), rolling
    env.qvel[:, env.free_dadr[1] + 1] = 0.5  # drag the idle robot sideways
    env.step(torch.zeros(2, 2, 2, device=DEV))
    sliding = env.m_friction[:, env.wheel_geom_lr[1][0], 0]
    assert torch.allclose(sliding, torch.full_like(sliding, k * mu), atol=0.02), sliding
