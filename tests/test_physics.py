"""Physics invariants (PLAN.md M1). RL exploits physics bugs, so every rule here guards a known way to cheat."""

import math
import time

import pytest
import torch

from sumo import physics
from sumo.config import get, load_spec, sample_randomization

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def spec():
    return load_spec()


def make(spec, n=1, fixed=None, **overrides):
    """Nominal-parameter batch; overrides = {param_name: value} applied to the randomization batch."""
    rand = sample_randomization(spec, n, device=DEVICE, randomize=False)
    for name, value in overrides.items():
        rand.params[name].fill_(value)
    p = physics.make_params(spec, rand, fixed=fixed)
    s = physics.zeros_state(n, DEVICE)
    return p, s


def place(s, p, xy0, th0, xy1=(10.0, 10.0), th1=0.0):
    n = s.pos.shape[0]
    center = torch.tensor([xy0, xy1], device=DEVICE, dtype=torch.float32).expand(n, 2, 2).clone()
    theta = torch.tensor([th0, th1], device=DEVICE, dtype=torch.float32).expand(n, 2).clone()
    physics.set_pose(s, p, center, theta)


def run(s, p, pwm, seconds, ext_force=None):
    infos = []
    for _ in range(round(seconds / p.dt)):
        infos.append(physics.step(s, p, pwm, ext_force))
    return infos


def cmd(n, left0, right0, left1=0.0, right1=0.0):
    return torch.tensor([[left0, right0], [left1, right1]], device=DEVICE).expand(n, 2, 2)


def free_speed(spec, volt_ratio):
    rpm = get(spec, "robot.motors.est_16v.no_load_rpm")
    return rpm * 2 * math.pi / 60 * get(spec, "robot.drive.wheel_diameter") / 2 * volt_ratio


def test_idle_robot_stays(spec):
    p, s = make(spec)
    place(s, p, (0.0, 0.0), 0.3)
    start = s.clone()
    run(s, p, cmd(1, 0, 0), 2.0)
    assert torch.allclose(s.pos, start.pos, atol=1e-7)
    assert torch.allclose(s.theta, start.theta, atol=1e-7)


def test_symmetric_command_drives_straight(spec):
    p, s = make(spec)
    place(s, p, (-0.5, 0.0), 0.0)
    run(s, p, cmd(1, 0.8, 0.8), 1.0)
    assert s.pos[0, 0, 0] > -0.2  # moved forward
    assert abs(s.pos[0, 0, 1].item()) < 1e-5
    assert abs(s.theta[0, 0].item()) < 1e-5


def test_top_speed(spec):
    p, s = make(spec, **{"robot.power.voltage": 16.0, "robot.motors.strength": 1.0})
    place(s, p, (-5.0, 0.0), 0.0)  # ring is irrelevant for the physics step
    run(s, p, cmd(1, 1, 1), 2.0)
    v = s.vel[0, 0, 0].item()
    v_free = free_speed(spec, 1.0)
    assert v_free == pytest.approx(0.78, abs=0.02)
    assert 0.85 * v_free < v < v_free  # only scoop drag and back-EMF slow it down


def test_spin_in_place_turns_left(spec):
    p, s = make(spec)
    place(s, p, (0.0, 0.0), 0.0)
    axle = torch.tensor([[p.axle_x, 0.0]], device=DEVICE)
    pivot0 = physics.body_to_world(s, p, axle)[0, 0, 0]
    run(s, p, cmd(1, -1, 1), 0.5)
    assert s.omega[0, 0] > 1.0
    drift = (physics.body_to_world(s, p, axle)[0, 0, 0] - pivot0).norm()
    assert drift < 0.03  # turns about the axle; scoop drag drifts it a little, no driving away


def test_push_against_wall_saturates_at_traction(spec):
    """Full throttle into an immovable robot: force capped by μ·N, wheels slip instead of pushing harder."""
    fixed = torch.tensor([[False, True]], device=DEVICE)
    p, s = make(spec, fixed=fixed, **{"robot.power.voltage": 16.8, "robot.motors.strength": 1.15})
    gap = 0.005
    place(s, p, (0.0, 0.0), 0.0, (0.2 + gap, 0.0), math.pi)
    infos = run(s, p, cmd(1, 1, 1), 2.0)
    push = torch.stack([i["contact_normal_force"] for i in infos[-200:]]).mean().item()
    traction = 2 * (p.tire_mu * p.n_wheel)[0, 0].item()
    assert 0.8 * traction < push < 1.02 * traction
    wheel_surface = (s.wheel_w[0, 0] * p.wheel_r).min().item()
    assert wheel_surface > 0.2  # wheels keep spinning (slip) while the body is stopped
    assert s.vel[0, 0].norm() < 0.01
    assert max(i["contact_depth"].max().item() for i in infos) < 0.005


def test_external_force_does_not_turn_locked_wheels(spec):
    p, s = make(spec)
    place(s, p, (0.0, 0.0), 0.0)
    static = (2 * p.tire_mu * p.n_wheel + p.scoop_mu * p.n_scoop)[0, 0].item()
    x0 = s.pos[0, 0, 0].item()
    weak = torch.tensor([[0.3 * static, 0.0], [0.0, 0.0]], device=DEVICE).expand(1, 2, 2)
    run(s, p, cmd(1, 0, 0), 1.0, ext_force=weak)
    assert torch.all(s.wheel_w == 0)
    assert abs(s.pos[0, 0, 0].item() - x0) < 0.01  # smoothed Coulomb creeps a little, but no rolling

    strong = torch.tensor([[1.5 * static, 0.0], [0.0, 0.0]], device=DEVICE).expand(1, 2, 2)
    run(s, p, cmd(1, 0, 0), 0.5, ext_force=strong)
    assert torch.all(s.wheel_w == 0)
    assert s.pos[0, 0, 0].item() - x0 > 0.1  # slides on locked wheels


def test_wheel_never_backdriven(spec):
    """Driving forward slowly while pushed backwards hard: the wheel stops, it never turns backwards."""
    p, s = make(spec)
    place(s, p, (0.0, 0.0), 0.0)
    push_back = torch.tensor([[-60.0, 0.0], [0.0, 0.0]], device=DEVICE).expand(1, 2, 2)
    for _ in range(500):
        physics.step(s, p, cmd(1, 0.4, 0.4), push_back)
        assert torch.all(s.wheel_w[0, 0] >= 0)
    assert s.vel[0, 0, 0] < 0  # robot slides backwards anyway


def test_head_on_collision_is_stable(spec):
    p, s = make(spec)
    place(s, p, (-0.25, 0.0), 0.0, (0.25, 0.0), math.pi)
    infos = run(s, p, cmd(1, 1, 1, 1, 1), 3.0)
    assert torch.isfinite(s.pos).all()
    assert max(i["contact_depth"].max().item() for i in infos) < 0.01
    assert s.pos[0, :, 0].mean().abs() < 1e-3  # identical robots: deadlock in the middle
    assert s.vel.norm(dim=-1).max() < 0.02


def test_frictionless_collision_conserves_momentum(spec):
    p, s = make(spec, **{"robot.drive.tire_mu": 0.0, "robot.front.friction_mu": 0.0})
    place(s, p, (-0.3, 0.05), 0.0, (0.0, 0.0), 0.4)
    s.vel[:, 0] = torch.tensor([1.0, 0.0], device=DEVICE)
    p0 = s.vel.sum(1).clone()
    e0 = 0.5 * (s.vel.pow(2).sum(-1) / p.inv_mass).sum() + 0.5 * (s.omega.pow(2) / p.inv_inertia).sum()
    infos = run(s, p, cmd(1, 0, 0), 1.0)
    assert max(i["contact_normal_force"].max().item() for i in infos) > 0  # they did collide
    assert torch.allclose(s.vel.sum(1), p0, atol=1e-4)
    e1 = 0.5 * (s.vel.pow(2).sum(-1) / p.inv_mass).sum() + 0.5 * (s.omega.pow(2) / p.inv_inertia).sum()
    assert e1 <= e0 * 1.001


def test_box_contacts_face_to_face():
    h = (0.1, 0.1)
    c = torch.tensor([[[0.0, 0.0], [0.195, 0.0]]])
    pts, n, depth = physics.box_contacts(c, torch.tensor([[0.0, math.pi]]), h)
    assert torch.allclose(n, torch.tensor([[1.0, 0.0]]), atol=1e-6)
    assert torch.allclose(depth, torch.full((1, 2), 0.005), atol=1e-6)
    assert torch.allclose(pts[0, :, 1].sort().values, torch.tensor([-0.1, 0.1]), atol=1e-6)


def test_box_contacts_corner_and_separated():
    h = (0.1, 0.1)
    corner = 0.1 + 0.1 * math.sqrt(2) - 0.004  # box 1 rotated 45°, its corner 4 mm into box 0
    c = torch.tensor([[[0.0, 0.0], [corner, 0.0]], [[0.0, 0.0], [0.5, 0.0]]])
    th = torch.tensor([[0.0, math.pi / 4], [0.0, 0.0]])
    pts, n, depth = physics.box_contacts(c, th, h)
    assert (depth[0] > 0).sum() == 1
    assert depth[0].max() == pytest.approx(0.004, abs=1e-5)
    assert torch.allclose(n[0], torch.tensor([1.0, 0.0]), atol=1e-5)
    assert torch.all(depth[1] == 0)


def test_out_of_ring(spec):
    p, s = make(spec)
    place(s, p, (0.0, 0.0), 0.0, (0.79, 0.0), 0.0)  # opponent CoM at 0.79 - 0.04 → inside 0.80
    assert not physics.com_out_of_ring(s, p).any()
    place(s, p, (0.0, 0.0), 0.0, (0.85, 0.0), 0.0)
    assert physics.com_out_of_ring(s, p)[0].tolist() == [False, True]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_throughput_4096_envs(spec):
    n = 4096
    rand = sample_randomization(spec, n, device="cuda", generator=torch.Generator("cuda").manual_seed(0))
    p = physics.make_params(spec, rand)
    s = physics.zeros_state(n, "cuda")
    place(s, p, (-0.3, 0.0), 0.0, (0.3, 0.0), math.pi)
    pwm = torch.rand(n, 2, 2, device="cuda") * 2 - 1
    substeps = round(1 / get(spec, "controller.control_rate_hz") / p.dt)
    for _ in range(50):
        physics.step(s, p, pwm)
    torch.cuda.synchronize()
    t = time.perf_counter()
    control_steps = 100
    for _ in range(control_steps * substeps):
        physics.step(s, p, pwm)
    torch.cuda.synchronize()
    rate = n * control_steps / (time.perf_counter() - t)
    print(f"\nphysics: {rate:,.0f} env-steps/s ({substeps} substeps each)")
    assert rate >= 50_000
