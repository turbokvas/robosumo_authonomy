import math

import pytest
import torch

from sumo import physics
from sumo.config import get, load_spec, sample_randomization
from sumo.sensors import Sensors, ray_box_distance

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def spec():
    return load_spec()


def make(spec, n, xy1, th1, xy0=(0.0, 0.0), th0=0.0, **overrides):
    rand = sample_randomization(spec, n, device=DEVICE, randomize=False)
    for name, value in overrides.items():
        rand.params[name].fill_(value)
    p = physics.make_params(spec, rand)
    s = physics.zeros_state(n, DEVICE)
    center = torch.tensor([xy0, xy1], device=DEVICE, dtype=torch.float32).expand(n, 2, 2).clone()
    theta = torch.tensor([th0, th1], device=DEVICE, dtype=torch.float32).expand(n, 2).clone()
    physics.set_pose(s, p, center, theta)
    sens = Sensors(spec, n, DEVICE, torch.Generator(DEVICE).manual_seed(0))
    sens.set_params(rand)
    sens.prime(s, p)
    return p, s, sens


def substep_dt(spec, n):
    return torch.full((n,), get(spec, "controller.physics_dt_s"), device=DEVICE)


def test_ray_box_distance():
    o = torch.tensor([[-1.0, 0.0], [-1.0, 0.0], [0.0, 0.0], [-1.0, 0.15]])
    d = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
    dist = ray_box_distance(o, d, (0.1, 0.1))
    assert dist[0] == pytest.approx(0.9)
    assert math.isinf(dist[1])
    assert dist[2] == 0  # origin inside
    assert math.isinf(dist[3])  # passes above


def test_ir_sees_opponent_ahead_within_range(spec):
    p, s, sens = make(spec, 1, (0.6, 0.0), math.pi)
    g = sens.geometry(s, p)
    assert g[0, 0, :3].tolist() == [False, True, False]  # only the centre beam hits
    p, s, sens = make(spec, 1, (1.5, 0.0), math.pi)  # beyond range
    assert not sens.geometry(s, p)[0, 0, :3].any()


def test_side_beam(spec):
    ang = math.radians(15)
    p, s, sens = make(spec, 1, (0.1 + 0.5 * math.cos(ang), 0.06 + 0.5 * math.sin(ang)), 0.0)
    assert sens.geometry(s, p)[0, 0, :3].tolist() == [True, False, False]


def test_nothing_behind(spec):
    p, s, sens = make(spec, 1, (-0.6, 0.0), 0.0)
    assert not sens.geometry(s, p)[0, 0, :3].any()


def test_line_sensors(spec):
    r = get(spec, "arena.ring_diameter") / 2
    p, s, sens = make(spec, 1, (5.0, 5.0), 0.0)
    assert not sens.geometry(s, p)[0, 0, 6:].any()  # centre of the ring: all white
    # front corners in the black band, rear still on white
    p, s, sens = make(spec, 1, (5.0, 5.0), 0.0, xy0=(r - 0.12, 0.0))
    assert sens.geometry(s, p)[0, 0, 6:].tolist() == [True, True, False, False]
    p, s, sens = make(spec, 1, (5.0, 5.0), 0.0, xy0=(r + 0.3, 0.0))  # fell off: no reflection
    assert sens.geometry(s, p)[0, 0, 6:].all()


def test_latency(spec):
    n = 1
    p, s, sens = make(spec, n, (3.0, 0.0), math.pi, **{"robot.sensors.opponent_ir.latency_s": 0.008})
    sens.fn_p = sens.fp_p = 0.0
    dt = substep_dt(spec, n)
    physics.set_pose(s, p, torch.tensor([[[0.0, 0.0], [0.6, 0.0]]], device=DEVICE), torch.tensor([[0.0, math.pi]], device=DEVICE))
    seen = []
    for _ in range(6):
        sens.push(s, p)
        seen.append(bool(sens.read(dt)["ir"][0, 0, 1]))
    assert seen == [False, False, False, False, True, True]  # 8 ms / 2 ms = 4 substeps late


def test_false_positive_rate(spec):
    n = 20000
    p, s, sens = make(spec, n, (5.0, 5.0), 0.0)
    ir = sens.read(substep_dt(spec, n))["ir"][:, 0]
    fp = get(spec, "robot.sensors.opponent_ir.false_positive_p")
    assert ir.float().mean().item() == pytest.approx(fp, abs=3 * math.sqrt(fp / (n * 3)) + 1e-4)


def test_crosstalk_beyond_range(spec):
    """Facing E18s 1.2 m apart: no beam hit (range 1.0) but the opponent's emitter shines into ours."""
    n = 20000
    p, s, sens = make(spec, n, (1.4, 0.0), math.pi)
    g = sens.geometry(s, p)
    assert not g[0, 0, :3].any()
    assert g[0, 0, 3:6].any()
    ir = sens.read(substep_dt(spec, n))["ir"][:, 0]
    xt_p = get(spec, "robot.sensors.opponent_ir.crosstalk_p")
    assert ir[:, 1].float().mean().item() > 0.8 * xt_p


def test_battery(spec):
    p, s, sens = make(spec, 1, (5.0, 5.0), 0.0, **{"robot.power.voltage": 14.0})
    assert sens.read(substep_dt(spec, 1))["battery"][0, 0].item() == pytest.approx(14.0 / 16.8)
