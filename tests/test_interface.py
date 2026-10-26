import math

import pytest
import torch

from sumo import firmware, physics
from sumo import interface as iface
from sumo.config import load_spec, sample_randomization
from sumo.env import SumoEnv
from sumo.sensors import Sensors
from train.models import Actor, RunningNorm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def spec():
    return load_spec()


def test_spec_matches_interface(spec):
    iface.check_spec(spec)
    assert iface.OBS_DIM == iface.FRAME_DIM * iface.OBS_STACK
    assert sum(w for _, w in iface.FRAME) == iface.FRAME_DIM


def test_checkpoint_version_is_enforced(spec):
    ok = iface.stamp({"obs_dim": iface.OBS_DIM}, spec)
    iface.check_checkpoint(ok)
    with pytest.raises(iface.InterfaceMismatch):
        iface.check_checkpoint({"obs_dim": iface.OBS_DIM})  # pre-v1, unstamped
    with pytest.raises(iface.InterfaceMismatch):
        iface.check_checkpoint({**ok, "interface_version": iface.INTERFACE_VERSION + 1})


def test_reserved_inputs_start_at_zero_weight():
    a = Actor(iface.OBS_DIM, list(iface.HIDDEN), zero_inputs=iface.reserved_obs_columns())
    w = a.net[0].weight
    cols = iface.reserved_obs_columns()
    assert torch.all(w[:, cols] == 0)
    others = [i for i in range(iface.OBS_DIM) if i not in cols]
    assert torch.all(w[:, others].abs().sum(0) > 0)


def test_norm_floor_keeps_constant_inputs_sane():
    n = RunningNorm(3)
    n.update(torch.tensor([[0.0, 1.0, 5.0], [0.0, 0.0, 5.0]]).repeat(100, 1))
    out = n(torch.tensor([[1.0, 1.0, 5.0]]))  # the always-0 input suddenly reads 1
    assert out[0, 0] <= 1 / n.min_std + 1e-3 and abs(out[0, 2]) < 1e-3


def test_deadband_compensation():
    db = torch.tensor(0.1)
    a = torch.tensor([0.0, 1e-4, 0.01, 0.5, -0.5, 1.0, -1.0])
    duty = firmware.deadband_comp(a, db)
    assert duty[0] == 0 and duty[1] == 0  # stop means stop
    assert duty[2] == pytest.approx(0.1 + 0.9 * 0.01)
    assert duty[3] == pytest.approx(0.55) and duty[4] == pytest.approx(-0.55)
    assert duty[5] == 1 and duty[6] == -1


def test_small_command_moves_robot_with_compensation(spec):
    """With compensation a 5% command drives; the true deadband (up to 15%) would otherwise eat it."""
    env = SumoEnv(spec, 2, DEVICE, randomize=False, seed=0, use_graph=False)
    env.p.deadband.fill_(0.15)
    env.p.deadband_cal.fill_(0.15)
    for _ in range(100):
        env.step(torch.tensor([[[0.05, 0.05], [0.0, 0.0]]] * 2, device=DEVICE))
    assert env.state.wheel_w[:, 0].min() > 0.5


def test_encoder_and_imu_models(spec):
    rand = sample_randomization(spec, 1, device=DEVICE, randomize=False)
    p = physics.make_params(spec, rand)
    s = physics.zeros_state(1, DEVICE)
    physics.set_pose(s, p, torch.tensor([[[0.0, 0.0], [5.0, 5.0]]], device=DEVICE), torch.zeros(1, 2, device=DEVICE))
    sens = Sensors(spec, 1, DEVICE, torch.Generator(DEVICE).manual_seed(0))
    s.wheel_w[:] = 10.0
    s.omega[:] = 2.0
    prev = s.vel.clone()
    s.vel[0, 0] = torch.tensor([0.05, 0.0], device=DEVICE)  # +0.05 m/s in 10 ms → 5 m/s² forward
    m = sens.motion(s, prev, torch.full((1,), 0.01, device=DEVICE))
    step = 2 * math.pi / spec["robot"]["sensors"]["encoders"]["counts_per_wheel_rev"] / 0.01
    assert torch.all((m["wheel_speed"] - 10.0).abs() <= step / 2 + 1e-4)  # quantized to one count per tick
    assert m["gyro_z"][0, 0].item() == pytest.approx(2.0, abs=0.1)
    assert m["accel_xy"][0, 0, 0].item() == pytest.approx(5.0, abs=0.3)
    assert abs(m["accel_xy"][0, 0, 1].item()) < 0.3
