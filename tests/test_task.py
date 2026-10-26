import pytest
import torch
import yaml

from sumo import physics
from sumo.config import get, load_spec
from sumo import interface as iface
from sumo.env import SumoTask

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CFG = yaml.safe_load(open("configs/m3.yaml", encoding="utf-8"))


@pytest.fixture(scope="module")
def spec():
    return load_spec()


def test_obs_shapes_and_actor_sees_only_mcu_signals(spec):
    task = SumoTask(spec, CFG, 32, DEVICE, seed=0, sensors="installed")
    obs = task.obs()
    assert obs.shape == (32, iface.OBS_DIM)
    frame = obs.view(32, iface.OBS_STACK, iface.FRAME_DIM)[:, -1]
    F = iface.FRAME_SLICES
    digital = torch.cat([frame[:, F["ir_front"]], frame[:, F["line"]]], -1)
    assert torch.all((digital == 0) | (digital == 1))
    assert torch.all(frame[:, F["battery"]] > 0.8)  # V / full_v
    reserved = torch.cat([frame[:, F[name]] for name in iface.RESERVED_FIELDS], -1)
    assert torch.all(reserved == 0)  # nothing installed on the current robot → zeros, flags 0
    assert torch.isfinite(task.critic_obs()).all()


def test_critic_layout_does_not_depend_on_pool_or_sensors(spec):
    from train.league import PolicyPool
    a = SumoTask(spec, CFG, 8, DEVICE, seed=0, sensors="none").critic_obs().shape[1]
    pool = PolicyPool(4, iface.OBS_DIM, list(iface.HIDDEN), DEVICE)
    b = SumoTask(spec, CFG, 8, DEVICE, seed=0, pool=pool, sensors="all").critic_obs().shape[1]
    assert a == b


def test_sensor_groups_dropped_per_match(spec):
    cfg = {**CFG, "env": {**CFG["env"], "sensor_dropout": {"side_ir": 0.5, "encoders": 0.5, "imu": 0.5}}}
    task = SumoTask(spec, cfg, 4096, DEVICE, seed=0)
    frac = task.present.mean((0, 1))
    assert torch.all((frac > 0.45) & (frac < 0.55))
    frame = task.obs().view(4096, iface.OBS_STACK, iface.FRAME_DIM)[:, -1]
    F = iface.FRAME_SLICES
    absent_enc = task.present[:, 0, 1] == 0
    assert torch.all(frame[absent_enc][:, F["wheel_speed"]] == 0)
    assert torch.equal(frame[:, F["has_encoders"]].squeeze(-1), task.present[:, 0, 1])


def test_terminal_reward_and_autoreset(spec):
    task = SumoTask(spec, CFG, 2, DEVICE, seed=0)
    task.set_opponents({"stationary": 1.0})
    st, p = task.env.state, task.env.p
    r = p.ring_radius[0].item()
    ctr = torch.tensor([[[0.0, 0.0], [r + 0.2, 0.0]], [[r + 0.2, 0.0], [0.0, 0.3]]], device=DEVICE)
    physics.set_pose(st, p, ctr, torch.zeros(2, 2, device=DEVICE))
    reward, done, info = task.step(torch.zeros(2, 2, device=DEVICE))
    assert done.all()
    assert info["winner"].tolist() == [0, 1]
    assert reward[0] > CFG["reward"]["win"] - 1 and reward[1] < CFG["reward"]["loss"] + 1
    assert (task.env.t == 0).all() and (task.opp_type == 0).all()  # new episodes against the stationary target
    assert not physics.com_out_of_ring(task.env.state, task.env.p).any()
