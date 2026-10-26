import pytest
import torch
import yaml

from sumo import physics
from sumo.config import get, load_spec
from sumo.env import FRAME_DIM, SumoTask

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CFG = yaml.safe_load(open("configs/m3.yaml", encoding="utf-8"))


@pytest.fixture(scope="module")
def spec():
    return load_spec()


def test_obs_shapes_and_actor_sees_only_mcu_signals(spec):
    task = SumoTask(spec, CFG, 32, DEVICE, seed=0)
    stack = get(spec, "controller.policy.obs_frame_stack")
    obs = task.obs()
    assert obs.shape == (32, stack * FRAME_DIM)
    frame = obs.view(32, stack, FRAME_DIM)[:, -1]
    assert torch.all((frame[:, :7] == 0) | (frame[:, :7] == 1))  # IR + line are digital
    assert torch.all(frame[:, 9] > 0.8)  # battery V / full_v
    assert torch.isfinite(task.critic_obs()).all()


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
