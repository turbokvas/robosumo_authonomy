import math

import pytest
import torch

from sumo import physics
from sumo.config import get, load_spec
from sumo.env import DRAW, SumoEnv
from sumo.match import run_matches

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def spec():
    return load_spec()


@pytest.mark.parametrize("mode", ["face_to_face", "back_to_back", "side_by_side", "random"])
def test_start_poses(spec, mode):
    env = SumoEnv(spec, 2048, DEVICE, start_modes=[mode], seed=0)
    ctr = physics.centers(env.state, env.p)
    assert (physics.box_contacts(ctr, env.state.theta, env.p.half)[2] == 0).all()  # no overlap
    assert not physics.com_out_of_ring(env.state, env.p).any()
    to_other = ctr[:, 1] - ctr[:, 0]
    fwd0 = torch.stack([env.state.theta[:, 0].cos(), env.state.theta[:, 0].sin()], -1)
    facing = (fwd0 * to_other).sum(-1) / to_other.norm(dim=-1)
    jitter = math.radians(get(spec, "arena.start_jitter_deg"))
    if mode == "face_to_face":
        assert (facing > math.cos(2 * jitter + 0.1)).all()
    if mode == "back_to_back":
        assert (facing < -math.cos(2 * jitter + 0.1)).all()


def test_ring_out_wins_and_timeout_draws(spec):
    env = SumoEnv(spec, 2, DEVICE, randomize=False, start_modes=["face_to_face"], seed=0)
    r = env.p.ring_radius[0].item()
    ctr = torch.tensor([[[0.0, 0.0], [r + 0.2, 0.0]], [[-0.4, 0.0], [0.4, 0.0]]], device=DEVICE)
    physics.set_pose(env.state, env.p, ctr, torch.zeros(2, 2, device=DEVICE))
    _, done, winner = env.step(torch.zeros(2, 2, 2, device=DEVICE))
    assert done.tolist() == [True, False]
    assert winner[0] == 0
    env.t[:] = get(spec, "match.timeout_s")
    _, done, winner = env.step(torch.zeros(2, 2, 2, device=DEVICE))
    assert done[1] and winner[1] == DRAW


def test_slew_limit(spec):
    env = SumoEnv(spec, 4, DEVICE, seed=0)
    slew = get(spec, "controller.slew_limit_per_step")
    env.step(torch.ones(4, 2, 2, device=DEVICE))
    assert torch.allclose(env.action, torch.full_like(env.action, slew))
    env.step(-torch.ones(4, 2, 2, device=DEVICE))
    assert torch.allclose(env.action, torch.zeros_like(env.action), atol=1e-6)


def test_reset_only_masked_envs(spec):
    env = SumoEnv(spec, 8, DEVICE, seed=0)
    for _ in range(20):
        env.step(torch.ones(8, 2, 2, device=DEVICE))
    before = env.state.pos.clone()
    mask = torch.zeros(8, dtype=torch.bool, device=DEVICE)
    mask[:3] = True
    env.reset(mask)
    assert torch.equal(env.state.pos[3:], before[3:])
    assert (env.t[:3] == 0).all() and (env.t[3:] > 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (thousands of matches)")
def test_aggressive_beats_random_walker(spec):
    """PLAN M2 'done' criterion, on nominal hardware (all *_rand come in M5)."""
    r = run_matches(spec, "aggressive", "random_walker", 2048, "cuda", seed=0, randomize=False)
    print(f"\naggressive vs random_walker: {r['a_wins']:.1%} wins, {r['draws']:.1%} draws")
    assert r["a_wins"] > 0.8
