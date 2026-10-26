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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need CUDA")
def test_cuda_graph_matches_eager(spec):
    """The captured substep block must give the same trajectory as eager execution, across resets."""
    envs = [SumoEnv(spec, 64, "cuda", seed=7, use_graph=g) for g in (False, True)]
    gen = torch.Generator("cuda").manual_seed(1)
    for k in range(150):
        action = torch.rand(64, 2, 2, device="cuda", generator=gen) * 2 - 1
        outs = [e.step(action) for e in envs]
        if k % 50 == 49:
            mask = torch.arange(64, device="cuda") % 3 == 0
            for e in envs:
                e.reset(mask)
    a, b = envs
    for x, y in zip(a._sim_tensors(), b._sim_tensors()):
        assert torch.allclose(x.float(), y.float(), atol=1e-5)
    assert torch.equal(outs[0][1], outs[1][1])


def test_mirror_symmetry(spec):
    """Mirrored starts + identical commands must stay mirrored, also through contacts. Regression: float noise in
    near-tie wedge contests used to hand the win to robot 0, which a policy trained in slot 0 learned to exploit."""
    n = 128
    env = SumoEnv(spec, n, DEVICE, randomize=False, seed=0, use_graph=False)
    env.sensors.fn_p = env.sensors.fp_p = env.sensors.xt_p = 0.0
    g = torch.Generator(DEVICE).manual_seed(0)
    ang = torch.rand(n, device=DEVICE, generator=g) * 2 * math.pi
    c0 = 0.25 * torch.stack([-ang.cos(), -ang.sin()], -1)
    th0 = ang + (torch.rand(n, device=DEVICE, generator=g) - 0.5) * 0.6
    physics.set_pose(env.state, env.p, torch.stack([c0, -c0], 1), torch.stack([th0, th0 + math.pi], 1))
    env.sensors.prime(env.state, env.p)
    ir_diff = 0.0
    for _ in range(250):
        a = 0.3 + 0.7 * torch.rand(n, 2, device=DEVICE, generator=g)
        r, _, _ = env.step(torch.stack([a, a], 1))
        ir_diff += (r["ir"][:, 0] != r["ir"][:, 1]).float().mean().item() / 250
    assert ir_diff < 0.005  # only beams grazing a body edge may flip on 1e-6 m differences
    c = physics.centers(env.state, env.p)
    assert (c[:, 0] + c[:, 1]).norm(dim=-1).max() < 1e-4
    assert torch.allclose(env.state.lift[:, 0], env.state.lift[:, 1], atol=1e-4)


def test_side_by_side_sides_are_random(spec):
    """Regression: side_by_side always put the opponent on robot 0's right, a slot-0 advantage of ~98% in self-play."""
    env = SumoEnv(spec, 2048, DEVICE, start_modes=["side_by_side"], seed=0)
    ctr = physics.centers(env.state, env.p)
    to_other = ctr[:, 1] - ctr[:, 0]
    fwd0 = torch.stack([env.state.theta[:, 0].cos(), env.state.theta[:, 0].sin()], -1)
    on_right = (fwd0[:, 0] * to_other[:, 1] - fwd0[:, 1] * to_other[:, 0]) < 0
    assert 0.45 < on_right.float().mean().item() < 0.55
