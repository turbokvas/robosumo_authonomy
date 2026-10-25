import pytest
import torch

from sumo.config import load_spec
from sumo.opponents import ATTACK, ESC_BACK, ESC_TURN, REPO_BACK, SEARCH, FSM


@pytest.fixture(scope="module")
def spec():
    return load_spec()


def b(*bits):
    return torch.tensor([[bool(x) for x in bits]])


DT = torch.tensor([0.01])
NO_IR, NO_LINE = b(0, 0, 0), b(0, 0, 0, 0)


def fsm(spec, variant):
    return FSM(spec, 1, "cpu", variant, torch.Generator().manual_seed(0))


@pytest.mark.parametrize("variant", ["aggressive", "cautious_edge_waiter", "spinner"])
def test_attacks_what_it_sees(spec, variant):
    f = fsm(spec, variant)
    left_only = f.act(b(1, 0, 0), NO_LINE, DT)[0]
    assert left_only[1] > left_only[0]  # steer left
    right_only = f.act(b(0, 0, 1), NO_LINE, DT)[0]
    assert right_only[0] > right_only[1]
    centre = f.act(b(0, 1, 0), NO_LINE, DT)[0]
    assert (centre > 0).all() and f.mode[0] == ATTACK


def test_front_left_line_backs_up_then_turns_right(spec):
    f = fsm(spec, "aggressive")
    cmd = f.act(NO_IR, b(1, 0, 0, 0), DT)[0]
    assert (cmd < 0).all() and f.mode[0] == ESC_BACK
    for _ in range(100):
        cmd = f.act(NO_IR, NO_LINE, DT)[0]
        if f.mode[0] == ESC_TURN:
            break
    assert f.mode[0] == ESC_TURN
    assert cmd[0] > 0 > cmd[1]  # spins right, away from the left edge


def test_push_through_only_for_brave_variants(spec):
    brave = fsm(spec, "aggressive")
    assert (brave.act(b(0, 1, 0), b(1, 1, 0, 0), DT)[0] > 0).all()  # pushing it out, keep going
    cautious = fsm(spec, "cautious_edge_waiter")
    assert (cautious.act(b(0, 1, 0), b(1, 1, 0, 0), DT)[0] < 0).all()


def test_random_walker_ignores_ir(spec):
    f = fsm(spec, "random_walker")
    for _ in range(50):
        f.act(b(1, 1, 1), NO_LINE, DT)
        assert f.mode[0] == SEARCH


def test_clinch_triggers_reposition(spec):
    f = fsm(spec, "aggressive")
    steps = int(f.prm["attack_timeout_s"][0] / DT[0]) + 5
    modes = [int(f.mode[0]) for _ in range(steps) if f.act(b(0, 1, 0), NO_LINE, DT) is not None]
    assert REPO_BACK in modes


def test_variants_are_vectorized(spec):
    f = FSM(spec, 512, "cpu", None, torch.Generator().manual_seed(1))
    assert len(set(f.variant.tolist())) == 4
    cmd = f.act(torch.zeros(512, 3, dtype=torch.bool), torch.zeros(512, 4, dtype=torch.bool), torch.full((512,), 0.01))
    assert cmd.shape == (512, 2) and cmd.abs().max() <= 1
