import pytest
import torch

from sumo.config import get, load_spec, rand_ranges, sample_randomization

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(scope="module")
def spec():
    return load_spec()


def test_spec_loads(spec):
    assert get(spec, "robot.mass") == 3.0
    assert get(spec, "controller.physics_dt_s") == 0.002


def test_rand_names(spec):
    names = rand_ranges(spec)
    assert names["robot.mass"] == (2.85, 3.0)
    assert names["controller.control_rate_hz"] == (80.0, 120.0)
    assert names["arena.ring_diameter"] == (1.55, 1.65)
    assert "robot.sensors.opponent_ir.latency_s" in names
    assert "opponent.params" not in names  # bool flag, not a range


@pytest.mark.parametrize("device", DEVICES)
def test_sampling_within_ranges(spec, device):
    n = 4096
    g = torch.Generator(device=device).manual_seed(0)
    batch = sample_randomization(spec, n, device=device, generator=g)
    for name, (lo, hi) in rand_ranges(spec).items():
        v = batch[name]
        assert v.device.type == device
        assert v.shape == ((n, 2) if name.startswith("robot.") else (n,))
        assert v.min() >= lo and v.max() <= hi
        if hi > lo:  # actually spread over the range, not stuck at one value
            assert v.max() - v.min() > 0.9 * (hi - lo)


def test_sampling_deterministic(spec):
    a = sample_randomization(spec, 64, generator=torch.Generator().manual_seed(1))
    b = sample_randomization(spec, 64, generator=torch.Generator().manual_seed(1))
    for name in a.params:
        assert torch.equal(a[name], b[name])


def test_robots_sampled_independently(spec):
    batch = sample_randomization(spec, 256, generator=torch.Generator().manual_seed(2))
    m = batch["robot.motors.strength"]
    assert not torch.equal(m[:, 0], m[:, 1])


def test_nominal(spec):
    batch = sample_randomization(spec, 8, randomize=False)
    assert torch.all(batch["robot.mass"] == 3.0)  # base field exists
    assert torch.all(batch["controller.control_rate_hz"] == 100.0)
    assert torch.allclose(batch["robot.motors.left_right_mismatch"], torch.tensor(1.0))  # midpoint


def test_partial_randomize_and_resample(spec):
    batch = sample_randomization(spec, 16, randomize={"robot.mass"}, generator=torch.Generator().manual_seed(3))
    assert torch.all(batch["arena.ring_diameter"] == 1.60)  # base field, not the range
    fresh = sample_randomization(spec, 16, generator=torch.Generator().manual_seed(4))
    mask = torch.zeros(16, dtype=torch.bool)
    mask[:4] = True
    old = batch["robot.mass"].clone()
    batch.resample(fresh, mask)
    assert torch.equal(batch["robot.mass"][:4], fresh["robot.mass"][:4])
    assert torch.equal(batch["robot.mass"][4:], old[4:])
