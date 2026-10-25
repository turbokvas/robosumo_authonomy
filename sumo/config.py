"""Loading robot_spec.yaml and sampling domain randomization into batched tensors.

Every spec field whose name contains `_rand` and holds a [lo, hi] pair is a uniform randomization range.
Its parameter name is the dotted path with `_rand` removed, e.g.
    robot.motors.strength_rand        -> "robot.motors.strength"
    controller.control_rate_rand_hz   -> "controller.control_rate_hz"
Parameters under `robot.` are sampled per robot (shape [num_envs, n_robots], both robots share the hardware spec),
all others per environment (shape [num_envs]).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

SPEC_PATH = Path(__file__).resolve().parent.parent / "robot_spec.yaml"
RAND_TAG = "_rand"
PER_ROBOT_PREFIX = "robot."


def load_spec(path: str | Path = SPEC_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    for name, (lo, hi) in rand_ranges(spec).items():
        if lo > hi:
            raise ValueError(f"randomization range {name}: lo={lo} > hi={hi}")
    return spec


def get(spec: dict[str, Any], dotted: str) -> Any:
    node: Any = spec
    for key in dotted.split("."):
        node = node[key]
    return node


def _is_range(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value)


def rand_ranges(spec: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """All randomization ranges in the spec as {param_name: (lo, hi)}."""
    out: dict[str, tuple[float, float]] = {}

    def walk(node: dict[str, Any], prefix: str) -> None:
        for key, value in node.items():
            if isinstance(value, dict):
                walk(value, f"{prefix}{key}.")
            elif RAND_TAG in key and _is_range(value):
                out[prefix + key.replace(RAND_TAG, "")] = (float(value[0]), float(value[1]))

    walk(spec, "")
    return out


def nominal_value(spec: dict[str, Any], name: str, lo: float, hi: float) -> float:
    """Value used when randomization is off: the base spec field if it exists, otherwise the range midpoint."""
    try:
        base = get(spec, name)
    except (KeyError, TypeError):
        base = None
    if isinstance(base, (int, float)) and not isinstance(base, bool):
        return float(base)
    return 0.5 * (lo + hi)


@dataclass
class RandomizationBatch:
    params: dict[str, torch.Tensor]

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.params[name]

    def resample(self, other: "RandomizationBatch", env_mask: torch.Tensor) -> None:
        """Copy values from `other` into envs where env_mask is True (used on env reset)."""
        for name, value in self.params.items():
            mask = env_mask.view(-1, *([1] * (value.dim() - 1)))
            value.copy_(torch.where(mask, other.params[name], value))  # in place: views and CUDA graphs stay valid


def sample_randomization(
    spec: dict[str, Any],
    num_envs: int,
    n_robots: int = 2,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
    randomize: bool | set[str] = True,
) -> RandomizationBatch:
    """Sample every randomization range into tensors.

    randomize: True — all ranges; False — all nominal; a set of names — only those, the rest nominal.
    """
    params: dict[str, torch.Tensor] = {}
    for name, (lo, hi) in rand_ranges(spec).items():
        shape = (num_envs, n_robots) if name.startswith(PER_ROBOT_PREFIX) else (num_envs,)
        on = randomize if isinstance(randomize, bool) else name in randomize
        if on:
            u = torch.rand(shape, device=device, generator=generator)
            params[name] = lo + (hi - lo) * u
        else:
            params[name] = torch.full(shape, nominal_value(spec, name, lo, hi), device=device)
    return RandomizationBatch(params)
