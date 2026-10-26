"""Actor (goes to the robot's MCU) and privileged critic (sim only), plus running observation normalization."""

from __future__ import annotations

import torch
from torch import nn


MIN_STD = 0.1  # inputs are O(1) by construction; a constant input (absent sensor, fixed battery) must not blow up later


class RunningNorm(nn.Module):
    """Running mean/var normalizer. Statistics are buffers, saved with the checkpoint and frozen for export.

    The std has a floor (MIN_STD): a reserved input that was always 0 has var 0, and dividing by sqrt(var + 1e-8)
    would turn its first real reading into a huge number."""

    def __init__(self, dim: int, clip: float = 10.0, min_std: float = MIN_STD):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))
        self.clip = clip
        self.min_std = min_std

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        b_mean, b_var, b_n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_n
        self.mean += delta * b_n / tot
        self.var = (self.var * self.count + b_var * b_n + delta**2 * self.count * b_n / tot) / tot
        self.count = tot

    def std(self) -> torch.Tensor:
        return self.var.sqrt().clamp(min=self.min_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / self.std()).clamp(-self.clip, self.clip)


def mlp(sizes: list[int], act: type[nn.Module]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), act()]
    return nn.Sequential(*layers[:-1])


class Actor(nn.Module):
    """Deterministic part runs on the robot: norm → MLP(ReLU) → tanh mean. Exploration noise only in training."""

    def __init__(
        self, obs_dim: int, hidden: list[int], act_dim: int = 2, init_log_std: float = -0.5, zero_inputs: list[int] = ()
    ):
        """zero_inputs: obs columns whose first-layer weights start at 0 (reserved sensors, interface.py)."""
        super().__init__()
        self.norm = RunningNorm(obs_dim)
        self.net = mlp([obs_dim, *hidden, act_dim], nn.ReLU)
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))
        nn.init.orthogonal_(self.net[-1].weight, 0.01)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[0].weight[:, list(zero_inputs)] = 0

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(self.norm(obs)))

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(self(obs), self.log_std.exp().expand(obs.shape[0], -1))


class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden: list[int]):
        super().__init__()
        self.norm = RunningNorm(obs_dim)
        self.net = mlp([obs_dim, *hidden, 1], nn.ELU)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(obs)).squeeze(-1)
