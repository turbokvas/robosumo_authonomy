"""SumoEnv: batched matches on the GPU. Both robot slots take [left, right] actions at the control tick.

One `step` = one control period of each env (jittered rate from the spec), split into a fixed number of physics
substeps, so envs stay in lockstep while their control periods differ. Actions go through the firmware's slew limit.
Match rules: CoM out of the ring loses; both out on the same tick or timeout → draw.
M2 scope: readings + match outcome. Observations and rewards for RL come in M3.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from sumo import physics
from sumo.config import get, sample_randomization
from sumo.sensors import Sensors

DRAW = -1


class SumoEnv:
    def __init__(
        self,
        spec: dict[str, Any],
        num_envs: int,
        device: str | torch.device = "cpu",
        randomize: bool | set[str] = True,
        start_modes: list[str] | None = None,
        seed: int | None = None,
    ):
        self.spec, self.n, self.device, self.randomize = spec, num_envs, device, randomize
        self.gen = torch.Generator(device=device)
        if seed is not None:
            self.gen.manual_seed(seed)
        self.all_modes: list[str] = get(spec, "arena.start_modes")
        self.start_modes = start_modes or self.all_modes
        self.slew = get(spec, "controller.slew_limit_per_step")
        self.timeout = get(spec, "match.timeout_s")
        self.jitter_pos = get(spec, "arena.start_jitter_pos")
        self.jitter_th = math.radians(get(spec, "arena.start_jitter_deg"))
        self.random_r = get(spec, "arena.random_start_max_r")
        self.half = tuple(x / 2 for x in get(spec, "robot.footprint"))

        self.rand = sample_randomization(spec, num_envs, device=device, generator=self.gen, randomize=randomize)
        self.p = physics.make_params(spec, self.rand)
        self.state = physics.zeros_state(num_envs, device)
        self.sensors = Sensors(spec, num_envs, device, self.gen)
        self.sensors.set_params(self.rand)
        self.substeps = self.sensors.substeps
        z = lambda *s, dtype=torch.float32: torch.zeros(num_envs, *s, device=device, dtype=dtype)  # noqa: E731
        self.t = z()
        self.action = z(physics.N_ROBOTS, 2)
        self.start_mode = z(dtype=torch.long)
        self.reset()

    @property
    def ctrl_dt(self) -> torch.Tensor:
        return 1 / self.rand["controller.control_rate_hz"]

    def _rand(self, *shape: int) -> torch.Tensor:
        return torch.rand(*shape, device=self.device, generator=self.gen)

    def _start_poses(self, k: int, modes: torch.Tensor, dist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Robot centres [k,2,2] and headings [k,2] for each start mode, then rotated about the ring centre."""
        name = lambda m: self.all_modes.index(m)  # noqa: E731
        x = torch.stack([-dist, dist], -1)
        center = torch.stack([x, torch.zeros_like(x)], -1)
        pi = math.pi
        th = torch.zeros(k, 2, device=self.device)
        th = torch.where((modes == name("face_to_face")).unsqueeze(-1), torch.tensor([0.0, pi], device=self.device), th)
        th = torch.where((modes == name("back_to_back")).unsqueeze(-1), torch.tensor([pi, 0.0], device=self.device), th)
        th = torch.where((modes == name("side_by_side")).unsqueeze(-1), torch.tensor([pi / 2, pi / 2], device=self.device), th)

        # random: uniform in a disc, re-drawn until the two robots are clear of each other
        rnd = modes == name("random")
        if rnd.any():
            n_r = int(rnd.sum())
            pos = torch.zeros(n_r, 2, 2, device=self.device)
            todo = torch.ones(n_r, dtype=torch.bool, device=self.device)
            # bounding circles apart for any headings, with room for the position jitter added below
            min_gap = 2 * math.hypot(*self.half) + 2 * math.hypot(self.jitter_pos, self.jitter_pos) + 0.01
            for _ in range(50):
                idx = todo.nonzero().squeeze(-1)
                if idx.numel() == 0:
                    break
                r = self.random_r * self._rand(idx.numel(), 2).sqrt()
                a = 2 * pi * self._rand(idx.numel(), 2)
                pos[idx] = torch.stack([r * a.cos(), r * a.sin()], -1)
                todo[idx] = (pos[idx, 0] - pos[idx, 1]).norm(dim=-1) < min_gap
            center[rnd] = pos
            th[rnd] = 2 * pi * self._rand(n_r, 2)

        rot = 2 * pi * self._rand(k, 1)
        c, s = rot.cos(), rot.sin()
        center = torch.stack([c * center[..., 0] - s * center[..., 1], s * center[..., 0] + c * center[..., 1]], -1)
        center = center + self.jitter_pos * (2 * self._rand(k, 2, 2) - 1)
        th = th + rot + self.jitter_th * (2 * self._rand(k, 2) - 1)
        return center, th

    def reset(self, env_mask: torch.Tensor | None = None) -> None:
        m = torch.ones(self.n, dtype=torch.bool, device=self.device) if env_mask is None else env_mask
        k = int(m.sum())
        if k == 0:
            return
        fresh = sample_randomization(self.spec, self.n, device=self.device, generator=self.gen, randomize=self.randomize)
        self.rand.resample(fresh, m)
        self.p = physics.make_params(self.spec, self.rand)
        self.sensors.set_params(self.rand)

        allowed = torch.tensor([self.all_modes.index(s) for s in self.start_modes], device=self.device)
        modes = allowed[torch.randint(len(allowed), (k,), device=self.device, generator=self.gen)]
        self.start_mode[m] = modes
        center = physics.centers(self.state, self.p).clone()
        theta = self.state.theta.clone()
        center[m], theta[m] = self._start_poses(k, modes, self.rand["arena.start_center_dist"][m])
        physics.set_pose(self.state, self.p, center, theta, m)
        self.sensors.prime(self.state, self.p, m)
        self.t[m] = 0
        self.action[m] = 0

    def readings(self) -> dict[str, torch.Tensor]:
        return self.sensors.read(self.ctrl_dt / self.substeps)

    def step(self, action: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """action [N,R,2] in [-1, 1] → (readings, done [N], winner [N]: 0/1 or DRAW). No auto-reset."""
        self.action = self.action + (action.clamp(-1, 1) - self.action).clamp(-self.slew, self.slew)
        dt = self.ctrl_dt / self.substeps
        for _ in range(self.substeps):
            physics.step(self.state, self.p, self.action, dt=dt)
            self.sensors.push(self.state, self.p)
        self.t += self.ctrl_dt

        out = physics.com_out_of_ring(self.state, self.p)
        lost = out.any(-1)
        timeout = self.t >= self.timeout
        done = lost | timeout
        winner = torch.full((self.n,), DRAW, device=self.device, dtype=torch.long)
        winner = torch.where(out[:, 1] & ~out[:, 0], 0, winner)
        winner = torch.where(out[:, 0] & ~out[:, 1], 1, winner)
        return self.readings(), done, winner
