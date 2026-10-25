"""SumoEnv: batched matches on the GPU. Both robot slots take [left, right] actions at the control tick.

One `step` = one control period of each env (jittered rate from the spec), split into a fixed number of physics
substeps, so envs stay in lockstep while their control periods differ. Actions go through the firmware's slew limit.
Match rules: CoM out of the ring loses; both out on the same tick or timeout → draw.
On CUDA the substep block (physics + sensor history) is captured once as a CUDA graph and replayed: the sim is
launch-bound otherwise. Everything the graph touches is updated in place (params, randomization, actions).
M2 scope: readings + match outcome. Observations and rewards for RL come in M3.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from sumo import physics
from sumo.config import get, rand_ranges, sample_randomization
from sumo.opponents import FSM
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
        use_graph: bool = True,
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
        self.use_graph = use_graph and torch.device(device).type == "cuda"
        self._graph: torch.cuda.CUDAGraph | None = None
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
        physics.copy_params_(self.p, physics.make_params(self.spec, self.rand))

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

    def _substeps(self) -> None:
        dt = self.ctrl_dt / self.substeps
        for _ in range(self.substeps):
            physics.step(self.state, self.p, self.action, dt=dt)
            self.sensors.push(self.state, self.p)
        self.t.add_(self.ctrl_dt)

    def _sim_tensors(self) -> list[torch.Tensor]:
        st = self.state
        return [st.pos, st.theta, st.vel, st.omega, st.wheel_w, st.u_lag, st.lift, st.wedge,
                self.sensors.hist, self.sensors.head, self.t]

    def _capture(self) -> None:
        """Warm up on a side stream, capture `_substeps`, then roll the sim back to where it was."""
        saved = [x.clone() for x in self._sim_tensors()]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._substeps()
        torch.cuda.current_stream().wait_stream(side)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._substeps()
        for x, s in zip(self._sim_tensors(), saved):
            x.copy_(s)

    def step(self, action: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """action [N,R,2] in [-1, 1] → (readings, done [N], winner [N]: 0/1 or DRAW). No auto-reset."""
        self.action.add_((action.clamp(-1, 1) - self.action).clamp(-self.slew, self.slew))
        if self.use_graph:
            if self._graph is None:
                self._capture()
            self._graph.replay()
        else:
            self._substeps()

        out = physics.com_out_of_ring(self.state, self.p)
        lost = out.any(-1)
        timeout = self.t >= self.timeout
        done = lost | timeout
        winner = torch.full((self.n,), DRAW, device=self.device, dtype=torch.long)
        winner = torch.where(out[:, 1] & ~out[:, 0], 0, winner)
        winner = torch.where(out[:, 0] & ~out[:, 1], 1, winner)
        return self.readings(), done, winner


STATIONARY = "stationary"
FRAME_DIM = 10  # 3 IR + 4 line + previous action (2) + battery


class SumoTask:
    """RL view of a match: robot 0 = learner, robot 1 = opponent (an FSM variant or a stationary target).

    Actor obs = only what the robot's MCU has: frame stack of [3 IR, 4 line, previous applied action (2), battery].
    Critic obs = privileged state of both robots, randomization, opponent type and FSM mode (sim only).
    Reward = terminal win/loss/draw + shaping (approach, push the opponent off centre while close, action jerk);
    approach/push are multiplied by `shaping_scale`, which the trainer anneals to 0.
    Finished envs auto-reset; the obs returned for them belong to the new episode.
    """

    def __init__(self, spec: dict[str, Any], cfg: dict[str, Any], num_envs: int, device: str | torch.device, seed: int):
        self.spec, self.n, self.device = spec, num_envs, device
        self.env = SumoEnv(spec, num_envs, device, randomize=cfg["env"]["randomize"], seed=seed)
        self.fsm = FSM(spec, num_envs, device, None, self.env.gen)
        self.opp_names = [STATIONARY] + self.fsm.variants
        self.opp_weights = torch.ones(len(self.opp_names), device=device)
        self.opp_type = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.stack = get(spec, "controller.policy.obs_frame_stack")
        self.frames = torch.zeros(num_envs, self.stack, FRAME_DIM, device=device)
        self.rew = cfg["reward"]
        self.shaping_scale = 1.0
        self.ranges = sorted(rand_ranges(spec).items())
        motors = get(spec, "robot.motors.est_16v")
        self.omega0 = motors["no_load_rpm"] * 2 * math.pi / 60
        self.prev_raw = torch.zeros(num_envs, 2, device=device)
        self.readings = self.env.readings()
        self.ep_len = torch.zeros(num_envs, device=device)
        self._new_episodes(torch.ones(num_envs, dtype=torch.bool, device=device), reset_env=False)

    # --- opponents -------------------------------------------------------------------------------------------------
    def set_opponents(self, weights: dict[str, float]) -> None:
        """Sampling weights over opponent types for the next resets, e.g. {"stationary": 0.2, "random_walker": 0.8}."""
        w = torch.zeros(len(self.opp_names), device=self.device)
        for name, value in weights.items():
            w[self.opp_names.index(name)] = value
        self.opp_weights = w

    def _new_episodes(self, m: torch.Tensor, reset_env: bool = True) -> None:
        k = int(m.sum())
        if k == 0:
            return
        if reset_env:
            self.env.reset(m)
        self.fsm.reset(m)
        self.opp_type[m] = torch.multinomial(self.opp_weights, k, replacement=True, generator=self.env.gen)
        self.fsm.variant[m] = (self.opp_type[m] - 1).clamp(min=0)
        fresh = self.env.readings()
        for key in self.readings:
            mm = m.view(-1, *([1] * (fresh[key].dim() - 1)))
            self.readings[key] = torch.where(mm, fresh[key], self.readings[key])
        self.frames[m] = 0
        self.prev_raw[m] = 0
        self.ep_len[m] = 0
        self._push_frame()
        self.prev_dist, self.prev_opp_r = self._geometry()

    # --- observations ----------------------------------------------------------------------------------------------
    def _frame(self) -> torch.Tensor:
        r = self.readings
        return torch.cat([r["ir"][:, 0].float(), r["line"][:, 0].float(), self.env.action[:, 0], r["battery"][:, :1]], -1)

    def _push_frame(self) -> None:
        self.frames = torch.cat([self.frames[:, 1:], self._frame().unsqueeze(1)], 1)

    def obs(self) -> torch.Tensor:
        return self.frames.flatten(1)

    def critic_obs(self) -> torch.Tensor:
        env, st, p = self.env, self.env.state, self.env.p
        R = p.ring_radius.unsqueeze(-1)  # [N,1]
        ctr = physics.centers(st, p)
        c, s = st.theta.cos(), st.theta.sin()
        v_ref = self.omega0 * p.wheel_r
        rel = ctr[:, 1] - ctr[:, 0]
        rel_own = torch.stack([c[:, 0] * rel[:, 0] + s[:, 0] * rel[:, 1], -s[:, 0] * rel[:, 0] + c[:, 0] * rel[:, 1]], -1)
        rel_opp = torch.stack([-(c[:, 1] * rel[:, 0] + s[:, 1] * rel[:, 1]), s[:, 1] * rel[:, 0] - c[:, 1] * rel[:, 1]], -1)
        dth = st.theta[:, 1] - st.theta[:, 0]
        parts = [
            (ctr / R.unsqueeze(-1)).flatten(1), c, s, (st.vel / v_ref).flatten(1), st.omega / 10,
            (st.wheel_w / self.omega0).flatten(1), st.u_lag.flatten(1), st.lift, st.wedge,
            ctr.norm(dim=-1) / R, rel_own / R, rel_opp / R, torch.stack([dth.cos(), dth.sin()], -1),
            rel.norm(dim=-1, keepdim=True) / R, (1 - env.t / env.timeout).unsqueeze(-1),
            torch.nn.functional.one_hot(self.opp_type, len(self.opp_names)).float(),
            torch.nn.functional.one_hot(self.fsm.mode, 7).float() * (self.opp_type > 0).float().unsqueeze(-1),
            self.obs(),
        ]
        for name, (lo, hi) in self.ranges:
            if hi > lo:
                v = (env.rand[name] - lo) / (hi - lo)
                parts.append(v if v.dim() == 2 else v.unsqueeze(-1))
        return torch.cat(parts, -1)

    # --- step ------------------------------------------------------------------------------------------------------
    def _geometry(self) -> tuple[torch.Tensor, torch.Tensor]:
        ctr = physics.centers(self.env.state, self.env.p)
        return (ctr[:, 0] - ctr[:, 1]).norm(dim=-1), ctr[:, 1].norm(dim=-1)

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """action [N,2] for robot 0 → (reward [N], done [N], info). Obs for the next step via obs()/critic_obs()."""
        raw = action.clamp(-1, 1)
        opp = self.fsm.act(self.readings["ir"][:, 1], self.readings["line"][:, 1], self.env.ctrl_dt)
        opp = torch.where((self.opp_type == 0).unsqueeze(-1), torch.zeros_like(opp), opp)
        self.readings, done, winner = self.env.step(torch.stack([raw, opp], 1))
        self.ep_len += 1

        dist, opp_r = self._geometry()
        close = dist < self.rew["push_gate_dist"]
        shaping = self.rew["approach"] * (self.prev_dist - dist) + self.rew["push"] * torch.where(
            close, opp_r - self.prev_opp_r, torch.zeros_like(opp_r)
        )
        jerk = ((raw - self.prev_raw) ** 2).sum(-1)
        reward = self.shaping_scale * shaping - self.rew["jerk"] * jerk
        terminal = torch.where(winner == 0, self.rew["win"], torch.where(winner == 1, self.rew["loss"], self.rew["draw"]))
        reward = reward + torch.where(done, terminal, torch.zeros_like(reward))
        self.prev_dist, self.prev_opp_r, self.prev_raw = dist, opp_r, raw

        info = {"winner": winner, "opp_type": self.opp_type.clone(), "ep_len": self.ep_len.clone()}
        self._push_frame()
        self._new_episodes(done)
        return reward, done, info
