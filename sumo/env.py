"""SumoEnv: batched matches on the GPU. Both robot slots take [left, right] actions at the control tick.

One `step` = one control period of each env (jittered rate from the spec), split into a fixed number of physics
substeps, so envs stay in lockstep while their control periods differ. Actions (fractions of max drive) go through
the firmware layer (sumo/firmware.py): slew limit, then deadband compensation → motor duty.
Match rules: CoM out of the ring loses; both out on the same tick or timeout → draw.
On CUDA the substep block (physics + sensor history) is captured once as a CUDA graph and replayed: the sim is
launch-bound otherwise. Everything the graph touches is updated in place (params, randomization, actions).
M2 scope: readings + match outcome. Observations and rewards for RL come in M3.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from sumo import firmware, physics
from sumo import interface as iface
from sumo.config import get, sample_randomization
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
        self.deadband_comp = get(spec, "controller.deadband_compensation")
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
        self.action = z(physics.N_ROBOTS, 2)  # applied action (after slew), what the firmware remembers
        self.duty = z(physics.N_ROBOTS, 2)    # motor duty after deadband compensation (physics input)
        self.prev_vel = z(physics.N_ROBOTS, 2)  # CoM velocity at the previous control tick (IMU)
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
        dev = self.device
        th = torch.zeros(k, 2, device=dev)
        th = torch.where((modes == name("face_to_face")).unsqueeze(-1), physics.const((0.0, pi), dev), th)
        th = torch.where((modes == name("back_to_back")).unsqueeze(-1), physics.const((pi, 0.0), dev), th)
        # side by side: which robot has the other on its right is random, else slot 0 always gets the same side
        side = torch.where(self._rand(k, 1) < 0.5, -1.0, 1.0)
        th = torch.where((modes == name("side_by_side")).unsqueeze(-1), side * physics.const((pi / 2, pi / 2), dev), th)

        # random: uniform in a disc; take the first of a few candidate pairs whose bounding circles are clear
        # (with room for the jitter added below). All candidates at once, no host sync. If none is clear
        # (~1e-3 of resets) the env keeps the fixed layout computed above.
        tries = 8
        min_gap = 2 * math.hypot(*self.half) + 2 * math.hypot(self.jitter_pos, self.jitter_pos) + 0.01
        r = self.random_r * self._rand(k, tries, 2).sqrt()
        a = 2 * pi * self._rand(k, tries, 2)
        cand = torch.stack([r * a.cos(), r * a.sin()], -1)  # [k,tries,2(robot),2]
        clear = (cand[:, :, 0] - cand[:, :, 1]).norm(dim=-1) >= min_gap
        pick = cand[torch.arange(k, device=dev), clear.float().argmax(1)]
        rnd = (modes == name("random")) & clear.any(1)
        center = torch.where(rnd.view(-1, 1, 1), pick, center)
        th = torch.where(rnd.unsqueeze(-1), 2 * pi * self._rand(k, 2), th)

        rot = 2 * pi * self._rand(k, 1)
        c, s = rot.cos(), rot.sin()
        center = torch.stack([c * center[..., 0] - s * center[..., 1], s * center[..., 0] + c * center[..., 1]], -1)
        center = center + self.jitter_pos * (2 * self._rand(k, 2, 2) - 1)
        th = th + rot + self.jitter_th * (2 * self._rand(k, 2) - 1)
        return center, th

    def reset(self, env_mask: torch.Tensor | None = None) -> None:
        if env_mask is None:
            env_mask = torch.ones(self.n, dtype=torch.bool, device=self.device)
        self.reset_idx(env_mask.nonzero().squeeze(-1))

    def reset_idx(self, idx: torch.Tensor) -> None:
        """Start new matches in envs `idx` (long). Index ops only: one host sync (the caller's nonzero) per reset."""
        k = idx.numel()
        if k == 0:
            return
        # sample and derive params only for the envs being reset (full-batch resampling dominated short episodes)
        fresh = sample_randomization(self.spec, k, device=self.device, generator=self.gen, randomize=self.randomize)
        self.rand.scatter_(fresh, idx)
        physics.scatter_params_(self.p, physics.make_params(self.spec, fresh), idx)

        allowed = physics.const(tuple(self.all_modes.index(s) for s in self.start_modes), self.device, torch.long)
        modes = allowed[torch.randint(len(allowed), (k,), device=self.device, generator=self.gen)]
        self.start_mode.index_copy_(0, idx, modes)
        center = physics.centers(self.state, self.p)
        theta = self.state.theta.clone()
        new_c, new_th = self._start_poses(k, modes, self.rand["arena.start_center_dist"][idx])
        center.index_copy_(0, idx, new_c)
        theta.index_copy_(0, idx, new_th)
        m = torch.zeros(self.n, dtype=torch.bool, device=self.device).index_fill_(0, idx, True)
        physics.set_pose(self.state, self.p, center, theta, m)
        self.sensors.prime(self.state, self.p, idx)
        self.t.index_fill_(0, idx, 0)
        for x in (self.action, self.duty, self.prev_vel):
            x.index_fill_(0, idx, 0)

    def readings(self) -> dict[str, torch.Tensor]:
        r = self.sensors.read(self.ctrl_dt / self.substeps)
        r.update(self.sensors.motion(self.state, self.prev_vel, self.ctrl_dt))
        return r

    def _substeps(self) -> None:
        dt = self.ctrl_dt / self.substeps
        for _ in range(self.substeps):
            physics.step(self.state, self.p, self.duty, dt=dt)
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
        self.action.copy_(firmware.slew(self.action, action, self.slew))
        if self.deadband_comp:
            self.duty.copy_(firmware.deadband_comp(self.action, self.p.deadband_cal.unsqueeze(-1)))
        else:
            self.duty.copy_(self.action)
        self.prev_vel.copy_(self.state.vel)
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


class SumoTask:
    """RL view of a match: robot 0 = learner, robot 1 = opponent.

    Opponent types: "stationary", the FSM variants, and (when a policy pool is attached, M4) frozen policy
    snapshots "pool0".."poolK-1" that see robot 1's own actor obs.
    Actor obs and action follow the frozen interface (sumo/interface.py): only what the robot's MCU has.
    Critic obs = privileged state of both robots, their physical parameters, opponent type, FSM mode/params.
    Reward = terminal win/loss/draw + shaping (approach, push the opponent off centre while close, action jerk);
    approach/push are multiplied by `shaping_scale`, which the trainer anneals to 0.
    Reserved sensor groups are present/absent per robot per match: `sensors` = "train" (random: cfg
    env.sensor_dropout gives, per group, the probability it is absent, and `bare` the probability that all of them
    are absent at once, i.e. today's hardware), "installed" (spec `installed` flags), "all" or "none".
    Finished envs auto-reset; the obs returned for them belong to the new episode.
    """

    def __init__(
        self,
        spec: dict[str, Any],
        cfg: dict[str, Any],
        num_envs: int,
        device: str | torch.device,
        seed: int,
        pool: Any = None,
        sensors: str = "train",
    ):
        """pool: object with `capacity`, `size` and `act(obs [N,D], slot [N]) -> [N,2]` (train.league.PolicyPool)."""
        iface.check_spec(spec)
        self.spec, self.n, self.device = spec, num_envs, device
        if cfg["env"].get("sim", "2d") == "3d":
            from sim3d.env3d import SumoEnv3D  # MuJoCo Warp; imported only when used

            self.env = SumoEnv3D(spec, num_envs, device, randomize=cfg["env"]["randomize"], seed=seed)
        else:
            self.env = SumoEnv(spec, num_envs, device, randomize=cfg["env"]["randomize"], seed=seed)
        self.fsm = FSM(spec, num_envs, device, None, self.env.gen)
        self.pool = pool
        self.n_fsm = len(self.fsm.variants)
        self.opp_names = [STATIONARY] + self.fsm.variants + [f"pool{i}" for i in range(pool.capacity if pool else 0)]
        self.opp_weights = torch.ones(len(self.opp_names), device=device)
        self.opp_type = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.frames = torch.zeros(num_envs, physics.N_ROBOTS, iface.HISTORY_LEN, iface.FRAME_DIM, device=device)
        self.lag_idx = torch.tensor([iface.HISTORY_LEN - 1 - lag for lag in iface.FRAME_LAGS], device=device)
        self.memory = firmware.SensorMemory(num_envs, physics.N_ROBOTS, device)
        groups = list(iface.SENSOR_GROUPS)
        if sensors == "train":
            drop = cfg["env"].get("sensor_dropout", {})
            self.absent_p = torch.tensor([drop.get(g, 1.0) for g in groups], device=device)
            self.bare_p = drop.get("bare", 0.0)
        elif sensors == "installed":
            sens = get(spec, "robot.sensors")
            self.absent_p = torch.tensor([0.0 if sens[g]["installed"] else 1.0 for g in groups], device=device)
        elif sensors in ("all", "none"):
            self.absent_p = torch.full((len(groups),), 0.0 if sensors == "all" else 1.0, device=device)
        else:
            raise ValueError(f"sensors={sensors!r}")
        if sensors != "train":
            self.bare_p = 0.0
        self.present = torch.zeros(num_envs, physics.N_ROBOTS, len(groups), device=device)
        self.rew = cfg["reward"]
        self.shaping_scale = 1.0
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
        idx = m.nonzero().squeeze(-1)  # the only host sync of a reset
        k = idx.numel()
        if k == 0:
            return
        if reset_env:
            self.env.reset_idx(idx)
        self.fsm.reset_idx(idx)
        opp = torch.multinomial(self.opp_weights, k, replacement=True, generator=self.env.gen)
        self.opp_type.index_copy_(0, idx, opp)
        self.fsm.variant.index_copy_(0, idx, (opp - 1).clamp(0, self.n_fsm - 1))
        u = torch.rand(k, physics.N_ROBOTS, len(self.absent_p), device=self.device, generator=self.env.gen)
        bare = torch.rand(k, physics.N_ROBOTS, 1, device=self.device, generator=self.env.gen) < self.bare_p
        self.present.index_copy_(0, idx, ((u >= self.absent_p) & ~bare).float())
        fresh = self.env.readings()
        for key in self.readings:
            mm = m.view(-1, *([1] * (fresh[key].dim() - 1)))
            self.readings[key] = torch.where(mm, fresh[key], self.readings[key])
        self.frames = torch.where(m.view(-1, 1, 1, 1), torch.zeros_like(self.frames), self.frames)
        self.memory.reset(idx)
        self._push_frame(m)  # first frame of the new episodes only: other envs already got this tick's frame
        self.prev_raw = torch.where(m.view(-1, 1), torch.zeros_like(self.prev_raw), self.prev_raw)
        self.ep_len = torch.where(m, torch.zeros_like(self.ep_len), self.ep_len)
        self.prev_dist, self.prev_opp_r = self._geometry()

    # --- observations ----------------------------------------------------------------------------------------------
    def _frame(self) -> torch.Tensor:
        """Current MCU frame of both robots per the frozen interface, [N,R,FRAME_DIM]."""
        r, pres = self.readings, self.present
        side, enc, imu = pres[..., 0:1], pres[..., 1:2], pres[..., 2:3]
        parts = {
            "ir_front": r["ir"].float(),
            "line": r["line"].float(),
            "prev_action": self.env.action,
            "battery": r["battery"].unsqueeze(-1),
            "ir_side": r["ir_side"].float() * side,
            "wheel_speed": r["wheel_speed"] / iface.WHEEL_SPEED_SCALE * enc,
            "gyro_z": r["gyro_z"].unsqueeze(-1) / iface.GYRO_SCALE * imu,
            "accel_xy": r["accel_xy"] / iface.ACCEL_SCALE * imu,
            "has_side_ir": side,
            "has_encoders": enc,
            "has_imu": imu,
        }
        return torch.cat([parts[name] for name, _ in iface.FRAME], -1)

    def _push_frame(self, m: torch.Tensor | None = None) -> None:
        """New control tick: frame history and the firmware memory (only what the MCU has); only envs in m if given.

        Regression: resets used to push a second frame into every env, so with thousands of envs (some reset
        nearly every tick) the history was full of duplicated frames."""
        r = self.readings
        mem = self.memory
        old = None if m is None else (mem.ir_age.clone(), mem.line_age.clone(), mem.last_side.clone(), mem.engaged.clone())
        mem.update(r["ir"], r["ir_side"] * self.present[..., 0:1], r["line"], self.env.ctrl_dt)
        frames = torch.cat([self.frames[:, :, 1:], self._frame().unsqueeze(2)], 2)
        if m is None:
            self.frames = frames
            return
        self.frames = torch.where(m.view(-1, 1, 1, 1), frames, self.frames)
        for name, prev in zip(("ir_age", "line_age", "last_side", "engaged"), old):
            cur = getattr(mem, name)
            setattr(mem, name, torch.where(m.view(-1, *([1] * (cur.dim() - 1))), cur, prev))

    def obs(self, slot: int = 0) -> torch.Tensor:
        """Actor obs of robot `slot` (its own sensors only), [N, OBS_DIM]: lagged frames, then memory."""
        frames = self.frames[:, slot].index_select(1, self.lag_idx).flatten(1)
        return torch.cat([frames, self.memory.features()[:, slot]], -1)

    def opp_category(self) -> torch.Tensor:
        """0 = stationary, 1..n_fsm = FSM variant, n_fsm+1 = any pool policy."""
        return self.opp_type.clamp(max=self.n_fsm + 1)

    def critic_obs(self) -> torch.Tensor:
        """Privileged critic input (layout fixed by sumo/interface.py: same with or without a policy pool)."""
        env, st, p = self.env, self.env.state, self.env.p
        R = p.ring_radius.unsqueeze(-1)  # [N,1]
        ctr = physics.centers(st, p)
        c, s = st.theta.cos(), st.theta.sin()
        v_ref = (p.omega0 * p.wheel_r).unsqueeze(-1)
        rel = ctr[:, 1] - ctr[:, 0]
        rel_own = torch.stack([c[:, 0] * rel[:, 0] + s[:, 0] * rel[:, 1], -s[:, 0] * rel[:, 0] + c[:, 0] * rel[:, 1]], -1)
        rel_opp = torch.stack([-(c[:, 1] * rel[:, 0] + s[:, 1] * rel[:, 1]), s[:, 1] * rel[:, 0] - c[:, 1] * rel[:, 1]], -1)
        dth = st.theta[:, 1] - st.theta[:, 0]
        is_fsm = self._is_fsm().float().unsqueeze(-1)
        fsm_prm = torch.stack([self.fsm.prm[k] for k in sorted(self.fsm.prm)], -1) * is_fsm
        parts = [
            (ctr / R.unsqueeze(-1)).flatten(1), c, s, (st.vel / v_ref).flatten(1), st.omega / 10,
            (st.wheel_w / p.omega0.unsqueeze(-1)).flatten(1), st.u_lag.flatten(1), st.lift, st.wedge,
            ctr.norm(dim=-1) / R, rel_own / R, rel_opp / R, torch.stack([dth.cos(), dth.sin()], -1),
            rel.norm(dim=-1, keepdim=True) / R, (1 - env.t / env.timeout).unsqueeze(-1),
            torch.nn.functional.one_hot(self.opp_category(), self.n_fsm + 2).float(),
            torch.nn.functional.one_hot(self.fsm.mode, iface.N_FSM_MODES).float() * is_fsm,
            fsm_prm, self.present.flatten(1), 1 / env.ctrl_dt.unsqueeze(-1) / 100,
            *[getattr(p, name) for name in iface.CRITIC_ROBOT_PARAMS],
            p.tau_stall.flatten(1),
            *[getattr(p, name).unsqueeze(-1) for name in iface.CRITIC_ENV_PARAMS],
            env.sensors.ir_range, env.sensors.ir_latency * 100, env.sensors.line_latency * 100,
            self.obs(),
        ]
        return torch.cat(parts, -1)

    # --- step ------------------------------------------------------------------------------------------------------
    def _is_fsm(self) -> torch.Tensor:
        return (self.opp_type > 0) & (self.opp_type <= self.n_fsm)

    def _geometry(self) -> tuple[torch.Tensor, torch.Tensor]:
        ctr = physics.centers(self.env.state, self.env.p)
        return (ctr[:, 0] - ctr[:, 1]).norm(dim=-1), ctr[:, 1].norm(dim=-1)

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """action [N,2] for robot 0 → (reward [N], done [N], info). Obs for the next step via obs()/critic_obs()."""
        raw = action.clamp(-1, 1)
        opp = self.fsm.act(self.readings["ir"][:, 1], self.readings["line"][:, 1], self.env.ctrl_dt)
        if self.pool is not None and self.pool.size > 0:
            slot = (self.opp_type - self.n_fsm - 1).clamp(min=0)
            pol = self.pool.act(self.obs(1), slot)
            opp = torch.where((self.opp_type > self.n_fsm).unsqueeze(-1), pol, opp)
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
