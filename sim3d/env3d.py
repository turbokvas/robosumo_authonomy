"""SumoEnv3D: the 3D (MuJoCo Warp) twin of sumo.env.SumoEnv, with the same interface.

SumoTask, the league, PPO and eval run on top of it unchanged (cfg `env.sim: 3d`). What is shared with 2D:
the spec, randomization batch and the derived parameters (physics.make_params: masses, CoM, friction, motors —
pushed into the MuJoCo model per world), start poses, the firmware layer (slew, deadband compensation), motor
deadband and lag, sensor latency/noise/crosstalk, encoder/IMU models, match rules. What is 3D: rigid-body contact
of the CAD geometry (wedge lifting, wheels leaving the ground, falling off the platform), E18 beams as real rays.

The 2D-compatible `state` is a view refreshed every control tick: pos = CoM xy, theta = yaw, vel = CoM velocity xy,
omega = yaw rate, wheel_w, u_lag; `lift` = how far the CoM is above its resting height (/ 0.05 m) and `wedge` =
body tilt (rad) — 3D stand-ins for the 2D lift state, used by the critic and the renderer only.
Sensor latency is modelled at control-tick resolution (the 2D sim does it per physics substep).
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import mujoco_warp as mjw
import torch
import warp as wp
from mujoco_warp._src.types import vec6f  # geomgroup mask type of mjw.rays

from sim3d.model import load_geometry, scene_xml
from sumo import firmware, physics
from sumo.config import get, sample_randomization
from sumo.env import DRAW, SumoEnv
from sumo.sensors import Sensors

ROBOTS = ("a", "b")


def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v [...,3] by unit quaternions q [...,4] (w, x, y, z)."""
    w, u = q[..., :1], q[..., 1:]
    t = 2 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


class Sensors3D(Sensors):
    """Same sensor models as 2D; the geometry comes from MuJoCo rays and 3D poses. Pushed once per control tick."""

    def __init__(self, spec: dict[str, Any], n_envs: int, device: str, generator: torch.Generator | None, env: "SumoEnv3D"):
        super().__init__(spec, n_envs, device, generator)
        self.env = env
        self.mount_h = get(spec, "robot.sensors.opponent_ir.mount_height")
        n_ir = self.ir_xy.shape[0]
        self.n_ray = physics.N_ROBOTS * n_ir * len(self.beam_offsets)
        self.ray_pnt = wp.zeros((n_envs, self.n_ray), dtype=wp.vec3)
        self.ray_vec = wp.zeros((n_envs, self.n_ray), dtype=wp.vec3)
        self.ray_dist = wp.zeros((n_envs, self.n_ray), dtype=float)
        self.ray_geom = wp.zeros((n_envs, self.n_ray), dtype=int)
        self.ray_normal = wp.zeros((n_envs, self.n_ray), dtype=wp.vec3)
        per_robot = n_ir * len(self.beam_offsets)
        self.ray_exclude = wp.array([env.body_id[r] for r in range(physics.N_ROBOTS) for _ in range(per_robot)], dtype=int)
        self.t_pnt, self.t_vec = wp.to_torch(self.ray_pnt), wp.to_torch(self.ray_vec)
        self.t_dist, self.t_geom = wp.to_torch(self.ray_dist), wp.to_torch(self.ray_geom)

    def geometry3d(self) -> torch.Tensor:
        env = self.env
        n, R = env.n, physics.N_ROBOTS
        q = env.body_quat  # [N,R,4]
        origin_b = torch.cat([self.ir_xy, torch.full_like(self.ir_xy[:, :1], self.mount_h)], -1)  # [5,3]
        origin = env.body_pos.unsqueeze(2) + _quat_rotate(q.unsqueeze(2), origin_b.expand(n, R, -1, -1))  # [N,R,5,3]
        ang = self.ir_angle.view(1, 1, -1, 1) + self.beam_offsets.view(1, 1, 1, -1)  # [1,1,5,3]
        d_b = torch.stack([ang.cos(), ang.sin(), torch.zeros_like(ang)], -1).expand(n, R, -1, -1, -1)
        dirs = _quat_rotate(q.view(n, R, 1, 1, 4), d_b)  # [N,R,5,3,3]
        self.t_pnt.copy_(origin.unsqueeze(3).expand_as(dirs).reshape(n, -1, 3))
        self.t_vec.copy_(dirs.reshape(n, -1, 3))
        env._to_warp()
        # beams test only the robots' ray boxes (group 4: body bounding box, tilts with the robot), not the meshes
        mjw.rays(env.mm, env.d, self.ray_pnt, self.ray_vec, vec6f(0, 0, 0, 0, 1, 0), False, self.ray_exclude,
                 self.ray_dist, self.ray_geom, self.ray_normal)
        env._to_torch()
        geom = self.t_geom.view(n, R, -1, len(self.beam_offsets)).long()
        dist = self.t_dist.view(n, R, -1, len(self.beam_offsets))
        owner = env.geom_robot[geom.clamp(min=0)]
        other = torch.arange(R, device=self.device).flip(0).view(1, R, 1, 1)
        hit = (geom >= 0) & (owner == other) & (dist >= 0) & (dist <= self.ir_range.view(n, R, 1, 1))
        beam = hit.any(-1)  # [N,R,5]

        # crosstalk and line sensors in the ground plane, as in 2D
        heading = env.state.theta.unsqueeze(-1) + self.ir_angle
        o2 = origin[..., :2]
        emit = o2.flip(1)
        emit_dir = torch.stack([heading.flip(1).cos(), heading.flip(1).sin()], -1)
        v = emit.unsqueeze(2) - o2.unsqueeze(3)
        dist_v = v.norm(dim=-1).clamp(min=1e-6)
        our_dir = torch.stack([heading.cos(), heading.sin()], -1).unsqueeze(3)
        looks = (v * our_dir).sum(-1) / dist_v > self.xt_cos
        shines = (-v * emit_dir.unsqueeze(2)).sum(-1) / dist_v > self.xt_cos
        xt = (looks & shines & (dist_v < self.xt_range)).any(-1) if self.crosstalk else torch.zeros_like(beam)

        line_b = torch.cat([self.line_xy, torch.zeros_like(self.line_xy[:, :1])], -1)
        line_pts = env.body_pos.unsqueeze(2) + _quat_rotate(q.unsqueeze(2), line_b.expand(n, R, -1, -1))
        line = line_pts[..., :2].norm(dim=-1) > (env.p.ring_radius - self.edge_w).view(-1, 1, 1)
        return torch.cat([beam, xt, line], -1)

    def push3d(self) -> None:
        self.hist.index_copy_(0, self.head, self.geometry3d().unsqueeze(0))
        self.head.add_(1).remainder_(self.hist_len)

    def prime3d(self, env_idx: torch.Tensor | None = None) -> None:
        g = self.geometry3d()
        if env_idx is None:
            self.hist[:] = g
        else:
            self.hist[:, env_idx] = g[env_idx]


class SumoEnv3D:
    """Batched 3D matches. Same constructor, attributes and step/reset/readings contract as SumoEnv."""

    _start_poses = SumoEnv._start_poses
    _rand = SumoEnv._rand
    ctrl_dt = SumoEnv.ctrl_dt

    def __init__(
        self,
        spec: dict[str, Any],
        num_envs: int,
        device: str | torch.device = "cuda",
        randomize: bool | set[str] = True,
        start_modes: list[str] | None = None,
        seed: int | None = None,
        use_graph: bool = True,
    ):
        assert torch.device(device).type == "cuda", "SumoEnv3D runs on the GPU"
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
        geo = load_geometry()
        self.half = (geo["body_length"] / 2, geo["body_width"] / 2)
        s3 = get(spec, "sim.sim3d")
        self.stop_drive = s3["stop_drive"]
        self.worm_damping = s3["worm_brake_damping"]
        self.worm_lock_speed = s3["worm_lock_speed"]
        geo_m = load_geometry()
        self.wheel_share = geo_m["wheel_volume_m3"] / (geo_m["body_volume_m3"] + 2 * geo_m["wheel_volume_m3"])

        wp.init()
        # Warp works on its own stream (a graph cannot be captured on torch's legacy default stream); the two
        # streams are ordered with events around every Warp call (_to_warp / _to_torch)
        self._wp_stream = wp.get_stream("cuda:0")
        self._torch_stream = wp.stream_from_torch(torch.cuda.current_stream())
        self.mjm = mujoco.MjModel.from_xml_string(scene_xml(spec))
        self.substeps = round(1 / (get(spec, "controller.control_rate_hz") * get(spec, "sim.sim3d.physics_dt_s")))
        self.platform = get(spec, "sim.sim3d.platform")
        self.mm = mjw.put_model(self.mjm)
        # the solver occasionally hitting its (validated) iteration cap is expected; keep the buffer-overflow warnings
        self.mm.opt.warn_overflow &= ~(int(mjw.OverflowType.ITERATIONS) | int(mjw.OverflowType.LS_ITERATIONS))
        self._batch_model_fields()
        self.d = mjw.make_data(self.mjm, nworld=num_envs, nconmax=s3["nconmax"], njmax=s3["njmax"])
        self._index_model()
        t = wp.to_torch
        self.qpos, self.qvel, self.ctrl = t(self.d.qpos), t(self.d.qvel), t(self.d.ctrl)
        self.xipos, self.xquat, self.xpos = t(self.d.xipos), t(self.d.xquat), t(self.d.xpos)

        self.rand = sample_randomization(spec, num_envs, device=device, generator=self.gen, randomize=randomize)
        self.p = physics.make_params(spec, self.rand)
        self.state = physics.zeros_state(num_envs, device)
        z = lambda *s, dtype=torch.float32: torch.zeros(num_envs, *s, device=device, dtype=dtype)  # noqa: E731
        self.t = z()
        self.action = z(physics.N_ROBOTS, 2)
        self.duty = z(physics.N_ROBOTS, 2)
        self.prev_vel = z(physics.N_ROBOTS, 2)
        self.start_mode = z(dtype=torch.long)
        self.rest_z = z(physics.N_ROBOTS)
        self.n_diverged = z(dtype=torch.long)  # per world, diagnostics
        self.sensors = Sensors3D(spec, num_envs, device, self.gen, self)
        self.sensors.set_params(self.rand)
        self.use_graph = use_graph
        self._graph = None
        mjw.step(self.mm, self.d)  # compile the kernels before any graph capture
        wp.synchronize()
        self.reset()

    # --- model --------------------------------------------------------------------------------------------------
    def _batch_model_fields(self) -> None:
        """Give the per-world fields a world dimension of n (MuJoCo Warp broadcasts size-1 world dims)."""
        import numpy as np

        for name in ("body_mass", "body_ipos", "body_inertia", "geom_friction", "geom_size", "geom_rbound",
                     "actuator_gainprm", "actuator_biasprm", "dof_damping", "body_pos", "geom_pos"):
            a = getattr(self.mm, name)
            setattr(self.mm, name, wp.array(np.repeat(a.numpy(), self.n, axis=0), dtype=a.dtype))
        t = wp.to_torch
        self.m_mass, self.m_ipos, self.m_inertia = t(self.mm.body_mass), t(self.mm.body_ipos), t(self.mm.body_inertia)
        self.m_friction, self.m_size, self.m_rbound = t(self.mm.geom_friction), t(self.mm.geom_size), t(self.mm.geom_rbound)
        self.m_gain, self.m_bias = t(self.mm.actuator_gainprm), t(self.mm.actuator_biasprm)
        self.m_damping = t(self.mm.dof_damping)
        self.m_body_pos = t(self.mm.body_pos)
        self.m_geom_pos = t(self.mm.geom_pos)

    def _index_model(self) -> None:
        m = self.mjm
        dev = self.device
        self.body_id = [m.body(r).id for r in ROBOTS]
        self.wheel_body = [[m.body(f"{r}_w{s}").id for s in "lr"] for r in ROBOTS]
        self.free_qadr = [m.jnt_qposadr[m.body_jntadr[b]] for b in self.body_id]
        self.free_dadr = [m.jnt_dofadr[m.body_jntadr[b]] for b in self.body_id]
        self.wheel_dadr = [[m.jnt_dofadr[m.joint(f"{r}_j{s}").id] for s in "lr"] for r in ROBOTS]
        self.wheel_qadr = [[m.jnt_qposadr[m.joint(f"{r}_j{s}").id] for s in "lr"] for r in ROBOTS]
        self.motor = [[m.actuator(f"{r}_motor_{s}").id for s in "lr"] for r in ROBOTS]
        owner = torch.full((m.ngeom,), -1, dtype=torch.long)
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]
            for r in range(physics.N_ROBOTS):
                if b == self.body_id[r] or b in self.wheel_body[r]:
                    owner[g] = r
        self.geom_robot = owner.to(dev)
        self.wheel_geoms = [[g for g in range(m.ngeom) if m.geom_bodyid[g] in self.wheel_body[r]] for r in range(2)]
        self.body_geoms = [[g for g in range(m.ngeom) if m.geom_bodyid[g] == self.body_id[r] and m.geom_contype[g]]
                           for r in range(2)]
        self.wheel_half_w = float(m.geom_size[self.wheel_geoms[0][0]][1])
        # per robot, (left, right): the wheel geom and its axle offset (x, y) from the body origin
        self.wheel_geom_lr = [[next(g for g in range(m.ngeom) if m.geom_bodyid[g] == wb) for wb in self.wheel_body[r]]
                              for r in range(physics.N_ROBOTS)]
        geo = load_geometry()
        self.axle_z, self.cad_wheel_r = geo["axle_z"], geo["wheel_radius"]
        half = geo["track_width"] / 2
        self.wheel_xy = torch.tensor([[geo["axle_x"], half], [geo["axle_x"], -half]], device=dev)  # [2 wheels, xy]
        self.slide_vel = get(self.spec, "robot.drive.tire_slide_vel")
        self.edge_geom = [m.geom(f"{r}_edge").id for r in ROBOTS]
        self.edge_tip_x = geo["blade_edge"]["x"] + geo["blade_edge"]["radius"]
        self.edge_half_len = float(m.geom_size[self.edge_geom[0]][1])
        self.dohyo = m.geom("dohyo").id
        self.dohyo_halfh = float(m.geom_size[self.dohyo][1])

    def _apply_params(self, idx: torch.Tensor) -> None:
        """Push the (randomized) physical parameters of worlds idx into the MuJoCo model."""
        p = self.p
        mb = (1 / p.inv_mass[idx]) * (1 - 2 * self.wheel_share)  # [k,R] body share of the whole robot's mass
        com_h = self.rand["robot.com_height"][idx]  # [k,R]
        edge_r = get(self.spec, "sim.sim3d.blade_edge_radius") + p.clearance[idx]  # blunter edge: higher axis
        L, W = self.half[0] * 2, self.half[1] * 2
        H = load_geometry()["body_height"]
        for r in range(physics.N_ROBOTS):
            b = self.body_id[r]
            self.m_mass[idx, b] = mb[:, r]
            for wb in self.wheel_body[r]:  # body + 2 wheels = robot.mass exactly
                self.m_mass[idx, wb] = (1 / p.inv_mass[idx, r]) * self.wheel_share
            ipos = torch.stack([p.com_x[idx, r], torch.zeros_like(mb[:, r]), com_h[:, r]], -1)
            self.m_ipos[idx, b] = ipos
            box = torch.tensor([W**2 + H**2, L**2 + H**2, L**2 + W**2], device=mb.device, dtype=mb.dtype)
            self.m_inertia[idx, b] = box * mb[:, r:r + 1] / 12
            for g in self.wheel_geoms[r]:
                self.m_friction[idx, g, 0] = p.tire_mu[idx, r]
                # wheel radius (M5 randomization). The axle mount moves with the wheel so the ride height stays
                # (fitting Ø60 wheels on the Ø70 axle would sit the 1.4-3 mm underbody on the ring)
                self.m_size[idx, g, 0] = p.wheel_r[idx, r]
                self.m_rbound[idx, g] = torch.sqrt(p.wheel_r[idx, r] ** 2 + self.wheel_half_w**2)
            for wb in self.wheel_body[r]:
                self.m_body_pos[idx, wb, 2] = self.axle_z + (p.wheel_r[idx, r] - self.cad_wheel_r)
            for g in self.body_geoms[r]:
                self.m_friction[idx, g, 0] = p.scoop_mu[idx, r]
            # blade edge: front stays at the CAD tip, bottom on the ring (the plates start inside it)
            e, er = self.edge_geom[r], edge_r[:, r]
            self.m_size[idx, e, 0] = er
            self.m_rbound[idx, e] = er + self.edge_half_len
            self.m_geom_pos[idx, e, 0] = self.edge_tip_x - er
            self.m_geom_pos[idx, e, 2] = er
            for w in range(2):
                tau = p.tau_stall[idx, r, w]
                self.m_gain[idx, self.motor[r][w], 0] = tau
                self.m_bias[idx, self.motor[r][w], 2] = -tau / p.omega0[idx, r]
        if self.platform == "cylinder":
            R = p.ring_radius[idx]
            self.m_size[idx, self.dohyo, 0] = R
            self.m_rbound[idx, self.dohyo] = torch.sqrt(R**2 + self.dohyo_halfh**2)

    # --- reset / step -------------------------------------------------------------------------------------------
    def reset(self, env_mask: torch.Tensor | None = None) -> None:
        if env_mask is None:
            env_mask = torch.ones(self.n, dtype=torch.bool, device=self.device)
        self.reset_idx(env_mask.nonzero().squeeze(-1))

    def reset_idx(self, idx: torch.Tensor) -> None:
        k = idx.numel()
        if k == 0:
            return
        fresh = sample_randomization(self.spec, k, device=self.device, generator=self.gen, randomize=self.randomize)
        self.rand.scatter_(fresh, idx)
        physics.scatter_params_(self.p, physics.make_params(self.spec, fresh), idx)
        self._apply_params(idx)

        allowed = physics.const(tuple(self.all_modes.index(s) for s in self.start_modes), self.device, torch.long)
        modes = allowed[torch.randint(len(allowed), (k,), device=self.device, generator=self.gen)]
        self.start_mode.index_copy_(0, idx, modes)
        center, theta = self._start_poses(k, modes, self.rand["arena.start_center_dist"][idx])

        mask = torch.zeros(self.n, dtype=torch.bool, device=self.device).index_fill_(0, idx, True)
        self._to_warp()
        mjw.reset_data(self.mm, self.d, wp.from_torch(mask))
        self._to_torch()
        for r in range(physics.N_ROBOTS):
            a = self.free_qadr[r]
            self.qpos[idx, a:a + 2] = center[:, r]
            self.qpos[idx, a + 2] = 0.002
            half = theta[:, r] / 2
            self.qpos[idx, a + 3] = half.cos()
            self.qpos[idx, a + 4:a + 6] = 0
            self.qpos[idx, a + 6] = half.sin()
        self._to_warp()
        mjw.kinematics(self.mm, self.d)
        self._to_torch()
        self.t.index_fill_(0, idx, 0)
        for x in (self.action, self.duty, self.prev_vel, self.state.u_lag, self.state.vel, self.state.lift):
            x.index_fill_(0, idx, 0)
        self._refresh_state(None)
        self.rest_z.index_copy_(0, idx, self.xipos[idx][:, self.body_id, 2])
        self.sensors.prime3d(idx)

    def _to_warp(self) -> None:
        self._wp_stream.wait_stream(self._torch_stream)

    def _to_torch(self) -> None:
        self._torch_stream.wait_stream(self._wp_stream)

    def _substeps(self) -> None:
        for _ in range(self.substeps):
            mjw.step(self.mm, self.d)

    def _refresh_state(self, prev_pos: torch.Tensor | None) -> None:
        st = self.state
        self.body_pos = self.xpos[:, self.body_id]           # [N,R,3] frame origin = footprint centre
        self.body_quat = self.xquat[:, self.body_id]         # [N,R,4]
        com = self.xipos[:, self.body_id]
        w, x, y, z = self.body_quat.unbind(-1)
        st.theta = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        if prev_pos is not None:
            st.vel = (com[..., :2] - prev_pos) / self.ctrl_dt.view(-1, 1, 1)
        st.pos = com[..., :2].clone()
        ang = torch.stack([self.qvel[:, d + 3:d + 6] for d in self.free_dadr], 1)  # body-frame angular velocity
        st.omega = _quat_rotate(self.body_quat, ang)[..., 2]
        st.wheel_w = torch.stack([self.qvel[:, self.wheel_dadr[r]] for r in range(physics.N_ROBOTS)], 1)
        up = _quat_rotate(self.body_quat, torch.tensor([0.0, 0.0, 1.0], device=self.device).expand_as(com))
        st.wedge = torch.acos(up[..., 2].clamp(-1, 1))      # tilt
        if prev_pos is not None:
            st.lift = ((com[..., 2] - self.rest_z) / 0.05).clamp(min=0)

    def _tire_friction(self) -> None:
        """Static vs sliding tire friction, which MuJoCo does not distinguish: μ for a rolling wheel, falling to
        μ·tire_slide_ratio as the contact slips (exp(-slip/tire_slide_vel)). Uses the velocities of the last control
        step, so it lags by one tick."""
        p, st = self.p, self.state
        c, s = st.theta.cos(), st.theta.sin()                                    # [N,R]
        k = self.rand["robot.drive.tire_slide_ratio"]
        for r in range(physics.N_ROBOTS):
            v = self.qvel[:, self.free_dadr[r]:self.free_dadr[r] + 2]             # body origin, world xy
            for w in range(2):
                lx, ly = self.wheel_xy[w]
                ox, oy = c[:, r] * lx - s[:, r] * ly, s[:, r] * lx + c[:, r] * ly   # wheel offset in world
                vx = v[:, 0] - st.omega[:, r] * oy - st.wheel_w[:, r, w] * p.wheel_r[:, r] * c[:, r]
                vy = v[:, 1] + st.omega[:, r] * ox - st.wheel_w[:, r, w] * p.wheel_r[:, r] * s[:, r]
                slip = torch.sqrt(vx * vx + vy * vy)
                self.m_friction[:, self.wheel_geom_lr[r][w], 0] = p.tire_mu[:, r] * (
                    k[:, r] + (1 - k[:, r]) * torch.exp(-slip / self.slide_vel))

    def readings(self) -> dict[str, torch.Tensor]:
        r = self.sensors.read(self.ctrl_dt)  # one history slot per control tick
        r.update(self.sensors.motion(self.state, self.prev_vel, self.ctrl_dt))
        return {k: v.nan_to_num(0.0, 0.0, 0.0) if v.is_floating_point() else v for k, v in r.items()}

    def step(self, action: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """action [N,R,2] in [-1, 1] → (readings, done [N], winner [N]: 0/1 or DRAW). No auto-reset."""
        p, st = self.p, self.state
        self.action.copy_(firmware.slew(self.action, action, self.slew))
        if self.deadband_comp:
            self.duty.copy_(firmware.deadband_comp(self.action, p.deadband_cal.unsqueeze(-1)))
        else:
            self.duty.copy_(self.action)
        db = p.deadband.unsqueeze(-1)
        u_m = torch.sign(self.duty) * ((self.duty.abs() - db) / (1 - db)).clamp(min=0)
        alpha = (1 - torch.exp(-self.ctrl_dt.view(-1, 1) / p.motor_tc)).unsqueeze(-1)
        st.u_lag.add_(alpha * (u_m - st.u_lag))
        drive = st.u_lag * p.volt_ratio.unsqueeze(-1)
        # worm gear: holds a wheel that has (almost) stopped with no drive; a spinning wheel is slowed by the motor
        locked = ((drive.abs() < self.stop_drive) & (st.wheel_w.abs() < self.worm_lock_speed)).float() * self.worm_damping
        for r in range(physics.N_ROBOTS):
            self.ctrl[:, self.motor[r]] = drive[:, r]
            self.m_damping[:, self.wheel_dadr[r]] = locked[:, r]
        self._tire_friction()
        self.prev_vel.copy_(st.vel)
        prev_pos = st.pos.clone()
        self._to_warp()
        if self.use_graph:
            if self._graph is None:
                with wp.ScopedCapture(stream=self._wp_stream) as cap:
                    self._substeps()
                self._graph = cap.graph
            wp.capture_launch(self._graph, stream=self._wp_stream)
        else:
            self._substeps()
        self._to_torch()
        # Safety net: a diverged world (rare) must not leak NaN into obs/rewards: it ends as a draw and is reset by
        # the caller. The poses derived from qpos (xpos/xipos/xquat) are cleaned too: the state and the sensors read
        # them, and one NaN frame in the frame history or the obs normalizer used to take the whole run down.
        bad = ~torch.isfinite(self.qpos).all(1) | ~torch.isfinite(self.qvel).all(1)
        for x in (self.qpos, self.qvel, self.xpos, self.xipos, self.xquat):
            x.nan_to_num_(0.0, 0.0, 0.0)
        self.n_diverged += bad
        self._refresh_state(prev_pos)
        for x in (st.pos, st.vel, st.theta, st.omega, st.wheel_w, st.wedge, st.lift):
            x.nan_to_num_(0.0, 0.0, 0.0)
        self.sensors.push3d()
        self.t += self.ctrl_dt

        com_z = self.xipos[:, self.body_id, 2]
        out = physics.com_out_of_ring(st, p) | (com_z < self.rest_z - 0.03)  # beyond the edge, or fell off it
        done = out.any(-1) | (self.t >= self.timeout) | bad
        winner = torch.full((self.n,), DRAW, device=self.device, dtype=torch.long)
        winner = torch.where(out[:, 1] & ~out[:, 0], 0, winner)
        winner = torch.where(out[:, 0] & ~out[:, 1], 1, winner)
        winner = torch.where(bad, DRAW, winner)
        return self.readings(), done, winner
