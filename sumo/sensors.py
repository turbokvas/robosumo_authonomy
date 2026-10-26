"""Sensors as the robot's MCU sees them: 3 front + 2 side (reserved) E18-D80NK beams, 4 edge line sensors,
battery ADC, and the reserved wheel encoders and IMU.

Geometry is sampled every physics substep into a short history; reads at the control tick pick each robot's
own latency out of it, then apply E18 noise (false negatives/positives, crosstalk from the opponent's identical E18).
Output convention (sim side, before the firmware's active-low / colour mapping):
    ir      [N,R,3] bool, order = spec mounts (left +15°, centre, right −15°), True = opponent detected
    ir_side [N,R,2] bool, side E18 (left +90°, right −90°), same model as the front ones
    line    [N,R,4] bool, order = spec mounts (front-left, front-right, rear-left, rear-right), True = over the black edge
    battery [N,R] float = V / full_v
    wheel_speed [N,R,2] rad/s from encoder counts per control period (quantized)
    gyro_z [N,R] rad/s, accel_xy [N,R,2] m/s² in the body frame (CoM acceleration over the control period), noisy
"""

from __future__ import annotations

import math
from typing import Any

import torch

from sumo import physics
from sumo.config import RandomizationBatch, get

N_FRONT_IR = 3
N_SIDE_IR = 2
N_IR = N_FRONT_IR + N_SIDE_IR
N_LINE = 4
_IR, _XT, _LINE = slice(0, N_IR), slice(N_IR, 2 * N_IR), slice(2 * N_IR, 2 * N_IR + N_LINE)


def ray_box_distance(o: torch.Tensor, d: torch.Tensor, half: tuple[float, float]) -> torch.Tensor:
    """Slab test in the box frame. o, d: [..., 2] (d unit). Distance to the box along the ray, inf if missed."""
    h = physics.const(tuple(half), o.device, o.dtype)
    d_safe = torch.where(d.abs() < 1e-9, torch.full_like(d, 1e-9), d)
    t1, t2 = (-h - o) / d_safe, (h - o) / d_safe
    t_near = torch.minimum(t1, t2).amax(-1)
    t_far = torch.maximum(t1, t2).amin(-1)
    hit = t_far >= t_near.clamp(min=0)
    return torch.where(hit, t_near.clamp(min=0), torch.full_like(t_near, math.inf))


def _to_local(pts: torch.Tensor, center: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """World points/vectors → frame of a box (center [..,2], theta [..]); pass center=0 for directions."""
    c, s = theta.cos().unsqueeze(-1), theta.sin().unsqueeze(-1)
    v = pts - center
    return torch.cat([c * v[..., :1] + s * v[..., 1:], -s * v[..., :1] + c * v[..., 1:]], -1)


class Sensors:
    def __init__(self, spec: dict[str, Any], n_envs: int, device: str | torch.device, generator: torch.Generator | None = None):
        ir = get(spec, "robot.sensors.opponent_ir")
        self.device = device
        self.gen = generator
        side = get(spec, "robot.sensors.side_ir.mounts")
        mounts = torch.tensor(ir["mounts"] + side, device=device, dtype=torch.float32)  # [5, (x, y, deg)]
        self.ir_xy = mounts[:, :2]
        self.ir_angle = torch.deg2rad(mounts[:, 2])
        half_beam = math.radians(ir["beam_half_angle_deg"])
        self.beam_offsets = torch.tensor([-half_beam, 0.0, half_beam], device=device)
        self.fn_p = ir["false_negative_p"]
        self.fp_p = ir["false_positive_p"]
        self.crosstalk = ir["crosstalk"]
        self.xt_cos = math.cos(math.radians(ir["crosstalk_cone_deg"]))
        self.xt_range = ir["crosstalk_range"]
        self.xt_p = ir["crosstalk_p"]
        self.line_xy = torch.tensor(get(spec, "robot.sensors.edge.mounts"), device=device, dtype=torch.float32)
        self.edge_w = get(spec, "arena.edge_line_width")
        self.full_v = get(spec, "robot.power.full_v")
        enc, imu = get(spec, "robot.sensors.encoders"), get(spec, "robot.sensors.imu")
        self.counts_per_rev = enc["counts_per_wheel_rev"]
        self.gyro_std, self.accel_std = imu["gyro_noise_std"], imu["accel_noise_std"]
        self.gyro_range, self.accel_range = imu["gyro_range"], imu["accel_range"]

        # history long enough for the slowest latency at the fastest substep
        lat_max = max(get(spec, "robot.sensors.opponent_ir.latency_s_rand")[1], get(spec, "robot.sensors.edge.latency_s_rand")[1])
        rate_max = get(spec, "controller.control_rate_rand_hz")[1]
        self.substeps = round(1 / (get(spec, "controller.control_rate_hz") * get(spec, "controller.physics_dt_s")))
        self.hist_len = math.ceil(lat_max * rate_max * self.substeps) + 2
        self.hist = torch.zeros(self.hist_len, n_envs, physics.N_ROBOTS, 2 * N_IR + N_LINE, dtype=torch.bool, device=device)
        self.head = torch.zeros(1, dtype=torch.long, device=device)  # device-side, so push() is graph-capturable
        self._n = torch.arange(n_envs, device=device).view(-1, 1)
        self._r = torch.arange(physics.N_ROBOTS, device=device).view(1, -1)

    def set_params(self, rand: RandomizationBatch) -> None:
        """Views into the randomization batch; it is updated in place on reset, so these stay current."""
        self.ir_range = rand["robot.sensors.opponent_ir.range"]
        self.ir_latency = rand["robot.sensors.opponent_ir.latency_s"]
        self.line_latency = rand["robot.sensors.edge.latency_s"]
        self.voltage = rand["robot.power.voltage"]

    def geometry(self, state: physics.PhysState, p: physics.PhysParams) -> torch.Tensor:
        """Noise-free readings now: [N,R,10] = 3 IR beam hits, 3 crosstalk conditions, 4 line sensors."""
        origin = physics.body_to_world(state, p, self.ir_xy)  # [N,R,3,2]
        heading = state.theta.unsqueeze(-1) + self.ir_angle  # [N,R,3]
        ray_ang = heading.unsqueeze(-1) + self.beam_offsets  # [N,R,3,3]
        dirs = torch.stack([ray_ang.cos(), ray_ang.sin()], -1)

        ctr = physics.centers(state, p)
        opp_c = ctr.flip(1).view(-1, physics.N_ROBOTS, 1, 1, 2)
        opp_th = state.theta.flip(1).view(-1, physics.N_ROBOTS, 1, 1)
        o_loc = _to_local(origin.unsqueeze(-2), opp_c, opp_th)
        d_loc = _to_local(dirs, torch.zeros_like(opp_c), opp_th)
        dist = ray_box_distance(o_loc.expand_as(d_loc), d_loc, p.half)  # [N,R,3,3]
        beam = (dist.amin(-1) <= self.ir_range.unsqueeze(-1))  # [N,R,3]

        # crosstalk: opponent emitter j inside our receive cone of sensor i, and its beam pointing at us
        if self.crosstalk:
            emit = origin.flip(1)  # [N,R,3(j),2] opponent's sensors
            emit_dir = torch.stack([heading.flip(1).cos(), heading.flip(1).sin()], -1)
            v = emit.unsqueeze(2) - origin.unsqueeze(3)  # [N,R,3(i),3(j),2] from our sensor i to their emitter j
            dist_v = v.norm(dim=-1).clamp(min=1e-6)
            our_dir = torch.stack([heading.cos(), heading.sin()], -1).unsqueeze(3)
            looks = (v * our_dir).sum(-1) / dist_v > self.xt_cos
            shines = (-v * emit_dir.unsqueeze(2)).sum(-1) / dist_v > self.xt_cos
            xt = (looks & shines & (dist_v < self.xt_range)).any(-1)
        else:
            xt = torch.zeros_like(beam)

        line_pts = physics.body_to_world(state, p, self.line_xy)  # [N,R,4,2]
        line = line_pts.norm(dim=-1) > (p.ring_radius - self.edge_w).view(-1, 1, 1)
        return torch.cat([beam, xt, line], -1)

    def push(self, state: physics.PhysState, p: physics.PhysParams) -> None:
        self.hist.index_copy_(0, self.head, self.geometry(state, p).unsqueeze(0))
        self.head.add_(1).remainder_(self.hist_len)

    def prime(self, state: physics.PhysState, p: physics.PhysParams, env_idx: torch.Tensor | None = None) -> None:
        """Fill the whole history with the current geometry (after a reset), so latency reads stay valid."""
        g = self.geometry(state, p)
        if env_idx is None:
            self.hist[:] = g
        else:
            self.hist[:, env_idx] = g[env_idx]

    def _delayed(self, latency: torch.Tensor, substep_dt: torch.Tensor, ch: slice) -> torch.Tensor:
        delay = (latency / substep_dt.view(-1, 1)).round().long().clamp(0, self.hist_len - 1)  # [N,R]
        idx = (self.head - 1 - delay) % self.hist_len
        return self.hist[idx, self._n, self._r, ch]

    def read(self, substep_dt: torch.Tensor) -> dict[str, torch.Tensor]:
        """Readings at the control tick, with per-robot latency (substep_dt: [N]) and E18 noise."""
        beam = self._delayed(self.ir_latency, substep_dt, _IR)
        xt = self._delayed(self.ir_latency, substep_dt, _XT)
        line = self._delayed(self.line_latency, substep_dt, _LINE)
        u = torch.rand(3, *beam.shape, device=self.device, generator=self.gen)
        ir = (beam & (u[0] >= self.fn_p)) | (u[1] < self.fp_p) | (xt & (u[2] < self.xt_p))
        return {"ir": ir[..., :N_FRONT_IR], "ir_side": ir[..., N_FRONT_IR:], "line": line,
                "battery": self.voltage / self.full_v}

    def motion(self, state: physics.PhysState, prev_vel: torch.Tensor, ctrl_dt: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encoder wheel speed and IMU readings at the control tick. prev_vel: CoM velocity one control period ago."""
        dt = ctrl_dt.view(-1, 1, 1)
        rad_per_count = 2 * math.pi / self.counts_per_rev
        counts = torch.round(state.wheel_w * dt / rad_per_count)
        wheel_speed = counts * rad_per_count / dt
        fwd = torch.stack([state.theta.cos(), state.theta.sin()], -1)
        left = torch.stack([-fwd[..., 1], fwd[..., 0]], -1)
        acc = (state.vel - prev_vel) / dt
        acc_body = torch.stack([(acc * fwd).sum(-1), (acc * left).sum(-1)], -1)
        noise = lambda x, std: x + std * torch.randn(x.shape, device=x.device, generator=self.gen)  # noqa: E731
        gyro = noise(state.omega, self.gyro_std).clamp(-self.gyro_range, self.gyro_range)
        accel = noise(acc_body, self.accel_std).clamp(-self.accel_range, self.accel_range)
        return {"wheel_speed": wheel_speed, "gyro_z": gyro, "accel_xy": accel}
