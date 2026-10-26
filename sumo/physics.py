"""Batched 2D rigid-body physics: two diff-drive sumo robots per env, fully on GPU.

Conventions
- World frame centred on the ring, θ counter-clockwise from +x.
- Robot body frame (spec): x forward, y left, origin at the geometric centre of the footprint.
- The state stores the centre-of-mass pose; the CoM sits at (com_x, 0) in the body frame.
- Shapes: N envs, R = 2 robots (0 = ours, 1 = opponent), wheel index 0 = left, 1 = right.

Model (PLAN.md, M1)
- Motor: τ = u·(V/V_nom)·τ_stall − (τ_stall/ω₀)·ω, PWM deadband, first-order lag on the command.
- Worm gear: the wheel is driven only by the motor. External forces can slow it down but never back-drive it,
  so the wheel speed stays between 0 and the free speed for the current command; at u = 0 it is locked.
  Everything the wheel cannot follow goes into tire slip.
- Tire / scoop / body friction: μ·N·tanh(|slip|/s₀) along −slip (smoothed Coulomb, friction circle in 2D).
- Normal loads: static split of the weight between the axle and the scoop contact point.
- Robot-robot: oriented boxes, SAT + face clipping (two-point manifold), penalty spring-damper + friction.
- Wedge: a robot touching the opponent with its front can get under it and take part of the opponent's wheel
  load onto its own scoop (state `lift`). Front into side/rear always gets under; front into front goes to whoever
  came in at an angle, then to the lower blade (see `_update_lift`).
- Integration: semi-implicit Euler; the wheel ODE is linearised-implicit for stability with stiff tire slip.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any

import torch

from sumo.config import RandomizationBatch, get

N_ROBOTS = 2
_CONSTS: dict[tuple, torch.Tensor] = {}


def const(values: tuple[float, ...], device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Small constant tensor, created once per device. Keeps hot paths free of host→device copies (CUDA graphs)."""
    key = (values, str(device), dtype)
    if key not in _CONSTS:
        _CONSTS[key] = torch.tensor(values, device=device, dtype=dtype)
    return _CONSTS[key]
WEDGE_EPS = 1e-3  # below this a decaying wedge advantage counts as gone


@dataclass
class PhysParams:
    dt: float
    s0: float                    # slip smoothing velocity
    half: tuple[float, float]    # box half extents (x, y)
    k_contact: float
    c_contact: float
    body_mu: torch.Tensor        # [N]
    ring_radius: torch.Tensor    # [N]
    inv_mass: torch.Tensor       # [N,R]
    inv_inertia: torch.Tensor    # [N,R]
    com_x: torch.Tensor          # [N,R] CoM in body frame
    axle_x: float
    half_track: float
    wheel_r: torch.Tensor        # [N,R]
    scoop_x: float
    tire_mu: torch.Tensor        # [N,R]
    scoop_mu: torch.Tensor       # [N,R]
    n_wheel: torch.Tensor        # [N,R] normal load per wheel
    n_scoop: torch.Tensor        # [N,R]
    tau_stall: torch.Tensor      # [N,R,2]
    omega0: torch.Tensor         # [N,R] no-load wheel speed at nominal voltage, rad/s
    motor_tc: torch.Tensor       # [N,R]
    deadband: torch.Tensor       # [N,R] true motor deadband (duty)
    deadband_cal: torch.Tensor   # [N,R] the firmware's calibrated estimate of it (true − calibration error)
    volt_ratio: torch.Tensor     # [N,R] V / V_nom
    j_wheel: float
    lift_max: torch.Tensor       # [N,R] max share of the opponent's wheel load this robot's wedge can take
    clearance: torch.Tensor      # [N,R] blade height above the ring
    front_cos: float
    angle_adv: float             # rad
    clearance_adv: float
    adv_deadband: float
    lift_time: float
    drop_time: float


@dataclass
class PhysState:
    pos: torch.Tensor            # [N,R,2] CoM, world
    theta: torch.Tensor          # [N,R]
    vel: torch.Tensor            # [N,R,2]
    omega: torch.Tensor          # [N,R]
    wheel_w: torch.Tensor        # [N,R,2] wheel angular speed, rad/s
    u_lag: torch.Tensor          # [N,R,2] motor command after deadband and lag
    lift: torch.Tensor           # [N,R] share of this robot's wheel load carried by the opponent's wedge
    wedge: torch.Tensor          # [N,R] how far this robot's wedge got under the opponent, 0..1 (latched)

    def clone(self) -> "PhysState":
        return PhysState(**{f.name: getattr(self, f.name).clone() for f in fields(self)})


def make_params(
    spec: dict[str, Any],
    rand: RandomizationBatch,
    fixed: torch.Tensor | None = None,
) -> PhysParams:
    """Physical parameters for a batch. `fixed` [N,R] bool makes a robot an immovable obstacle (tests, curriculum)."""
    g = get(spec, "sim.gravity")
    mass = rand["robot.mass"]

    def val(name: str) -> torch.Tensor:
        """A per-robot parameter: randomized if the spec has `<name>_rand`, else the spec value. M5 widens/adds
        ranges in the spec only; the physics reads every such quantity through here."""
        if name in rand.params:
            return rand[name]
        return torch.full_like(mass, float(get(spec, name)))

    footprint = get(spec, "robot.footprint")
    half = (footprint[0] / 2, footprint[1] / 2)
    axle_x = get(spec, "robot.drive.axle_x")
    scoop_x = get(spec, "robot.front.contact_x")
    com_x = axle_x + rand["robot.com_x_from_axle"]

    yaw_inertia = get(spec, "robot.yaw_inertia")
    if yaw_inertia is None:  # spec: поки ~ m*(a²+b²)/12
        inertia = mass * (footprint[0] ** 2 + footprint[1] ** 2) / 12
    else:
        inertia = torch.full_like(mass, float(yaw_inertia))

    weight = mass * g
    n_wheels = weight * (scoop_x - com_x) / (scoop_x - axle_x)

    motors = get(spec, "robot.motors")
    est = motors["est_16v"]
    omega0 = val("robot.motors.est_16v.no_load_rpm") * 2 * math.pi / 60
    mismatch = rand["robot.motors.left_right_mismatch"].sqrt()
    tau_stall = (val("robot.motors.est_16v.stall_torque_nm") * rand["robot.motors.strength"]).unsqueeze(-1) * torch.stack(
        [mismatch, 1 / mismatch], dim=-1
    )

    inv_mass = 1 / mass
    inv_inertia = 1 / inertia
    if fixed is not None:
        inv_mass = torch.where(fixed, torch.zeros_like(inv_mass), inv_mass)
        inv_inertia = torch.where(fixed, torch.zeros_like(inv_inertia), inv_inertia)

    contact = get(spec, "sim.contact")
    wedge = get(spec, "sim.wedge")
    dt = get(spec, "controller.physics_dt_s")
    return PhysParams(
        dt=dt,
        s0=get(spec, "sim.slip_smoothing_vel"),
        half=half,
        k_contact=contact["stiffness"],
        c_contact=contact["damping"],
        body_mu=rand["sim.contact.body_mu"],
        ring_radius=rand["arena.ring_diameter"] / 2,
        inv_mass=inv_mass,
        inv_inertia=inv_inertia,
        com_x=com_x,
        axle_x=axle_x,
        half_track=get(spec, "robot.drive.track_width") / 2,
        wheel_r=val("robot.drive.wheel_diameter") / 2,
        scoop_x=scoop_x,
        tire_mu=rand["robot.drive.tire_mu"],
        scoop_mu=rand["robot.front.friction_mu"],
        n_wheel=n_wheels / 2,
        n_scoop=weight - n_wheels,
        tau_stall=tau_stall,
        omega0=omega0,
        motor_tc=rand["robot.motors.time_constant_s"],
        deadband=rand["robot.motors.deadband_pwm"],
        deadband_cal=(rand["robot.motors.deadband_pwm"] - val("robot.motors.deadband_comp_error")).clamp(min=0),
        volt_ratio=rand["robot.power.voltage"] / get(spec, "robot.power.nominal_v_for_motor_model"),
        j_wheel=motors["reflected_inertia_kgm2"],
        lift_max=rand["robot.front.lift_fraction"],
        clearance=rand["robot.front.clearance"],
        front_cos=math.cos(math.radians(wedge["front_cone_deg"])),
        angle_adv=math.radians(wedge["angle_advantage_deg"]),
        clearance_adv=wedge["clearance_advantage"],
        adv_deadband=wedge["advantage_deadband"],
        lift_time=wedge["lift_time_s"],
        drop_time=wedge["drop_time_s"],
    )


def copy_params_(dst: PhysParams, src: PhysParams) -> None:
    """Overwrite dst in place (tensors keep their storage, so captured CUDA graphs see the new values)."""
    for f in fields(dst):
        a, b = getattr(dst, f.name), getattr(src, f.name)
        if isinstance(a, torch.Tensor):
            a.copy_(b)
        elif a != b:
            raise ValueError(f"non-tensor parameter {f.name} changed: {a} → {b}")


def scatter_params_(dst: PhysParams, subset: PhysParams, env_idx: torch.Tensor) -> None:
    """Write params computed for the envs `env_idx` (long) only into those envs of dst, in place."""
    for f in fields(dst):
        a, b = getattr(dst, f.name), getattr(subset, f.name)
        if isinstance(a, torch.Tensor):
            a.index_copy_(0, env_idx, b)
        elif a != b:
            raise ValueError(f"non-tensor parameter {f.name} changed: {a} → {b}")


def zeros_state(n_envs: int, device: str | torch.device = "cpu") -> PhysState:
    z = lambda *s: torch.zeros(n_envs, N_ROBOTS, *s, device=device)  # noqa: E731
    return PhysState(pos=z(2), theta=z(), vel=z(2), omega=z(), wheel_w=z(2), u_lag=z(2), lift=z(), wedge=z())


def _rot(theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit forward and left vectors, [..., 2]."""
    c, s = theta.cos(), theta.sin()
    return torch.stack([c, s], -1), torch.stack([-s, c], -1)


def _cross(r: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    return r[..., 0] * f[..., 1] - r[..., 1] * f[..., 0]


def _point_vel(vel: torch.Tensor, omega: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Velocity of a point at arm r from the CoM: v + ω × r."""
    w = omega.unsqueeze(-1)
    return vel + torch.cat([-w * r[..., 1:2], w * r[..., 0:1]], -1)


def _friction(slip: torch.Tensor, mu_n: torch.Tensor, s0: float) -> torch.Tensor:
    """Smoothed Coulomb force opposing a 2D slip velocity: −μN·tanh(|s|/s₀)·ŝ."""
    mag = slip.norm(dim=-1, keepdim=True)
    gain = torch.where(mag > 1e-9, torch.tanh(mag / s0) / mag.clamp(min=1e-9), torch.full_like(mag, 1 / s0))
    return -mu_n.unsqueeze(-1) * gain * slip


def centers(state: PhysState, p: PhysParams) -> torch.Tensor:
    """Geometric centres of the robots, [N,R,2]."""
    fwd, _ = _rot(state.theta)
    return state.pos - p.com_x.unsqueeze(-1) * fwd


def body_to_world(state: PhysState, p: PhysParams, pts: torch.Tensor) -> torch.Tensor:
    """Body-frame points [K,2] (relative to the geometric centre) → world [N,R,K,2]."""
    fwd, left = _rot(state.theta)
    rel_x = pts[:, 0] - p.com_x.unsqueeze(-1)  # [N,R,K]
    rel_y = pts[:, 1].expand_as(rel_x)
    return state.pos.unsqueeze(-2) + rel_x.unsqueeze(-1) * fwd.unsqueeze(-2) + rel_y.unsqueeze(-1) * left.unsqueeze(-2)


def set_pose(
    state: PhysState, p: PhysParams, center: torch.Tensor, theta: torch.Tensor, env_mask: torch.Tensor | None = None
) -> None:
    """Place robots by geometric centre [N,R,2] and heading [N,R], at rest. Only envs in env_mask [N] if given."""
    fwd, _ = _rot(theta)
    new = PhysState(
        pos=center + p.com_x.unsqueeze(-1) * fwd,
        theta=theta,
        vel=torch.zeros_like(state.vel),
        omega=torch.zeros_like(state.omega),
        wheel_w=torch.zeros_like(state.wheel_w),
        u_lag=torch.zeros_like(state.u_lag),
        lift=torch.zeros_like(state.lift),
        wedge=torch.zeros_like(state.wedge),
    )
    for f in fields(state):
        cur, val = getattr(state, f.name), getattr(new, f.name)
        if env_mask is None:
            cur.copy_(val)
        else:
            m = env_mask.view(-1, *([1] * (cur.dim() - 1)))
            cur.copy_(torch.where(m, val, cur))


def com_out_of_ring(state: PhysState, p: PhysParams) -> torch.Tensor:
    """[N,R] bool: CoM beyond the ring edge → that robot lost."""
    return state.pos.norm(dim=-1) > p.ring_radius.unsqueeze(-1)


def box_contacts(
    c: torch.Tensor, theta: torch.Tensor, half: tuple[float, float]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Contact manifold between box 0 and box 1 of each env (same half extents).

    c: centres [N,2,2], theta [N,2].
    Returns points [N,2,2] (world), normal [N,2] pointing from box 0 to box 1, depth [N,2] (0 = no contact).
    """
    h = const(tuple(half), c.device, c.dtype)  # [2]
    fwd, left = _rot(theta)
    axes = torch.stack([fwd, left], dim=2)  # [N,box,axis,2]

    def sat(a: int, b: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Max separation of box b from box a's face axes, and that axis index."""
        d = c[:, b] - c[:, a]  # [N,2]
        aa, ab = axes[:, a], axes[:, b]  # [N,2(axis),2]
        dots = torch.einsum("nkd,njd->nkj", aa, ab).abs()  # |a_k · b_j|
        r_b = (dots * h).sum(-1)  # projection radius of b on a_k
        sep = torch.einsum("nkd,nd->nk", aa, d).abs() - h - r_b
        return sep.max(dim=-1)

    sep0, k0 = sat(0, 1)
    sep1, k1 = sat(1, 0)
    ref_is_1 = sep1 > sep0 + 1e-6
    overlap = torch.maximum(sep0, sep1) < 0

    ri = ref_is_1.long()
    idx = torch.arange(c.shape[0], device=c.device)
    c_ref, c_inc = c[idx, ri], c[idx, 1 - ri]
    ax_ref, ax_inc = axes[idx, ri], axes[idx, 1 - ri]  # [N,2,2]
    k = torch.where(ref_is_1, k1, k0)

    a_k = ax_ref[idx, k]
    a_t = ax_ref[idx, 1 - k]
    sign = torch.sign((a_k * (c_inc - c_ref)).sum(-1, keepdim=True))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    n = a_k * sign  # reference face normal, towards the incident box
    face_c = c_ref + n * h[k].unsqueeze(-1)
    ht = h[1 - k]

    # incident face: most anti-parallel to n
    inc_dots = torch.einsum("njd,nd->nj", ax_inc, n)
    j = inc_dots.abs().argmax(-1)
    a_j = ax_inc[idx, j]
    inc_n = -torch.sign(inc_dots[idx, j]).unsqueeze(-1) * a_j
    inc_c = c_inc + inc_n * h[j].unsqueeze(-1)
    t_inc = ax_inc[idx, 1 - j] * h[1 - j].unsqueeze(-1)
    v1, v2 = inc_c + t_inc, inc_c - t_inc

    # clip the incident edge to the reference face's tangent extent
    s1 = ((v1 - face_c) * a_t).sum(-1)
    s2 = ((v2 - face_c) * a_t).sum(-1)
    ds = s2 - s1  # |ds| >= 2·h·cos45° > 0
    la, lb = (-ht - s1) / ds, (ht - s1) / ds
    lo = torch.minimum(la, lb).clamp(min=0)
    hi = torch.maximum(la, lb).clamp(max=1)
    lam = torch.stack([lo, hi], -1)  # [N,2]
    pts = v1.unsqueeze(1) + lam.unsqueeze(-1) * (v2 - v1).unsqueeze(1)  # [N,2,2]

    sep = ((pts - face_c.unsqueeze(1)) * n.unsqueeze(1)).sum(-1)  # <0 = penetrating
    valid = overlap.unsqueeze(-1) & (sep < 0) & (lo <= hi).unsqueeze(-1)
    depth = torch.where(valid, -sep, torch.zeros_like(sep))
    pts = pts - 0.5 * sep.unsqueeze(-1) * n.unsqueeze(1)  # midpoint between the two surfaces
    normal01 = torch.where(ref_is_1.unsqueeze(-1), -n, n)
    return pts, normal01, depth


def _update_lift(
    state: PhysState, p: PhysParams, fwd: torch.Tensor, n01: torch.Tensor, touching: torch.Tensor, dt2: torch.Tensor | float
) -> None:
    """Who gets under whom, from the contact normal and both headings.

    A robot can lift only when it touches with its front (normal within the front cone of its heading).
    Front into side/rear: full lift. Front into front: advantage = (own misalignment − opponent's) / angle_adv
    + (opponent's clearance − own) / clearance_adv, i.e. whoever came in at an angle, then the lower blade.
    The advantage is latched: whoever got under at impact stays under while the front contact lasts, even after
    the bodies swing flush (that happens within tens of ms), and the loser cannot get under the winner.
    How far the blade gets under is proportional to the advantage: lift → lift_max · share while pushing (over
    lift_time), so a slightly better angle gives a slight lift, not the whole contest. An all-or-nothing rule made
    tiny, unobservable differences decide matches and drove policy styles into rock-paper-scissors cycles.
    Lift drops quickly once contact is lost.
    """
    if isinstance(dt2, torch.Tensor):
        down = 1 - torch.exp(-dt2 / p.drop_time)
    else:
        down = 1 - math.exp(-dt2 / p.drop_time)
    n_out = torch.stack([n01, -n01], dim=1)  # [N,R,2] from each robot towards the other
    cos = (n_out * fwd).sum(-1)
    front = cos > p.front_cos
    mis = torch.atan2(_cross(fwd, n_out).abs(), cos)  # atan2, not acos: acos near 1 turns float noise into 1e-4 rad
    adv = (mis - mis.flip(1)) / p.angle_adv + (p.clearance.flip(1) - p.clearance) / p.clearance_adv
    # a near-tie is a clinch: without the deadband, float noise in symmetric contacts systematically picked robot 0
    adv = (adv - p.adv_deadband) / (1 - p.adv_deadband)
    share = torch.where(front.flip(1), adv.clamp(0, 1), torch.ones_like(adv))  # opponent's front too → contest
    pushing = front & touching.unsqueeze(-1)
    latched = torch.maximum(state.wedge, torch.where(state.wedge.flip(1) > 0, torch.zeros_like(share), share))
    decayed = state.wedge * (1 - down)
    decayed = torch.where(decayed < WEDGE_EPS, torch.zeros_like(decayed), decayed)  # exp decay never hits 0
    state.wedge.copy_(torch.where(pushing, latched, decayed))
    by_me = torch.where(pushing, p.lift_max * state.wedge, torch.zeros_like(share))
    target = by_me.flip(1)  # lift of each robot = what the other one's wedge achieves
    if isinstance(dt2, torch.Tensor):
        up = 1 - torch.exp(-dt2 / p.lift_time)
    else:
        up = 1 - math.exp(-dt2 / p.lift_time)
    alpha = torch.where(target > state.lift, up, down)
    state.lift.add_(alpha * (target - state.lift))


def step(
    state: PhysState,
    p: PhysParams,
    pwm: torch.Tensor,
    ext_force: torch.Tensor | None = None,
    dt: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Advance one physics step in place. pwm: [N,R,2] in [-1, 1]. ext_force: [N,R,2] at the CoM (tests).

    dt: per-env step [N] (control-loop jitter: each env splits its own control period into substeps); default p.dt.
    """
    if dt is None:
        dt2 = dt3 = p.dt
    else:
        dt2, dt3 = dt.view(-1, 1), dt.view(-1, 1, 1)
    fwd, left = _rot(state.theta)

    # --- motor: deadband → first-order lag → torque-speed line
    u = pwm.clamp(-1, 1)
    db = p.deadband.unsqueeze(-1)
    u_m = torch.sign(u) * ((u.abs() - db) / (1 - db)).clamp(min=0)
    alpha = (1 - torch.exp(-dt2 / p.motor_tc)).unsqueeze(-1)
    state.u_lag.add_(alpha * (u_m - state.u_lag))
    drive = state.u_lag * p.volt_ratio.unsqueeze(-1)
    b = p.tau_stall / p.omega0.unsqueeze(-1)
    tau_m = p.tau_stall * drive - b * state.wheel_w
    r_w = p.wheel_r.unsqueeze(-1)  # [N,R,1]

    # --- tires, wheel contact arms relative to the CoM: [N,R,2(wheel),2]
    arm_x = (p.axle_x - p.com_x).unsqueeze(-1)
    arm_y = const((p.half_track, -p.half_track), pwm.device)
    arms = arm_x.unsqueeze(-1) * fwd.unsqueeze(-2) + arm_y.view(1, 1, 2, 1) * left.unsqueeze(-2)
    v_pt = _point_vel(state.vel.unsqueeze(-2), state.omega.unsqueeze(-1), arms)
    fwd_w = fwd.unsqueeze(-2)
    slip = v_pt - (state.wheel_w * r_w).unsqueeze(-1) * fwd_w
    mu_n = (p.tire_mu * p.n_wheel * (1 - state.lift)).unsqueeze(-1).expand_as(state.wheel_w)
    f_tire = _friction(slip, mu_n, p.s0)  # on the robot

    # --- wheel speed: J·dω/dt = τ_m − F_long·r, linearised backward Euler, then worm-gear clamp
    f_long = (f_tire * fwd_w).sum(-1)
    slip_mag = slip.norm(dim=-1)
    k_t = mu_n * r_w**2 / p.s0 * (1 - torch.tanh(slip_mag / p.s0) ** 2)
    w_new = state.wheel_w + dt3 * (tau_m - f_long * r_w) / (p.j_wheel + dt3 * (b + k_t))
    w_free = drive * p.omega0.unsqueeze(-1)
    state.wheel_w.copy_(torch.clamp(w_new, torch.minimum(w_free, torch.zeros_like(w_free)), torch.maximum(w_free, torch.zeros_like(w_free))))

    # --- scoop sliding on the ring
    arm_s = (p.scoop_x - p.com_x).unsqueeze(-1) * fwd
    carried = (2 * p.n_wheel * state.lift).flip(1)  # opponent's weight resting on this robot's wedge
    f_scoop = _friction(_point_vel(state.vel, state.omega, arm_s), p.scoop_mu * (p.n_scoop + carried), p.s0)

    force = f_tire.sum(-2) + f_scoop
    torque = _cross(arms, f_tire).sum(-1) + _cross(arm_s, f_scoop)
    if ext_force is not None:
        force = force + ext_force

    # --- robot-robot contact
    ctr = state.pos - p.com_x.unsqueeze(-1) * fwd
    pts, n01, depth = box_contacts(ctr, state.theta, p.half)  # [N,P,2], [N,2], [N,P]
    r0 = pts - state.pos[:, 0:1]
    r1 = pts - state.pos[:, 1:2]
    v_rel = _point_vel(state.vel[:, 1:2], state.omega[:, 1:2], r1) - _point_vel(state.vel[:, 0:1], state.omega[:, 0:1], r0)
    n = n01.unsqueeze(1)
    vn = (v_rel * n).sum(-1)
    fn = torch.where(depth > 0, (p.k_contact * depth - p.c_contact * vn).clamp(min=0), torch.zeros_like(depth))
    vt = v_rel - vn.unsqueeze(-1) * n
    f_on_1 = fn.unsqueeze(-1) * n + _friction(vt, p.body_mu.unsqueeze(-1) * fn, p.s0)
    force = force + torch.stack([-f_on_1.sum(1), f_on_1.sum(1)], dim=1)
    torque = torque + torch.stack([-_cross(r0, f_on_1).sum(1), _cross(r1, f_on_1).sum(1)], dim=1)

    _update_lift(state, p, fwd, n01, fn.sum(-1) > 0, dt2)

    # --- semi-implicit Euler
    state.vel.add_(dt3 * force * p.inv_mass.unsqueeze(-1))
    state.omega.add_(dt2 * torque * p.inv_inertia)
    state.pos.add_(dt3 * state.vel)
    state.theta.add_(dt2 * state.omega)

    return {"contact_normal_force": fn.sum(-1), "tire_force": f_tire, "contact_depth": depth.amax(-1)}
