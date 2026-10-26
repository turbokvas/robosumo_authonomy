"""Frozen policy interface, version 1: what the actor sees, what it outputs, what the critic sees.

Checkpoints are only interchangeable (fine-tuning, league, export) while this file is unchanged. Rules:
- Do not change the composition, order, scaling or stack depth of the actor obs, or the action semantics, without
  bumping INTERFACE_VERSION and noting it in PLAN.md. Loading a checkpoint of another version fails loudly.
- Physics changes go through robot_spec.yaml only; they never touch this file.
- The actor gets only what the robot's MCU has. Privileged state goes to the critic only.

Actor observation = OBS_STACK frames (oldest first, newest last), each frame = FRAME fields in this order.
Values are raw sensor readings divided by the fixed scales below (not by spec values, so the meaning of an input
does not shift when the hardware in the spec changes). The actor's RunningNorm then standardizes them; its
statistics are part of the checkpoint.

Reserved sensors (side IR, wheel encoders, IMU) are not on the robot yet. The sim always models them; each
group is present or absent for a whole match (absent → zeros and its has_* flag = 0). Training drops groups at
random (env.sensor_dropout) so one policy works with and without them; eval uses spec `installed` by default.
First-layer weights of reserved inputs start at zero, so adding them changes nothing until training uses them.

Action = [u_left, u_right] in [-1, 1]: fraction of the maximum drive (duty at the current battery voltage),
forward positive. The firmware layer (sumo/firmware.py, mirrored in C) applies the slew limit in action space,
then deadband compensation, and outputs the motor duty.
"""

from __future__ import annotations

import hashlib
from typing import Any

import yaml

INTERFACE_VERSION = 1

OBS_STACK = 3
HIDDEN = (64, 64)  # actor MLP; changing it needs an MCU timing measurement and a new interface version
ACT_DIM = 2

# fixed input scales (physical units → O(1)); part of the interface
WHEEL_SPEED_SCALE = 25.0   # rad/s
GYRO_SCALE = 10.0          # rad/s
ACCEL_SCALE = 9.81         # m/s²

# (field, width) in frame order
FRAME: tuple[tuple[str, int], ...] = (
    ("ir_front", 3),       # E18 left (+15°), centre, right (−15°): 1 = opponent seen
    ("line", 4),           # front-left, front-right, rear-left, rear-right: 1 = over the black edge
    ("prev_action", 2),    # last applied action (after the slew limit), [-1, 1]
    ("battery", 1),        # V / full_v
    ("ir_side", 2),        # reserved: side E18 left (+90°), right (−90°)
    ("wheel_speed", 2),    # reserved: encoder wheel speed left, right / WHEEL_SPEED_SCALE (quantized)
    ("gyro_z", 1),         # reserved: IMU yaw rate / GYRO_SCALE
    ("accel_xy", 2),       # reserved: IMU body-frame acceleration (forward, left) / ACCEL_SCALE
    ("has_side_ir", 1),    # 1 if the side E18 are present this match
    ("has_encoders", 1),
    ("has_imu", 1),
)
SENSOR_GROUPS: dict[str, tuple[str, ...]] = {   # group → frame fields it fills; order = has_* flags order
    "side_ir": ("ir_side",),
    "encoders": ("wheel_speed",),
    "imu": ("gyro_z", "accel_xy"),
}
RESERVED_FIELDS = ("ir_side", "wheel_speed", "gyro_z", "accel_xy", "has_side_ir", "has_encoders", "has_imu")

FRAME_SLICES: dict[str, slice] = {}
_o = 0
for _name, _w in FRAME:
    FRAME_SLICES[_name] = slice(_o, _o + _w)
    _o += _w
FRAME_DIM = _o
OBS_DIM = FRAME_DIM * OBS_STACK


def reserved_obs_columns() -> list[int]:
    """Indices of reserved inputs in the flattened actor obs (all stacked frames)."""
    cols = []
    for k in range(OBS_STACK):
        for name in RESERVED_FIELDS:
            s = FRAME_SLICES[name]
            cols += [k * FRAME_DIM + i for i in range(s.start, s.stop)]
    return cols


def check_spec(spec: dict[str, Any]) -> None:
    """The spec must agree with the frozen interface on everything the interface fixes."""
    pol = spec["controller"]["policy"]
    if pol["obs_frame_stack"] != OBS_STACK or tuple(pol["hidden"]) != HIDDEN:
        raise ValueError(
            f"robot_spec.yaml policy shape (stack {pol['obs_frame_stack']}, hidden {pol['hidden']}) differs from "
            f"interface v{INTERFACE_VERSION} (stack {OBS_STACK}, hidden {list(HIDDEN)}): bump the interface version"
        )
    if len(spec["robot"]["sensors"]["opponent_ir"]["mounts"]) != FRAME_SLICES["ir_front"].stop - FRAME_SLICES["ir_front"].start:
        raise ValueError("opponent_ir mount count differs from the interface")


def spec_hash(spec: dict[str, Any]) -> str:
    return hashlib.sha256(yaml.safe_dump(spec, sort_keys=True, allow_unicode=True).encode()).hexdigest()[:16]


def stamp(ckpt: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    ckpt["interface_version"] = INTERFACE_VERSION
    ckpt["spec_hash"] = spec_hash(spec)
    return ckpt


class InterfaceMismatch(RuntimeError):
    pass


def check_checkpoint(ckpt: dict[str, Any], path: Any = "") -> None:
    v = ckpt.get("interface_version")
    if v != INTERFACE_VERSION:
        raise InterfaceMismatch(
            f"checkpoint {path} has interface_version {v}, code expects {INTERFACE_VERSION}; "
            "checkpoints of different interface versions cannot be loaded or mixed (see sumo/interface.py)"
        )
    if ckpt.get("obs_dim") != OBS_DIM:
        raise InterfaceMismatch(f"checkpoint {path} obs_dim {ckpt.get('obs_dim')} != {OBS_DIM}")


# ---- critic (sim only) --------------------------------------------------------------------------------------------
# The critic input is privileged and versioned with the interface too (resume needs a matching critic), but it is
# cheap to re-learn: a fine-tune may start a fresh critic with a warm-up. Its layout is built in SumoTask.critic_obs
# from these fixed lists, so widening or adding *_rand ranges in the spec does not change it.
CRITIC_ROBOT_PARAMS = (     # PhysParams fields per robot, own then opponent
    "inv_mass", "inv_inertia", "com_x", "wheel_r", "omega0", "tire_mu", "scoop_mu", "n_wheel", "n_scoop",
    "motor_tc", "deadband", "deadband_cal", "volt_ratio", "lift_max", "clearance",
)
CRITIC_ROBOT_TAU = True     # tau_stall left/right per robot
CRITIC_ENV_PARAMS = ("ring_radius", "body_mu")
N_FSM_MODES = 7
