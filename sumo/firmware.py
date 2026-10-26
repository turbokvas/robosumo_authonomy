"""The firmware's actuation layer, shared by the sim and (ported 1:1 to C) the robot.

policy action (fraction of max, [-1, 1]) → slew limit (in action space) → deadband compensation → motor duty.
Same code path for every robot in the sim, FSM opponents included, since they run on the same firmware.
"""

from __future__ import annotations

import torch

ZERO_ACTION = 1e-3  # |action| below this means "stop": no deadband kick


def slew(prev: torch.Tensor, target: torch.Tensor, limit: float) -> torch.Tensor:
    """Move from `prev` towards `target` (clamped to [-1, 1]) by at most `limit` per control step."""
    return prev + (target.clamp(-1, 1) - prev).clamp(-limit, limit)


def deadband_comp(action: torch.Tensor, deadband: torch.Tensor) -> torch.Tensor:
    """Map a drive fraction to duty so that any non-zero command moves the motor: duty = sign·(db + (1 − db)·|a|).

    deadband: the calibrated deadband per motor, broadcastable to action ([N,R,1] for [N,R,2])."""
    mag = deadband + (1 - deadband) * action.abs()
    return torch.where(action.abs() > ZERO_ACTION, torch.sign(action) * mag, torch.zeros_like(action))
