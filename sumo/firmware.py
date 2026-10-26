"""The firmware layer, shared by the sim and (ported 1:1 to C) the robot.

Actuation: policy action (fraction of max, [-1, 1]) → slew limit (in action space) → deadband compensation → duty.
Same code path for every robot in the sim, FSM opponents included, since they run on the same firmware.
Memory: SensorMemory keeps the counters the policy sees as MEMORY inputs (sumo/interface.py).
"""

from __future__ import annotations

import torch

from sumo import interface as iface

ZERO_ACTION = 1e-3  # |action| below this means "stop": no deadband kick


def slew(prev: torch.Tensor, target: torch.Tensor, limit: float) -> torch.Tensor:
    """Move from `prev` towards `target` (clamped to [-1, 1]) by at most `limit` per control step."""
    return prev + (target.clamp(-1, 1) - prev).clamp(-limit, limit)


def deadband_comp(action: torch.Tensor, deadband: torch.Tensor) -> torch.Tensor:
    """Map a drive fraction to duty so that any non-zero command moves the motor: duty = sign·(db + (1 − db)·|a|).

    deadband: the calibrated deadband per motor, broadcastable to action ([N,R,1] for [N,R,2])."""
    mag = deadband + (1 - deadband) * action.abs()
    return torch.where(action.abs() > ZERO_ACTION, torch.sign(action) * mag, torch.zeros_like(action))


class SensorMemory:
    """Per-robot counters updated once per control tick; on the MCU these are a few floats per sensor.

    Ages count seconds since a sensor last fired and saturate at MEMORY_HORIZON_S ("not recently"); a new match
    starts saturated. Absent side E18 never fire, so their ages stay saturated."""

    def __init__(self, n: int, n_robots: int, device: str | torch.device):
        z = lambda *s: torch.zeros(n, n_robots, *s, device=device)  # noqa: E731
        self.ir_age, self.line_age = z(5), z(4)
        self.last_side, self.engaged = z(), z()
        self.reset()

    def reset(self, idx: torch.Tensor | None = None) -> None:
        h = iface.MEMORY_HORIZON_S
        for t, v in ((self.ir_age, h), (self.line_age, h), (self.last_side, 0.0), (self.engaged, 0.0)):
            if idx is None:
                t.fill_(v)
            else:
                t.index_fill_(0, idx, v)

    def update(self, ir_front: torch.Tensor, ir_side: torch.Tensor, line: torch.Tensor, dt: torch.Tensor) -> None:
        """ir_front [N,R,3], ir_side [N,R,2] (zeros when absent), line [N,R,4] (bool/0-1); dt [N] seconds."""
        d = dt.view(-1, 1, 1)
        ir = torch.cat([ir_front, ir_side], -1) > 0
        self.ir_age = torch.where(ir, torch.zeros_like(self.ir_age), (self.ir_age + d).clamp(max=iface.MEMORY_HORIZON_S))
        ln = line > 0
        self.line_age = torch.where(ln, torch.zeros_like(self.line_age), (self.line_age + d).clamp(max=iface.MEMORY_HORIZON_S))
        left = ir[..., 0] | ir[..., 3]
        right = ir[..., 2] | ir[..., 4]
        centre = ir[..., 1]
        side = torch.where(centre, 0.0, torch.where(left & ~right, 1.0, torch.where(right & ~left, -1.0, self.last_side)))
        self.last_side = torch.where(ir.any(-1), side, self.last_side)
        self.engaged = torch.where(centre, (self.engaged + dt.view(-1, 1)).clamp(max=iface.ENGAGED_HORIZON_S),
                                   torch.zeros_like(self.engaged))

    def features(self) -> torch.Tensor:
        """[N,R,MEMORY_DIM] in interface order."""
        h = iface.MEMORY_HORIZON_S
        return torch.cat([self.ir_age[..., :3] / h, self.ir_age[..., 3:] / h, self.last_side.unsqueeze(-1),
                          self.line_age / h, (self.engaged / iface.ENGAGED_HORIZON_S).unsqueeze(-1)], -1)
