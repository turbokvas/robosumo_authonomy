"""One-off: move the M3 policy (pre-interface, 30 inputs) onto interface v1 (60 inputs) without changing what it does.

The old frame [ir_front 3, line 4, prev_action 2, battery 1] is the start of the v1 frame, so those weights and
normalization stats map 1:1; reserved inputs get zero first-layer weights and mean 0 / var 1. The result is an
init checkpoint (actor only, no critic/optimizer): fine-tune from it with `init_ckpt`.

    uv run python scripts/migrate_m3_to_v1.py results/m3/ckpt_00500M.pt results/m3/ckpt_00500M_v1.pt
"""

import sys

import torch

from sumo import interface as iface
from sumo.config import load_spec
from train.models import Actor

OLD_FRAME = 10
OLD_STACK = 3


def main(src: str, dst: str) -> None:
    old = torch.load(src, map_location="cpu", weights_only=False)
    assert old["obs_dim"] == OLD_FRAME * OLD_STACK and "interface_version" not in old, "not a pre-v1 M3 checkpoint"
    assert iface.OBS_STACK == OLD_STACK and [f for f, _ in iface.FRAME[:4]] == ["ir_front", "line", "prev_action", "battery"]
    cols = torch.tensor([k * iface.FRAME_DIM + i for k in range(OLD_STACK) for i in range(OLD_FRAME)])

    sd = old["actor"]
    new = Actor(iface.OBS_DIM, list(iface.HIDDEN)).state_dict()
    for key, value in sd.items():
        if key == "net.0.weight":
            w = torch.zeros(value.shape[0], iface.OBS_DIM)
            w[:, cols] = value
            new[key] = w
        elif key in ("norm.mean", "norm.var"):
            v = torch.zeros(iface.OBS_DIM) if key == "norm.mean" else torch.ones(iface.OBS_DIM)
            v[cols] = value
            new[key] = v
        else:
            new[key] = value

    actor_old = Actor(OLD_FRAME * OLD_STACK, list(iface.HIDDEN))
    actor_old.load_state_dict(sd)
    actor_new = Actor(iface.OBS_DIM, list(iface.HIDDEN))
    actor_new.load_state_dict(new)
    x_old = torch.rand(4096, OLD_FRAME * OLD_STACK).round()
    x_new = torch.zeros(4096, iface.OBS_DIM)
    x_new[:, cols] = x_old
    with torch.no_grad():
        diff = (actor_old(x_old) - actor_new(x_new)).abs().max().item()
    print(f"max |old - new| on random obs with reserved inputs = 0: {diff:.2e}")
    assert diff < 1e-5

    ckpt = {
        "actor": new, "obs_dim": iface.OBS_DIM, "step": old["step"], "stage": old["stage"], "cfg": old["cfg"],
        "spec": load_spec(), "git": old["git"], "migrated_from": src,
    }
    torch.save(iface.stamp(ckpt, ckpt["spec"]), dst)
    print(f"wrote {dst} (interface v{iface.INTERFACE_VERSION})")


if __name__ == "__main__":
    main(*sys.argv[1:3])
