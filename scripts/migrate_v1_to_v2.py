"""Move interface-v1 policies (3 frames, 60 inputs) onto v2 (frames at FRAME_LAGS + firmware memory) unchanged.

The v1 frames were lags (2, 1, 0) with the same frame layout, so their weights map 1:1. New inputs (older frames,
memory) get zero first-layer weights; the old frames' normalization stats are reused for the new lags, memory
starts at mean 0 / var 1. Output: actor-only init checkpoint (the critic input changed; train a fresh one).

    uv run python scripts/migrate_v1_to_v2.py SRC.pt DST.pt [SRC2.pt DST2.pt ...]
"""

import sys

import torch

from sumo import interface as iface
from sumo.config import load_spec
from train.models import Actor

V1_LAGS = (2, 1, 0)
V1_DIM = len(V1_LAGS) * iface.FRAME_DIM  # v1 frame layout == v2 frame layout


def migrate(src: str, dst: str) -> None:
    old = torch.load(src, map_location="cpu", weights_only=False)
    assert old.get("interface_version") == 1 and old["obs_dim"] == V1_DIM, f"{src} is not an interface-v1 checkpoint"
    F = iface.FRAME_DIM
    old_cols = torch.arange(V1_DIM)
    new_cols = torch.tensor([iface.FRAME_LAGS.index(lag) * F + i for lag in V1_LAGS for i in range(F)])

    sd = old["actor"]
    new = Actor(iface.OBS_DIM, list(iface.HIDDEN)).state_dict()
    for key, value in sd.items():
        if key == "net.0.weight":
            w = torch.zeros(value.shape[0], iface.OBS_DIM)
            w[:, new_cols] = value[:, old_cols]
            new[key] = w
        elif key in ("norm.mean", "norm.var"):
            v = torch.zeros(iface.OBS_DIM) if key == "norm.mean" else torch.ones(iface.OBS_DIM)
            newest = value[-F:]  # stats of the newest v1 frame for every lag
            for k in range(len(iface.FRAME_LAGS)):
                v[k * F:(k + 1) * F] = newest
            v[new_cols] = value[old_cols]
            new[key] = v
        else:
            new[key] = value

    a_old, a_new = Actor(V1_DIM, list(iface.HIDDEN)), Actor(iface.OBS_DIM, list(iface.HIDDEN))
    a_old.load_state_dict(sd)
    a_new.load_state_dict(new)
    x_old = torch.rand(4096, V1_DIM).round()
    x_new = torch.rand(4096, iface.OBS_DIM)  # new inputs random: zero weights must ignore them
    x_new[:, new_cols] = x_old
    with torch.no_grad():
        diff = (a_old(x_old) - a_new(x_new)).abs().max().item()
    assert diff < 1e-5, diff

    ckpt = {
        "actor": new, "obs_dim": iface.OBS_DIM, "step": old["step"], "cfg": old["cfg"], "spec": load_spec(),
        "git": old["git"], "migrated_from": src,
    }
    torch.save(iface.stamp(ckpt, ckpt["spec"]), dst)
    print(f"{src} -> {dst}  (v{iface.INTERFACE_VERSION}, max output diff {diff:.1e})")


if __name__ == "__main__":
    args = sys.argv[1:]
    for s, d in zip(args[::2], args[1::2]):
        migrate(s, d)
