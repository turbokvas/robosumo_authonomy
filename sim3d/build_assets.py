"""STEP from Fusion → MuJoCo assets in the robot frame, plus a geometry report against robot_spec.yaml.

Robot frame (same as the 2D sim): x forward, y left, z up; origin at the footprint's geometric centre, on the
ground plane (z = 0 at the wedge tip). The Fusion export has the wedge towards −X, so the frame is rotated 180°
about z and shifted. Outputs (sim3d/assets/generated/, regenerated from the STEP, not edited by hand):
    body.stl              visual mesh of the body
    body_col_XX.stl       convex slabs of the body (kept for inspection; not used for contacts any more)
    geometry.yaml         also lists the collision boxes: main body, front lip and sloped plates following the
                          scoop profile. Boxes because box-box / box-plane contacts are analytic in MuJoCo; mesh and
                          cylinder pairs go through GJK/EPA, whose fixed EPA horizon produced bogus depths and launched
                          robots into the air.
    wheel.stl             one wheel, centred on its axle, axle along y
    geometry.yaml         measured dimensions used by the MJCF builder (and to update the spec)

    uv run python -m sim3d.build_assets [--step sim3d/assets/robot_mdl.step]
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import cascadio
import numpy as np
import trimesh
import yaml

from sumo.config import get, load_spec

ASSETS = Path(__file__).resolve().parent / "assets"
OUT = ASSETS / "generated"


def load_solids(step: Path) -> list[trimesh.Trimesh]:
    """Closed solids of the STEP (zero-volume decal surfaces dropped), in metres."""
    with tempfile.TemporaryDirectory() as tmp:
        glb = Path(tmp) / "model.glb"
        if cascadio.step_to_glb(str(step), str(glb), tol_linear=0.05, tol_angular=0.2) != 0:
            raise RuntimeError(f"STEP conversion failed: {step}")
        mesh = trimesh.load(glb, force="scene").to_geometry()
    mesh.merge_vertices(digits_vertex=6)
    solids = [p for p in mesh.split(only_watertight=False) if p.is_watertight and abs(p.volume) > 1e-6]
    return sorted(solids, key=lambda p: -abs(p.volume))


def collision_boxes(body: trimesh.Trimesh, t: float, clearance: float, scoop_z=(0.004, 0.011)) -> list[dict]:
    """Boxes (centre, half sizes, pitch about y in degrees) approximating the body for contacts, from the CAD profile
    at y = 0: main box behind the vertical front face, the front lip, and plates along the scoop curve split where
    it reaches the heights in scoop_z."""
    sec = body.section(plane_origin=[0, 0, 0], plane_normal=[0, 1, 0])
    prof = max(sec.discrete, key=len)[:, [0, 2]]
    half_w = (body.bounds[1, 1] - body.bounds[0, 1]) / 2
    tip_x = prof[:, 0].max()
    curve = prof[(prof[:, 1] < 0.025) & (prof[:, 0] > tip_x - 0.05)]
    curve = curve[np.argsort(-curve[:, 0])]
    face_x = prof[(prof[:, 1] > 0.03) & (prof[:, 1] < 0.055), 0].max()  # vertical front face
    top = body.bounds[1, 2]
    lip = prof[prof[:, 1] > 0.06]
    lip_x, lip_z = lip[:, 0].max(), lip[lip[:, 0] > face_x, 1]
    scoop_top = curve[:, 1].max()
    knots = [curve[0]] + [curve[np.argmin(np.abs(curve[:, 1] - z))] for z in scoop_z] + [curve[np.argmax(curve[:, 1])]]

    boxes = [{"name": "main", "pos": [(face_x + body.bounds[0, 0]) / 2, 0.0, (clearance + top) / 2],
              "size": [(face_x - body.bounds[0, 0]) / 2, half_w, (top - clearance) / 2], "pitch_deg": 0.0}]
    if len(lip_z):
        boxes.append({"name": "lip", "pos": [(face_x + lip_x) / 2, 0.0, (lip_z.min() + lip_z.max()) / 2],
                      "size": [(lip_x - face_x) / 2, half_w, (lip_z.max() - lip_z.min()) / 2], "pitch_deg": 0.0})
    for i, (a, b) in enumerate(zip(knots[:-1], knots[1:])):
        d = b - a  # from the front-low end to the back-high end of this stretch of the scoop
        length = float(np.hypot(*d))
        th = float(np.arctan2(d[1], -d[0]))  # slope from horizontal
        inward = np.array([-np.sin(th), -np.cos(th)])  # into the material, below/behind the surface
        c = (a + b) / 2 + inward * t / 2
        low = min((a + inward * t)[1], (b + inward * t)[1])
        c[1] -= min(low, 0.0)  # keep the plate above the ring (tip stays at ~0)
        boxes.append({"name": f"scoop{i}", "pos": [float(c[0]), 0.0, float(c[1])],
                      "size": [length / 2, half_w, t / 2], "pitch_deg": float(np.degrees(th))})
    for bx in boxes:
        bx["pos"] = [float(v) for v in bx["pos"]]
        bx["size"] = [float(v) for v in bx["size"]]
    print("collision boxes:", ", ".join(f"{b['name']}@{b['pitch_deg']:.0f}°" for b in boxes),
          f"(scoop top {scoop_top * 1000:.1f} mm)")
    return boxes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=Path, default=ASSETS / "robot_mdl.step")
    ap.add_argument("--scoop-slabs", type=int, default=2, help="convex slabs across the curved scoop")
    ap.add_argument("--plate-thickness", type=float, default=0.002, help="scoop plates (collision boxes), m")
    ap.add_argument("--underbody-clearance", type=float, default=0.003,
                    help="the main collision piece starts this high: only wheels and the blade touch the ring")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    solids = load_solids(args.step)
    if len(solids) != 3:
        raise RuntimeError(f"expected 3 solids (body + 2 wheels), got {len(solids)}")
    body, w1, w2 = solids  # the body is by far the largest

    # Fusion → robot frame: wedge tip is the min-X end; rotate 180° about z, put the footprint centre at the origin
    lo, hi = body.bounds
    centre = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    T = trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1])
    T[:3, 3] = -(T[:3, :3] @ centre)
    for m in (body, w1, w2):
        m.apply_transform(T)

    wheels = sorted([w1, w2], key=lambda w: -w.centroid[1])  # left (+y) first
    wc = [w.bounds.mean(0) for w in wheels]
    r = (wheels[0].bounds[1, 2] - wheels[0].bounds[0, 2]) / 2
    width = wheels[0].bounds[1, 1] - wheels[0].bounds[0, 1]
    wheel = wheels[0].copy()
    wheel.apply_translation(-wc[0])
    wheel.export(OUT / "wheel.stl")
    body.export(OUT / "body.stl")

    # Collision: CoACD gave ~15 pieces → up to 225 mesh-mesh pairs between two robots, slow and occasionally
    # unstable in the solver. The body is convex across y-z almost everywhere; along x only the scoop curves.
    # So: slabs along x — several thin ones over the scoop (keeps its real profile), one for the rest.
    tip_x = body.bounds[1, 0]
    scoop_end = body.vertices[body.vertices[:, 2] > 0.02][:, 0].max()  # where the vertical front face starts
    cuts = list(np.linspace(tip_x, scoop_end, args.scoop_slabs + 1)) + [body.bounds[0, 0]]
    parts = []
    for i, (x_hi, x_lo) in enumerate(zip(cuts[:-1], cuts[1:])):
        inside = body.vertices[(body.vertices[:, 0] <= x_hi + 1e-9) & (body.vertices[:, 0] >= x_lo - 1e-9)]
        pts = [inside]
        for x in (x_hi, x_lo):  # outline of the body on each cut plane
            sec = body.section(plane_origin=[x, 0, 0], plane_normal=[1, 0, 0])
            if sec is not None:
                pts.append(sec.vertices)
        pts = np.concatenate(pts)
        if i == len(cuts) - 2:
            # main piece: the real underbody has 1.4-3 mm clearance, but soft contacts let the wheels sink ~1 mm and
            # the flat bottom then rests on the ring (wrong load split, many contacts). Lift its bottom face.
            pts[:, 2] = np.maximum(pts[:, 2], args.underbody_clearance)
        hull = trimesh.convex.convex_hull(pts)
        parts.append((hull.vertices, hull.faces))
    for old in OUT.glob("body_col_*.stl"):
        old.unlink()
    for i, (v, f) in enumerate(parts):
        trimesh.Trimesh(v, f).export(OUT / f"body_col_{i:02d}.stl")

    b_lo, b_hi = body.bounds
    tip = body.vertices[np.argmax(body.vertices[:, 0])]
    boxes = collision_boxes(body, args.plate_thickness, args.underbody_clearance)
    geom = {
        "source": str(args.step.relative_to(ASSETS.parent.parent)).replace("\\", "/"),
        "body_length": float(b_hi[0] - b_lo[0]),
        "body_width": float(b_hi[1] - b_lo[1]),
        "body_height": float(b_hi[2] - b_lo[2]),
        "body_volume_m3": float(abs(body.volume)),
        "wedge_tip_x": float(tip[0]),
        "wheel_radius": float(r),
        "wheel_volume_m3": float(abs(wheels[0].volume)),
        "wheel_width": float(width),
        "track_width": float(wc[0][1] - wc[1][1]),
        "axle_x": float((wc[0][0] + wc[1][0]) / 2),
        "axle_z": float((wc[0][2] + wc[1][2]) / 2),
        "wheel_bottom_z": float(min(w.bounds[0, 2] for w in wheels)),
        "collision_pieces": len(parts),
        "collision_boxes": boxes,
    }
    (OUT / "geometry.yaml").write_text(yaml.safe_dump(geom, sort_keys=False), encoding="utf-8")

    spec = load_spec()
    rows = [
        ("length (footprint x)", geom["body_length"], get(spec, "robot.footprint")[0]),
        ("width (footprint y)", geom["body_width"], get(spec, "robot.footprint")[1]),
        ("height", geom["body_height"], get(spec, "robot.height")),
        ("wheel diameter", 2 * r, get(spec, "robot.drive.wheel_diameter")),
        ("wheel width", width, get(spec, "robot.drive.wheel_width")),
        ("track width", geom["track_width"], get(spec, "robot.drive.track_width")),
        ("axle x", geom["axle_x"], get(spec, "robot.drive.axle_x")),
        ("wedge contact x", geom["wedge_tip_x"], get(spec, "robot.front.contact_x")),
    ]
    print(f"{'':22}{'CAD':>9}{'spec':>9}   (mm)")
    for name, cad, sp in rows:
        flag = "   <-- differs" if abs(cad - sp) > 0.003 else ""
        print(f"{name:22}{cad * 1000:9.1f}{sp * 1000:9.1f}{flag}")
    print(f"collision: {len(parts)} convex pieces → {OUT}")


if __name__ == "__main__":
    main()
