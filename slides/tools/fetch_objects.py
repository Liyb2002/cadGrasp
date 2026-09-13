"""Stage 0a: fetch the Passive Grippers test suite and compute mass properties.

Writes objects/<name>/mesh.stl and objects/<name>/meta.json.
The mesh is stored exactly as downloaded -- every pose in the pipeline is
expressed in this untouched mesh frame, so nothing has to be un-normalized later.

    python slides/tools/fetch_objects.py [--force]
"""
from __future__ import annotations

import argparse
import urllib.request

import numpy as np
import trimesh

from common import DENSITY, OBJ_DIR, PG_OBJECTS, PG_URL, obj_path, write_json


def fetch_one(name: str, force: bool) -> dict:
    d = obj_path(name)
    d.mkdir(parents=True, exist_ok=True)
    stl = d / "mesh.stl"
    if force or not stl.exists():
        urllib.request.urlretrieve(PG_URL.format(name=name), stl)

    mesh = trimesh.load(stl, force="mesh")
    watertight = bool(mesh.is_watertight)
    if not watertight:
        mesh.fill_holes()
        mesh.export(stl)
        mesh = trimesh.load(stl, force="mesh")

    mesh.density = DENSITY
    meta = {
        "name": name,
        "source": f"milmillin/passive-gripper data/stl/{name}.stl",
        "units": "m",
        "n_vertices": int(len(mesh.vertices)),
        "n_faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "was_repaired": not watertight,
        "extents_m": [float(v) for v in mesh.extents],
        "bounds_m": [[float(v) for v in row] for row in mesh.bounds],
        "volume_m3": float(mesh.volume),
        "density_kg_m3": DENSITY,
        "mass_kg": float(mesh.mass),
        # Centre of mass and inertia tensor about the CoM, in the mesh frame.
        "com_mesh_frame": [float(v) for v in mesh.center_mass],
        "inertia_com": [[float(v) for v in row] for row in mesh.moment_inertia],
        "convex": bool(mesh.is_convex),
        # How far the volume is from its own convex hull: a rough concavity score.
        "hull_volume_ratio": float(mesh.volume / mesh.convex_hull.volume),
    }
    write_json(d / "meta.json", meta)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    names = args.only or PG_OBJECTS
    rows = []
    for name in names:
        m = fetch_one(name, args.force)
        rows.append(m)
        print(f"{name:6s} V={m['n_vertices']:6d} F={m['n_faces']:6d} "
              f"wt={str(m['watertight']):5s} ext={np.round(m['extents_m'], 3)} "
              f"mass={m['mass_kg']:.3f}kg hullratio={m['hull_volume_ratio']:.2f}")

    write_json(OBJ_DIR / "index.json", {
        "source": "Passive Grippers (Kodnongbua et al., SIGGRAPH 2022) test suite",
        "url": "https://github.com/milmillin/passive-gripper",
        "density_kg_m3": DENSITY,
        "objects": [
            {k: r[k] for k in ("name", "extents_m", "volume_m3", "mass_kg",
                               "n_faces", "watertight", "hull_volume_ratio")}
            for r in rows
        ],
    })
    print(f"\n{len(rows)} objects -> {OBJ_DIR}")


if __name__ == "__main__":
    main()
