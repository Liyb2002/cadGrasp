"""Stage 0b: approximate convex decomposition of every object (CoACD).

MuJoCo collides meshes as their convex hulls, so a hook or a C-shaped part would
settle on a hull face that does not exist in reality. Each object is therefore
split into convex pieces which become separate collision geoms.

Writes objects/<name>/collision/part_XX.obj and adds the piece count to meta.json.

    python slides/tools/decompose.py [--force] [--jobs 6]
"""
from __future__ import annotations

import argparse
import shutil
import time
from multiprocessing import Pool

import trimesh

from common import obj_path, objects_with_meshes, read_json, write_json

# Objects at least this close to their convex hull are kept as a single piece.
CONVEX_TOL = 0.97
COACD_THRESHOLD = 0.04  # concavity tolerance; smaller = more pieces
# CoACD occasionally emits zero-thickness slivers; MuJoCo refuses to compile those.
SLIVER_FRACTION = 1e-5  # of the object's volume


def prune_slivers(parts: list, total_volume: float) -> list:
    keep = [p for p in parts if p.volume > SLIVER_FRACTION * total_volume]
    return keep or parts[:1]


def decompose_one(args: tuple[str, bool]) -> tuple[str, int, int, float]:
    name, force = args
    d = obj_path(name)
    col = d / "collision"
    meta = read_json(d / "meta.json")
    t0 = time.time()

    cached = col.exists() and not force and any(col.glob("part_*.obj"))
    if cached:
        # Cheap re-run: keep the decomposition, just re-apply the sliver filter.
        parts = [trimesh.load(p, force="mesh") for p in sorted(col.glob("part_*.obj"))]
    elif meta["hull_volume_ratio"] >= CONVEX_TOL:
        parts = [trimesh.load(d / "mesh.stl", force="mesh").convex_hull]
    else:
        import coacd

        coacd.set_log_level("error")
        mesh = trimesh.load(d / "mesh.stl", force="mesh")
        raw = coacd.run_coacd(
            coacd.Mesh(mesh.vertices, mesh.faces),
            threshold=COACD_THRESHOLD,
            max_convex_hull=48,
        )
        parts = [trimesh.Trimesh(v, f).convex_hull for v, f in raw]

    n_before = len(parts)
    parts = prune_slivers(parts, meta["volume_m3"])

    if not (cached and len(parts) == n_before):
        if col.exists():
            shutil.rmtree(col)
        col.mkdir(parents=True)
        for i, p in enumerate(parts):
            p.export(col / f"part_{i:02d}.obj")

    meta["n_collision_parts"] = len(parts)
    meta["n_slivers_pruned"] = n_before - len(parts)
    meta["coacd_threshold"] = (None if meta["hull_volume_ratio"] >= CONVEX_TOL
                               else COACD_THRESHOLD)
    write_json(d / "meta.json", meta)
    return name, len(parts), n_before - len(parts), time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    names = args.only or objects_with_meshes()
    with Pool(args.jobs) as pool:
        for name, n, pruned, secs in pool.imap_unordered(
                decompose_one, [(n, args.force) for n in names]):
            note = f"  (pruned {pruned} sliver{'s' if pruned > 1 else ''})" if pruned else ""
            print(f"{name:6s} {n:3d} convex pieces  ({secs:5.1f}s){note}", flush=True)


if __name__ == "__main__":
    main()
