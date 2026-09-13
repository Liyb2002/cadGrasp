"""The two-piece merge, run on the sharp cube instead of A1-f.

Everything about joining loose walls into printable pieces -- routing plates over
the free floor, ranking every two-group split by how much plate it needs, and
checking the result with an exact boolean -- is `merge_a1f`'s and is reused
verbatim. Only one thing differs between the two objects and it is upstream of
all of that: the work region. A1-f is sprayed by an unbounded gun pass, while a
six-faced cube needs `spot_region`'s finite beam, because an unbounded pass on a
part whose faces are a sixth of it each simply eats whole faces.

So this file is a replicate() with the other region sampler and nothing else. It
asserts against `capped/capped_k1.json` for the same reason `merge_a1f` does:
everything downstream draws the recorded answer rather than recomputing it, so
replicating the wrong region would produce a careful picture of a different
problem.

The cube is the harder case for a merge and that is why it is worth running. Its
contacts are face normals of a box, so they point at right angles to each other
and their walls stand on different sides of the part -- there is less chance that
two of them share a side to route along. Five or six contacts into two pieces may
simply not be possible; if so the run says so rather than shrinking anything.

    python slides/tools/merge_cuboid.py cuboid_baseline --pose 0
"""
from __future__ import annotations

import sys

import numpy as np
import trimesh
from scipy.spatial import cKDTree

import merge_a1f
import pipeline_a1f
from capped_a1f import lp_answered
from common import obj_path, read_json
from cover import UP
from opposing_supports import tiling
from shrink_support import angled_pushes
from supports import CONTACT_EPS
from work_regions import BOUNDARY_EDGE, spot_region


def replicate(name: str, pose: int, k: float = 1.0, seed: int = 0,
              n_points: int = 90, n_dirs: int = 24, check: bool = True) -> dict:
    """`pipeline_a1f.replicate` with the spot region, and the same four assertions.

    The four recorded numbers a wrong replication cannot match by accident are
    how many requirements the region generated, how much surface it covers, how
    many distinct push directions the part offers, and what fraction the chosen
    set answers. All four are asserted rather than printed.
    """
    d = obj_path(name)
    mesh = trimesh.load(d / "mesh.stl", force="mesh")
    examples = read_json(d / "tips" / "tips.json")["examples"]
    T = np.asarray(examples[pose]["T_world_mesh"])
    R, t = T[:3, :3], T[:3, 3]

    rng = np.random.default_rng(seed + 1000 * pose)
    reg = spot_region(mesh, T, rng, float(rng.uniform(0.17, 0.33)),
                      BOUNDARY_EDGE * float(np.linalg.norm(mesh.extents)))
    part, inside = reg.pop("mesh"), reg.pop("mask")
    n_passes = reg["n_passes"]
    area = float(part.area_faces[inside].sum() / part.area)

    pw, pu = angled_pushes(part, T, inside, n_points, n_dirs, seed + pose)
    targets = UP - k * pu
    keep = np.linalg.norm(targets, axis=1) > 1e-9
    targets, pw, pu = targets[keep], pw[keep], pu[keep]

    on_floor = (part.triangles_center @ R.T + t)[:, 2] <= CONTACT_EPS
    off = np.flatnonzero(~inside & ~on_floor)
    push = -(part.face_normals[off] @ R.T)
    push /= np.linalg.norm(push, axis=1, keepdims=True)
    ico, tiles = tiling(4)
    tree = cKDTree(tiles)
    hit = tree.query(push)[1]
    first = {}
    for j, h in enumerate(hit):
        first.setdefault(h, j)
    rep = np.array([first[h] for h in sorted(first)])
    avail, avail_face = push[rep], off[rep]

    rec = {r["pose"]: r for r in read_json(d / "capped" / f"capped_k{k:g}.json")["poses"]}[pose]
    # matched BY DIRECTION, not by index: only the directions are stable across runs
    chosen = [int(np.argmax(avail @ np.asarray(s["push"]))) for s in rec["supports"]]
    G = np.column_stack([UP] + [avail[c] for c in chosen])
    lp = lp_answered(G, targets)
    pts = part.triangles_center[avail_face[chosen]] @ R.T + t
    us = avail[chosen]

    if check:
        for got, want, what in [(len(targets), rec["n_targets"], "n_targets"),
                                (len(avail), rec["n_available"], "n_available")]:
            assert got == want, f"pose {pose} {what}: replicated {got}, recorded {want}"
        assert abs(area - rec["region_area_fraction"]) < 1e-12, "region area drifted"
        assert abs(float(lp.mean()) - rec["answered_fraction"]) < 1e-9, "answer drifted"
        assert np.allclose(pts, [s["p"] for s in rec["supports"]], atol=1e-9)
        assert np.allclose(us, [s["push"] for s in rec["supports"]], atol=1e-9)

    return dict(mesh=mesh, part=part, T=T, inside=inside, n_passes=n_passes,
                area=area, pw=pw, pu=pu, targets=targets, avail=avail,
                avail_face=avail_face, chosen=chosen, pts=pts, us=us, G=G,
                lp=lp, ico=ico, tiles=tiles, tree=tree, record=rec, k=k)


if __name__ == "__main__":
    # the merge is `merge_a1f`'s from here on; only the region sampler is swapped,
    # and it is swapped in both places because the drawing code reaches for the
    # module attribute rather than the name imported at the top
    merge_a1f.replicate = replicate
    pipeline_a1f.replicate = replicate
    if len(sys.argv) == 1:
        sys.argv += ["cuboid_baseline"]
    merge_a1f.main()
