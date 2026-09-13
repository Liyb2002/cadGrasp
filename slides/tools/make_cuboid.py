"""The sharp-edged control: a cube, and ten tilted rests of it.

Every other object in the library is a fetched part with filleted edges, and a
fillet is the only thing that puts a *band* on the direction sphere: a flat face,
however large, carries one normal and therefore lands on one point. A1-f offers
ten points plus a band that is half the sphere; a cube offers six points and
nothing else, in three antipodal pairs. Once a work region takes one or two of
those faces there are four or five push directions in the entire universe of
candidates, and that is the whole point of building it -- it is the extreme
against which "the method needs curvature" can be read off.

Edge length is set from A1-f's volume, not its area: at the fixed density in
common.py that is what makes the mass equal, and the mass is what "one body
weight" in the disturbance scale means. 12 triangles, because a finer cube adds
no normal and only slows the direction dedup.

The ten poses are not what tip_examples.py would give. A cube's six stable
placements are congruent and its four footprint edges and four footprint corners
are congruent too, so that stage produces one edge tip and one corner tip, over
and over. What a rotational symmetry cannot absorb is the tip ANGLE, so the ten
are five edge tips and five corner tips at ten different angles, each carried out
by the same tip_examples machinery.

    python slides/tools/make_cuboid.py
"""
from __future__ import annotations
import coordinates as COORD

import argparse
import itertools

import numpy as np
import trimesh

from PIL import Image, ImageDraw

from common import DENSITY, obj_path, read_json, se3, write_json
from tip_examples import _font, make_example, render, tip_limits

NAME = "cuboid_baseline"
# A1-f, so that the mass and hence the "one body weight" disturbance scale match
REFERENCE_VOLUME_M3 = 0.0005551458716590356

# Below the balance limit -- 45 deg about a footprint edge, atan(sqrt 2) = 54.7
# about a footprint corner -- and spread over what is left of each range.
EDGE_TIPS = (12.0, 20.0, 27.0, 35.0, 43.0)
CORNER_TIPS = (15.0, 25.0, 34.0, 44.0, 52.0)


def cube_rotations() -> np.ndarray:
    """The 24 rotations that carry a cube onto itself: signed permutations, det +1."""
    out = []
    for p in itertools.permutations(range(3)):
        P = np.eye(3)[list(p)]
        for s in itertools.product((1.0, -1.0), repeat=3):
            S = P * np.asarray(s)[:, None]
            if np.linalg.det(S) > 0:
                out.append(S)
    return np.asarray(out)


def apart_deg(Ra: np.ndarray, Rb: np.ndarray, group: np.ndarray) -> float:
    """Angle between two orientations after the cube's own symmetry is divided out."""
    D = np.einsum("ij,njk->nik", Ra.T @ Rb, group)
    c = np.clip((np.trace(D, axis1=1, axis2=2) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(c)).min())


def tilt_apart_deg(Ra: np.ndarray, Rb: np.ndarray, group: np.ndarray) -> float:
    """The same, but also ignoring which way the pose faces on the floor.

    A turn about the world z axis is a symmetry of the environment -- the floor
    and gravity are both unchanged by it -- so two poses that differ only by one
    are the same tilt on a different compass bearing. All that survives both
    quotients is where gravity points in the mesh frame, so that is what is
    compared here. This is the stricter of the two tests and the one that decides
    whether ten poses are ten experiments.
    """
    ga, gb = Ra.T @ np.array([0.0, -1, 0]), Rb.T @ np.array([0.0, -1, 0])
    c = np.clip(np.einsum("nij,j->ni", group, ga) @ gb, -1, 1)
    return float(np.degrees(np.arccos(c)).min())


def flatten(T0: np.ndarray, half: float) -> np.ndarray:
    """Snap a settled placement to the exact face-down rest it is a sample of.

    The drop lands within a thousandth of a degree of flat, which is as close as
    a physics sample gets and not close enough here: on a cube a tilt that small
    still decides which single corner is the lowest, so the footprint comes back
    as a point instead of a square and there is no edge to tip about. The yaw of
    the sampled pose is kept -- it is the only part of it that is not congruent
    to every other placement anyway.
    """
    R = np.asarray(T0)[:3, :3]
    g = R.T @ np.array([0.0, -1, 0])
    down = np.zeros(3)
    down[int(np.argmax(np.abs(g)))] = np.sign(g[int(np.argmax(np.abs(g)))])
    A = trimesh.geometry.align_vectors(down, np.array([0.0, -1, 0]))[:3, :3]
    M = R @ A.T                                        # what is left is a turn about y
    yaw = float(np.arctan2(M[2, 0] - M[0, 2], M[0, 0] + M[2, 2]))
    c, s = np.cos(yaw), np.sin(yaw)
    return se3(np.array([[c, 0., -s], [0., 1., 0.], [s, 0., c]]) @ A,
               np.array([0.0, half, 0]))


def footprint(V: np.ndarray, T0: np.ndarray) -> np.ndarray:
    """The four corners the cube stands on, counter-clockwise."""
    W = V @ T0[:3, :3].T + T0[:3, 3]
    xy = COORD.floor(W[W[:, 1] < W[:, 1].min() + 1e-9])
    a = np.arctan2(*(xy - xy.mean(axis=0)).T[::-1])
    return xy[np.argsort(a)]


def pivots(poly: np.ndarray, j: int) -> tuple:
    """Side `j` of the footprint and corner `j` of it, as ground lines.

    Both are supporting lines of the square, which is what a tip needs: the side
    lies along one, and at a corner the line perpendicular to the outward
    diagonal touches the square at that corner alone.
    """
    a, b = poly[j], poly[(j + 1) % len(poly)]
    edge = (COORD.lift_floor((a+b)/2), -COORD.lift_floor(b-a))
    out = poly[j] - poly.mean(axis=0)
    corner = (COORD.lift_floor(poly[j]), -COORD.lift_floor([-out[1],out[0]]))
    return edge, corner, float(np.linalg.norm(b - a))


def build_mesh(d) -> trimesh.Trimesh:
    L = REFERENCE_VOLUME_M3 ** (1 / 3)
    mesh = trimesh.creation.box(extents=(L, L, L))     # 8 vertices, 12 triangles
    mesh.density = DENSITY
    d.mkdir(parents=True, exist_ok=True)
    mesh.export(d / "mesh.stl")
    write_json(d / "meta.json", {
        "name": NAME,
        "source": "slides/tools/make_cuboid.py: cube of A1-f's volume, edges left sharp",
        "units": "m",
        "n_vertices": int(len(mesh.vertices)),
        "n_faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "was_repaired": False,
        "extents_m": [float(v) for v in mesh.extents],
        "bounds_m": [[float(v) for v in row] for row in mesh.bounds],
        "volume_m3": float(mesh.volume),
        "density_kg_m3": DENSITY,
        "mass_kg": float(mesh.mass),
        "com_mesh_frame": [float(v) for v in mesh.center_mass],
        "inertia_com": [[float(v) for v in row] for row in mesh.moment_inertia],
        "convex": bool(mesh.is_convex),
        "hull_volume_ratio": float(mesh.volume / mesh.convex_hull.volume),
    })
    return mesh


def build_tips(d, mesh: trimesh.Trimesh) -> list:
    V = np.asarray(mesh.vertices)
    com = np.asarray(read_json(d / "meta.json")["com_mesh_frame"])
    placements = read_json(d / "poses.json")["poses"]
    records = []
    # the yaw and the choice of side/corner only change where the pose points on
    # the floor; they are varied so the ten renders do not all look alike, but
    # the angle is what makes them ten different problems
    for k, (kind, deg) in enumerate([("edge", g) for g in EDGE_TIPS]
                                    + [("point", g) for g in CORNER_TIPS]):
        pl = placements[k % len(placements)]
        T0 = flatten(np.asarray(pl["T_world_mesh"]), float(mesh.extents[1]) / 2)
        poly = footprint(V, T0)
        edge, corner, edge_len = pivots(poly, k % 4)
        q, axis = edge if kind == "edge" else corner
        lim = tip_limits(V, T0[:3, :3] @ com + T0[:3, 3], T0, q, axis)
        ex = make_example(V, com, T0, q, axis, kind, fraction=deg / lim["limit_deg"])
        ex.update({"placement": pl["index"], "object": NAME,
                   "probability": pl["probability"],
                   "half_length": edge_len / 2 if kind == "edge" else 0.0})
        if kind == "edge":
            ex["edge_length_m"] = edge_len
        records.append(ex)
    write_json(d / "tips" / "tips.json",
               {"object": NAME, "tip_fraction": None, "examples": records})
    return records


def sheet(d, records: list, px: int) -> None:
    """The ten poses, from the side as well, where touching the floor is visible."""
    label_w, head_h = int(px * 0.62), int(px * 0.18)
    img = Image.new("RGB", (label_w + 2 * px, head_h + px * len(records)), "white")
    dr = ImageDraw.Draw(img)
    f, fs = _font(int(px * 0.075)), _font(int(px * 0.058))
    for i, h in enumerate(["tipped", "the same, seen along the pivot"]):
        dr.text((label_w + i * px + 8, int(head_h * 0.28)), h, fill=(20, 20, 20), font=fs)
    for r, ex in enumerate(records):
        y = head_h + r * px
        for c, view in enumerate(("iso", "side")):
            img.paste(render(NAME, ex, px, view), (label_w + c * px, y))
        dr.text((10, y + px // 2 - int(px * 0.15)),
                f"pose {r}\n{ex['pivot']} pivot\n\n{ex['tip_deg']:.0f} deg\n"
                f"of {ex['limit_deg']:.0f} max\ngap {ex['gap_max_m'] * 1000:.0f} mm",
                fill=(20, 20, 20), font=f)
        dr.line([(0, y), (img.size[0], y)], fill=(220, 220, 220))
    img.save(d / "tips" / "sheet.png")
    print(f"{d / 'tips' / 'sheet.png'}  {img.size[0]}x{img.size[1]}\n")


def report(records: list) -> None:
    group = cube_rotations()
    assert len(group) == 24, len(group)
    R = [np.asarray(r["T_world_mesh"])[:3, :3] for r in records]
    pairs = list(itertools.combinations(range(len(R)), 2))
    full = min(apart_deg(R[i], R[j], group) for i, j in pairs)
    tilt = min(tilt_apart_deg(R[i], R[j], group) for i, j in pairs)
    for i, r in enumerate(records):
        print(f"pose {i:2d}: {r['pivot']:5s} pivot, tipped {r['tip_deg']:5.1f} deg of "
              f"{r['limit_deg']:5.1f} ({r['limited_by']}), lowest point "
              f"{r['ground_clearance_min_m'] * 1e6:+7.3f} um, "
              f"gap {r['gap_max_m'] * 1000:5.1f} mm", flush=True)
    print(f"\nclosest pair, modulo the 24 rotations of the cube : {full:5.2f} deg")
    print(f"closest pair, modulo those and the world yaw too   : {tilt:5.2f} deg")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tips-only", action="store_true",
                    help="rebuild tips.json, once poses.json and scene.xml exist")
    ap.add_argument("--size", type=int, default=320)
    args = ap.parse_args()

    d = obj_path(NAME)
    mesh = build_mesh(d) if not args.tips_only else trimesh.load(d / "mesh.stl",
                                                                 force="mesh")
    meta = read_json(d / "meta.json")
    print(f"{NAME}: edge {meta['extents_m'][0] * 1000:.3f} mm, "
          f"{meta['n_faces']} faces, {mesh.area * 1e6:.0f} mm2, "
          f"{meta['volume_m3'] * 1e9:.0f} mm3, {meta['mass_kg']:.6f} kg\n")
    if (d / "poses.json").exists():
        records = build_tips(d, mesh)
        sheet(d, records, args.size)
        report(records)


if __name__ == "__main__":
    main()
