"""For each pose: a work region, the sphere of forces it admits, and the supports.

The pipeline, straight through:

    a pose, and a work region drawn on it
    every point of the region can be pushed along its inward normal
        -> the set of those directions is a patch on the sphere of forces
    reverse the patch: those are the directions a support has to push in
    look for each one on the workpiece, off the work region
        -> not there? say so, and say which of the two reasons it is
        -> there? keep the fewest positions that cover the whole patch

A direction can be missing for two quite different reasons, and they mean
different things. Either the workpiece has no face pointing that way at all,
which is a fact about the shape, or it has one but the work region is sitting on
it, which is a fact about this particular job and could be fixed by spraying it
differently or tipping it further.

    python slides/tools/opposing_supports.py A1-f
    python slides/tools/opposing_supports.py A1-f --tolerance 15 --poses 0 1 2
"""
from __future__ import annotations

import argparse
import io

import matplotlib
import mujoco
from yup_render import Renderer as YUpRenderer
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection    # noqa: E402

from common import mat_to_quat_wxyz, obj_path, read_json, write_json   # noqa: E402
from disturbances import _font                             # noqa: E402
from supports import paint_region, region_mask             # noqa: E402
from work_regions import SCENE, gun_directions             # noqa: E402

INK, MUTED, PAPER = "#1b1b1a", "#6b6b66", "#ffffff"
PROCESS = "#B45309"
COVERED, MISSING_SHAPE, MISSING_REGION = "#2F855A", "#8C1D18", "#2563EB"
EMPTY = "#e6e6e0"
SUPPORT_RGBA = (0.18, 0.51, 0.35, 1.0)
ARROW_FRAC, ARROW_WIDTH = 0.42, 3.4e-3


def tiling(subdivisions: int = 4):
    ico = trimesh.creation.icosphere(subdivisions=subdivisions)
    c = np.array(ico.triangles_center, dtype=float)
    return ico, c / np.linalg.norm(c, axis=1, keepdims=True)


def work_region(mesh, T, rng):
    """A spray job: a random number of passes, from random directions above."""
    n = int(rng.integers(1, 7))
    dirs = gun_directions(rng, n)
    inside = region_mask(mesh, T, dirs, mesh.triangles_center, mesh.face_normals)
    return n, dirs, inside


def solve(mesh, T, inside, tiles, tree, tol_deg):
    """Which reversed directions a support can supply, and the fewest places to do it."""
    R, t = T[:3, :3], T[:3, 3]
    push = -(mesh.face_normals @ R.T)
    push /= np.linalg.norm(push, axis=1, keepdims=True)
    centres = mesh.triangles_center @ R.T + t
    hit = tree.query(push)[1]

    process = np.unique(hit[inside])                     # the sphere of forces
    needed = np.unique(tree.query(-tiles[process])[1])   # reversed
    off = np.flatnonzero(~inside)

    # one representative place on the workpiece per available direction: the
    # biggest face pointing that way, so a support has something to sit against
    place, direction = {}, {}
    order = off[np.argsort(-mesh.area_faces[off])]
    for f in order:
        place.setdefault(hit[f], centres[f])
        direction.setdefault(hit[f], push[f])
    avail = np.array(sorted(place), dtype=int)

    cos_tol = np.cos(np.radians(tol_deg))
    if len(avail):
        # cover[j] = the needed directions that the j-th available place answers
        gram = tiles[avail] @ tiles[needed].T >= cos_tol
    else:
        gram = np.zeros((0, len(needed)), bool)
    answered = gram.any(axis=0)

    # a direction can go missing two ways, and they mean different things
    on_region = np.isin(needed, process)
    reason = np.where(answered, "covered",
                      np.where(on_region, "the work region is on it",
                               "the workpiece has no such face"))

    # fewest places, greedily: an upper bound, the exact one is set cover
    chosen, left = [], answered.copy()
    while left.any():
        gain = (gram & left).sum(axis=1)
        j = int(np.argmax(gain))
        if gain[j] == 0:
            break
        chosen.append(int(avail[j]))
        left &= ~gram[j]
    return {"process": process, "needed": needed, "answered": answered,
            "reason": reason, "chosen": chosen,
            "contacts": [(place[c], direction[c]) for c in chosen],
            "avail": avail}


# ------------------------------------------------------------------ drawing ---

def globe(ax, ico, colours, title):
    v = ico.vertices / np.linalg.norm(ico.vertices, axis=1, keepdims=True)
    ax.add_collection3d(Poly3DCollection(v[ico.faces], facecolors=colours,
                                         edgecolors="none", zsort="average"))
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(1.002 * np.cos(th), 1.002 * np.sin(th), np.zeros_like(th), color=MUTED,
            lw=0.7, alpha=0.55)
    ax.plot([0, 0], [0, 0], [1.0, 1.22], color=INK, lw=0.9)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1.35)
    ax.set_box_aspect((1, 1, 1.18))
    ax.view_init(elev=14, azim=-62)
    ax.set_axis_off()
    ax.set_title(title, color=INK, fontsize=9.5, pad=-6)


def sphere_png(ico, colour_sets, titles, px):
    fig = plt.figure(figsize=(3.0 * len(colour_sets), 3.1), dpi=px / 3.1,
                     facecolor=PAPER)
    for i, (cols, title) in enumerate(zip(colour_sets, titles)):
        globe(fig.add_subplot(1, len(colour_sets), i + 1, projection="3d",
                              facecolor=PAPER), ico, cols, title)
    fig.subplots_adjust(0, 0, 1, 1, 0, 0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=PAPER)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def render_part(name, T, parts, contacts, px, azimuth):
    obj = obj_path(name)
    xml = SCENE.format(
        assets="\n".join(f'    <mesh name="p{i}" file="{f}"/>'
                         for i, (f, _) in enumerate(parts)),
        geoms="\n".join(f'      <geom type="mesh" mesh="p{i}" material="{m}"/>'
                        for i, (_, m) in enumerate(parts)),
        pos=" ".join(f"{v:.9g}" for v in T[:3, 3]),
        quat=" ".join(f"{v:.9g}" for v in mat_to_quat_wxyz(T[:3, :3])))
    tmp = obj / ".opp_tmp.xml"
    tmp.write_text(xml)
    try:
        model = mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    mesh = trimesh.load(obj / "mesh.stl", force="mesh")
    W = np.asarray(mesh.vertices) @ T[:3, :3].T + T[:3, 3]
    lo, hi = W.min(axis=0), W.max(axis=0)
    size = float(np.linalg.norm(hi - lo))
    L = ARROW_FRAC * float(np.linalg.norm(mesh.extents))

    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation = azimuth, -15.0
    cam.lookat[:] = (lo + hi) / 2
    cam.distance = 1.75 * size / 2 / np.tan(np.deg2rad(model.vis.global_.fovy / 2))

    with YUpRenderer(model, px, px, max_geom=4000) as r:
        r.update_scene(data, camera=cam)
        scn = r.scene
        col = np.array(SUPPORT_RGBA, np.float32)
        for p, u in contacts:
            if scn.ngeom + 2 >= scn.maxgeom:
                break
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3),
                                np.zeros(3), np.zeros(9), col)
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, ARROW_WIDTH,
                                 np.asarray(p) - np.asarray(u) * L, np.asarray(p))
            scn.ngeom += 1
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.full(3, ARROW_WIDTH * 1.5), np.asarray(p),
                                np.eye(3).ravel(), col)
            scn.ngeom += 1
        return Image.fromarray(r.render())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("object")
    ap.add_argument("--poses", type=int, nargs="+", default=None)
    ap.add_argument("--tolerance", type=float, default=10.0,
                    help="degrees a support direction may differ from the one wanted")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subdivisions", type=int, default=4)
    ap.add_argument("--size", type=int, default=380)
    args = ap.parse_args()

    d = obj_path(args.object)
    mesh = trimesh.load(d / "mesh.stl", force="mesh")
    examples = read_json(d / "tips" / "tips.json")["examples"]
    poses = args.poses if args.poses is not None else range(len(examples))
    ico, tiles = tiling(args.subdivisions)
    tree = cKDTree(tiles)

    px = args.size
    rows, records = [], []
    for pose in poses:
        ex = examples[pose]
        T = np.asarray(ex["T_world_mesh"])
        rng = np.random.default_rng(args.seed + 1000 * pose)
        n_passes, dirs, inside = work_region(mesh, T, rng)
        res = solve(mesh, T, inside, tiles, tree, args.tolerance)

        base = [matplotlib.colors.to_rgba(EMPTY, 0.5)] * len(tiles)
        left = np.array(base)
        left[res["process"]] = matplotlib.colors.to_rgba(PROCESS, 0.95)
        right = np.array(base)
        for tag, colour in (("covered", COVERED),
                            ("the work region is on it", MISSING_REGION),
                            ("the workpiece has no such face", MISSING_SHAPE)):
            sel = res["needed"][res["reason"] == tag]
            if len(sel):
                right[sel] = matplotlib.colors.to_rgba(colour, 0.95)

        parts, area = paint_region(mesh, T, dirs, d / "work_regions" / "paint",
                                   f"opp_p{pose}")
        part_img = render_part(args.object, T, parts, res["contacts"], px, 135.0)
        part_img2 = render_part(args.object, T, parts, res["contacts"], px, 315.0)
        globes = sphere_png(ico, [left, right],
                            ["the forces the process can apply",
                             "the reversed forces: can a support supply them?"],
                            px)
        rows.append((pose, n_passes, area, res, part_img, part_img2, globes))

        n_need = len(res["needed"])
        ok = int(res["answered"].sum())
        by_region = int((res["reason"] == "the work region is on it").sum())
        by_shape = int((res["reason"] == "the workpiece has no such face").sum())
        records.append({"pose": pose, "pivot": ex["pivot"], "tip_deg": ex["tip_deg"],
                        "n_passes": n_passes, "region_area_fraction": area,
                        "n_directions_needed": n_need, "n_covered": ok,
                        "n_missing_work_region_on_it": by_region,
                        "n_missing_no_such_face": by_shape,
                        "n_supports": len(res["chosen"]),
                        "found": bool(ok == n_need),
                        "contacts": [{"p": [float(x) for x in p],
                                      "push": [float(x) for x in u]}
                                     for p, u in res["contacts"]]})
        verdict = (f"{len(res['chosen'])} supports"
                   if ok == n_need else
                   f"NOT FOUND ({n_need - ok} of {n_need} directions)")
        print(f"pose {pose:2d} ({ex['pivot']:5s}, tipped {ex['tip_deg']:4.1f} deg, "
              f"{n_passes} pass, region {100 * area:4.1f}%): "
              f"{n_need:4d} directions wanted, {ok:4d} answered "
              f"[{by_shape:4d} no such face, {by_region:3d} under the work region] "
              f"-> {verdict}", flush=True)

    label = int(px * 1.15)
    gw = rows[0][6].size[0]
    sheet = Image.new("RGB", (label + 2 * px + gw, int(px * 0.42) + px * len(rows)),
                      "white")
    dr = ImageDraw.Draw(sheet)
    f, fs = _font(int(px * 0.062)), _font(int(px * 0.052))
    dr.text((14, 12),
            f"{args.object} -- one random spray job per pose, "
            f"support directions matched to {args.tolerance:.0f} deg\n"
            "green arrows = the supports found.   on the right sphere: "
            "green = a support can supply it,\n"
            "blue = the work region is sitting on the only face that could, "
            "dark red = the workpiece has no such face.",
            fill=(20, 20, 20), font=f)
    for r, (pose, n_passes, area, res, im1, im2, gl) in enumerate(rows):
        y = int(px * 0.42) + r * px
        sheet.paste(im1, (label, y))
        sheet.paste(im2, (label + px, y))
        sheet.paste(gl, (label + 2 * px, y + (px - gl.size[1]) // 2))
        rec = records[r]
        dr.text((12, y + int(px * 0.10)),
                f"pose {pose}  ({rec['pivot']}, {rec['tip_deg']:.0f}deg)\n"
                f"{n_passes} spray pass(es)\n"
                f"work region {100 * area:.0f}% of the surface\n\n"
                f"{rec['n_directions_needed']} directions wanted\n"
                f"{rec['n_covered']} of them answered\n\n"
                + (f"{rec['n_supports']} support points"
                   if rec["found"] else
                   f"NOT FOUND\n{rec['n_missing_no_such_face']} no such face\n"
                   f"{rec['n_missing_work_region_on_it']} under the work region"),
                fill=(40, 40, 40), font=fs)
        dr.line([(0, y), (sheet.size[0], y)], fill=(215, 215, 215))

    out = d / "opposing" / f"sheet_tol{args.tolerance:g}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    write_json(d / "opposing" / f"opposing_tol{args.tolerance:g}.json",
               {"object": args.object, "tolerance_deg": args.tolerance,
                "seed": args.seed, "poses": records})
    print(f"\n{out}  {sheet.size[0]}x{sheet.size[1]}")


if __name__ == "__main__":
    main()
