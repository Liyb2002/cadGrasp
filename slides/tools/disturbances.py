"""Step 1: the set of disturbances a support design has to defeat.

At a target pose the workpiece is held out of equilibrium. Anything in the cell
can push on it: a tool, a nozzle, a hose, an operator. A rigid push on a surface
acts along the inward normal at the point it touches, so the complete set of
disturbances is one push per point of the workpiece surface -- every point except
the patch already resting on the ground, which cannot be pushed on.

This samples that set and reports, for each push, the moment it makes about the
ground contact. The sign against the direction gravity is already pulling splits
the set in two: pushes that add to the fall the workpiece is already trying to
make, and pushes that hold it up. Only the first group needs a support; the
second is doing our job for us.

Magnitudes are not modelled. A push is a direction, and a support answers it if
it can push back at all.

    python slides/tools/disturbances.py A1-f
    python slides/tools/disturbances.py A1-f --samples 400
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
from mujoco import Renderer
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

from common import mat_to_quat_wxyz, obj_path, read_json, write_json

CONTACT_EPS = 1.5e-3   # m; surface this close to the floor is the ground patch
NEUTRAL = 1e-4         # N*m per unit force, below which a push barely turns it
ARROW_FRAC = 0.20      # arrow length, as a fraction of the workpiece size
ARROW_WIDTH = 1.6e-3
ARROWS_DRAWN = 70      # how many of the sampled pushes to draw, for legibility

ADDS, HOLDS, NEUTRALS = "adds to the fall", "holds it up", "barely turns it"
COLOR = {ADDS: (0.85, 0.20, 0.15, 1.0),
         HOLDS: (0.20, 0.45, 0.85, 1.0),
         NEUTRALS: (0.60, 0.60, 0.62, 1.0)}

SCENE = """<mujoco>
  <compiler meshdir="." angle="radian"/>
  <visual>
    <global offwidth="1400" offheight="1400"/>
    <quality shadowsize="4096" offsamples="8"/>
    <map znear="0.005" zfar="30"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.45 0.45 0.45" specular="0.05 0.05 0.05"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="1 1 1"
             width="256" height="256"/>
    <material name="floor" rgba="0.97 0.97 0.97 1"/>
    <material name="adds" rgba="0.86 0.34 0.28 1" specular="0.15" shininess="0.2"/>
    <material name="holds" rgba="0.36 0.55 0.82 1" specular="0.15" shininess="0.2"/>
    <material name="ground_patch" rgba="0.55 0.55 0.57 1" specular="0.1"/>
{assets}
  </asset>
  <worldbody>
    <light pos="0.4 1.2 -0.5" dir="-0.3 -1 0.4" directional="true" castshadow="true"
           diffuse="0.55 0.55 0.55"/>
    <geom type="plane" quat="0.7071067811865476 -0.7071067811865476 0 0" size="2 2 0.05" material="floor"/>
    <body pos="{pos}" quat="{quat}">
{geoms}
    </body>
  </worldbody>
</mujoco>
"""


# ------------------------------------------------------------------ sample ---

def sample_surface(mesh: trimesh.Trimesh, n: int, seed: int = 0):
    """Area-uniform points on the surface, with the outward normal at each."""
    rng = np.random.default_rng(seed)
    area = mesh.area_faces
    faces = rng.choice(len(area), size=n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    tri = mesh.triangles[faces]
    pts = tri[:, 0] + u[:, None] * (tri[:, 1] - tri[:, 0]) + v[:, None] * (tri[:, 2] - tri[:, 0])
    return pts, mesh.face_normals[faces]


def disturbances(mesh: trimesh.Trimesh, T: np.ndarray, pivot: np.ndarray,
                 com_local: np.ndarray, n: int, seed: int = 0) -> dict:
    """Every push that can act on the workpiece at pose T, and what it turns.

    A push at a surface point acts along the inward normal there. Its moment is
    taken about the ground contact, since that is the only thing the workpiece is
    resting on: a push with no moment about it cannot turn the workpiece at all.
    """
    R, t = T[:3, :3], T[:3, 3]
    pts, nrm = sample_surface(mesh, n, seed)
    p = pts @ R.T + t
    n_out = nrm @ R.T
    keep = p[:, 2] > CONTACT_EPS          # the ground patch cannot be pushed on
    p, n_out = p[keep], n_out[keep]
    push = -n_out                         # a rigid contact can only push inwards

    com = R @ com_local + t
    # The direction gravity is already turning the workpiece about the contact.
    fall = np.cross(com - pivot, np.array([0.0, 0.0, -1.0]))
    fall_axis = fall / (np.linalg.norm(fall) + 1e-15)

    moment = np.cross(p - pivot, push)
    along = moment @ fall_axis                       # + means it helps it fall
    across = np.linalg.norm(moment - along[:, None] * fall_axis, axis=1)

    kind = np.where(along > NEUTRAL, ADDS,
                    np.where(along < -NEUTRAL, HOLDS, NEUTRALS))
    return {"points": p, "push": push, "moment": moment, "along": along,
            "across": across, "kind": kind, "fall_axis": fall_axis, "com": com}


# ------------------------------------------------------------------ render ---

def paint_surface(mesh: trimesh.Trimesh, T: np.ndarray, pivot: np.ndarray,
                  fall_axis: np.ndarray, out_dir: Path, tag: str) -> list[tuple[str, str]]:
    """Split the surface into the three kinds and write one mesh per kind.

    Painting the surface rather than drawing an arrow per sample makes the
    boundary visible: it is the set of points whose push line passes through the
    pivot, and it is what separates the pushes a support has to answer from the
    ones already working in our favour.
    """
    R, t = T[:3, :3], T[:3, 3]
    centre = mesh.triangles_center @ R.T + t
    push = -(mesh.face_normals @ R.T)
    along = np.cross(centre - pivot, push) @ fall_axis

    on_ground = centre[:, 2] <= CONTACT_EPS
    groups = [("ground_patch", on_ground),
              ("adds", ~on_ground & (along > NEUTRAL)),
              ("holds", ~on_ground & (along <= NEUTRAL))]

    out_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for material, sel in groups:
        if not sel.any():
            continue
        patch = mesh.submesh([np.flatnonzero(sel)], append=True)
        # A patch that happens to be one flat face has no convex hull, which the
        # MuJoCo compiler refuses. Break the coplanarity by an invisible amount.
        v = patch.vertices - patch.vertices.mean(axis=0)
        if np.linalg.matrix_rank(v, tol=1e-9 * max(patch.scale, 1e-9)) < 3:
            n = patch.face_normals[0]
            patch.vertices[::2] += 2e-5 * n
        fname = f"{tag}_{material}.obj"
        patch.export(out_dir / fname)
        parts.append((f"disturbances/paint/{fname}", material))
    return parts


def render(name: str, T: np.ndarray, d: dict, parts, px: int, azimuth: float,
           arrow_len: float, arrows: np.ndarray | None) -> Image.Image:
    obj = obj_path(name)
    assets = "\n".join(f'    <mesh name="p{i}" file="{f}"/>' for i, (f, _) in enumerate(parts))
    geoms = "\n".join(f'      <geom type="mesh" mesh="p{i}" material="{m}"/>'
                      for i, (_, m) in enumerate(parts))
    xml = SCENE.format(assets=assets, geoms=geoms,
                       pos=" ".join(f"{v:.9g}" for v in T[:3, 3]),
                       quat=" ".join(f"{v:.9g}" for v in mat_to_quat_wxyz(T[:3, :3])))
    tmp = obj / ".dist_tmp.xml"
    tmp.write_text(xml)
    try:
        model = mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    V = np.asarray(trimesh.load(obj / "mesh.stl", force="mesh").vertices)
    W = V @ T[:3, :3].T + T[:3, 3]
    lo, hi = W.min(axis=0), W.max(axis=0)

    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation = azimuth, -18.0
    cam.lookat[:] = (lo + hi) / 2
    cam.distance = 1.55 * float(np.linalg.norm(hi - lo)) / 2 / np.tan(
        np.deg2rad(model.vis.global_.fovy / 2))

    with Renderer(model, px, px, max_geom=20000) as r:
        r.update_scene(data, camera=cam)
        scn = r.scene
        if arrows is not None:
            for i in arrows:
                if scn.ngeom >= scn.maxgeom:
                    break
                p, push, kind = d["points"][i], d["push"][i], d["kind"][i]
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3),
                                    np.zeros(3), np.zeros(9),
                                    np.array(COLOR[kind], np.float32))
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, ARROW_WIDTH,
                                     p - push * arrow_len, p)
                scn.ngeom += 1
        return Image.fromarray(r.render())


def _font(size: int):
    try:
        from matplotlib import font_manager
        return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
    except Exception:
        return ImageFont.load_default()


# -------------------------------------------------------------------- main ---

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("object")
    ap.add_argument("--samples", type=int, default=260)
    ap.add_argument("--size", type=int, default=380)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    name = args.object
    d = obj_path(name)
    mesh = trimesh.load(d / "mesh.stl", force="mesh")
    com_local = np.asarray(read_json(d / "meta.json")["com_mesh_frame"])
    examples = read_json(d / "tips" / "tips.json")["examples"]
    arrow_len = ARROW_FRAC * float(np.linalg.norm(mesh.extents))

    out = d / "disturbances"
    (out / "renders").mkdir(parents=True, exist_ok=True)
    rows, records = [], []

    for ex in examples:
        T = np.asarray(ex["T_world_mesh"])
        pivot = np.asarray(ex["point"])
        dist = disturbances(mesh, T, pivot, com_local, args.samples, args.seed)

        n_add = int((dist["kind"] == ADDS).sum())
        n_hold = int((dist["kind"] == HOLDS).sum())
        n_neu = int((dist["kind"] == NEUTRALS).sum())
        tag = f"p{ex['placement']}_{ex['pivot']}"
        records.append({"placement": ex["placement"], "pivot": ex["pivot"],
                        "tip_deg": ex["tip_deg"], "n_sampled": int(len(dist["kind"])),
                        "n_adds_to_the_fall": n_add, "n_holds_it_up": n_hold,
                        "n_barely_turns_it": n_neu,
                        "fall_axis": [float(v) for v in dist["fall_axis"]],
                        "samples": [
                            {"p": [float(v) for v in p], "push": [float(v) for v in q],
                             "moment_along_fall": float(a), "moment_across": float(c),
                             "kind": str(k)}
                            for p, q, a, c, k in zip(dist["points"], dist["push"],
                                                     dist["along"], dist["across"],
                                                     dist["kind"])]})

        parts = paint_surface(mesh, T, pivot, dist["fall_axis"], out / "paint", tag)
        rng = np.random.default_rng(args.seed)
        adds_idx = np.flatnonzero(dist["kind"] == ADDS)
        drawn = rng.choice(adds_idx, size=min(ARROWS_DRAWN, len(adds_idx)), replace=False)
        imgs = [render(name, T, dist, parts, args.size, 135.0, arrow_len, None),
                render(name, T, dist, parts, args.size, 135.0, arrow_len, drawn),
                render(name, T, dist, parts, args.size, 315.0, arrow_len, drawn)]
        for i, im in enumerate(imgs):
            im.save(out / "renders" / f"{tag}_{i}.png")
        rows.append((ex, dist, imgs, (n_add, n_hold, n_neu)))
        print(f"placement {ex['placement']} {ex['pivot']:5s} pivot, tipped "
              f"{ex['tip_deg']:4.1f} deg: {len(dist['kind']):4d} pushes -> "
              f"{n_add:4d} add to the fall, {n_hold:4d} hold it up, "
              f"{n_neu:3d} barely turn it", flush=True)

    write_json(out / "disturbances.json",
               {"object": name, "samples_requested": args.samples,
                "seed": args.seed, "examples": records})

    px = args.size
    label_w, head_h = int(px * 0.85), int(px * 0.22)
    sheet = Image.new("RGB", (label_w + 3 * px, head_h + px * len(rows)), "white")
    dr = ImageDraw.Draw(sheet)
    f, fs = _font(int(px * 0.050)), _font(int(px * 0.052))
    for i, h in enumerate(["red: pushing here adds to the fall\nblue: pushing here holds it up",
                           "the pushes that add to the fall",
                           "the same, from the other side"]):
        dr.text((label_w + i * px + 8, int(head_h * 0.3)), h, fill=(20, 20, 20), font=f)
    for r, (ex, dist, imgs, (n_add, n_hold, n_neu)) in enumerate(rows):
        y = head_h + r * px
        for c, im in enumerate(imgs):
            sheet.paste(im, (label_w + c * px, y))
        dr.text((10, y + int(px * 0.12)),
                f"placement {ex['placement']}, {ex['pivot']} pivot\n"
                f"tipped {ex['tip_deg']:.0f} deg\n\n"
                f"{n_add} add to the fall\n{n_hold} hold it up\n{n_neu} barely turn it",
                fill=(40, 40, 40), font=fs)
        dr.line([(0, y), (sheet.size[0], y)], fill=(215, 215, 215))
    sheet.save(out / "sheet.png")
    print(f"\n{out / 'sheet.png'}  {sheet.size[0]}x{sheet.size[1]}")


if __name__ == "__main__":
    main()
