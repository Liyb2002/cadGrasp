"""Start from a support that grips everything, then take away all it can spare.

The other way round -- looking for a contact that opposes each push one at a time
-- asks for something the workpiece usually cannot give, and it is asking for too
much anyway: contacts add up, so what has to be opposed is the total, not each
push separately. Starting from the largest possible support avoids both mistakes.

    grip the whole surface outside the work region
    can that hold the workpiece against every push the process can make?
        no  -> nothing weaker can either. the part has no support at this pose.
        yes -> hand back area, patch by patch, for as long as it still holds

What survives is a support that cannot give up any of its remaining patches, and
its area is the answer to the question the point models could not reach: how much
contact does holding this actually take. Removal is greedy, so this is one local
answer, not the smallest one -- but unlike a sparsified point set it is a real
support that has been checked at every step.

    python slides/tools/shrink_support.py A1-f
    python slides/tools/shrink_support.py A1-f --poses 0 --k 1 --patches 160
"""
from __future__ import annotations

import argparse

import mujoco
from yup_render import Renderer as YUpRenderer
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.cluster.vq import kmeans2

from common import mat_to_quat_wxyz, obj_path, read_json, write_json
from disturbances import _font, sample_surface
from opposition import opposable, wrenches
from supports import CONTACT_EPS, region_mask
from work_regions import SCENE, gun_directions

# the two that must never be confused get opposite hues, not two greens
KEPT = "0.85 0.30 0.05 1"       # the contact that has to stay
GIVEN_UP = "0.87 0.86 0.82 1"   # surface a support could have used but need not
WORK = "0.42 0.74 0.53 1"       # the work region, which must stay clear


def contact_wrenches(mesh, T, inside, n_cand, n_dist, seed):
    """Places a support may touch, and the pushes it has to answer."""
    R, t = T[:3, :3], T[:3, 3]
    scale = float(np.linalg.norm(mesh.extents))
    ref = mesh.center_mass @ R.T + t

    pts, nrm, face = sample_surface_faces(mesh, (n_cand + n_dist) * 4, seed)
    p = pts @ R.T + t
    push = -(nrm @ R.T)
    off = ~inside[face]

    cand_p, cand_u, cand_f = p[off][:n_cand], push[off][:n_cand], face[off][:n_cand]
    dist_p, dist_u = p[~off][:n_dist], push[~off][:n_dist]

    # the workpiece rests on a line once it is tipped, so area sampling never
    # lands there; that contact is free and has to be put in by hand
    V = mesh.vertices @ R.T + t
    touching = V[V[:, 1] <= CONTACT_EPS]
    if len(touching):
        step = max(1, len(touching) // 24)
        floor = touching[::step]
        cand_p = np.vstack([floor, cand_p])
        cand_u = np.vstack([np.tile([0.0, 0.0, 1.0], (len(floor), 1)), cand_u])
        cand_f = np.concatenate([np.full(len(floor), -1), cand_f])

    return {"W": wrenches(cand_p, cand_u, ref, scale),
            "D": wrenches(dist_p, dist_u, ref, scale),
            "p": cand_p, "u": cand_u, "face": cand_f,
            "gravity": np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0])}


def sample_surface_faces(mesh, n, seed):
    """`sample_surface`, but it also says which face each point came from."""
    rng = np.random.default_rng(seed)
    area = mesh.area_faces
    faces = rng.choice(len(area), size=n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    tri = mesh.triangles[faces]
    pts = tri[:, 0] + u[:, None] * (tri[:, 1] - tri[:, 0]) + v[:, None] * (tri[:, 2] - tri[:, 0])
    return pts, mesh.face_normals[faces], faces


GRAZE = 0.08           # a push has to go into the surface, not skim along it


def angled_pushes(mesh, T, inside, n_points, n_dirs, seed):
    """Every push a tool could make, not just the ones square to the surface.

    A cut can come in at an angle, so the direction is not the inward normal but
    anything that goes into the surface. What rules the rest out is not the
    normal, it is getting there: the tool arrives along a straight line out of
    the air, so a direction counts only if that line reaches the point without
    running through the workpiece first. Self-occluded angles drop out on their
    own, and so does anything that would have to come from inside the part.
    """
    R, t = T[:3, :3], T[:3, 3]
    rng = np.random.default_rng(seed)
    # sample the region itself rather than the whole surface and reject: a small
    # region gets nothing back from rejection sampling
    where = np.flatnonzero(inside)
    if not len(where):
        return np.zeros((0, 3)), np.zeros((0, 3))
    w = mesh.area_faces[where]
    face = rng.choice(where, size=n_points, p=w / w.sum())
    a, b = rng.random(n_points), rng.random(n_points)
    flip = a + b > 1
    a[flip], b[flip] = 1 - a[flip], 1 - b[flip]
    tri = mesh.triangles[face]
    pts = tri[:, 0] + a[:, None] * (tri[:, 1] - tri[:, 0]) + b[:, None] * (tri[:, 2] - tri[:, 0])
    nrm = mesh.face_normals[face]
    sel = np.arange(n_points)
    origins, escape = [], []
    for i in sel:
        v = rng.normal(size=(n_dirs * 3, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        v = v[v @ nrm[i] > GRAZE][:n_dirs]
        origins.append(np.tile(pts[i] + nrm[i] * 1e-5, (len(v), 1)))
        escape.append(v)
    O, V = np.vstack(origins), np.vstack(escape)
    clear = ~mesh.ray.intersects_any(ray_origins=O, ray_directions=V)
    return O[clear] @ R.T + t, -(V[clear] @ R.T)      # the push runs against the escape


def holds(W, D, gravity, k, active) -> bool:
    """Can these contacts balance gravity plus k times any one push?"""
    Wa = W[:, active]
    if not Wa.shape[1] or not opposable(Wa, -gravity)[0]:
        return False
    return all(opposable(Wa, -(k * D[:, i] + gravity))[0] for i in range(D.shape[1]))


def features(p, u, scale):
    """Where a bit of surface is and which way it faces, in one vector."""
    return np.hstack([p / max(scale, 1e-9), 0.6 * u])


def patches(p, u, scale, n, seed):
    """Cut the touchable surface into pads a support could be built on.

    Grouped by where they are and which way they face together, so a pad is
    something that could be one printed conforming face rather than a scatter.
    A pad is a region of the surface, not the samples inside it: keeping one
    keeps the whole share of the workpiece it stands for, which is what a printed
    support actually touches.
    """
    x = features(p, u, scale)
    n = min(n, len(x))
    centre, label = kmeans2(x, n, minit="++", seed=seed, iter=40)
    return label, centre


def shrink(W, D, gravity, k, label, weight):
    """Hand back whole patches, biggest first, for as long as it still holds."""
    ids = np.array(sorted(set(label.tolist())))
    area = np.array([weight[label == i].sum() for i in ids])
    active = np.ones(len(label), bool)
    kept = set(ids.tolist())
    for i in ids[np.argsort(-area)]:
        trial = active & (label != i)
        if trial.any() and holds(W, D, gravity, k, trial):
            active, _ = trial, kept.discard(i)
    return active, sorted(kept)


# ------------------------------------------------------------------ drawing ---

def paint(mesh, T, inside, keep_faces, out_dir, tag, rel="shrink/paint"):
    """Three colours: the work region, the contact kept, the surface let go."""
    out_dir.mkdir(parents=True, exist_ok=True)
    kept = np.zeros(len(mesh.faces), bool)
    kept[list(keep_faces)] = True
    parts = []
    for material, sel in (("work", inside), ("kept", kept & ~inside),
                          ("spare", ~kept & ~inside)):
        if not sel.any():
            continue
        patch = mesh.submesh([np.flatnonzero(sel)], append=True)
        v = patch.vertices - patch.vertices.mean(axis=0)
        if np.linalg.matrix_rank(v, tol=1e-9 * max(patch.scale, 1e-9)) < 3:
            patch.vertices[::2] += 2e-5 * patch.face_normals[0]
        name = f"{tag}_{material}.obj"
        patch.export(out_dir / name)
        parts.append((f"{rel}/{name}", material))
    return parts


def render(name, T, parts, px, azimuth, marks=(), arrows=()):
    obj = obj_path(name)
    xml = SCENE.replace(
        '<material name="work" rgba="0.16 0.68 0.40 1" specular="0.15" shininess="0.2"/>',
        f'<material name="work" rgba="{WORK}" specular="0.1"/>').replace(
        '<material name="rest" rgba="0.85 0.79 0.68 1" specular="0.1"/>',
        f'<material name="rest" rgba="{GIVEN_UP}" specular="0.1"/>\n'
        f'    <material name="kept" rgba="{KEPT}" specular="0.25" shininess="0.3"/>\n'
        f'    <material name="spare" rgba="{GIVEN_UP}" specular="0.05"/>').format(
        assets="\n".join(f'    <mesh name="p{i}" file="{f}"/>'
                         for i, (f, _) in enumerate(parts)),
        geoms="\n".join(f'      <geom type="mesh" mesh="p{i}" material="{m}"/>'
                        for i, (_, m) in enumerate(parts)),
        pos=" ".join(f"{v:.9g}" for v in T[:3, 3]),
        quat=" ".join(f"{v:.9g}" for v in mat_to_quat_wxyz(T[:3, :3])))
    tmp = obj / ".shrink_tmp.xml"
    tmp.write_text(xml)
    try:
        model = mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    mesh = trimesh.load(obj / "mesh.stl", force="mesh")
    V = np.asarray(mesh.vertices) @ T[:3, :3].T + T[:3, 3]
    lo, hi = V.min(axis=0), V.max(axis=0)
    size = float(np.linalg.norm(hi - lo))
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation = azimuth, -16.0
    cam.lookat[:] = (lo + hi) / 2
    cam.distance = 1.7 * size / 2 / np.tan(np.deg2rad(model.vis.global_.fovy / 2))
    rgba = np.array([float(x) for x in KEPT.split()], np.float32)
    with YUpRenderer(model, px, px, max_geom=3000) as r:
        r.update_scene(data, camera=cam)
        scn = r.scene
        for q, n, col in arrows:               # whatever the caller wants drawn
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3),
                                np.zeros(3), np.zeros(9),
                                np.array(col, np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 2.0e-3,
                                 np.asarray(q) - np.asarray(n) * 0.16 * size,
                                 np.asarray(q))
            scn.ngeom += 1
        for q, n in marks:                     # a pin per pad, so small ones show
            for geom, sz, a, b in ((mujoco.mjtGeom.mjGEOM_ARROW, 5.5e-3,
                                    np.asarray(q) - np.asarray(n) * 0.32 * size,
                                    np.asarray(q)),):
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g, geom, np.zeros(3), np.zeros(3), np.zeros(9), rgba)
                mujoco.mjv_connector(g, geom, sz, a, b)
                scn.ngeom += 1
        return Image.fromarray(r.render())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("object")
    ap.add_argument("--poses", type=int, nargs="+", default=None)
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--patches", type=int, default=140)
    ap.add_argument("--candidates", type=int, default=1400)
    ap.add_argument("--disturbances", type=int, default=140)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=400)
    args = ap.parse_args()

    d = obj_path(args.object)
    mesh = trimesh.load(d / "mesh.stl", force="mesh")
    examples = read_json(d / "tips" / "tips.json")["examples"]
    poses = args.poses if args.poses is not None else range(len(examples))
    scale = float(np.linalg.norm(mesh.extents))

    rows, records = [], []
    for pose in poses:
        ex = examples[pose]
        T = np.asarray(ex["T_world_mesh"])
        rng = np.random.default_rng(args.seed + 1000 * pose)
        n_passes = int(rng.integers(1, 7))
        dirs = gun_directions(rng, n_passes)
        inside = region_mask(mesh, T, dirs, mesh.triangles_center, mesh.face_normals)
        area_region = float(mesh.area_faces[inside].sum() / mesh.area)

        s = contact_wrenches(mesh, T, inside, args.candidates, args.disturbances,
                             args.seed + pose)
        W, D, g = s["W"], s["D"], s["gravity"]

        everything = np.ones(W.shape[1], bool)
        if not holds(W, D, g, args.k, everything):
            print(f"pose {pose:2d} ({n_passes} pass, region {100 * area_region:4.1f}%): "
                  f"gripping everything outside the work region is NOT enough "
                  f"-> None", flush=True)
            records.append({"pose": pose, "n_passes": n_passes,
                            "region_area_fraction": area_region, "feasible": False})
            continue

        label, centre = patches(s["p"], s["u"], scale, args.patches, args.seed)
        weight = np.ones(len(label))
        active, kept_ids = shrink(W, D, g, args.k, label, weight)

        # a pad stands for a whole region of the surface, so hand every face
        # outside the work region to its nearest pad and keep the pads that stayed
        off = np.flatnonzero(~inside)
        fx = features(mesh.triangles_center[off] @ T[:3, :3].T + T[:3, 3],
                      -(mesh.face_normals[off] @ T[:3, :3].T), scale)
        owner = np.argmin(((fx[:, None, :] - centre[None, :, :]) ** 2).sum(axis=2),
                          axis=1)
        faces = set(off[np.isin(owner, kept_ids)].tolist())
        held_area = float(mesh.area_faces[sorted(faces)].sum() / mesh.area)
        parts = paint(mesh, T, inside, faces, d / "shrink" / "paint", f"p{pose}")
        marks = []
        for i in kept_ids:
            sel = off[owner == i]
            if len(sel):
                q = mesh.triangles_center[sel].mean(axis=0) @ T[:3, :3].T + T[:3, 3]
                nrm = -(mesh.face_normals[sel].mean(axis=0) @ T[:3, :3].T)
                marks.append((q, nrm / max(np.linalg.norm(nrm), 1e-9)))
        rows.append((pose, n_passes, area_region, len(kept_ids), int(active.sum()),
                     held_area, [render(args.object, T, parts, args.size, az, marks)
                                 for az in (135.0, 315.0)]))
        records.append({"pose": pose, "pivot": ex["pivot"], "tip_deg": ex["tip_deg"],
                        "n_passes": n_passes, "region_area_fraction": area_region,
                        "feasible": True, "n_patches_start": len(set(label.tolist())),
                        "n_patches_kept": len(kept_ids),
                        "n_contact_points_kept": int(active.sum()),
                        "kept_area_fraction": held_area,
                        "contacts": [{"p": [float(x) for x in s["p"][i]],
                                      "push": [float(x) for x in s["u"][i]]}
                                     for i in np.flatnonzero(active)]})
        print(f"pose {pose:2d} ({n_passes} pass, region {100 * area_region:4.1f}%): "
              f"holds -> shrank {len(set(label.tolist())):3d} pads to "
              f"{len(kept_ids):3d}, contact area {100 * held_area:4.1f}% of the "
              f"surface ({100 * held_area / max(1 - area_region, 1e-9):4.1f}% of what "
              f"was available)", flush=True)

    write_json(d / "shrink" / f"shrink_k{args.k:g}.json",
               {"object": args.object, "k": args.k, "patches": args.patches,
                "candidates": args.candidates, "disturbances": args.disturbances,
                "seed": args.seed, "poses": records})
    if not rows:
        print("nothing to draw")
        return

    px, label_w, head = args.size, int(args.size * 1.2), int(args.size * 0.34)
    sheet = Image.new("RGB", (label_w + 2 * px, head + px * len(rows)), "white")
    dr = ImageDraw.Draw(sheet)
    f, fs = _font(int(px * 0.058)), _font(int(px * 0.050))
    dr.text((14, 12),
            f"{args.object} -- grip everything outside the work region, then hand back "
            f"every pad that can be spared  (K = {args.k:g})\n"
            "green = the work region, which must stay clear\n"
            "ORANGE = the conforming contact face that survived, with an arrow per pad; no pad of it can be given up\n"
            "beige = surface a support could have touched but does not need to",
            fill=(20, 20, 20), font=f)
    for r, (pose, n_passes, ar, npk, npt, ha, imgs) in enumerate(rows):
        y = head + r * px
        for c, im in enumerate(imgs):
            sheet.paste(im, (label_w + c * px, y))
        dr.text((12, y + int(px * 0.14)),
                f"pose {pose}   {n_passes} spray pass(es)\n"
                f"work region {100 * ar:.0f}% of the surface\n\n"
                f"contact area {100 * ha:.1f}% of the surface\n"
                f"in {npk} pads",
                fill=(40, 40, 40), font=fs)
        dr.line([(0, y), (sheet.size[0], y)], fill=(215, 215, 215))
    out = d / "shrink" / f"sheet_k{args.k:g}.png"
    sheet.save(out)
    print(f"\n{out}  {sheet.size[0]}x{sheet.size[1]}")


if __name__ == "__main__":
    main()
