"""Work regions: the patch of the workpiece a process has to reach.

The work region is an input of the problem, and it is what makes a support design
non-trivial: the supports have to stay off it and clear of whatever reaches it.
Which process it is does not matter to anything downstream -- only where it can
push does -- so there is one generator, for the shape a coating job takes.

**spray** -- coating, painting, blasting. A real coating job is several passes
from several directions, so the region is the union of the passes: for each one,
everything that both faces the gun and is in its line of sight. With the guns
above the workpiece the union lands on the upper side and covers many faces.

A region has to be able to cover *part* of a face, which the raw mesh cannot
express. Refining everything is not an option -- these meshes are mostly
sub-millimetre triangles with a handful of 100 mm flat ones, and a uniform refine
runs into millions of faces. Only the faces the region boundary actually runs
through are split, and only until they are small enough.

    python slides/tools/work_regions.py A1-f
    python slides/tools/work_regions.py A1-f --pose 2 --passes 1 2 3 6
"""
from __future__ import annotations

import argparse

import mujoco
from mujoco import Renderer
import numpy as np
import trimesh
from PIL import Image, ImageDraw

from common import mat_to_quat_wxyz, obj_path, read_json, write_json
from disturbances import _font

SPRAY_INCIDENCE = 60.0   # deg; past this the spray hits too obliquely to count
GUN_MIN_ELEVATION = 0.35 # the gun sits above the workpiece, not level with it
BOUNDARY_EDGE = 0.012    # of the workpiece size; how finely the region edge is cut
SPLIT_ROUNDS = 7
SPOT_SAMPLES = 60000     # surface points the finite-beam radius is read off
ARROW_WIDTH = 2.4e-3

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
    <material name="work" rgba="0.16 0.68 0.40 1" specular="0.15" shininess="0.2"/>
    <material name="rest" rgba="0.85 0.79 0.68 1" specular="0.1"/>
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


# ------------------------------------------------------ boundary refinement ---

def split_along_boundary(occluder: trimesh.Trimesh, label, target_edge: float):
    """Refine only where the label changes; return the refined mesh and labels.

    The label is recomputed each round, so it lands on ever finer faces exactly
    where the region boundary runs and nowhere else. Rays are always cast against
    the original mesh, which is the real occluder.
    """
    v, f = occluder.vertices.copy(), occluder.faces.copy()
    for _ in range(SPLIT_ROUNDS):
        m = trimesh.Trimesh(v, f, process=False)
        lab = label(m)
        adj = m.face_adjacency
        diff = lab[adj[:, 0]] != lab[adj[:, 1]]
        edge = np.zeros(len(f), bool)
        edge[adj[diff].ravel()] = True
        big = m.area_faces > target_edge ** 2 / 2
        sel = np.flatnonzero(edge & big)
        if not len(sel):
            return m, lab
        v, f = trimesh.remesh.subdivide(v, f, face_index=sel)
    m = trimesh.Trimesh(v, f, process=False)
    return m, label(m)


# --------------------------------------------------------------- generators ---

def spray_pass(mesh, occluder, R, u_world) -> np.ndarray:
    """One pass: faces that both face the gun and are in its line of sight."""
    n = mesh.face_normals @ R.T
    local = u_world @ R
    origins = mesh.triangles_center + mesh.face_normals * 1e-5
    clear = ~occluder.ray.intersects_any(
        ray_origins=origins, ray_directions=np.tile(local, (len(origins), 1)))
    return clear & (n @ u_world > np.cos(np.radians(SPRAY_INCIDENCE)))


def gun_directions(rng, n_passes: int) -> list:
    """Where the passes come from: random, but always above the workpiece."""
    dirs = []
    for _ in range(n_passes):
        while True:
            u = rng.normal(size=3)
            u /= np.linalg.norm(u)
            if u[2] > GUN_MIN_ELEVATION:               # the gun is above the part
                break
        dirs.append(u)
    return dirs


def spray_region(occluder, T, rng, n_passes, target_edge) -> dict:
    R = T[:3, :3]
    dirs = gun_directions(rng, n_passes)

    def label(m):
        mask = np.zeros(len(m.faces), bool)
        for u in dirs:
            mask |= spray_pass(m, occluder, R, u)
        return mask

    mesh, mask = split_along_boundary(occluder, label, target_edge)
    per_pass = [float(mesh.area_faces[spray_pass(mesh, occluder, R, u)].sum() / mesh.area)
                for u in dirs]
    return {"kind": "spray", "n_passes": n_passes,
            "gun_directions": [[float(v) for v in u] for u in dirs],
            "pass_area_fractions": per_pass,
            "incidence_limit_deg": SPRAY_INCIDENCE, "mesh": mesh, "mask": mask}


def in_beam(pts_local, R, u_world, centre_local, radius) -> np.ndarray:
    """Points within `radius` of the gun's axis: the line through the centre along u."""
    u = u_world @ R                                    # the axis, in the mesh frame
    r = np.asarray(pts_local) - centre_local
    return np.linalg.norm(r - np.outer(r @ u, u), axis=1) <= radius


def spot_pass(mesh, occluder, R, u_world, centre_local, radius) -> np.ndarray:
    """One pass of a gun whose beam has a finite width."""
    return spray_pass(mesh, occluder, R, u_world) & in_beam(
        mesh.triangles_center, R, u_world, centre_local, radius)


def spot_region(occluder, T, rng, target_fraction, target_edge) -> dict:
    """A gun with a beam of finite width, opened until it covers the work asked for.

    An unbounded pass takes every face that meets the gun at all. On a part with
    six of them that is half the workpiece, and worse, its boundary runs along
    the edges rather than across the faces, so the region can never cover part of
    a face -- which is the one thing a work region has to be able to do. A real
    gun has a spot. How large the region is is an input of the problem rather
    than something to be discovered, so the beam is opened until it covers the
    fraction asked for, and where the beam falls is what is left to chance.
    """
    R = T[:3, :3]
    u = gun_directions(rng, 1)[0]

    # The radius is read off a sample of the surface rather than bisected on the
    # mesh: line of sight does not depend on it, so which points the gun can see
    # is settled once and the radius is then a quantile of their distance from
    # the axis. Area-uniform samples are what make a count an area.
    pts, face = trimesh.sample.sample_surface(occluder, SPOT_SAMPLES,
                                              seed=int(rng.integers(1 << 30)))
    n = occluder.face_normals[face]
    lit = (n @ R.T) @ u > np.cos(np.radians(SPRAY_INCIDENCE))
    lit &= ~occluder.ray.intersects_any(
        ray_origins=pts + n * 1e-5, ray_directions=np.tile(u @ R, (len(pts), 1)))

    centre = pts[rng.choice(np.flatnonzero(lit))]
    axis = u @ R
    off = pts - centre
    d = np.linalg.norm(off - np.outer(off @ axis, axis), axis=1)[lit]
    want = int(round(target_fraction * SPOT_SAMPLES))
    # asking for more than the gun can see at all: the beam opens to the whole
    # lit patch and the region simply comes out smaller than asked
    radius = float(np.sort(d)[want if want < len(d) else -1])

    mesh, mask = split_along_boundary(
        occluder, lambda m: spot_pass(m, occluder, R, u, centre, radius), target_edge)
    return {"kind": "spot", "n_passes": 1,
            "gun_directions": [[float(v) for v in u]],
            "beam_centre_mesh": [float(v) for v in centre],
            "beam_radius_m": radius,
            "target_area_fraction": float(target_fraction),
            "lit_area_fraction": float(lit.mean()),
            "incidence_limit_deg": SPRAY_INCIDENCE, "mesh": mesh, "mask": mask}


def describe(mesh, T, mask) -> dict:
    area = mesh.area_faces
    n = mesh.face_normals @ T[:3, :3].T
    up = area[mask & (n[:, 2] > 0)].sum() / max(area[mask].sum(), 1e-12)
    idx = np.flatnonzero(mask)
    pairs = mesh.face_adjacency
    keep = mask[pairs[:, 0]] & mask[pairs[:, 1]]
    comps = trimesh.graph.connected_components(pairs[keep], nodes=idx) if len(idx) else []
    areas = sorted((area[list(c)].sum() for c in comps), reverse=True)
    return {"n_faces": int(mask.sum()),
            "area_fraction": float(area[mask].sum() / area.sum()),
            "fraction_facing_up": float(up),
            "n_pieces": int(len(comps)),
            "largest_piece_fraction": float(areas[0] / sum(areas)) if areas else 0.0}


# ------------------------------------------------------------------ render ---

def paint(mesh, mask, out_dir, tag: str) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for material, sel in (("work", mask), ("rest", ~mask)):
        if not sel.any():
            continue
        patch = mesh.submesh([np.flatnonzero(sel)], append=True)
        v = patch.vertices - patch.vertices.mean(axis=0)
        if np.linalg.matrix_rank(v, tol=1e-9 * max(patch.scale, 1e-9)) < 3:
            patch.vertices[::2] += 2e-5 * patch.face_normals[0]   # MuJoCo needs a hull
        fname = f"{tag}_{material}.obj"
        patch.export(out_dir / fname)
        parts.append((f"work_regions/paint/{fname}", material))
    return parts


def render(name, T, parts, guns, px, azimuth) -> Image.Image:
    obj = obj_path(name)
    xml = SCENE.format(
        assets="\n".join(f'    <mesh name="p{i}" file="{f}"/>' for i, (f, _) in enumerate(parts)),
        geoms="\n".join(f'      <geom type="mesh" mesh="p{i}" material="{m}"/>'
                        for i, (_, m) in enumerate(parts)),
        pos=" ".join(f"{v:.9g}" for v in T[:3, 3]),
        quat=" ".join(f"{v:.9g}" for v in mat_to_quat_wxyz(T[:3, :3])))
    tmp = obj / ".work_tmp.xml"
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

    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation = azimuth, -18.0
    cam.lookat[:] = (lo + hi) / 2
    cam.distance = 1.7 * size / 2 / np.tan(np.deg2rad(model.vis.global_.fovy / 2))

    with Renderer(model, px, px, max_geom=200) as r:
        r.update_scene(data, camera=cam)
        if guns is not None:
            scn, mid = r.scene, (lo + hi) / 2
            for u in np.atleast_2d(guns):
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                                    np.zeros(9), np.array([0.15, 0.15, 0.15, 1], np.float32))
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, ARROW_WIDTH,
                                     mid + u * size * 0.72, mid + u * size * 0.45)
                scn.ngeom += 1
        return Image.fromarray(r.render())


# -------------------------------------------------------------------- main ---

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("object")
    ap.add_argument("--pose", type=int, default=0)
    ap.add_argument("--passes", type=int, nargs="+", default=[1, 2, 3, 6],
                    help="one spray region per entry, unioning that many passes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=380)
    args = ap.parse_args()

    d = obj_path(args.object)
    occluder = trimesh.load(d / "mesh.stl", force="mesh")
    target_edge = BOUNDARY_EDGE * float(np.linalg.norm(occluder.extents))
    ex = read_json(d / "tips" / "tips.json")["examples"][args.pose]
    T = np.asarray(ex["T_world_mesh"])
    rng = np.random.default_rng(args.seed)

    out = d / "work_regions"
    regions, rows = [], []
    for i, k in enumerate(args.passes):
        reg = spray_region(occluder, T, rng, k, target_edge)
        mesh, mask = reg.pop("mesh"), reg.pop("mask")
        reg.update(describe(mesh, T, mask))
        reg["n_faces_after_refinement"] = int(len(mesh.faces))
        regions.append(reg)

        tag = f"p{args.pose}_spray{i}"
        parts = paint(mesh, mask, out / "paint", tag)
        rows.append((reg, [render(args.object, T, parts,
                                  np.asarray(reg["gun_directions"]), args.size, az)
                           for az in (135.0, 315.0)]))
        print(f"spray x{k}: {100 * reg['area_fraction']:5.1f}% of the surface, "
              f"{reg['n_pieces']:3d} piece(s), largest is "
              f"{100 * reg['largest_piece_fraction']:5.1f}% of it, "
              f"{100 * reg['fraction_facing_up']:5.1f}% faces up "
              f"({len(occluder.faces)} -> {len(mesh.faces)} faces)", flush=True)

    write_json(out / f"regions_pose{args.pose}.json",
               {"object": args.object, "pose": args.pose, "pivot": ex["pivot"],
                "tip_deg": ex["tip_deg"], "seed": args.seed,
                "boundary_edge_m": target_edge, "regions": regions})

    px = args.size
    label_w, head_h = int(px * 0.85), int(px * 0.16)
    sheet = Image.new("RGB", (label_w + 2 * px, head_h + px * len(rows)), "white")
    dr = ImageDraw.Draw(sheet)
    f, fs = _font(int(px * 0.052)), _font(int(px * 0.050))
    dr.text((label_w + 8, int(head_h * 0.3)),
            "green = the work region; grey arrows = where the spray comes from",
            fill=(20, 20, 20), font=f)
    for r, (reg, imgs) in enumerate(rows):
        y = head_h + r * px
        for c, im in enumerate(imgs):
            sheet.paste(im, (label_w + c * px, y))
        head = f"{reg['n_passes']} spray pass(es)"
        dr.text((10, y + int(px * 0.14)),
                f"{head}\n\n"
                f"{100 * reg['area_fraction']:.1f}% of the surface\n"
                f"{reg['n_pieces']} piece(s)\n"
                f"largest is {100 * reg['largest_piece_fraction']:.0f}% of it\n"
                f"{100 * reg['fraction_facing_up']:.0f}% faces up",
                fill=(40, 40, 40), font=fs)
        dr.line([(0, y), (sheet.size[0], y)], fill=(215, 215, 215))
    sheet.save(out / f"sheet_pose{args.pose}.png")
    print(f"\n{out / f'sheet_pose{args.pose}.png'}  {sheet.size[0]}x{sheet.size[1]}")


if __name__ == "__main__":
    main()
