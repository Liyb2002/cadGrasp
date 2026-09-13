"""One page that walks a single pose from the process to a thing you could print.

`capped_a1f.py` answers the force question and stops at a list of directions.
`demo_solid.py` starts from that list and makes geometry. Between them there is a
gap a reader has to jump: the sphere panels never touch the part, and the solid
never says why those four places and not four others. This figure closes it.

The argument, left to right:

  1  THE JOB.  Every direction the process can push the workpiece in.
  2  WHAT IS OWED.  Reverse them -- what has to be produced is the opposite of
     what is applied -- and add the body's own weight. That is the requirement
     `T = up - d`, a patch on the sphere with a MAGNITUDE, up to 2 body weights.
  3  WHAT WE HAVE.  The inward normal of every face outside the work region: the
     directions this particular workpiece is able to be pushed in, one body
     weight each.
  4  ONE POINT AT A TIME.  The chosen contacts go in one by one. Each panel shows
     the strength field `rho` the set can produce and how much of the requirement
     that answers, and the bar under it says how much THIS point added. The
     numbers climb for a reason: the order is the greedy one, so each panel is
     the largest bite still available.
  5  THE SUPPORT.  The same four contacts as solid walls standing on the floor.
     The face that bears on the part is the tangent plane at the contact, so its
     normal IS the push direction the search chose -- the geometry is the force
     model, not an illustration of it.

Nothing in the physics moves; this file re-runs `capped_a1f`'s pipeline and
asserts it lands on the recorded answer before it draws anything. The only new
modelling is in `pad_wall`: `demo_solid` sweeps the pad's RIM downward, and for a
contact on a vertical face that rim is a circle standing on edge, so the sweep
comes out a sheet of paper. Giving the pad a body first and sweeping that turns
the same contact into a wall with a thickness.

--- why pose 6 ---------------------------------------------------------------

Because it is the pose where the drawing is not lying about anything.

Of the ten, only 2 and 6 take four supports AND give every one of them a clear
drop to the floor -- on the other eight at least one contact has the workpiece
directly underneath it, so a wall dropped from it runs through the part and the
last panel would be a picture of something that cannot be built. Between those
two, pose 2's contact 4 is unusable in a different way: it bears on a sliver of
surface about a tenth of a square millimetre across, tucked beside a concave
edge, and a seven-millimetre pad laid flat on it cuts 1.07 mm into the workpiece
whatever draft it is given. Pose 6's four contacts all seat with room to spare.

Pose 6 also happens to make the cleanest argument: the three contacts that do any
work take almost exactly a third of the requirement each.

The one thing it costs is that contacts 1 and 2 are thirteen millimetres apart,
which on a shared camera prints as one blob -- hence `spread`.

    python slides/tools/pipeline_a1f.py A1-f
    python slides/tools/pipeline_a1f.py A1-f --pose 2
"""
from __future__ import annotations

import argparse
import shutil

import mujoco
from yup_render import Renderer as YUpRenderer
import numpy as np
import coordinates as COORD
import trimesh
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from capped_a1f import (BARE, FLOOR_MARK, MARK, SHORT, answered, least_effort,
                        legend, lp_answered, radial, screen, shades)
from common import mat_to_quat_wxyz, obj_path, read_json
from cover import (DONE, HAVE, INK, NEED, PUSH_BLUE, UP, globe_png, neighbours,
                   paint, sheet, shot, tiling)
from disturbances import _font
from reach import compass
from shrink_support import angled_pushes
from supports import CONTACT_EPS, region_mask
from work_regions import gun_directions

PAD_R = 0.007          # 7 mm bearing pad -- the face that actually touches
WALL_T = 0.008         # material behind it, so the sweep is a wall and not a sheet
FLOOR = 0.0
STEP_INK = "#d2450f"   # the same orange the sphere panels number the contacts in
GAIN = "#186158"       # the slice a point adds, in the darkest teal of the ramp
HELD = "#a3d8cf"       # what was already answered before it went in

# one scene for every solid render: the part in its pose, the floor, and the
# fixture. Two prop materials, because figure 2 has to tell two pieces apart and
# the two figures must not disagree about what orange means.
SCENE = """<mujoco>
  <compiler meshdir="." angle="radian"/>
  <visual>
    <global offwidth="2200" offheight="2200" fovy="45"/>
    <quality shadowsize="8192" offsamples="8"/>
    <map znear="0.005" zfar="30"/>
    <headlight ambient="0.44 0.44 0.44" diffuse="0.50 0.50 0.50" specular="0.08 0.08 0.08"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="1 1 1"
             width="256" height="256"/>
    <material name="floor" rgba="0.97 0.97 0.97 1"/>
    <material name="work" rgba="0.55 0.82 0.62 1" specular="0.1"/>
    <material name="kept" rgba="0.90 0.89 0.85 1" specular="0.1"/>
    <material name="spare" rgba="0.90 0.89 0.85 1" specular="0.1"/>
    <material name="propA" rgba="0.85 0.34 0.06 1" specular="0.25" shininess="0.35"/>
    <material name="propB" rgba="0.16 0.42 0.68 1" specular="0.25" shininess="0.35"/>
{assets}
  </asset>
  <worldbody>
    <light pos="0.4 1.2 -0.5" dir="-0.3 -1 0.4" directional="true" castshadow="true"
           diffuse="0.55 0.55 0.55"/>
    <geom type="plane" quat="0.7071067811865476 -0.7071067811865476 0 0" size="6 6 0.05" material="floor"/>
    <body pos="{pos}" quat="{quat}">
{geoms}
    </body>
{props}
  </worldbody>
</mujoco>
"""


# ------------------------------------------------------------- the pipeline ---

def replicate(name: str, pose: int, k: float = 1.0, seed: int = 0,
              n_points: int = 90, n_dirs: int = 24, check: bool = True) -> dict:
    """Run `capped_a1f`'s pipeline again and refuse to go on unless it lands.

    Everything downstream of here draws the answer rather than recomputing it,
    so the one thing that can go silently wrong is replicating with the wrong
    seed or the wrong region and drawing a picture of a different problem. The
    recorded json holds four numbers that a wrong replication cannot match by
    accident -- how many requirements the region generated, how much surface it
    covers, how many distinct push directions the part offers, and what fraction
    the chosen set answers -- so all four are asserted, not printed and ignored.

    The chosen supports are matched to the rebuilt candidate list BY DIRECTION,
    not by index: the list is rebuilt here and only the directions themselves are
    stable across runs. That is `capped_a1f --reuse`'s own rule.
    """
    d = obj_path(name)
    mesh = trimesh.load(d / "mesh.stl", force="mesh")
    examples = read_json(d / "tips" / "tips.json")["examples"]
    T = np.asarray(examples[pose]["T_world_mesh"])
    R, t = T[:3, :3], T[:3, 3]

    rng = np.random.default_rng(seed + 1000 * pose)
    n_passes = int(rng.integers(1, 4))
    part = mesh
    inside = region_mask(mesh, T, gun_directions(rng, n_passes),
                         mesh.triangles_center, mesh.face_normals)
    area = float(part.area_faces[inside].sum() / part.area)

    pw, pu = angled_pushes(part, T, inside, n_points, n_dirs, seed + pose)
    targets = UP - k * pu
    keep = np.linalg.norm(targets, axis=1) > 1e-9
    targets, pw, pu = targets[keep], pw[keep], pu[keep]

    on_floor = (part.triangles_center @ R.T + t)[:, 1] <= CONTACT_EPS
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
        assert abs(float(lp.mean()) - rec["answered_fraction"]) < 1e-12, "answer drifted"
        assert np.allclose(pts, [s["p"] for s in rec["supports"]], atol=1e-12)
        assert np.allclose(us, [s["push"] for s in rec["supports"]], atol=1e-12)

    return dict(mesh=mesh, part=part, T=T, inside=inside, n_passes=n_passes,
                area=area, pw=pw, pu=pu, targets=targets, avail=avail,
                avail_face=avail_face, chosen=chosen, pts=pts, us=us, G=G,
                lp=lp, ico=ico, tiles=tiles, tree=tree, record=rec, k=k)


def steps_of(targets, avail, chosen):
    """The fraction answered after each contact goes in, and what each one added."""
    frac = []
    for i in range(len(chosen) + 1):
        Gi = np.column_stack([UP] + [avail[c] for c in chosen[:i]])
        frac.append(float(answered(Gi, targets).mean()))
    return np.array(frac), np.diff(frac)


# ---------------------------------------------------------------- the solid ---

def pad_wall(p, u, r: float = PAD_R, t: float = WALL_T, floor: float = FLOOR,
             n: int = 40, draft: float = 0.0) -> trimesh.Trimesh:
    """The contact made solid: a puck in the tangent plane, swept down to the floor.

    The bearing face is the disc at `p` whose normal is `u`, which is the whole
    of the contact model -- frictionless, one direction, that direction. Behind
    it goes `t` of material along `-u`, and THAT is what gets swept to the floor.
    Sweeping the bare rim instead (what `demo_solid` does) is exact for a contact
    on a horizontal face and degenerate for one on a vertical face, where the rim
    is a circle standing on edge and its shadow is a line.

    The swept volume of a convex solid under a translation is the convex hull of
    the solid and its translate, so the hull here is the sweep exactly and not an
    approximation of it.

    `draft` is the angle by which the sweep leans AWAY from the surface as it
    goes down, and it is not cosmetic. A tipped workpiece has no exactly vertical
    faces: A1-f's pose-2 contact 4 bears on a face a degree and a half off plumb,
    and a wall dropped straight from a pad ninety millimetres up follows that face
    inward and ends up a millimetre inside the part. Draft is the fixture-maker's
    answer and it costs nothing that matters -- the pad at `p` is untouched, so
    the contact the search verified is exactly the contact that gets built.
    """
    u = np.asarray(u, float)
    u = u / np.linalg.norm(u)
    a = np.array([0.0, 1.0, 0.0]) if abs(u[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = -np.cross(u, a); e1 /= np.linalg.norm(e1)
    e2 = -np.cross(u, e1)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    face = np.asarray(p) + r * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)
    puck = np.vstack([face, face - t * u])
    drop = puck[:, 1].min() - floor
    back = COORD.floor(u) / max(float(np.linalg.norm(COORD.floor(u))), 1e-9)
    shift = COORD.lift_floor(-drop * np.tan(np.deg2rad(draft)) * back, -drop)
    pts = np.vstack([puck, puck + shift]) if drop > 0 else puck
    solid = trimesh.convex.convex_hull(pts)
    if pts[:, 1].min() < floor - 1e-12:
        # a contact only a few millimetres up leaves the pad hanging through the
        # ground. Cutting it off is the honest fix: the wall loses the part of
        # itself that was never buildable, and keeps the whole bearing face that
        # is above the floor.
        box = trimesh.creation.box(extents=[1.0, 1.0, 1.0],
                                   transform=trimesh.transformations.translation_matrix(
                                       [0, floor + 0.5, 0]))
        solid = trimesh.boolean.intersection([solid, box], engine="manifold")
    return solid


DRAFTS = (0.0, 1.0, 2.0, 3.0, 4.5, 6.0, 9.0, 14.0)


def fit_wall(p, u, world: trimesh.Trimesh, **kw):
    """The least draft that keeps the wall out of the workpiece, and the proof.

    The alternative -- subtracting the part from the wall -- would also produce a
    wall that does not overlap, and it would quietly change the model: the bearing
    face would stop being a plane and the support would touch over a patch with a
    spread of normals, which is not the single direction the search checked. Draft
    keeps the contact a contact.
    """
    for draft in DRAFTS:
        w = pad_wall(p, u, draft=draft, **kw)
        inter = trimesh.boolean.intersection([w, world], engine="manifold")
        vol = float(inter.volume) if len(inter.faces) else 0.0
        if vol <= 1e-12:
            return w, draft, vol
    return w, DRAFTS[-1], vol


def camera(lo, hi, az, el, zoom=1.30, lift=0.42, fovy=45.0):
    """A camera framing the part AND whatever stands on the floor beside it."""
    look = np.array([(lo[0] + hi[0]) / 2, hi[1] * lift, (lo[2] + hi[2]) / 2])
    span = max(float(np.linalg.norm(COORD.floor(hi-lo))), float(hi[1]) * 1.5)
    dist = zoom * span / 2 / np.tan(np.deg2rad(fovy / 2))
    a, e = np.deg2rad(az), np.deg2rad(el)
    fwd = np.array([np.cos(e) * np.cos(a), np.sin(e), np.cos(e) * np.sin(a)])
    right = np.array([np.sin(a), 0.0, -np.cos(a)])
    return dict(look=look, dist=dist, az=az, el=el, fwd=fwd, right=right,
                up=-np.cross(right, fwd), fovy=fovy)


def project(cam, w, h, pts):
    """Where world points land on a render made with `cam`, so they can be numbered.

    mujoco's fovy is the VERTICAL field, so a wide render stretches the horizontal
    half-angle by the aspect ratio. Getting that backwards puts every number a few
    per cent off centre, which on a 1800-pixel render is the width of a wall.
    """
    v = np.asarray(pts) - (cam["look"] - cam["dist"] * cam["fwd"])
    fov = np.tan(np.deg2rad(cam["fovy"] / 2))
    z = (v @ cam["fwd"])[:, None]
    s = np.stack([v @ cam["right"] / (fov * w / h), v @ cam["up"] / fov], 1) / z
    return np.stack([(0.5 + 0.5 * s[:, 0]) * w, (0.5 - 0.5 * s[:, 1]) * h], 1)


def solid_shot(name, T, parts, solids, cam, w, h):
    """The part in its pose with a set of (mesh, material) props standing beside it."""
    obj = obj_path(name)
    assets = [f'    <mesh name="p{i}" file="{f}"/>' for i, (f, _) in enumerate(parts)]
    geoms = [f'      <geom type="mesh" mesh="p{i}" material="{m}"/>'
             for i, (_, m) in enumerate(parts)]
    props = []
    for i, (path, material) in enumerate(solids):
        assets.append(f'    <mesh name="s{i}" file="{path}"/>')
        props.append(f'    <geom type="mesh" mesh="s{i}" material="{material}"/>')
    xml = SCENE.format(assets="\n".join(assets), geoms="\n".join(geoms),
                       props="\n".join(props),
                       pos=" ".join(f"{v:.9g}" for v in T[:3, 3]),
                       quat=" ".join(f"{v:.9g}" for v in mat_to_quat_wxyz(T[:3, :3])))
    f = obj / ".pipeline.xml"
    f.write_text(xml)
    try:
        model = mujoco.MjModel.from_xml_path(str(f))
    finally:
        f.unlink(missing_ok=True)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mj = mujoco.MjvCamera()
    # mujoco names the direction the camera looks ALONG; `camera` names where the
    # eye stands, the same convention `cover.shot` uses, so the two agree
    mj.azimuth, mj.elevation = cam["az"], cam["el"]
    mj.lookat[:] = cam["look"]
    mj.distance = cam["dist"]
    with YUpRenderer(model, h, w) as r:
        r.update_scene(data, camera=mj)
        return Image.fromarray(r.render())


def spread(xy, px, sep=0.070):
    """Push labels apart until none sits on top of another, and say where they went.

    Two contacts a centimetre apart on a hundred-millimetre part project a few
    tens of pixels apart, and two numbers that size print as one illegible blob --
    which is what a fixed camera did to `capped_a1f`'s poses 1, 5 and 6. Nudging
    the labels and leaving a leader line to the real point keeps both the count
    and the position.
    """
    q = np.array(xy, float)
    s = sep * px
    for _ in range(300):
        moved = False
        for i in range(len(q)):
            for j in range(i + 1, len(q)):
                d = q[j] - q[i]
                n = float(np.linalg.norm(d))
                if n < s:
                    if n < 1e-6:
                        d, n = np.array([1.0, 0.0]), 1.0
                    q[i] -= (s - n) / 2 * d / n
                    q[j] += (s - n) / 2 * d / n
                    moved = True
        if not moved:
            break
    return q


def number(im, xy, labels, px, ink=MARK, size=0.048, seen=None):
    """Contacts written on rather than left to be seen: one camera buries some.

    Where `seen` says the contact is behind something, the number goes on hollow
    rather than solid. Drawing a buried contact the same as a visible one puts a
    confident marker on a face it is not on, which is worse than not drawing it;
    dropping it altogether leaves the reader counting three walls out of four.
    """
    dr = ImageDraw.Draw(im)
    font = _font(int(px * size))
    ghost = _font(int(px * size * 0.78))
    xy = np.asarray(xy, float)
    q = spread(xy, px, sep=size * 1.55)
    for i, lab in enumerate(labels):
        vis = True if seen is None else bool(seen[i])
        if np.linalg.norm(q[i] - xy[i]) > px * 0.012:
            dr.line([tuple(xy[i]), tuple(q[i])], fill=ink, width=max(2, int(px * .003)))
        dr.text(tuple(q[i]), str(lab), fill="white" if vis else "#b9b4a8",
                font=font if vis else ghost, anchor="mm",
                stroke_width=max(3, int(px * 0.005)),
                stroke_fill=ink if vis else "white")
    return im


def visible(cam, part_world: trimesh.Trimesh, solids) -> np.ndarray:
    """Which of the solids the camera can actually see any of.

    Shooting at the CONTACT POINT instead answers a different question and always
    answers it no: the wall sits between the camera and its own contact whenever
    the camera is on the support's side, which is exactly when the wall is in
    plain view. What the number marks is the wall, so the wall is what is tested
    -- visible if any point of it is the first thing its ray meets.
    """
    eye = cam["look"] - cam["dist"] * cam["fwd"]
    scene = trimesh.util.concatenate([part_world] + list(solids))
    out = np.zeros(len(solids), bool)
    for j, s in enumerate(solids):
        v = np.asarray(s.vertices)
        d = v - eye
        n = np.linalg.norm(d, axis=1)
        loc, ray, _ = scene.ray.intersects_location(
            np.tile(eye, (len(v), 1)), d / n[:, None], multiple_hits=False)
        reach = np.full(len(v), np.inf)
        reach[ray] = np.linalg.norm(loc - eye, axis=1)
        out[j] = bool((reach >= n - 5e-4).any())
    return out


# --------------------------------------------------------------- the figure ---

def gain_bar(dr, x, y, w, h, before, after, font):
    """How much of the requirement THIS point brought in, as a slice of the whole.

    The percentage in the panel title is cumulative, and cumulative numbers hide
    the thing the reader is being asked to watch: a point that takes the answer
    from 61 to 100 did more than one that took it from 0 to 26, and two titles
    reading `61%` and `26%` say the opposite. The bar puts the increment back.
    """
    dr.rectangle([x, y, x + w, y + h], fill="#eeece6", outline="#d5d2c9")
    dr.rectangle([x, y, x + w * before, y + h], fill=HELD)
    dr.rectangle([x + w * before, y, x + w * after, y + h], fill=GAIN)
    lab = f"+{100 * (after - before):.0f}"
    if w * (after - before) > 2.2 * h:
        dr.text((x + w * (before + after) / 2, y + h / 2), lab, fill="white",
                font=font, anchor="mm")
    else:            # a contact that added nothing has no slice to write inside
        dr.text((x + w * after + h * 0.4, y + h / 2), lab, fill=(110, 110, 106),
                font=font, anchor="lm")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("object")
    ap.add_argument("--pose", type=int, default=6)
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--size", type=int, default=840)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    px = args.size
    d = obj_path(args.object)
    out_dir = d / "pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = d / "_pipe_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)

    r = replicate(args.object, args.pose, args.k, args.seed)
    part, T, inside = r["part"], r["T"], r["inside"]
    targets, pu, pw = r["targets"], r["pu"], r["pw"]
    avail, chosen, pts, us = r["avail"], r["chosen"], r["pts"], r["us"]
    ico, tiles, tree = r["ico"], r["tiles"], r["tree"]
    adjacency = neighbours(ico)
    mag = np.linalg.norm(targets, axis=1)
    unit = targets / mag[:, None]
    frac, gain = steps_of(targets, avail, chosen)
    print(f"pose {args.pose}: replication checks out against capped_k{args.k:g}.json "
          f"({len(targets)} requirements, {len(avail)} available pushes, "
          f"{len(chosen)} supports)")
    print("  per contact: " + "  ".join(
        f"{i + 1}: {100 * frac[i]:.0f}->{100 * frac[i + 1]:.0f}% (+{100 * gain[i]:.0f})"
        for i in range(len(chosen))))

    # ---- the spheres. A force is drawn WHERE IT COMES FROM, at -F, arrow running
    # into the centre, so the floor's G sits at the south pole where the floor is.
    h_job, h_back, h_want = tree.query(-pu)[1], tree.query(pu)[1], tree.query(-unit)[1]
    job = np.flatnonzero(sheet(adjacency, h_job))
    back = np.flatnonzero(sheet(adjacency, h_back))
    want = sheet(adjacency, h_want)
    near = cKDTree(-unit).query(tiles)[1]
    band = np.flatnonzero(sheet(adjacency, tree.query(-avail)[1]))
    few = np.random.default_rng(11).choice(len(pu), min(26, len(pu)), replace=False)
    fewt = np.random.default_rng(14).choice(len(unit), min(26, len(unit)), replace=False)
    span = compass(targets)

    top = [([(job, PUSH_BLUE, .95)], [], [(pu[few], PUSH_BLUE, False)], [],
            "1  the job\nevery way the process can push"),
           ([(back, NEED, .95)], [], [(-pu[few], NEED, False)], [],
            "2  reverse it\nwhat has to be produced"),
           ([(np.flatnonzero(want), NEED, .95)], [], [(unit[fewt], NEED, False)],
            [(-UP, FLOOR_MARK, "G")],
            f"3  + its own weight = what is owed\n{span:.0f}° wide, |T| up to {mag.max():.2f}w"),
           ([(band, HAVE, .95)], [(-avail, HAVE, 26, True)], [(avail, HAVE, False)],
            [(-UP, FLOOR_MARK, "G")],
            f"4  what we have\n{len(avail)} pushes, 1w each")]
    # the floor on its own closes band A and opens band B: it is the same panel
    # the contacts get, drawn with no contacts, so the reader sees the field start
    rho0 = radial(UP.reshape(3, 1), -tiles)
    top.append((shades(rho0) + [(np.flatnonzero(want), NEED, .95)], [], [],
                [(-UP, FLOOR_MARK, "G")],
                "5  the floor alone\n0% done  ·  it only points up"))

    bottom = []
    for i in range(1, len(chosen) + 1):
        Gi = np.column_stack([UP] + [avail[c] for c in chosen[:i]])
        rho = radial(Gi, -tiles)
        sti = answered(Gi, targets)
        marks = [(-UP, FLOOR_MARK, "G")]
        marks += [(-avail[chosen[j]], STEP_INK, str(j + 1)) for j in range(i)]
        # what is still owed is owed for one of two reasons and under a cap they
        # are the whole distinction: deep red is a direction no mixture points in,
        # violet is one that IS available and simply not strong enough
        owed = want & ~sti[near]
        fills = shades(rho)
        fills += [(np.flatnonzero(want & sti[near]), DONE, .95),
                  (np.flatnonzero(owed & (rho > 0)), SHORT, .95),
                  (np.flatnonzero(owed & (rho <= 0)), NEED, .95)]
        arr = [(unit[fewt][~sti[fewt]], NEED, False)] if not sti.all() else []
        lam = least_effort(Gi, targets[sti]) if sti.any() else None
        # the first contact answers nothing and that is not a defect of the choice:
        # `up` and one direction span a PLANE, and a plane has no volume, so no
        # single contact can ever score. Saying so in the panel, because a reader
        # who is not told reads the first one as a bad pick. Two lines and no more:
        # a third overflows the axes and gets clipped.
        title = ("contact 1  ·  0% done\nup + one push spans a plane" if i == 1 else
                 f"contact {i}\n{100 * frac[i]:.0f}% done")
        if i > 1 and lam is not None and len(lam) and np.isfinite(lam).any():
            title += f"  ·  peak {np.nanmax(lam):.2f} w"
        bottom.append((fills, [], arr, marks, title))
    # the field on its own, the requirement only outlined, so the strength UNDER
    # the requirement can be read instead of being painted over
    rho = radial(r["G"], -tiles)
    st = answered(r["G"], targets)
    inner = want.copy()
    for _ in range(2):
        inner &= ~((adjacency @ ~inner) > 0)
    tight = int(np.argmin(radial(r["G"], unit) / mag))
    head = float(np.min(radial(r["G"], unit) / mag))
    bottom.append((shades(rho) + [(np.flatnonzero(want & ~inner), INK, .95),
                                  (np.flatnonzero(want & ~st[near]), NEED, .95)],
                   [], [], [(-UP, FLOOR_MARK, "G"), (-unit[tight], "#B02A26", "tight")],
                   f"the strength field\nasks {mag.max():.2f}w · margin {head:.2f}"))

    # ---- the renders under the spheres, one camera throughout, as cover.shot uses
    parts = paint(part, T, inside, set(), tmp, f"pipe_p{args.pose}", rel="_pipe_tmp")
    blue = [(pw[j], pu[j]) for j in few]
    # every band-A panel carries the pushes, including 'the job' -- a panel whose
    # title says `every way the process can push` over a part with no arrows on it
    # is the one pairing a reader will not forgive
    forces = shot(args.object, T, parts, px, blue, [], triad=False)
    xy = screen(r["mesh"], T, px, pts)
    top_shots = [forces] * len(top)
    bot_shots = [number(shot(args.object, T, parts, px, blue,
                             [(pts[j], us[j]) for j in range(i)], triad=False),
                        xy[:i], range(1, i + 1), px)
                 for i in range(1, len(chosen) + 1)]
    bot_shots.append(bot_shots[-1])

    # ---- the two globe bands, spheres over the part they belong to
    font = _font(int(px * 0.038))
    small = _font(int(px * 0.026))
    tiny = _font(int(px * 0.032))
    blocks = []
    for panels, sh in [(top, top_shots), (bottom, bot_shots)]:
        g = globe_png(ico, panels, px)
        gw = g.size[0] // len(panels)
        gh = int(g.size[1] * 0.99)
        blk = Image.new("RGB", (gw * len(panels), gh + gw), "white")
        blk.paste(g.crop((0, 0, g.size[0], gh)), (0, 0))
        for i, im in enumerate(sh):
            blk.paste(im.resize((gw, gw)), (i * gw, gh))
        blocks.append(blk)
    gw = blocks[0].size[0] // len(top)
    wide = blocks[0].size[0]

    # the increment bars get their own strip under the step spheres. Inside the
    # sphere panel they would land on the floor's G marker, and the one thing
    # they must not do is read as part of the ball.
    bh = int(gw * 0.075)
    bars = Image.new("RGB", (wide, int(bh * 2.6)), "white")
    dbr = ImageDraw.Draw(bars)
    dbr.text((int(gw * 0.06), int(bh * 0.35)), "what this contact added",
             fill=(120, 120, 116), font=small)
    for i in range(len(chosen)):
        gain_bar(dbr, i * gw + int(gw * 0.10), int(bh * 1.2), int(gw * 0.80), bh,
                 frac[i], frac[i + 1], tiny)

    # ---- band C: the same contacts as walls, from two sides, so none is hidden
    world = r["mesh"].copy()
    world.apply_transform(T)
    walls, wall_meshes, drafts = [], [], []
    for i in range(len(chosen)):
        w, dr_, vol = fit_wall(pts[i], us[i], world)
        w.export(tmp / f"wall{i}.stl")
        walls.append((f"_pipe_tmp/wall{i}.stl", "propA"))
        wall_meshes.append(w)
        drafts.append(dr_)
        print(f"  wall {i + 1}: {1000 * (pts[i][1] - FLOOR):5.1f} mm tall, "
              f"{w.volume * 1e6:5.2f} cm3, draft {dr_:.1f} deg, "
              f"inside the part {vol * 1e9:.4f} mm3")
    # frame the part AND the fixture: the walls stand beside it and the plate in
    # figure 2 runs further still, so the part's own box is not the subject
    V = np.vstack([world.vertices] + [w.vertices for w in wall_meshes])
    lo, hi = V.min(axis=0), V.max(axis=0)
    sw = (wide - 10) // 2
    sh_ = int(sw * 0.70)
    labh = int(px * 0.085)
    strip = Image.new("RGB", (wide, sh_ + labh), "white")
    dsr = ImageDraw.Draw(strip)
    for i, (az, el, lab) in enumerate(
            [(118.0, -18.0, "5   the support   ·   from the front"),
             (298.0, -18.0, "the same four walls, from behind   ·   a pale number "
                            "means that wall is hidden in this view")]):
        cam = camera(lo, hi, az, el, zoom=1.14, lift=0.40)
        im = solid_shot(args.object, T, parts, walls, cam, sw, sh_)
        number(im, project(cam, sw, sh_, pts), range(1, len(chosen) + 1), sw,
               size=0.055, seen=visible(cam, world, wall_meshes))
        strip.paste(im, (i * (sw + 10), labh))
        dsr.text((i * (sw + 10) + sw // 2, int(labh * 0.45)), lab, fill=(40, 40, 38),
                 font=_font(int(px * 0.046)), anchor="mm")

    # ---- the page. Three lines, because one long enough to name every colour is
    # wider than the panels and would set the page width on its own
    caption = (
        f"{args.object}  pose {args.pose}   ·   a force is drawn WHERE IT COMES FROM, "
        f"at -F, with its arrow running into the centre; the floor sits at the south pole\n"
        f"BLUE the disturbances  ·  RED what is owed  ·  GREY answered  ·  VIOLET that "
        f"way is available but not hard enough  ·  ORANGE a push this workpiece can supply\n"
        f"TEAL how much force the chosen pushes can put that way, STONE none   ·   "
        f"G the floor, 1 2 3 4 the supports   ·   the walls bear on the tangent plane, "
        f"so a wall's face IS its push direction")
    headh = int(px * 0.20)
    capw = ImageDraw.Draw(Image.new("RGB", (1, 1))).multiline_textbbox(
        (16, 0), caption, font=font)[2] + 16
    page = Image.new("RGB", (max(wide, capw + int(px * 0.45)),
                             headh + blocks[0].size[1] + blocks[1].size[1]
                             + bars.size[1] + strip.size[1]), "white")
    page.paste(blocks[0], (0, headh))
    y = headh + blocks[0].size[1]
    page.paste(blocks[1], (0, y))
    # the bars sit under the step block's own part renders, not under the spheres,
    # so the column they belong to is unambiguous
    y += blocks[1].size[1]
    page.paste(bars, (0, y))
    page.paste(strip, (0, y + bars.size[1]))
    dr = ImageDraw.Draw(page)
    dr.text((16, int(headh * .12)), caption, fill=(90, 90, 90), font=font)
    legend(dr, capw + int(px * 0.05), int(headh * .34), int(px * 0.038), small)
    out = out_dir / f"pipeline_pose{args.pose}.png"
    page.save(out)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"  {out}  {page.size[0]}x{page.size[1]}")


if __name__ == "__main__":
    main()
