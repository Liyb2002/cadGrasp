"""One page from the disturbance to a printable support, for the sharp cube.

`capped_cuboid.py` answers a question about DIRECTIONS and stops there. It ends
holding a list of unit vectors and the faces they were taken from, which is a
proof and not a fixture. This walks the same pose the rest of the way -- the job,
what is owed, what the workpiece can supply, each support as it goes in, and then
the same contacts as solid pads standing on the floor -- so that direction,
contact point and printed thing can be followed across one sheet.

Nothing in the search moves. The region, the pushes, the candidate directions and
the exhaustive subset search are `capped_cuboid`'s, seed for seed, and the
figure's numbers are checked against `capped/capped_k1.json` before anything is
drawn. Three things are added.

WHY THE CUBE IS THE BASELINE. This part has no fillets at all, so the entire
universe of push directions is six face normals: six isolated points on the ball
and no band anywhere. Nothing can be traded off, nothing can be nudged a few
degrees, and a work region that swallows a face destroys a direction outright.
That is the whole reason the cube needs five or six supports where a filleted
part needs three, and the "what we have" panel is where it has to be visible.

PER-POINT COVERAGE. The obvious question -- how much does each support cover? --
has a startling answer here, and it is the reason this figure exists. With the
floor and ONE support the reachable set is a plane through the origin, and the
requirement is a solid patch, so a plane covers none of it: every candidate on
this part, alone with the floor, answers 0.0%. Coverage is not a property a point
has, it is a property a SET has. So each point gets two honest numbers instead:
what it ADDED the moment it went in, painted dark on the ball, and what the
finished design LOSES if it is taken out again. They disagree, and the ledger
panel shows where -- the support that adds almost nothing on entry can still be
the one without which the design falls to 98%.

THE SUPPORT AS A SOLID. `demo_solid.py` drops a column straight down from each
pad. On this pose that is wrong for four of the five contacts, and for one of
them it is badly wrong: the cube is tipped, so two of its faces OVERHANG, and a
column dropped from a pad on an overhanging face passes straight through the
workpiece. The support drawn here is therefore a pad, a shank retreating along
the push axis in pure compression, and a vertical post from there to the floor,
with the shank made exactly long enough that the post clears the part -- measured,
not assumed, by an exact box-versus-box separation test.

    python slides/tools/pipeline_cuboid.py --poses 0
    python slides/tools/pipeline_cuboid.py --poses 0 2
"""
from __future__ import annotations

import argparse
import io
import shutil

import matplotlib
import mujoco
from mujoco import Renderer
import numpy as np
import coordinates as COORD
import trimesh
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402

from capped_cuboid import (CAP, LEVELS, MARK, WEAK, bands, capacity, cone,   # noqa: E402
                           effort, exhaustive, feasible, screen)
from common import mat_to_quat_wxyz, obj_path, read_json, write_json  # noqa: E402
from cover import (AZIM, DONE, ELEV, HAVE, INK, MUTED, NEED, PAPER,   # noqa: E402
                   PUSH_BLUE, UP, globe_png, neighbours, paint, sheet, shot, tiling)
from disturbances import _font                             # noqa: E402
from reach import FLOOR_MARK, compass                      # noqa: E402
from shrink_support import angled_pushes                   # noqa: E402
from supports import CONTACT_EPS                           # noqa: E402
from work_regions import BOUNDARY_EDGE, spot_region        # noqa: E402

# Not a fourth state: a subdivision of the grey one, marking the part of
# "answered" that the support just added. It wants to be the same grey turned
# dark, and that was tried first -- but the requirement sits on the FAR side of
# the ball and is read through the near hemisphere and the capacity haze, which
# compresses value differences until two greys are one grey. A hue survives that
# and a value does not, so gold it is, and the caption says what it is a part of.
FRESH = "#E0A526"

PAD_W = 0.007          # half-width of the square pad, 14 mm across the flats
PAD_T = 0.003          # how thick the pad is behind the contact
POST_W = 0.004         # half-width of the shank and of the post
CLEAR = 0.002          # how far a post must stay off the workpiece
FLOOR = 0.0
PROP = "0.90 0.45 0.12 1"

SOLID_SCENE = """<mujoco>
  <compiler meshdir="." angle="radian"/>
  <visual>
    <global offwidth="1600" offheight="1600" fovy="45"/>
    <quality shadowsize="4096" offsamples="8"/>
    <map znear="0.005" zfar="30"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.45 0.45 0.45" specular="0.06 0.06 0.06"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="1 1 1"
             width="256" height="256"/>
    <material name="floor" rgba="0.97 0.97 0.97 1"/>
    <material name="work" rgba="0.55 0.82 0.62 1" specular="0.1"/>
    <material name="rest" rgba="0.90 0.89 0.85 1" specular="0.1"/>
    <material name="spare" rgba="0.90 0.89 0.85 1" specular="0.1"/>
{mats}
{assets}
  </asset>
  <worldbody>
    <light pos="0.4 1.2 -0.5" dir="-0.3 -1 0.4" directional="true" castshadow="true"
           diffuse="0.55 0.55 0.55"/>
    <geom type="plane" quat="0.7071067811865476 -0.7071067811865476 0 0" size="2 2 0.05" material="floor"/>
    <body pos="{pos}" quat="{quat}">
{geoms}
    </body>
{props}
  </worldbody>
</mujoco>
"""


# --------------------------------------------------------------- the solids ---

def obb(c, R, h):
    """An oriented box: centre, a frame in its columns, half-extents along it."""
    return {"c": np.asarray(c, float), "R": np.asarray(R, float),
            "h": np.asarray(h, float)}


def obb_mesh(b):
    s = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
    return trimesh.convex.convex_hull(b["c"] + (s * b["h"]) @ b["R"].T)


def separation(a, b):
    """Exact signed separation of two oriented boxes: >0 a gap, <0 the overlap.

    The question a picture cannot answer -- does this support pass through the
    workpiece -- and the reason it is answered this way rather than with a mesh
    boolean is that a boolean needs a backend, reports only volume, and calls a
    pad resting flush on a face an intersection. Fifteen axes settle two boxes
    exactly: three faces each and the nine edge pairs. Every piece built here is
    a box and so is this workpiece, so nothing is approximated anywhere.
    """
    axes = [a["R"][:, i] for i in range(3)] + [b["R"][:, i] for i in range(3)]
    for i in range(3):
        for j in range(3):
            v = np.cross(a["R"][:, i], b["R"][:, j])
            n = np.linalg.norm(v)
            if n > 1e-9:
                axes.append(v / n)
    best = -np.inf
    for n in axes:
        ra = float(np.abs(a["R"].T @ n) @ a["h"])
        rb = float(np.abs(b["R"].T @ n) @ b["h"])
        best = max(best, abs(float(n @ (b["c"] - a["c"]))) - ra - rb)
        if best > 0:
            return best                       # one separating axis is a proof
    return best


def frame(u):
    """A right-handed frame whose third axis is u."""
    u = np.asarray(u, float) / np.linalg.norm(u)
    a = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(a, u)
    e1 /= np.linalg.norm(e1)
    return np.column_stack([e1, np.cross(u, e1), u])


def bracket(p, u, part, clear=CLEAR, reach=0.15, step=0.0005):
    """Pad on the contact, shank back along the push, post down to the floor.

    The pad's face IS the contact plane, so its normal is the push the search
    chose and it rests flush. The shank retreats along -push and nothing else,
    because that is the load path: the pad is pressed straight into the shank and
    the shank works in pure compression. How far it retreats is not a taste --
    it is the shortest retreat for which a vertical post clears the workpiece,
    found by the separation test above. On a face that overhangs, that distance
    is what turns a column into a bracket; on a vertical face it is barely the
    post's own half-width; under the part it is zero.
    """
    u = np.asarray(u, float) / np.linalg.norm(u)
    F, I = frame(u), np.eye(3)
    pad = obb(p - 0.5 * PAD_T * u, F, [PAD_W, PAD_W, 0.5 * PAD_T])
    s, post, f = 0.0, None, None
    for s in np.arange(0.0, reach, step):
        f = p - (PAD_T + s) * u
        top = max(f[2], 1e-4)
        post = obb([f[0], f[1], 0.5 * top], I, [POST_W, POST_W, 0.5 * top])
        if separation(post, part) >= clear:
            break
    out = [pad]
    if s > 1e-9:
        out.append(obb(p - (PAD_T + 0.5 * s) * u, F, [POST_W, POST_W, 0.5 * s]))
    return out + [post], float(s), f


def part_box(T, mesh):
    """This workpiece as one oriented box -- which, being a cube, it exactly is."""
    return obb(T[:3, 3], T[:3, :3], mesh.extents / 2)


# --------------------------------------------------------------- rendering ---

def camera(lookat, span, azim, elev, px, fovy=45.0):
    """A mujoco camera and the pixel projection that matches it.

    The numbers on the renders have to land on the contacts they name, and mujoco
    hands back no camera matrix, so the projection is rebuilt from the same three
    numbers the camera was built from and returned alongside it.
    """
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation = azim, elev
    cam.lookat[:] = lookat
    fov = np.tan(np.deg2rad(fovy / 2))
    cam.distance = span / 2 / fov
    a, e = np.deg2rad(azim), np.deg2rad(elev)
    fwd = np.array([np.cos(e) * np.cos(a), np.sin(e), np.cos(e) * np.sin(a)])
    right = np.array([np.sin(a), 0.0, -np.cos(a)])
    up = -np.cross(right, fwd)
    eye = np.asarray(lookat, float) - cam.distance * fwd

    def project(pts):
        v = np.asarray(pts, float).reshape(-1, 3) - eye
        s = np.stack([v @ right, v @ up], 1) / (v @ fwd)[:, None] / fov
        return np.stack([(0.5 + 0.5 * s[:, 0]) * px, (0.5 - 0.5 * s[:, 1]) * px], 1)

    return cam, project


def solid_shot(name, T, parts, solids, px, cam, tmp, rel, tag):
    """The part, the floor and a list of oriented boxes, in one render."""
    obj = obj_path(name)
    mats, assets, props = [], [], []
    for i, (b, colour) in enumerate(solids):
        obb_mesh(b).export(tmp / f"{tag}_{i}.stl")
        mats.append(f'    <material name="m{i}" rgba="{colour}" specular="0.25" '
                    f'shininess="0.35"/>')
        assets.append(f'    <mesh name="s{i}" file="{rel}/{tag}_{i}.stl"/>')
        props.append(f'    <geom type="mesh" mesh="s{i}" material="m{i}"/>')
    xml = SOLID_SCENE.format(
        mats="\n".join(mats),
        assets="\n".join(assets)
        + "\n" + "\n".join(f'    <mesh name="p{i}" file="{f}"/>'
                           for i, (f, _) in enumerate(parts)),
        geoms="\n".join(f'      <geom type="mesh" mesh="p{i}" material="{m}"/>'
                        for i, (_, m) in enumerate(parts)),
        props="\n".join(props),
        pos=" ".join(f"{v:.9g}" for v in T[:3, 3]),
        quat=" ".join(f"{v:.9g}" for v in mat_to_quat_wxyz(T[:3, :3])))
    f = obj / f".{tag}.xml"
    f.write_text(xml)
    try:
        model = mujoco.MjModel.from_xml_path(str(f))
    finally:
        f.unlink(missing_ok=True)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with Renderer(model, px, px, max_geom=3000) as r:
        r.update_scene(data, camera=cam)
        return Image.fromarray(r.render())


def number(im, xy, labels, px, colour=MARK):
    dr = ImageDraw.Draw(im)
    f = _font(int(px * 0.052))
    for (x, y), lab in zip(xy, labels):
        dr.text((x, y), lab, fill="white", font=f, anchor="mm",
                stroke_width=4, stroke_fill=colour)
    return im


def ledger(entered, lost, px, note):
    """What each point is worth, twice over, because the two answers differ.

    A grouped bar and not a table: the eye has to see that the support which
    ADDED almost nothing is not the support the design can spare, and two numbers
    side by side per row say that in one glance. The solo figure is not plotted
    because it is identically zero and a row of zeros is not a chart -- it is
    written above the axes instead, where a fact of that size belongs.
    """
    fig = plt.figure(figsize=(px / 200, px / 200), dpi=200, facecolor=PAPER)
    ax = fig.add_axes([0.11, 0.09, 0.85, 0.55], facecolor=PAPER)
    y = np.arange(len(entered))[::-1]
    ax.barh(y + 0.19, 100 * np.asarray(entered), 0.34, color=FRESH,
            label="added the moment it went in")
    ax.barh(y - 0.19, 100 * np.asarray(lost), 0.34, facecolor="none",
            edgecolor=MARK, linewidth=1.6, label="lost if it is taken out again")
    for i, (a, b) in enumerate(zip(entered, lost)):
        for v, dy in ((a, 0.19), (b, -0.19)):
            ax.text(100 * v + 1.2, y[i] + dy, f"{100 * v:.0f}", va="center",
                    ha="left", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([str(i + 1) for i in range(len(entered))], fontsize=11, color=MARK,
                       fontweight="bold")
    ax.set_xlim(0, 112)
    ax.set_ylim(-0.7, len(entered) - 0.3)
    ax.set_xlabel("per cent of the requirement", fontsize=9, color=MUTED)
    ax.tick_params(labelsize=8, colors=MUTED, length=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    # above the axes, not inside them: the top bar is the shortest, so a legend
    # parked in the plot would sit on the one row a reader looks at first
    ax.legend(fontsize=9, frameon=False, labelcolor=INK, loc="lower left",
              bbox_to_anchor=(0.0, 1.01), ncol=1, handlelength=1.6)
    fig.text(0.05, 0.955, "what each point covers", fontsize=15, color=INK, va="top")
    fig.text(0.05, 0.905, note, fontsize=9.5, color=MUTED, va="top", linespacing=1.45)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=PAPER)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB").resize((px, px))


# ------------------------------------------------------------- the pipeline ---

def solve(name, pose, k, seed, n_points, n_dirs, mesh, examples, ico, tiles, tree):
    """capped_cuboid's pipeline, unchanged, up to the chosen supports."""
    T = np.asarray(examples[pose]["T_world_mesh"])
    R, t = T[:3, :3], T[:3, 3]
    rng = np.random.default_rng(seed + 1000 * pose)
    reg = spot_region(mesh, T, rng, float(rng.uniform(0.17, 0.33)),
                      BOUNDARY_EDGE * float(np.linalg.norm(mesh.extents)))
    part, inside = reg.pop("mesh"), reg.pop("mask")
    area = float(part.area_faces[inside].sum() / part.area)

    pw, pu = angled_pushes(part, T, inside, n_points, n_dirs, seed + pose)
    targets = UP - k * pu
    keep = np.linalg.norm(targets, axis=1) > 1e-9
    targets, pw, pu = targets[keep], pw[keep], pu[keep]

    on_floor = (part.triangles_center @ R.T + t)[:, 2] <= CONTACT_EPS
    off = np.flatnonzero(~inside & ~on_floor)
    push = -(part.face_normals[off] @ R.T)
    push /= np.linalg.norm(push, axis=1, keepdims=True)
    hit = tree.query(push)[1]
    first = {}
    for j, h in enumerate(hit):
        first.setdefault(h, j)
    rep = np.array([first[h] for h in sorted(first)])
    avail, avail_face = push[rep], off[rep]

    chosen, frac, ceiling = exhaustive(targets, avail)
    return {"T": T, "part": part, "inside": inside, "area": area, "pw": pw, "pu": pu,
            "targets": targets, "avail": avail, "chosen": chosen, "frac": frac,
            "ceiling": ceiling, "span": compass(targets),
            "pts": part.triangles_center[avail_face[chosen]] @ R.T + t,
            "us": avail[chosen]}


def check(name, pose, S, rec):
    """Refuse to draw a figure whose numbers are not the recorded ones.

    An earlier figure in this project was replicated with the wrong seed and
    nothing caught it, because a picture cannot be wrong out loud. These four
    numbers pin the region, the sample and the search, so if they agree the rest
    of the pipeline is the same pipeline.
    """
    want = {"n_targets": len(S["targets"]), "region_area_fraction": S["area"],
            "n_available": len(S["avail"]), "n_supports": len(S["chosen"]),
            "answered_fraction": S["frac"], "ceiling_fraction": S["ceiling"]}
    bad = [f"{key}: mine {v!r} vs recorded {rec[key]!r}" for key, v in want.items()
           if not np.isclose(v, rec[key], rtol=0, atol=1e-9)]
    if bad:
        raise SystemExit(f"{name} pose {pose} does not replicate capped_k1.json:\n  "
                         + "\n  ".join(bad))
    print(f"pose {pose}: replicates capped_k1.json "
          f"({want['n_targets']} targets, region {want['region_area_fraction']:.6f}, "
          f"{want['n_available']} candidates, {want['n_supports']} supports, "
          f"{100 * want['answered_fraction']:.1f}%)", flush=True)


def page(name, pose, S, px, tmp, rel, out_dir):
    part, T, inside = S["part"], S["T"], S["inside"]
    targets, avail, chosen = S["targets"], S["avail"], S["chosen"]
    pts, us, pu = S["pts"], S["us"], S["pu"]
    ico, tiles = tiling(4)
    tree, adjacency = cKDTree(tiles), neighbours(ico)
    unit = targets / np.linalg.norm(targets, axis=1, keepdims=True)

    # ---- the four setup panels, verbatim from capped_cuboid so the two figures
    # cannot drift apart; only the "what we have" title is rewritten, because on
    # this part the count of directions IS the finding
    h_job, h_back, h_want = tree.query(-pu)[1], tree.query(pu)[1], tree.query(-unit)[1]
    job = np.flatnonzero(sheet(adjacency, h_job))
    back = np.flatnonzero(sheet(adjacency, h_back))
    want = sheet(adjacency, h_want)
    near = cKDTree(-unit).query(tiles)[1]
    band = np.flatnonzero(sheet(adjacency, tree.query(-avail)[1]))
    few = np.random.default_rng(11).choice(len(pu), min(26, len(pu)), replace=False)
    fewt = np.random.default_rng(14).choice(len(unit), min(26, len(unit)), replace=False)
    need_mag = float(np.linalg.norm(targets, axis=1).max())
    faces = len(np.unique(np.round(avail, 3), axis=0))

    panels = [([(job, PUSH_BLUE, .95)], [], [(pu[few], PUSH_BLUE, False)], [], "the job"),
              ([(back, NEED, .95)], [], [(-pu[few], NEED, False)], [],
               "reverse the pushes\nthe far side, sideways in full"),
              ([(np.flatnonzero(want), NEED, .95)], [], [(unit[fewt], NEED, False)],
               [(-UP, FLOOR_MARK, "G")],
               f"+ its own weight\n{S['span']:.0f}° wide, up to {need_mag:.2f} W"),
              ([(band, HAVE, .95)], [(-avail, HAVE, 26, True)], [(avail, HAVE, False)],
               [(-UP, FLOOR_MARK, "G")],
               f"what we have\n{faces} of 6 faces, no band at all")]

    # ---- the steps. The three states are settled; what is added is a fourth
    # SHADE, not a fourth state: the tiles this support answered that the
    # previous ones did not. Coverage on this part is joint, so the increment is
    # the only per-point number that is ever non-zero, and it has to be visible.
    solo = [float(feasible([avail[c]], targets).mean()) for c in chosen]
    entered, lost, cum = [], [], []
    prev = feasible([], targets)
    for i, c in enumerate(chosen):
        ok = feasible([avail[j] for j in chosen[:i + 1]], targets)
        entered.append(float((ok & ~prev).mean()))
        cum.append(float(ok.mean()))
        lost.append(float(S["frac"] - feasible([avail[j] for j in chosen if j != c],
                                               targets).mean()))
        prev = ok
    lam = np.zeros((len(targets), 0))
    prev = feasible([], targets)
    for i in range(len(chosen) + 1):
        gens = [avail[c] for c in chosen[:i]]
        field = capacity(gens, -tiles)
        ok = feasible(gens, targets)
        inreach = cone(gens, targets)
        lam = effort(gens, targets) if i else lam
        marks = [(-UP, FLOOR_MARK, "G")]
        marks += [(-avail[chosen[j]], MARK, str(j + 1)) for j in range(i)]
        new = ok & ~prev
        fills = bands(field, want)
        fills += [(np.flatnonzero(want & ok[near] & ~new[near]), DONE, .95),
                  (np.flatnonzero(want & new[near]), FRESH, .95),
                  (np.flatnonzero(want & inreach[near] & ~ok[near]), WEAK, .95),
                  (np.flatnonzero(want & ~inreach[near]), NEED, .95)]
        arr = [(unit[fewt][inreach[fewt] & ~ok[fewt]], WEAK, False),
               (unit[fewt][~inreach[fewt]], NEED, False)]
        # two lines and no more: a panel is three inches wide and a third line
        # runs up into the row above it
        title = (f"floor only\n{100 * ok.mean():.0f}% done" if i == 0 else
                 f"+{i}   {100 * ok.mean():.0f}% done\n"
                 f"this point +{100 * entered[i-1]:.0f}"
                 + (f"  ·  peak {np.nanmax(lam):.2f} W" if ok.any() else ""))
        panels.append((fills, [], arr, marks, title))
        prev = ok
    peak = (np.nanmax(lam, axis=0) if len(chosen) and np.isfinite(lam).any()
            else np.zeros(len(chosen)))

    # ---- when a pose cannot be finished, say what stopped it rather than
    # leaving a red patch to be interpreted. The candidate list is the faces the
    # work region left behind, so the test is direct: hand the design back the
    # normals the region swallowed and see whether the shortfall goes away. If it
    # does, the pose was not hard, it was ROBBED, and that is a different finding.
    done_ok = feasible([avail[c] for c in chosen], targets)
    done_reach = cone([avail[c] for c in chosen], targets)
    red = float((~done_reach).mean())
    violet = float((done_reach & ~done_ok).mean())
    allpush = np.unique(np.round(-(mesh.face_normals @ T[:3, :3].T), 6), axis=0)
    eaten = [u for u in allpush
             if np.min(np.linalg.norm(avail - u, axis=1)) > 1e-3]
    restored = (float(feasible(list(us) + eaten, targets).mean()) if eaten
                else float(S["frac"]))

    # ---- the renders that sit under the globes, exactly capped_cuboid's
    parts = paint(part, T, inside, set(), tmp, f"pipe_p{pose}", rel=rel)
    blue = [(S["pw"][j], pu[j]) for j in few]
    bare = shot(name, T, parts, px, [], [], triad=False)
    forces = shot(name, T, parts, px, blue, [], triad=False)
    shots = [bare, forces, forces, forces]
    mesh = trimesh.load(obj_path(name) / "mesh.stl", force="mesh")
    xy = screen(mesh, T, px, pts) if len(chosen) else np.zeros((0, 2))
    for i in range(len(chosen) + 1):
        im = shot(name, T, parts, px, blue, [(pts[j], us[j]) for j in range(i)],
                  triad=False)
        shots.append(number(im, xy[:i], [str(j + 1) for j in range(i)], px))

    # ---- and the step the search never took: the same contacts as solids
    box = part_box(T, mesh)
    solids, per_support, stand, gaps, touch, floor_z = [], [], [], [], [], []
    for i in range(len(chosen)):
        pieces, s, _ = bracket(pts[i], us[i], box)
        solids += [(b, PROP) for b in pieces]
        per_support.append([(b, PROP) for b in pieces])
        stand.append(s)
        # the pad is MEANT to touch, so it is excluded from the clearance: what
        # has to be shown clear is everything that only gets the load to the
        # ground. Its own separation is reported instead, and should be zero.
        touch.append(separation(pieces[0], box))
        gaps.append(min(separation(b, box) for b in pieces[1:]))
        # the post always ends on the floor, so the only piece whose height off
        # the ground says anything is the pad: on a contact tucked under the part
        # it is the pad, not the post, that has to fit in the gap
        floor_z.append(float((pieces[0]["c"] - np.abs(pieces[0]["R"]) @ pieces[0]["h"])[2]))
    # what demo_solid would have built: the same pad, a post dropped straight down
    naive = []
    for i in range(len(chosen)):
        q = pts[i] - PAD_T * us[i]
        top = max(q[2], 1e-4)
        naive.append(separation(obb([q[0], q[1], 0.5 * top], np.eye(3),
                                    [POST_W, POST_W, 0.5 * top]), box))

    V = np.asarray(mesh.vertices) @ T[:3, :3].T + T[:3, 3]
    lo, hi = V.min(axis=0), V.max(axis=0)
    mid = COORD.lift_floor(COORD.floor(lo+hi)/2, .46*hi[2])
    span = 1.34 * max(float(np.linalg.norm(COORD.floor(hi-lo))), float(hi[2]))
    # the close-up goes to the pad squeezed nearest the ground, because that is
    # the one a reader will not believe until they see it. Seen ACROSS the wedge
    # rather than into it: the gap under a tipped part is a triangle, and a
    # triangle only reads as one when the eye is level with it and side on, so
    # the camera looks along the horizontal perpendicular to the way it opens.
    # The close-up looks straight INTO the wedge, from the side it opens on, with
    # the eye a hair above the pad. Anywhere else and the workpiece is in the
    # way: side on, its own near face hides the gap entirely, and from above
    # there is no gap to see. Everything but that one support is left out for the
    # same reason -- the other posts stand between the eye and it.
    tight = int(np.argmin(pts[:, 2])) if len(pts) else 0
    at = pts[tight]
    opens = -COORD.floor(us[tight])                     # the way the wedge widens
    close = 0.75 * span
    dist = close / 2 / np.tan(np.deg2rad(22.5))
    views = [(mid, span, AZIM + 180.0, -ELEV, range(len(pts)),
              "the same view as the balls"),
             (mid, span, AZIM, -ELEV, range(len(pts)), "and from behind"),
             (at, close, float(np.degrees(np.arctan2(-opens[1], -opens[0]))),
              -float(np.degrees(np.arcsin(min(1.0, at[2] / dist)))), [tight],
              f"into the gap: {tight + 1} alone, "
              f"{1000 * at[2]:.1f} mm of headroom")]
    solid_shots = []
    for j, (at, sp, az, el, show, cap_) in enumerate(views):
        cam, proj = camera(at, sp, az, el, px)
        keep = [b for i in show for b in per_support[i]]
        im = solid_shot(name, T, parts, keep, px, cam, tmp, rel, f"sol{pose}_{j}")
        im = number(im, proj(pts[list(show)]), [str(i + 1) for i in show], px)
        dr = ImageDraw.Draw(im)
        dr.text((int(px * 0.03), int(px * 0.93)), cap_, fill=(90, 90, 90),
                font=_font(int(px * 0.038)))
        solid_shots.append(im)

    # ---- the page: two blocks of globes-over-renders, then the solids
    per = (len(panels) + 1) // 2
    blocks = []
    for ps, sh_ in ((panels[:per], shots[:per]), (panels[per:], shots[per:])):
        g = globe_png(ico, ps, px)
        gw = g.size[0] // max(len(ps), 1)
        gh = int(g.size[1] * 0.99)
        blk = Image.new("RGB", (gw * len(ps), gh + gw), "white")
        blk.paste(g.crop((0, 0, g.size[0], gh)), (0, 0))
        for i, im in enumerate(sh_):
            blk.paste(im.resize((gw, gw)), (i * gw, gh))
        blocks.append(blk)

    wide = max(b.size[0] for b in blocks)
    cell = wide // 4
    note = ("alone with the floor, every one of them answers 0%:\n"
            "one support and the floor span a PLANE, and the requirement\n"
            "is solid, so a plane covers none of it. coverage is a property\n"
            "of the SET, and these are the only two honest per-point numbers.")
    head3 = int(px * 0.10)
    row = Image.new("RGB", (wide, cell + head3), "white")
    ImageDraw.Draw(row).text((16, int(head3 * 0.18)),
                             "the support:  every contact a pad on the face it "
                             "bears against, a shank along the push, a post to the floor",
                             fill=(90, 90, 90), font=_font(int(px * 0.046)))
    row.paste(ledger(entered, lost, cell, note), (0, head3))
    for i, im in enumerate(solid_shots):
        row.paste(im.resize((cell, cell)), ((i + 1) * cell, head3))
    blocks.append(row)

    caption = (
        f"{name}  pose {pose}   ·   every support pushes at most {CAP:g} body weight, "
        f"each direction is used once, the floor is unbounded\n"
        f"BLUE = the disturbances   ·   GREY = answered   ·   "
        f"GOLD = the part of the grey the support just added   ·   "
        f"RED = the direction is not there at all\n"
        f"VIOLET = the direction is there but the supports are not strong enough   ·   "
        f"ORANGE = a push this workpiece can supply   ·   G = the floor\n"
        f"TEAL = the hardest the chosen supports can push that way "
        f"(pale ≤{LEVELS[0]:g} W … dark >{LEVELS[-1]:g} W)   ·   "
        f"numbered contacts are the supports, some behind the part")
    if S["frac"] < 1 - 1e-9:
        caption += (
            f"\nTHIS POSE CANNOT BE FINISHED: it stops at {100 * S['frac']:.1f}% and "
            f"{100 * red:.1f}% of the requirement is RED — those directions are not "
            f"available at any force, so no cap and no extra support can reach them.\n"
            f"the work region swallowed {len(eaten)} of the cube's six faces; hand "
            f"{'that push' if len(eaten) == 1 else 'those pushes'} back and the same "
            f"design answers {100 * restored:.1f}%. The pose is not hard, it is robbed.")
    font = _font(int(px * 0.042))
    head = int(px * 0.26)
    wide = max(wide, ImageDraw.Draw(Image.new("RGB", (1, 1))).multiline_textbbox(
        (16, 0), caption, font=font)[2] + 16)
    sheet_ = Image.new("RGB", (wide, head + sum(b.size[1] for b in blocks)), "white")
    y = head
    for b in blocks:
        sheet_.paste(b, (0, y))
        y += b.size[1]
    ImageDraw.Draw(sheet_).text((16, int(head * .12)), caption, fill=(90, 90, 90),
                                font=font)
    out = out_dir / f"pose{pose}.png"
    sheet_.save(out)

    return out, {"pose": pose, "n_targets": int(len(targets)),
                 "region_area_fraction": S["area"], "n_available": int(len(avail)),
                 "n_supports": len(chosen), "answered_fraction": S["frac"],
                 "ceiling_fraction": S["ceiling"], "compass_span_deg": S["span"],
                 "solo_fraction": solo, "added_on_entry": entered,
                 "cumulative": cum, "lost_if_removed": lost,
                 "unanswered_red_fraction": red, "unanswered_violet_fraction": violet,
                 "faces_eaten_by_region": len(eaten),
                 "answered_if_faces_restored": restored,
                 "supports": [{"p": [float(v) for v in pts[i]],
                               "push": [float(v) for v in us[i]],
                               "peak_force": float(peak[i]),
                               "standoff_m": stand[i],
                               "pad_separation_m": touch[i],
                               "clearance_m": gaps[i],
                               "lowest_point_m": floor_z[i],
                               "straight_post_separation_m": float(naive[i])}
                              for i in range(len(chosen))]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="cuboid_baseline")
    ap.add_argument("--poses", type=int, nargs="+", default=[0])
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--size", type=int, default=840)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--push-points", type=int, default=90)
    ap.add_argument("--push-dirs", type=int, default=24)
    args = ap.parse_args()

    d = obj_path(args.object)
    mesh = trimesh.load(d / "mesh.stl", force="mesh")
    examples = read_json(d / "tips" / "tips.json")["examples"]
    rec = {r["pose"]: r for r in read_json(d / "capped" / "capped_k1.json")["poses"]}
    ico, tiles = tiling(4)
    tree = cKDTree(tiles)
    out_dir = d / "pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    rel = "_pipeline_tmp"
    tmp = d / rel
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)

    records = []
    for pose in args.poses:
        S = solve(args.object, pose, args.k, args.seed, args.push_points, args.push_dirs,
                  mesh, examples, ico, tiles, tree)
        check(args.object, pose, S, rec[pose])
        out, r = page(args.object, pose, S, args.size, tmp, rel, out_dir)
        records.append(r)
        print(f"          added on entry: "
              + ", ".join(f"{100 * v:.1f}" for v in r["added_on_entry"]))
        print(f"          lost if removed: "
              + ", ".join(f"{100 * v:.1f}" for v in r["lost_if_removed"]))
        print(f"          shortfall {100 * (1 - r['answered_fraction']):.1f}% "
              f"({100 * r['unanswered_red_fraction']:.1f}% red, "
              f"{100 * r['unanswered_violet_fraction']:.1f}% violet); the region ate "
              f"{r['faces_eaten_by_region']} face(s), restore them and the same design "
              f"answers {100 * r['answered_if_faces_restored']:.1f}%")
        for i, s in enumerate(r["supports"]):
            print(f"          {i + 1}: pad {1000 * s['p'][2]:5.1f} mm up, "
                  f"standoff {1000 * s['standoff_m']:5.1f} mm, pad touches at "
                  f"{1000 * s['pad_separation_m']:+.3f} mm, rest of it clears by "
                  f"{1000 * s['clearance_m']:+5.2f} mm, lowest point "
                  f"{1000 * s['lowest_point_m']:+5.2f} mm; a post dropped straight down "
                  f"would sit {1000 * s['straight_post_separation_m']:+6.2f} mm from the part")
        print(f"          {out}", flush=True)

    shutil.rmtree(tmp, ignore_errors=True)
    out_json = out_dir / f"pipeline_k{args.k:g}.json"
    keep = {r["pose"]: r for r in (read_json(out_json)["poses"] if out_json.exists() else [])}
    keep.update({r["pose"]: r for r in records})
    write_json(out_json, {"object": args.object, "k": args.k, "cap_per_support": CAP,
                          "pad_half_width_m": PAD_W, "pad_thickness_m": PAD_T,
                          "post_half_width_m": POST_W, "clearance_m": CLEAR,
                          "poses": [keep[p] for p in sorted(keep)]})


if __name__ == "__main__":
    main()
