r"""slides/sys_floor — the setup's own target pose, with the landings on the floor.

    python slides/sys_floor/on_the_floor.py
        -> slides/sys_floor/on_the_floor_A1-f.png
        -> slides/sys_floor/on_the_floor_B.png
        -> slides/sys_floor/on_the_floor_C5.png

`slides/setup/poses/big_tip.py`'s target pose, unchanged -- same pose, same camera
angles, same grey floor, same green work region -- with **the places the
load can land scattered on the ground**, and NOTHING ELSE ON IT.  The setup
page's orange-red contact target is deliberately not carried over: the landings
are the one thing this page adds, and a second mark in the same warm hue reads
as one of them.

Two other pages drew the same quantity -- a true plan view with the region as a
star hull, and the region with its convex hull and the ring of feet that answers
it -- and both were cut: they were correct and neither read beside the setup
slide.  This one is the setup slide with a single thing added.
`support_polygon.py` is what is left of them, the two closed forms this page
imports rather than re-derives.

THE LANDINGS AT K = 0.5, WITH A FOLD DIAGNOSTIC

Row (3) cuts the free body around the workpiece AND its supports, so the
supports' forces on it are internal and cancel, and where the load lands on the
floor depends only on gravity and the process push:

    R = mg - F_push . z,   M = c x (-mg z) + q x F_push,
    p = (M_y/R, -M_x/R, 0),   F_push = K mg d,   K = 0.5

`support_polygon.cop`, imported and not re-derived, at `K` body weights of push.
Row (3) is solved when every one of those landings lies inside the convex hull
of everything touching the floor.

The current load case is the full process push, t = K = 0.5 body weights.
The page samples contact points and N_PHI directions around each cone's rim,
then filters those directions by line of sight. It is a point cloud, not a
certificate of the continuous boundary. The algebra explaining the rim is:

1. **A lever rule.**  With `g = (c_x, c_y, 0)` the plumb point and
   `h = q - (q_z/d_z) d` the point where the push's OWN line of action crosses
   the floor,

       p = ( 1 * g  +  (-t d_z) * h ) / ( 1 - t d_z )

   -- the centre of pressure is the average of the plumb point and the push's
   floor intercept, weighted by their vertical force components.  `t = 0` gives
   `g`; a downward push puts `p` between the two; an upward one puts it on the
   far side of `g`, and `t d_z -> 1` sends it to infinity, which is the assembly
   on the point of floating.

2. **The full-strength landing.** With `r = q - g`,
   `p - g = K (r_z d_h - d_z r_h) / (1 - K d_z)`, and `R >= 1 - K = 0.5`.
   If the magnitude were varied over [0,K], each (q,d) would trace the segment
   from g to this landing. That extension is not part of the current check.

3. **Stereographic projection gives a rational map.**
   Substituting `u = d_h / (1 - d_z)`, the projection of the direction from the
   north pole, with `1/(1-d_z) = (1+|u|^2)/2`:

       p(u) = g + K [(1-|u|^2) r_h + 2 r_z u]
                    / [(1+K) + (1-K)|u|^2]
            = g + [(1-|u|^2) r_h + 2 r_z u] / [3+|u|^2]  at K = 0.5.

   The old quadratic map applies only at K = 1. An unoccluded cone that excludes
   the north pole maps to a disc; a cone containing it needs the other chart.

4. **The rim and critical set bound the image.** For r_z > 0 and K > 0,
   the critical directions obey `d . r = K r_z`, so at K = 0.5 they obey
   `d . r = 0.5 r_z`. For the full, unoccluded cone,

       boundary  is contained in  Phi(rim of the cone)  U  Phi(fold circle)

   `folds` counts the contact points whose full cone meets that circle. It does
   not draw the fold or apply the line-of-sight filter to it. Occlusion can add
   boundary curves inside the original cone, so a zero fold count alone does
   not certify the rim after that filter. The 96 rim angles are sampled too.

DRAWN FLAT, AND OCCLUDED BY HAND

The dots are projected onto the finished render rather than put in the scene as
geometry, because 8 640 of them is 8 640 geoms and `render` is handed 64.  What
that costs is the depth test, so it is done by hand and exactly: a dot is drawn
only if the ray from its landing point to the FINAL eye misses the workpiece.
The frame first fits the workpiece and every retained landing, then visibility
is evaluated using that camera. This avoids coupling framing to a visibility
mask computed before the camera moves, and keeps every final visible dot in
the frame. The part therefore hides the ground behind it, as it should.
"""
from __future__ import annotations

import hashlib
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
from PIL import ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "slides" / "setup" / "poses"))
sys.path.insert(0, str(ROOT / "slides/tools"))

from common import obj_path                                         # noqa: E402
from cone_model import CONE_HALF_DEG, frame                         # noqa: E402

import tip_sequence as T                                            # noqa: E402
import big_tip as G                                                 # noqa: E402
import support_polygon as S                                         # noqa: E402
from tip_sequence import PX, page, render, screen                   # noqa: E402

# THE COLUMN TITLE IS SET ONLY ROUND THIS PAGE'S OWN `page` CALL, and that is
# not tidiness.  `sweep` below calls `big_tip.sweep`, which writes
# `slides/poses`'s three pages on its way past; with `T.TITLES` rebound at import
# those pages came out with THIS page's single column head instead of their own
# three, and their md5s moved every time this script ran.  A shared module global
# is a shared module global.
SETUP_TITLES = T.TITLES        # big_tip's, and the pages `G.sweep` writes keep them
MY_TITLES = ("the landings",)  # short: `page` sets the column title in the same
                               # 68 px face as the object name, and at one column
                               # a long one runs into it

K = S.K                 # shared slides declaration: 0.5 body weights
OBJECTS = ("A1-f", "B", "C5")
POINTS, N_PHI = 220, 96  # contact points over the work region, and directions
                        # round the cone's rim; both are finite samples
CLIP = 3.0              # landings further than this many box diagonals from the
                        # plumb point are dropped, and the count prints.  On an
                        # unbounded pose there is no outer edge to reach and a
                        # frame that held every landing would hold no workpiece
DOT = 0.0038            # a dot's radius, in panel widths
FILL = (219, 138, 36)   # `support_polygon`'s own ORANGE, which is what that page
                        # and `torque/code/demo_scene.py` already spend on this
                        # quantity.  It is neither the setup page's orange-red
                        # contact ring nor its green work region
TMP = G.TMP             # the scratch directory `paint_parts` writes relative to


def rim(mesh, T, inside, seed):
    """Contact points over the work region, and the CONE'S RIM at each.

    `cone_model.cone_pushes`' point stream, call for call -- the same generator
    seeded the same way, so this page's contacts are the pipeline's -- with its
    interior directions replaced by `N_PHI` evenly spaced ones at exactly the
    cone's half-angle. The full-cone boundary can also contain a fold; the
    diagnostic below checks for that. Visibility may introduce additional
    boundary curves, which this rim-only point cloud does not reconstruct.

    The reachability test is unchanged: a push counts only if a straight line
    out of the air reaches the point without going through the workpiece.
    """
    R, t = T[:3, :3], T[:3, 3]
    rng = np.random.default_rng(seed)
    where = np.flatnonzero(inside)
    w = mesh.area_faces[where]
    face = rng.choice(where, size=POINTS, p=w / w.sum())
    a, b = rng.random(POINTS), rng.random(POINTS)
    flip = a + b > 1
    a[flip], b[flip] = 1 - a[flip], 1 - b[flip]
    tri = mesh.triangles[face]
    pts = (tri[:, 0] + a[:, None] * (tri[:, 1] - tri[:, 0])
           + b[:, None] * (tri[:, 2] - tri[:, 0]))
    nrm = mesh.face_normals[face]
    c0, s0 = np.cos(np.radians(CONE_HALF_DEG)), np.sin(np.radians(CONE_HALF_DEG))
    ph = np.linspace(0.0, 2 * np.pi, N_PHI, endpoint=False)
    e1, e2 = frame(nrm)
    V = (c0 * nrm[:, None, :]
         + (s0 * np.cos(ph))[None, :, None] * e1[:, None, :]
         + (s0 * np.sin(ph))[None, :, None] * e2[:, None, :]).reshape(-1, 3)
    O = np.repeat(pts + nrm * 1e-5, N_PHI, axis=0)
    assert np.abs(np.linalg.norm(V, axis=1) - 1.0).max() < 1e-12
    clear = ~mesh.ray.intersects_any(ray_origins=O, ray_directions=V)
    return O[clear] @ R.T + t, -(V[clear] @ R.T), pts @ R.T + t, nrm @ R.T


def folds(pts, nrm, com, t=K):
    """How many contact points have the FOLD circle inside their cone.

    For points above the floor, the critical set is `{d : d . r = t r_z}`.
    It meets the full cone exactly when `t r_z` lies between the extremes of `d . r`
    over the cap, which are `|r| cos(beta -+ a)` with `beta` the angle between the
    inward normal and `r`. This is before the visibility filter; it counts
    possible missing folds, not their visible subset.
    """
    g = np.array([com[0], com[1], 0.0])
    r = pts - g
    L = np.linalg.norm(r, axis=1)
    beta = np.arccos(np.clip((r * (-nrm)).sum(axis=1) / np.maximum(L, 1e-12), -1, 1))
    a = np.radians(CONE_HALF_DEG)
    hi = L * np.cos(np.maximum(beta - a, 0.0))
    lo = L * np.cos(np.minimum(beta + a, np.pi))
    level = t * r[:, 2]
    return int(((level >= lo) & (level <= hi)).sum())


def landings(mesh, T_star, inside, seed):
    """Where the load can land, one point per rim push.  `support_polygon.cop`."""
    com = T_star[:3, :3] @ mesh.center_mass + T_star[:3, 3]
    q, d, pts, nrm = rim(mesh, T_star, inside, seed)
    P, R = S.cop(q, d, com, t=K)
    return np.column_stack([P, np.zeros(len(P))]), com, folds(pts, nrm, com), len(pts)


def visible(mesh, T_star, P, cam):
    """Which landings the workpiece does not stand in front of.

    `tip_sequence.buried`'s test, one ray a dot instead of one ray a contact:
    from the landing toward the eye, against the mesh in the pose it is drawn
    in.  Vectorised, because there are thousands of them.
    """
    d = cam.eye - P
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    R, t = T_star[:3, :3], T_star[:3, 3]
    eps = 1e-4 * float(np.linalg.norm(mesh.extents))
    return ~mesh.ray.intersects_any(ray_origins=((P + eps * d) - t) @ R,
                                    ray_directions=d @ R)


def scatter(im, P, cam):
    """The dots, on the finished panel, furthest away drawn first."""
    xy = screen(P, cam)
    order = np.argsort(-((P - cam.eye) @ cam.fwd))
    dr = ImageDraw.Draw(im)
    r = DOT * PX
    for i in order:
        x, y = xy[i]
        if -r <= x <= PX + r and -r <= y <= PX + r:
            dr.ellipse([x - r, y - r, x + r, y + r], fill=FILL)
    return im


def sweep(name, log):
    """One workpiece: `big_tip`'s own rows, each with its landings scattered."""
    d = obj_path(name)
    mesh, _ = G.refine(trimesh.load(d / "mesh.stl", force="mesh"))
    rows = G.sweep(name)                # the SETUP page's own poses and cameras
    size = float(np.linalg.norm(mesh.extents))
    log(f"\n=== {name}   {len(rows)} rows from slides/setup/poses/big_tip.py, "
        f"part {1000 * size:.0f} mm across")
    tmp = d / TMP
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    grid, labels, notes, out = [], [], [], []
    try:
        for k, r in enumerate(rows):
            T_star, cam, inside = r["T_star"], r["cam"], r["take"]
            P, com, nfold, npts = landings(mesh, T_star, inside, G.SEED + 7 * k)
            box = r["box"]
            span = float(np.linalg.norm(box.max(axis=0) - box.min(axis=0)))
            g = com[:2]
            far = np.linalg.norm(P[:, :2] - g, axis=1) > CLIP * span
            reach_all = float(np.linalg.norm(P[:, :2] - g, axis=1).max())
            P = P[~far]
            Rmin, nz_min, unb = S.exact_R_min(
                mesh, T_star, np.flatnonzero(inside), t=K)
            parts = G.paint_parts(mesh, inside, tmp, f"floor_{name}_r{k}")
            # THE FRAME IS PULLED BACK UNTIL THE LANDINGS ARE IN IT.  The setup
            # page frames on the workpiece and the load lands further out than
            # that on every row, so at its framing the dots run off the panel.
            # Keep the row's elevation and azimuth; fit the look-at point and
            # distance using ALL retained landings, including hidden ones.
            # Visibility must be computed after that move, using the same
            # final camera as rendering and projection. Fitting all P avoids
            # a framing/visibility loop and includes newly revealed dots.
            cam = T.fit(cam.elev, cam.azim, np.vstack([box, P]))
            seen = visible(mesh, T_star, P, cam)
            # NO OVERLAY.  `render`'s `draw` is left empty, so `tip_sequence`'s
            # orange-red contact target is not painted on: the landings are this
            # page's one added thing and a second mark in the same warm hue reads
            # as one of them.  The setup page still carries the contact.
            panel = render(name, T_star, parts, cam)
            grid.append([G.apart(scatter(panel, P[seen], cam))])
            labels.append(f"tip {k + 1}")
            notes.append(("",))
            log(f"  tip {k + 1}: tipped {r['tip']:.1f} deg, {npts} contact points "
                f"x {N_PHI} rim directions, {len(P) + int(far.sum())} landings "
                f"reachable, {int(far.sum())} of them beyond {CLIP:g} box "
                f"diagonals and dropped, {int(seen.sum())} of the rest not behind "
                f"the workpiece"
                + ("  -- UNBOUNDED: R_min is 0, so the region has no outer edge"
                   if unb else "") + "\n"
                f"          they reach {1000 * reach_all:8.1f} mm from the plumb "
                f"point = {reach_all / size:5.2f} part widths; the fold circle is "
                f"inside the full cone on {nfold} of {npts} contact points "
                f"at K = {K:g}; R_min >= {Rmin:.6f} before visibility")
            out.append(dict(name=name, tip=k + 1, unbounded=unb,
                            widths=reach_all / size, seen=int(seen.sum()),
                            dropped=int(far.sum()), fold=nfold, npts=npts))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    T.TITLES = MY_TITLES
    try:
        page(HERE / f"on_the_floor_{name}.png", name, grid, labels, notes)
    finally:
        T.TITLES = SETUP_TITLES
    return out


def main():
    t0 = time.time()

    def log(s):
        print(s, flush=True)

    log(f"slides/setup/poses' own target poses, with row (3)'s landings scattered on "
        f"the floor.  Process cone {CONE_HALF_DEG:.0f} deg, {POINTS} contact "
        f"points x {N_PHI} sampled rim directions, K = {K:g}; the fold "
        f"diagnostic uses d . r = K r_z")
    rows = []
    for name in OBJECTS:
        rows += sweep(name, log)
        f = HERE / f"on_the_floor_{name}.png"
        log(f"  md5 {hashlib.md5(f.read_bytes()).hexdigest()}")
    log(f"\n{len(rows)} rows.  The load reaches "
        f"{min(r['widths'] for r in rows):.2f} to "
        f"{max(r['widths'] for r in rows):.2f} part widths from the plumb point; "
        f"{sum(r['unbounded'] for r in rows)} of {len(rows)} poses have no outer "
        f"edge at all, and the FOLD circle is inside the cone on "
        f"{sum(r['fold'] for r in rows)} of {sum(r['npts'] for r in rows)} "
        f"contact points over all of them. Both contact points and rim angles "
        f"are sampled; folds and visibility-boundary curves are not drawn.")
    log(f"{time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
