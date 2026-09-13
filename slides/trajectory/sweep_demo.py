r"""slides/trajectory -- does the support's swept volume hit the workpiece?

    python slides/trajectory/sweep_demo.py

        slides/trajectory/sweep_demo.png     two panels, the same support twice:
                                             slid in along a1 (BLOCKED) and a2 (clear)
        slides/trajectory/sweep_demo.md      the numbers

THE IDEA.  A support is a rigid solid standing on the floor, slid into place
along the floor in a horizontal straight line `a`.  Everything it swept on the
way,

    Sweep(supp, a) = { x - t a : x in supp, t >= 0 }

-- the support and the whole corridor behind it along `a`, out to the edge of
the scene -- must not enter the workpiece at its target pose.  Two cases on ONE
workpiece and ONE support placement: a direction whose corridor cuts through
the part (the path is blocked) and one whose corridor is clear.

THE WORKPIECE is object B at tip 1 of the slides -- mesh, `T_star`, work
region, contact and camera taken from the shared setup:
`big_tip.refine`, `big_tip.sweep("B")[0]`, the work region
painted through `big_tip.paint_parts`, rendered through `tip_sequence.render`
with the distance refit on this pose (`tip_sequence.fit`).  Metres throughout.

THE SUPPORT IS A CLAW THAT CLOSES ROUND THE WORKPIECE'S TAIL (owner's
revision, 2026-08-30; before it, two cone pads under the haunch, and before
that a bare wedge).  SIXTEEN CONVEX PIECES: SIX JAWS round the tail, a HUB
past the tail's tip with SIX SPOKES out to the jaws, a COLUMN up from the
floor, a BRACE up its back and a base SLAB.  Every piece embeds a couple of
millimetres into its neighbour and the drawn solid is their exact union
(`manifold3d`), one watertight body.  Colour: teal, the raw-generator colour
(`capped_a1f.SHADE`, the 2026-08-20 rule).

AND THE BORE IS NOT A SHAPE ANYBODY DREW.  It is the WORKPIECE'S OWN
SILHOUETTE along the axis the claw slides in on, offset by `CLEAR`, and the
six jaws sit on six supporting lines of it.  That is forced, not chosen: a
socket comes off along an axis only if its bore contains the silhouette of
everything it holds, so this is the TIGHTEST bore that comes off at all, and
every millimetre of the fit is the tail's and not the designer's.  It is also
why the claw makes the page's point where the old two-pad fixture could not --
a pad presses on a face and can be withdrawn nearly anywhere, a claw round a
lobe has an axis and the corridor is what enforces it.

THE ANGLE IS THE WHOLE ANSWER, AND IT IS NARROW.  The claw goes on along the
tail's own bearing and comes off along it; every run walks the clear window's
two edges in at `FINE` degrees and prints them.  It is 58 degrees wide out of
360 -- 60 of the 72 five-degree bearings are blocked -- against the 42 of 72
the two-pad fixture blocked.  And what is left over is NOT the clearance.  A jaw
`DEPTH` long with `CLEAR` of gap would allow 6.3 degrees off the axis; the
tail TAPERS, 18.9 mm of radius at the mouth down to 12.7 at the tip over 14,
a half angle of 23.8 -- and a bore that has to swallow the widest section is
loose by that much everywhere further in.  The two add to 30.1 and the
measured half window is 29.  Every run prints both.  Neither number is designed: shorten the jaws and the window widens
(88 degrees at 13 mm), lengthen them and the bore has to swallow the haunch
and it widens again (72 degrees at 26 mm).  18 mm is the floor of that curve,
which is a fact about the tail.

THE SWEPT VOLUME IS EXACT, AND IT IS THE SUPPORT'S OWN SHAPE -- not a box
round it (owner's revision, 2026-08-30).  Translation distributes over a
union, so `Sweep(P1 u ... u Pn, a) = Sweep(P1, a) u ... u Sweep(Pn, a)`; each
piece is convex, and the sweep of a convex solid under a translation is the
convex hull of the solid and its translate (METHOD s4.6).  So the corridor is

    Union_i  conv( Pi u (Pi - L a) ),        L well past the frame,

an exact boolean union (`manifold3d`), and its silhouette carries the six jaw
tubes, the spokes, the column and the slab, which is what makes it readable as
THIS support's corridor.  Its intersection with the workpiece is an exact boolean
too, reported in cm3; the part is then drawn as its two exact pieces -- the
part MINUS the corridor in its own colours, the part INSIDE the corridor in
red -- so the red patch is the boolean and not a per-face approximation of
it, and nothing overlaps anything (no z-fighting).

THE TWO DIRECTIONS are compass angles of `a`, the direction the support MOVES
in; the corridor lies behind it along `-a`.  The BLOCKED one puts the corridor
through the bunny's flank; the CLEAR one is the tail's own axis, and pulls the
claw off along the tail and out over open floor.  Every 5 degrees of the
compass is priced in the log and in `sweep_demo.md`, so the two chosen are two
of seventy-two and not a lucky pair, and the window's edges are then walked in
at 1 degree because 5 is too coarse to say where an arc stops.

THE DRAWING.  Two 700 px panels, one camera (`tip_sequence.fit`, on the
part AND the support so the fixture is in frame), no words on the panels.  The
corridor is a translucent solid (alpha `ALPHA`, a greyed teal) the bunny
shows through; the intersection is opaque red `#B02A26`, METHOD s3.0's "what
is owed", given emission so it stays red under the corridor; the support is
teal; a small ink arrow lies on the floor beside the support, along `a`, on
the eye's side of the corridor.  One muted line under each panel carries the
case, its bearing and its cm3 -- and on the clear panel, how much of the
compass is clear at all, because that is the page's claim.  Renderer: mujoco, through `tip_sequence.render`, with this
page's four materials patched into `T.SCENE` for the call the way `skin.py`
does it.

Deterministic: the one sampled thing is the silhouette the bore is read off
(`SAMPLES` area-uniform surface points at a fixed seed -- a 4000-face part's
VERTICES are not its outline).  The GPU is not bit-reproducible on
this machine (`skin.py`'s docstring), so each panel is rendered until two
renders agree byte for byte (`shot`).  `slides/setup/poses/tip_B.png` is rewritten
by `big_tip.sweep` on the way past and checked byte-identical.  Scratch goes
under `objects/B/_big_tip_tmp` and is removed in a `finally`.
"""
from __future__ import annotations

import hashlib
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for p in ("slides/setup/poses", "slides/tools"):
    sys.path.insert(0, str(ROOT / p))

from common import obj_path                                         # noqa: E402
from capped_a1f import SHADE                                        # noqa: E402
from disturbances import _font                                      # noqa: E402
from mesh_export import export                                       # noqa: E402
import big_tip as G                                                 # noqa: E402
import tip_sequence as T                                            # noqa: E402
from tip_sequence import PX, fit, render, screen                    # noqa: E402
from cover import PAPER, INK, MUTED                                # noqa: E402

NAME, TIP = "B", 1
GAP = 18
# ---- the support: a CLAW that closes round the workpiece's TAIL -------------
# The bore is not a shape somebody drew.  It is the WORKPIECE'S OWN SILHOUETTE
# along the axis the claw slides in on, offset by `CLEAR` -- which is forced:
# a socket is removable along an axis only if its bore contains the silhouette
# of everything it holds, so the bore is the SMALLEST removable one, and the
# jaws sit on that silhouette's own supporting lines.
TAIL = (1.0, 0.0, 0.0)  # the tail's axis IN THE MESH FRAME (the bunny's tail
                        # points along its own +x).  Under `T_star` its
                        # horizontal bearing is 328.0 deg; the claw slides in
                        # against it, along 148.0, and that is the only family
                        # of bearings that gets it on or off
CLEAR = 0.0020          # the bore's clearance round that silhouette, m
DEPTH = 0.018           # a jaw's run along the axis
JAWS = 6                # jaws round the bore, one on each supporting line
JAW_T = 0.005           # a jaw's wall thickness
JAW_GAP = 0.010         # the gap between neighbours -- far under the tail's own
                        # width, so nothing leaves sideways between two jaws
JAW_TAPER = 0.005       # each jaw narrows this much at the mouth
TIP_GAP = 0.004         # the plate stands this far beyond the tail's tip
PLATE_T = 0.008         # the back of the claw: how thick the hub and spokes are
HUB_R = 0.009           # the hub, past the tail's tip
SPOKE_HV = 0.005        # a spoke's half width
SIMPLIFY = 0.0004       # the bore polygon's tolerance, m: a rounded offset of a
                        # 50-vertex hull is 500 points and its swept hull comes
                        # back degenerate, so the polygon is simplified first
COL_OFF, COL_LEN = 0.006, 0.016     # the column, behind the plate, m
COL_HV = 0.009                      # its half width; it climbs to the hub
BRACE_RUN, BRACE_HV = 0.016, 0.005  # the brace up the column's back
SLAB_OFF, SLAB_LEN = -0.004, 0.034  # the base slab, behind the plate
SLAB_HV, SLAB_H = 0.020, 0.006
SAMPLES = 200000        # area-uniform surface samples the silhouette is read
                        # off -- a 4000-face part's VERTICES are not its
                        # outline, and the bore is an outline
# ---- the two cases: compass angle of `a`, the direction the support MOVES ---
CLEAR_DEG = 148.0       # the tail's own axis, reversed: the claw goes ON
                        # along this bearing, and the window of bearings that
                        # work at all is measured round it
CASES = (("blocked", 280.0), ("clear", CLEAR_DEG))
L = 1.0                 # the corridor's length, m -- six part widths, off the page
STEP = 5.0              # the compass is priced every this many degrees
FINE = 1.0              # and the clear window's two edges are walked in at this
SKIN_EPS = 0.00025      # the DRAWN corridor is inflated by this much, m.  The
                        # sweep contains the support with a shared boundary --
                        # `conv(P u (P - L a))` keeps P's own leading faces --
                        # and two coincident faces z-fight (a comb of stripes on
                        # the fixture, measured 2026-08-30).  Every boolean and
                        # every number uses the EXACT corridor; only the
                        # translucent solid handed to the renderer is inflated,
                        # a quarter millimetre radially, under a pixel here
# ---- the drawing ------------------------------------------------------------
ALPHA = 0.35            # the corridor's opacity
SUPPORT_HEX = SHADE[4]  # teal: a raw generator
SWEEP_HEX = "#8fb5b0"   # the corridor: a GREYED teal.  `SHADE[2]` at this alpha
                        # turned the red under it salmon and a plain grey made
                        # the corridor vanish into the floor; a teal pulled
                        # halfway to grey keeps the corridor in the support's
                        # colour family and the red under it red (measured on
                        # three renders, 2026-08-30)
OWED_HEX = "#B02A26"    # METHOD s3.0's red: what is owed
OWED_GLOW = 1.0         # the red's emission: the cut is on the part's underside,
                        # lit by ambient alone, and seen through the corridor a
                        # shaded red read as brown.  Emission keeps it red
ARROW_RGBA = (0.13, 0.13, 0.12)     # the outside agent, `tip_sequence.PUSH_RGBA`
ARROW_L = 0.040         # the arrow's length, m
ARROW_W = 0.0025        # its shaft radius
ARROW_AIR = 0.008       # the air between the arrow and the support's silhouette
HEAD_R, HEAD_L, SECTIONS = 1.75, 1 / 3, 16      # `skin.py`'s arrow proportions
BAND = 56               # the caption band under a panel, px
AGREE = 5               # renders of one panel before giving up on two agreeing
EXTRA = [0]


def hexf(h):
    return tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def compass(deg):
    return np.array([np.cos(np.radians(deg)), 0.0, np.sin(np.radians(deg))])


# ------------------------------------------------------------- the geometry ---

def prism(poly, w0, w1, M):
    """A convex polygon in the bore plane, extruded along the claw's axis."""
    P = np.asarray(poly, float)
    P = P[ConvexHull(P).vertices]
    n = len(P)
    V = np.vstack([np.c_[np.full(n, w0), P], np.c_[np.full(n, w1), P]])
    F = []
    for i in range(1, n - 1):
        F += [[0, i + 1, i], [n, n + i, n + i + 1]]
    for i in range(n):
        j = (i + 1) % n
        F += [[i, j, n + j], [i, n + j, n + i]]
    m = trimesh.Trimesh(V @ M.T, np.asarray(F), process=False)
    if m.volume < 0:
        m.invert()
    return m


def slot(n, t, n0, n1, t0, t1):
    """A rectangle in the bore plane, given by an outward normal and a tangent."""
    return np.array([n0 * n + t0 * t, n1 * n + t0 * t,
                     n1 * n + t1 * t, n0 * n + t1 * t])


def bore_frame(Tm):
    """The claw's own axes: the tail's axis LAID HORIZONTAL (the support slides
    on the floor, so its path has no vertical component and the axis it is
    designed round must not either), then across, then up."""
    w = Tm[:3, :3] @ np.asarray(TAIL)
    ax = np.array([w[0], 0.0, w[2]])
    ax /= np.linalg.norm(ax)
    return ax, -np.cross([0.0, 1.0, 0.0], ax), np.array([0.0, 1.0, 0.0])


def support(bunny, Tm):
    """The claw, in the world.  Returns the union solid (one watertight body),
    the convex pieces it is the union of, its frame, and the bore's own record.

    Every jaw is ONE convex piece, so `corridor`'s per-piece hull formula
    applies to the claw as it stands.  The six jaws sit on six supporting lines
    of the bore polygon, which is the part's own silhouette beyond the mouth,
    offset by `CLEAR`: that is the tightest bore the claw can have and still
    come off along the axis, and it is why nothing about the fit is a choice.
    """
    ax, cr, up = bore_frame(Tm)
    M = np.c_[ax, cr, up]                       # (along, across, up) -> world
    pts, _ = trimesh.sample.sample_surface_even(bunny, SAMPLES, seed=0)
    w = pts @ ax
    w_tip = float(w.max())
    w_plate = w_tip + TIP_GAP                   # the plate caps the tip
    w_mouth = w_plate - DEPTH                   # the jaws' open end
    sel = w >= w_mouth
    P = np.c_[pts[sel] @ cr, pts[sel] @ up]
    hull = P[ConvexHull(P).vertices]
    bore = Polygon(hull).buffer(CLEAR, quad_segs=4).simplify(SIMPLIFY)
    Q = np.asarray(bore.exterior.coords)[:-1]
    c = np.asarray(bore.centroid.coords)[0]
    pieces, lines = [], []
    for k in range(JAWS):
        a = np.radians(90.0 + 360.0 * k / JAWS)
        n = np.array([np.cos(a), np.sin(a)])
        t = np.array([-n[1], n[0]])
        h = float((Q @ n).max())                # the bore's supporting line
        lines.append((n, t, h))
        d = Q - c                               # the jaw takes its own sector
        ang = np.degrees(np.arctan2(d @ t, d @ n))
        ang = np.where(ang > 180.0, ang - 360.0, ang)
        s = Q[np.abs(ang) <= 180.0 / JAWS] @ t
        t0, t1 = s.min() + JAW_GAP / 2, s.max() - JAW_GAP / 2
        V = np.vstack([
            np.c_[np.full(4, w_plate + PLATE_T / 2), slot(n, t, h, h + JAW_T, t0, t1)],
            np.c_[np.full(4, w_mouth),
                  slot(n, t, h, h + JAW_T, t0 + JAW_TAPER, t1 - JAW_TAPER)]])
        pieces.append(trimesh.convex.convex_hull(V @ M.T))
    # the back: a hub past the tail's tip and one spoke out to each jaw.  It is
    # a SPIDER and not a plate because a plate hides what the claw is doing --
    # and it costs nothing, since everything at `w >= w_plate` is beyond the
    # tail and can cross the bore freely
    th = np.radians(np.arange(6) * 60.0 + 30.0)
    pieces.append(prism(c + HUB_R * np.c_[np.cos(th), np.sin(th)],
                        w_plate, w_plate + PLATE_T, M))
    for n, t, h in lines:
        pieces.append(prism(slot(n, t, float(c @ n), h + JAW_T,
                                 float(c @ t) - SPOKE_HV, float(c @ t) + SPOKE_HV),
                            w_plate, w_plate + PLATE_T, M))
    e = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    pieces.append(prism(slot(*e, c[0] - COL_HV, c[0] + COL_HV, 0.0, c[1]),
                        w_plate + COL_OFF, w_plate + COL_OFF + COL_LEN, M))
    wb = w_plate + COL_OFF + COL_LEN            # the brace, up the column's back
    tri = np.array([[wb - 0.002, SLAB_H], [wb + BRACE_RUN, SLAB_H],
                    [wb - 0.002, c[1] - 0.004]])
    V = np.vstack([np.c_[tri[:, 0], np.full(3, c[0] - BRACE_HV), tri[:, 1]],
                   np.c_[tri[:, 0], np.full(3, c[0] + BRACE_HV), tri[:, 1]]])
    pieces.append(trimesh.convex.convex_hull(V @ M.T))
    pieces.append(prism(slot(*e, c[0] - SLAB_HV, c[0] + SLAB_HV, 0.0, SLAB_H),
                        w_plate + SLAB_OFF, w_plate + SLAB_OFF + SLAB_LEN, M))
    S = trimesh.boolean.union(pieces, engine="manifold")
    assert S.is_watertight and S.body_count == 1, "the claw is not one body"
    return S, pieces, (ax, cr, up), dict(w_mouth=w_mouth, w_tip=w_tip,
                                         w_plate=w_plate, hull=hull, bore=Q, c=c)


def corridor(pieces, a, eps=0.0):
    """`Sweep(supp, a)` for a union of convex pieces: translation distributes over
    the union, and each convex piece sweeps to the hull of itself and its
    translate (METHOD s4.6).  Returns the exact union and the per-piece hulls.
    `eps > 0` inflates every piece radially about its centroid first -- the
    render-only copy, see `SKIN_EPS`; every measurement runs at `eps = 0`.
    """
    hulls = []
    for P in pieces:
        V = P.vertices
        if eps:
            d = V - P.centroid
            V = V + eps * d / np.linalg.norm(d, axis=1, keepdims=True)
        hulls.append(trimesh.convex.convex_hull(np.vstack([V, V - L * a])))
    return trimesh.boolean.union(hulls, engine="manifold"), hulls


def cut(bunny, sweep):
    """The part inside the corridor and the part outside it, exactly."""
    inter = bunny.intersection(sweep, engine="manifold")
    rest = bunny.difference(sweep, engine="manifold")
    vi = float(inter.volume) if len(inter.faces) else 0.0
    assert abs(vi + rest.volume - bunny.volume) < 1e-7 * bunny.volume, \
        "the two pieces do not add up to the part"      # manifold: ~4e-9 relative
    return inter, rest, vi


def inside_hull(hull, pts):
    """Which of `pts` lie strictly inside one convex hull (all its planes)."""
    d = (pts[:, None, :] - hull.triangles_center[None]) * hull.face_normals[None]
    return d.sum(axis=2).max(axis=1) < 0


def inside_any(hulls, pts):
    got = np.zeros(len(pts), bool)
    for h in hulls:
        got |= inside_hull(h, pts)
    return got


def arrow(a, S, azim):
    """A mesh arrow along `a` on the floor beside the support, on the eye's side
    of the corridor: `skin.py`'s cylinder-and-cone, lying down.  It stands just
    clear of the support's own silhouette across `a`, so it can never lie in
    the corridor whatever the fixture's shape.  The side is chosen off the
    row's AZIMUTH alone -- the eye's horizontal heading, fixed before the
    refit -- so both arrows exist before the camera is fitted and can be
    fitted INTO the frame (the first cut clipped them at the panel's edge)."""
    up = np.array([0.0, 1.0, 0.0])
    side = -np.cross(up, a)
    if side @ compass(azim) < 0:
        side = -side
    c = S.centroid.copy()
    off = float(((S.vertices - c) @ side).max()) + ARROW_AIR + HEAD_R * ARROW_W
    c[[0,2]] += off * side[[0,2]]
    c[1] = ARROW_W + 0.0005
    p0 = c - 0.5 * ARROW_L * a
    M = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], a)
    hl = HEAD_L * ARROW_L
    shaft = trimesh.creation.cylinder(ARROW_W, ARROW_L - hl, sections=SECTIONS)
    shaft.apply_translation([0.0, 0.0, (ARROW_L - hl) / 2])
    head = trimesh.creation.cone(HEAD_R * ARROW_W, hl, sections=SECTIONS)
    head.apply_translation([0.0, 0.0, ARROW_L - hl])
    out = []
    for m in (shaft, head):
        m.apply_transform(M)
        m.apply_translation(p0)
        out.append(m)
    return trimesh.util.concatenate(out)


# --------------------------------------------------------------- the render ---

def to_mesh_frame(world, Tm):
    R, t = Tm[:3, :3], Tm[:3, 3]
    return trimesh.Trimesh((world.vertices - t) @ R, world.faces, process=False)


def piece(world, Tm, material, tmp, tag):
    """One world-frame solid, exported in the part's frame for the scene."""
    export(to_mesh_frame(world, Tm), tmp / f"{tag}.obj")
    return [(f"{G.TMP}/{tag}.obj", material)]


def paint_rest(rest, mesh, Tm, take, tmp, tag):
    """The part outside the corridor, painted work / spare by the ORIGINAL face
    each of its faces lies on (`closest_point` on the untouched part: for a
    point on the surface that is the face containing it)."""
    R, t = Tm[:3, :3], Tm[:3, 3]
    local = to_mesh_frame(rest, Tm)
    _, _, fid = trimesh.proximity.closest_point(mesh, local.triangles_center)
    work = take[fid]
    parts = []
    for material, sel in (("work", work), ("spare", ~work)):
        if sel.any():
            sub = local.submesh([np.flatnonzero(sel)], append=True)
            export(sub, tmp / f"{tag}_{material}.obj")
            parts.append((f"{G.TMP}/{tag}_{material}.obj", material))
    return parts


def shot(Tm, parts, cam):
    """`tip_sequence.render` with this page's four materials patched into
    `T.SCENE` for the one call (`skin.shot`'s way), rendered until two renders
    agree byte for byte."""
    mats = "\n".join(
        f'    <material name="{n}" rgba="{r:.3f} {g:.3f} {b:.3f} {al}" specular="0.1" '
        f'emission="{em}"/>'
        for n, (r, g, b), al, em in [("supp", hexf(SUPPORT_HEX), 1, 0),
                                     ("sweep", hexf(SWEEP_HEX), ALPHA, 0),
                                     ("owed", hexf(OWED_HEX), 1, OWED_GLOW),
                                     ("ink", ARROW_RGBA, 1, 0)])
    anchor = '<material name="work"'
    assert anchor in T.SCENE
    was = T.SCENE
    T.SCENE = was.replace(anchor, mats + "\n    " + anchor, 1)
    try:
        seen = []
        for _ in range(AGREE):
            im = render(NAME, Tm, parts, cam)
            raw = np.asarray(im)
            if any(np.array_equal(done, raw) for done in seen):
                break
            seen.append(raw)
        else:
            raise AssertionError(f"{AGREE} renders of one panel and no two agree")
        EXTRA[0] += len(seen) - 1
    finally:
        T.SCENE = was
    assert im.size == (PX, PX)
    return im


def page(out, panels, lines):
    """Two panels side by side on page colour, one muted line under each."""
    W = 2 * PX + 3 * GAP
    H = GAP + PX + BAND + GAP
    im = Image.new("RGB", (W, H), PAPER)
    dr = ImageDraw.Draw(im)
    f = _font(26)
    for i, (panel, line) in enumerate(zip(panels, lines)):
        x = GAP + i * (PX + GAP)
        im.paste(panel, (x, GAP))
        dr.text((x + PX // 2, GAP + PX + BAND // 2), line, fill=MUTED, font=f,
                anchor="mm")
    im.save(out)
    return im


# -------------------------------------------------------------------- main ---

def main():
    t0 = time.time()
    log = lambda s: print(s, flush=True)          # noqa: E731
    d = obj_path(NAME)
    mesh, _ = G.refine(trimesh.load(d / "mesh.stl", force="mesh"))
    tip_png = ROOT / "slides" / "setup" / "poses" / f"tip_{NAME}.png"
    before = md5(tip_png) if tip_png.exists() else None
    rows = G.sweep(NAME)
    after = md5(tip_png)
    log(f"\n=== {NAME} tip {TIP} of {len(rows)} from slides/poses; tip_{NAME}.png "
        f"{'rewritten byte-identically' if before == after else 'CHANGED'} ({after})")
    r = rows[TIP - 1]
    Tm, take = r["T_star"], r["take"]
    R, t = Tm[:3, :3], Tm[:3, 3]
    bunny = trimesh.Trimesh(mesh.vertices @ R.T + t, mesh.faces, process=False)
    assert bunny.is_watertight

    # the claw, and what it closes on
    S, pieces, (ax, cr, up), bore = support(bunny, Tm)
    ext = S.bounds[1] - S.bounds[0]
    assert not inside_any([P.convex_hull for P in pieces], bunny.vertices).any() \
        and not bunny.contains(S.vertices).any(), "the claw is inside the part"
    clash = bunny.intersection(S, engine="manifold")
    assert (float(clash.volume) if len(clash.faces) else 0.0) == 0.0, \
        "the claw intersects the part"
    out_deg = np.degrees(np.arctan2(ax[1], ax[0])) % 360.0
    skin, _ = trimesh.sample.sample_surface_even(bunny, SAMPLES, seed=1)
    held = skin[skin @ ax >= bore["w_mouth"]]
    probe = S.copy()                            # the boolean leaves a few slivers
    probe.update_faces(probe.nondegenerate_faces())     # and they divide by zero
    near, _, _ = trimesh.proximity.closest_point(probe, held)
    gap = np.linalg.norm(near - held, axis=1)
    hull, c = bore["hull"], bore["c"]
    log(f"  the claw: {len(pieces)} convex pieces -- {JAWS} jaws, the hub past the "
        f"tail's tip and its {JAWS} spokes, a column, a brace and the base slab -- unioned to "
        f"one body of {1e6 * S.volume:.2f} cm3, {(1000 * ext).round(1)} mm across its "
        f"world box; the tail's axis bears {out_deg:.2f} deg and the claw goes on "
        f"along {(out_deg + 180) % 360:.2f}; the bore is the part's own silhouette "
        f"beyond the mouth ({1000 * np.ptp(hull, axis=0).round(4)} mm across, centred "
        f"({1000 * c[0]:.1f}, {1000 * c[1]:.1f}) mm) offset {1000 * CLEAR:g} mm; jaws "
        f"from {1000 * bore['w_mouth']:.1f} to {1000 * (bore['w_plate'] + PLATE_T / 2):.1f} mm "
        f"along the axis, the tail's tip at {1000 * bore['w_tip']:.1f}; the held skin "
        f"stands {1000 * gap.min():.2f} to {1000 * gap.max():.2f} mm off the claw "
        f"({len(held)} samples); claw-part intersection 0 cm3 (exact)")
    # the tail's own taper, because it and not `CLEAR` is what sets the window:
    # the bore has to swallow the widest section, so it is loose by the taper
    # everywhere further in
    q = np.c_[held @ cr, held @ up] - bore["c"]
    rad, wq = np.linalg.norm(q, axis=1), held @ ax
    r0 = rad[wq <= bore["w_mouth"] + 0.002].max()
    r1 = rad[wq >= bore["w_tip"] - 0.002].max()
    log(f"  the tail tapers from {1000 * r0:.1f} mm of radius at the jaws' mouth to "
        f"{1000 * r1:.1f} at its tip over {1000 * (bore['w_tip'] - bore['w_mouth']):.1f} mm "
        f"-- a half angle of {np.degrees(np.arctan2(r0 - r1, bore['w_tip'] - bore['w_mouth'])):.1f} "
        f"deg, against the {np.degrees(np.arctan2(CLEAR, DEPTH)):.1f} deg the bore's own "
        f"clearance would allow on a straight-sided tail.  THAT is the window")

    # METHOD s2: a support has to stay OFF the work region, and this one is at
    # the other end of the part from it -- the number, not the assurance
    wr = bunny.triangles_center[take]
    off, _, _ = trimesh.proximity.closest_point(probe, wr)
    log(f"  the work region is {100 * take.mean():.1f} % of the faces and the claw "
        f"clears it by {1000 * np.linalg.norm(off - wr, axis=1).min():.0f} mm at its "
        f"nearest ({len(wr)} faces) -- it is not far off it, and that is a fact "
        f"about this pose rather than a margin anybody chose")

    # the camera: the row's direction, the distance refit on the part, the
    # support AND both cases' arrows, so nothing of the page is clipped
    arrs = {case: arrow(compass(deg), S, r["cam"].azim) for case, deg in CASES}
    V = np.asarray(mesh.vertices) @ R.T + t
    cam = fit(r["cam"].elev, r["cam"].azim,
              np.vstack([V, S.vertices] + [x.vertices for x in arrs.values()]))
    com = bunny.center_mass
    log(f"  camera elev {cam.elev:.0f} azim {cam.azim:.1f} (eye at heading "
        f"{np.degrees(np.arctan2(cam.eye[1] - cam.lookat[1], cam.eye[0] - cam.lookat[0])):.1f} "
        f"deg); part {(1000 * mesh.extents).round(0)} mm, contact "
        f"({1000 * r['contact'][0]:.1f}, {1000 * r['contact'][1]:.1f}) mm, centre of mass "
        f"({1000 * com[0]:.1f}, {1000 * com[1]:.1f}, {1000 * com[2]:.1f}) mm")

    # the compass, priced every STEP degrees
    log(f"  the compass, every {STEP:g} deg: cm3 of the part inside the corridor")
    table = []
    for deg in np.arange(0.0, 360.0, STEP):
        sweep, _ = corridor(pieces, compass(deg))
        inter = bunny.intersection(sweep, engine="manifold")
        table.append((float(deg), float(inter.volume) * 1e6 if len(inter.faces) else 0.0))
    blocked = [deg for deg, v in table if v > 0]
    log("    " + "  ".join(f"{deg:.0f}:{v:.1f}" for deg, v in table))
    log(f"    blocked on {len(blocked)} of {len(table)} bearings")

    # the window's own edges, walked at FINE degrees -- the page's whole claim is
    # how narrow this arc is, and 5 deg is too coarse to say where it stops
    edges = []
    for lo, hi in ((CLEAR_DEG - 90.0, CLEAR_DEG), (CLEAR_DEG, CLEAR_DEG + 90.0)):
        for deg in np.arange(lo, hi + 1e-9, FINE) if lo < CLEAR_DEG else \
                np.arange(hi, lo - 1e-9, -FINE):
            sweep, _ = corridor(pieces, compass(deg % 360.0))
            inter = bunny.intersection(sweep, engine="manifold")
            if not len(inter.faces) or float(inter.volume) == 0.0:
                edges.append(deg % 360.0)
                break
    span = (edges[1] - edges[0]) % 360
    log(f"    the clear window, walked in from both sides at {FINE:g} deg: "
        f"{edges[0]:.0f} to {edges[1]:.0f} deg, {span:.0f} deg "
        f"wide, centred {(edges[0] + span / 2) % 360:.0f}; the "
        f"tail's own axis is at {CLEAR_DEG:.0f}")

    tmp = d / G.TMP
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    panels, lines, recs = [], [], []
    try:
        for case, deg in CASES:
            a = compass(deg)
            sweep, hulls = corridor(pieces, a)
            inter, rest, vol = cut(bunny, sweep)
            cm3 = 1e6 * vol
            assert (cm3 > 0) == (case == "blocked"), (case, cm3)
            # the corridor leaves the frame: its far end's projection is off the panel
            far = screen(sweep.vertices[sweep.vertices @ a < (S.vertices @ a).min()
                                        - 0.5 * L], cam)
            off = ((far < 0) | (far >= PX)).any(axis=1).all()
            arr = arrs[case]
            assert not inside_any(hulls, arr.vertices).any() and \
                not bunny.contains(arr.vertices).any(), "the arrow is in something"
            # the red skin the eye can see, for the record
            c, n = bunny.triangles_center, bunny.face_normals
            dd = cam.eye - c
            dn = dd / np.linalg.norm(dd, axis=1, keepdims=True)
            vis = ((n * dd).sum(axis=1) > 0) & ~bunny.ray.intersects_any(
                ray_origins=c + 1e-6 * n, ray_directions=dn)
            red = inside_any(hulls, c)
            skin = 1e4 * bunny.area_faces[red].sum()
            skin_vis = 1e4 * bunny.area_faces[red & vis].sum()
            tag = f"sweep_{NAME}_{case}"
            if cm3 > 0:
                parts = paint_rest(rest, mesh, Tm, take, tmp, tag) \
                    + piece(inter, Tm, "owed", tmp, tag + "_owed")
            else:
                parts = G.paint_parts(mesh, take, tmp, tag)
            parts += piece(S, Tm, "supp", tmp, tag + "_supp") \
                + piece(arr, Tm, "ink", tmp, tag + "_arrow") \
                + piece(corridor(pieces, a, eps=SKIN_EPS)[0], Tm, "sweep", tmp,
                        tag + "_sweep")   # translucent last, and the inflated copy
            panels.append(shot(Tm, parts, cam))
            lines.append(
                f"{case} \N{MIDDLE DOT} a at {deg:.0f}\N{DEGREE SIGN} "
                f"\N{MIDDLE DOT} {cm3:.1f} cm\N{SUPERSCRIPT THREE} inside the workpiece"
                if cm3 > 0 else
                f"{case} \N{MIDDLE DOT} a at {deg:.0f}\N{DEGREE SIGN} "
                f"\N{MIDDLE DOT} 0 cm\N{SUPERSCRIPT THREE}, and so is {span:.0f}\N{DEGREE SIGN} "
                f"of the 360")
            recs.append(dict(case=case, deg=deg, cm3=cm3, skin=skin, skin_vis=skin_vis,
                             off=off, n_inter=len(inter.faces), n_rest=len(rest.faces)))
            log(f"  {case}: a at {deg:.0f} deg = ({a[0]:+.3f}, {a[1]:+.3f}, 0), the corridor "
                f"behind it toward {(deg + 180) % 360:.0f} deg; part inside the corridor "
                f"{cm3:.2f} cm3 ({100 * cm3 / (1e6 * bunny.volume):.1f} % of "
                f"{1e6 * bunny.volume:.1f}), red skin {skin:.1f} cm2 of which "
                f"{skin_vis:.1f} faces the eye; the corridor is one body of "
                f"{len(sweep.faces)} faces, far end {'off' if off else 'ON'} the panel; "
                f"pieces {len(inter.faces)} / {len(rest.faces)} faces")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert not (d / G.TMP).exists()
    out = HERE / "sweep_demo.png"
    im = page(out, panels, lines)
    log(f"  wrote {out.name}  {im.size[0]} x {im.size[1]} px  md5 {md5(out)}; "
        f"renders that disagreed with their first and bought a third: {EXTRA[0]}")
    log(f"{time.time() - t0:.1f} s")
    return recs


if __name__ == "__main__":
    # The current presentation uses a fixed B/pose_2 snapshot. Preserve the
    # historical experiment and its library functions behind an explicit flag.
    import runpy
    import sys
    if '--legacy' in sys.argv:
        sys.argv.remove('--legacy')
        main()
    elif len(sys.argv) > 1:
        raise SystemExit('Use slides/render.py for current figures; --legacy enables historical options.')
    else:
        runpy.run_path(str(Path(__file__).with_name('presentation.py')), run_name='__main__')
