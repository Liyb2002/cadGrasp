"""The CORNER tip, in three pictures a pose: at rest -> pushed over -> falling back.

    python slides/setup/poses/tip_sequence.py
        -> slides/setup/poses/tip_A1-f.png
        -> slides/setup/poses/tip_B.png
        -> slides/setup/poses/tip_C5.png

A PROBLEM-STATEMENT SLIDE, not a result.  It is judged on whether a viewer with
no caption can see, in ten seconds, that the workpiece was standing flat, that
somebody pushed it over onto ONE POINT of its own footprint, that it is now
hanging in a pose IT CANNOT HOLD -- its weight comes down a measured number of
millimetres away from that point and is pulling it back -- and that the patch a
process has to reach is up near the top.  Anything that does not survive at
slide size is left out.

**FIVE ROWS AN OBJECT, and they are the five `pivot == "point"` poses.**  That is
a DELIBERATE NARROWING and not a truncated ten: `tips.json` holds five corner
tips and five edge tips per object, in the order `e p e p e p e p e p`, and this
page now draws only poses 1, 3, 5, 7, 9.  The edge poses are not deleted from
anything and nothing downstream changed; they are simply not what this page is
for.  `POSES` is derived from `tips.json`'s own `pivot` field rather than typed,
and `sweep` asserts that what it gets is five poses and all of them corners, so a
page of five rows can never be a page of ten that lost some.

The reason for the narrowing is the reason for the camera below.  A CORNER tip is
the harder half of METHOD s0's distinction -- one contact, nothing else on the
floor -- and it is the half a camera can actually serve, because a corner has no
EDGE whose length has to survive the projection.

**Almost nothing here is computed.  Nearly everything drawn is read off disk** --
METHOD s0's tip examples (`objects/<name>/tips/tips.json`), the stable placements
they were tipped from (`poses.json`), and the STORED work region
(`objects/<name>/region/region.json`, ONE CONNECTED PATCH of 8-15 % of the
surface a pose).  The exceptions are named and both are measurements of this
page's own drawing rather than new claims about the workpiece: how much of the
green each camera can see, and the TIPPING MARGIN, which is METHOD s11.4's `w(0)`
recomputed from the mesh and the pose.  It never calls `cover.covered()` or
`cover.grow()`; METHOD s7 records both as broken.

Three columns:

    at rest                  the tip                    the target pose
    the stable placement     the same part part-way     `T_world_mesh`, held
    `examples[i].placement`  through, with the PIVOT     nowhere, with the STORED
    indexes into poses.json  and a push arrow            region green AND THE
                                                        WEIGHT HANGING OFF THE
                                                        ONE CONTACT

**Column 2 carries the contact, and there is exactly one of it.**  A corner pivot
is a single point: one disc, drawn as a filled mark inside a separated ring so
that its ISOLATION is the thing the eye gets -- not a contact among others, THE
contact, with nothing else of the workpiece on the floor.  That is what makes a
corner tip worth less than an edge tip and it is the whole content of the middle
picture.

**Column 3 carries the instability, and it is an exact quantity, not a mood.**
Leaning is not falling, and a picture of a leaning part is not a picture of a
part that cannot stand.  So column 3 draws the four things that make the
difference: the CENTRE OF MASS (the ISO half-filled circle, drawn flat -- see
below), a PLUMB LINE dropped from it to the floor, the PIVOT, and the horizontal
GAP between where the plumb line lands and the pivot -- marked in METHOD s3.0's
red for `what is owed`, and labelled in millimetres.

That gap is **METHOD s11.4's tipping margin**: gravity pulls at the centre of
mass, the floor pushes up at a contact that is not under it, and the pair is a
COUPLE that no floor reaction can cancel.  `margin` computes it the way s11.4
does -- minimise `|SUM_i lam_i (p_i - c) x zhat|` over the split of one body
weight across the ground contacts -- and then ASSERTS that it equals the flat
horizontal distance the page draws.  s11.4 records that equality holding to
`6.9e-18 m`; run over these three objects AND `cuboid_baseline` the largest
disagreement is 6.938893903907228e-18 m, on cuboid pose 5, so s11.4's figure IS
this quantity and this is where it comes from.  **It is computed and asserted on
ALL TEN poses of every object even though five are drawn**, because the edge
branch of that calculation is the harder one and a check that never runs is not a
check.

The margins on the five drawn rows: A1-f 29.8-108.5 mm, B 33.7-94.3 mm, C5
16.1-24.1 mm, against `cuboid_baseline`'s recorded 2.03-45.50 mm.  The two
biggest (A1-f poses 1 and 5) are tips cut short by ANOTHER PART OF THE WORKPIECE
LANDING, so they stop nowhere near balance and hang a long way out -- which is
the same fact that makes them the two poses whose tip angle is smallest.

**LOOK ALONG THE TIP AXIS.  One camera per ROW, shared by its three columns.**
A rotation about an axis carries every point round a circle in the plane
perpendicular to it, and that circle projects to an ELLIPSE whose axis ratio is
`cos phi`, with `phi` the angle between the axis and the view direction.  Look
along the axis and the circle is a circle: the tip reads at its true angle.  Look
across it and the tip reads as the part getting SHORTER.  A previous version of
this page sat at `phi = 45` deg, because an EDGE pivot also has a contact segment
whose apparent length goes as `sin phi` and 45 is the only compromise that keeps
either -- and it charged every row 29 % of its rotation: a 39.75 deg tip drew as
28.7 deg.  **A corner pivot has no segment to foreshorten, so there is no
compromise to make and the whole of `cos phi` is taken.**

Two things stop `phi` at zero rather than none:

  - **The elevation is a floor on `phi`, exactly.**  The pivot axis is horizontal,
    so an eye tilted up by `ROW_ELEV` can never come closer to it than `ROW_ELEV`:
    `cos phi = cos(ROW_ELEV) cos(OUT)` has no solution below it.  `ROW_ELEV` is
    therefore chosen as LOW as the page's own floor drawing allows -- 12 deg,
    against `sin 14 = 0.242` on METHOD s3.0's camera and `sin 30 = 0.500` on the
    version this replaces.  What 12 deg buys back is not only the rotation: the
    tipping margin lies along the pivot NORMAL and the eye is now nearly along the
    AXIS, so the gap is drawn at 0.99 of its true length and crosses the plumb
    line at 91.7 deg on the page -- a right angle at full size, where the 45 deg
    camera drew it at 0.785 and 54.7 deg.
  - **The eye stands OUTSIDE the pivot**, by `OUT` degrees of swing off the axis
    toward the outward normal.  The part turns TOWARD its corner and leans out
    over it, so an eye on the far side of the contact from the footprint has the
    pivot nearest it.  At `OUT = 0` that is not true of the eye at all: it lies
    in the vertical plane THROUGH the axis, its outward component is exactly
    zero, and the contact sits on the silhouette rather than inside or outside
    it.  `BAND[0]` = 8 deg is the smallest swing that makes the rule true, and it
    is honest about what it buys: measured over the fifteen rows on the cameras
    actually drawn, `OUT` = 0 and `OUT` = 8 show the SAME mean work region
    (67.7 %) and bury the SAME four contacts.  It is a geometric requirement,
    not an empirical gain, and it costs `cos 14.4 / cos 12` = 0.991 of the drawn
    angle -- under one per cent.  `outward` fixes the side and the assert in
    `choose` says the eye landed there, which at `OUT` = 0 it could not.

`OUT` starts at `BAND[0]` = 8 deg -- `phi` = 14.4 deg, `cos phi` = 0.969, so a
39.75 deg tip draws as 38.6 deg instead of 28.7 -- and WALKS OUTWARD ONE RUNG AT A
TIME, stopping the first time the next rung is not worth `GAIN` = 5 more points
of the visible work region.  That is "drift off-axis only as far as the shape
requires", made a rule: thirteen of the fifteen rows stop on the first rung, and
the fifteenth (C5 pose 1) walks to 22 deg because its patch is nearly edge-on
until it does.  Deterministic: a fixed ladder, a measured test, ties to the inner
rung and then to the first side.

**The pivot, the plumb line, the gap and the centre of mass are drawn OVER the
render, not in it.**  The pivot lies on the floor at the edge of the footprint
and the workpiece stands over it -- `buried` measures per pose whether a sphere
placed there would be behind the part, and prints it under this camera and under
METHOD s3.0's.  The centre of mass is not a camera problem at all: on A1-f and B
it is 6.93 and 33.18 mm INSIDE a watertight solid, and on C5 it is 4.85 mm
OUTSIDE the surface, in the void a thin open frame encloses, where a sphere would
float in mid-air and read as part of the scene.  Both roads end at a mark painted
flat, with a white casing that reads as annotation.  `cover.globe`'s `rings_front`
and `rays_front` are the same decision for the same reason: a ruler entirely
buried in the ball is not a ruler.

**The intermediate angle stays at HALF of `tip_deg`** (`MIDWAY = 0.5`), and the
better camera is not a reason to move it.  Column 2 has to be as far as it can
from BOTH pictures beside it, and the fraction that maximises `min(f, 1 - f)` is
0.5 and nothing else; every alternative shortens one of the two steps.  Nor is
the projection an argument any more: at `phi` = 14.4 the apparent angle
`2 atan(cos phi tan(a/2))` is within 0.4 % of linear in `a`, so half the tip is
half the drawn tip.  A fraction chosen PER ROW -- larger where the tip is small --
was considered and rejected: that is an exaggeration wearing a formula, and it
would draw B pose 9's 6.34 deg tip as though the middle picture meant the same
thing as everywhere else on the page.

**Where the tip still does not read, and why no camera fixes it.**
`tip_fraction` is 0.6, so `tip_deg = 0.6 x limit_deg`, and that is NOT this
page's to change.  `travel` measures what is left on the drawing -- the largest
distance any vertex covers between column 1 and column 3, over the part's own
on-screen extent -- and `READS` calls it at a tenth of the part's width.  C5's
five corner tips are 28.9-37.9 deg and all five read.  A1-f's poses 3 and 7 are
39.75 deg and read; poses 1 and 5 are 13.7 and 8.0 deg and do not, and BOTH are
cut short by another part of the workpiece hitting the floor at 22.8 and 13.3 deg
where balance alone would have allowed 71.4 and 69.6.  All five of B's are
6.3-23.8 deg because B's own limits are 10.6-39.6 deg.  The log names them with
their `tip_deg` and their `limited_by` and the README says the same: this is the
workpiece, not the drawing.

**The push goes on the far side of the part from the pivot, in its upper half.**
The far side is the largest lever arm.  The upper half is a restriction with a
reason: the tangential velocity of a point is `w x r`, so a point at FLOOR level
far from the axis moves straight UP and one at floor level on the descending side
moves straight DOWN -- so an unrestricted "largest arm" puts the arrow at the
bottom of the part pointing vertically, and a slide reader calls that a lift, or
a push through the floor, and not a tip.

**And the arrow is then brought FORWARD along the eye ray, which cannot move it
on the page.**  A central projection is a scaling about the eye, so shrinking the
arrow toward the eye -- points and width together -- leaves its picture pixel for
pixel where it was and puts it in front of the workpiece.  The per-row camera
does not retire it: that camera is chosen for the pivot and the patch, and the
push a tip needs is still often applied on a face it cannot see.  What it costs
is the arrow's SHADOW, cast from where the arrow now is rather than from where
the push is; it lands as a faint grey streak near the part's own.  Paid
knowingly.

Two traps come with any mujoco camera and both are METHOD s3.0's: mujoco's angles
name the direction the camera looks ALONG where matplotlib's name where the eye
STANDS (so the same view is the elevation negated and the azimuth turned half a
circle), and `mjv_connector` fills only HALF the segment it is handed when the
geom is an arrow, so the push arrow is handed a segment CENTRED on its contact
and lands its head exactly there.

Framing is fitted in SCREEN space to the union of the row's three transforms plus
the pivot plus the arrow, so THE WORLD DOES NOT MOVE between panels and the part
is seen to rotate through it.  Framing each panel on its own pose would have
re-centred the workpiece three times and thrown the motion away.

Colours are `stored_region_demand`'s so the two pages read as one family: the
part warm white, the work region green, both straight out of `cover.shot`'s own
material overrides.  What this page adds gets roles nothing else has spoken for
-- the PIVOT is orange-red, the PUSH is INK (the colour the weight arrow already
uses for "an outside agent acts"), the CENTRE OF MASS and its PLUMB LINE are the
same ink because gravity is that agent, and the TIPPING MARGIN is METHOD s3.0's
red `#B02A26`, which means WHAT IS OWED everywhere else in this project and means
exactly that here.  Blue was available and was not taken: blue means a PROCESS
push (METHOD s3.0) and none of these are one.

**Deterministic, and MEASURED to be** -- run it twice and md5 the three pages.
There is nothing random in this file to seed.  That is worth stating because the
closest page to this one cannot say it: `stored_region_demand.stored_shot` found
that the same renderer -- MuJoCo through `work_regions.SCENE`'s 8x multisampling
and its 4096 shadow map -- settles on one of two images for the same scene from
the same bytes, stable within a process and flipping between them, differing by
fewer than ten pixels at the edge of an ARROW'S SHADOW, and it had to store its
renders on disk to get a byte-exact re-run.  That page draws eight arrows a
panel; this one draws one, on five of its fifteen panels, and does not flip.  So
there is no render store here and no cache to invalidate.  If a page ever does
change under an unchanged script, that is the failure to look for first, and
`stored_region_demand` holds the remedy already written.

Two problems in the STORED data, found here, reported here, and NOT fixed here:
A1-f poses 1 and 2 carry the identical work region -- the same 4469 faces from
the same seed face 146 -- although their `target_area_fraction` differs, which is
`pipeline/region/make_region.py`'s bug and not this page's (pose 1 is drawn here
and pose 2 is not, so the collision no longer shows on the page, and the script
still prints it); and `region.json`'s `shown_area_fraction` measures a different
thing from this page's `seen` -- three turns of a contact sheet against the one
view drawn -- while its name says neither.  Both print on every run.
"""
from __future__ import annotations

import shutil
import sys
from collections import namedtuple
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent          # slides/poses
ROOT = HERE.parents[2]                          # the repository root
sys.path.insert(0, str(ROOT / "slides/tools"))
from mujoco import Renderer
import coordinates as COORD

from common import mat_to_quat_wxyz, obj_path, read_json      # noqa: E402
from cover import AZIM, ELEV, PAPER                           # noqa: E402
from disturbances import _font                                # noqa: E402
from shrink_support import paint                              # noqa: E402
from work_regions import SCENE                                # noqa: E402

OBJECTS = ("A1-f", "B", "C5")
KIND = "point"          # THE CORNER TIPS ONLY.  Five of the ten poses an object,
                        # and which five is read from `tips.json`'s own `pivot`
                        # field rather than typed: a deliberate narrowing, and
                        # `sweep` asserts it got five and that all five are
                        # corners, so five rows can never be a truncated ten
ALL = tuple(range(10))  # what the tipping-margin identity is CHECKED on, drawn
                        # or not: the edge branch of that calculation is the
                        # harder one and a check that never runs is not a check
MIDWAY = 0.5                    # column 2's angle, as a fraction of `tip_deg`
PX = 700                        # one panel, square; SCENE's offscreen buffer is
                                # 1400 x 1400 and this must stay under it
FOVY = 45.0                     # MuJoCo's default, which SCENE keeps
PAD = 1.10                      # the fitted frame's margin: the widest thing on
                                # the panel reaches 1/PAD of the half-frame
FIT_ROUNDS = 24                 # fixed-point rounds of `frame_for`
Z = np.array([0.0, 0.0, 1.0])

# ---- the row's camera.  One per ROW, computed from that row's pivot ----------
# A rotation carries every point round a circle in the plane perpendicular to its
# axis, and that circle projects to an ellipse of axis ratio `cos phi`, with
# `phi` the angle between the axis and the view direction.  So LOOK ALONG THE
# AXIS: `phi` small is the tip drawn at its own size.  A corner pivot is a single
# point and has no contact edge whose apparent length (`sin phi`) has to survive,
# so unlike the edge poses this page no longer draws, there is no compromise here
# and the whole of `cos phi` is taken.
ROW_ELEV = 12.0         # every row -- and it is a FLOOR ON `phi`, exactly: the
                        # pivot axis is horizontal, so an eye tilted up by this
                        # can never come within less than this of it
                        # (`cos phi = cos(ROW_ELEV) cos(OUT)`).  It is therefore
                        # as low as the page's own floor drawing allows: `sin 12
                        # = 0.208` against METHOD s3.0's `sin 14 = 0.242`.  It
                        # pays twice over -- the eye now looks nearly ALONG the
                        # axis and the tipping margin lies along the pivot
                        # NORMAL, so the gap is drawn at 0.99 of its true length
                        # and meets the plumb line at 91.7 deg on the page
BAND = (8.0, 15.0, 22.0, 30.0)  # how far off the axis the eye may swing toward
                        # the OUTSIDE, in degrees.  8 is `phi` = 14.4 and
                        # `cos phi` = 0.969; the ladder is walked outward one
                        # rung at a time and stops the first time the next is not
                        # worth `GAIN` more of the work region.  NOT zero -- and
                        # not because zero measures worse: over the fifteen
                        # drawn cameras `OUT` 0 and `OUT` 8 show the same mean
                        # patch (67.7 %) and bury the same four contacts.  At
                        # zero the eye lies in the vertical plane THROUGH the
                        # axis, its outward component is exactly zero, and
                        # "outside the pivot" stops being true of it.  8 is the
                        # smallest rung that makes the rule true, and it costs
                        # cos 14.4 / cos 12 = 0.991 of the drawn angle
GAIN = 0.05             # how much more of the work region the next rung of BAND
                        # must show before the eye leaves the one it is on

INK = (30, 30, 28)
RULE = (214, 214, 210)
MUTED = (120, 120, 114)
PIVOT = (219, 61, 15)                   # the ground contact(s) the tip turns on
OWED = (176, 42, 38)                    # METHOD s3.0's `#B02A26`, WHAT IS OWED:
                                        # column 3's tipping margin is exactly
                                        # that and gets exactly that colour
PUSH_RGBA = (0.13, 0.13, 0.12, 1.0)     # the outside agent doing the pushing
DISC = 0.034            # the ONE contact, as a fraction of the panel
RING = 2.05             # and the separated ring round it, in disc radii: with a
                        # single point on the floor its ISOLATION is what the row
                        # has to say, so the mark is a target and not a dot
RING_W = 0.008          # the ring's stroke, in panel widths
CASE = 0.008            # the white casing round every flat mark, ditto
PLUMB = 0.008           # the plumb line's width, ditto
DASH = (0.022, 0.016)   # its dash and gap, ditto
GAP_W = 0.010           # the tipping-margin bar's width, ditto
FOOT = 0.028            # the disc where the plumb line lands, ditto
MASS = 0.046            # the centre-of-mass symbol's diameter, ditto
MM = 0.058              # the millimetre label's type size, ditto
LABEL_R = (1.0, 1.30)   # the two radii the millimetre label is tried at, as
                        # multiples of the smallest that cannot overlap the bar
                        # it names: half the label's own diagonal plus half the
                        # bar's length plus `LABEL_AIR`
LABEL_N = 12            # and how many bearings at each radius
LABEL_AIR = 0.022       # the white it keeps round itself, in panel widths
LABEL_PULL = 60.0       # how much ink the label will cross to sit one panel
                        # width nearer the bar it names, in units of darkness
NOTE = (0.26, 0.18)     # the bottom-left corner `page` writes the row's angle
                        # into, kept clear of the millimetre label: `page` draws
                        # it after the tiles are pasted, so nothing in the panel
                        # itself can see it coming
ARROW_L = 0.26          # the push arrow's drawn length, in box diagonals
ARROW_W = 0.014         # and its shaft width, ditto
GRIP = 0.25             # how squarely the push has to go into the face it
                        # lands on: `u . n <= -GRIP`, never a graze, never a
                        # pull
VIEW = 0.75             # how far off the screen plane the arrow may point
                        # before it is foreshortened past reading, |u . fwd|
FRONT = 0.02            # the face the push lands on should turn toward the
                        # camera by at least this much, `facing`'s own test
CLEAR = 0.97            # how far in front of the nearest bit of workpiece the
                        # arrow is brought, as a share of that depth
READS_DEG = 25.0        # a tip READS when it DRAWS at least this many degrees.
                        # Not derived from optics -- CALIBRATED to the judgement
                        # already passed on this page: C5's five corner tips draw
                        # 28.1-36.8 deg and were called legible, A1-f's poses 3
                        # and 7 draw 38.6, and the rows called weak draw
                        # 6.1-23.1.  25 is the gap between those two sets and
                        # reproduces that classification exactly.  It is a
                        # judgement stated as a number, not a measurement
                        # dressed as one
READS = 0.10            # the second test, and the one measured on the page: the
                        # share of the part's own on-screen size the
                        # furthest-travelling vertex covers between column 1 and
                        # column 3.  A long part swung a few degrees about a
                        # corner moves its far end a long way, so this passes
                        # rows the angle fails, and both are reported
SHOWN = 0.25            # under this share of its work region a row is called
                        # hidden, in the log, under either camera
TMP = "_tip_sequence_tmp"       # the painted patch, written under the object and
                                # deleted in a `finally`, `make_region`'s way
HEAD = 0.20                     # the title band, in panel heights
TITLES = ("at rest", "the tip", "the target pose")

Cam = namedtuple("Cam", "elev azim fwd right up lookat dist eye")


# ------------------------------------------------------------ the geometry ---

def rot_about_line(axis, point, ang):
    """The 4x4 that turns the world by `ang` about the line (`point`, `axis`).

    This is the tip itself, and it is CHECKED rather than assumed: composed with
    the stable placement at `ang = sign * tip_deg` it has to reproduce the
    stored `T_world_mesh`, which is what licenses drawing any intermediate angle
    at all.  `sweep` asserts it on every pose of every run.
    """
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    R = np.eye(3) + np.sin(ang) * K + (1.0 - np.cos(ang)) * (K @ K)
    p = np.asarray(point, float)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p - R @ p
    return T


def axes_for(elev, azim):
    """A camera's forward, screen right and screen up, in mujoco's convention.

    `cover.shot`'s own derivation, kept in one place because every hand-projected
    thing on this page is a combination of the three.  `elev` and `azim` are
    where the EYE STANDS (matplotlib's convention, and METHOD s3.0's), so the eye
    sits from the lookat along `(cos elev cos azim, cos elev sin azim, sin elev)`
    and `fwd` is its negative.
    """
    a, e = np.deg2rad(azim + 180.0), np.deg2rad(-elev)
    fwd = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    right = np.array([np.sin(a), -np.cos(a), 0.0])
    return fwd, right, np.cross(right, fwd)


def frame_for(pts, ax):
    """Lookat and distance that FIT `pts` on a square panel, with `PAD` to spare.

    `cover.shot` frames on the bounding box's DIAGONAL, which is right when one
    pose fills one panel and wasteful here: a row's box holds the part at three
    transforms plus the pivot plus the arrow, and framed by its diagonal the
    workpiece came out a third of the panel with white all round it.  A slide
    cannot afford that.  So the frame is fitted in SCREEN space instead -- centre
    on the projected extent, then push the camera out or in until the widest
    point sits at `1/PAD` of the half-frame -- which is a fixed point, reached in
    a few rounds and run for `FIT_ROUNDS` of them so the answer does not depend
    on where it started.  Perspective is why it has to iterate at all: moving the
    camera changes the projection that decided where to move it.
    """
    fwd, right, up = ax
    pts = np.atleast_2d(np.asarray(pts, float))
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    lookat = (lo + hi) / 2
    t = np.tan(np.radians(FOVY / 2))
    dist = float(np.linalg.norm(hi - lo)) / 2 / t
    for _ in range(FIT_ROUNDS):
        for _ in range(2):                       # recentre, then rescale
            v = pts - (lookat - dist * fwd)
            z = v @ fwd
            sx, sy = (v @ right) / z / t, (v @ up) / z / t
            cx, cy = (sx.min() + sx.max()) / 2, (sy.min() + sy.max()) / 2
            lookat = lookat + dist * t * (cx * right + cy * up)
        v = pts - (lookat - dist * fwd)
        z = v @ fwd
        m = max(np.abs((v @ right) / z / t).max(), np.abs((v @ up) / z / t).max())
        dist *= m * PAD
    return lookat, float(dist)


def fit(elev, azim, pts):
    """A whole camera: direction from (`elev`, `azim`), framing from `pts`."""
    fwd, right, up = axes_for(elev, azim)
    lookat, dist = frame_for(pts, (fwd, right, up))
    return Cam(elev, azim, fwd, right, up, lookat, dist, lookat - dist * fwd)


def screen(P, cam):
    """World points -> pixels in a `PX` x `PX` panel, through the row's camera.

    `cover.shot` projects its three triad letters exactly this way and for the
    same reason: mujoco geoms carry no text and no flat overlay, so anything
    that has to sit ON the picture rather than IN it is projected by hand.
    """
    v = np.atleast_2d(np.asarray(P, float)) - cam.eye
    t = np.tan(np.radians(FOVY / 2))
    s = np.stack([v @ cam.right, v @ cam.up], axis=1) / (v @ cam.fwd)[:, None] / t
    return np.stack([(0.5 + 0.5 * s[:, 0]) * PX, (0.5 - 0.5 * s[:, 1]) * PX], axis=1)


def outward(axis, point, com_rest):
    """The horizontal normal of the pivot axis that points AWAY from the part.

    The tip turns the workpiece TOWARD this line, so the far side of the
    footprint lifts and the body leans out over the contact.  Everything the row
    has to show -- the contact itself, the lean, and the plumb line landing on
    the wrong side of it -- is nearest an eye standing out here, and the
    resting centre of mass is the cheapest thing on disk that says which side
    `here` is.  `point` is on the boundary of the footprint and the resting
    centre of mass is strictly inside it, so the sign is never in doubt; the
    assert says so rather than assuming it.
    """
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    assert abs(a[2]) < 1e-12, (
        f"the pivot axis is not horizontal ({a[2]:.3e} in z) -- it is supposed "
        f"to be a supporting line of the footprint, lying on the floor")
    n = np.array([a[1], -a[0], 0.0])
    n /= np.linalg.norm(n)
    s = float(n @ (np.asarray(com_rest, float) - np.asarray(point, float)))
    assert abs(s) > 1e-6, (
        f"the resting centre of mass sits on the pivot line to {s:.3e} m, so "
        f"there is no outside to stand on")
    return -n if s > 0 else n


def look(nrm, axis, elev, out, side):
    """The azimuth of an eye at `elev` looking `out` degrees off `axis`, OUTSIDE.

    The horizontal plane is spanned by the pivot axis and its outward normal, so
    the eye's horizontal direction is `sin(out) * nrm + side * cos(out) * axis`
    -- `out = 0` is straight along the axis, and every `out` in `BAND` leaves a
    positive component along the outward normal, which is what "outside" means
    here.  Tilt that up by `elev` and

        cos phi = |ehat . ahat| = cos(elev) * cos(out)

    so the elevation is a FLOOR on `phi` and `out` is what is left to choose.
    `choose` asserts the `phi` it gets back against this identity.
    """
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    c, o = np.cos(np.radians(elev)), np.radians(out)
    e = c * (np.sin(o) * np.asarray(nrm, float) + side * np.cos(o) * a) \
        + np.sin(np.radians(elev)) * Z
    return float(np.degrees(np.arctan2(e[1], e[0])))


def phi_of(elev, out):
    """The angle between the view direction and the pivot axis, in degrees."""
    return float(np.degrees(np.arccos(np.cos(np.radians(elev))
                                      * np.cos(np.radians(out)))))


def apparent_tip(tip, phi):
    """How big the tip draws, in degrees, at `phi` off its own axis.

    The circle a point traces about the axis projects to an ellipse of axis ratio
    `cos phi`, so a turn of `tip` about that axis, taken symmetrically about the
    compressed direction, subtends

        2 atan(cos phi * tan(tip / 2))

    on the page.  This is the number the whole camera argument is about: at the
    45 deg the edge poses needed it took A1-f pose 3's 39.75 deg tip down to
    28.7, and at `phi` = 14.4 it draws 38.6.
    """
    return float(np.degrees(2 * np.arctan(np.cos(np.radians(phi))
                                          * np.tan(np.radians(tip) / 2))))


def margin(mesh, T, ex):
    """METHOD s11.4's `w(0)`: the arm the supports owe with the process OFF.

    Gravity pulls one body weight down at the centre of mass; the floor pushes
    the same weight up at a contact that is NOT under it; the pair is a couple
    and no floor reaction can cancel it.  The floor may split its weight across
    the ground contacts however it likes -- two of them on an edge pivot, one on
    a corner -- so what is owed is the SMALLEST couple that split can leave:

        w(0) = min over lam >= 0, sum lam = 1 of |SUM_i lam_i (p_i - c) x zhat|

    computed here with real cross products in three dimensions.  On an edge that
    is a one-parameter quadratic in `lam` and is minimised in closed form and
    then CHECKED against a sweep of the whole segment.

    What the page draws is the flat horizontal distance from where the plumb line
    lands to that contact, which is a different calculation entirely -- two
    coordinates, no cross product, no minimisation -- and this asserts the two
    agree.  METHOD s11.4 records the same equality holding to `6.9e-18 m`, and
    over these three objects and `cuboid_baseline` the worst case is
    6.938893903907228e-18 m on cuboid pose 5, which is that number: s11.4's
    figure IS this quantity.

    Returns the arm in metres, the contact the floor should push at, the point
    the plumb line lands on, how the floor split its weight, and how far the two
    calculations disagreed.
    """
    c = T[:3, :3] @ mesh.center_mass + T[:3, 3]
    a = np.asarray(ex["axis"], float)
    a = a / np.linalg.norm(a)
    p = np.asarray(ex["point"], float)
    if ex["pivot"] == "edge":
        hl = float(ex["half_length"])
        p0, p1 = p - hl * a, p + hl * a
        t0, d = np.cross(p0 - c, Z), np.cross(p1 - p0, Z)
        lam = float(np.clip(-(t0 @ d) / (d @ d), 0.0, 1.0))
        q = p0 + lam * (p1 - p0)
        grid = np.linspace(0.0, 1.0, 2001)[:, None]
        swept = float(np.linalg.norm(np.cross((p0 + grid * (p1 - p0)) - c, Z),
                                     axis=1).min())
        assert float(np.linalg.norm(np.cross(q - c, Z))) <= swept + 1e-15, (
            "the closed-form split of the floor's reaction is not the minimum "
            "the segment allows")
    else:
        lam, q = None, p
    owed = float(np.linalg.norm(np.cross(q - c, Z)))
    foot = np.array([c[0], c[1], 0.0])
    gap = float(np.hypot(*COORD.floor(q-foot)))
    assert abs(owed - gap) <= 1e-17, (
        f"the couple's arm is {owed:.9e} m and the horizontal distance this page "
        f"draws between the plumb point and the contact is {gap:.9e} m; METHOD "
        f"s11.4 records those agreeing to 6.9e-18 m and they differ by "
        f"{abs(owed - gap):.3e} m, so the drawn gap is not the tipping margin")
    return owed, q, foot, lam, abs(owed - gap)


def push_site(mesh, T, axis, point, sign, fwd):
    """Where the push goes, which way it points, and how hard that was.

    THE FAR SIDE OF THE PART FROM THE PIVOT -- the largest lever arm, which is
    the brief -- taken over the faces that can carry the arrow.  Four conditions
    stand in front of that `argmax`, listed in the order they give way:

    1. `u . n <= -GRIP`.  A contact can only PUSH, never pull, so the push goes
       INTO the face.  It is also what puts the SHAFT outside the part:
       `mjv_connector` lands the head on the contact and hangs the shaft behind
       it along `-u`, which leaves the surface exactly when `u` enters it.  THIS
       ONE NEVER RELAXES -- an arrow that is not a push is not worth drawing.
    2. THE UPPER HALF of the part.  The tangential velocity is `w x r`, so at
       floor level it is vertical -- straight up on the rising side, straight
       down on the descending one -- and an unrestricted largest arm therefore
       puts the arrow at the bottom of the part pointing vertically, which a
       slide reader calls a lift, or a push through the floor, and not a tip.
    3. `|u . fwd| <= VIEW`: the arrow is not foreshortened to a stub by pointing
       along the eye.
    4. `n . fwd <= -FRONT`: the face it lands on turns toward the camera.  Last,
       and the first to give way, because `pull_forward` has already made the
       arrow visible whatever face it sits on; all this buys is that the head
       lands somewhere the reader can see it land.

    The ladder is walked best-first and the rung it stops on is REPORTED, since
    a rung below the first is a pose whose natural push is somewhere awkward and
    the drawing says so.  Deterministic: `argmax` breaks its ties on the lower
    index, and nothing here is sampled.

    What is NOT in this list is whether the arrow would be OCCLUDED, and that is
    deliberate: `pull_forward` settles it afterwards, for every pose at once and
    without moving the arrow on the page.  Filtering for it here instead cost
    half the poses their natural push site and still left A1-f pose 3 with no
    arrow at all.

    The direction is the site's own velocity under the tip, `w x r` with
    `w = sign * axis`, so the arrow cannot disagree with the rotation drawn
    beside it.  It is not always a horizontal shove: for a corner pivot the axis
    is any line through the corner, so a face can sit on the DESCENDING side of
    it and its arrow points down -- pressing one end to lever the other up,
    which is a tip and is drawn as one.
    """
    R = T[:3, :3]
    c = mesh.triangles_center @ R.T + T[:3, 3]
    n = mesh.face_normals @ R.T
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    r = c - np.asarray(point, float)
    u = np.cross(sign * a, r)
    u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-15)
    arm = np.linalg.norm(r - np.outer(r @ a, a), axis=1)
    push = (u * n).sum(axis=1) <= -GRIP
    high = c[:, 2] >= 0.5 * (c[:, 2].min() + c[:, 2].max())
    front = n @ fwd <= -FRONT
    flat = np.abs(u @ fwd) <= VIEW
    for sel, why in ((push & high & front & flat, ""),
                     (push & high & flat, "the face it lands on is turned away"),
                     (push & high & front, "it points along the eye"),
                     (push & high, "it points along the eye AND the face it "
                                   "lands on is turned away"),
                     (push, "the only faces that can be pushed into are in the "
                            "part's lower half")):
        if sel.any():
            break
    assert sel.any(), (
        "no face of this part can be pushed into so as to turn it the way the "
        "stored tip turns it -- there is no arrow to draw")
    j = int(np.argmax(np.where(sel, arm, -np.inf)))
    return c[j], u[j], why


def pull_forward(a, b, w, cam, near):
    """The arrow brought in FRONT of the workpiece, without moving on the page.

    A central projection is a scaling about the EYE, so shrinking a thing toward
    the eye by a factor `k` -- points AND width together -- leaves its picture
    pixel for pixel where it was and puts it nearer the camera.  Applied so the
    whole arrow sits just inside the nearest bit of workpiece, that makes the
    push arrow unoccludable while keeping it a lit 3d object rather than a flat
    sticker, and without the site having to be chosen for visibility.

    It is the same decision as drawing the pivot flat over the render, and it is
    forced by the same thing.  The per-row camera does not retire it: that camera
    is chosen for the PIVOT and for the work region, and the push a tip needs is
    still often applied on a face it cannot see.

    What it costs is the arrow's SHADOW, which is cast from where the arrow now
    is rather than from where the push is; it lands as a faint grey streak near
    the part's own shadow.  Paid knowingly: a shadow in the wrong place is a
    smaller error on a slide than an arrow that is not there.
    """
    k = min(1.0, CLEAR * near / max((a - cam.eye) @ cam.fwd, (b - cam.eye) @ cam.fwd))
    return cam.eye + k * (a - cam.eye), cam.eye + k * (b - cam.eye), k * w


def buried(mesh, T, pts, cam):
    """Would these ground contacts be hidden by the workpiece, drawn in scene?

    The claim that the pivot HAS to be drawn flat over the render, rather than
    as spheres in it, is worth a number rather than an assertion, so this is the
    number: one ray from each contact toward the eye, against the mesh in the
    pose it is drawn in.  It changes nothing about the picture -- the pivot goes
    on top either way -- and it is measured under BOTH cameras so the per-row
    camera's share of the credit is visible rather than claimed.
    """
    R = T[:3, :3]
    pts = np.atleast_2d(np.asarray(pts, float))
    d = cam.eye - pts
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    eps = 1e-4 * float(np.linalg.norm(mesh.extents))
    return bool(mesh.ray.intersects_any(
        ray_origins=((pts + eps * d) - T[:3, 3]) @ R,
        ray_directions=d @ R).any())


def facing(mesh, T, inside, cam):
    """How much of the green patch this camera actually shows.

    `stored_region_demand.facing`, unchanged in substance and pointed at this
    page's own fitted camera instead of at `cover.shot`'s.  It was a PRICE under
    the old fixed camera and it is a CRITERION now: the row's azimuth may go
    either way round the pivot and its `phi` may sit anywhere in `BAND`, and this
    is what settles both.  Reported under both cameras on every pose.
    """
    idx = np.flatnonzero(inside)
    c, n = mesh.triangles_center[idx], mesh.face_normals[idx]
    R, t = T[:3, :3], T[:3, 3]
    d = cam.eye - (c @ R.T + t)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    seen = (((n @ R.T) * d).sum(axis=1) > 0.02) & ~mesh.ray.intersects_any(
        ray_origins=c + n * 1e-5, ray_directions=d @ R)
    a = mesh.area_faces[idx]
    return float(a[seen].sum() / a.sum())


def travel(mesh, T_rest, T_star, cam):
    """How far the tip actually moves the part ACROSS THE PAGE, as a share.

    The largest distance any vertex covers between column 1 and column 3, over
    the diagonal of the part's own on-screen extent at rest.  This is the honest
    test of "does it read as a tip", because it is measured on the drawing the
    reader is given: it already contains the camera, the framing and the angle.
    `tip_fraction` is 0.6 and is not this page's to change, so a row that fails
    it is reported and left alone.
    """
    V = np.asarray(mesh.vertices)
    s1 = screen(V @ T_rest[:3, :3].T + T_rest[:3, 3], cam)
    s3 = screen(V @ T_star[:3, :3].T + T_star[:3, 3], cam)
    ext = float(np.linalg.norm(s1.max(axis=0) - s1.min(axis=0)))
    return float(np.linalg.norm(s1 - s3, axis=1).max() / ext)


def choose(mesh, ex, T_star, inside, com_rest, frame):
    """The row's camera: along the tip axis, a little outside, then measured.

    The eye starts on `BAND`'s inner rung -- as near the axis as `ROW_ELEV` and
    "outside" together allow -- and WALKS OUTWARD one rung at a time, stopping the
    first time the next rung does not show at least `GAIN` more of the STORED work
    region than the one it is on.  Walking rather than taking the best of the
    ladder is the point: every degree off the axis is paid for out of the tip's
    own drawn size, so a distant rung has to earn its way past every rung between,
    not merely beat the start.  Thirteen of the fifteen rows stop on the first;
    A1-f pose 3 walks to 15 deg and C5 pose 1 to 22, because their patches are
    nearly edge-on until they do.

    The side is free at every rung and goes to whichever shows more, ties to `+1`.

    **Every candidate is measured on the camera it would actually be DRAWN with**
    -- `frame` is the row's own closure, which picks the push site for that
    direction and refits the frame to include the arrow's tail -- because the two
    are not interchangeable.  A1-f pose 3's patch is a large flat face sitting
    within a hair of edge-on to this eye, and `facing`'s test is a step function
    on a flat face: pulling the camera back the 6 % the arrow costs takes that
    pose from 70.5 % to 43.4 %.  Choosing on a frame the page does not draw would
    have been choosing on the wrong number.
    """
    a = np.asarray(ex["axis"], float)
    a = a / np.linalg.norm(a)
    nrm = outward(a, ex["point"], com_rest)

    def rung(out):
        best = None
        for side in (1.0, -1.0):
            cam = frame(ROW_ELEV, look(nrm, a, ROW_ELEV, out, side))[0]
            got = float(np.degrees(np.arccos(min(1.0, abs(cam.fwd @ a)))))
            assert abs(got - phi_of(ROW_ELEV, out)) < 1e-9, (
                f"asked for a view {out} deg off the axis at elevation "
                f"{ROW_ELEV} and got {got:.6f} deg from it")
            assert (cam.eye - cam.lookat) @ nrm > 0, (
                "the eye did not land outside the pivot")
            seen = facing(mesh, T_star, inside, cam)
            if best is None or seen > best[0]:
                best = (seen, out, side, cam)
        return best

    got = rung(BAND[0])
    for out in BAND[1:]:
        nxt = rung(out)
        if nxt[0] <= got[0] + GAIN:
            break
        got = nxt
    return got


# ------------------------------------------------------------- the drawing ---

def dashed(dr, p, q, width, colour):
    """A straight dashed line between two pixel points, PIL having none."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    L = float(np.linalg.norm(q - p))
    if L < 1e-9:
        return
    u = (q - p) / L
    on, off = DASH[0] * PX, DASH[1] * PX
    s = 0.0
    while s < L:
        e = min(s + on, L)
        dr.line([tuple(p + s * u), tuple(p + e * u)], fill=colour, width=int(width))
        s = e + off


def place(im, at, half, size):
    """Where the millimetre label goes: the clearest space near the bar it names.

    The label has to sit ON the panel, and the panel is a photograph -- there is
    no margin to put it in and no reliable empty corner, because which corner is
    empty changes with the row's camera.  A perpendicular offset is no good
    either: on an edge pivot the gap is BY CONSTRUCTION perpendicular to the
    contact line, so stepping off the bar sideways steps straight along the
    pivot and lands on it.

    So it is measured.  `LABEL_N` bearings at each of `LABEL_R` radii round the
    middle of the bar, each scored by how much ink its box would cover -- floor
    is nothing, the part a little, the shadow more, the work region more again,
    and the pivot and the centre of mass most of all -- plus `LABEL_PULL` for
    every panel width it sits away from what it names, so the clearest distant
    corner never beats a clear space right beside the bar.  Deterministic: a
    fixed candidate list, scored, `argmin`, ties to the lower index.
    """
    a = np.asarray(im.convert("RGB"), np.int16)
    ink = (255 - a.min(axis=2)).astype(np.float64)
    ink[int((1 - NOTE[1]) * PX):, :int(NOTE[0] * PX)] = 255.0
    ink *= ink / 255.0          # SQUARED, so the cost of crossing a mark is
                                # nothing like the cost of crossing the part:
                                # floor 0.3, the part 6, its shadow 28, the work
                                # region 52, the pivot and the centre of mass
                                # 226.  A linear score treats a label lying over
                                # the contact line as slightly worse than one
                                # lying over a white face, which is not what a
                                # reader finds
    w, h = size
    best, at = None, np.asarray(at, float)
    clear = float(np.hypot(w, h)) / 2 + half + LABEL_AIR * PX
    m = LABEL_AIR * PX
    for k in LABEL_R:
        r = k * clear
        for k in range(LABEL_N):
            t = 2 * np.pi * k / LABEL_N
            c = at + r * np.array([np.cos(t), np.sin(t)])
            c = np.clip(c, [w / 2 + m, h / 2 + m], [PX - w / 2 - m, PX - h / 2 - m])
            x0, y0 = int(c[0] - w / 2), int(c[1] - h / 2)
            cost = float(ink[y0:y0 + h, x0:x0 + w].mean()) \
                + LABEL_PULL * float(np.linalg.norm(c - at)) / PX
            if best is None or cost < best[0]:
                best = (cost, c)
    return best[1]


def overlay(im, cam, draw):
    """Everything that sits ON the picture rather than IN it, in a fixed order.

    The pivot, the plumb line, the centre of mass and the tipping-margin bar are
    all projected by hand and painted flat, for one reason each of them shares:
    they are either behind the workpiece or inside it.  `buried` counts the first
    case per pose; the second needs no counting, because the centre of mass is
    interior to a watertight solid on all thirty poses and no camera anywhere
    sees it.

    The order is fixed HERE and not by the caller, because it is a z-order and it
    is part of what the marks mean: the margin bar runs UNDER the pivot, so the
    pivot's white casing cuts it exactly where the contact is, and the centre of
    mass goes on top of everything because it is the thing the plumb line hangs
    from.
    """
    dr = ImageDraw.Draw(im)
    rad, case = DISC * PX / 2, CASE * PX
    want = {k: [a for j, a in draw if j == k]
            for k in ("plumb", "gap", "pivot", "mass")}

    for com, foot in want["plumb"]:
        p, q = screen(np.vstack([com, foot]), cam)
        dashed(dr, p, q, PLUMB * PX + 2 * case, (255, 255, 255))
        dashed(dr, p, q, PLUMB * PX, INK)

    labels, feet = [], []
    for foot, contact, com, mm in want["gap"]:
        p, q = screen(np.vstack([foot, contact]), cam)
        # the bar, then the disc the plumb line lands in.  Three marks carry the
        # whole of column 3: an INK disc where the weight comes down, a RED bar,
        # and the ORANGE contact at the other end of it.  The bar needs no ticks
        # once both its ends are things
        for w, col in ((GAP_W * PX / 2 + case, (255, 255, 255)), (GAP_W * PX / 2, OWED)):
            dr.line([tuple(p), tuple(q)], fill=col, width=int(2 * w))
        feet.append(p)
        labels.append(((p + q) / 2, float(np.linalg.norm(q - p)) / 2,
                       f"{1000 * mm:.1f} mm"))

    # THE ONE CONTACT.  A filled disc inside a ring it does not touch: with a
    # single point of the workpiece on the floor, what the row has to say is that
    # there is ONE of them and nothing else, and a bare dot says "a contact"
    # where a target says "this contact, on its own".
    for (pt,) in want["pivot"]:
        c = screen(pt, cam)[0]

        def circle(r, fill=None, outline=None, w=0):
            dr.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r],
                       fill=fill, outline=outline, width=w)

        out = RING * rad
        circle(out + case, fill=(255, 255, 255))
        circle(out, outline=PIVOT, w=int(RING_W * PX))
        circle(rad + case, fill=(255, 255, 255))
        circle(rad, fill=PIVOT)

    # the foot disc goes ON TOP of the pivot, not under it: on B pose 6 the
    # margin is 0.95 mm and the two marks land on each other, and the one that
    # has to survive is WHERE THE WEIGHT COMES DOWN
    for p in feet:
        for rr, col in ((FOOT * PX / 2 + case, (255, 255, 255)), (FOOT * PX / 2, INK)):
            dr.ellipse([p[0] - rr, p[1] - rr, p[0] + rr, p[1] + rr], fill=col)

    for (com,) in want["mass"]:
        c = screen(com, cam)[0]
        r = MASS * PX / 2
        box = [c[0] - r, c[1] - r, c[0] + r, c[1] + r]
        dr.ellipse([box[0] - case, box[1] - case, box[2] + case, box[3] + case],
                   fill=(255, 255, 255))
        dr.ellipse(box, fill=(255, 255, 255), outline=INK, width=max(2, int(case)))
        # the ISO centre-of-mass symbol: a circle with two opposite quadrants
        # filled.  It says `centre of mass` to an engineer without a word on it,
        # which is the only kind of label this page can afford.
        for a0 in (180, 0):
            dr.pieslice(box, a0, a0 + 90, fill=INK)
        dr.ellipse(box, outline=INK, width=max(2, int(case)))

    # last, so `place` scores the label against the finished panel and not
    # against a picture half of its own marks are still missing from
    for at, half, text in labels:
        f = _font(int(MM * PX))
        x0, y0, x1, y1 = dr.textbbox((0, 0), text, font=f)
        w, h = int(x1 - x0), int(y1 - y0)
        dr.text(tuple(place(im, at, half, (w, h))), text, fill=OWED, font=f,
                anchor="mm", stroke_width=int(case), stroke_fill=(255, 255, 255))
    return im


def render(name, T, parts, cam, draw=()):
    """One panel: the part at `T` on the floor, down the row's camera.

    `work_regions.SCENE` with `cover.shot`'s two material overrides VERBATIM --
    part warm white, work region green -- so this page and
    `setup_test/stored_region_demand`'s are the same picture of the same
    workpiece.  What is not `cover.shot`'s is the camera, which is handed in
    because all three columns of a row share one and every row has its own, and
    the flat overlay, which goes on the finished image.
    """
    obj = obj_path(name)
    # THE SKY IS FLATTENED TO WHITE, and only here.  `work_regions.SCENE` gives
    # the skybox a gradient from white at the bottom to 0.94 at the top, which is
    # invisible at METHOD s3.0's 14 deg and at the 30 deg this page used to sit
    # at, because at those elevations THE HORIZON IS OUT OF FRAME -- the view
    # axis is more than the 22.5 deg half-frame below the horizontal.  At 12 deg
    # it is not, and the lit floor meets a 0.94 sky in a hard grey band across
    # the top quarter of every panel, which a reader takes for a wall.  Flat
    # white removes it and costs nothing: these panels are cropped to the
    # workpiece and the gradient was never doing any work in them.  A string
    # replacement on this script's own copy; `slides/tools/work_regions.py` is untouched.
    xml = SCENE.replace(
        'rgb1="1 1 1" rgb2="0.94 0.94 0.95"', 'rgb1="1 1 1" rgb2="1 1 1"').replace(
        '<material name="work" rgba="0.16 0.68 0.40 1" specular="0.15" shininess="0.2"/>',
        '<material name="work" rgba="0.55 0.82 0.62 1" specular="0.1"/>').replace(
        '<material name="rest" rgba="0.85 0.79 0.68 1" specular="0.1"/>',
        '<material name="rest" rgba="0.90 0.89 0.85 1" specular="0.1"/>\n'
        '    <material name="spare" rgba="0.90 0.89 0.85 1" specular="0.1"/>').format(
        assets="\n".join(f'    <mesh name="p{i}" file="{f}"/>'
                         for i, (f, _) in enumerate(parts)),
        geoms="\n".join(f'      <geom type="mesh" mesh="p{i}" material="{m}"/>'
                        for i, (_, m) in enumerate(parts)),
        pos=" ".join(f"{v:.9g}" for v in T[:3, 3]),
        quat=" ".join(f"{v:.9g}" for v in mat_to_quat_wxyz(T[:3, :3])))
    tmpx = obj / ".tip_sequence.xml"
    tmpx.write_text(xml)
    try:
        model = mujoco.MjModel.from_xml_path(str(tmpx))
    finally:
        tmpx.unlink(missing_ok=True)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cm = mujoco.MjvCamera()
    # METHOD s3.0: mujoco's angles name the direction the camera looks ALONG,
    # matplotlib's name where the eye stands, so the same view is the elevation
    # negated and the azimuth turned half a circle.
    cm.azimuth, cm.elevation = cam.azim + 180.0, -cam.elev
    cm.lookat[:] = cam.lookat
    cm.distance = cam.dist
    with Renderer(model, PX, PX, max_geom=64) as r:
        r.update_scene(data, camera=cm)
        scn = r.scene
        for kind, args in draw:
            if kind != "arrow":
                continue
            tail, beyond, w = args
            # METHOD s3.0: `mjv_connector` fills only HALF the segment it is
            # handed when the geom is an arrow, so it is handed one CENTRED on
            # the contact -- `tail` to `beyond`, the contact halfway along --
            # and the head lands exactly on the contact.
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3),
                                np.zeros(3), np.zeros(9),
                                np.array(PUSH_RGBA, np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, float(w),
                                 np.asarray(tail, float),
                                 np.asarray(beyond, float))
            scn.ngeom += 1
        im = Image.fromarray(r.render())
    return overlay(im, cam, draw)


def page(out, name, grid, labels, notes):
    """Ten rows pasted flush under one title band.

    The band carries the object and the three column names; a row carries its
    pose number in the gutter and, in columns 2 and 3, the angle it is drawn at.
    The only other words on the sheet are column 3's millimetres, printed on the
    bar they measure, where a caption cannot be mistaken for a caption of
    something else.  `stored_region_demand` allows itself only the pose number;
    the extra here is what a viewer needs to read a row without a caption, which
    is what this page is for.
    """
    ch = grid[0][0].size[1]
    head = int(ch * HEAD)
    lab, tit, small = _font(int(ch * .085)), _font(int(ch * .095)), _font(int(ch * .062))
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lw = int(max(probe.textlength(t, font=lab) for t in labels) + ch * .12)
    widths = [im.size[0] for im in grid[0]]
    im = Image.new("RGB", (lw + sum(widths), head + len(grid) * ch), PAPER)
    dr = ImageDraw.Draw(im)
    dr.text((int(ch * .06), head // 2), name, fill=INK, font=tit, anchor="lm")
    x = lw
    for w, t in zip(widths, TITLES):
        dr.text((x + w // 2, head // 2), t, fill=INK, font=tit, anchor="mm")
        x += w
    dr.line([(0, head), (im.size[0], head)], fill=RULE, width=2)
    for i, (row, text, note) in enumerate(zip(grid, labels, notes)):
        y = head + i * ch
        x = lw
        for j, tile in enumerate(row):
            im.paste(tile, (x, y))
            if note[j]:
                dr.text((x + int(ch * .05), y + ch - int(ch * .05)), note[j],
                        fill=MUTED, font=small, anchor="lb")
            x += tile.size[0]
        dr.text((int(ch * .06), y + ch // 2), text, fill=INK, font=lab, anchor="lm")
        if i:
            dr.line([(0, y), (im.size[0], y)], fill=RULE, width=1)
    im.save(out)
    print(f"-> {out}  {im.size[0]} x {im.size[1]} px", flush=True)


# ------------------------------------------------------------------- sweep ---

def sweep(name):
    """One object: its five corner rows drawn, and every number they rest on."""
    d = obj_path(name)
    mesh = trimesh.load(d / "mesh.stl", force="mesh")
    placements = read_json(d / "poses.json")["poses"]
    filed = read_json(d / "tips" / "tips.json")
    examples, frac = filed["examples"], filed["tip_fraction"]
    region = read_json(d / "region" / "region.json")
    assert len(examples) >= len(ALL), (
        f"{name}: tips.json holds {len(examples)} examples and this page reads "
        f"all {len(ALL)} of them, drawn or not")
    kinds = [examples[p]["pivot"] for p in ALL]
    # WHICH FIVE, read off the data and not typed.  A deliberate narrowing to the
    # corner tips is not a truncated ten and must not be able to become one by
    # accident, so the selection is derived and then checked.
    POSES = tuple(p for p in ALL if examples[p]["pivot"] == KIND)
    assert len(POSES) == 5 and all(examples[p]["pivot"] == KIND for p in POSES), (
        f"{name}: expected five {KIND} pivots among the ten poses and found "
        f"{len(POSES)} ({''.join(k[0] for k in kinds)})")
    print(f"\n=== {name}   {len(mesh.faces)} faces, tip_fraction {frac:g}, region "
          f"band {region['band']}   the ten poses are "
          f"{kinds.count('edge')} edge / {kinds.count('point')} corner "
          f"({' '.join(k[0] for k in kinds)}); THIS PAGE DRAWS THE "
          f"{len(POSES)} CORNER TIPS, poses {', '.join(map(str, POSES))}",
          flush=True)
    # WHY THE CENTRE OF MASS CANNOT BE A GEOM.  It is not a camera problem like
    # the pivot's: on A1-f and B the centre of mass is INSIDE a watertight solid,
    # so no view recovers it; and on C5 it is not on the workpiece at all --
    # 4.85 mm OUTSIDE the surface, in the void a thin open frame encloses -- so a
    # sphere there would float in mid-air and read as a part of the scene.  Both
    # roads end at a mark drawn flat over the render.  Measured every run.
    depth = float(trimesh.proximity.signed_distance(mesh, [mesh.center_mass])[0])
    print(f"  the centre of mass is {abs(1000 * depth):.2f} mm "
          f"{'INSIDE' if depth > 0 else 'OUTSIDE'} the surface (watertight "
          f"{mesh.is_watertight}, winding consistent {mesh.is_winding_consistent})"
          f", so it is drawn flat over the render: "
          + ("a sphere there is buried in solid on every pose"
             if depth > 0 else
             "a sphere there floats in the void this frame encloses"), flush=True)

    # THE TIPPING MARGIN ON ALL TEN, drawn or not.  `margin`'s edge branch is the
    # one with a minimisation in it and it would otherwise never run.
    every = {p: margin(mesh, np.asarray(examples[p]["T_world_mesh"]), examples[p])
             for p in ALL}
    print("  METHOD \N{SECTION SIGN}11.4's w(0) over ALL TEN poses "
          "(\N{DAGGER} drawn): "
          + "  ".join(f"{1000 * every[p][0]:.2f}"
                      + ("\N{DAGGER}" if examples[p]["pivot"] == KIND else "")
                      for p in ALL)
          + f" mm; the couple's arm and the horizontal distance this page draws "
            f"agree to {max(every[p][4] for p in ALL):.3e} m "
            f"(\N{SECTION SIGN}11.4 records 6.9e-18 m)", flush=True)

    tmp = d / TMP
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    rows = []
    # the stored regions are an INPUT and are not touched here, but two poses
    # sharing one is worth saying out loud even when only one of them is drawn
    same = {}
    for pose in ALL:
        same.setdefault(tuple(region["poses"][str(pose)]["faces"]), []).append(pose)
    for faces, poses in same.items():
        if len(poses) > 1:
            print(f"  STORED DATA: poses {', '.join(map(str, poses))} carry the "
                  f"SAME work region \N{EM DASH} the same {len(faces)} faces, "
                  f"grown from the same seed face "
                  f"{region['poses'][str(poses[0])]['seed_face']}, at "
                  f"target_area_fraction "
                  f"{', '.join(f'{region['poses'][str(p)]['target_area_fraction']:g}' for p in poses)}"
                  f". Read, not fixed: make_region wrote it, region.json is an "
                  f"input to this page. Drawn here: "
                  f"{', '.join(str(p) for p in poses if p in POSES) or 'neither'}",
                  flush=True)
    try:
        # ONE bare part for all ten un-painted panels: identical content under one
        # filename, so mujoco's mesh-asset cache is a saving here rather than the
        # trap `stored_region_demand` records (a name reused across poses with
        # DIFFERENT content comes back as the patch that compiled first)
        bare = paint(mesh, np.eye(4), np.zeros(len(mesh.faces), bool), set(),
                     tmp, f"tip_{name}_bare", rel=TMP)
        V = np.asarray(mesh.vertices)
        grid, labels, notes = [], [], []
        for pose in POSES:
            ex = examples[pose]
            T_star = np.asarray(ex["T_world_mesh"])
            T_rest = np.asarray(placements[ex["placement"]]["T_world_mesh"])
            axis, point, sign = ex["axis"], ex["point"], float(ex["sign"])
            tip, mid = float(ex["tip_deg"]), MIDWAY * float(ex["tip_deg"])
            # the tip, CHECKED and not assumed: the same construction at the full
            # angle has to land on the stored T*
            err = float(np.abs(rot_about_line(axis, point, np.radians(sign * tip))
                               @ T_rest - T_star).max())
            assert err < 1e-9, (
                f"{name} pose {pose}: turning the stable placement by "
                f"sign*tip_deg about (point, axis) misses the stored "
                f"T_world_mesh by {err:.3e} -- the stored tip is not the rotation "
                f"it says it is, and no intermediate angle drawn from it would "
                f"mean anything")
            T_mid = rot_about_line(axis, point, np.radians(sign * mid)) @ T_rest

            Vs = [V @ t[:3, :3].T + t[:3, 3] for t in (T_rest, T_mid, T_star)]
            marks = np.asarray(point, float)[None]        # ONE contact, always
            box = np.vstack(Vs + [marks])
            com_rest = T_rest[:3, :3] @ mesh.center_mass + T_rest[:3, 3]
            com_star = T_star[:3, :3] @ mesh.center_mass + T_star[:3, 3]
            arm, contact, foot, lam, ident = every[pose]

            def frame(elev, azim):
                """The whole drawn camera for one direction, arrow and all.

                Handed to `choose` as well as used for the drawing, so that a
                candidate is scored on the picture it would produce.  The box is
                grown in two passes because the arrow's own length is quoted in
                box diagonals: the part and the pivot first, then the arrow tail
                that scale implies.
                """
                cam0 = fit(elev, azim, box)
                q, u, why = push_site(mesh, T_mid, axis, point, sign, cam0.fwd)
                size = float(np.linalg.norm(box.max(axis=0) - box.min(axis=0)))
                tail, beyond = q - ARROW_L * size * u, q + ARROW_L * size * u
                cam = fit(elev, azim, np.vstack([box, tail[None]]))
                near = float(((Vs[1] - cam.eye) @ cam.fwd).min())
                return cam, pull_forward(tail, beyond, ARROW_W * size, cam, near), why

            ent = region["poses"][str(pose)]
            inside = np.zeros(len(mesh.faces), bool)
            inside[np.asarray(ent["faces"], int)] = True

            # WHAT THE FIXED CAMERA WOULD HAVE SHOWN, computed here so the
            # improvement is measured in the same run rather than quoted from a
            # previous one: METHOD s3.0's direction, this page's framing, its own
            # push site, its own numbers.
            old, _, _ = frame(ELEV, AZIM)
            was_seen = facing(mesh, T_star, inside, old)
            was_dark = buried(mesh, T_mid, marks, old)

            seen, out, side, cam = choose(mesh, ex, T_star, inside, com_rest, frame)
            _, arrow, why = frame(cam.elev, cam.azim)
            phi = phi_of(ROW_ELEV, out)
            if why:
                print(f"      the push arrow is drawn where it is even though "
                      f"{why}: nothing better exists on this pose", flush=True)
            dark = buried(mesh, T_mid, marks, cam)
            moved = travel(mesh, T_rest, T_star, cam)
            drawn = apparent_tip(tip, phi)
            was_drawn = apparent_tip(tip, 45.0)

            work = paint(mesh, np.eye(4), inside, set(), tmp,
                         f"tip_{name}_p{pose}", rel=TMP)
            two = [("arrow", arrow), ("pivot", (marks,))]
            # column 3's: the weight hanging off the one contact, and by how much
            three = [("pivot", (marks,)), ("plumb", (com_star, foot)),
                     ("gap", (foot, contact, com_star, arm)), ("mass", (com_star,))]
            print(f"  pose {pose}  placement {ex['placement']}  ONE contact  "
                  f"tip {tip:5.2f}\N{DEGREE SIGN} of a "
                  f"{float(ex['limit_deg']):5.2f}\N{DEGREE SIGN} limit "
                  f"({ex['limited_by']}), column 2 at {mid:5.2f}\N{DEGREE SIGN}\n"
                  f"           margin {1000 * arm:6.2f} mm owed at rest   "
                  f"region {100 * float(ent['area_fraction']):5.2f} % of the "
                  f"surface ({ent['n_faces']:6d} faces, {ent['n_components']} patch, "
                  f"{float(ent['area_cm2']):6.1f} cm\N{SUPERSCRIPT TWO})\n"
                  f"           camera elev {cam.elev:.0f}\N{DEGREE SIGN} azim "
                  f"{cam.azim:7.1f}\N{DEGREE SIGN}, {out:4.1f}\N{DEGREE SIGN} "
                  f"outside the axis \N{RIGHTWARDS ARROW} phi {phi:4.1f}"
                  f"\N{DEGREE SIGN} (cos phi {np.cos(np.radians(phi)):.3f}); the "
                  f"tip DRAWS as {drawn:5.2f}\N{DEGREE SIGN} of its "
                  f"{tip:5.2f}\N{DEGREE SIGN}, where the 45\N{DEGREE SIGN} rule "
                  f"drew {was_drawn:5.2f}\N{DEGREE SIGN}\n"
                  f"           seen {100 * was_seen:5.1f} % "
                  f"\N{RIGHTWARDS ARROW} {100 * seen:5.1f} %   pivot "
                  f"{'hidden' if was_dark else 'clear '} \N{RIGHTWARDS ARROW} "
                  f"{'hidden' if dark else 'clear '}   the part moves "
                  f"{100 * moved:5.1f} % of its own width", flush=True)
            grid.append([render(name, T_rest, bare, cam),
                         render(name, T_mid, bare, cam, two),
                         render(name, T_star, work, cam, three)])
            labels.append(f"pose {pose}")
            notes.append(("", f"{mid:.0f}\N{DEGREE SIGN}", f"{tip:.0f}\N{DEGREE SIGN}"))
            rows.append(dict(name=name, pose=pose, phi=phi, out=out, seen=seen,
                             was_seen=was_seen, dark=dark, was_dark=was_dark,
                             moved=moved, arm=arm, tip=tip, drawn=drawn,
                             was_drawn=was_drawn, limit=float(ex["limit_deg"]),
                             why=ex["limited_by"]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"  the pivot, drawn as a sphere IN the scene: hidden by the workpiece "
          f"on {sum(r['was_dark'] for r in rows)} of the {len(rows)} drawn poses "
          f"under METHOD \N{SECTION SIGN}3.0's fixed camera and "
          f"{sum(r['dark'] for r in rows)} under this one. It is drawn flat OVER "
          f"the render either way.", flush=True)
    print(f"  {sum(r['out'] != BAND[0] for r in rows)} of the {len(rows)} rows "
          f"walked past {BAND[0]:g}\N{DEGREE SIGN} off the axis, because the next "
          f"rung showed more than {100 * GAIN:g} points more of the work region",
          flush=True)
    page(HERE / f"tip_{name}.png", name, grid, labels, notes)
    return rows


def main():
    print(f"objects {list(OBJECTS)} \N{EM DASH} the FIVE `{KIND}` pivots of each, "
          f"three panels a row.  A deliberate narrowing to the CORNER tips, not a "
          f"truncated ten: which five is read off tips.json's own `pivot` field "
          f"and asserted, and the other five poses are still checked here even "
          f"though they are not drawn.  Everything drawn is READ from disk: "
          f"tips/tips.json, poses.json, region/region.json.", flush=True)
    print(f"column 2 is drawn at {MIDWAY:g} \N{MULTIPLICATION SIGN} tip_deg. ONE "
          f"CAMERA A ROW, LOOKING ALONG THAT ROW'S TIP AXIS: elevation "
          f"{ROW_ELEV:g}\N{DEGREE SIGN} on every row, which is a floor on phi, and "
          f"{BAND[0]:g}\N{DEGREE SIGN} of swing OUTSIDE the axis, walked out "
          f"through {BAND[1:]} only while each rung shows {100 * GAIN:g} points "
          f"more of the work region.  A corner pivot has no contact EDGE whose "
          f"apparent length must survive, so the 45\N{DEGREE SIGN} compromise the "
          f"edge poses needed is gone and the whole of cos phi is taken.",
          flush=True)
    got = {}
    for name in OBJECTS:
        got[name] = sweep(name)
    every = [r for name in OBJECTS for r in got[name]]

    print("\nWHAT LOOKING ALONG THE AXIS BOUGHT \N{EM DASH} how big the stored tip "
          "actually draws, at the 45\N{DEGREE SIGN} the edge poses needed and at "
          "the phi this page uses:", flush=True)
    for name in OBJECTS:
        print(f"   {name}: "
              + "  ".join(f"pose {r['pose']} {r['tip']:.1f}\N{DEGREE SIGN} \N{EM DASH} "
                          f"{r['was_drawn']:.1f}\N{DEGREE SIGN} "
                          f"\N{RIGHTWARDS ARROW} {r['drawn']:.1f}\N{DEGREE SIGN}"
                          for r in got[name]), flush=True)
    print(f"   all fifteen: the drawn angle recovers "
          f"{100 * np.mean([r['drawn'] / r['tip'] for r in every]):.1f} % of the "
          f"stored tip, against "
          f"{100 * np.mean([r['was_drawn'] / r['tip'] for r in every]):.1f} % under "
          f"the 45\N{DEGREE SIGN} rule; phi runs "
          f"{min(r['phi'] for r in every):.1f} to "
          f"{max(r['phi'] for r in every):.1f}\N{DEGREE SIGN}", flush=True)

    print("\nWHAT IT COST, AND WHAT IT DID NOT \N{EM DASH} the share of the STORED "
          "work region visible at T*, METHOD \N{SECTION SIGN}3.0's fixed camera "
          "\N{RIGHTWARDS ARROW} this one, and whether the one contact would have "
          "been buried:", flush=True)
    for name in OBJECTS:
        rows = got[name]
        print(f"   {name}: mean {100 * np.mean([r['was_seen'] for r in rows]):5.1f} % "
              f"\N{RIGHTWARDS ARROW} {100 * np.mean([r['seen'] for r in rows]):5.1f} %"
              f"   under {100 * SHOWN:g} %: "
              f"{sum(r['was_seen'] < SHOWN for r in rows)} "
              f"\N{RIGHTWARDS ARROW} {sum(r['seen'] < SHOWN for r in rows)} poses"
              f"   pivot buried: {sum(r['was_dark'] for r in rows)} "
              f"\N{RIGHTWARDS ARROW} {sum(r['dark'] for r in rows)}", flush=True)
        worse = [r for r in rows if r["seen"] < r["was_seen"] - GAIN]
        if worse:
            print("      WORSE OFF: "
                  + ", ".join(f"pose {r['pose']} ({100 * r['was_seen']:.1f} "
                              f"\N{RIGHTWARDS ARROW} {100 * r['seen']:.1f} %)"
                              for r in worse)
                  + " \N{EM DASH} the patch faces back the way the part came "
                    "from, and no eye near the axis and outside the pivot sees "
                    "all of it", flush=True)
    print(f"   all fifteen: {100 * np.mean([r['was_seen'] for r in every]):.1f} % "
          f"\N{RIGHTWARDS ARROW} {100 * np.mean([r['seen'] for r in every]):.1f} % "
          f"of the patch, {sum(r['was_dark'] for r in every)} "
          f"\N{RIGHTWARDS ARROW} {sum(r['dark'] for r in every)} contacts buried",
          flush=True)

    print(f"\nWHERE THE TIP STILL DOES NOT READ \N{EM DASH} a row reads when it "
          f"DRAWS at least {READS_DEG:g}\N{DEGREE SIGN}. The second number is "
          f"measured on the page: how far the furthest vertex travels between "
          f"column 1 and column 3, over the part's own on-screen width \N{EM DASH} "
          f"a long part swung a few degrees about a corner moves its far end a "
          f"long way, so the two tests do not agree and both are given.",
          flush=True)
    for name in OBJECTS:
        bad = [r for r in got[name] if r["drawn"] < READS_DEG]
        print(f"   {name}: "
              + (", ".join(f"pose {r['pose']} (draws {r['drawn']:.1f}"
                           f"\N{DEGREE SIGN} of a {r['tip']:.2f}\N{DEGREE SIGN} "
                           f"tip, {r['limit']:.2f}\N{DEGREE SIGN} limit, "
                           f"{r['why']}; moves {100 * r['moved']:.1f} %)"
                           for r in bad)
                 if bad else "none \N{EM DASH} all five read"), flush=True)

    # WHAT tip_fraction WOULD HAVE TO BE, and where no value of it is enough.
    # `drawn = 2 atan(cos phi tan(tip/2))` inverts, and `tip = f * limit_deg`.
    bad = [r for r in every if r["drawn"] < READS_DEG]
    print(f"\n   {len(bad)} of the fifteen. tip_fraction is 0.6 and NOTHING HERE "
          f"CHANGES IT; what it would have to be, per row, to draw "
          f"{READS_DEG:g}\N{DEGREE SIGN} at that row's own camera:", flush=True)
    hopeless = []
    for r in bad:
        need = 2 * np.degrees(np.arctan(np.tan(np.radians(READS_DEG) / 2)
                                        / np.cos(np.radians(r["phi"]))))
        f = need / r["limit"]
        if f > 1.0:
            hopeless.append(r)
        print(f"      pose {r['pose']} on {r['name']:<5s} needs a {need:.2f}\N{DEGREE SIGN} tip of a {r['limit']:.2f}"
              f"\N{DEGREE SIGN} limit \N{EM DASH} tip_fraction {f:.2f}"
              + ("   IMPOSSIBLE: past 1.0, i.e. past the angle the workpiece is "
                 "allowed to reach at all" if f > 1.0 else ""), flush=True)
    inner = 2 * np.degrees(np.arctan(np.tan(np.radians(READS_DEG) / 2)
                                     / np.cos(np.radians(phi_of(ROW_ELEV, BAND[0])))))
    print(f"   So {len(hopeless)} of the {len(bad)} CANNOT be fixed by any "
          f"tip_fraction: their whole limit angle is under the {inner:.2f}"
          f"\N{DEGREE SIGN} a legible tip would need, and "
          f"{sum(r['why'] != 'balance' for r in hopeless)} of those "
          f"{len(hopeless)} are cut short by ANOTHER PART OF THE WORKPIECE "
          f"HITTING THE FLOOR, not by balance. THIS IS THE WORKPIECE AND NOT THE "
          f"DRAWING. Raising tip_fraction is a DATA decision for whoever owns "
          f"tips.json, and even then it buys only the rows whose limit is large "
          f"enough; nothing here changes it, regenerates tips.json, or "
          f"exaggerates an angle.", flush=True)

    print("\nTHE TIPPING MARGIN ON THE DRAWN ROWS \N{EM DASH} METHOD "
          "\N{SECTION SIGN}11.4's w(0), the mm of arm the supports owe at one body "
          "weight with the process switched OFF, drawn on column 3 and asserted "
          "equal to the horizontal offset from the centre of mass to the contact "
          "on every pose of every run (cuboid_baseline's recorded row: 2.03 to "
          "45.50 mm):", flush=True)
    for name in OBJECTS:
        print(f"   {name}: "
              + "  ".join(f"{1000 * r['arm']:.2f}" for r in got[name])
              + " mm", flush=True)


if __name__ == "__main__":
    main()
