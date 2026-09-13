"""slides/poses — THE BIG TIP: flat on the floor, then over onto ONE point.

`tip_sequence.py` draws `tips/tips.json`'s poses, and those stop where METHOD
s0 says a tip stops: at the balance point, short of the angle at which gravity
takes over.  On its three workpieces that is 13.7 to 33.1 degrees, which is the
honest target pose and is NOT a picture of the problem -- a viewer reads a 14
degree lean as a part standing slightly crooked, and the whole difficulty, that
the pose is one the workpiece has no equilibrium in, has to be argued from a red
bar and a number rather than seen.

This page turns workpieces MUCH FURTHER: 60 to 120 degrees, about ONE point of
their own footprint, past the balance point and into a pose gravity is actively
driving them out of.  Three columns a row -- the stable placement it starts
from with the chosen point already marked, the turn halfway through with the
push that makes it, and the target pose with its work region painted on.  Every
number the row rests on prints on every run.

WHAT IS CLAIMED, AND HOW EACH IS CHECKED

1. **It is one rotation about one point of the ground, and nothing is lifted
   into place.**  METHOD s0's condition -- the contact is never broken and never
   relocated -- with one measured exception the tessellation forces and which
   is priced rather than hidden: see `candidates`, `roll` and `lift`.
2. **The workpiece clears the floor the whole way, and that is exact, not
   sampled.**  A vertex at `(x, z)` in the plane of the tip stands at
   `r sin(phi - t)` after a turn of `t`, with `phi = atan2(z, x)`, so it reaches
   the floor exactly at `t = phi` and the limit is `min phi` -- closed form, no
   marching.  The drawn angle stands a `WEDGE` under it -- see `candidates`:
   that angle is not a safety margin, it IS the daylight under the workpiece --
   and the assert at the end of `one` re-checks the finished transform rather
   than the formula that chose it.
3. **It stands on ONE point.**  Tested as what it means: at the drawn angle, the
   set of vertices still ON the floor is at most `PATCH` wide.  An edge pivot
   fails that -- its far end never lifts at all -- and so does a corner whose
   next feature has already come down.  Both readings print, because they differ
   and the difference is a fact about the workpiece: at `TOUCH` the patch is
   0.00 to 0.73 mm wide over the twelve rows, and at CONTACT_EPS's 1.5 mm
   tolerance it is 2.18-2.72 mm on A1-f, 2.49-3.74 on C5 and **12.23-12.42 on
   B**, whose feet are low-curvature pads that lie nearly flat.  That is METHOD
   s11.10's strip, met again.
4. **The pose cannot balance, and it is PAST the balance point** -- not leaning
   back toward where it came from but over the top, where gravity drives it on.
   `tip_sequence`'s column 3 draws four marks for this: the centre of mass, its
   plumb line, the contact, and the horizontal gap between them in red
   millimetres, METHOD s11.4's `w(0)`.  **This page computes all four, asserts
   the identity between them and prints them, and draws NONE of them.**  It was
   asked for one mark -- the point the workpiece turns on -- so the balance
   angle and `w(0)` are carried by the caption and the log instead of by ink on
   the picture.

UP TO FIVE ROWS, AND WHY B HAS TWO

`spread` takes the DIFFERENT tips -- a different stable placement, or on the same
one a contact `APART` away or a bearing `SEP` away -- and A1-f offers 1008
qualifying bearings and C5 250, so five of each is a choice.  B has exactly TWO
and gets a TWO-ROW page.  **A row is a different tip or it is not drawn**: the
row count is variable, `ROWS` is a ceiling and not a quota, and no page is
padded.

The one thing that widened is WHERE THE PLACEMENTS COME FROM.  `poses.json` is a
sample and not a set: `slides/tools/drop_sample.py` dropped B forty times, found nine
distinct placements and kept five.  So when `spread` comes up short of `ROWS` --
and ONLY then, which is what keeps A1-f and C5 byte-identical -- `hull_places`
enumerates the rest off the workpiece's own convex hull, exhaustively: every
facet, seated on the floor, kept when the centre of mass projects strictly inside
the footprint.  B has FIFTEEN statically stable hull facets; five are
`poses.json`'s, and one of the other ten -- facet 113 -- carries a tip this page
can draw.

WHAT WAS BELIEVED HERE UNTIL 2026-08-28, AND WHY IT WAS WRONG.  This section used
to say that B's eight qualifying bearings were one corner of one placement over a
3.5 degree fan, that a `top_up` function therefore had to draw a true tip four
more times to fill the sheet, and that this was the workpiece and not the code --
reasoning from "five placements, 720 bearings each, 3600 pairs, nothing sampled".
The sweep was exhaustive over what it was handed and what it was handed was
incomplete: five sampled placements out of fifteen.  **The placement count was
never the limit.**  A ROUND workpiece rolls onto another part of itself before
the tip clears its own balance point, and that is what B's other placements fail
on: their best landing limits run 1.99 to 67.89 degrees and only two of the
fifteen reach the 68 that `WANT[0] + MIN_WEDGE` asks for.  Read as the window a
tip has to live in, `limit - balance` is over 30 degrees on three of the fifteen,
and the third of those -- facet 233, 39.73 degrees of window -- misses the 68 by
0.11.  Widening the source found the second tip; it did not find a fifth, and
relaxing the band, the wedge or `LIE` to manufacture one was measured, refused,
and is still refused (`big_tip.md`).

WHICH WORKPIECES

`A1-f`, `B` and `C5` -- the working set the rest of `slides/poses` and
`PIPELINE` s3 draw, so the same three parts carry every page of the deck.  They
are not the easy three: swept over the whole library with the 60 degree floor
lifted (720 bearings a placement, five placements an object), **15 of the 21
objects clear 60 degrees somewhere and 6 cannot** -- C1 stops at 54.8, C8 at
23.2, D1 at 54.2, D2 at 11.1, D3 at 45.7, D4 at 20.5 -- and **B is the tightest
of the fifteen, with 9 of its 1784 bearings clearing at all** (its own body
overhangs its feet, so nearly every direction is stopped in the twenties).  The
roomiest are `A2` at 119.2 degrees on every one of its bearings and `A5` at
120.0; `A4` and `cuboid_baseline` reach 80 everywhere.  Any of them can be drawn
by editing `OBJECTS` -- nothing else in the file knows which parts these are.

THE WORK REGION IS REGENERATED HERE, AND THAT IS FORCED

`objects/<name>/region/region.json` stores one patch per pose of `tips.json`,
and these are not those poses.  The rules are that file's own, quoted from its
`seed_rule` and `growth_rule`: an area-uniform seed among faces whose WORLD
normal at the target pose has `n_z > work_regions.GUN_MIN_ELEVATION`, whose
centre stands clear of `supports.CONTACT_EPS`, and which is visible from outside
along its own normal; then a geodesic disc grown by Dijkstra on the
face-adjacency graph weighted by the distance between face centres, never
entering the floor band and never carried past the top of the `[0.08, 0.15]`
band.  `N_SEEDS` of them are grown and the one that DRAWS largest is the one
painted -- a work region is a declared INPUT to the problem (PIPELINE s1), so
choosing which patch a slide declares is legitimate; choosing one and not saying
how is not, and every candidate's size and drawn share prints.

THE CAMERA is `tip_sequence`'s elevation and its axis-first argument, with its
WALK removed (`nearest`): the eye stands `OUT` = 8 degrees outside the pivot and
stays there, so `phi` is 14.4 degrees on every row and the tip always draws at
`cos 14.4` = 0.969 of itself.  That module walks outward while the work region
keeps coming into view, which is right for a 14 degree tip and wrong here: on B
it went out to 22 degrees and drew a 62.7 degree tip as 57.9, under the 60 this
page is for.  What the walk bought is bought instead where it costs no angle --
the region is a DECLARED input and `regions` grows `N_SEEDS` of them, so the
patch is chosen for the camera rather than the camera for the patch.

The pivot is drawn flat OVER the render, for `tip_sequence`'s reasons, and that
is what answers "the contact point must be visible": it is on top of the picture
and never behind the part.  Whether it would ALSO have been visible in the scene
is measured and printed per row rather than assumed -- on these three rows it
would not, at either angle, because a big tip carries the workpiece OVER its own
contact and an eye outside the pivot then has the part in front of it.

Deterministic: the only draws are the region seeds, from a generator seeded by
`SEED` and the object's NAME (METHOD s7: never by list index).

    cd /Users/yuanboli/Documents/GitHub/cadGrasp
    source ~/miniforge3/etc/profile.d/conda.sh; conda activate cadgrasp
    python -u slides/setup/poses/big_tip.py

Writes ONE PAGE AN OBJECT, `tip_<name>.png`, up to `ROWS` rows of it.  **Those
are the filenames `tip_sequence.py` writes too**, and this page is what they now
hold: same three columns, same page fitter, same shape on the slide, the tips
five times bigger and one mark instead of four.  Running `tip_sequence.py` puts
its own version back, so run one or the other and not both -- `slides/setup/poses/
README.md` carries the same warning beside that script's own md5 table.

It used to write one combined `big_tip.png` and, before that, every panel
separately; both were dropped on request.  `sweep` has each row's three panels
in memory if either is ever wanted back.
"""
from __future__ import annotations

import heapq
import shutil
import sys
import zlib
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, QhullError

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                # slides/setup/poses -> repo root
sys.path.insert(0, str(ROOT / "slides/tools"))
sys.path.insert(0, str(HERE))
import coordinates as COORD

from common import obj_path, read_json                          # noqa: E402
from PIL import Image                                           # noqa: E402
from cover import AZIM, ELEV, PAPER                             # noqa: E402
from supports import CONTACT_EPS                                # noqa: E402
from work_regions import GUN_MIN_ELEVATION                      # noqa: E402

import tip_sequence as T                                        # noqa: E402
from tip_sequence import (ARROW_L, ARROW_W, ROW_ELEV,           # noqa: E402
                          apparent_tip, buried, facing, fit, look, margin,
                          outward, page, phi_of, push_site, pull_forward,
                          render, rot_about_line, travel)
from tip_sequence import screen as T_screen                     # noqa: E402

# `page` writes these across the top band.  Set here rather than copied, so this
# file adds a fourth caller to that fitter and not a fifth copy of it
# (`torque/demo_fig/README.md`'s standing complaint).
T.TITLES = ("at rest", "over the point", "the target pose")
# and `place` keeps the millimetre label off the WHOLE bottom strip rather than
# `tip_sequence`'s bottom-left corner of it: that page's column-3 note is one
# angle and this one's carries the balance angle beside it, so the label was
# landing on the words -- and a wider corner was not enough, because the label
# simply moved to the end of the same line.
T.NOTE = (1.0, 0.13)
# and the note itself is written DARKER than `tip_sequence`'s muted grey, which
# is within a few levels of the floor's own shadow: that page's angles sit on
# white, and this page's parts throw a long shadow across the bottom of the
# panel at 12 degrees of elevation and 80 of tip, with the note on top of it.
T.MUTED = (55, 55, 52)

OBJECTS = ("A1-f", "B", "C5")
NB = 720                # bearings tried a placement: the horizontal direction
                        # the workpiece is pushed over in, and with it the one
                        # vertex of the footprint it turns on
WANT = (60.0, 120.0)    # the band of tip angles this page is for
GUTTER = 10             # a margin of page colour round every panel, in pixels.
                        # `tip_sequence.page` pastes its tiles FLUSH, which is
                        # right when a panel ends in white sky and wrong the
                        # moment the ground is drawn: the grey runs to the
                        # panel's edge, so the three panels of a row join into
                        # one continuous floor and the eye reads a single scene
                        # cut by two vertical seams.  Ten pixels each side is
                        # twenty between neighbours, and `PAPER` (#ffffff) is
                        # the sheet's own colour, so the margin shows only where
                        # it has something to separate
FLOOR_RGBA = "0.80 0.80 0.81 1"         # the ground, a plain grey.  It is
                        # `work_regions.SCENE`'s own infinite plane -- no
                        # thickness, no edge, not a table -- darkened until it
                        # reads.  At the 0.97 it ships with it is invisible
                        # against the white sky; the horizon then sits near the
                        # top of the panel and the workpiece, which is warm
                        # white, stands against it
HIDDEN = 60.0           # what a camera that HIDES the contact behind the
                        # workpiece is charged, in page pixels of a 700 px
                        # panel -- about the drawn width of the marker.  See
                        # `cameras`: both failures are measured as the distance
                        # on the page between the mark and where the eye takes
                        # the ground contact to be
OUT = (8.0, 15.0, 22.0)  # how far off the tip's own axis the eye may stand, in
                        # degrees -- `tip_sequence.BAND` without its last rung.
                        # The inner one is always preferred; `cameras` walks out
                        # only to stop the workpiece standing in front of the
                        # mark, and pays `PER_DEG` a degree for it
PER_DEG = 3.0           # what a degree off the axis costs, in the same page
                        # pixels `cameras` scores everything else in.  Fourteen
                        # degrees of walk is 42 px against a hidden contact's
                        # 60, so the eye walks to show the contact and for
                        # nothing else
WEDGE = 20.0            # how much of the limit is spent on DAYLIGHT rather than
                        # on tip, in degrees, when there is that much to spare.
                        # It is not a safety margin: it is the angle the
                        # workpiece's underside makes with the floor, which is
                        # the whole of what makes a panel read as PERCHED ON A
                        # POINT rather than as lying down.  Twenty degrees on a
                        # 200 mm part lifts its far end 68 mm off the ground
MIN_WEDGE = 8.0         # and under this much daylight a bearing is not drawn at
                        # all, however big its tip: the picture would not say
                        # what the caption says
TILT = 3.0              # how far off level the recorded placement's own resting
                        # hull facet may stand before `rest_pose` refuses it
DEDUPE = 1.0            # two hull facets whose normals are within this many
                        # degrees stand the workpiece on the SAME face of itself,
                        # so `hull_places` offers one of the two.  B's hull is
                        # 262 facets over 231 distinct normals, and its resting
                        # pads are tessellated into several facets apiece
TOUCH = 1e-4            # what counts as ON the floor when the contact is picked.
                        # NOT supports.CONTACT_EPS: that is a 1.5 mm tolerance
                        # band (METHOD s11.10 measures what it catches), and a
                        # vertex anywhere in it can be the one that reaches
                        # furthest along the tip -- which puts the workpiece on a
                        # 1.5 mm pedestal, floating, in every panel of the row.
                        # The point this page turns on has to be ON the ground
NEAR = 0.002            # vertices this close to the contact are the SAME
                        # contact: a filleted corner's own tessellation, which
                        # the tip rolls along rather than lands on (see
                        # `candidates`).  Excluded from the limit, priced by
                        # `DEPTH` instead
DEPTH = 2e-4            # how far the deepest vertex of the mesh may go under the
                        # floor at the drawn angle before the bearing is refused.
                        # The pose is then lifted by exactly that much, so the
                        # drawn workpiece is never through the floor and the
                        # contact patch is on it: 0.2 mm is a fifth of a
                        # tessellation edge on these parts and invisible at any
                        # page size, and every row prints what it actually used
LIE = 0.030             # how much of the workpiece may sit within CONTACT_EPS of
                        # the floor before the pose READS as lying on a feature
                        # rather than standing on a point.  A picture criterion,
                        # not the "one point" test -- that is `PATCH`, below
PATCH = 0.004           # and how wide what still TOUCHES the floor may be at the
                        # drawn angle -- 4 mm on a 100-180 mm workpiece, with
                        # "touches" read at `TOUCH` and not at CONTACT_EPS, which
                        # is a 1.5 mm tolerance meant for a different question.
                        # This is what "ONE point" is tested as: an edge pivot's
                        # far end is 80 mm away and never lifts at all, and fails
                        # it; a filleted corner's own neighbours are inside it
MIDWAY = 0.5            # column 2's angle, as a fraction of the tip
BAND_AREA = (0.08, 0.15)        # region.json's own `band`
TARGET_AREA = (0.09, 0.14)      # and its own `target_drawn_from`
ROWS = 5                # how many different tips a page carries at most, which
                        # is `tip_sequence`'s five corner poses an object -- the
                        # shape this page replaces
SEP = 25.0              # two tips of the same contact are the same picture
                        # unless their push bearings differ by this much
APART = 0.015           # or unless the points they turn on are this far apart
N_SEEDS = 12            # candidate work regions grown a row, of which one is
                        # drawn and all are printed
COARSE = 2000           # a mesh this size or smaller is refined uniformly;
                        # a finer one is left exactly as it is
QUANTUM = 0.005         # the largest share of the surface one face may hold
                        # before it is split, and PIPELINE s6's standing trap --
                        # "subdivide before patching" -- is why: a region has to
                        # be able to cut ACROSS a face, and what the danger
                        # scales with is FACE SIZE AGAINST THE PATCH, not face
                        # count against the part.  A4 is 92 faces over a 150 mm
                        # frame and its largest face is 13.9 % of the surface,
                        # so an 8-15 % patch cannot be grown on it at all until
                        # that face is split; A1-f's largest is 4.5 % and A2's
                        # 14.4 %.  ONLY the faces over the quantum are split --
                        # subdividing to a uniform edge instead took A1-f from
                        # 25k faces to 1.2M and sampled its fillets so finely
                        # that every bearing came back a roll (`candidates`)
SEED = 20260827
TMP = "_big_tip_tmp"
Z = np.array([0.0, 1.0, 0.0])  # historical symbol; world up is Y


# ------------------------------------------------------------ the geometry ---

def down(n):
    """The smallest rotation carrying `n` onto -z: about their common
    perpendicular, by the angle between them.  Seating a hull facet on the floor
    is this and nothing else, and both `rest_pose` and `hull_places` need it.
    """
    v = np.cross(n, -Z)
    s = float(np.linalg.norm(v))
    if s <= 1e-12:
        return np.eye(3)
    v = v / s
    K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    a = np.arctan2(s, float(n @ -Z))
    return np.eye(3) + np.sin(a) * K + (1.0 - np.cos(a)) * (K @ K)


def rest_pose(mesh, T_place):
    """The stable placement, RE-SEATED on the hull facet it actually rests on.

    `poses.json` records where a physics drop settled the part, and a settle
    stops when the motion stops, not when the contact is exact: the recorded
    placements carry a residual tilt of a tenth of a degree or so, which over a
    60 mm part is a tenth of a millimetre of height across the contact patch.
    That is invisible on a page and fatal here, because this page turns the
    workpiece about a VERTEX and picks that vertex as the one reaching furthest
    along the tip among those ON the floor.  Under a residual tilt the patch's
    true outer vertex sits a hair ABOVE the cut, the vertex behind it is picked
    instead, and the tip then ends a few degrees later when the outer one comes
    down -- which threw out every bearing of every C-series part before this
    function existed.

    A body resting on a plane rests on a facet of its own CONVEX HULL, so the
    exact seat is available without a simulator: take the hull face whose world
    normal points most nearly straight down, turn the pose by the smallest
    rotation that makes it point exactly down, and drop.  Two asserts say the
    facet found is the one the part is actually on -- it was already within
    `TILT` of level, and the centre of mass projects INSIDE its polygon, which
    is what "stable placement" means.

    Returns the seated pose, the tilt taken out, and the drop.
    """
    hull = mesh.convex_hull
    R = T_place[:3, :3]
    n = np.asarray(hull.face_normals) @ R.T
    j = int(np.argmin(n[:, 1]))
    tilt = float(np.degrees(np.arccos(np.clip(-n[j, 1], -1.0, 1.0))))
    assert tilt < TILT, (
        f"the most downward-facing hull facet of this placement stands {tilt:.2f} "
        f"deg off level, so the recorded pose is not resting on it")
    F = down(n[j])
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = F @ R, F @ T_place[:3, 3]
    V = np.asarray(mesh.vertices) @ out[:3, :3].T + out[:3, 3]
    drop = float(V[:, 1].min())
    out[1, 3] -= drop
    # the placement is STABLE, which is what makes it something to tip FROM: the
    # weight comes down inside the polygon the part stands on
    P = np.asarray(hull.vertices) @ out[:3, :3].T + out[:3, 3]
    foot = COORD.floor(P[P[:, 1] <= TOUCH])
    com = COORD.floor(out[:3, :3] @ mesh.center_mass + out[:3, 3])
    poly = ConvexHull(foot)
    deep = float((poly.equations[:, :2] @ com + poly.equations[:, 2]).max())
    assert deep <= 0.0, (
        f"the weight comes down {1000 * deep:.2f} mm OUTSIDE the polygon the "
        f"part stands on, so this placement is not stable and there is nothing "
        f"to tip from")
    return out, tilt, drop, float(-deep)


def refine(mesh, rounds=6):
    """Split a COARSE mesh uniformly until no face is bigger than the quantum.

    PIPELINE s6's standing trap -- "subdivide before patching" -- with its own
    correction beside it: what the danger scales with is FACE SIZE AGAINST THE
    PATCH, not face count against the part.  A4 is 92 faces over a 150 mm frame
    and its largest face is 13.9 % of the surface, so a patch grown there is one
    or two triangles and its boundary can only run along edges; A1-f is 25k faces
    whose largest is 4.5 % and needs nothing.

    UNIFORM rounds, and only on a mesh coarse enough to afford them.  Splitting
    just the faces over the quantum leaves T-JUNCTIONS -- the neighbours keep
    their own edges, the shared edge is no longer shared, and `face_adjacency`
    quietly loses it.  That is not a cosmetic loss here: the region is grown by
    Dijkstra ON that graph, and A1-f's patches stalled at 4-7 % of the surface,
    under the band, because the disc could not cross the seam.  Uniform 4-splits
    keep every adjacency; the cost is why `COARSE` gates them.

    Subdivision moves no surface -- every new vertex is a convex combination of
    old ones -- so the tip's own geometry is untouched, and the asserts say so.
    """
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    got = 0
    if len(f) <= COARSE:
        for got in range(1, rounds + 1):
            v, f = trimesh.remesh.subdivide(v, COORD.faces(f))
            f = COORD.faces(f)
            m = trimesh.Trimesh(v, f, process=False)
            if (m.area_faces / m.area).max() <= QUANTUM:
                break
    out = trimesh.Trimesh(v, f, process=False)
    assert abs(out.area - mesh.area) <= 1e-9 * mesh.area, "splitting moved the surface"
    assert abs(out.volume - mesh.volume) <= 1e-9 * abs(mesh.volume), \
        "splitting moved the solid"
    assert len(out.face_adjacency) == len(mesh.face_adjacency) * 4 ** got, \
        "splitting lost a face adjacency, which is the graph the region grows on"
    return out, got


def cameras(mesh, T_star, T_mid, marks, nrm, axis, frame):
    """The row's candidate cameras: four, and the ones that SHOW the contact.

    Two decisions, and the second is the one this page got wrong for a while.

    **As near the tip's own axis as the page allows.**  `tip_sequence.choose`
    starts on `BAND`'s inner rung and WALKS OUTWARD while each next rung shows
    `GAIN` more of the work region.  That trade is right for a 14 degree tip,
    where the angle is not what the page is about, and wrong here, where it is:
    every degree off the axis is paid out of the tip's own drawn size, because
    the circle a point traces about the axis projects to an ellipse of ratio
    `cos phi`.  On B the walk went out to 22 degrees for the green and drew a
    62.7 degree tip as 57.9.  So the eye stays on the inner rung, `phi` is 14.4
    degrees on every row, and the tip draws at `cos 14.4` = 0.969 of itself.

    **And it stands where the CONTACT can be seen.**  `tip_sequence` puts the
    eye OUTSIDE the pivot, on the side the workpiece leans toward, because at
    14 degrees of tip that is where the contact is nearest the eye.  Past the
    balance point it is the opposite: the workpiece has gone OVER its own
    contact and an eye out there has the part in front of it.  Measured on every
    row of this page before the fix -- `the contact would be HIDDEN by the
    workpiece at T*`, all of them -- and it shows: the marker is drawn flat over
    the render, so a buried contact lands in the MIDDLE OF A FACE, metres of
    perspective behind it, and reads as a sticker on the part rather than as the
    point it stands on.

    So TWELVE cameras are offered -- each rung of `OUT`, either side of the axis,
    the eye at either end of it -- and `one` takes the one minimising
    `drop + HIDDEN * hidden + PER_DEG * (out - OUT[0])`, all three terms in page
    pixels.  Standing inside the pivot costs nothing else: at 8 degrees off the
    axis the parallax is small, and the contact is a corner of the footprint
    either way.
    """
    out = []
    for side in (1.0, -1.0):
        for off in [o * k for o in OUT for k in (1.0, -1.0)]:
            cam = frame(ROW_ELEV, look(nrm, axis, ROW_ELEV, off, side))[0]
            got = float(np.degrees(np.arccos(min(1.0, abs(cam.fwd @ axis
                                                          / np.linalg.norm(axis))))))
            assert abs(got - phi_of(ROW_ELEV, abs(off))) < 1e-9, (
                f"asked for a view {off} deg off the axis at elevation "
                f"{ROW_ELEV} and got {got:.6f} deg from it")
            # and how far the workpiece hangs BELOW the marked contact on the
            # page.  Not in the world -- on the page: perspective puts a floor
            # point nearer the eye lower in the image than a far one, so a part
            # leaning toward the camera draws its near end under a contact that
            # is genuinely the lowest thing there is, and the panel reads as an
            # edge resting on the ground with the mark stuck on beside it.  The
            # camera that puts the contact at the BOTTOM of the silhouette is
            # the one that says "it stands on this point" without a caption
            V = np.asarray(mesh.vertices) @ T_star[:3, :3].T + T_star[:3, 3]
            drop = float(T_screen(V, cam)[:, 1].max()
                         - T_screen(marks, cam)[0, 1])
            out.append(dict(cam=cam, out=off, side=side, drop=drop,
                            walk=PER_DEG * (abs(off) - OUT[0]),
                            dark=buried(mesh, T_star, marks, cam),
                            dark_mid=buried(mesh, T_mid, marks, cam)))
    return out


# NO CAST SHADOW, AND A GREY FLOOR.  Three things were tried here and the two
# that failed are kept, because the reasoning for each was sound and the failure
# was not where it was expected.
#
# The shadow goes.  The scene lights from nearly overhead, so a workpiece perched
# with its far end 68 mm up throws its shadow only 34 mm from under itself: the
# daylight `candidates` spends up to 20 degrees of tip to buy came back filled
# with the part's own shadow, which is exactly the picture of something LYING on
# the ground.  Moving the lamp to the row's side does shift the shadow clear --
# and takes the floor's brightness with it, `sin 63` to `sin 41` of the lamp,
# most of the contrast between lit floor and shadow, so those panels came back
# with a flat grey floor and no readable shadow anyway.
#
# The floor stays, and is darkened to `FLOOR_RGBA`.  With no shadow it is
# invisible at the 0.97 it ships with: white plane, white sky.  A finite disc was
# tried, to dodge the horizon that sits in frame at 12 degrees of elevation, and
# a solid with a rim is a TABLE, which the workpiece is not standing on.  Six
# colours were rendered side by side; a plain grey at 0.80 is the one that lets a
# warm-white part stand against it without five rows of floor becoming the
# subject.
#
# String replacements on this page's copy of the scene; `slides/tools/work_regions.py`
# is untouched, and `tip_sequence.py` keeps its shadows.
T.SCENE = T.SCENE.replace('castshadow="true"', 'castshadow="false"').replace(
    '<material name="floor" rgba="0.97 0.97 0.97 1"/>',
    f'<material name="floor" rgba="{FLOOR_RGBA}"/>')


def shown(mesh, T, inside, cam):
    """How much of the DRAWN part the work region covers, in pixels.

    `tip_sequence.facing` measures the share of the region's own surface area
    that turns toward the eye, which is the right question when the camera is
    being walked round a fixed patch.  It is the wrong one when the PATCH is
    what is being chosen: a flat face seen nearly edge-on scores 100 % there and
    draws as a green thread.  A2's page went out that way -- one triangle,
    14.37 % of the surface, "100.0 % in view", a sliver.

    So a candidate region is scored on the picture: the projected area of the
    faces that both turn toward the eye and are not occluded, over the projected
    area of the whole front of the part.  Both are sums of screen-space triangle
    areas, which is exact for the front and an over-count for the part's own
    silhouette where it folds -- and that over-count divides out of a comparison
    between candidates on one camera, which is all this is used for.
    """
    V = np.asarray(mesh.vertices) @ T[:3, :3].T + T[:3, 3]
    S = T_screen(V, cam)
    tri = S[mesh.faces]
    a, b = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    px = 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])    # 2-d cross, by
    # hand: numpy 2.0 deprecated `np.cross` on two-vectors
    c = mesh.triangles_center @ T[:3, :3].T + T[:3, 3]
    n = mesh.face_normals @ T[:3, :3].T
    d = cam.eye - c
    front = (n * d).sum(axis=1) > 0
    idx = np.flatnonzero(inside & front)
    if not len(idx):
        return 0.0
    clear = ~mesh.ray.intersects_any(
        ray_origins=mesh.triangles_center[idx] + 1e-5 * mesh.face_normals[idx],
        ray_directions=(d[idx] / np.linalg.norm(d[idx], axis=1, keepdims=True))
        @ T[:3, :3])
    return float(px[idx][clear].sum() / px[front].sum())


def apart(im):
    """One panel on a slightly larger sheet of page colour, so floors do not join.

    See `GUTTER`.  It is done here rather than in `tip_sequence.page` because
    that function is shared and its own pages end in white sky, where flush is
    exactly right.
    """
    out = Image.new("RGB", (im.size[0] + 2 * GUTTER, im.size[1] + 2 * GUTTER), PAPER)
    out.paste(im, (GUTTER, GUTTER))
    return out


def paint_parts(mesh, inside, tmp, tag):
    """`shrink_support.paint`'s two-material split, with PIPELINE s6's guard.

    That guard: **mujoco refuses fewer than four vertices.**  It hulls every
    mesh geom and a hull needs a fourth point, so a ONE-TRIANGLE piece is a hard
    `ValueError`, not a warning.  `paint` cannot hit it on a work region grown
    over a fine mesh and this page does: A2's largest face is 14.37 % of its
    surface, which is a legal region all by itself, and it is one triangle.  So
    a fourth point goes on, 20 micrometres off the face along its own normal --
    the same trick and the same scale as the coplanarity jitter `paint` already
    carries, and `step4_heat.export`'s own remedy for the same crash.
    """
    parts = []
    for material, sel in (("work", inside), ("spare", ~inside)):
        if not sel.any():
            continue
        patch = mesh.submesh([np.flatnonzero(sel)], append=True)
        v = patch.vertices - patch.vertices.mean(axis=0)
        if np.linalg.matrix_rank(v, tol=1e-9 * max(patch.scale, 1e-9)) < 3:
            patch.vertices[::2] += 2e-5 * patch.face_normals[0]
        if len(patch.vertices) < 4:
            n = len(patch.vertices)
            patch = trimesh.Trimesh(
                np.vstack([patch.vertices,
                           patch.vertices.mean(axis=0) + 2e-5 * patch.face_normals[0]]),
                np.vstack([patch.faces, [[0, 1, n]]]), process=False)
        name = f"{tag}_{material}.obj"
        patch.export(tmp / name)
        parts.append((f"{TMP}/{name}", material))
    return parts


def candidates(mesh, T_rest):
    """Every (direction, contact, limit) this placement offers, exactly.

    For a horizontal push direction `e1` the contact is the floor vertex that
    reaches furthest along it -- a supporting line of the footprint, METHOD s0's
    condition -- and the tip is about the horizontal axis `z x e1` through it,
    which carries the part up and over that point.

    The limit is closed form.  Write a vertex in the plane of the tip as
    `(x, z)` relative to the contact; the rotation takes it to
    `(x cos t + z sin t, -x sin t + z cos t)`, so its height is `r sin(phi - t)`
    with `phi = atan2(z, x)` and it reaches the floor exactly at `t = phi`.  The
    limit is therefore `min phi` over the vertices, no sampling and no marching.

    **`min phi` over the vertices FARTHER than `NEAR`, and the difference is the
    whole of whether these workpieces can be tipped at all.**  A convex corner in
    this library is FILLETED and its fillet is tessellated, so the vertex that
    lands first is nearly always the contact's own NEIGHBOUR a fraction of a
    millimetre away -- C5's is 0.27 mm out and lands at 76.9 degrees, C3's is
    0.25 mm out and lands at 62.0.  Read as a limit, that is a workpiece that
    cannot be tipped; read as what it is, it is the contact ROLLING onto the next
    facet of its own fillet, which is a quarter of a millimetre of contact
    migration and not a second contact at all.  So the neighbourhood is excluded
    from the limit and its dip below the floor is priced instead: `depth` is how
    far the deepest vertex of the whole mesh goes under at the drawn angle, it is
    required to stay inside `DEPTH`, and the pose is lifted by exactly that much
    so that nothing is through the floor and the patch touches it.  Going eight
    degrees past a neighbour 0.27 mm away costs 38 micrometres.
    """
    V = np.asarray(mesh.vertices) @ T_rest[:3, :3].T + T_rest[:3, 3]
    com = T_rest[:3, :3] @ mesh.center_mass + T_rest[:3, 3]
    floor = np.flatnonzero(V[:, 1] <= TOUCH)
    out = []
    for k in range(NB):
        th = 2 * np.pi * k / NB
        e1 = np.array([np.cos(th), 0.0, np.sin(th)])
        x = V @ e1
        j = int(floor[np.argmax(x[floor])])
        q = V[j]
        dx, dz = x - x[j], V[:, 1] - q[1]
        r = np.hypot(dx, dz)
        on = r > 1e-9                           # what sits ON the axis never moves
        phi = np.arctan2(dz[on], dx[on])
        # `phi <= 0` is a vertex at or below the CONTACT's own height, and after
        # the drop that can only be another vertex of the resting patch, BEHIND
        # the contact -- `dz <= 0` forces `dx <= 0`, since anything at floor
        # level ahead of the contact would have been the contact.  The tip
        # LIFTS those (its admissible interval is `[phi + pi, phi + 2pi]`), so
        # they bind at `phi + 2pi >= 180` degrees and not at `phi`.  Reading
        # `min(phi)` straight off instead makes a vertex one micrometre lower
        # than the contact report a limit of -180 degrees and throws the whole
        # bearing away: A1-f's resting face carries 1199 vertices within 0.1 mm
        # of the floor and every one of its bearings died this way
        far = r[on] > NEAR
        ahead_of = np.degrees(phi[far & (phi > 0)])
        lim = float(min(ahead_of.min(), 180.0)) if len(ahead_of) else 180.0
        # THE TIP AND THE WEDGE ARE ONE BUDGET, and the budget is `lim`.
        # Requiring visible daylight under the workpiece -- every vertex higher
        # than `tan(alpha)` times its own horizontal distance from the contact,
        # a cone of air opening from the point it stands on -- reduces exactly
        # to `phi - t >= alpha` for a vertex in the plane of the tip.  So the
        # angle held back from the limit IS the wedge the reader sees, and at 5
        # degrees of it, which is what this page used to spend, the underside of
        # an 85 degree tip is 5 degrees off the floor and the panel reads as
        # LYING DOWN however big the caption's number.  So: spend everything
        # above the band's floor on daylight, up to `WEDGE`, and keep the rest.
        wedge = min(WEDGE, lim - WANT[0])
        if wedge < MIN_WEDGE:
            continue
        tip = min(lim - wedge, WANT[1])
        if tip < WANT[0]:
            continue
        # every height at the tip, over EVERY vertex this time.  What sits on the
        # axis has `r = 0` and stays exactly where it is, and that is the whole
        # of what makes an EDGE pivot an edge: its far end never lifts.  Leaving
        # those out of the patch test -- as the limit has to, since `atan2(0, 0)`
        # is 0 and would read as a landing at zero degrees -- let every edge
        # bearing through as "one contact"
        h = r * np.sin(np.arctan2(dz, dx) - np.radians(tip))
        # ONE CONTACT, tested directly: at the angle drawn, what still touches
        # the floor has to be one small patch round the chosen point.  Distance
        # from the contact is what the tip does not change -- the rotation is
        # about a line through it -- so the patch is measured in the pose on
        # disk.  This replaced a "the second nearest vertex must clear 4 mm"
        # rule that could not survive a refined mesh: on any fine tessellation
        # there are vertices 2 to 4 mm from the contact, they rise to 2 to 4 mm
        # at 85 degrees, and every bearing of A4 was thrown out for it
        touch = h <= TOUCH
        patch = float(np.linalg.norm(V[touch] - q, axis=1).max())
        if patch > PATCH:
            continue
        # and the same width read at CONTACT_EPS's 1.5 mm.  It is not the
        # "one point" test -- `patch` is -- but a workpiece with 50 mm of itself
        # within a millimetre and a half of the floor READS as lying on that
        # feature, whatever the tolerance says, and this page is a picture
        # before it is a measurement.  C5 has such a bearing: 0.52 mm wide at
        # `TOUCH` and 49.99 mm at 1.5 mm, its leg pad almost flat on the ground
        wide = float(np.linalg.norm(V[h <= CONTACT_EPS] - q, axis=1).max())
        if wide > LIE:
            continue
        clear = float(h[~touch].min())
        # and the patch's own dip, which is paid for by lifting the pose
        depth = float(max(0.0, -h.min()))
        if depth > DEPTH:
            continue
        # where the weight ends up: the com in the plane of the tip, turned
        xc, zc = float((com - q) @ e1), float(com[1] - q[1])
        s = float((com - q) @ np.cross(Z, e1))          # and out of that plane
        t = np.radians(tip)
        ahead = xc * np.cos(t) + zc * np.sin(t)
        out.append(dict(bearing=float(np.degrees(th)), e1=e1, point=q, vertex=j,
                        limit=lim, tip=tip, wedge=wedge, clear=clear, depth=depth,
                        patch=patch, wide=wide, ahead=ahead, aside=s,
                        balance=float(np.degrees(np.arctan2(-xc, zc))),
                        arm=float(np.hypot(ahead, s))))
    return out


def unlike(c, p):
    """How UNLIKE two candidates are, in units of `spread`'s own two tolerances.

    Under 1 is `spread`'s "the same picture" exactly: the same stable placement
    AND a contact inside `APART` AND a bearing inside `SEP`, all three at once.
    At or above 1 they are different tips, and how far above says HOW different,
    which a threshold cannot say.  Two candidates from different placements are
    infinitely unlike, which is right: the workpiece is standing on a different
    face of itself -- and is why `hull_places` must never hand the same placement
    back a second time under a second name.

    One function so that the test and the distance are the same rule and cannot
    drift apart, which is the whole reason `spread` was rewritten to call it.
    """
    if c["placement"] != p["placement"]:
        return float("inf")
    return max(float(np.linalg.norm(c["point"] - p["point"])) / APART,
               float(np.degrees(np.arccos(np.clip(c["e1"] @ p["e1"],
                                                  -1.0, 1.0)))) / SEP)


def spread(cands, k):
    """`k` tips of one workpiece that are actually DIFFERENT tips.

    Greedy, largest first, skipping anything too like a tip already taken:
    a bearing a degree away from one already drawn is the same picture, and a
    page of five of those says nothing a page of one does not.  Two candidates
    count as the same tip when they start from the same stable placement AND
    turn about contacts less than `APART` apart AND push within `SEP` of the
    same bearing -- so a different corner of the same placement is a row, and so
    is the same corner pushed a different way.  That is `unlike(c, p) < 1`.

    The order within that is `best`'s: the largest angle first, ties broken on
    the most legible overhang.  **A workpiece with fewer than `k` different tips
    gets fewer rows, and that is the whole of it**: `k` is a ceiling, this
    function is the only thing that decides a row, and nothing anywhere pads what
    it returns.  When it comes up short `sweep` widens the PLACEMENT source once
    (`hull_places`) and asks again; if the answer is still short, the page is
    short.  B's is two rows.
    """
    picked = []
    for c in sorted(cands, key=lambda c: (-c["tip"],
                                          -(c["ahead"] - abs(c["aside"])))):
        assert c["ahead"] > 0, (
            f"a candidate at {c['tip']:.1f} deg leaves the centre of mass "
            f"{1000 * c['ahead']:.2f} mm SHORT of the contact, so the workpiece "
            f"would fall back to where it started")
        if any(unlike(c, p) < 1.0 for p in picked):
            continue
        picked.append(c)
        if len(picked) == k:
            break
    return picked


def hull_places(mesh, used):
    """Every OTHER stable placement of the workpiece, off its own convex hull.

    **`poses.json` is a SAMPLE of the stable placements and not the set of
    them.**  `slides/tools/drop_sample.py` drops the part in mujoco a fixed number of
    times, clusters where it settles and keeps the commonest few: on B that is
    forty drops, nine distinct placements, five kept at 0.85 coverage.  A page
    that wants five DIFFERENT tips and gets fewer has therefore not been told
    what the workpiece can do -- it has been told what a physics sampler
    happened to see.

    A body resting on a plane rests on a facet of its own CONVEX HULL, so the
    whole set is enumerable without a simulator and without sampling anything.
    Every facet, in turn: seat it with `down`, drop the mesh onto the floor, take
    the footprint (the hull vertices left within `TOUCH` of the floor) and keep
    the facet when the centre of mass projects STRICTLY inside that polygon.
    That is `rest_pose`'s own stability test, run here as a FILTER on a placement
    nobody has vouched for -- `rest_pose`'s copy of it is an assert, and feeding
    it a facet the part cannot balance on would stop the run instead of skipping
    the facet.  Strictly inside, so `rest_pose`'s `deep <= 0` can never fire on
    what this returns.

    What comes back is deduped at `DEDUPE` -- a tessellated pad is several
    facets of one plane, and standing on any of them is the same placement --
    and against `used`, the placements `poses.json` has already given, so the
    pool never carries one placement twice under two names.  (It would not be
    harmless if it did: `unlike` calls two candidates from different placements
    infinitely unlike, so one placement entered twice would fill the page with
    near-copies of itself, which is the bug this replaced.)

    ON B: 262 hull facets, 15 of them statically stable, 5 of those already in
    `poses.json`, 10 offered here, and exactly ONE of the ten -- facet 113 --
    yields a candidate.  **Its weight comes down 1.72 mm inside its footprint**,
    against 9.55 mm for the placement `poses.json` did record, so it is a
    STATICALLY stable placement and is labelled as one: mujoco has never settled
    the bunny there in `drop_sample`'s forty drops, and nothing here re-verifies
    it dynamically.

    WHAT THIS REPLACED, AND WHY THAT WAS WRONG.  Until 2026-08-28 this slot held
    `top_up`, which filled a short page from the SAME pool by taking whatever was
    least like the rows already drawn, and labelled every row it added with the
    separation it actually achieved.  Its docstring asserted "**B is that
    workpiece and it is not a bug**" and reasoned from "five placements, 720
    bearings each, 3600 pairs" -- an enumeration that was exhaustive over
    `poses.json` and `poses.json` was 5 placements of 15.  B's page came out five
    rows of one corner of one placement over a 3.5 degree fan of bearing, all
    five at exactly 60.0 degrees because `wedge = min(WEDGE, lim - WANT[0])`
    absorbs the whole surplus whenever the limit is under 80.  The FINDING it
    reached is still right and the REASON it gave was not: **the placement count
    was never the limit.**  A round workpiece rolls onto another part of itself
    before the tip clears its own balance point, and B's fifteen stable
    placements have `limit - balance` windows of -2.94 to 39.73 degrees, over 30
    on three of them and clearing `WANT[0] + MIN_WEDGE` = 68 degrees of landing
    limit on two.  Two is still the answer for B; it is now two ROWS instead of
    five, because a page that cannot fill itself with different tips is short and
    says so by being short.

    Returns `(facet, T_place, margin)` per placement, in facet order.
    """
    hull = mesh.convex_hull
    N = np.asarray(hull.face_normals)
    HV = np.asarray(hull.vertices)
    V = np.asarray(mesh.vertices)
    seen = list(used)
    out = []
    for j in range(len(N)):
        F = down(N[j])
        drop = -float((V @ F.T)[:, 1].min())
        P = HV @ F.T
        P[:, 1] += drop
        foot = COORD.floor(P[P[:, 1] <= TOUCH])
        if len(foot) < 3:
            continue
        com = F @ mesh.center_mass
        com[1] += drop
        try:
            poly = ConvexHull(foot)
        except QhullError:
            continue            # the foot is a segment or a point, so this is
                                # an edge or a vertex balance and not a placement
        deep = float((poly.equations[:, :2] @ COORD.floor(com) + poly.equations[:, 2]).max())
        if deep >= 0.0:
            continue            # the weight comes down outside the footprint
        if any(float(np.degrees(np.arccos(np.clip(N[j] @ m, -1.0, 1.0)))) < DEDUPE
               for m in seen):
            continue            # the same face of the workpiece, already offered
        T = np.eye(4)
        T[:3, :3], T[1, 3] = F, drop
        out.append((j, T, float(-deep)))
        seen.append(N[j])
    return out


def admissible(mesh, T_star):
    """region.json's `seed_rule`, on the pose this page draws.

    Three tests, all of them that file's own words: the face turns up far enough
    for a gun above the workpiece to meet it, its centre stands clear of the
    floor band, and a ray leaving it along its own normal gets away without
    hitting the part again.
    """
    n = mesh.face_normals @ T_star[:3, :3].T
    c = mesh.triangles_center @ T_star[:3, :3].T + T_star[:3, 3]
    up = (n[:, 1] > GUN_MIN_ELEVATION) & (c[:, 1] > CONTACT_EPS)
    seen = ~mesh.ray.intersects_any(
        ray_origins=mesh.triangles_center + 1e-5 * mesh.face_normals,
        ray_directions=mesh.face_normals)
    return up & seen


def grow(mesh, seed, target, cap, blocked, adj, w):
    """region.json's `growth_rule`: a geodesic disc, in area, off the floor.

    Dijkstra on the face-adjacency graph weighted by the distance between face
    centres, which is a disc in surface distance and connected by construction.
    The floor band is not entered.  A face that would carry the patch past `cap`
    -- the top of the band -- is never added, though the disc still grows past
    it, so the stopping rule can never turn into a hole in the middle.
    """
    area = mesh.area_faces
    seen = np.zeros(len(area), bool)
    take = np.zeros(len(area), bool)
    got = 0.0
    heap = [(0.0, int(seed))]
    while heap and got < target:
        d, f = heapq.heappop(heap)
        if seen[f] or blocked[f]:
            continue
        seen[f] = True
        if got + area[f] <= cap:
            take[f] = True
            got += area[f]
        for g, e in zip(adj[f], w[f]):
            if not seen[g] and not blocked[g]:
                heapq.heappush(heap, (d + e, int(g)))
    return take, got


def components(mesh, take):
    """How many connected patches the drawn region is, over face adjacency."""
    idx = np.flatnonzero(take)
    if not len(idx):
        return 0
    pos = -np.ones(len(mesh.faces), int)
    pos[idx] = np.arange(len(idx))
    pair = mesh.face_adjacency
    keep = take[pair[:, 0]] & take[pair[:, 1]]
    g = trimesh.graph.connected_components(pos[pair[keep]], nodes=np.arange(len(idx)))
    return len(g)


def regions(mesh, T_star, name):
    """`N_SEEDS` candidate patches, each one connected and inside the band."""
    rng = np.random.default_rng(SEED ^ zlib.crc32(name.encode()))
    fit = admissible(mesh, T_star)
    ok = np.flatnonzero(fit)
    assert len(ok), "no face of this workpiece is admissible as a region seed"
    area = mesh.area_faces
    p = area[ok] / area[ok].sum()
    n = min(N_SEEDS, len(ok))
    seeds = rng.choice(ok, size=n, replace=False, p=p)
    target = rng.uniform(*TARGET_AREA, size=n) * mesh.area
    cap = BAND_AREA[1] * mesh.area
    # THE DISC GROWS ONLY THROUGH FACES THE PROCESS CAN REACH, which is the same
    # test the SEED is drawn from and was the whole of what growth used to skip.
    # `region.json`'s rule as written filters the seed and then lets the disc run
    # geodesically with nothing in its way but the floor band, so a patch starts
    # on a face a gun above the workpiece can see and crawls over the edge onto
    # the underside, which that gun cannot reach at all: 0.1 % to 42.9 % of the
    # drawn patches were surface no declared process touches.  It is not a
    # cosmetic point.  Those faces have their inward normal pointing UP, so the
    # 15 degree push cone on them contains directions within a hair of straight
    # up, and a push of one body weight there leaves the assembly's total normal
    # force at 0.027 of a body weight -- on the point of floating, with the
    # centre of pressure 3 to 8 part widths away (`slides/sys_floor`).  Growing
    # inside the admissible set brings that to 0.31-0.60.
    blocked = ~fit
    pair = mesh.face_adjacency
    adj = [[] for _ in range(len(mesh.faces))]
    w = [[] for _ in range(len(mesh.faces))]
    d = np.linalg.norm(mesh.triangles_center[pair[:, 0]]
                       - mesh.triangles_center[pair[:, 1]], axis=1)
    for (i, j), e in zip(pair, d):
        adj[i].append(j)
        w[i].append(e)
        adj[j].append(i)
        w[j].append(e)
    out = []
    for s, t in zip(seeds, target):
        take, got = grow(mesh, s, t, cap, blocked, adj, w)
        frac = got / mesh.area
        n = components(mesh, take)
        out.append(dict(seed=int(s), take=take, frac=float(frac), n=int(n),
                        faces=int(take.sum()), want=float(t / mesh.area),
                        band=BAND_AREA[0] <= frac <= BAND_AREA[1] and n == 1))
    return out


# ------------------------------------------------------------------ a row ---

def sweep(name):
    """One workpiece: `ROWS` different big tips of it, drawn as one page."""
    d = obj_path(name)
    raw = trimesh.load(d / "mesh.stl", force="mesh")
    mesh, rounds = refine(raw)
    places = read_json(d / "poses.json")["poses"]
    print(f"\n=== {name}   {len(raw.faces)} faces, largest "
          f"{100 * (raw.area_faces / raw.area).max():.2f} % of the surface "
          f"\N{RIGHTWARDS ARROW} {len(mesh.faces)} after {rounds} uniform "
          f"round(s) (target: no face over {100 * QUANTUM:g} %), largest now "
          f"{100 * (mesh.area_faces / mesh.area).max():.2f} %, "
          f"{1000 * mesh.extents[0]:.0f} x {1000 * mesh.extents[1]:.0f} x "
          f"{1000 * mesh.extents[2]:.0f} mm, {len(places)} stable placements",
          flush=True)

    pool, rests = [], []
    for i, p in enumerate(places):
        T_rest, tilt, drop, inside = rest_pose(mesh, np.asarray(p["T_world_mesh"]))
        rests.append(T_rest)
        cs = candidates(mesh, T_rest)
        print(f"  placement {i}: seated by taking {tilt:.3f}\N{DEGREE SIGN} of "
              f"residual tilt out and dropping {1000 * drop:5.2f} mm, weight "
              f"{1000 * inside:5.1f} mm inside the footprint; "
              f"{len(cs):4d} of {NB} bearings give a {WANT[0]:.0f}-"
              f"{WANT[1]:.0f}\N{DEGREE SIGN} tip on ONE point"
              + (f", limits {min(c['limit'] for c in cs):5.1f}-"
                 f"{max(c['limit'] for c in cs):5.1f}\N{DEGREE SIGN}"
                 if cs else ""), flush=True)
        for c in cs:
            c["placement"], c["T_rest"] = i, T_rest
        pool += cs
    picks = spread(pool, ROWS) if pool else []
    if pool:
        print(f"  {len(pool)} bearings over {len(places)} placements, tips "
              f"{min(c['tip'] for c in pool):.1f}-{max(c['tip'] for c in pool):.1f}"
              f"\N{DEGREE SIGN}; {len(picks)} of them are DIFFERENT tips by "
              f"{SEP:.0f}\N{DEGREE SIGN} of bearing or {1000 * APART:.0f} mm of "
              f"contact, and those are the rows", flush=True)
    if len(picks) < ROWS:
        # THE PLACEMENT SOURCE WIDENS, AND ONLY HERE.  `poses.json` is what a
        # physics sampler happened to find; the convex hull carries the rest.
        # A page `spread` already fills never reaches this branch, which is what
        # keeps A1-f's and C5's pages byte-identical to what they were before it
        # existed.  See `hull_places`.
        more = hull_places(mesh, [t[:3, :3].T @ -Z for t in rests])
        print(f"  \N{EM DASH} which is under the {ROWS} this page carries, so the "
              f"PLACEMENT SOURCE widens (`hull_places`): `poses.json` holds "
              f"{len(places)} placements a physics drop happened to settle in, "
              f"and of this workpiece's "
              f"{len(mesh.convex_hull.faces)} convex-hull facets "
              f"{len(more)} MORE are statically stable and are none of those "
              f"{len(places)}", flush=True)
        for j, T_place, held in more:
            T_rest, tilt, drop, inside = rest_pose(mesh, T_place)
            assert abs(inside - held) < 1e-6, (
                f"hull facet {j}: the screen `hull_places` applied and the "
                f"assert `rest_pose` makes disagree by "
                f"{1000 * abs(inside - held):.4f} mm, so they are not the "
                f"same stability test")
            cs = candidates(mesh, T_rest)
            # AND THE ONES THAT STOP SHORT OF THEIR OWN BALANCE POINT ARE
            # SCREENED OUT HERE.  `spread` asserts `ahead > 0` rather than
            # filtering it, on purpose -- a recorded placement offering a
            # candidate that would fall BACK is a fault upstream and stops the
            # run.  A placement nobody has vouched for is different: on a round
            # workpiece a bearing can clear the landing limit and still leave
            # the weight behind the contact, and that is a fact about the part,
            # not a fault.  It is counted and printed, never drawn.
            back = [c for c in cs if c["ahead"] <= 0]
            cs = [c for c in cs if c["ahead"] > 0]
            print(f"  hull facet {j}: seated exactly, dropped "
                  f"{1000 * drop:5.2f} mm, weight {1000 * inside:5.1f} mm inside "
                  f"the footprint; {len(cs):4d} of {NB} bearings give a "
                  f"{WANT[0]:.0f}-{WANT[1]:.0f}\N{DEGREE SIGN} tip on ONE point"
                  + (f", limits {min(c['limit'] for c in cs):5.1f}-"
                     f"{max(c['limit'] for c in cs):5.1f}\N{DEGREE SIGN}"
                     if cs else "")
                  + (f" ({len(back)} more cleared the limit but leave the weight "
                     f"SHORT of the contact, so they are not offered)"
                     if back else ""), flush=True)
            for c in cs:
                c["placement"], c["T_rest"] = f"hull {j}", T_rest
            pool += cs
        picks = spread(pool, ROWS)
        print(f"  {len(pool)} bearings over {len(places) + len(more)} placements "
              f"now, and {len(picks)} of them are DIFFERENT tips \N{EM DASH} "
              f"**that is the row count**.  Nothing is padded: a row is a "
              f"different tip or it is not drawn", flush=True)
    assert picks, f"{name}: no placement of this workpiece, recorded or " \
                  f"enumerated, tips {WANT[0]:.0f} deg onto one point of its " \
                  f"own footprint"
    grid, labels, notes, out = [], [], [], []
    for k, got in enumerate(picks):
        panels, label, note, rec = one(name, d, mesh, got, k, len(pool))
        grid.append(panels)
        labels.append(label)
        notes.append(note)
        out.append(rec)
    page(HERE / f"tip_{name}.png", name, grid, labels, notes)
    return out


def one(name, d, mesh, got, k, tied):
    """One row: the pose built, the region grown, three panels drawn."""
    T_rest = got["T_rest"]
    axis = np.cross(Z, got["e1"])
    point = got["point"]
    tip, mid = got["tip"], MIDWAY * got["tip"]
    # the rotation, and then the re-seat that `candidates` priced.  Both
    # directions of it are a fraction of a millimetre and both are the same
    # thing -- a real contact is a rolling one, and a mesh's is a chord of it:
    # DOWN by up to `TOUCH`, because the vertex the tip turns on is only
    # required to be on the floor to that tolerance and the pose would otherwise
    # stand on a 0.07 mm pedestal; UP by up to `DEPTH`, because the fillet's
    # next facet dips under as the contact rolls onto it.
    def turn(a):
        R = rot_about_line(axis, point, np.radians(a)) @ T_rest
        V = np.asarray(mesh.vertices) @ R[:3, :3].T + R[:3, 3]
        R[1, 3] -= float(V[:, 1].min())
        return R

    T_star, T_mid = turn(tip), turn(mid)
    lift = float(T_star[2, 3]
                 - (rot_about_line(axis, point, np.radians(tip)) @ T_rest)[1, 3])

    # the pose is legal, checked on the drawing's own transform rather than on
    # the closed form that chose it.  THE CONTACT IS READ BACK OFF THE DRAWN
    # POSE -- the lowest vertex of the workpiece as it is drawn, which after the
    # re-seat is exactly on the floor -- so the marker, the couple's arm and the
    # red bar all stand on the point the part actually touches rather than on
    # the point the rotation was built about.  `roll` is the distance between
    # those two, and it is the contact migration this page allows and prints.
    V = np.asarray(mesh.vertices) @ T_star[:3, :3].T + T_star[:3, 3]
    low = float(V[:, 1].min())
    contact = V[int(np.argmin(V[:, 1]))].copy()
    contact[1] = 0.0
    roll = float(np.linalg.norm(COORD.floor(contact) - COORD.floor(point)))
    point = contact
    touch = int((V[:, 1] <= TOUCH).sum())
    band = int((V[:, 1] <= CONTACT_EPS).sum())
    spread = float(np.linalg.norm(
        COORD.floor(V[V[:, 1] <= TOUCH]) - COORD.floor(contact), axis=1).max())
    # and the same width read at the 1.5 mm tolerance, which is the generous
    # reading of "touching" and the one a sceptic would ask for
    wide = float(np.linalg.norm(
        COORD.floor(V[V[:, 1] <= CONTACT_EPS]) - COORD.floor(contact), axis=1).max())
    assert low > -1e-12, f"{name}: the target pose puts the workpiece " \
                         f"{-1000 * low:.3f} mm through the floor"
    assert low < 1e-9, f"{name}: the target pose floats {1000 * low:.3f} mm " \
                       f"above the floor -- it is on a pedestal, not on the ground"
    assert spread <= PATCH, f"{name}: the workpiece touches the floor {1000 * spread:.2f} " \
                            f"mm from the point it turns on, which is not one contact"
    print(f"  placement {got['placement']}, bearing {got['bearing']:5.1f}"
          f"\N{DEGREE SIGN}, contact vertex {got['vertex']} \N{EM DASH} tip "
          f"{tip:5.1f}\N{DEGREE SIGN} of a {got['limit']:5.1f}\N{DEGREE SIGN} "
          f"limit, and the balance point is at {got['balance']:5.1f}"
          f"\N{DEGREE SIGN}, so the pose is {tip - got['balance']:.1f}"
          f"\N{DEGREE SIGN} PAST it: gravity drives it ON, not back\n"
          f"  the contact: what still touches at that angle is {1000 * spread:.2f} "
          f"mm wide at {1000 * TOUCH:g} mm ({touch} vertices) and "
          f"{1000 * wide:.2f} mm wide at CONTACT_EPS's 1.5 mm ({band}); the next "
          f"thing outside it stands {1000 * got['clear']:.1f} mm up; inside the "
          f"patch the tessellation rolls {1000 * roll:.2f} mm and dips "
          f"{1e6 * got['depth']:.0f} \N{MICRO SIGN}m, so the pose is re-seated "
          f"by {1e6 * lift:+.0f} \N{MICRO SIGN}m\n"
          f"  {tied} bearings were within 1\N{DEGREE SIGN} of this tip and this "
          f"one draws the weight furthest across the page "
          f"({1000 * got['ahead']:.1f} mm past the contact, "
          f"{1000 * abs(got['aside']):.1f} mm along the eye)", flush=True)

    ex = dict(axis=axis.tolist(), point=point.tolist(), pivot="point")
    arm, contact, foot, _, ident = margin(mesh, T_star, ex)
    com_rest = T_rest[:3, :3] @ mesh.center_mass + T_rest[:3, 3]
    com_star = T_star[:3, :3] @ mesh.center_mass + T_star[:3, 3]
    # the centre of mass at the target pose is kept because it is what makes the
    # margin readable as a statement -- it is 105.5 mm from the contact
    # horizontally on A1-f and the plumb line lands that far outside the one
    # point holding the part up -- and it is printed rather than drawn
    marks = np.asarray(point, float)[None]
    box = np.vstack([np.asarray(mesh.vertices) @ t[:3, :3].T + t[:3, 3]
                     for t in (T_rest, T_mid, T_star)] + [marks])
    nrm = outward(axis, point, com_rest)
    assert float(nrm @ got["e1"]) > 1 - 1e-9, (
        "the tip does not turn the workpiece toward the outside of its own "
        "footprint")

    def frame(elev, azim):
        """`tip_sequence.sweep`'s closure: the camera a candidate is DRAWN with.

        Two passes, because the arrow's length is quoted in box diagonals: fit
        the part and the contact, then refit around the tail that scale implies.
        """
        cam0 = fit(elev, azim, box)
        q, u, why = push_site(mesh, T_mid, axis, point, 1.0, cam0.fwd)
        size = float(np.linalg.norm(box.max(axis=0) - box.min(axis=0)))
        tail, beyond = q - ARROW_L * size * u, q + ARROW_L * size * u
        cam = fit(elev, azim, np.vstack([box, tail[None]]))
        near = float(((np.asarray(mesh.vertices) @ T_mid[:3, :3].T + T_mid[:3, 3]
                       - cam.eye) @ cam.fwd).min())
        return cam, pull_forward(tail, beyond, ARROW_W * size, cam, near), why

    eyes = cameras(mesh, T_star, T_mid, marks, nrm, axis, frame)
    # ONE score in ONE currency -- pixels on the page between the mark and where
    # a reader takes the ground contact to be.  A camera can fail two ways and
    # they are the same failure: the workpiece hangs BELOW the mark (`drop`), or
    # it stands in FRONT of it, which puts the flat mark on a face instead of on
    # the floor.  Being hidden is charged `HIDDEN` pixels, about the width of the
    # marker itself, so a hidden-but-low camera beats a visible-but-high one and
    # a visible one wins ties.
    eye = min(eyes, key=lambda e: e["drop"] + HIDDEN * e["dark"] + e["walk"])
    print(f"  cameras: {len(eyes)} offered ({', '.join(f'{o:.0f}' for o in OUT)}"
          f"\N{DEGREE SIGN} either side of the axis, the eye at either end) "
          f"\N{EM DASH} on the inner rung, "
          + ", ".join(f"{e['out']:+.0f}/{e['side']:+.0f} "
                      f"{'HIDES' if e['dark'] else 'shows'} it, hangs "
                      f"{e['drop']:.0f} px under it"
                      for e in eyes if abs(e["out"]) == OUT[0])
          + f".  TAKEN: {eye['out']:+.0f}/{eye['side']:+.0f} at "
            f"{eye['drop'] + HIDDEN * eye['dark'] + eye['walk']:.0f} px of cost "
            f"({'hidden' if eye['dark'] else 'clear'}, {eye['drop']:.0f} px of "
            f"hang, {eye['walk']:.0f} px of walk), and the region is then "
            f"chosen for it", flush=True)
    cand = regions(mesh, T_star, name)
    print(f"  {N_SEEDS} candidate work regions, region.json's own rules, on THIS "
          f"pose (band {BAND_AREA[0]:.0%}-{BAND_AREA[1]:.0%}, target drawn from "
          f"{TARGET_AREA[0]:.0%}-{TARGET_AREA[1]:.0%}):", flush=True)
    pick = None
    for r in cand:
        if not r["band"]:
            print(f"    seed {r['seed']:6d}: {100 * r['frac']:5.2f} % in "
                  f"{r['n']} patch(es) \N{EM DASH} OUT OF BAND, not offered",
                  flush=True)
            continue
        drawn = shown(mesh, T_star, r["take"], eye["cam"])
        seen = facing(mesh, T_star, r["take"], eye["cam"])
        r.update(seen=seen, drawn=drawn, **eye)
        print(f"    seed {r['seed']:6d}: {100 * r['frac']:5.2f} % of the surface "
              f"({r['faces']:6d} faces, {r['n']} patch), asked for "
              f"{100 * r['want']:5.2f} %; its best camera turns {100 * seen:5.1f} % "
              f"of it toward the eye and draws it over {100 * drawn:5.1f} % of the "
              f"part", flush=True)
        if pick is None or drawn > pick["drawn"]:
            pick = r
    assert pick is not None, f"{name}: no candidate region landed inside the band"
    cam = pick["cam"]
    _, arrow, why = frame(cam.elev, cam.azim)
    phi = phi_of(ROW_ELEV, abs(pick["out"]))
    old = frame(ELEV, AZIM)[0]
    print(f"  DRAWN: seed {pick['seed']}, {100 * pick['frac']:.2f} % of the "
          f"surface in one patch, {100 * pick['seen']:.1f} % of it turned toward "
          f"the eye and covering {100 * pick['drawn']:.1f} % of the drawn part "
          f"(METHOD \N{SECTION SIGN}3.0's fixed camera would have turned "
          f"{100 * facing(mesh, T_star, pick['take'], old):.1f} % \N{RIGHTWARDS ARROW} "
          f"{100 * shown(mesh, T_star, pick['take'], old):.1f} %)\n"
          f"  camera elev {cam.elev:.0f}\N{DEGREE SIGN} azim {cam.azim:7.1f}"
          f"\N{DEGREE SIGN}, {pick['out']:4.1f}\N{DEGREE SIGN} outside the axis "
          f"\N{RIGHTWARDS ARROW} phi {phi:4.1f}\N{DEGREE SIGN}; the tip draws as "
          f"{apparent_tip(tip, phi):5.1f}\N{DEGREE SIGN} of its {tip:5.1f}"
          f"\N{DEGREE SIGN} and the part moves "
          f"{100 * travel(mesh, T_rest, T_star, cam):.0f} % of its own width\n"
          f"  the contact, drawn as a sphere IN the scene, would be "
          f"{'HIDDEN by the workpiece' if buried(mesh, T_star, marks, cam) else 'clear of it'}"
          f" at T* and {'HIDDEN' if buried(mesh, T_mid, marks, cam) else 'clear'} "
          f"halfway; it is drawn flat OVER the render either way, so the point "
          f"the tip turns on is visible on all three panels\n"
          f"  METHOD \N{SECTION SIGN}11.4's w(0) at this pose: "
          f"{1000 * arm:.2f} mm of arm owed at one body weight, with the process "
          f"OFF — the weight comes down at "
          f"({1000 * com_star[0]:.1f}, {1000 * com_star[1]:.1f}, "
          f"{1000 * com_star[2]:.1f}) mm and the one contact is that far away "
          f"horizontally (the couple's arm and the plumb offset agree to "
          f"{ident:.1e} m).  NOT DRAWN: this page marks the contact and nothing "
          f"else"
          + (f"\n  the push arrow sits where it does even though {why}" if why else ""),
          flush=True)

    tmp = d / TMP
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        bare = paint_parts(mesh, np.zeros(len(mesh.faces), bool), tmp,
                           f"big_{name}_r{k}_bare")
        work = paint_parts(mesh, pick["take"], tmp, f"big_{name}_r{k}_work")
        # THE PIVOT AND NOTHING ELSE.  `tip_sequence`'s column 3 also draws the
        # centre of mass, its plumb line and the red tipping-margin bar; they are
        # computed here, asserted here and printed here on every run, and they
        # are NOT drawn -- the page was asked for one mark, the point the
        # workpiece turns on.  The push arrow stays, because it is not a mark on
        # the workpiece but the agent that puts it there.
        panels = [apart(render(name, T_rest, bare, cam, [("pivot", (marks,))])),
                  apart(render(name, T_mid, bare, cam,
                               [("arrow", arrow), ("pivot", (marks,))])),
                  apart(render(name, T_star, work, cam, [("pivot", (marks,))]))]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # NO ANGLES ON THE PANELS.  They were `tip_sequence`'s, and they are in the
    # log, in `big_tip.md`'s table and in the row's own record; on the page they
    # were the last text left on a picture whose point is that it needs none.
    note = ("", "", "")
    # the row's own ingredients travel with its record, so a page in another
    # folder can draw THE SAME POSE THROUGH THE SAME CAMERA rather than
    # re-deriving it and drifting.  `slides/sys_floor/on_the_floor.py` is the
    # caller this exists for.
    return panels, f"tip {k + 1}", note, dict(
        name=name, tip=tip, limit=got["limit"], balance=got["balance"],
        bearing=got["bearing"], T_star=T_star, T_mid=T_mid, T_rest=T_rest,
        cam=cam, take=pick["take"], contact=contact, axis=axis, box=box,
        arm=arm, frac=pick["frac"], seen=pick["seen"], drawn=pick["drawn"], phi=phi,
        apparent=apparent_tip(tip, phi), placement=got["placement"],
        vertex=got["vertex"], seed=pick["seed"])


def main():
    print(f"objects {list(OBJECTS)} \N{EM DASH} one PAGE each, as many rows as "
          f"the workpiece has DIFFERENT big tips up to {ROWS}, three panels a "
          f"row.  These are "
          f"NOT tips.json's poses: each is turned {WANT[0]:.0f} to {WANT[1]:.0f}"
          f"\N{DEGREE SIGN} about ONE point of the footprint, PAST the balance "
          f"point, and its work region is regenerated on it under region.json's "
          f"own rules because region.json holds patches for the other poses.",
          flush=True)
    rows = []
    for name in OBJECTS:
        rows += sweep(name)
    print(f"\nTHE {len(rows)} ROWS, as numbers:", flush=True)
    print(f"  {'object':8s} {'placement':>10s} {'pivot':>8s} {'bearing':>8s} "
          f"{'tip':>7s} {'limit':>7s} {'balance':>8s} {'past by':>8s} "
          f"{'w(0)':>9s} {'seed':>7s} {'region':>8s} {'of page':>8s} "
          f"{'draws as':>9s}", flush=True)
    for r in rows:
        print(f"  {r['name']:8s} {str(r['placement']):>10s} "
              f"{'v' + str(r['vertex']):>8s} {r['bearing']:7.1f}\N{DEGREE SIGN} "
              f"{r['tip']:6.1f}\N{DEGREE SIGN} "
              f"{r['limit']:6.1f}\N{DEGREE SIGN} {r['balance']:7.1f}"
              f"\N{DEGREE SIGN} {r['tip'] - r['balance']:7.1f}\N{DEGREE SIGN} "
              f"{1000 * r['arm']:7.1f} mm {r['seed']:7d} {100 * r['frac']:7.2f} % "
              f"{100 * r['drawn']:7.1f} % {r['apparent']:8.1f}\N{DEGREE SIGN}",
              flush=True)
    print(f"\nEvery tip is inside [{WANT[0]:.0f}, {WANT[1]:.0f}]\N{DEGREE SIGN}, "
          f"every one is past its own balance point, every region is one patch "
          f"inside [{100 * BAND_AREA[0]:.0f}, {100 * BAND_AREA[1]:.0f}] % of the "
          f"surface, and no workpiece touches the floor anywhere but the one "
          f"point it turns on.  All {len(rows)} rows are DIFFERENT tips by "
          f"`spread`'s rule \N{EM DASH} a page carries as many as its workpiece "
          f"has, up to {ROWS}, and none of them is padded.", flush=True)


if __name__ == "__main__":
    main()
