"""Three pushes on a ball: which directions they reach, and how hard.

Three contacts, each able to push one way only and each capped at the same F --
one support's whole strength. What they can supply together is the cone of
non-negative combinations

    v = sum_i c_i u_i,   c_i >= 0

and, since scaling the whole combination scales every coefficient together, the
cap binds on the largest of them:

    m(v) = F / max_i c_i

That single quantity is the whole figure. `spanned` in slides/tools/dimension.py answers
WHICH directions the three reach and says nothing about how hard, and on its own
that half is not merely incomplete, it is the misleading half. Here both halves
are drawn at once: the cone is the painted patch, and the paint IS the magnitude.

Three balls, one camera, one colour scale, and ONE ARRANGEMENT POSED THE SAME WAY
in all three: u_1 straight up, because in the scene these figures are about one
generator is not a chock but the FLOOR, and the floor's push does not move when
the chocks are re-cut. What changes across the row is the other two.

  (a) the three pushes MUTUALLY SQUARE -- one up and two horizontal, the
      coordinate axes. The cone is the corner of a cube: an OCTANT, exactly an
      eighth of the ball, and every direction in it takes F or more -- from
      exactly F at the three pushes up to sqrt(3) F down the middle.

  (b) the three pushes CLOSE TOGETHER, 50 deg. The two swing UP towards the
      vertical one and cluster round it -- shims driven under an edge rather than
      blocks stood beside it. A much smaller patch -- 2.9 % of the ball against
      12.5 % -- and again everything in it takes F or more, now up to 2.62 F.
      Narrowing buys strength in the middle and costs reach, and it does NOT
      raise the floor: pushing along one support's own ray decomposes as
      c = (1,0,0) and no other support can contribute, so that direction is worth
      exactly F whatever the angles are. The floor is pinned at F for every
      arrangement up to 90 deg and cannot be raised.

  (c) the three pushes SPREAD WIDE, 110 deg. The two swing DOWN to 20 deg below
      horizontal and splay apart -- clamps hooked over an edge. Twice the reach
      of the octant -- 25.5 % of the ball -- and the forces inside it have fallen
      through F: the weakest direction of the patch is worth 0.80 F, and only
      4.9 % of the patch still takes a full support's worth. Reach went up and
      capability went down, which is why the angular size of a cone is the wrong
      thing to quote about a support arrangement.

So 90 deg is a corner and not a preference, and the mechanism is one direction
per PAIR. The weakest direction of a pair {u_j, u_k} is the one perpendicular to
both, d_i = u_j x u_k: neither of them can put anything along it, so only the
third support is left to push, and wherever the three can reach d_i at all it is
worth m(d_i) = F (u_i . d_i), which is at most F. As the pairwise angle opens,
d_i walks INTO the patch -- outside it below 90 deg, where the three cannot reach
it and the bound is vacuous; exactly on top of u_i at 90 deg, where u_i . d_i = 1
and the bound is tight at F; strictly inside past 90 deg, where it bites. The crossing is algebraically
clean: decomposing d_i gives the two other coefficients equal to

    -cos t / [ (1 - cos t)(1 + 2 cos t) |u^i| ]

whose every factor but the leading `-cos t` is positive over the whole range the
three directions are independent in. The sign therefore turns at cos t = 0 and
nowhere else.

THE 90 DEG THRESHOLD IS ABOUT THREE SUPPORTS AND DOES NOT SURVIVE A FOURTH, and
this docstring used to say the opposite here -- that "a fourth support does not
mend a direction that two of its neighbours are square to". It does. What bounds
the blind direction d_i = unit(u_j x u_k) is what the OTHER supports can put along
it,

    h(d_i) = F sum_{l not in {j,k}} max(0, d_i . u_l)

which with THREE supports has ONE term, F cos(beta), and so is at most F, with
equality only when the third support IS d_i -- the mutually square arrangement,
and the whole of the 90 deg rule. With four or more the sum acquires further terms
and can reach F: the square pyramid {+z, +-x, +-y} contains two EXACTLY OPPOSED
pairs, 180 deg apart, and its weakest direction is still exactly F, because the
direction its pair {+z, +x} is blind to is +-y and +y is itself a support.
Meanwhile the regular tetrahedron, every pair 109.47 deg, falls to 0.8165 F. Past
three supports the pairwise angle is not the criterion at all.
`slides/tools/more_pushes.py` draws that.

Read the paint as two states with a hard break, not as one scale: red is a
direction the three can push but not with a full support behind it, blue is one
support's strength or more, and bare shell is a direction they cannot push at
all. The break at F is a jump in lightness as well as in hue, because where that
edge runs is the whole question.

    python slides/tools/three_pushes.py   ->  slides/tools/figures/three_pushes.png
"""
from __future__ import annotations

from common import figure_path

import itertools
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
from matplotlib.colors import LinearSegmentedColormap          # noqa: E402
from matplotlib.patches import FancyArrowPatch                 # noqa: E402
from mpl_toolkits.mplot3d import proj3d                        # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection        # noqa: E402

# the house palette, the same one dimension.py takes
INK, MUTED, PAPER = "#1b1b1a", "#6b6b66", "#ffffff"
ORANGE = "#E08A24"
# the unpainted shell: the directions the three pushes cannot reach at all. It is
# dimension.py's own SHELL, and it is left NEUTRAL on purpose. An opaque ball has
# to paint that set SOMETHING -- every pixel says a value, so "unreachable" ends
# up as the darkest step of the scale and reads as "reachable, but barely". A
# translucent shell says it by being unpainted: one fewer colour to key, and it
# reads as "nothing here"
SHELL = "#ffffff"
F = 1.0

# --- the two ramps, and the break between them ------------------------------
#
# Shared verbatim with more_pushes.py so that a colour means the same thing in
# both figures that answer this question: below F a hot red, at and above F a
# blue that leaves white fast. The pale band just above F is deliberately narrow,
# because a direction that only just reaches F is the fact panels (a) and (b) are
# about -- their three push directions sit at exactly F and have to show up as
# bright spots at the corners of an otherwise deep patch.
DEAD = LinearSegmentedColormap.from_list(
    "dead", ["#171215", "#3d1420", "#6e1e28", "#9c2a26", "#c03a25"])
LIVE = LinearSegmentedColormap.from_list(
    "live", [(0.00, "#f6fcfd"), (0.06, "#d3ebf4"), (0.18, "#9fd2e6"),
             (0.42, "#5ba7ca"), (0.70, "#2f7ba2"), (1.00, "#17415e")])

# the three arrangements, in the order the argument wants them rather than in
# order of angle. 90 is the hub: it is the last arrangement that still guarantees
# F everywhere, and the two panels beside it are the two ways off it. Narrowing
# (50) is drawn second because the thing it fails to buy -- a better floor -- is
# only surprising once the reader has seen that 90 already sits at the floor
CASES = ((90.0, "a"), (50.0, "b"), (110.0, "c"))
# what the blue ramp's far end stands for: the largest magnitude anywhere in the
# figure, which is the middle of the 50 deg patch. Read off the closed form in
# `strongest` rather than typed, so the key cannot drift from the paint.
# (more_pushes.py's own top is 3.54 F, the middle of its five-ray cluster; the
# two figures share the ramp and the break and differ in this one number, because
# they contain different maxima and neither may clip its own strongest point)
VMAX = None                                                    # set in main()

SUBDIV = 5                                                     # 20 * 4^5 = 20480 cells
# where an arrow leaving the ball starts, and how long it is drawn ON THE PAGE.
# Length carries no number in this figure -- the paint carries the magnitude --
# so every arrow is drawn the same page length, which is what says the three
# supports are three of the same thing. See `onpage`
STAND, UNIT_LN = 1.03, 0.42
MARK, LABEL_OUT = 0.34, 1.52   # the angle arcs' radius, and where the number sits
# how much of the panel the ball is allowed to take, and where the ball sits in
# it. The framing box has to stay a CUBE -- mplot3d rescales whatever box it is
# given to a fixed diagonal and then stretches the result to fill the axes
# rectangle, so a box that is not a cube comes out both shrunk and squashed --
# which leaves exactly two levers: how big the cube is drawn, and where its
# centre is put. Both are set by what has to fit rather than by taste.
#
# An arrow reaches 1.03 * seen + 0.42 ball radii from the middle of the page, and
# the two terms trade against each other: the arrows drawn longest are the ones
# pointing most nearly at the reader, whose tails the same foreshortening has
# pulled in. Measured over all three panels, the drawing runs from -1.000 to
# +1.388 radii vertically and from -1.062 to +1.305 sideways -- 2.388 x 2.367 --
# and it is NOT centred on the ball, because the vertical push now stands up out
# of the top of every panel and nothing balances it underneath. The top of that
# is the vertical arrow's own head, in all three panels at once.
#
# So the cube is lowered in the frame by LOWER of a ball radius along world +z,
# which is page-up: the cube stays a cube and only its centre moves. Without it
# the drawing runs to 0.906 of the page height and the arrowhead is drawn THROUGH
# the second line of the caption -- measured, and it is what this revision's
# first render did. At LOWER = 0.28 the drawing occupies 0.286 .. 0.844 of the
# page, which clears the caption line above by 0.018 and the number line below by
# 0.019; ZOOM is then capped by the two of them together, not by the panel.
ZOOM, LOWER = 1.80, 0.28
# the least of a length the projection may leave on the page before `onpage`
# stops undoing it. In this pose no push comes near the view axis -- the closest
# is the 50 deg panel's u_3 at 27.17 deg, which keeps 0.457 of its length, and
# `pose` fails the run if anything gets inside 20 deg -- so the clamp never binds
# and is a guard and nothing else, there so that moving the camera degrades the
# figure instead of exploding it. (It DID bind in the pose this replaced, where
# the camera was aimed down the tripod's own axis at one of the pushes and that
# push kept 0.245 of its length.)
SEEN_FLOOR = 0.20

# --------------------------------------------------------------------- camera
#
# ONE camera for all three panels, and it has to be one: the reader compares the
# SIZE of the three patches, and a camera that moved between panels would make
# that comparison a lie. The arrangement is not free to turn either -- `tripod`
# pins u_1 to +z and the mirror plane of the other two to azimuth 0 -- so the
# whole of the camera is these two numbers.
#
# Read them in the PAGE basis, where roll is 0 and so world +z always projects
# straight up the page. A world direction w lands at (w . SCREEN_RIGHT,
# w . SCREEN_UP), and for the two families that matter that comes out as
#
#     +z                                 ->  (0, cos elev)
#     horizontal at page-angle th        ->  (cos th, -sin th sin elev)
#
# th measured from page-right, and NEGATIVE page-y meaning coming OUT of the page
# at the reader. So at 90 deg the three pushes draw as a coordinate triad -- one
# straight up, one lying nearly in the page, one foreshortened towards the eye --
# which is the whole point of the pose and is what fixes both numbers below.
#
# AZIM is the offset between the eye and the arrangement's own mirror plane, and
# it is squeezed from both sides. At 0 the panel is left-right symmetric: the two
# horizontals come out at th = 45 and 135, mirror images of each other, the same
# page length, and NEITHER of them reads as "across the page" against the other's
# "towards us". Opening it to 12 deg splits them to 33 and 123, page lengths
# 0.859 and 0.616 against the vertical's 0.940 -- the ordinary way a triad is
# drawn. Past about 13 the 110 deg panel breaks: its receding push is already
# 81.6 deg from the eye at 12 and goes over the limb at 17.
#
# ELEV trades the same two things again. Lower opens the AZIM budget -- the limb
# is reached at 15 deg of offset from elev 15 and at only 7 from elev 30 -- and
# flattens the ball towards a disc, since the horizontal plane of directions
# projects to an ellipse of semi-minor axis sin(elev). 20 deg keeps that ellipse
# open enough to read as a horizon and still affords the 12 deg of offset.
#
# The brief this revision was written to asked for elev 30 with the horizontals
# at th = 20 and 110, which is a better triad than this one -- page lengths 0.955
# and 0.581. It is not reachable. That camera stands 25 deg off the mirror plane,
# and measured over the other two panels it puts the 50 deg panel's near push
# 12.16 deg off the view axis, where an arrow has 0.211 of its length left on the
# page, and the 110 deg panel's far push 96.28 deg from the eye, which is 6 deg
# ROUND THE BACK of the ball. No elevation rescues the second of those: at 25 deg
# of offset the eye cannot come within 23.55 deg of the 110 deg tripod's 3-fold
# axis, and its pushes stand 71.06 deg off that axis, so one of them is always
# over the horizon. `main` prints every ray's off-axis angle and page length on
# every run; the rule they are checked against is 20 deg clear of the view axis
# and 84 deg clear of the far side.
ELEV, AZIM = 20.0, -12.0
EYE = np.array([np.cos(np.radians(ELEV)) * np.cos(np.radians(AZIM)),
                np.cos(np.radians(ELEV)) * np.sin(np.radians(AZIM)),
                np.sin(np.radians(ELEV))])
# screen up is +z with the camera's share of it taken out, which is mplot3d's own
# convention at roll 0; screen right follows. Only the light is built on these
SCREEN_UP = np.array([0.0, 0.0, 1.0]) - (np.array([0.0, 0.0, 1.0]) @ EYE) * EYE
SCREEN_UP /= np.linalg.norm(SCREEN_UP)
SCREEN_RIGHT = np.cross(SCREEN_UP, EYE)
# one light, fixed to the CAMERA and not to the world, so the ball is lit
# identically in all three panels however the pushes are turned -- otherwise the
# reader has a second thing changing across the row. Over the reader's left
# shoulder, as everywhere else in these figures
LIGHT = -0.42 * SCREEN_RIGHT + 0.40 * SCREEN_UP + 0.82 * EYE
LIGHT /= np.linalg.norm(LIGHT)
# how much of the colour the shading is allowed to move, for the bare shell and
# for the painted patch. The patch is modelled with the LESS of the two: the
# field is the whole message there, and a Lambert term deep enough to round the
# ball would also be deep enough to be mistaken for the field falling off
SHELL_LIGHT, PAINT_LIGHT = (0.94, 0.06), (0.90, 0.13)
# the transparency itself, and it is not symmetric. The bare shell is what makes
# the ball a BALL rather than a disc -- the far half of the equator and the far
# half of every arc are seen through it -- so it is nearly clear. The paint is
# OPAQUE, for two reasons: the reader has to match its colour against a key, and
# a colour read through a third of a grey ball is not the colour on the key; and
# a translucent mesh leaves a hairline of paper at every cell edge, which on a
# saturated field draws the tessellation itself across the picture. Opaque cells
# can be given their own colour as an EDGE as well as a fill, which closes those
# seams exactly; translucent ones cannot, because the two would blend twice
SHELL_A, PAINT_A = 0.13, 1.0
# the shell is WHITE and all but clear, on the human's instruction, so that the
# three radii and the angles between them can be read THROUGH it -- they are
# inside the ball and there is no honest way to draw them that does not go
# through its near half. The Lambert term is nearly all ambient for the same
# reason: shading is what made the old grey shell read as a solid, and a solid
# is exactly what this is not. Two things have to be given back once the fill
# stops carrying the form. The LIMB, drawn in `ball` -- the great circle square
# to the eye, which is the ball's own outline and the only thing left saying
# where it ends, since a white fill on near-white paper has no edge of its own.
# And the equator, which was already there and now does more work than it did

N = 1_000_000                                                  # directions in the sweep


def tripod(deg):
    """Three unit vectors, pairwise `deg` apart, with ONE OF THEM STRAIGHT UP.

    The arrangement is the same regular tripod at every angle -- half-cone angle
    arccos(sqrt((cos t + 0.5)/1.5)) about its own 3-fold axis -- but it is posed
    so that u_1 = +z whatever `deg` is, which is what connects the picture to the
    scene it is about. One generator there is not a chock: it is THE FLOOR, and
    the floor's push does not move when the chocks are re-cut. So the row is a
    one-parameter family with the vertical held fixed: the two others tilt UP
    towards it as the angle narrows -- shims driven under an edge -- and DOWN
    below the horizontal as it opens -- clamps hooked over an edge. At 90 deg
    they are exactly horizontal and the three are the coordinate axes.
    (Posed the other way -- symmetric about +z, which is what this drew before --
    NONE of the three is vertical at any angle, and panel (a) reads as two rays
    up-left and up-right with one straight down. Same cone, wrong pose.)

    Placing them needs one number. With u_1 at the pole the two arcs u_1 -> u_2
    and u_1 -> u_3 are MERIDIANS, so the angle between them is just the
    difference in azimuth -- and that angle is the spherical triangle's own
    corner angle A at u_1, the same A `covers` integrates:

        cos A = cos t / (1 + cos t)

    so the two stand at polar angle t and at azimuths +-A/2. Every pairwise
    angle is then exactly t: u_1 . u_2 = cos t by the polar angle, and
    u_2 . u_3 = sin^2 t cos A + cos^2 t = cos t by that identity. Checked in
    `main`, at every angle drawn.

    The cosine is snapped before it is used, which is not fussiness -- it is a
    bug from an earlier pass, kept fixed. cos(120 deg) comes back as
    -0.4999999999999998,
    and carried through it makes cos A = -1.0000000000000004, which is a NaN out
    of arccos rather than the flat degenerate tripod that 120 deg really is.
    """
    g = np.round(np.cos(np.radians(deg)), 12)
    half = np.arccos(np.clip(g / (1 + g), -1.0, 1.0)) / 2
    s, c = np.sqrt(max(0.0, 1 - g * g)), g
    return np.array([[0.0, 0.0, 1.0],
                     [s * np.cos(half), s * np.sin(half), c],
                     [s * np.cos(half), -s * np.sin(half), c]])


def sweep(n):
    """A near-uniform set of directions, for measuring areas by counting.

    The Fibonacci spiral, as everywhere else in this repo, and the choice is
    load-bearing rather than stylistic. The obvious alternative is to count the
    cells of the ball this figure is DRAWN on, and it is worth knowing exactly
    how that fails, because it fails HERE and not in dimension.py.

    A lat-lon mesh has cells equal in longitude and latitude and so with areas
    going as sin(theta), which over-weights the poles -- and in this pose the
    pole is not somewhere else, it is exactly where u_1 stands, a VERTEX of all
    three patches. Counting a 96 x 48 lat-lon mesh against the closed form, and
    against the geodesic mesh this figure is drawn on:

          t      exact     sweep   lat-lon    geodesic
         90     12.500    12.500    12.500      12.480
         50      2.902     2.903     4.818       2.773
        110     25.550    25.549    22.526      25.503

    Read the lat-lon column across. It is EXACT at 90 deg -- that octant is cut
    by the equator and by two meridians, so the over-weighting is symmetric
    across its own boundaries and cancels, which is the same reason
    dimension_ball.png gets away with counting quads. One panel over, at 50 deg,
    the same mesh reports two thirds more patch than there is, because that patch
    hugs the pole where the cells are slivers; at 110 deg it reports a tenth too
    little. A method that is exact on the panel you check it on and 66 % out on
    the panel beside it is worse than one that is uniformly rough.

    The geodesic mesh is far better behaved -- within half a per cent everywhere
    -- and is still not used for measuring: a drawing mesh answers to the drawing.
    """
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r, t = np.sqrt(1 - z * z), np.pi * (1 + 5 ** 0.5) * i
    return np.stack([r * np.cos(t), r * np.sin(t), z], axis=1)


def decompose(U, V):
    """The unique c with v = sum c_i u_i, for every direction at once.

    Three independent generators span space, so there is exactly one
    decomposition and no choice to make -- which is what lets `capped` be a
    formula rather than a search. (`strongest` in dimension.py has to take a
    minimum over subsets because it is written for generator sets that may be
    dependent; here they never are, and the two agree wherever both apply.)
    """
    return np.linalg.solve(np.asarray(U, float).T, np.atleast_2d(V).T).T


def capped(U, V, cap=F):
    """Which directions the three can push in, and the most they can put there.

    A contact can only push, so `v` is reachable exactly when every c_i >= 0.
    Scaling the combination scales every coefficient together, so the cap binds
    on the largest and the answer is cap / max_i c_i. Outside the cone there is
    no answer at all and this returns 0, which is a THIRD state and not a small
    magnitude -- the figure keeps it distinct by leaving those directions
    unpainted rather than by giving them the bottom of the ramp.
    """
    c = decompose(U, V)
    inside = (c >= -1e-12).all(axis=1)
    return inside, np.where(inside, cap / np.maximum(c.max(axis=1), 1e-300), 0.0)


def covers(deg):
    """The exact share of the ball the cone takes, as a percentage.

    The closed form for a regular spherical triangle of side t: the spherical law
    of cosines gives cos A = cos t / (1 + cos t) for its angle, and its area is
    the excess 3A - pi. At 90 deg that is exactly pi/2, an eighth of the sphere.
    Kept beside the swept figure so the sweep is checked rather than trusted.
    """
    g = np.cos(np.radians(deg))
    return (3 * np.arccos(g / (1 + g)) - np.pi) / (4 * np.pi) * 100


def weakest(deg, cap=F):
    """The floor of m over the patch, in closed form, and where it sits.

    m = cap / max_i c_i, so the floor is at the direction that makes ONE
    coefficient as large as possible: maximise c_1 over {c >= 0, c'Gc = 1}. The
    unconstrained maximum is sqrt((G^-1)_11), at c proportional to G^-1 e_1 --
    but those weights are only non-negative, and so only name a direction the
    three can actually reach, when the pairwise cosine is negative. For a
    non-negative cosine the constrained maximum falls back on c = e_1, the push
    direction itself, and the answer is exactly cap.

    That switch IS the threshold and it happens at cos t = 0. It is why narrowing
    below 90 deg cannot buy a better floor: below 90 the floor is pinned at one
    support's strength by the push directions themselves, and no angle moves it.
    """
    g = np.cos(np.radians(deg))
    return cap if g >= 0 else cap * np.sqrt((1 + 2 * g) * (1 - g) / (1 + g))


def strongest(deg, cap=F):
    """The ceiling of m over the patch, in closed form.

    m = cap / max_i c_i, so the ceiling is at the direction that keeps every
    coefficient small: minimise max_i c_i over {c >= 0, |sum c_i u_i| = 1}, which
    is the same as MAXIMISING |sum c_i u_i| over the box 0 <= c_i <= 1. A convex
    function over a box takes its maximum at a vertex, so there are only four to
    try, and by symmetry only three distinct ones:

        c = (1,1,1)   all three pushing   |sum| = sqrt(3 + 6 cos t)
        c = (1,1,0)   two of them          |sum| = sqrt(2 + 2 cos t)
        c = (1,0,0)   one alone            |sum| = 1

    The first wins while cos t > -1/4, i.e. below 104.48 deg, and the second past
    it: spread wide enough, a pair pulls harder across its own bisector than all
    three do down the middle, because the third one is then more than 90 deg from
    that middle and its neighbours have to cancel each other. The third entry is
    what keeps the ceiling from ever falling below cap -- a single support can
    always push along itself.
    """
    g = np.cos(np.radians(deg))
    return cap * max(np.sqrt(3 + 6 * g), np.sqrt(2 + 2 * g), 1.0)


def dual(U):
    """The reciprocal basis: the vectors that READ OFF the coefficients.

    u^i is defined by u^i . u_j = delta_ij, so c_i = v . u^i for any v -- and
    u^1 = (u_2 x u_3) / det, which is exactly the mechanism direction. Two facts
    follow immediately and both are used below:

      * unit(u^i) is the direction perpendicular to the OTHER TWO pushes, the one
        neither of them can contribute anything along;
      * the level set c_i = 1, where the cap on support i binds at exactly F, is
        the plane v . u^i = 1, which meets the ball in a circle centred on that
        direction at angular radius arccos(1 / |u^i|).

    So the boundary between "takes a full support's worth" and "does not" is not
    fitted to a picture: it is three small circles, drawn from the algebra.
    """
    # rows are u^i: W U^T = I is exactly W_i . u_j = delta_ij, so W = inv(U^T)
    # and NOT its transpose. Written the wrong way round it still returns three
    # vectors of about the right size, which is why the error showed up as a
    # plausible curve in the wrong place rather than as an exception
    return np.linalg.inv(np.asarray(U, float).T)


def mechanism(U):
    """d_i = unit(u_j x u_k) for each i, with the coefficients it decomposes into.

    The direction perpendicular to two of the three pushes: neither of them can
    put anything along it, so only the third is left. Its sign is taken towards
    u_i, which is the half of the line the third support can actually push in.
    """
    out = []
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        d = np.cross(U[j], U[k])
        d = d / np.linalg.norm(d) * np.sign(d @ U[i])
        out.append((d, decompose(U, d)[0]))
    return out


# ------------------------------------------------------------------- drawing

ICO = None                                                     # built once, on first use


def shell(k=SUBDIV):
    """The unit sphere as triangles: an icosahedron subdivided k times.

    dimension.py's ball is tessellated in longitude and latitude, and that mesh
    is exactly wrong here. Its cells shrink to slivers at the pole -- and in this
    pose the pole is where u_1 stands, the vertical push, which is a CORNER OF
    ALL THREE PATCHES and the one mark the whole figure is posed around. A
    lat-lon mesh would leave a bullseye of converging seams exactly there, on top
    of the mark the reader is being asked to look at, and would leave it in all
    three panels at once. A geodesic mesh has no pole to have that problem at:
    every triangle is within a few per cent of every other in size and shape, so
    the field's staircase is the same width all over the patch and the corner is
    no worse than the middle.

    k = 5 gives 20480 triangles at about 2 deg a side. That is the resolution the
    F CURVE needs to hide behind: the curve is drawn exactly, over the paint,
    with a casing wide enough to cover half a cell, and half a cell is 1 deg.

    ONE mesh drawn as ONE collection with a colour per triangle is what lets a
    translucent ball sort against itself: handed two collections mplot3d gives
    each a single averaged depth, and the near half of one can end up behind the
    far half of the other.

    These triangles are for DRAWING only. Nothing is ever counted on them --
    `sweep` does the measuring, on a set with no mesh in it at all.
    """
    global ICO
    if ICO is not None:
        return ICO
    t = (1 + 5 ** 0.5) / 2
    V = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
                  [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    faces = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                      [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                      [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                      [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    T = V[faces] / np.linalg.norm(V[faces], axis=2, keepdims=True)
    for _ in range(k):
        a, b, c = T[:, 0], T[:, 1], T[:, 2]
        # each edge midpoint is a function of the two ends alone, so the two
        # triangles sharing an edge compute the same point and the mesh cannot
        # crack. Normalised at every level rather than once at the end, which is
        # what keeps the cells even instead of bunching them at the face centres
        ab, bc, ca = [(p + q) / np.linalg.norm(p + q, axis=1, keepdims=True)
                      for p, q in ((a, b), (b, c), (c, a))]
        T = np.concatenate([np.stack([a, ab, ca], 1), np.stack([ab, b, bc], 1),
                            np.stack([ca, bc, c], 1), np.stack([ab, bc, ca], 1)])
    mid = T.mean(axis=1)
    ICO = (T, mid / np.linalg.norm(mid, axis=1, keepdims=True))
    return ICO


def shaded(normals, rgb, amb, key):
    """A colour per face: how squarely each one meets the light.

    Lambert, clipped at zero -- a face turned away from the light receives
    nothing from it, not a negative amount -- so what is left on an unlit face is
    the ambient. On a sphere the normal IS the direction, so the limb darkens by
    itself and nothing extra is needed to round the edge off.
    """
    n = np.atleast_2d(np.asarray(normals, float))
    f = amb + key * np.clip(n @ LIGHT, 0, None)
    return np.clip(f[:, None] * np.atleast_2d(rgb), 0, 1)


def ramp(m):
    """The field as colours: two ramps meeting at F, with a hard break between.

    Read as two states and not as one number. The break is the whole question the
    figure asks, so it is a jump in lightness as well as in hue -- hot red just
    below F, near white just above it. A reader who takes in nothing else takes
    in where that edge runs.
    """
    m = np.asarray(m, float)
    out = np.zeros(m.shape + (3,))
    lo = m < F
    out[lo] = DEAD(np.clip(m[lo] / F, 0, 1))[..., :3]
    out[~lo] = LIVE(np.clip((m[~lo] - F) / (VMAX - F), 0, 1))[..., :3]
    return out


def onpage(d, floor=SEEN_FLOOR):
    """What fraction of a length along `d` survives the projection to the page.

    A radial arrow on a ball has no foreshortening cue on it to contradict -- it
    is one straight mark pointing away from the middle of a disc -- so giving it
    the 3-D length whose PAGE length is the one wanted costs nothing and buys
    everything. Uncorrected, the nine pushes keep between 0.457 and 0.989 of
    their length -- `pose` prints all nine every run -- and the reader is handed
    nine arrows of nine different lengths all standing for the same identical F.
    Worse here than a spread of lengths would be anywhere else: the three pushes
    of one panel are drawn at three angles to the eye ON PURPOSE, so a reader who
    took length for magnitude would read the pose as the physics.

    Only the LENGTH is treated this way. The tail still stands at STAND on the
    shell, because that is a position, and positions are the one thing this
    figure projects honestly.
    """
    return max(float(np.sqrt(max(1.0 - (np.asarray(d, float) @ EYE) ** 2, 0.0))), floor)


class Arrow3D(FancyArrowPatch):
    """A flat arrow with both ends pinned to points in space.

    mplot3d has no arrow of its own, and `quiver` builds its head out of two
    barbs in a plane it picks from the shaft alone -- so the more nearly an arrow
    points at the reader the more nearly its head is projected edge on, and it
    thins away to nothing. The nine pushes here stand between 27.17 and 81.59 deg
    off the camera's axis on purpose, that spread being the whole content of the
    pose, so nine heads built that way would come out at nine different weights
    and the reader would be handed a difference where there is none. A
    FancyArrowPatch re-projects both ends and draws its head on the PAGE, so the
    head is the same head whichever way the arrow points.
    """

    def __init__(self, tail, tip, **kw):
        super().__init__((0, 0), (0, 0), mutation_scale=13, shrinkA=0.0, shrinkB=0.0, **kw)
        self.ends = np.array([tail, tip], float)

    def do_3d_projection(self, renderer=None):
        x, y, _ = proj3d.proj_transform(*self.ends.T, self.axes.M)
        self.set_positions((x[0], y[0]), (x[1], y[1]))
        return 0.0                                             # the axes is told not to sort


def split(ax, P, near_kw, far_kw, lift=1.006):
    """A curve on the ball, cut at the silhouette and drawn twice.

    mplot3d gives a whole line ONE depth and would put all of it in front, so the
    far half of a great circle reads as a loop floating over the ball rather than
    as a curve lying on it. Cutting it at the silhouette and drawing the two
    halves at different weights is the fix dimension.py's ball uses, and it is
    also the only cue that says the far half is round the back.
    """
    P = np.asarray(P, float)
    if len(P) < 2:
        return
    for near, kw in ((False, far_kw), (True, near_kw)):
        keep = (P @ EYE > 0) == near
        ax.plot(*np.where(keep[:, None], lift * P, np.nan).T, **kw)


def arc(a, b, n=361):
    """The great-circle arc between two directions, walked evenly.

    An edge of the patch: the pair {a, b} reaches exactly this curve and the cone
    of all three closes on the three of them.
    """
    t = np.linspace(0.0, 1.0, n)[:, None]
    g = (1 - t) * np.asarray(a, float) + t * np.asarray(b, float)
    return g / np.linalg.norm(g, axis=1, keepdims=True)


def contour(U, i, n=721, least=2.0):
    """The curve where support i is working at exactly its cap, inside the cone.

    c_i = 1 is the plane v . u^i = 1, so on the ball it is the circle centred at
    unit(u^i) -- the mechanism direction -- at angular radius arccos(1/|u^i|).
    Every one of the three passes through u_i itself, since u_i decomposes as
    e_i; what changes with the angle is whether anything ELSE of the circle lies
    inside the cone.

    Below 90 deg nothing does, and the whole below-F region is that single
    touching point, which is why this hands back nothing there rather than a
    hairline: runs shorter than `least` degrees are dropped as the touch they
    are. Past 90 deg the circle cuts a real lens out of the patch and this is its
    boundary, exact rather than traced off a raster.

    Two conditions and not one. c_i = 1 holds all the way round the circle, so a
    stretch of it can perfectly well run through the INSIDE of the region where
    another support is over ITS cap -- where c_j > 1 the field is below F on both
    sides of the curve and the curve is not a boundary of anything. Dropped, the
    three circles come back as one long arc straight across the 110 deg patch
    instead of as the three symmetric lenses that are really there.
    """
    W = dual(U)[i]
    r = np.linalg.norm(W)
    if r <= 1.0 + 1e-12:                                       # the plane misses the ball
        return []
    c, b = W / r, np.arccos(1 / r)
    e1 = np.cross(c, [0.0, 0.0, 1.0] if abs(c[2]) < 0.9 else [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(c, e1)
    th = np.linspace(0, 2 * np.pi, n)[:, None]
    P = np.cos(b) * c + np.sin(b) * (np.cos(th) * e1 + np.sin(th) * e2)
    C = decompose(U, P)
    ok = (C >= -1e-12).all(axis=1) & (C <= 1.0 + 1e-9).all(axis=1)
    out, k, step = [], 0, 360.0 / (n - 1)
    while k < len(ok):                                         # the runs that are inside
        if ok[k]:
            j = k
            while j + 1 < len(ok) and ok[j + 1]:
                j += 1
            if (j - k) * step >= least:
                out.append(P[k:j + 1])
            k = j
        k += 1
    return out


def ball(ax, U):
    """One panel: the ball of directions, painted with what the three can push.

    Everything drawn here is a consequence of the same three vectors and nothing
    is placed by hand: the paint is `capped` on the quad centres, the rim is the
    three great-circle arcs between them, the F curve is the three small circles
    from `dual`, and the rings are the mechanism directions.
    """
    cells, mid = shell()
    inside, m = capped(U, mid)
    # TWO collections, and they are split by the one thing that has to differ
    # between them: the line width.
    #
    # Agg rasterises every polygon on its own, so two that share an edge leave a
    # hairline of whatever is behind them between the two -- 20480 cells' worth of
    # hairline, which prints the tessellation right across the ball. Giving each
    # cell an outline in its OWN colour closes that, but only where the fill is
    # opaque: over the translucent shell the outline blends a second time and
    # draws the same mesh a shade darker instead. So the painted cells want a line
    # width and the bare ones want none. (Nor does turning antialiasing off help:
    # with it off Agg hands the boundary pixels to BOTH polygons and the
    # translucent shell comes out with a dark lattice over all of it.)
    #
    # A per-face `linewidths` array cannot do it. Poly3DCollection re-orders its
    # faces by depth on every draw and permutes the face and edge COLOURS to
    # match, but the line widths are the base Collection's and are left where they
    # were -- so the widths land on whichever cells happen to be at those depths,
    # and the ball comes out sprinkled with a few hundred outlined triangles in
    # both the bare and the painted parts. That is what this split is for.
    shell_kw = dict(zsort="average", edgecolors="none", linewidths=0.0)
    ax.add_collection3d(Poly3DCollection(
        cells[~inside], zorder=1.0, **shell_kw,
        facecolors=np.hstack([shaded(mid[~inside], matplotlib.colors.to_rgb(SHELL),
                                     *SHELL_LIGHT),
                              np.full((int((~inside).sum()), 1), SHELL_A)])))
    paint = np.hstack([shaded(mid[inside], ramp(m[inside]), *PAINT_LIGHT),
                       np.full((int(inside.sum()), 1), PAINT_A)])
    # splitting them costs the one thing a single collection was bought for --
    # mplot3d gives a whole collection one averaged depth, so these two cannot
    # interleave -- and here that costs nothing, because the painted set is
    # entirely on the near side in all three panels and so belongs in front of the
    # whole shell anyway. That is not luck: the patch is the cone ON the three
    # pushes, so for any v in it with |v| = 1, v . EYE = sum c_i (u_i . EYE) and
    # sum c_i >= 1, which puts the whole patch no further from the eye than its
    # furthest PUSH. Keeping every push inside 84 deg -- which `pose` checks --
    # therefore keeps every painted cell on the near side, and the widest panel is
    # the one that could break it: at 110 deg the receding push stands 81.59 deg
    # from the eye and the furthest painted CELL CENTRE is 80.36 deg, nine and a
    # half degrees clear of the limb. Anyone moving the camera trips this first
    assert (mid[inside] @ EYE > 0).all(), "the painted patch has reached the far side"
    ax.add_collection3d(Poly3DCollection(cells[inside], facecolors=paint,
                                         edgecolors=paint, linewidths=0.4,
                                         zsort="average", zorder=1.1))
    # the LIMB: the great circle square to the eye, which is where the ball turns
    # away and so is its outline on the page. A grey shell did not need it -- the
    # fill ended somewhere and that was the edge. A white one at 13 % on
    # near-white paper has no edge at all, and without this the ball reads as a
    # patch and some arrows floating near it
    th = np.linspace(0, 2 * np.pi, 481)[:, None]
    ax.plot(*(np.cos(th) * SCREEN_RIGHT + np.sin(th) * SCREEN_UP).T,
            color=MUTED, lw=1.1, alpha=0.75, zorder=2.9)
    interior(ax, U, np.degrees(np.arccos(np.clip(U[0] @ U[1], -1, 1))))
    # the world equator, which in this pose is the HORIZON: +z is the floor's own
    # push, so z = 0 is the set of horizontal directions and the panel's ellipse
    # is the ground plane seen from 20 deg above it, semi-minor axis sin(20) =
    # 0.342 of a radius. It is what says the ball is round and not a disc -- and
    # in this pose it also says which way is up, which is why the 90 deg panel's
    # two horizontal pushes land ON it and the 110 deg panel's two land under it.
    # It costs nothing: the same circle in all three panels, carrying no part of
    # the argument. (In the pose this replaced the camera was 15 deg off the pole,
    # the equator projected to a sliver hugging the silhouette, and it meant
    # nothing at all -- there was no up.)
    th = np.linspace(0, 2 * np.pi, 361)
    split(ax, np.stack([np.cos(th), np.sin(th), 0 * th], axis=1),
          dict(color=MUTED, lw=0.9, alpha=0.65, zorder=2.0),
          dict(color=MUTED, lw=0.9, alpha=0.22, zorder=0.4), lift=1.02)
    # the rim of the patch. For a PAIR of pushes this arc is the whole of what
    # they reach, so drawing all three says the patch is three pair-cones closed
    # on each other -- and it is where the field jumps from something to nothing,
    # which a quad edge alone renders as a staircase
    for a, b in itertools.combinations(np.asarray(U, float), 2):
        split(ax, arc(a, b),
              dict(color=ORANGE, lw=3.2, alpha=1.0, solid_capstyle="round", zorder=2.6),
              dict(color=ORANGE, lw=2.2, alpha=0.40, solid_capstyle="round", zorder=0.42))
    # the F curve: where a support is working at exactly its cap. Cased in paper
    # so it survives a hot red field on one side and a near-white one on the
    # other -- one colour cannot cross both. Thin casing on purpose: the curve
    # runs through the palest part of the field and a fat white line there would
    # swallow the very thing it is drawn to point out
    for i in range(3):
        for piece in contour(U, i):
            split(ax, piece,
                  dict(color=PAPER, lw=3.4, alpha=0.95, solid_capstyle="round", zorder=2.7),
                  dict(color=PAPER, lw=2.4, alpha=0.30, solid_capstyle="round", zorder=0.44))
            split(ax, piece,
                  dict(color=INK, lw=1.5, alpha=1.0, solid_capstyle="round", zorder=2.75),
                  dict(color=INK, lw=1.1, alpha=0.35, solid_capstyle="round", zorder=0.45))
    # the three pushes themselves, standing out of the ball along their own
    # directions. Same colour, same page length, same weight in every panel, so
    # what changes across the row is only where they stand
    for u in np.asarray(U, float):
        near = u @ EYE > 0
        tail = STAND * u
        ax.add_artist(Arrow3D(tail, tail + UNIT_LN / onpage(u) * u, color=ORANGE,
                              lw=3.2, alpha=1.0 if near else 0.40,
                              zorder=3.6 if near else 0.48,
                              arrowstyle="-|>,head_width=.20,head_length=.42"))
        ax.scatter(*(1.02 * u), s=86, c=ORANGE, edgecolors="white", linewidths=1.2,
                   depthshade=False, zorder=3.5)
    # the mechanism directions, one per pair. A hollow ring and nothing more: it
    # is a single point, it carries no magnitude, and where it LIES is the entire
    # content -- outside the patch at 50 deg, exactly ON the push at 90, well
    # inside it at 110. Drawn LAST and wide enough to encircle a push dot rather
    # than to hide under one, because the 90 deg panel is the one that has to
    # show the two marks landing on each other
    #
    # Measured on the page, in ball radii, ring to its OWN push's dot: 0.694 and
    # 0.763 at 50 deg for the two near ones, 0.012 to 0.019 at 90 deg -- which is
    # only the 1.04 against 1.02 stand-off, i.e. exactly on top -- and 0.278 to
    # 0.458 at 110 deg, on the inward side. That walk is the threshold, drawn.
    #
    # The pose this replaced put every ring on the same page RAY as its own push,
    # because the camera stood in a mirror plane of the arrangement, and at 50 deg
    # the arrow was drawn straight through the ring. Off that plane by 12 deg the
    # rings come clear: the nearest any of them now comes to any arrowhead is
    # 0.563 of a radius. The paper casing is kept anyway, because at 110 deg the
    # rings sit on a saturated red field
    for d, _ in mechanism(np.asarray(U, float)):
        near = d @ EYE > 0
        for s, col, w in ((250, PAPER, 2.6), (190, INK, 1.6)):
            ax.scatter(*(1.04 * d), s=s, facecolor="none", edgecolor=col,
                       linewidths=w if near else w * 0.7,
                       alpha=(1.0 if col == INK else 0.9) if near else 0.32,
                       depthshade=False, zorder=3.9 if near else 0.46)
    # orthographic, and that is not a preference. mplot3d's default is a
    # PERSPECTIVE projection at focal length 1, strong enough that the near side
    # of a ball of radius 1 draws visibly larger than the far side -- and this
    # figure asks the reader to compare the AREA of three patches which, in this
    # pose, sit at three different distances from the view axis (their centroids
    # stand 29.2, 54.7 and 71.1 deg off the vertical push, so they cannot all face
    # the camera). Under perspective the reader would be comparing three areas
    # scaled by three different factors he was given no way to know about
    ax.set_proj_type("ortho")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    # the same cube, moved: world +z is page-up at roll 0, so raising the z
    # window lowers the ball in the frame and makes room over it for the arrow
    # that now stands out of the top of every panel. See LOWER
    ax.set_zlim(-1.5 + LOWER, 1.5 + LOWER)
    ax.set_box_aspect((1, 1, 1), zoom=ZOOM)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_axis_off()


def interior(ax, U, deg):
    """The three radii, the angle between each pair, and the centre they meet at.

    This is what the shell was made clear for. The three pushes are directions,
    so on the ball they are POINTS, and what a reader wants to see is the thing
    those points are directions OF: three radii out of one centre, with the
    angle standing between them. Drawn inside the ball because that is where
    they are -- at `zorder` below the shell, so the near half veils them by the
    13 % it is worth and the depth still reads.

    The angle marks are three arcs of radius `MARK`, one per pair, joined end to
    end at the radii. They are a scaled copy of the patch's own rim, which is
    not a coincidence and is the point: the solid angle at the centre and the
    patch on the surface are the same object seen at two radii. Only one is
    labelled -- all three pairs are equal by construction, `tripod` asserts it
    to nine decimals, and three copies of the same number is noise. The one that
    gets it is the arc whose midpoint projects furthest from the other two, so
    the text never lands on a radius.
    """
    # DASHED, and drawn over everything rather than under it. Honest depth would
    # hide them: for any point of the disc the sphere's near surface is at
    # sqrt(1 - r^2) and every interior line is nearer the centre than that, so
    # the painted patch occludes all three radii wherever it covers them -- which
    # in (a) and (c) is the whole middle of the ball, and drawing them properly
    # buried showed nothing at all. A dash is the ordinary way to say "this is
    # behind what it crosses", it costs no accuracy, and it is the only reading
    # that survives an opaque patch. Thin, so the patch still wins the eye
    for u in U:
        ax.plot(*np.array([[0.0, 0.0, 0.0], u]).T, color=INK, lw=1.3, alpha=0.80,
                dashes=(4.5, 3.0), solid_capstyle="butt", zorder=3.0)
    ax.scatter([0.0], [0.0], [0.0], s=30, c=INK, depthshade=False, zorder=3.05)
    marks = [(i, j, MARK * arc(U[i], U[j])) for i, j in ((0, 1), (1, 2), (2, 0))]
    for _, _, P in marks:
        ax.plot(*P.T, color=INK, lw=1.3, alpha=0.80, dashes=(4.5, 3.0), zorder=3.02)
    page = np.array([SCREEN_RIGHT, SCREEN_UP])
    mids = np.array([P[len(P) // 2] for _, _, P in marks])
    flat = mids @ page.T
    apart = [min(np.linalg.norm(flat[i] - flat[j]) for j in range(3) if j != i)
             for i in range(3)]
    at = mids[int(np.argmax(apart))]
    ax.text(*(at * LABEL_OUT), f"{deg:.0f}\u00b0", color=INK, fontsize=13,
            ha="center", va="center", zorder=3.06,
            bbox=dict(boxstyle="round,pad=0.16", facecolor=PAPER, edgecolor="none",
                      alpha=0.85))


def onpaper(m):
    """The colour a reading of `m` actually lands on the page as.

    A swatch has to match the ball, and the ball's colours have been through two
    things the ramp knows nothing about: the Lambert term, and -- for the bare
    shell -- being seen at 28 % over near-white paper. Both are undone here
    rather than eyeballed, so the key follows the paint automatically if either
    the light or the transparency is ever changed. `m` of None means the shell.
    """
    if m is None:
        c = shaded(EYE, matplotlib.colors.to_rgb(SHELL), *SHELL_LIGHT)[0]
        return (1 - SHELL_A) * np.array(matplotlib.colors.to_rgb(PAPER)) + SHELL_A * c
    return shaded(EYE, ramp(np.array([m]))[0], *PAINT_LIGHT)[0]


def key(ax):
    """The scale: one ramp, broken at F, and the ticks that put numbers on it.

    Deliberately narrow. It is a legend and not a second figure, and the reading
    it has to support is "which side of F", which is a break rather than a value.
    """
    t = np.linspace(0, VMAX, 512)
    ax.imshow(shaded(EYE, ramp(t), *PAINT_LIGHT)[None, :, :], extent=(0, VMAX, 0, 1),
              origin="lower", aspect="auto")
    ax.plot([F, F], [0, 1], color=INK, lw=1.8)                 # the break itself
    ax.set_xlim(0, VMAX)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, F, 2 * F, VMAX])
    ax.set_xticklabels(["0", "F", "2F", f"{VMAX:.2f} F"], fontsize=9.5, color=MUTED)
    ax.tick_params(colors=MUTED, length=3, pad=2)
    for s in ax.spines.values():
        s.set_color(MUTED)
    ax.text(0.5, 1.9, "m(v) = F / max$_i$ c$_i$", transform=ax.transAxes, ha="center",
            va="bottom", color=INK, fontsize=11)


def band(fig, rows, marks):
    """The bottom strip: what the colours mean on the right, what the marks mean
    on the left, with the scale itself between them.

    Three groups across the width rather than a stack down the page, because the
    figure is three times as wide as it is tall and the space beside a narrow
    colour bar is the only space left. A stack put the words under the bar, where
    they ran into the panel captions above and off the bottom below.
    """
    for k, (x, y, sw, line) in enumerate(rows):
        # the swatch is squared against the FIGURE's aspect, not against the page,
        # or a 13.4 x 6.4 in canvas turns every square into a letterbox
        w, h = 0.0092, 0.0092 * fig.get_figwidth() / fig.get_figheight()
        fig.add_artist(plt.Rectangle((x, y - h / 2), w, h, facecolor=sw,
                                     edgecolor=MUTED, lw=0.5, transform=fig.transFigure))
        fig.text(x + w + 0.008, y, line, ha="left", va="center", color=MUTED,
                 fontsize=9.6)
    x, ya, yb, wide = marks
    # the two marks are drawn with the very artists the panels use, not with
    # look-alike glyphs, so a change of arrowhead or ring size cannot leave the
    # key describing the previous version of the figure
    fig.add_artist(FancyArrowPatch((x, ya), (x + wide, ya), transform=fig.transFigure,
                                   arrowstyle="-|>,head_width=.20,head_length=.42",
                                   mutation_scale=13, color=ORANGE, lw=3.2,
                                   shrinkA=0.0, shrinkB=0.0))
    fig.add_artist(matplotlib.lines.Line2D([x + wide / 2], [yb], marker="o", ls="none",
                                           markersize=np.sqrt(190), markerfacecolor="none",
                                           markeredgecolor=INK, markeredgewidth=1.6,
                                           transform=fig.transFigure))
    fig.text(x + wide + 0.010, ya, "the three pushes, each capped at F", ha="left",
             va="center", color=MUTED, fontsize=9.6)
    fig.text(x + wide + 0.010, yb, "u$_j$ × u$_k$ : square to two of the three, so only\n"
             "the third can push there at all", ha="left", va="center", color=MUTED,
             fontsize=9.6, linespacing=1.4)


def trim(out, pad=18):
    """Cut the saved image down to what was drawn on it.

    mplot3d rescales whatever framing box it is given to a fixed diagonal and
    then stretches the result to fill its axes rectangle, so the box has to stay
    a cube and a cube framed to the width leaves the page empty above and below.
    Cropping the raster afterwards is the one fix that cannot distort anything,
    because it moves no pixel that it keeps.
    """
    a = plt.imread(out)
    ink = (a[:, :, :3] < 0.96).any(axis=2)
    rows, cols = np.where(ink.any(axis=1))[0], np.where(ink.any(axis=0))[0]
    y0, y1 = max(rows.min() - pad, 0), min(rows.max() + 1 + pad, a.shape[0])
    x0, x1 = max(cols.min() - pad, 0), min(cols.max() + 1 + pad, a.shape[1])
    matplotlib.image.imsave(out, a[y0:y1, x0:x1])
    return a[y0:y1, x0:x1].shape


def numbers(deg, ins, m):
    """Everything the captions quote, measured, and checked against closed forms."""
    inpatch = m[ins]
    return dict(
        deg=deg,
        cover=100 * ins.mean(), cover_exact=covers(deg),
        lo=inpatch.min(), lo_exact=weakest(deg),
        hi=inpatch.max(), hi_exact=strongest(deg),
        share=100 * (inpatch >= F - 1e-12).mean(),
    )


def onpaper_xy(w):
    """Where a direction lands on the PAGE, in ball radii, and how long it draws.

    The page basis is the camera's, so this is the projection the panels are
    actually drawn with rather than a model of it -- `main` asserts the two agree
    by re-deriving the same numbers through mplot3d's own transform.
    """
    w = np.asarray(w, float)
    return np.array([w @ SCREEN_RIGHT, w @ SCREEN_UP])


def pose(cases=(90.0, 50.0, 110.0)):
    """The pose check the revision exists for, run on every render.

    Three things, and all three are numbers rather than opinions:

      * ONE PUSH IS VERTICAL in every panel. u_1 = +z exactly, so it projects to
        (0, cos elev) -- straight up the page, the same length in all three.
      * NO PUSH POINTS AT THE CAMERA. A ray within about 20 deg of the view axis
        has under a third of its length left on the page and reads as a dot.
      * NO PUSH IS ROUND THE BACK. Past 90 deg from the eye a push is behind the
        ball; the painted patch is bounded by the pushes, so a push over the limb
        also puts paint on the far side and trips `ball`'s own assert.
    """
    print("the pose: one push straight up in every panel, and where each one lands\n")
    print(f"{'t':>7s}{'push':>18s}{'polar':>9s}{'page x':>10s}{'page y':>10s}"
          f"{'page len':>11s}{'off-axis':>11s}")
    lo, hi = 180.0, 0.0
    for deg in cases:
        U = tripod(deg)
        assert np.allclose(U[0], [0, 0, 1]), "u_1 is not vertical"
        G = U @ U.T
        assert np.allclose(G[np.triu_indices(3, 1)], np.cos(np.radians(deg))), \
            "the pairwise angle is not t"
        for i, u in enumerate(U):
            x, y = onpaper_xy(u)
            off = np.degrees(np.arccos(np.clip(float(u @ EYE), -1, 1)))
            lo, hi = min(lo, off), max(hi, off)
            name = ("u_1  the floor", "u_2", "u_3")[i]
            print(f"{deg:6.0f}°{name:>18s}{np.degrees(np.arccos(u[2])):8.1f}°"
                  f"{x:+10.3f}{y:+10.3f}{np.hypot(x, y):11.3f}{off:10.2f}°")
    print(f"\nclosest any push comes to the view axis {lo:.2f}° (the rule is 20°), "
          f"furthest\n from it {hi:.2f}° (the rule is 84°, since 90° is the limb and the "
          f"patch hangs\n on the pushes)")
    assert lo >= 20.0 and hi <= 84.0, "a push is aimed at the camera or is round the back"


def main(out=figure_path('three_pushes.png')):
    global VMAX
    t0 = time.time()
    # the ramp's far end is the largest magnitude the figure contains, read off
    # the closed form so the key cannot clip the paint or leave slack above it
    VMAX = max(strongest(d) for d, _ in CASES)

    pose(tuple(d for d, _ in CASES))
    V = sweep(N)
    print()
    print(f"a uniform sweep of {N} directions -- NOT the ball's own quads, whose areas "
          f"go as sin(theta)\n")
    print(f"{'t':>7s}{'cone covers':>22s}{'weakest':>20s}{'strongest':>20s}"
          f"{'of the patch,':>16s}")
    print(f"{'':>7s}{'sweep':>11s}{'exact':>11s}{'sweep':>10s}{'exact':>10s}"
          f"{'sweep':>10s}{'exact':>10s}{'at or above F':>16s}")
    # the four rows the human's own run reports, so a regression shows up as a
    # number moving rather than as a figure looking different
    for deg in (50.0, 60.0, 90.0, 110.0):
        ins, m = capped(tripod(deg), V)
        n = numbers(deg, ins, m)
        print(f"{deg:6.0f}°{n['cover']:10.3f}%{n['cover_exact']:10.3f}%"
              f"{n['lo']:10.3f}{n['lo_exact']:10.3f}{n['hi']:10.3f}{n['hi_exact']:10.3f}"
              f"{n['share']:14.2f}%")
    print("\n(the swept floor sits a little ABOVE the exact one below 90 deg because the "
          "floor\n is AT the push directions there and no Fibonacci sample lands on one; "
          "the swept\n ceiling sits a little below for the same reason at the centroid)")

    # why the sweep and not a mesh, measured rather than asserted. A 96 x 48
    # lat-lon mesh -- dimension.py's own ball -- against the geodesic mesh this
    # figure draws on, both counted as CELLS, both against the closed form
    th = (np.arange(48) + 0.5) * np.pi / 48
    ph = (np.arange(96) + 0.5) * 2 * np.pi / 96
    T, P = np.meshgrid(th, ph, indexing="ij")
    LATLON = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P),
                       np.cos(T)], axis=-1).reshape(-1, 3)
    _, GEO = shell()
    print("\nwhy the sweep and not a mesh -- the same three cones, counted three ways:\n")
    print(f"{'t':>7s}{'exact':>10s}{'sweep':>10s}{'lat-lon cells':>16s}"
          f"{'geodesic cells':>16s}")
    for deg, _ in CASES:
        U = tripod(deg)
        print(f"{deg:6.0f}°{covers(deg):10.3f}{100 * capped(U, V)[0].mean():10.3f}"
              f"{100 * capped(U, LATLON)[0].mean():16.3f}"
              f"{100 * capped(U, GEO)[0].mean():16.3f}")
    print("\nthe lat-lon column is EXACT at 90 deg -- that octant is cut by the equator and"
          "\n by two meridians, so the sin(theta) over-weighting cancels across its own\n"
          " boundaries -- and two thirds too big one panel over, where the patch hugs the"
          "\n pole. Exact on the case you check and 66 % out on the one beside it")

    # the mechanism, and the one number that decides the threshold
    print(f"\nd_i = unit(u_j x u_k), the direction square to two of the three:\n")
    print(f"{'t':>7s}{'c_i':>10s}{'c_j = c_k':>12s}{'in the cone':>14s}"
          f"{'angle to u_i':>15s}{'m(d_i)':>10s}{'cos t':>10s}")
    for deg in (50.0, 60.0, 89.0, 90.0, 91.0, 110.0):
        U = tripod(deg)
        d, c = mechanism(U)[0]
        ins, m = capped(U, d[None])
        # + 0.0 only to stop the 90 deg row printing a negative zero: the two
        # other coefficients are exactly 0 there and which side of 0 the last bit
        # falls on is the arithmetic's business, not a fact about the arrangement
        print(f"{deg:6.0f}°{c[0]:10.4f}{c[1] + 0.0:12.4f}{str(bool(ins[0])):>14s}"
              f"{np.degrees(np.arccos(np.clip(d @ U[0], -1, 1))):14.2f}°"
              f"{m[0]:10.4f}{np.cos(np.radians(deg)):+10.4f}")
    print("\nc_j = c_k = -cos t / [(1 - cos t)(1 + 2 cos t) |u^i|], and every factor but "
          "the\n leading -cos t is positive for 0 < t < 120 deg, so the sign turns at "
          "90 deg and\n nowhere else -- and nothing in that counts the supports")

    # ---------------------------------------------------------------- the figure
    fig = plt.figure(figsize=(13.4, 6.6), dpi=200, facecolor=PAPER)
    # the first line of each is now the POSE -- what the reader is looking at --
    # because the three panels are one arrangement turned, and the turning is the
    # thing the eye has to follow across the row. The vertical push is the floor
    # and never moves; the other two tilt up towards it or down under the
    # horizontal, which is the difference between a shim driven under an edge and
    # a clamp hooked over one
    said = {
        "a": ("one push straight UP — the floor — and two across:",
              "the coordinate axes, and the cone is the corner of a cube.",
              "Every direction in it takes F or more"),
        "b": ("the two swing UP towards the vertical one and cluster,",
              "so the cone is far smaller — and every direction still",
              "takes F or more: the floor has not moved"),
        "c": ("the two swing DOWN to 20° below horizontal and splay,",
              "so the cone is twice the octant — but the field has",
              "fallen THROUGH F: the weak spot is inside the patch"),
    }
    print()
    for k, (deg, tag) in enumerate(CASES):
        U = tripod(deg)
        ins, m = capped(U, V)
        n = numbers(deg, ins, m)
        ax = fig.add_subplot(1, 3, k + 1, projection="3d", computed_zorder=False)
        ax.set_facecolor(PAPER)
        ball(ax, U)
        x = (2 * k + 1) / 6
        fig.text(x, 0.972, f"({tag})  three pushes {deg:.0f}° apart", ha="center",
                 va="center", color=INK, fontsize=14.5, fontweight="bold")
        for j, line in enumerate(said[tag]):
            fig.text(x, 0.935 - 0.031 * j, line, ha="center", va="center", color=MUTED,
                     fontsize=10.4)
        # the numbers under the picture are formatted from the same arrays the
        # paint is, so the words and the colours cannot drift apart. The octant
        # is named as a FRACTION as well as a percentage, in (a) only, because
        # that is what the reader was promised: 12.500 % is a measurement and
        # "one eighth" is the fact, and the tail is written only where it is
        # exactly true -- covers(90) is pi/2 out of 4 pi to machine precision,
        # and no other angle in the figure lands on a round fraction at all
        eighth = abs(n['cover_exact'] - 12.5) < 1e-9
        fig.text(x, 0.255, f"reaches {n['cover_exact']:.3f} % of the ball"
                 + ("  =  exactly one eighth" if eighth else ""), ha="center",
                 va="center", color=INK, fontsize=11.4)
        fig.text(x, 0.220, f"weakest {n['lo_exact']:.2f} F   ·   strongest "
                 f"{n['hi_exact']:.2f} F", ha="center", va="center", color=INK,
                 fontsize=11.4)
        fig.text(x, 0.185, f"{n['share']:.1f} % of the cone takes F or more", ha="center",
                 va="center", color=MUTED if n['share'] > 99.99 else "#a33224",
                 fontsize=10.6,
                 fontweight="normal" if n['share'] > 99.99 else "bold")
        print(f"({tag}) {deg:5.0f} deg   covers {n['cover_exact']:7.3f} %   "
              f"weakest {n['lo_exact']:.4f} F   strongest {n['hi_exact']:.4f} F   "
              f"{n['share']:6.2f} % of it at or above F")
    fig.subplots_adjust(left=0.0, right=1.0, top=0.900, bottom=0.245, wspace=0.0)
    key(fig.add_axes([0.412, 0.062, 0.176, 0.020]))
    band(fig, [(0.655, 0.126, onpaper(0.90), "under one support's strength"),
               (0.655, 0.086, onpaper(1.60), "one support's strength or more"),
               (0.655, 0.046, onpaper(None), "left bare: they cannot push that way at all")],
         (0.052, 0.124, 0.068, 0.030))
    fig.savefig(out, facecolor=PAPER)
    plt.close(fig)
    shape = trim(out)
    print(f"\nwrote {out}  {shape[1]} x {shape[0]}   ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
