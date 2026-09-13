"""Four, five and six pushes on a ball: what the fourth ray breaks, and what it buys.

`slides/tools/three_pushes.py` draws THREE contacts, each able to push one way only and
each capped at the same F, and paints the ball with the most they can put in each
direction. With three independent generators the decomposition

    v = sum_i c_i u_i

is UNIQUE, so `m(v) = F / max_i c_i` is a solve and a formula. THAT IS THE ONE
THING THAT DOES NOT SURVIVE A FOURTH RAY, and it does not survive it silently:
four or more generators leave a whole family of decompositions and the contacts
get to choose the best one, so

    m(v)  =  F / min { max_i c_i  :  c >= 0,  sum_i c_i u_i = v }

which is a LINEAR PROGRAM. Equivalently, and this is how `reach_lp` states it:
maximise r subject to r v = sum c_i u_i and 0 <= c_i <= F. A least-squares or a
pseudo-inverse hands back one of the many decompositions -- never the cheapest --
and so gets the answer wrong. `wrong_ways` RUNS the three obvious substitutions
rather than warning about them, on this figure's own arrangements:

  * `numpy.linalg.solve`, three_pushes.py's own decomposition, on a flat set. At
    tripod(120 deg) the three rays are coplanar, the matrix's condition number is
    1.9e16, and solve DOES NOT RAISE: it returns coefficients of order 2.8e15 for
    a direction the three cannot reach at all. Nothing downstream can tell.
  * `lstsq` / `pinv`, the minimum-NORM decomposition. With four or more rays it
    has a negative entry in almost every direction, so a `capped` written on it
    reports the cone as EMPTY -- 0 % where the truth is 100 %.
  * `nnls`, the minimum-norm NON-NEGATIVE decomposition, which is the dangerous
    one because it is genuinely feasible and genuinely smooth. Discard its
    residual and it reports the octant of panel (a) as 87.5 % of the ball; keep
    the residual and its magnitudes still come out too small, on 18.8 % of an
    unevenly spread four-ray cone and by up to a third. On the four symmetric
    sets drawn here its magnitudes happen to agree with the LP's, which is the
    worst possible luck: a check run only on them would pass.

The LP is the definition here, and it is also too slow to paint 20480 mesh cells
with four times over. `reach` is the closed form it is checked against, and the
check is the whole licence for using it: the set the contacts can supply within
the caps is the ZONOTOPE

    Z = { sum_i c_i u_i : 0 <= c_i <= F }

and m(v) is exactly Z's radial function -- the furthest one can go along v and
still be inside. A zonotope in three dimensions is bounded by parallelogram
facets whose normals are the pairs' cross products, and its support function is
h(a) = F sum_i max(0, a . u_i) by inspection, so

    m(v) = min over facet normals a with a . v > 0 of  h(a) / (a . v)

exactly, and vectorised over a million directions at once. `main` runs the LP on
1200 directions per arrangement and asserts the two agree; they agree to 1e-15.

WHAT THE FIGURE IS FOR. `slides/tools/dimension.md` records, for THREE pushes, that
the guarantee "every direction the cone reaches takes at least F" holds exactly
while the pairwise angles are at most 90 deg. THAT RULE DOES NOT SURVIVE A FOURTH
RAY, and the two middle panels are the two halves of the point:

  (a) THREE, every pair 90 deg -- the floor and two chocks square to it and to
      each other. An octant, exactly one eighth of the ball, everything in it at
      F or more. This is three_pushes.png's panel (a), redrawn as the baseline.

  (b) FOUR, every pair 109.47 deg -- the regular tetrahedron, four spread as
      evenly as four can be spread, one of them up. It reaches EVERY direction,
      and it has lost the guarantee: the weakest direction takes 0.8165 F, and
      only 5.99 % of the ball still takes a full support's worth. Pairwise angles
      barely past 90 deg, and the floor is gone.

  (c) FIVE, the square pyramid -- the floor and four chocks round it. It contains
      TWO EXACTLY OPPOSED PAIRS, 180 deg apart, the worst pairwise angle there
      is, and the weakest direction is exactly F. Half the ball, and every
      direction of it at F or more. Against (a): four times the reach for
      nothing.

  (d) SIX, the three axes both ways -- (c) with a lid over the top, one more ray.
      The whole ball, and still everything at F or more.

So 109.47 deg loses the guarantee and 180 deg keeps it. PAIRWISE ANGLE IS SIMPLY
NOT THE CRITERION once there are more than three generators.

THE CRITERION THAT REPLACES IT is in `worst_pair`, and it is the old one with the
thing that was invisible at n = 3 made visible. For a pair {u_i, u_j} let
a = unit(u_i x u_j) be the direction square to both: neither of them can put
anything along it. The most the arrangement can put there is what is LEFT,

    h(a) = F sum_{k not in {i,j}} max(0, a . u_k)

and the guarantee survives that pair exactly when h(a) >= F -- or when the cap
that pair's plane cuts off misses the cone entirely, which is the "d_i walks into
the patch" of three_pushes.py. With THREE rays that sum has ONE term, F cos(beta)
with beta the angle from the third ray to a, so it is at most F and reaches F
only when the third ray IS a -- which is precisely the mutually-square
arrangement. "Every pair at most 90 deg" was never a statement about pairs: it
was the n = 3 shadow of "whatever is left over must be able to cover the pair's
blind direction", and at n = 4 the sum acquires more terms and the shadow
detaches. The square pyramid's pair {+z, +x} is blind along +-y, and +y and -y
are themselves rays: h = F exactly. The tetrahedron's pairs leave two rays whose
positive parts along a sum to 0.8165 F, and that IS its weakest direction.

    python slides/tools/more_pushes.py   ->  slides/tools/figures/more_pushes.png
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
from scipy.integrate import quad                               # noqa: E402
from scipy.optimize import linprog, nnls                       # noqa: E402

# the house palette, the same one dimension.py and three_pushes.py take
INK, MUTED, PAPER = "#1b1b1a", "#6b6b66", "#ffffff"
ORANGE = "#E08A24"
SHELL = "#ffffff"
F = 1.0
EPS = 1e-9

# the two ramps and the break between them, taken unchanged from three_pushes.py
# so that a colour means the same thing in the two figures that answer the same
# question about three rays and about more than three
DEAD = LinearSegmentedColormap.from_list(
    "dead", ["#171215", "#3d1420", "#6e1e28", "#9c2a26", "#c03a25"])
LIVE = LinearSegmentedColormap.from_list(
    "live", [(0.00, "#f6fcfd"), (0.06, "#d3ebf4"), (0.18, "#9fd2e6"),
             (0.42, "#5ba7ca"), (0.70, "#2f7ba2"), (1.00, "#17415e")])
VMAX = None                                                    # set in main()

SUBDIV = 5                                                     # 20 * 4^5 = 20480 cells
STAND, UNIT_LN = 1.03, 0.42
MARK, LABEL_OUT = 0.34, 1.52   # the angle arc's radius, and how far out its number sits
LENS_W = 0.45                  # see `label_spot`: how much a lens outweighs a dash
# how big the ball is drawn and where its centre is put. The framing box has to
# stay a CUBE -- mplot3d rescales whatever box it is given to a fixed diagonal and
# then stretches the result to fill the axes rectangle -- which leaves exactly
# those two levers, and both are set by what has to fit.
#
# three_pushes.py can LOWER its cube by 0.28 of a radius, because in all three of
# its panels the vertical push stands out of the top and nothing balances it
# underneath. THAT IS NO LONGER TRUE: panel (d)'s sixth ray is the lid, it points
# straight DOWN, and it reaches 1.03 + 0.42/0.940 = 1.477 radii below the middle
# -- outside a cube lifted by 0.28, which clips the arrowhead off inside the axes
# rectangle rather than overflowing it. So LOWER is 0, the row is symmetric about
# the balls' own centres, and the PAGE has to grow to pay for the extra 0.28 at
# the bottom instead.
#
# It has to be the page and not the zoom, and that is worth knowing before anyone
# tries the other lever. Measured: the drawn size of the ball depends on the axes
# rectangle's WIDTH alone and not at all on its height -- a figure 0.9 in taller
# at the same zoom draws a ball of exactly the same 280 pixels. So a shorter page
# cannot be paid for by zooming out without shrinking every ball, and at 6.9 in
# the zoom that fitted (d)'s downward arrow was 1.62, which drew balls 2.80 in
# across against three_pushes.png's 3.20 in -- 12 % smaller than the figure whose
# panel (a) this one repeats. At 7.8 in the fit is 1.82 and the balls are 3.15 in,
# and the text is placed at the same INCH offsets from the edges as at 6.9 in
# rather than at the same fractions, so nothing else moves.
ZOOM, LOWER = 1.82, 0.0
SEEN_FLOOR = 0.20
N = 1_000_000                                                  # directions in the sweep
N_LP = 1200                                                    # directions the LP is run on

# --------------------------------------------------------------------- camera
#
# ONE camera for all four panels, and it is three_pushes.py's, unchanged, for its
# reasons: the reader compares the SIZE of four patches, so the camera cannot
# move; elev 20 keeps the horizon ellipse open enough to read as a ground plane
# (semi-minor axis sin 20 = 0.342) while affording enough offset from the
# arrangements' own mirror planes that a triad does not draw as a symmetric star.
# raised from 20, -12 on the human's instruction -- "转一下转一下，让这个蓝块向着我".
# The clustered cones all point up the same axis, so at a low camera their patches
# were seen at 70 deg from square on, foreshortened into slivers at the top of the
# ball. The camera can climb until a RAY runs into the view axis and projects to
# nothing, which is what SHY guards; scanning both angles with the phases these
# four sets already use, the closest approach at elev 60 is 26.05 deg -- the
# roomiest the band gets -- against 25.24 at 45 and 24.86 at 55. So 60, and the
# patches now face the reader 30 deg off square instead of 70
ELEV, AZIM = 60.0, -90.0
EYE = np.array([np.cos(np.radians(ELEV)) * np.cos(np.radians(AZIM)),
                np.cos(np.radians(ELEV)) * np.sin(np.radians(AZIM)),
                np.sin(np.radians(ELEV))])
SCREEN_UP = np.array([0.0, 0.0, 1.0]) - (np.array([0.0, 0.0, 1.0]) @ EYE) * EYE
SCREEN_UP /= np.linalg.norm(SCREEN_UP)
SCREEN_RIGHT = np.cross(SCREEN_UP, EYE)
LIGHT = -0.42 * SCREEN_RIGHT + 0.40 * SCREEN_UP + 0.82 * EYE
LIGHT /= np.linalg.norm(LIGHT)
SHELL_LIGHT, PAINT_LIGHT = (0.94, 0.06), (0.90, 0.13)
SHELL_A, PAINT_A = 0.13, 1.0

# THE ONE RULE OF THE CAMERA THAT HAD TO BE LOOSENED, and it was forced rather
# than chosen. three_pushes.py keeps every push between 20 and 84 deg of the eye:
# 20 because a ray nearer the view axis than that keeps under a third of its
# length on the page and reads as a dot, and 84 because 90 is the limb and a push
# round the back puts the paint it bounds round the back with it. FOUR AND FIVE
# AND SIX RAYS CANNOT ALL BE ON THE NEAR SIDE. Five rays whose azimuths run all
# the way round -- which is what a square pyramid is -- put at least two of
# themselves more than 90 deg from any eye whatever. So the rule becomes: no ray
# within SHY of the view axis OR of its opposite, which is the condition an ARROW
# needs to read; and the far ones are drawn as far ones. The paint's own
# condition is separate and is asserted in `ball` where it belongs.
SHY = 20.0


# ---------------------------------------------------------------- the sets drawn
#
# The azimuth of each arrangement about the vertical is FREE and is used. Turning
# a set about +z carries it onto a congruent set, so every number in this file --
# coverage, weakest, strongest, share -- is invariant under it, and only the pose
# moves. three_pushes.py deliberately refuses that freedom, because its three
# panels are ONE arrangement opening and closing and turning it between panels
# would make "the two swing up" false. Here the four panels are four different
# objects and there is no such sentence to protect, so each is turned to the
# phase that draws best. The phases were picked by `pose`'s own numbers, printed
# on every run: (c) and (d) are given the SAME phase, because (d) is (c) plus one
# ray and the reader has to see that at a glance.
UP = np.array([0.0, 0.0, 1.0])
# the azimuth of the four horizontals, shared by (c) and (d). `pose` prints the
# scan it came from: at 25 deg no ray comes within 41.4 deg of the view axis or
# of its opposite, and the four draw at two page lengths rather than one, which
# a square seen at 45 deg to the eye does not
PHASE_4 = 25.0


def horizontals(k, phase=0.0):
    """k unit vectors evenly round the horizon, the first at azimuth `phase`."""
    a = np.radians(phase) + 2 * np.pi * np.arange(k) / k
    return np.stack([np.cos(a), np.sin(a), np.zeros(k)], axis=1)


def cluster(k, theta=45.0, phase=0.0):
    """k rays leaning `theta` off a common axis, evenly round it.

    The arrangement the whole figure was rebuilt around, because the first
    version answered a question nobody had asked. Four rays spread as far apart
    as four can be -- the regular tetrahedron -- positively span space, so their
    cone is ALL of it and the "patch" is the whole ball. True, and it is what
    force closure means, but it is not what a reader looking for a patch expects
    to see, and it made the figure unreadable to the person it was drawn for.

    Clustered instead, the k rays are the CORNERS of a spherical polygon and the
    cone is the small piece they close on -- a triangle at k = 3, a quadrilateral
    at 4, a pentagon at 5, exactly as the three-push figure's octant is a
    triangle. Two closed forms make the row worth putting side by side, and both
    are checked by `assert` against the LP every run:

      * along any CORNER ray the reach is exactly F, whatever k and theta are.
        Only that ray can push there without something else needing to be
        cancelled, so the corners are the weak points -- the same fact the
        three-ray figure shows, and it does not care how many rays there are.
      * along the AXIS the reach is exactly k cos(theta) F, because every ray
        contributes its full cos(theta) and none of them fight. So clustering
        more rays hardly widens the patch and multiplies the middle: at
        theta = 45 the axis takes 2.12 F with three rays, 2.83 with four and
        3.54 with five.
    """
    t = np.radians(theta)
    a = np.radians(phase) + 2 * np.pi * np.arange(k) / k
    return np.stack([np.sin(t) * np.cos(a), np.sin(t) * np.sin(a),
                     np.full(k, np.cos(t))], axis=1)


def tripod(deg, phase=0.0):
    """Three unit vectors pairwise `deg` apart with ONE OF THEM STRAIGHT UP.

    three_pushes.py's `tripod`, verbatim apart from the free azimuth. With u_1 at
    the pole the arcs u_1 -> u_2 and u_1 -> u_3 are meridians, so the angle
    between them is the difference in azimuth -- which is the spherical
    triangle's own corner angle A at u_1, cos A = cos t / (1 + cos t). The cosine
    is snapped to 12 places first because cos(120 deg) comes back as
    -0.4999999999999998 and carried through makes arccos's argument -1.0000000004.
    """
    g = np.round(np.cos(np.radians(deg)), 12)
    half = np.degrees(np.arccos(np.clip(g / (1 + g), -1.0, 1.0))) / 2
    s = np.sqrt(max(0.0, 1 - g * g))
    low = horizontals(1, phase + half)[0], horizontals(1, phase - half)[0]
    return np.array([UP] + [s * h + g * UP for h in low])


def tetrahedron(phase=0.0):
    """Four unit vectors pairwise 109.47 deg apart, ONE OF THEM STRAIGHT UP.

    cos t = -1/3 for the regular tetrahedron, so the other three stand at polar
    angle arccos(-1/3) = 109.47 deg -- that is 19.47 deg BELOW the horizontal,
    which is the physical reading: the floor's push straight up and three clamps
    hooked over an edge and pulling down and out. It is the natural thing to try
    when three at 90 deg does not reach far enough, and it is the panel that
    loses the guarantee.
    """
    c = -1.0 / 3.0
    s = np.sqrt(1 - c * c)
    return np.array([UP] + [s * h + c * UP for h in horizontals(3, phase)])


# tag, the heading, the ray set, and the three lines of caption under the heading
def cases():
    return [
        ("a", "three pushes, leaning 45°", cluster(3, 45.0, 90.0),
         ("three rays round one axis: the corners of a spherical",
          "TRIANGLE, and the cone is the piece they close on.",
          "The corners are the weak points, at exactly F")),
        ("b", "four pushes, leaning 45°", cluster(4, 45.0, 45.0),
         ("one more ray, same lean. A QUADRILATERAL, barely",
          "wider than the triangle — but the middle has gone",
          "from 2.12 F to 2.83 F, because nothing fights")),
        ("c", "five pushes, leaning 45°", cluster(5, 45.0, 90.0),
         ("a PENTAGON. Adding rays to a cluster buys almost no",
          "reach and a great deal of strength: 12.2 % of the ball",
          "against 7.7 %, and 3.54 F down the middle")),
        ("d", "the same four, spread right out", tetrahedron(20.0),
         ("panel (b)'s four rays opened out to the regular",
          "tetrahedron. They now positively span space, so the",
          "cone is EVERYTHING — and 94 % of it is under F")),
    ]


# the rows printed but not drawn, so a regression shows up here as a number
# moving rather than as a figure merely looking different. The bipyramid is the
# sharpest of them: it has the SAME widest pair as the square pyramid, 180 deg,
# and unlike the square pyramid it loses the guarantee
def extra_rows():
    return [
        ("5, triangular bipyramid", np.vstack([UP, -UP, horizontals(3)])),
        ("4, the tetrahedron, cube corners",
         np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float) / np.sqrt(3)),
    ]


# ------------------------------------------------------------------ the physics


def fib(n):
    """A near-uniform set of directions, for measuring areas by COUNTING.

    The Fibonacci spiral, as everywhere else in this repo, and the choice is
    load-bearing rather than stylistic: a lat-lon mesh has cells whose areas go
    as sin(theta), so counting them weights a patch by where it sits, and this
    figure's patches sit in four different places. three_pushes.py measures that
    trap rather than asserting it -- at 50 deg a lat-lon count reports two thirds
    more patch than there is -- and the same measurement is repeated in `main`
    here, on this figure's own four sets.
    """
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r, t = np.sqrt(1 - z * z), np.pi * (1 + 5 ** 0.5) * i
    return np.stack([r * np.cos(t), r * np.sin(t), z], axis=1)


FLAT = 1e-7          # below this, a generator counts as lying IN a facet plane
WALL_EPS = 1e-7      # how far a direction must clear a cone wall to be outside it


def reach_lp(U, V, cap=F):
    """m(v) for each v, by linear programming. THE DEFINITION.

    Four or more generators in three dimensions leave a whole affine family of
    decompositions v = sum c_i u_i, and the contacts get to pick the one that
    keeps the largest coefficient smallest. So the reachable magnitude is

        m(v) = cap / min { max_i c_i : c >= 0, sum c_i u_i = v }

    which is a minimax over a polytope, not a solve. Written as an LP in the
    variables (r, c) it is: MAXIMISE r subject to sum_i c_i u_i - r v = 0 and
    0 <= c_i <= cap, with r >= 0. Feasibility is free -- r = 0, c = 0 is always
    feasible -- so an unreachable direction comes back as r = 0 rather than as an
    exception, which is the third state this figure needs and is why the LP is
    posed this way round rather than as a minimisation of max_i c_i.

    Slow, deliberately kept, and used on a subsample only: it is what `reach` is
    checked against. A least-squares or pseudo-inverse decomposition is available
    and is WRONG -- it returns some decomposition, never the cheapest -- and the
    wrongness is invisible, because the field it paints is smooth and plausible.
    """
    U = np.asarray(U, float)
    n = len(U)
    A = np.zeros((3, n + 1))
    A[:, 1:] = U.T
    obj = np.zeros(n + 1)
    obj[0] = -1.0                                              # maximise r
    out = np.zeros(len(V))
    for k, v in enumerate(np.atleast_2d(V)):
        A[:, 0] = -v
        res = linprog(obj, A_eq=A, b_eq=np.zeros(3),
                      bounds=[(0, None)] + [(0, cap)] * n, method="highs")
        assert res.status == 0, f"the LP did not solve: {res.message}"
        out[k] = res.x[0]
    return out


def facets(U, cap=F):
    """The zonotope's facet normals and support values.

    Z = { sum c_i u_i : 0 <= c_i <= cap } is a zonotope, and every facet of a
    zonotope in three dimensions is a zonogon spanned by generators lying in the
    facet's plane -- so every facet normal is square to at least TWO of the
    generators and is therefore +- unit(u_i x u_j) for some pair. The support
    function is h(a) = cap sum_i max(0, a . u_i), by inspection: to go as far as
    possible along a, turn on exactly the generators that have a positive
    component along it, each to its cap.

    Parallel and ANTIPARALLEL pairs give no facet and are skipped -- which is not
    an edge case here but the ordinary case, since the square pyramid contains
    +x with -x and the three axes contain three such pairs.

    RANK-DEFICIENT SETS ARE REFUSED rather than served. If the generators span a
    plane the zonotope is flat, every cross product is parallel to that plane's
    normal, and the facet enumeration comes back with two facets through the
    origin and no bound at all in the plane -- so the radial function it implies
    is +infinity on the very directions the set can actually reach. That is a
    silent lie of exactly the kind this module exists to avoid, so it raises. The
    LP has no such failure mode and `main` prints the pair side by side.
    """
    U = np.asarray(U, float)
    if np.linalg.matrix_rank(U, tol=1e-9) < 3:
        raise ValueError("the facet form needs generators that span space; use reach_lp")
    A = []
    for i, j in itertools.combinations(range(len(U)), 2):
        a = np.cross(U[i], U[j])
        r = np.linalg.norm(a)
        if r < 1e-9:                                           # parallel or opposed
            continue
        A += [a / r, -a / r]
    A = np.unique(np.round(np.array(A), 9), axis=0)            # each facet once
    # A GENERATOR LYING IN THE FACET PLANE MUST CONTRIBUTE EXACTLY NOTHING, and
    # deciding that needs a tolerance because `a` came out of a cross product.
    # `a . u` for such a generator is zero in exact arithmetic and ~5e-10 here;
    # `max(0, .)` keeps the residue, and a WALL of the cone -- whose h is zero
    # precisely because every generator lies in it or behind it -- comes back
    # with h ~ 5e-10 instead. The radial function then reads h/(a.v) on the
    # outside of that wall, which is small but not zero, so `reach` declares a
    # broad band OUTSIDE the cone reachable: 6416 of 20480 mesh cells for a cone
    # covering 1574 of them, painted at the ramp's black end. It never showed
    # while every set was axis-aligned and every such dot product was an exact
    # zero. It is not a tolerance to tune: FLOOR is far below any real
    # contribution, which is O(0.1) or more
    D = A @ U.T
    D[np.abs(D) < FLAT] = 0.0
    return A, cap * np.clip(D, 0, None).sum(axis=1)


def reach(U, V, cap=F):
    """m(v) for every direction at once. The closed form the LP licenses.

    m(v) is the radial function of the zonotope Z: the furthest one may go along
    v and still be inside it. Z is the intersection of its facets' half-spaces,
    a . x <= h(a), so along a ray r v the binding facet is whichever runs out
    first:

        m(v) = min over facet normals a with a . v > 0 of h(a) / (a . v)

    A facet with h(a) = 0 passes through the origin: that is a WALL OF THE CONE,
    and any v on its outward side gets m = 0, which is how "cannot push that way
    at all" arrives here as a value rather than as a special case.

    THE WALLS NEED A WIDER DEADBAND THAN THE REST, and it is not fussiness. On a
    direction lying in a wall the quotient is 0/0: `h` is exactly zero there and
    `a . v` is zero to within rounding, and whether the rounding lands at +1e-17
    or -1e-17 decides between `m = 0` and `m = F`. With the earlier arrangements
    every wall was axis-aligned and the cancellation was exact, so nothing showed.
    Cluster the rays and it does: `mid` cells straddling the rim came back at
    m ~ 0, the ramp painted them at its black end, and the patch grew a torn
    black fringe -- the same rounding that made the LP and the closed form
    disagree by 1.8e-6 on grazing directions. A wall only ever binds where it
    binds properly, so it is skipped until `a . v` clears WALL_EPS; the other
    facets, whose h is bounded away from zero, keep the tight one.
    """
    A, h = facets(U, cap)
    d = np.atleast_2d(np.asarray(V, float)) @ A.T
    eps = np.where(h <= 1e-12, WALL_EPS, 1e-12)[None, :]
    live = d > eps
    q = np.where(live, h[None, :] / np.where(live, d, 1.0), np.inf)
    return q.min(axis=1)


def walls(U):
    """The cone's own bounding planes: inward normals a with a . u_k >= 0 for all k.

    Every wall of a polyhedral cone contains at least two generators, so its
    normal is again +- unit(u_i x u_j); what picks the walls out of that list is
    that every OTHER generator lies on one side.
    """
    U = np.asarray(U, float)
    out = []
    for i, j in itertools.combinations(range(len(U)), 2):
        a = np.cross(U[i], U[j])
        r = np.linalg.norm(a)
        if r < 1e-9:
            continue
        for s in (1.0, -1.0):
            n = s * a / r
            if (U @ n >= -1e-9).all():
                out.append(n)
    return np.unique(np.round(np.array(out).reshape(-1, 3), 9), axis=0)


def covers(U):
    """The exact share of the ball the cone reaches, as a percentage.

    The cone meets the sphere in a CONVEX SPHERICAL POLYGON whose sides lie on
    the cone's walls, so its area is Girard's excess: sum of interior angles
    minus (k - 2) pi, for k walls. The interior angle where two walls meet is
    pi minus the angle between their inward normals. Three degenerate counts sit
    below the polygon: no wall at all is the whole sphere, one wall is a
    half-sphere, two walls are a lune of twice their dihedral.

    Checked against the sweep on every run. At 90 deg the three-push cone gives
    exactly pi/2, an eighth; the square pyramid has the single wall z = 0 and
    gives exactly a half; the other three arrangements have no wall and give the
    whole ball.
    """
    W = walls(U)
    k = len(W)
    if k == 0:
        return 100.0
    if k == 1:
        return 50.0
    if k == 2:
        return 100.0 * 2 * (np.pi - np.arccos(np.clip(W[0] @ W[1], -1, 1))) / (4 * np.pi)
    total, seen = 0.0, 0
    for i, j in itertools.combinations(range(k), 2):
        e = np.cross(W[i], W[j])
        r = np.linalg.norm(e)
        if r < 1e-9:
            continue
        for s in (1.0, -1.0):
            if (W @ (s * e / r) >= -1e-9).all():               # they meet on the cone
                total += np.pi - np.arccos(np.clip(W[i] @ W[j], -1, 1))
                seen += 1
    assert seen == k, f"{seen} vertices for {k} walls -- the cone is not simple"
    return 100.0 * (total - (k - 2) * np.pi) / (4 * np.pi)


def weakest(U, cap=F):
    """The floor of m over everything the cone reaches, exactly, and it is a min.

    m = min_a h(a) / (a . v), so the floor over the cone is

        min over facets a of  h(a) / max { a . v : v in the cone, |v| = 1 }

    and that inner maximum is |P(a)|, the length of a's projection ONTO the cone
    -- a non-negative least squares, `dimension.py`'s `gap` seen from the other
    side. If P(a) = 0 nothing in the cone has a positive component along a and
    that facet never binds; if h(a) = 0 the facet is a wall and bounds the cone
    rather than the field inside it.

    Reading it back through `worst_pair` is the whole finding: h(a) for
    a = unit(u_i x u_j) is what the rays OTHER THAN i and j can put along the one
    direction i and j are both blind to, and with three rays that is a single
    cosine and so at most cap.
    """
    A, h = facets(U, cap)
    U = np.asarray(U, float)
    best = np.inf
    for a, ha in zip(A, h):
        if ha <= EPS:
            continue
        p = U.T @ nnls(U.T, a)[0]
        n = np.linalg.norm(p)
        if n <= EPS:
            continue
        best = min(best, ha / n)
    return best


def strongest(U, cap=F):
    """The ceiling of m over the cone, exactly.

    Minimising max_i c_i over {c >= 0, |sum c_i u_i| = 1} is the same as
    MAXIMISING |sum c_i u_i| over the box 0 <= c_i <= cap -- the furthest point
    of the zonotope from the origin -- and a convex function over a box takes its
    maximum at a vertex, so the answer is the longest subset sum. 2^n of them,
    n <= 6, and no cleverness is worth the risk of missing one.
    """
    U = np.asarray(U, float)
    best = 0.0
    for r in range(1, len(U) + 1):
        for S in itertools.combinations(range(len(U)), r):
            best = max(best, np.linalg.norm(cap * U[list(S)].sum(axis=0)))
    return best


def half_lens(alpha, theta):
    """Area of a cap of angular radius `alpha` beyond the bisector of two centres
    `theta` apart. Two equal caps' overlap is twice this, by symmetry."""
    k = np.tan(theta / 2)
    return quad(lambda p: 2 * np.arccos(np.clip(k / np.tan(p), -1, 1)) * np.sin(p),
                theta / 2, alpha, limit=200)[0]


def share(U, cap=F):
    """The share OF THE BALL at or above F, exactly, as a percentage.

    m(v) >= cap means v is inside Z, so the set is the sphere with one cap cut
    off by each facet that comes closer to the origin than the sphere does:
    a . v > h(a) with h(a) < 1. Every arrangement here has all such facets at the
    SAME depth, so the caps are equal and their pairwise overlaps are two copies
    of `half_lens`; triple overlaps are checked to be empty rather than assumed,
    by intersecting each overlapping pair's boundary circles and testing the two
    points against every other cap. For the tetrahedron three adjacent caps meet
    in a single point and inclusion-exclusion stops at pairs.

    The sweep in `main` reproduces this to three decimals and is what says the
    combinatorics were right.
    """
    # TWO cases the cap formula below cannot see, and both bite once the rays are
    # clustered. It subtracts a cap for every facet with 0 < h < 1 and ignores the
    # facets with h <= 0 -- but those are the CONE'S OWN walls, and while the cone
    # was the whole ball there were none of them. For a cluster there are, and
    # leaving them out counts the entire sphere minus a few caps: it returned
    # 92 % of the ball for a cone covering 7.7 % of it.
    #
    # So the easy case is taken first and exactly. If the weakest direction of the
    # cone is already at or above the cap, then every direction the set reaches is,
    # and the share at or above F IS the cover -- no caps, no inclusion-exclusion.
    # That is every clustered panel. What is left for the formula is the case it
    # was written for, and it says so rather than being trusted: the cone must be
    # the whole ball, so that the sphere really is the thing the caps are cut from
    if weakest(U) >= cap - 1e-9:
        return covers(U)
    A, h = facets(U, cap)
    keep = (h > EPS) & (h < 1.0 - EPS)
    if not keep.any():
        return covers(U)                                       # nothing is below F
    assert covers(U) > 100.0 - 1e-6, \
        "share()'s cap formula needs a full-ball cone; this one is a proper cone " \
        "with a region below F, which needs the cone's own walls in the count too"
    A, h = A[keep], h[keep]
    # loosened from 1e-9 to FLAT. The depths are equal by symmetry, but they are
    # computed from cross products of irrational coordinates and agree only to
    # about 1e-9; genuinely unequal depths would differ by O(0.1), so this still
    # catches what it is for
    assert np.ptp(h) < FLAT, "unequal cap depths -- share() is written for equal ones"
    d = float(h[0])
    alpha = np.arccos(d)
    area = 4 * np.pi - len(A) * 2 * np.pi * (1 - d)
    for i, j in itertools.combinations(range(len(A)), 2):
        c = float(np.clip(A[i] @ A[j], -1, 1))
        th = np.arccos(c)
        if th >= 2 * alpha - 1e-12:                            # the caps do not meet
            continue
        area += 2 * half_lens(alpha, th)
        # the two points where the caps' boundary circles cross, so that a triple
        # overlap is DETECTED rather than assumed away
        ab = (d - d * c) / (1 - c * c)
        p = ab * (A[i] + A[j])
        g = np.sqrt(max(0.0, 1 - p @ p))
        nrm = np.cross(A[i], A[j])
        nrm = nrm / np.linalg.norm(nrm)
        for s in (1.0, -1.0):
            v = p + s * g * nrm
            k = [q for q in range(len(A)) if q not in (i, j) and A[q] @ v > d + 1e-9]
            assert not k, "three caps overlap -- inclusion-exclusion needs a third term"
    return 100.0 * area / (4 * np.pi)


def worst_pair(U, cap=F):
    """For each pair, the direction it is blind to and what the OTHERS can do there.

    a = unit(u_i x u_j) is square to both, so neither can put anything along it;
    h(a) = cap sum_{k not in (i,j)} max(0, a . u_k) is what is left over. The
    guarantee survives that pair when h(a) >= cap, or when the cap of directions
    the constraint cuts off misses the cone altogether -- which is measured here
    as h(a) / |P(a)|, the same quantity `weakest` minimises, with P the
    projection onto the cone.

    With THREE rays the sum has one term and is cap cos(beta): at most cap, and
    equal to cap only when the third ray IS a. That is the whole of the 90 deg
    rule, and it is why the rule has nothing to say once there are four.
    """
    U = np.asarray(U, float)
    out = []
    for i, j in itertools.combinations(range(len(U)), 2):
        a = np.cross(U[i], U[j])
        r = np.linalg.norm(a)
        if r < 1e-9:
            continue
        for s in (1.0, -1.0):
            n = s * a / r
            others = [k for k in range(len(U)) if k not in (i, j)]
            h = cap * sum(max(0.0, n @ U[k]) for k in others)
            if h <= EPS:
                continue                                       # a wall of the cone
            p = np.linalg.norm(U.T @ nnls(U.T, n)[0])
            out.append((i, j, n, h, np.inf if p <= EPS else h / p))
    return out


def pair_angles(U):
    """Every pairwise angle in degrees, sorted -- the quantity that stops working."""
    U = np.asarray(U, float)
    return np.sort([np.degrees(np.arccos(np.clip(U[i] @ U[j], -1, 1)))
                    for i, j in itertools.combinations(range(len(U)), 2)])


# ------------------------------------------------------------------- drawing

ICO = None                                                     # built once, on first use


def shell(k=SUBDIV):
    """The unit sphere as triangles: an icosahedron subdivided k times.

    Geodesic and not lat-lon, for three_pushes.py's reason: the pole is where the
    floor's push stands, a mark every panel is posed around and a CORNER of two of
    the four patches, and a lat-lon mesh would leave a bullseye of converging
    seams exactly on it in all four panels at once. k = 5 gives 20480 triangles
    about 2 deg a side, so half a cell is 1 deg -- which is what the exact curves
    drawn over the paint are wide enough to hide.

    These triangles are for DRAWING only. Nothing is ever counted on them; `fib`
    does the measuring, on a set with no mesh in it at all.
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
        # crack. Normalised at every level rather than once at the end
        ab, bc, ca = [(p + q) / np.linalg.norm(p + q, axis=1, keepdims=True)
                      for p, q in ((a, b), (b, c), (c, a))]
        T = np.concatenate([np.stack([a, ab, ca], 1), np.stack([ab, b, bc], 1),
                            np.stack([ca, bc, c], 1), np.stack([ab, bc, ca], 1)])
    mid = T.mean(axis=1)
    ICO = (T, mid / np.linalg.norm(mid, axis=1, keepdims=True))
    return ICO


def shaded(normals, rgb, amb, key):
    """A colour per face: how squarely each one meets the light.

    Lambert, clipped at zero, so what is left on an unlit face is the ambient. On
    a sphere the normal IS the direction, so the limb darkens by itself.
    """
    n = np.atleast_2d(np.asarray(normals, float))
    f = amb + key * np.clip(n @ LIGHT, 0, None)
    return np.clip(f[:, None] * np.atleast_2d(rgb), 0, 1)


def ramp(m):
    """The field as colours: two ramps meeting at F, with a hard break between.

    Read as two states and not as one number. The break is the whole question,
    so it is a jump in lightness as well as in hue -- hot red just below F, near
    white just above it.
    """
    m = np.asarray(m, float)
    out = np.zeros(m.shape + (3,))
    lo = m < F
    out[lo] = DEAD(np.clip(m[lo] / F, 0, 1))[..., :3]
    out[~lo] = LIVE(np.clip((m[~lo] - F) / (VMAX - F), 0, 1))[..., :3]
    return out


def onpage(d, floor=SEEN_FLOOR):
    """What fraction of a length along `d` survives the projection to the page.

    A radial arrow on a ball has no foreshortening cue on it to contradict, so
    giving it the 3-D length whose PAGE length is the one wanted costs nothing
    and buys everything: eighteen arrows all standing for the same identical F
    are drawn the same length. Only the LENGTH is treated this way; the tail
    still stands at STAND on the shell, because that is a position.
    """
    return max(float(np.sqrt(max(1.0 - (np.asarray(d, float) @ EYE) ** 2, 0.0))), floor)


class Arrow3D(FancyArrowPatch):
    """A flat arrow with both ends pinned to points in space.

    `quiver` builds its head out of two barbs in a plane it picks from the shaft
    alone, so the more nearly an arrow points at the reader the more nearly its
    head is edge on. The eighteen pushes here stand between 20 and 160 deg off
    the camera's axis, so eighteen quiver heads would come out at eighteen
    weights. A FancyArrowPatch re-projects both ends and draws its head on the
    PAGE, so the head is the same head whichever way the arrow points.
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
    as a curve lying on it.
    """
    P = np.asarray(P, float)
    if len(P) < 2:
        return
    for near, kw in ((False, far_kw), (True, near_kw)):
        keep = (P @ EYE > 0) == near
        ax.plot(*np.where(keep[:, None], lift * P, np.nan).T, **kw)


def arc(a, b, n=361):
    """The great-circle arc between two directions, walked evenly."""
    t = np.linspace(0.0, 1.0, n)[:, None]
    g = (1 - t) * np.asarray(a, float) + t * np.asarray(b, float)
    return g / np.linalg.norm(g, axis=1, keepdims=True)


def rim(U, n=721):
    """The boundary of the cone on the ball: one arc per wall, cut to the cone.

    A wall of the cone contains two or more rays and is a whole great circle; the
    part of it that bounds the cone is where every OTHER wall is still satisfied.
    Drawn rather than left to the paint's own edge because a triangle mesh renders
    a boundary as a 2 deg staircase, and in (a) that boundary is most of what the
    panel says.
    """
    W = walls(U)
    out = []
    for w in W:
        e1 = np.cross(w, [0.0, 0.0, 1.0] if abs(w[2]) < 0.9 else [1.0, 0.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(w, e1)
        th = np.linspace(0, 2 * np.pi, n)[:, None]
        P = np.cos(th) * e1 + np.sin(th) * e2
        out += runs(P, (P @ W.T >= -1e-9).all(axis=1))
    return out


def contour(U, n=721):
    """The curve where m = F exactly: the zonotope's boundary met with the ball.

    m(v) >= F is v inside Z, so the curve is the sphere's intersection with each
    facet plane a . v = h(a) -- a circle at angular radius arccos(h(a)) -- kept
    only where the point is actually ON the boundary of Z, i.e. where no OTHER
    facet is already violated.

    That second condition is easy to miss and is not decoration: a . v = h(a)
    holds all the way round its circle, and a stretch of it can run through the
    inside of the region where another facet is over ITS bound, where the field
    is below F on both sides and the curve bounds nothing. Only the tetrahedron
    has any of this curve at all; the other three arrangements' facets all stand
    at distance F or more and merely touch the ball.
    """
    A, h = facets(U)
    out = []
    for a, d in zip(A, h):
        if not (EPS < d < 1.0 - 1e-9):
            continue
        e1 = np.cross(a, [0.0, 0.0, 1.0] if abs(a[2]) < 0.9 else [1.0, 0.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(a, e1)
        b = np.arccos(d)
        th = np.linspace(0, 2 * np.pi, n)[:, None]
        P = np.cos(b) * a + np.sin(b) * (np.cos(th) * e1 + np.sin(th) * e2)
        out += runs(P, (P @ A.T <= h[None, :] + 1e-9).all(axis=1))
    return out


def runs(P, ok, least=2.0):
    """The stretches of a closed sampled curve where `ok` holds, as arrays.

    Runs shorter than `least` degrees are dropped as the tangencies they are --
    a facet at distance exactly F touches the ball at one point and would
    otherwise come back as a hairline.
    """
    out, k, step = [], 0, 360.0 / (len(ok) - 1)
    while k < len(ok):
        if ok[k]:
            j = k
            while j + 1 < len(ok) and ok[j + 1]:
                j += 1
            if (j - k) * step >= least:
                out.append(P[k:j + 1])
            k = j
        k += 1
    if len(out) > 1 and ok[0] and ok[-1]:                      # the wrap-around join
        out[0] = np.vstack([out[-1], out[0]])
        out.pop()
    return out


def angle_mark(U, n=361):
    """The WIDEST pair, as an arc of radius MARK inside the ball, and its number.

    One mark and not C(n,2) of them. What the figure is arguing is that the
    widest pairwise angle predicts nothing -- 109.5 deg loses the guarantee and
    180 deg keeps it -- so the one angle worth drawing is the widest, and the
    reader is meant to read it straight off against the line underneath saying
    whether the guarantee held.

    An OPPOSED pair has no arc: every great semicircle from u to -u is 180 deg,
    and picking one is picking a plane. The one picked is the semicircle through
    the direction square to the pair that stands FURTHEST FROM EVERY OTHER RAY,
    so the mark threads the gap instead of lying along a neighbour -- which is
    what it would do in (d), where a careless choice runs the 180 deg mark
    straight down two of the other four pushes.
    """
    U = np.asarray(U, float)
    pairs = list(itertools.combinations(range(len(U)), 2))
    cosines = np.array([U[i] @ U[j] for i, j in pairs])
    wide = np.flatnonzero(cosines <= cosines.min() + 1e-9)
    # among equally wide pairs, the one whose worse end faces the camera most
    i, j = pairs[max(wide, key=lambda k: min(U[pairs[k][0]] @ EYE, U[pairs[k][1]] @ EYE))]
    deg = np.degrees(np.arccos(np.clip(U[i] @ U[j], -1, 1)))
    if U[i] @ U[j] > -1 + 1e-9:
        P = MARK * arc(U[i], U[j], n)
        return P, deg, label_spot(P, U)
    e1 = np.cross(U[i], [0.0, 0.0, 1.0] if abs(U[i][2]) < 0.9 else [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(U[i], e1)
    th = np.linspace(0, 2 * np.pi, 721)[:, None]
    C = np.cos(th) * e1 + np.sin(th) * e2                      # the choices of plane
    clear = -np.abs(C @ U.T).max(axis=1)                       # how far from every ray
    near = np.flatnonzero(clear >= clear.max() - 1e-9)
    # the choice is symmetric about the pair's own axis and so always comes in
    # mirror pairs; the tie goes to the one facing the camera, or the label lands
    # behind the ball as often as in front of it
    through = C[near[int(np.argmax(C[near] @ EYE))]]
    P = MARK * np.vstack([arc(U[i], through, n // 2), arc(through, U[j], n // 2)])
    return P, deg, label_spot(P, U)


def label_spot(P, U):
    """Where along the mark arc the number goes: wherever nothing else is drawn.

    NOT the arc's midpoint, which is the obvious place and is the wrong one. For
    every regular arrangement here the midpoint of the arc between two rays is the
    direction that PAIR pulls hardest in -- the tetrahedron's six strongest
    directions ARE its six edge midpoints, and (b)'s pale lenses stand exactly
    there -- so a paper-backed label at the midpoint sits on the one feature the
    panel exists to show. Measured on the first render: it hid the largest of (b)'s
    three visible lenses almost entirely.

    So the label is put at the point of the arc whose PAGE position stands
    furthest from everything else in the picture: the rays' dots, the dashed radii
    running out to them (as segments, not as endpoints -- a label clear of both
    ends of a line can still sit across its middle), the centre they meet at, and
    the strongest directions, which are the normalised vertices of the zonotope
    and are where the pale lenses are. One number, computed rather than nudged.
    """
    U = np.asarray(U, float)
    peaks = []
    for r in range(2, len(U) + 1):
        for S in itertools.combinations(range(len(U)), r):
            v = U[list(S)].sum(axis=0)
            if np.linalg.norm(v) > 1e-9:
                peaks.append(v / np.linalg.norm(v))
    A = np.array([onpaper_xy(w) for w in peaks])
    ends = np.array([onpaper_xy(u) for u in U])                # the radii, as segments
    C = np.array([onpaper_xy(q) for q in P]) * LABEL_OUT
    best, at = -1.0, P[len(P) // 2]
    for k, c in enumerate(C):
        # a lens is a REGION about 30 deg across and a radius is a hairline, so a
        # ball radius of clearance is worth less next to the first than the second.
        # LENS_W buys the difference: without it the search settles for clipping
        # the corner of (b)'s largest lens in order to stand further off a dash
        d = min(LENS_W * np.linalg.norm(A - c, axis=1).min(), np.linalg.norm(c))
        for e in ends:                                         # distance to segment 0 -> e
            t = np.clip((c @ e) / (e @ e), 0.0, 1.0)
            d = min(d, np.linalg.norm(c - t * e))
        if d > best:
            best, at = d, P[k]
    return at


def ball(ax, U):
    """One panel: the ball of directions, painted with what the rays can push.

    Everything drawn is a consequence of the same rays and nothing is placed by
    hand: the paint is `reach` on the mesh centres, the rim is the cone's own
    walls, the F curve is the zonotope's facet circles, the arrows are the rays.
    """
    cells, mid = shell()
    m = reach(U, mid)
    inside = m > EPS
    # TWO collections, split by the one thing that has to differ between them:
    # the line width. Agg rasterises every polygon on its own, so two that share
    # an edge leave a hairline of whatever is behind them between the two --
    # 20480 cells' worth, which prints the tessellation across the ball. Giving
    # each cell an outline in its OWN colour closes it, but only where the fill
    # is opaque; over the translucent shell the outline blends a second time and
    # draws the mesh a shade darker instead. A per-face `linewidths` array cannot
    # do it: Poly3DCollection re-orders its faces by depth on every draw and
    # permutes the face and edge COLOURS to match, but the line widths are the
    # base Collection's and stay where they were.
    if (~inside).any():
        ax.add_collection3d(Poly3DCollection(
            cells[~inside], zorder=1.0, zsort="average", edgecolors="none", linewidths=0.0,
            facecolors=np.hstack([
                shaded(mid[~inside], matplotlib.colors.to_rgb(SHELL), *SHELL_LIGHT),
                np.full((int((~inside).sum()), 1), SHELL_A)])))
    paint = np.hstack([shaded(mid[inside], ramp(m[inside]), *PAINT_LIGHT),
                       np.full((int(inside.sum()), 1), PAINT_A)])
    # the paint is split at the LIMB and drawn twice, near half over the shell and
    # far half under it. The earlier version drew it as ONE collection in front of
    # the shell and asserted that no bare near-side cell ever covered a painted
    # far-side one -- which held for the arrangements it had, and stopped holding
    # the moment the rays were clustered: a cone leaning 45 deg round the vertical
    # wraps past the limb, so part of its patch really is round the back, and a
    # single collection in front drew that part THROUGH the ball. Splitting costs
    # nothing here because the shell between them is 13 % opaque and is what makes
    # the far half read as far
    seen = mid[inside] @ EYE > 0
    # SPLITTING COSTS DEPTH SORTING BETWEEN THE TWO, and here that is affordable
    # for a reason that has to be checked rather than assumed. three_pushes.py
    # could assert the whole patch was on the near side; that is false here, since
    # the square pyramid paints the entire upper half of the ball. What is needed
    # is weaker and is exactly what "paint in front of shell" requires: wherever a
    # PAINTED cell lies on the far side, the point of the ball at the same place
    # on the page -- its mirror in the plane square to the eye -- must be painted
    # too, so that nothing bare is being drawn over. For the square pyramid the
    # mirror of a far point with z >= 0 has a LARGER z, so it holds; for the two
    # full-ball panels there is no bare cell at all; for the octant no painted
    # cell is on the far side in the first place. Measured on 40000 directions,
    # not argued
    for half, z in ((~seen, 0.30), (seen, 1.1)):
        if half.any():
            ax.add_collection3d(Poly3DCollection(
                cells[inside][half], facecolors=paint[half], edgecolors=paint[half],
                linewidths=0.4, zsort="average", zorder=z))
    # the LIMB: the great circle square to the eye, which is where the ball turns
    # away and so is its outline on the page. A white fill at 13 % on near-white
    # paper has no edge of its own, and without this (a) and (c) read as a patch
    # with some arrows near it
    th = np.linspace(0, 2 * np.pi, 481)[:, None]
    ax.plot(*(np.cos(th) * SCREEN_RIGHT + np.sin(th) * SCREEN_UP).T,
            color=MUTED, lw=1.1, alpha=0.75, zorder=2.9)
    interior(ax, U)
    # the world equator, which in this pose is the HORIZON: +z is the floor's own
    # push, so z = 0 is the set of horizontal directions, seen from 20 deg above.
    # It is what says the ball is round and not a disc, and it is where (c)'s and
    # (d)'s four chock pushes stand
    th = np.linspace(0, 2 * np.pi, 361)
    split(ax, np.stack([np.cos(th), np.sin(th), 0 * th], axis=1),
          dict(color=MUTED, lw=0.9, alpha=0.65, zorder=2.0),
          dict(color=MUTED, lw=0.9, alpha=0.22, zorder=0.4), lift=1.02)
    for piece in rim(U):
        split(ax, piece,
              dict(color=ORANGE, lw=3.2, alpha=1.0, solid_capstyle="round", zorder=2.6),
              dict(color=ORANGE, lw=2.2, alpha=0.40, solid_capstyle="round", zorder=0.42))
    # the F curve, cased in paper so it survives a hot red field on one side and
    # a near-white one on the other -- one colour cannot cross both
    for piece in contour(U):
        split(ax, piece,
              dict(color=PAPER, lw=2.4, alpha=0.90, solid_capstyle="round", zorder=2.7),
              dict(color=PAPER, lw=1.8, alpha=0.28, solid_capstyle="round", zorder=0.44))
        split(ax, piece,
              dict(color=INK, lw=1.3, alpha=1.0, solid_capstyle="round", zorder=2.75),
              dict(color=INK, lw=1.0, alpha=0.35, solid_capstyle="round", zorder=0.45))
    # the rays themselves. Same colour, same page length, same weight in every
    # panel, so what changes across the row is only how many there are and where
    # they stand.
    #
    # THE FAR ONES ARE DRAWN OVER THE BALL, DASHED, and that is the same
    # deliberate lie about depth the interior radii tell. Honest depth deletes
    # them: an arrow standing on the far side projects INSIDE the disc -- its
    # tail lands at 1.03 * sqrt(1 - (u.EYE)^2) of a radius, under 1 whenever the
    # ray is more than a few degrees over the limb -- so an opaque painted ball
    # hides it completely, and in (d) that would silently turn six pushes into
    # three. A dash is the ordinary way to say "behind what it crosses"; the dot
    # at its foot is drawn hollow for the same reason
    for u in np.asarray(U, float):
        near = u @ EYE > 0
        tail = STAND * u
        ax.add_artist(Arrow3D(tail, tail + UNIT_LN / onpage(u) * u, color=ORANGE,
                              lw=3.2, alpha=1.0 if near else 0.55, zorder=3.6 if near else 3.4,
                              linestyle="solid" if near else (0, (3.4, 2.2)),
                              arrowstyle="-|>,head_width=.20,head_length=.42"))
        ax.scatter(*(1.02 * u), s=86, c=ORANGE if near else "none", edgecolors=ORANGE,
                   linewidths=1.2 if not near else 1.2,
                   alpha=1.0 if near else 0.55, depthshade=False,
                   zorder=3.5 if near else 3.38)
    # orthographic, and that is not a preference: the reader is being asked to
    # compare the AREA of four patches sitting at four different places on the
    # ball, and mplot3d's default perspective at focal length 1 draws the near
    # side of a unit ball visibly larger than the far side
    ax.set_proj_type("ortho")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5 + LOWER, 1.5 + LOWER)
    ax.set_box_aspect((1, 1, 1), zoom=ZOOM)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_axis_off()


def interior(ax, U):
    """The radii, the widest angle between two of them, and the centre they meet at.

    This is what the shell was made clear for. The rays are directions, so on the
    ball they are POINTS, and what a reader wants to see is the thing those points
    are directions OF: radii out of one centre with an angle standing between two
    of them.

    DASHED and drawn over everything rather than under it. Honest depth hides
    them: at any point of the disc the sphere's near surface is at sqrt(1 - r^2)
    and every interior line is nearer the centre than that, so an opaque patch
    occludes all of them wherever it covers them -- which in (b) and (d) is the
    whole ball. A dash is the ordinary way to say "behind what it crosses".
    """
    for u in U:
        ax.plot(*np.array([[0.0, 0.0, 0.0], u]).T, color=INK, lw=1.3, alpha=0.80,
                dashes=(4.5, 3.0), solid_capstyle="butt", zorder=3.0)
    ax.scatter([0.0], [0.0], [0.0], s=30, c=INK, depthshade=False, zorder=3.05)
    P, deg, spot = angle_mark(U)
    ax.plot(*P.T, color=INK, lw=1.3, alpha=0.80, dashes=(4.5, 3.0), zorder=3.02)
    at = spot * LABEL_OUT
    ax.text(*at, f"{deg:.0f}°" if abs(deg - round(deg)) < 0.05 else f"{deg:.1f}°",
            color=INK, fontsize=13, ha="center", va="center", zorder=3.06,
            bbox=dict(boxstyle="round,pad=0.16", facecolor=PAPER, edgecolor="none",
                      alpha=0.85))


def onpaper(m):
    """The colour a reading of `m` actually lands on the page as.

    A swatch has to match the ball, and the ball's colours have been through the
    Lambert term and -- for the bare shell -- through being seen at 13 % over
    near-white paper. Both are undone here rather than eyeballed, so the key
    follows the paint if either is ever changed. `m` of None means the shell.
    """
    if m is None:
        c = shaded(EYE, matplotlib.colors.to_rgb(SHELL), *SHELL_LIGHT)[0]
        return (1 - SHELL_A) * np.array(matplotlib.colors.to_rgb(PAPER)) + SHELL_A * c
    return shaded(EYE, ramp(np.array([m]))[0], *PAINT_LIGHT)[0]


def key(ax):
    """The scale: one ramp, broken at F, and the ticks that put numbers on it."""
    t = np.linspace(0, VMAX, 512)
    ax.imshow(shaded(EYE, ramp(t), *PAINT_LIGHT)[None, :, :], extent=(0, VMAX, 0, 1),
              origin="lower", aspect="auto")
    ax.plot([F, F], [0, 1], color=INK, lw=1.8)                 # the break itself
    ax.set_xlim(0, VMAX)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, F, VMAX])
    ax.set_xticklabels(["0", "F", f"{VMAX:.2f} F"], fontsize=9.5, color=MUTED)
    ax.tick_params(colors=MUTED, length=3, pad=2)
    for s in ax.spines.values():
        s.set_color(MUTED)
    ax.text(0.5, 1.9, "m(v) = F / min$_c$ max$_i$ c$_i$", transform=ax.transAxes,
            ha="center", va="bottom", color=INK, fontsize=11)


def band(fig, rows, marks):
    """The bottom strip: what the marks mean on the left, what the colours mean on
    the right, with the scale itself between them."""
    for x, y, sw, line in rows:
        # the swatch is squared against the FIGURE's aspect, not against the page,
        # or a wide canvas turns every square into a letterbox
        w, h = 0.0072, 0.0072 * fig.get_figwidth() / fig.get_figheight()
        fig.add_artist(plt.Rectangle((x, y - h / 2), w, h, facecolor=sw,
                                     edgecolor=MUTED, lw=0.5, transform=fig.transFigure))
        fig.text(x + w + 0.006, y, line, ha="left", va="center", color=MUTED, fontsize=9.6)
    x, ya, yb, wide = marks
    # drawn with the very artists the panels use, not with look-alike glyphs, so
    # a change of arrowhead cannot leave the key describing the last version
    for y, style, alpha, text in (
            (ya, "solid", 1.0, "a push, capped at F — near side of the ball"),
            (yb, (0, (3.4, 2.2)), 0.55, "the same, standing round the BACK of the ball")):
        fig.add_artist(FancyArrowPatch((x, y), (x + wide, y), transform=fig.transFigure,
                                       arrowstyle="-|>,head_width=.20,head_length=.42",
                                       mutation_scale=13, color=ORANGE, lw=3.2,
                                       linestyle=style, alpha=alpha,
                                       shrinkA=0.0, shrinkB=0.0))
        fig.text(x + wide + 0.008, y, text, ha="left", va="center", color=MUTED, fontsize=9.6)


def trim(out, pad=18):
    """Cut the saved image down to what was drawn on it.

    mplot3d rescales whatever framing box it is given to a fixed diagonal and then
    stretches the result to fill its axes rectangle, so the box has to stay a cube
    and a cube framed to the width leaves the page empty above and below. Cropping
    the raster afterwards moves no pixel that it keeps.
    """
    a = plt.imread(out)
    ink = (a[:, :, :3] < 0.96).any(axis=2)
    rows, cols = np.where(ink.any(axis=1))[0], np.where(ink.any(axis=0))[0]
    y0, y1 = max(rows.min() - pad, 0), min(rows.max() + 1 + pad, a.shape[0])
    x0, x1 = max(cols.min() - pad, 0), min(cols.max() + 1 + pad, a.shape[1])
    matplotlib.image.imsave(out, a[y0:y1, x0:x1])
    return a[y0:y1, x0:x1].shape


# --------------------------------------------------------------- what it prints


def onpaper_xy(w):
    """Where a direction lands on the PAGE, in ball radii.

    The page basis is the camera's, so this is the projection the panels are
    actually drawn with rather than a model of it.
    """
    w = np.asarray(w, float)
    return np.array([w @ SCREEN_RIGHT, w @ SCREEN_UP])


def pose(sets):
    """Where every ray lands, and the one rule the camera still has to keep.

    Every set's AXIS is vertical -- the mean of its rays points along +z, so the
    patch sits square on the page in all four panels and their sizes can be
    compared by eye. (The earlier version asserted one RAY was vertical instead.
    That held while the sets each contained the floor's own push; it does not
    hold for a symmetric cluster, where the rays lean evenly round the vertical
    and none of them is it.) And NO RAY POINTS AT THE CAMERA OR STRAIGHT AWAY
    FROM IT. A ray within SHY of either
    end of the view axis keeps under a third of its length on the page and reads
    as a dot rather than as an arrow; the free azimuth of each arrangement is
    spent on that and on nothing else.
    """
    print("the pose: every set axis vertical, and where each ray lands\n")
    print(f"{'panel':>7s}{'ray':>7s}{'polar':>9s}{'page x':>10s}{'page y':>10s}"
          f"{'page len':>11s}{'off-axis':>11s}{'margin':>10s}")
    worst = 180.0
    for tag, _, U, _ in sets:
        axis = np.sum(np.asarray(U, float), axis=0)
        assert np.linalg.norm(axis) < 1e-9 or abs(axis @ UP) > 0.999 * np.linalg.norm(axis), \
            f"panel {tag}: the set's axis is not vertical"
        for i, u in enumerate(np.asarray(U, float)):
            x, y = onpaper_xy(u)
            off = np.degrees(np.arccos(np.clip(float(u @ EYE), -1, 1)))
            worst = min(worst, off, 180 - off)
            print(f"{tag:>7s}{'u_' + str(i + 1):>7s}{np.degrees(np.arccos(u[2])):8.1f}°"
                  f"{x:+10.3f}{y:+10.3f}{np.hypot(x, y):11.3f}{off:10.2f}°"
                  f"{min(off, 180 - off):9.2f}°")
    print(f"\nno ray comes within {worst:.2f}° of the view axis or of its opposite "
          f"(the rule is {SHY:.0f}°)")
    assert worst >= SHY, "a ray is aimed at the camera or straight away from it"


def wrong_ways(sets):
    """The three obvious substitutions for the LP, RUN rather than warned about.

    Every one of them is one line, every one is in the standard library of this
    repo already, and every one is wrong in a way that leaves no exception behind.
    The numbers below are the reason the LP is kept even though it is fifty times
    slower than the closed form it licenses.
    """
    print("\nthe three obvious substitutions for the LP, run rather than warned about\n")

    # 1. the direct solve, on a set that is flat and does not say so
    print(" 1. np.linalg.solve -- three_pushes.py's own decomposition -- as the three\n"
          "    pushes are opened towards coplanar. The target is one fixed direction\n"
          "    out of the tripod's plane:\n")
    print(f"{'':>10s}{'t':>12s}{'cond(U^T)':>14s}{'solve max|c|':>15s}{'raised?':>10s}"
          f"{'the LP says':>14s}")
    v = np.array([0.3, 0.0, 0.954])
    v /= np.linalg.norm(v)
    for deg in (110.0, 119.0, 119.99, 119.9999, 120.0):
        U = tripod(deg)
        try:
            got, raised = f"{np.abs(np.linalg.solve(U.T, v)).max():15.4e}", "no"
        except np.linalg.LinAlgError:
            got, raised = f"{'--':>15s}", "yes"
        print(f"{'':>10s}{deg:11.4f}°{np.linalg.cond(U.T):14.3e}{got}{raised:>10s}"
              f"{reach_lp(U, v[None])[0] + 0.0:12.4f} F")
    print("\n    At 120° the three are coplanar, the matrix is singular to 1.9e16, and\n"
          "    solve RETURNS -- coefficients of order 1e15 for a direction the three\n"
          "    cannot reach at all. `facets` refuses that set instead; the LP answers 0.")
    for name, U in (("{up, u, -u}", np.array([UP, [1.0, 0, 0], [-1.0, 0, 0]])),
                    ("{±x, ±y}", np.array([[1.0, 0, 0], [0, 1.0, 0],
                                           [-1.0, 0, 0], [0, -1.0, 0]]))):
        try:
            facets(U)
            raise AssertionError("the facet form should have refused a flat set")
        except ValueError as e:
            print(f"    {name:>12s}, rank {np.linalg.matrix_rank(U, tol=1e-9)}: {e}")

    # 2 and 3. the two decompositions that are available for n > 3
    V = fib(4000)
    print("\n 2. lstsq / pinv, the minimum-NORM decomposition, and\n"
          " 3. nnls, the minimum-norm NON-NEGATIVE one -- both against the LP, on the\n"
          "    four sets drawn and on one four-ray set that is NOT symmetric:\n")
    print(f"{'':>26s}{'the cone is':>13s}{'lstsq calls it':>16s}{'nnls calls it':>15s}"
          f"{'nnls m too small on':>21s}{'by up to':>10s}")
    uneven = np.vstack([UP, np.sin(np.radians(80)) * horizontals(3)
                        + np.cos(np.radians(80)) * UP])
    for name, U in [(f"({t}) {len(U)} rays", U) for t, _, U, _ in sets] + \
                   [("four, unevenly spread", uneven)]:
        A = np.asarray(U, float).T
        C = np.linalg.lstsq(A, V.T, rcond=None)[0].T
        ls = (C >= -1e-9).all(axis=1)
        nn = np.array([nnls(A, x)[0] for x in V])
        top = nn.max(axis=1)
        mn = np.where(top > 1e-12, 1.0 / np.maximum(top, 1e-300), 0.0)
        m = reach(U, V)
        live = m > EPS
        rel = (m[live] - mn[live]) / m[live]
        print(f"{name:>26s}{100 * live.mean():11.2f} %{100 * ls.mean():14.2f} %"
              f"{100 * (mn > EPS).mean():13.2f} %{100 * (rel > 1e-9).mean():19.2f} %"
              f"{100 * rel.max():9.1f} %")
    print("\n    lstsq's minimum-norm c has a negative entry in every direction once\n"
          "    there are four rays, so a `capped` written on it paints NOTHING. nnls\n"
          "    is the dangerous one: it is feasible, it is smooth, and if its residual\n"
          "    is discarded it calls the octant 87 % of the ball. On the four symmetric\n"
          "    sets its MAGNITUDES happen to match the LP exactly -- which is the worst\n"
          "    luck there is, since a check run only on them would pass -- and on an\n"
          "    uneven four they are short over a fifth of the cone by up to a third.")


ANCHOR_N = 4000                    # the sweep the handoff numbers were measured on


def anchors():
    """The four rows this figure was handed, reproduced digit for digit.

    They came measured on a 4000-direction Fibonacci sweep, and reproducing them
    pins down two things a coarser statement of the sets would leave loose.

    THE ORIENTATION MATTERS TO THE SWEPT COLUMNS AND TO NOTHING ELSE. The
    Fibonacci set is fixed in space, so turning an arrangement through it changes
    which samples land where the field is extreme -- that is how the handoff's
    tetrahedron is identified as the one at the CUBE'S CORNERS and not the one
    with a vertex up: at the cube's corners the sweep returns 1.1433 F and 6.10 %,
    and with a vertex up it returns 1.1504 F and 6.05 %. The exact columns are the
    same to every digit either way, which is the point of having both. The figure
    draws the vertex-up one, because in the scene these figures are about one ray
    is the FLOOR.

    AND TWO OF THE HANDOFF'S ENTRIES ARE CLOSED FORMS, NOT SWEEPS. The square
    pyramid's 1.0000 F and 1.7321 F are exactly F and exactly sqrt(3) F; the
    4000-direction sweep returns 1.0002 and 1.7157, because its strongest
    direction is the single corner (1,1,1)/sqrt(3) of a box and no sample lands
    within a degree of it. The bipyramid's row, by contrast, is swept throughout
    and reproduces to all four decimals.
    """
    V = fib(ANCHOR_N)
    print(f"\nthe rows this figure was handed, on their own orientations and their own "
          f"{ANCHOR_N}-\ndirection sweep, beside the closed forms:\n")
    print(f"{'':>30s}{'reaches':>10s}{'weakest':>21s}{'strongest':>21s}"
          f"{'at or above F':>16s}")
    print(f"{'':>30s}{'':>10s}{'sweep':>11s}{'exact':>10s}{'sweep':>11s}{'exact':>10s}"
          f"{'of the ball':>16s}")
    for name, U in (("4, tetrahedron 109.47 deg",
                     np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
                     / np.sqrt(3)),
                    ("5, square pyramid", np.vstack([UP, horizontals(4)])),
                    ("5, triangular bipyramid", np.vstack([UP, -UP, horizontals(3)])),
                    ("6, the three axes both ways", np.vstack([UP, -UP, horizontals(4)]))):
        m = reach(U, V)
        ins = m > EPS
        print(f"{name:>30s}{100 * ins.mean():8.2f} %{m[ins].min():9.4f} F"
              f"{weakest(U):8.4f} F{m[ins].max():9.4f} F{strongest(U):8.4f} F"
              f"{100 * (m >= F - 1e-12).mean():14.2f} %")


def named(sets):
    """The four panels and the two extra rows, with names short enough to tabulate."""
    return [(f"({t}) {len(U)} rays, {n}", U) for t, _, U, n in
            [(t, h, U, {3: "mutually square", 4: "the tetrahedron",
                        5: "the square pyramid", 6: "the three axes both ways"}[len(U)])
             for t, h, U, _ in sets]] + \
           [(f"    {n}", U) for n, U in extra_rows()]


def numbers(U, V):
    """Everything a caption quotes, exact, with the sweep beside it."""
    m = reach(U, V)
    ins = m > EPS
    return dict(cover=covers(U), cover_sweep=100 * ins.mean(),
                lo=weakest(U), lo_sweep=m[ins].min(),
                hi=strongest(U), hi_sweep=m[ins].max(),
                ball=share(U), ball_sweep=100 * (m >= F - 1e-12).mean(),
                cone=100 * share(U) / covers(U),
                wide=pair_angles(U)[-1], rays=len(U))


def main(out=figure_path('more_pushes.png')):
    global VMAX
    t0 = time.time()
    sets = cases()
    # the ramp's far end is the largest magnitude the figure contains, read off
    # the closed form so the key cannot clip the paint or leave slack above it.
    # It comes out sqrt(3) F, and three of the four panels attain it; the
    # tetrahedron's own ceiling is 2/sqrt(3) F = 1.155 F, less than a fifth of the
    # way up the blue. (three_pushes.png's top is 2.62 F, the middle of its 50 deg
    # patch. The two figures share the ramp and the break at F and differ in this
    # one number, because neither may clip its own strongest point)
    VMAX = max(strongest(U) for _, _, U, _ in sets)

    pose(sets)
    anchors()
    V = fib(N)
    print(f"\na uniform sweep of {N} directions -- NOT the ball's own cells, whose "
          f"areas go as sin(theta)\n")

    # the LP is the definition and `reach` is what paints 20480 cells four times
    # over; this is the whole licence for the substitution and it is re-earned on
    # every run
    # compared only where the comparison means anything: STRICTLY INSIDE the cone.
    # On a direction grazing the boundary the closed form is h(a)/(a.v) with both
    # terms going to zero together, and for a set whose coordinates are not exact
    # -- every clustered set here -- the rounding in `h` no longer cancels the
    # rounding in `a.v`. The quotient then wanders by ~1e-6 on directions the LP
    # simply calls unreachable, which is a floating-point artefact of the ratio
    # and not a disagreement about the geometry. Checked on 400 random interior
    # directions of the clustered triple, the two agree to 2.2e-16. So: interior
    # only, and relative, since the reach itself runs from F to 3.5 F
    print(f"the LP against the closed form, on {N_LP} directions of each set:\n")
    L = fib(N_LP)
    for name, U in named(sets):
        lp, cf = reach_lp(U, L), reach(U, L)
        inside = lp > 1e-6
        d = (np.abs(lp[inside] - cf[inside]) / lp[inside]).max() if inside.any() else 0.0
        print(f"{name:>38s}   max relative |LP - closed form| = {d:.2e}"
              f"   on {inside.sum()} interior of {len(L)}")
        assert d < 1e-9, "the closed form and the LP disagree inside the cone"

    print(f"\n{'':>38s}{'widest':>9s}{'reaches':>19s}{'weakest':>10s}{'strongest':>11s}"
          f"{'at or above F':>26s}")
    print(f"{'':>38s}{'pair':>9s}{'exact':>10s}{'sweep':>9s}{'':>10s}{'':>11s}"
          f"{'of the ball':>16s}{'of the cone':>15s}")
    for name, U in named(sets):
        n = numbers(U, V)
        print(f"{name:>38s}{n['wide']:8.2f}°{n['cover']:9.3f}%{n['cover_sweep']:8.3f}%"
              f"{n['lo']:10.4f}{n['hi']:11.4f}{n['ball']:15.3f}%{n['cone']:14.3f}%")
    print("\n(the 'of the cone' column is the share of what the arrangement REACHES; the\n"
          " 'of the ball' column is the share of all directions, and they differ only\n"
          " where the cone is not the whole ball. The sweep reproduces every exact\n"
          " share to three decimals -- 5.992 against 5.988 for the tetrahedron at a\n"
          " million directions -- which is what says the inclusion-exclusion was right)")

    # THE FINDING. Two arrangements with the same widest pair and opposite verdicts,
    # and one with a narrower widest pair than either and the worse verdict of all
    print("\nwhy the pairwise angle stops being the criterion past three rays:\n")
    print(f"{'':>38s}{'rays':>6s}{'widest pair':>14s}{'weakest':>10s}{'guarantee':>12s}")
    for name, U in named(sets):
        w = weakest(U)
        print(f"{name:>38s}{len(U):6d}{pair_angles(U)[-1]:13.2f}°{w:10.4f}"
              f"{'holds' if w >= F - 1e-9 else 'LOST':>12s}")
    print("\n109.47° loses it and 180° keeps it, and the bipyramid keeps the SAME widest\n"
          " pair as the square pyramid and loses it anyway. Whatever the criterion is,\n"
          " it is not an angle between two rays")

    print("\nthe criterion that does work: for a pair {u_i, u_j}, a = unit(u_i x u_j) is\n"
          " the direction both are blind to, and h(a) is what the OTHER rays can put\n"
          " there. The binding pair of each set, and what was left over:\n")
    print(f"{'':>38s}{'blind pair':>13s}{'rays left':>11s}{'h(a)/F':>10s}"
          f"{'h(a)/|P(a)|':>14s}")
    for name, U in named(sets):
        i, j, a, h, bound = min(worst_pair(U), key=lambda r: r[4])
        print(f"{name:>38s}{'u_' + str(i + 1) + ', u_' + str(j + 1):>13s}"
              f"{len(U) - 2:11d}{h:10.4f}{bound:14.4f}")
    print("\nwith THREE rays exactly one is left over and h(a) = F cos(beta) <= F, equal\n"
          " only when that ray IS a -- the mutually square arrangement. With four the\n"
          " sum has two terms and 0.8165 F is all they manage; with five the square\n"
          " pyramid's blind direction is itself a ray and the sum is exactly F")

    # why the sweep and not a mesh, measured rather than asserted -- three_pushes.py
    # makes the same measurement on its own three cones and this repeats it here
    th = (np.arange(48) + 0.5) * np.pi / 48
    ph = (np.arange(96) + 0.5) * 2 * np.pi / 96
    T, P = np.meshgrid(th, ph, indexing="ij")
    LATLON = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P),
                       np.cos(T)], axis=-1).reshape(-1, 3)
    _, GEO = shell()
    print("\nwhy the sweep and not a mesh -- the same four cones, counted four ways:\n")
    print(f"{'':>10s}{'exact':>10s}{'sweep':>10s}{'lat-lon cells':>16s}{'geodesic cells':>16s}")
    for tag, _, U, _ in sets:
        print(f"{'(' + tag + ')':>10s}{covers(U):10.3f}{100 * (reach(U, V) > EPS).mean():10.3f}"
              f"{100 * (reach(U, LATLON) > EPS).mean():16.3f}"
              f"{100 * (reach(U, GEO) > EPS).mean():16.3f}")
    print("\n(the lat-lon column is exact on (a)'s octant -- cut by the equator and by two\n"
          " meridians, so the sin(theta) over-weighting cancels across its own boundaries\n"
          " -- and it is exact on the three whose cone is a hemisphere or the whole ball\n"
          " for the same kind of reason. It is the patches that hug a pole it ruins, and\n"
          " three_pushes.py has one: at 50° it reports two thirds more patch than exists)")

    wrong_ways(sets)

    # ---------------------------------------------------------------- the figure
    fig = plt.figure(figsize=(17.4, 7.8), dpi=210, facecolor=PAPER)
    print()
    for k, (tag, head, U, said) in enumerate(sets):
        n = numbers(U, V)
        ax = fig.add_subplot(1, 4, k + 1, projection="3d", computed_zorder=False)
        ax.set_facecolor(PAPER)
        ball(ax, U)
        x = (2 * k + 1) / 8
        fig.text(x, 0.975, f"({tag})  {head}", ha="center", va="center", color=INK,
                 fontsize=14.5, fontweight="bold")
        for j, line in enumerate(said):
            fig.text(x, 0.944 - 0.027 * j, line, ha="center", va="center", color=MUTED,
                     fontsize=10.4)
        # the numbers are formatted from the same values the paint is, so the
        # words under a picture cannot drift from the colours in it. The two round
        # fractions are named as fractions as well as percentages, and only where
        # they are exactly true: covers() lands on 12.5 and on 50 to machine
        # precision and no other panel lands on a round fraction at all
        # 12.5 and 50 land on those values to machine precision and no other
        # panel lands on a round fraction at all. 100 % is left unglossed: it
        # needs no help, and the gloss it had ran off the right of panel (d)
        gloss = {12.5: "  =  exactly one eighth", 50.0: "  =  exactly half"}
        tail = next((g for v, g in gloss.items() if abs(n['cover'] - v) < 1e-9), "")
        fig.text(x, 0.221, f"reaches {n['cover']:.3f} % of the ball" + tail, ha="center",
                 va="center", color=INK, fontsize=11.4)
        fig.text(x, 0.190, f"weakest {n['lo']:.4f} F   ·   strongest {n['hi']:.4f} F",
                 ha="center", va="center", color=INK, fontsize=11.4)
        ok = n['cone'] > 99.999
        fig.text(x, 0.159, (f"every direction it reaches takes F or more"
                            if ok else
                            f"only {n['cone']:.1f} % of the cone still takes F or more"),
                 ha="center", va="center", color=MUTED if ok else "#a33224",
                 fontsize=10.8, fontweight="normal" if ok else "bold")
        print(f"({tag}) {len(U)} rays, widest pair {n['wide']:6.2f}°   "
              f"covers {n['cover']:7.3f} %   weakest {n['lo']:.4f} F   "
              f"strongest {n['hi']:.4f} F   {n['cone']:6.2f} % of the cone at or above F")
    fig.subplots_adjust(left=0.0, right=1.0, top=0.912, bottom=0.212, wspace=0.0)
    key(fig.add_axes([0.437, 0.051, 0.126, 0.018]))
    band(fig, [(0.652, 0.108, onpaper(0.90), "under one support's strength"),
               (0.652, 0.076, onpaper(1.45), "one support's strength or more"),
               (0.652, 0.044, onpaper(None), "left bare: they cannot push that way at all")],
         (0.050, 0.108, 0.076, 0.052))
    fig.savefig(out, facecolor=PAPER)
    plt.close(fig)
    shape = trim(out)
    print(f"\nwrote {out}  {shape[1]} x {shape[0]}   ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
