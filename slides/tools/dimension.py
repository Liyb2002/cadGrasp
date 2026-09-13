"""Why the first support looks like it does nothing.

The same statics as cover.py, stripped to the one thing that settles the answer.
A contact contributes only the direction it pushes, the floor is a free
generator pointing straight up, and a disturbance `d` of one body weight is
survived exactly when

    T = up - d          lies in the cone the contacts span

With the floor alone that cone is a single ray, so the set of survivable
disturbances is a single direction -- one arrow, nothing more. Every support
adds a generator, and every generator adds a DIMENSION to that set: the ray
opens into a wedge and the set becomes an arc, the wedge opens into a solid cone
and the arc becomes a patch of the sphere. The count of supports is really a
count of dimensions, and a table of percentages hides exactly that: what the
first support buys is an arc, an arc has no area, and so it is entered as 0 %.

The third panel only exists in space, so the solid figure carries the argument
and the plane figure is kept beside it as the same statics read at a glance.

Nothing here depends on WHERE a push is applied: the model is translation only,
so a force enters as a direction and nothing else. The surviving pushes are
therefore spread over a face rather than pinned to a vertex.

The body is a flat plate LYING ON ITS BROAD FACE, stopped from sliding by a
chock against each of two vertical sides. Earlier versions balanced it on a
single corner, which is dropped: balance puts the centre of mass over the
contact, so a plate that stands unaided stands on end and can never be seen
whole. Lying down costs the statics nothing -- a face on the floor is the same
one upward generator a corner was -- and it makes all three answers exact. The
pushes are horizontal, so a disturbance survives exactly when it leans away from
each of them: one direction, then a half great circle, then a quarter of the
sphere. The solid figure is written out from each of the four isometric azimuths
in turn, one scene seen from its four sides, so that the view can be chosen by
looking rather than by argument.

Each panel also carries a ROW of arrows across the plate itself, and they are
the other quantity: not what the supports survive but what they can PUSH WITH.
Every support is capped at F, so a direction in the cone they span reaches
F / max_i c_i, and each arrow is drawn with its length AND its line width in
proportion to that -- the encoding `dimension_pairs.png` uses, off the same
function. The row says the thing a reader does not expect: the weakest places in
the cone are the PUSH DIRECTIONS THEMSELVES, at exactly F, and everything between
them is stronger, sqrt(2) F between any two and sqrt(3) F down the middle. What
is sampled is the cone's rim, because the camera looks down its axis and the
strongest direction of all is the one aimed at the reader.

The scene is only half of it. An orthographic view turns a direction's
inclination and its azimuth into one drawn angle, so arrows standing on the body
cannot say which way they point -- a push 16.6 deg BELOW horizontal comes out
drawn ABOVE one exactly in it. The ball figure makes the same argument from the
other end and with nothing projected twice: it draws the CONE THE CONTACTS SPAN,
which is the region between the push arrows themselves. One arrow reaches one
direction, two reach the arc between them, three reach the patch they close on --
point, arc, patch, drawn as what they are. It is the same counting as the scene
and not the same set, so the two figures quote different percentages by right:
the cone is an eighth of the ball, the disturbances it survives a quarter.

    python slides/tools/dimension.py     ->  slides/tools/figures/dimension.png       three panels, in space
                                      slides/tools/figures/dimension_ball.png  the three sets themselves
                                      slides/tools/figures/dimension_pairs.png 180, 175 and 90 deg apart
                                      slides/tools/figures/dimension_iso_a..d  the scene, four sides
                                      slides/tools/figures/dimension_flat.png  two of them, in the plane
                                      slides/tools/figures/normals.png
"""
from __future__ import annotations

from common import figure_path

import itertools

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon  # noqa: E402
from mpl_toolkits.mplot3d import proj3d                  # noqa: E402
from mpl_toolkits.mplot3d.art3d import (Line3DCollection,  # noqa: E402
                                        Poly3DCollection)

INK, MUTED, PAPER = "#1b1b1a", "#6b6b66", "#ffffff"
BLUE, ORANGE = "#2563EB", "#E08A24"
BODY, SUPPORT, GROUND = "#dfe3df", "#c9ccc6", "#8d8f89"
UP = np.array([0.0, 1.0])

HALF = np.array([0.75, 0.40])                            # half length, half height


def block():
    """A rectangle tipped until its lowest corner sits under its own centre.

    Balanced on that corner it is in equilibrium and nothing is touching it but
    the floor, which is the whole point: the configuration is legitimate, and
    still only one disturbance in the plane leaves it standing.
    """
    a, b = HALF
    phi = np.arctan2(a, b)                               # puts the corner under the centre
    R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    local = np.array([[-a, -b], [a, -b], [a, b], [-a, b]])
    corners = local @ R.T
    centre = -corners[0]                                 # lift the resting corner to z = 0
    return corners + centre, centre, R


def sweep(dim, n):
    """A dense, near-uniform set of unit directions to put to the test.

    The circle can be walked at even spacing and lands exactly on straight down;
    the sphere cannot be, so this is the Fibonacci spiral, which spreads evenly
    and, unlike a lat-lon grid, has no pole where the samples pile up.
    """
    if dim == 2:
        ang = np.linspace(-np.pi, np.pi, n, endpoint=False)
        return np.stack([np.cos(ang), np.sin(ang)], axis=1)
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r, t = np.sqrt(1 - z * z), np.pi * (1 + 5 ** 0.5) * i
    return np.stack([r * np.cos(t), r * np.sin(t), z], axis=1)


def gap(G, T):
    """How far each demanded total force is from anything the contacts can supply.

    Zero means it is a non-negative combination of the generators. With at most
    three of them the active set is one of eight subsets, so the non-negative
    least squares projection is the best of eight ordinary solves -- a closed
    form, where a linear program would be a solver aimed at a problem this size
    already answers.
    """
    out = np.linalg.norm(T, axis=1)                      # supply nothing at all
    for r in range(1, G.shape[1] + 1):
        for S in itertools.combinations(range(G.shape[1]), r):
            lam = np.linalg.lstsq(G[:, S], T.T, rcond=None)[0]
            res = np.linalg.norm(T.T - G[:, S] @ lam, axis=0)
            out = np.where((lam >= -1e-12).all(axis=0), np.minimum(out, res), out)
    return out


def survivable(gens, n, tol=1e-9):
    """Which unit disturbances the floor and these supports can answer.

    Same test in both dimensions: the contacts must produce `up - d` with no
    pulling, so it is a non-negative combination of the generators and nothing
    else. On the circle the default tolerance is floating-point slack and no
    more, because the sweep lands on straight down exactly. No finite grid on
    the sphere lands on a set of measure zero, so there `tol` is set to the grid
    spacing: a direction counts when the set passes within one step of it, which
    is what lets a point, an arc and a patch be counted on the same footing.
    """
    d = sweep(len(gens[0]), n)
    T = np.eye(len(gens[0]))[-1] - d
    return d, gap(np.array(gens, float).T, T) <= tol


def spans(ang, ok, floor_deg=2.0):
    """The admissible angles as runs, with the hairline ones thrown away.

    Straight up always scrapes through: a disturbance of exactly one body weight
    pulling straight up leaves the contacts nothing to do. It is a single angle,
    it survives on an equality, and drawing it puts a lone arrow through the
    middle of the block that reads as a whole capability. Anything thinner than
    a couple of degrees is that kind of artefact, not a range.
    """
    out, i, n, step = [], 0, len(ok), ang[1] - ang[0]
    if ok.all():
        return [(-np.pi, np.pi)]
    while i < n:
        if ok[i] and not ok[i - 1]:
            j = i
            while ok[(j + 1) % n] and (j + 1) % n != i:
                j += 1
            if (j - i) * step >= np.deg2rad(floor_deg):
                out.append((ang[i], ang[i] + (j - i) * step))
        i += 1
    return out


def com(ax, c, r=0.075):
    """The usual centre-of-mass roundel."""
    ax.add_patch(plt.Circle(c, r, facecolor="white", edgecolor=INK, lw=1.3, zorder=8))
    for a0 in (90, 270):
        ax.add_patch(matplotlib.patches.Wedge(c, r, a0, a0 + 90, facecolor=INK,
                                              edgecolor="none", zorder=9))


def generator(ax, at, u, ln=0.42):
    """One contact, drawn as the only thing it contributes: a push direction.

    Same colour and weight everywhere so the panels can be read by counting
    arrows -- one generator on the left, two on the right.
    """
    ax.annotate("", at, at - ln * u, zorder=12,
                arrowprops=dict(arrowstyle="-|>,head_width=.22,head_length=.42",
                                color=ORANGE, lw=3.0))


def arrow(ax, at, v, ln):
    """A disturbance that gets survived, landing on the face."""
    ax.annotate("", at, at - ln * v, zorder=11,
                arrowprops=dict(arrowstyle="-|>,head_width=.20,head_length=.40",
                                color=BLUE, lw=2.4))


def along(edge, n, lo=0.14, hi=0.94):
    """Anchor points spread over a face, kept off its two corners.

    The low end is inset further because the shallowest arrow in the fan comes in
    from below and would otherwise reach the floor.
    """
    e0, e1 = edge
    s = np.linspace(lo, hi, n)[:, None]
    return e0 + s * (e1 - e0)


def draw(ax, corners, centre, edge, support, gens, n=9, ln=0.62):
    ax.add_patch(Polygon([[-3, -0.55], [3, -0.55], [3, 0], [-3, 0]],
                         facecolor=GROUND, edgecolor="none", alpha=.28, zorder=0))
    ax.plot([-3, 3], [0, 0], color=MUTED, lw=1.6, zorder=1)
    if support is not None:
        pad, at, u = support
        ax.add_patch(Polygon(pad, facecolor=SUPPORT, edgecolor=MUTED, lw=1.2, zorder=3))
        generator(ax, at, u)
    ax.add_patch(Polygon(corners, facecolor=BODY, edgecolor=INK, lw=1.8, zorder=4))
    com(ax, centre)
    generator(ax, corners[0], UP)                        # the floor, always free

    d, ok = survivable(gens, 1440)
    runs = spans(np.arctan2(d[:, 1], d[:, 0]), ok)
    width = sum(b - a for a, b in runs)
    if not runs:                                         # a set of measure zero
        # no run to read a direction off, so read it off the survivors: they are
        # isolated samples, and dropping the straight-up equality `spans` throws
        # away leaves the one direction the floor actually answers
        v = d[ok & (d @ UP < 1 - 1e-6)].mean(axis=0)
        # one direction is one arrow -- copies along the face would inflate a
        # 0-dimensional set into a row of capabilities. it takes the anchor the
        # other panel's straight-down arrow uses, so that fan is this arrow opened
        arrow(ax, along(edge, n)[-1], v / np.linalg.norm(v), ln)
    else:
        for a, b in runs:
            # the arc runs b..a up the face, not a..b: that way the straight-down
            # push sits at the top in both panels and the tails never cross
            t = np.linspace(b, a, 120)[:, None]
            an = along(edge, 120)
            # sweep the shading along the face with the arrows -- a wedge pinned
            # to a vertex would claim the push has to land there
            ax.add_patch(Polygon(np.vstack([an, (an - ln * np.hstack([np.cos(t),
                                                                     np.sin(t)]))[::-1]]),
                                 facecolor=BLUE, alpha=.16, edgecolor="none", zorder=5))
            for p, k in zip(along(edge, n), np.linspace(b, a, n)):
                arrow(ax, p, np.array([np.cos(k), np.sin(k)]), ln)
    ax.set_xlim(-1.42, 1.42)
    ax.set_ylim(-0.55, 2.45)                             # the tallest arrow ends at 2.24
    ax.set_aspect("equal")
    ax.set_axis_off()
    return np.degrees(width)


def rounded(corners, k, r, n=90):
    """Replace corner k with an arc, and hand back its centre of curvature.

    A right-angled corner puts the tangent points a distance r back along each
    edge and the centre r*sqrt(2) along the bisector, which is all this needs.
    """
    V, A, B = corners[k], corners[(k - 1) % len(corners)], corners[(k + 1) % len(corners)]
    e1, e2 = (A - V) / np.linalg.norm(A - V), (B - V) / np.linalg.norm(B - V)
    bis = (e1 + e2) / np.linalg.norm(e1 + e2)
    half = np.arccos(np.clip(e1 @ e2, -1, 1)) / 2
    hub = V + bis * r / np.sin(half)
    t1, t2 = V + e1 * r / np.tan(half), V + e2 * r / np.tan(half)
    a1 = np.arctan2(*(t1 - hub)[::-1])
    a2 = np.arctan2(*(t2 - hub)[::-1])
    if (a2 - a1) % (2 * np.pi) > np.pi:
        a1, a2 = a2, a1
    t = np.linspace(a1, a1 + (a2 - a1) % (2 * np.pi), n)
    arc = hub + r * np.stack([np.cos(t), np.sin(t)], axis=1)
    shape = np.vstack([corners[:k], arc, corners[k + 1:]])
    return shape, arc, hub


def normals(corners, centre, R, out, r=0.24, n=7):
    """What a frictionless contact force actually points along.

    The question this answers is whether the pushes on a tilted face are parallel
    or aimed at the centre of mass. Neither is the rule: a contact pushes along
    the LOCAL normal, and that is a flat face's one direction repeated, or a
    fillet's directions fanning into its centre of curvature. The mass centre is
    drawn in both panels precisely so it can be seen that nothing points at it.

    The fillet is drawn far fatter than the real one -- A1-f's is about 3 mm on a
    190 mm face -- because at true scale the fan would be a single pixel.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 6.2), dpi=210, facecolor=PAPER)
    inward = -(R @ np.array([0.0, 1.0]))                 # into the upper-left face
    for ax, curved in zip(axes, (False, True)):
        ax.set_facecolor(PAPER)
        ax.add_patch(Polygon([[-3, -0.55], [3, -0.55], [3, 0], [-3, 0]],
                             facecolor=GROUND, edgecolor="none", alpha=.28, zorder=0))
        ax.plot([-3, 3], [0, 0], color=MUTED, lw=1.6, zorder=1)
        if curved:
            shape, arc, hub = rounded(corners, 2, r)
            ax.add_patch(Polygon(shape, facecolor=BODY, edgecolor=INK, lw=1.8, zorder=4))
            for p in arc[np.linspace(6, len(arc) - 7, n).astype(int)]:
                generator(ax, p, (hub - p) / np.linalg.norm(hub - p))
            ax.scatter(*hub, s=46, facecolor="white", edgecolor=MUTED, lw=1.4, zorder=10)
        else:
            ax.add_patch(Polygon(corners, facecolor=BODY, edgecolor=INK, lw=1.8, zorder=4))
            for p in along((corners[3], corners[2]), n, .10, .90):
                generator(ax, p, inward)
        com(ax, centre)
        ax.set_xlim(-1.42, 1.42)
        ax.set_ylim(-0.55, 2.45)
        ax.set_aspect("equal")
        ax.set_axis_off()
    fig.subplots_adjust(0.01, 0.02, 0.99, 0.98, 0.02)
    fig.savefig(out, facecolor=PAPER)
    plt.close(fig)
    return inward


# ----------------------------------------------------------------- and in space

# a flat plate, 5.43 : 5.00 : 1, SET DOWN ON ITS BROAD FACE. Every earlier pass
# balanced it on one corner, and that is what had to go: balance means the
# centre of mass sits over the contact, so a plate that stands unaided stands on
# END, and the thinner it is the straighter it stands -- the tilt is
# arccos(h_2/|h|), which runs to 90 deg as h_2 runs to 0. Resting on an edge
# does not escape it either (68.5 and 66.8 deg for the two bottom edges, against
# 82.3 for the corner). So there is exactly one pose that shows a flat plate
# whole, and this is it. The reader now sees the broad face from above and the
# two thin ones edge on, which is the shape read straight off the page rather
# than inferred, and the body lies 2.07 wide by 1.28 tall there instead of 1.64
# by 1.97. The statics do not notice: a contact contributes its push direction
# and nothing else, so a face resting on the floor is the same one upward
# generator a corner was
HALF3 = np.array([0.76, 0.70, 0.14])                     # half extents of the plate
UP3 = np.array([0.0, 0.0, 1.0])
# the axis it lies on -- the thinnest, since the broad face is the one on the
# floor. Everything else about the pose follows from this single index
LEVEL = int(np.argmin(HALF3))
# laying the plate down retired both of the elevation ceilings this figure used
# to answer to. They existed to protect one point: the corner the body balanced
# on had to stay visible, and had to stay the lowest thing on the page. There is
# no such corner now -- the contact is a whole face, and every part of it that
# matters is on the silhouette at any elevation -- so nothing above caps the
# camera and nothing below it either.
#
# What is left is a plain trade, and 36 is where it was settled by looking. Too
# low and the broad face closes up: it is the face the argument is read off,
# and the page area it keeps goes as sin(elev). Too high and the two thin faces
# close up instead, until the plate reads as a rectangle painted on the ground
# rather than a solid standing on it, and the chocks lose the height that tells
# them apart from their own shadows. 36 is very nearly the balance point: the
# broad face keeps sin(elev) = 58.8 % of its true area on the page and each thin
# face cos(elev) cos45 = 57.2 % of its own, so no face is favoured over another
ELEV, AZIM = 36.0, -45.0
AZIMS = (-45.0, 45.0, 135.0, 225.0)                      # the four isometric corners
BEST = -45.0                                             # the one dimension.png takes


def look(azim):
    """Stand the camera at one of the four isometric corners of the scene.

    Everything that culls a face, seats a pad or sizes the floor asks the camera
    where it is, so the whole figure follows from this one vector and the four
    variants differ in nothing else.
    """
    global AZIM, VIEW
    AZIM = float(azim)
    VIEW = np.array([np.cos(np.radians(ELEV)) * np.cos(np.radians(AZIM)),
                     np.cos(np.radians(ELEV)) * np.sin(np.radians(AZIM)),
                     np.sin(np.radians(ELEV))])


look(AZIM)
# the spin about the vertical changes no statics whatever, so it is free, and
# with the plate lying down what it buys is the whole 3-D read. Its four sides
# stand vertical, so a spin that aims one of them at the camera hands the reader
# a rectangle head on: the plate flattens into a shape on the page and the floor
# under it stops receding. Square to the world axes, and with the camera at any
# of the four isometric corners, both visible sides come in at 45 deg and the
# body projects to a hexagon that can only be a solid. So the spin is zero and
# the cameras do the work. It cannot be spent on the light as it used to be --
# see LAZIM, which is spent on that instead
SPIN = 0.0
# both supports press a face that STANDS VERTICAL, which is every axis except
# the one the plate lies on. There is no choosing here and no taste in it: a
# support can only push along the outward normal of the face it presses, the
# broad face's normal is straight up and the floor already supplies that
# direction, and nothing can press the underside at all
PROPPED = tuple(k for k in range(3) if k != LEVEL)
# a chock: how much of the side it covers, how far out it reaches, how high it
# stands. The crest is 2.0 times the plate's own thickness, which is what lets
# it read over the top of a plate it stands behind
SEAT = (0.46, 0.46, 0.56)                                # size, depth, crest
# the plane figure fills the body and the floor at almost the same lightness,
# which it gets away with because there the body has a hard outline and the
# floor is a band behind it. here they meet along a silhouette the reader has to
# read as a solid standing on a plane, so the two are pulled well apart: a mid
# grey plate on a floor barely off the paper. kept separate from BODY/GROUND so
# the two plane figures are left exactly as they were, and SUPPORT3 is forked off
# SUPPORT for the same reason
BODY3, SUPPORT3, GROUND3 = "#b0b5ae", "#bfc5c8", "#f1f2ed"
# outlines go the same way as the fills: heaviest and blackest on the body,
# lighter and cooler on what stands behind it, lightest on the floor itself
SUPPORT_INK, GROUND_INK = "#79838a", "#a3a9a4"
DUSK = "#49525c"                                         # what the light misses, cool
# one light for the whole figure. The Lambert term is measured against it and
# the pools on the floor are the same vertices dropped along it, so the two cues
# cannot disagree about where it stands: over the reader's left shoulder. The
# light does not turn with the camera: the four variants are four views of one
# scene, so the shadow falls the same way in all of them and only the first has
# the light over the reader's shoulder rather than off to one side.
#
# It used to have to be STEEP as well, because the floor is sized to hold
# whatever is thrown onto it and every unit that quad grows comes off the size
# of the body on the page. That constraint has lifted: a plate lying down throws
# a shadow barely longer than itself, and the floor is now sized by the arrow
# tails instead, which reach further than any shadow does at any light elevation
# from 45 up. Folding the shadow in costs 1.01 % of the widest side and 2.07 %
# of the area, and that figure does not move when the light is lowered.
#
# So the light can be spent on the shading, and the azimuth is what the spin
# used to buy. Lying down, the body shows the camera one vertical face at
# azimuth 0, one at -90 and the broad face straight up, and a vertical face
# takes cos(LELEV) cos(its azimuth - LAZIM) of the key while the broad one is
# fixed at amb + key sin(LELEV). Setting the two verticals a step apart and the
# broad face the same step above them solves out at tan(LELEV) = 2, or 63.4 deg,
# with the light square to one of them; 64 and -95 sit a whisker off that, and
# the 5 deg of slack past square is deliberate -- the far face is then clipped
# to the ambient by a margin instead of sitting exactly on the knife edge. The
# three come back at 0.800, 0.975 and 1.160 -- see SUN
LAZIM, LELEV = -95.0, 64.0
LIGHT = np.array([np.cos(np.radians(LELEV)) * np.cos(np.radians(LAZIM)),
                  np.cos(np.radians(LELEV)) * np.sin(np.radians(LAZIM)),
                  np.sin(np.radians(LELEV))])
# ambient, key, bounce. The ambient is set so the darkest face lands on the
# value the darkest face has always had, and the range is widened upwards only.
# The three visible faces come
# back at 0.800, 0.975 and 1.160 -- gaps of 0.175 and 0.185, the widest and the
# most even this figure has had (0.157 and 0.177 when the plate stood on a
# corner, 0.095 and 0.195 when it was thinned and still standing). The 0.175 is
# the number that matters, being the narrowest of the two and so the one that
# decides whether the reader can tell the faces apart, and the darkest face sits
# AT the ambient because it stands edge on to the light and takes no key at all.
#
# The bounce term has gone inert and that is worth saying rather than leaving to
# be discovered. It was put in because the only faces a key above a floor cannot
# reach are the ones looking DOWN, and blue arrows used to land on a downward
# face of a body balanced on its corner. Lying down, the body turns no face
# downwards that the reader can see, so `shaded`'s bounce multiplies zero on
# everything drawn. It is kept because it costs nothing and because the term is
# right; it is no longer load-bearing
SUN = 0.80, 0.40, 0.25
# the chocks stand behind the plate in every view, so they
# take the same light over a narrower range and a cooler, paler colour. Less
# contrast within themselves and less against the floor is the whole of aerial
# perspective
HAZE = 0.91, 0.17, 0.09
# a shadow is drawn as nested copies of itself, rim to core, each one a shade
# darker: a penumbra with no blur to do it. They shrink INWARDS -- a real
# penumbra spreads out, and out is where the floor has to grow to follow it.
# Enough steps that the steps are not visible as steps
SOFT, CORE, DEEP = 14, 0.45, 0.19
# contact is the one thing a cast shadow cannot show, because its own edge runs
# through the contact. So it gets a second ramp, from this radius down to that
# one: tighter and darker than the shadow around it, and small enough to stay a
# contact rather than turn into a second shadow
BED, BEDDED = (0.18, 0.035), 0.38
FILL = 0.62                                              # how much of a face anchors use
N = 40000                                                # directions swept over the sphere
# ------- the row of forces the contacts can PRODUCE, drawn on the plate itself
#
# The blue arrows say what the supports SURVIVE. They say nothing about what the
# supports can push with, and those are two different quantities: one is a set of
# disturbances, the other a set of forces. This row is the second. Each support
# is capped at F, a direction in the cone is v = sum c_i u_i with every c_i >= 0,
# and scaling the combination scales every coefficient together -- so the cap
# binds on the largest and the direction reaches m(v) = F / max_i c_i. It is
# `strongest`, the same function the pairs figure's fan is drawn from, and the
# encoding is that figure's too: LENGTH and LINE WIDTH both in proportion to m,
# so the two pictures speak one language and a reader who has learnt the fan can
# read this row without being told again.
#
# All three generators here stand mutually square, so c_i = v . u_i and the
# answers are exact: F straight up with the floor alone; F at each end of the
# quarter-plane wedge and sqrt(2) F across it; F at each of the three pushes,
# sqrt(2) F between any two of them and sqrt(3) F down the cone's own axis. The
# shape of that is the surprise and it is what the row is drawn to show -- THE
# PUSH DIRECTIONS ARE THE WEAKEST PLACES IN THE CONE, and everything between them
# is stronger, which is the opposite of where a reader looks for a weak spot.
#
# What is sampled is the cone's RIM -- its generators and the arcs joining them --
# and not its inside, and the reason is the camera rather than taste. The scene
# looks down elev 36 azim -45 and the cone's axis is elev 35.26 azim -45, so the
# sqrt(3) F direction is 0.7 deg off the view axis and keeps 1.3 % of its length
# on the page: it cannot be drawn here at any length, in any style. Worse, honest
# projection INVERTS the whole message -- measured over the octant, F straight up
# draws 0.809 on the page and sqrt(2) F draws 0.809 as well, dead level, and the
# sqrt(3) F peak draws 0.022. The rim is the part of the cone that stands clear of
# the camera: the worst foreshortening on it is 0.572, above the 0.55 floor, and
# every drawn arrow keeps a length worth reading. The peak is printed by the run
# and painted whole on `dimension_ball.png`, which is where interior directions
# are answered.
ROW_PITCH = 45.0                     # deg between samples along the rim
# so a 90 deg edge gives its two ends and its middle: F, sqrt(2) F, F. Three per
# edge is as coarse as the rim can be sampled and still carry the swell, and
# coarse is what this figure can afford -- panel 3 already holds 21 blue arrows
# and two chocks, and a 22.5 deg pitch would put 12 more arrows on it rather than
# 6. The three ends are not optional at any pitch: they are the push directions,
# and they are the point
ROW = False        # draw the row of producible forces on the body? see `panel`
VARIANTS = False   # also write the four dimension_iso_* views? see `main`
ROW_LN, ROW_LW, ROW_THIN = 0.28, 1.9, 1.0
# ROW_LN is a PAGE length, undone through `onpage` exactly as the ball's fan is,
# because a length that carries a number has to be read off the page and the
# projection scales it by sqrt(1 - (v.VIEW)^2) -- 0.809 at the floor's push and
# 0.572 at the middle of an edge, which would draw sqrt(2) F shorter than F. The
# TAIL is left where it is: it is a position, and positions are what this figure
# projects honestly.
#
# 0.28 is a seventh of the plate's page width, and it is a ceiling and not a
# taste: the row has to fit between two chock generators that are 2.06 apart,
# and every unit it grows brings its outermost arrow down along one of them (see
# `crowding`). It is also SHORTER than the 0.372 a generator arrow is drawn at,
# although both stand for F, and that is the one thing about this row that has to
# be taken on trust rather than read off. The generator arrows are direction
# marks of one fixed length, the row's are magnitudes; what anchors the scale
# instead is panel 1, where the single arrow IS F, and the same arrow stands in
# every panel because `up` is in every cone
ROW_HEAD, ROW_TIP = 11.0, 5.0        # arrowhead in mutation points, at F and at least
ROW_DOT = 22.0                       # the dot under each tail, in points squared
ROW_STEP, ROW_AT, ROW_OFF = 0.30, 0.02, 0.08    # between tails, and where the row sits
# fixed SPACING, not a fixed span, for the reason `anchors` spreads the blue
# arrows the way it does: with the pitch fixed the row GROWS as supports are
# added -- one arrow, three, six -- and what the reader sees change is the set
# rather than the spacing.
#
# The row lies along the widest chord of the top face, and the two offsets are
# small nudges along and across it. They are not free and they are not eyeballed:
# ROW_STEP, ROW_LN, ROW_AT and ROW_OFF together were picked by sweeping this
# whole face and maximising the clearance `crowding` measures, subject to the row
# keeping 0.20 from any other ORANGE arrow. Centred on the face, which is what
# they would be at zero, the row clears 0.028 -- and what it is 0.028 from is the
# ROUNDEL, in panel 1, where the row is a single arrow standing at the middle of
# the plate and the centre-of-mass mark is drawn at the middle of the plate. Move
# it off along the chord alone and panel 2's strongest arrow comes head to head
# with a blue one at 0.021 instead. Both together, at 0.02 and 0.08, give 0.058
# from anything at all and 0.252 from the chocks' own arrows
ROW_SEEN = 0.55                      # the same guard `onpage` takes on the ball


def slab(psi=SPIN):
    """The plate set down on its broad face: no tilt at all, only a spin.

    Every earlier version of this function tipped the box until its lowest
    corner came under its own centre. That is the pose this one replaces, and
    the reason is one line of trigonometry: balancing puts the centre of mass
    over the contact, so the tilt works out to arccos(h_2/|h|) and runs to 90
    deg as the plate is thinned. A flat body balanced on itself STANDS. Resting
    it on an edge gives 68.5 and 66.8 deg instead of 82.3 -- the same answer.

    So the plate is laid down, and what the statics lose by it is nothing: a
    contact enters the model as the direction it pushes and nothing else, so a
    broad face on the floor is the one upward generator that a single corner
    was. What the drawing gains is that the shape is now read rather than
    inferred, and that every camera constraint downstream of the balance -- two
    elevation ceilings, the spin, the choice of azimuth -- simply lifts.
    """
    c, s = np.cos(psi), np.sin(psi)
    return (np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]),
            HALF3[LEVEL] * UP3)


def quad(R, centre, k, s, at=(0.0, 0.0), size=1.0):
    """A rectangle on the face whose outward normal is s * R e_k.

    `at` and `size` are fractions of that face's own half extents, so one call
    serves for the whole face, for the patch a support presses on, and -- at
    size zero -- for a single point placed somewhere on it.
    """
    p, q = (k + 1) % 3, (k + 2) % 3
    out = np.zeros((4, 3))
    for row, (u, v) in zip(out, ((-1, -1), (1, -1), (1, 1), (-1, 1))):
        row[k] = s * HALF3[k]
        row[p] = (at[0] + u * size) * HALF3[p]
        row[q] = (at[1] + v * size) * HALF3[q]
    return out @ R.T + centre


def facing(R):
    """The faces of the plate turned towards the camera, with their normals.

    A convex body never hides one of these behind another, so drawing this set
    and no more disposes of the one ordering question the body itself raises.
    """
    every = [(k, s, s * (R @ np.eye(3)[k])) for k in range(3) for s in (-1, 1)]
    return [f for f in every if f[2] @ VIEW > 0]


def lit(polys):
    """The faces of a convex solid that point at the camera, same trick again.

    Here the normals are not known in advance, so each is taken from the winding
    and turned outwards by the solid's own centre before it is asked which way
    it looks. Handed back with the face, because once a normal is outward it is
    also the thing the light has to be measured against.
    """
    hub = np.vstack(polys).mean(axis=0)
    out = []
    for P in polys:
        n = np.cross(P[1] - P[0], P[2] - P[1])
        n = n / np.linalg.norm(n) * np.sign(n @ (P.mean(axis=0) - hub))
        if n @ VIEW > 0:
            out.append((P, n))
    return out


def shaded(normals, base, amb, key, bounce):
    """A colour per face: how squarely each one meets the light.

    Lambert, plus a term for the light the near-white floor throws back up,
    which is the only thing that can reach a face pointing downwards. Both are
    clipped at zero -- a face turned away from a source receives nothing from
    it, it does not receive a negative amount -- so what is left over on an
    unlit face is the ambient, and the ambient is what decides how dark the
    darkest thing in the figure gets.
    """
    n = np.atleast_2d(np.asarray(normals, float))
    f = amb + key * np.clip(n @ LIGHT, 0, None) + bounce * np.clip(-n[:, 2], 0, None)
    return np.clip(f[:, None] * np.array(matplotlib.colors.to_rgb(base)), 0, 1)


def pressed(R, k):
    """Which of axis k's two faces the support takes: the one turned AWAY.

    Both do identical work -- they are the same face of the same plate, mirrored
    -- so this is chosen for the picture, and it settles two things at once. A
    chock on a face the reader can see must come between the reader and the
    body; on the far face it never can, and it needs no depth sorting either,
    which is why `panel` can hand every chock the same zorder.

    It has one cost, and it is unavoidable rather than chosen. The survivable
    set is {d : d.u <= 0 for every push u}; pressing the far faces puts both
    pushes on the camera's side, so the direction pointing straight AWAY from
    the reader satisfies both and belongs to the set. An arrow along it has no
    length on the page. Pressing the near faces instead would put the direction
    straight TOWARDS the reader in the set and cost the same, with the chocks
    occluding as well, so there is nothing to be had by swapping. What is left
    is to keep the drawn samples off that axis, which is what the panel-3
    sampling density in `solid` is picked for: at m = 6 the shortest arrow keeps
    26 % of its length, at m = 5 it collapses to 6 %.
    """
    return -1.0 if (R @ np.eye(3)[k]) @ VIEW > 0 else 1.0


def chock(R, centre, k):
    """A block on the floor with its inner wall flush against one vertical side.

    This replaces the pad that used to be splayed under a downward-facing face:
    with the plate lying down there are no downward faces to take, and what
    stops it is a block beside it. The cross-section, measured out from that
    wall, rises to `crest`, runs a short flat top, then falls to the floor at
    `depth`; it is extruded along the side over the stretch the contact covers.
    Every point of it lies on the outer side of the wall, so it touches the
    plate and reaches into nothing -- the same guarantee the splay used to give.

    The crest is what makes it drawable at all. Standing behind a plate this
    thin, a block no taller than the plate is hidden by it; twice the thickness
    clears the top edge and reads as a solid rather than as a shadow.
    """
    size, depth, crest = SEAT
    s = pressed(R, k)
    n = s * (R @ np.eye(3)[k])                           # the pressed face's normal
    p = [j for j in range(3) if j != k and j != LEVEL][0]
    e = R @ np.eye(3)[p]
    # slide the block along the side towards the reader, so its end and the
    # contact on it come to the silhouette instead of hiding behind the plate
    mid = np.sign(e @ VIEW) * (1 - size) * HALF3[p]
    ends = [n * HALF3[k] + e * (mid + t * size * HALF3[p]) for t in (-1, 1)]
    ends = [q * [1, 1, 0] + centre * [1, 1, 0] for q in ends]
    sect = [(0.0, 0.0), (0.0, crest), (0.30 * depth, crest), (depth, 0.0)]
    ring = [[q + a * n + b * UP3 for a, b in sect] for q in ends]
    # wall, flat top, slope, then the two ends -- and the footprint LAST, since
    # everything that asks a solid where it meets the floor takes polys[-1]
    polys = [np.array([ring[0][i], ring[1][i], ring[1][i + 1], ring[0][i + 1]])
             for i in range(len(sect) - 1)]
    polys += [np.array(ring[0]), np.array(ring[1][::-1]),
              np.array([ring[0][-1], ring[1][-1], ring[1][0], ring[0][0]])]
    # mark the contact at the near end of the wall, half way up the plate: the
    # middle of a face turned away is behind the body, and an arrow ending there
    # ends nowhere the reader can see it land
    at = max(ends, key=lambda q: q @ VIEW) + HALF3[LEVEL] * UP3
    return polys, at


def reachable(gens, m):
    """The survivable directions themselves, read straight off the generators.

    `d` survives when `up - d` is a non-negative combination of them, so every
    such combination hands one back: take a ray of the cone, walk out along it
    until it meets |up - d| = 1, and the disturbance is what is left over.
    Sampling the weights on a simplex therefore sweeps the whole set -- the lone
    direction, the arc, the patch -- with no case analysis, and the corners of
    the simplex are its extreme directions.

    The one direction thrown out is `up` itself, and it is the same artefact
    `measure` drops: a disturbance of exactly one body weight pulling straight
    up leaves `T = 0`, so the contacts do nothing and it survives on an equality
    saying nothing about what a support bought. It never arose while the plate
    stood on a corner, because no push there was horizontal. Lying down every
    push is, and `g` horizontal gives `g.up = 0` and hands back `up` exactly --
    a whole edge of the simplex collapsing onto one useless direction, five of
    the twenty-one weights in panel 3 among them.
    """
    w = np.array([c for c in itertools.product(range(m + 1), repeat=len(gens))
                  if sum(c) == m], float)
    g = w @ np.array(gens, float)
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    d = UP3 - 2 * (g @ UP3)[:, None] * g
    return d[np.linalg.norm(UP3 - d, axis=1) > 1e-9]


def landing(R, d, faces):
    """The face a push arrives on, and which way it came in across that face.

    Position is no part of the model, so this is legibility and nothing else:
    send each push to the visible face it meets most squarely, which is also the
    only way its shaft can stay outside a convex body, and rank it within that
    face by the direction it arrived from. The arrows then spread the way the
    set does -- a line of them for an arc, a sheet of them for a patch -- and no
    two of them cross, with not one of them placed by hand.
    """
    k, s, n = max(faces, key=lambda f: -d @ f[2])
    v = -d + (d @ n) * n
    return (k, s), np.array([v @ (R @ np.eye(3)[(k + i) % 3]) for i in (1, 2)])


def anchors(R, centre, land):
    """Those rankings turned into points, one face's worth at a time.

    Ranking alone bunches the arrows, because incoming directions crowd towards
    the middle of a cone; stretching each face's own range across the face
    spaces them evenly instead. The range is taken over all three panels at
    once, so a push that survives in more than one of them keeps its place, and
    the growth from one arrow to a line to a sheet is the set growing and not
    the spacing changing.
    """
    box = {}
    for key, f in itertools.chain(*land):
        lo, hi = box.get(key, (f, f))
        box[key] = (np.minimum(lo, f), np.maximum(hi, f))
    out = []
    for L in land:
        row = []
        for (k, s), f in L:
            lo, hi = box[(k, s)]
            wide = hi - lo > 1e-9                        # a lone arrow has no range
            t = np.where(wide, (f - lo) / np.where(wide, hi - lo, 1.0), 0.5)
            row.append(quad(R, centre, k, s, FILL * (2 * t - 1), 0.0)[0])
        out.append(row)
    return out


class Arrow3D(FancyArrowPatch):
    """A flat arrow with both ends pinned to points in space.

    mplot3d has no arrow of its own and quiver draws its head as bare lines, so
    the plane figure's arrowhead survives the move only by re-projecting a
    FancyArrowPatch on every draw. The mutation scale is the one `annotate`
    applies by default, which is what makes these the same arrows as those.
    """

    def __init__(self, tail, tip, **kw):
        super().__init__((0, 0), (0, 0), mutation_scale=10, **kw)
        self.ends = np.array([tail, tip], float)

    def do_3d_projection(self, renderer=None):
        x, y, _ = proj3d.proj_transform(*self.ends.T, self.axes.M)
        self.set_positions((x[0], y[0]), (x[1], y[1]))
        return 0.0                                       # the axes is told not to sort


def generator3(ax, at, u, z, ln=0.46):
    """One contact, drawn as the only thing it contributes: a push direction."""
    ax.add_artist(Arrow3D(at - ln * u, at, zorder=z, color=ORANGE, lw=3.0,
                          arrowstyle="-|>,head_width=.22,head_length=.42"))


def arrow3(ax, at, v, ln=0.70, z=11):
    """A disturbance that gets survived, landing on the face."""
    ax.add_artist(Arrow3D(at - ln * v, at, zorder=z, color=BLUE, lw=2.4,
                          arrowstyle="-|>,head_width=.20,head_length=.40"))


def com3(ax, c, r=0.085):
    """The centre-of-mass roundel, built in the view plane so it stays a roundel."""
    e1 = np.cross(UP3, VIEW)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(VIEW, e1)                              # already unit, and screen up

    def rim(a, b):
        t = np.radians(np.arange(a, b + 1, 5))[:, None]
        return c + r * (np.cos(t) * e1 + np.sin(t) * e2)
    ax.add_collection3d(Poly3DCollection([rim(0, 360)], facecolor="white",
                                         edgecolor=INK, lw=1.3, zorder=8))
    for a0 in (90, 270):
        ax.add_collection3d(Poly3DCollection([np.vstack([c, rim(a0, a0 + 90)])],
                                             facecolor=INK, edgecolor="none", zorder=9))


def iso(ax, lo, hi):
    """The camera: same projection, same limits, same aspect in all three panels.

    Orthographic and standing well above the floor, looking down the diagonal at
    ELEV. The panels only exist to be compared with one another, so the one
    thing that matters more than the angle is that all three take it.
    """
    ax.set_proj_type("ortho")
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)                           # equal world units per inch
    ax.set_axis_off()


def trim(out, pad=20):
    """Cut the saved image down to what was drawn on it.

    A plate lying down covers about twice the width it does the height, and the
    framing box has to stay a cube: `set_box_aspect` rescales whatever box it is
    given to a fixed diagonal and then stretches the result to fill the axes
    rectangle, so a box that is not a cube comes out both shrunk and squashed,
    and the elevation the reader sees stops being the one the camera was set to.
    A cube framed to the width therefore leaves two fifths of the page height
    empty, above and below every panel alike.

    Cropping the raster afterwards is the one fix that cannot distort anything,
    since it moves no pixel it keeps. All three panels share a vertical extent,
    so the same cut serves for all of them, and `pad` puts the margin back.
    """
    a = plt.imread(out)
    ink = (a[:, :, :3] < 0.96).any(axis=2)
    rows, cols = np.where(ink.any(axis=1))[0], np.where(ink.any(axis=0))[0]
    y0, y1 = max(rows.min() - pad, 0), min(rows.max() + 1 + pad, a.shape[0])
    x0, x1 = max(cols.min() - pad, 0), min(cols.max() + 1 + pad, a.shape[1])
    matplotlib.image.imsave(out, a[y0:y1, x0:x1])


def hull(P):
    """The convex hull of a set of points in the plane, as its vertices in order.

    Monotone chain: sort, then walk each side dropping any point the two around
    it turn the wrong way about. Every solid standing on this floor is convex
    and the shadow of a convex solid is convex, so this is the shadow itself and
    not an outline fitted to a scatter of samples.
    """
    P = np.unique(np.asarray(P, float).round(12), axis=0)
    if len(P) < 3:
        return P
    out = []
    for Q in (P, P[::-1]):
        side = []
        for p in Q:
            while len(side) > 1:
                a, b = side[-1] - side[-2], p - side[-2]
                if a[0] * b[1] - a[1] * b[0] > 0:
                    break
                side.pop()
            side.append(p)
        out += side[:-1]
    return np.array(out)


def cast(P):
    """A solid's vertices dropped along the light onto the floor.

    The one cue that costs nothing here and pays for everything: a solid with a
    shadow touching it is standing on the plane the shadow lies in, and a solid
    without one is at an unknown height above it. Where the caster meets the
    floor its shadow meets it too, so the join is made by the geometry and not
    by being placed.
    """
    P = np.asarray(P, float)
    return hull(P[:, :2] - np.outer(P[:, 2] / LIGHT[2], LIGHT[:2]))


def bed(P, r, n=28):
    """The floor immediately around whatever a solid rests on it.

    The one thing a cast shadow cannot show is the contact, because its own edge
    runs through it. Sweeping a small disc over the points that lie ON the floor
    covers every case with one rule, and it survived the body being laid down
    unchanged: a rounded rectangle around the plate's own footprint, another
    around each chock's, where it used to be a disc under a single corner.
    """
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = r * np.stack([np.cos(t), np.sin(t)], axis=1)
    down = np.asarray(P, float)
    return hull((down[down[:, 2] < 1e-9][:, :2][:, None, :] + ring).reshape(-1, 2))


def veil(t):
    """The floor colour with `t` of the light taken out of it.

    Opaque and not transparent, which matters: two shadows crossing would
    otherwise darken each other and draw a seam along an edge that is not there.
    The floor is one flat colour, so the blend is worked out here once and an
    overlap simply looks like more of the same shadow.
    """
    g, d = (np.array(matplotlib.colors.to_rgb(c)) for c in (GROUND3, DUSK))
    return (1 - t) * g + t * d


def flat(P):
    """A polygon of the ground plane, lifted into space to be drawn.

    Nothing culls these. Back faces are dropped by asking a normal whether it
    looks at the camera, and every one of these lies in z = 0 with its normal
    straight up, which from a camera standing above the floor always does.
    """
    return np.hstack([P, np.zeros((len(P), 1))])


def casters(R, centre, props=None):
    """Everything standing on the floor, as the points its shadow is thrown from.

    One list, read by the floor when it works out how big it has to be and by
    the panel when it draws what lies on it, so the two cannot fall out of step.

    `props` defaults to None and not to PROPPED, because a default argument is
    bound once at import: written the obvious way this function would go on
    reading the PROPPED of the moment the module loaded, and hand `chock` an
    axis it cannot build against.
    """
    props = PROPPED if props is None else props
    return ([np.vstack([quad(R, centre, k, s) for k in range(3) for s in (-1, 1)])]
            + [np.vstack(chock(R, centre, k)[0]) for k in props])


def shadow(ax, R, centre, props, stood=None):
    """The floor under the scene: what the light misses, and where it is touched.

    Two ramps, one cast and one at the contacts, merged and laid down lightest
    first. The fills are opaque, so where any two of them cross the darker one
    is what shows -- which is what two shadows crossing do anyway.

    All of it sits between the ground and everything standing on it. With the
    depth sorting turned off that is a zorder like any other, and an artist not
    given one lands wherever it happened to be added.
    """
    stood = casters(R, centre, props) if stood is None else stood
    thrown, u = [cast(P) for P in stood], (np.arange(SOFT) + 1) / SOFT
    # both ramps run darkness up faster than radius runs in, so the fall-off is
    # weighted to the core and the rim fades into the floor rather than ending
    # on it -- the contact one less steeply, or all of its darkness would land
    # inside the few pixels the floor's own arrowhead covers
    steps = [(DEEP * t ** 2.0, [p.mean(axis=0) + (1 - CORE * t) * (p - p.mean(axis=0))
                                for p in thrown]) for t in u]
    steps += [(BEDDED * t ** 1.6, [bed(P, BED[0] + (BED[1] - BED[0]) * t)
                                   for P in stood]) for t in u]
    for z, (dark, polys) in enumerate(sorted(steps, key=lambda s: s[0])):
        ax.add_collection3d(Poly3DCollection([flat(p) for p in polys], facecolor=veil(dark),
                                             edgecolor="none", zorder=1.0 + 0.01 * z))


def silhouette(R, centre, faces):
    """The body's outline: the edges where a face you can see meets one you cannot.

    An interior edge of the drawing has solid on both sides of it and the
    outline has floor or paper on one, so they are not the same line and are not
    drawn at the same weight. Which is which needs nothing beyond the visibility
    `facing` has already settled.
    """
    seen = {(k, s) for k, s, _ in faces}
    out = []
    for k in range(3):
        p, q = (k + 1) % 3, (k + 2) % 3
        for sp, sq in itertools.product((-1, 1), repeat=2):
            if ((p, sp) in seen) != ((q, sq) in seen):
                e = np.zeros((2, 3))
                e[:, k] = (-HALF3[k], HALF3[k])
                e[:, p], e[:, q] = sp * HALF3[p], sq * HALF3[q]
                out.append(e @ R.T + centre)
    return out


def tile(R, centre, ends, margin=0.15, touching=0.20):
    """The floor, as the four corners of a bounded quad lying in z = 0.

    A sheet run off all four sides of the picture is the thing that was wrong
    here: whatever the elevation, the only boundary left in view is the cut
    across it, and a cut across an orthographic ground plane is dead straight
    and, at any of these four azimuths, dead horizontal. That is a band under a
    line -- the plane figure's floor -- and no elevation can make it recede,
    because a plane recedes on the page by the convergence of its OWN edges and
    there are none.

    Bounded, and squared to the world axes rather than to the page, all four
    edges are in view and none of them is horizontal: the quad projects to a
    rhombus that opens with sin(elev), and the plate sits in the middle of it.
    Sized to whatever rests on it, over all three panels at once, so that the
    floor is the same floor in each -- and to any arrow end that all but does.

    It is the ARROWS that size this floor now, and by a long way. The plate lies
    a tenth of its own width off the ground, so every survivable direction with
    any lift in it puts its tail under z = 0: the lowest reaches -0.625 and 15
    of the 31 tails are below the plane, against 3 of 25 when the body stood on
    a corner. `touching` therefore no longer picks out arrows that nearly touch
    the floor but arrows that cross it, and the floor is carried out to sit
    under them; an edge cut inside one would leave it hanging over bare paper at
    floor level. This is what makes the light cheap again -- the shadows no
    longer reach anywhere the arrows have not already been.

    Shadows go in too, and for every panel at once like the chocks above them:
    the same failure otherwise, a shadow running off the edge of the floor it is
    cast on. They go in without the margin. The margin is there to keep a solid off the
    edge, and a shadow has already faded to nothing at its own rim, so the floor
    has to reach it and not a step further -- with the margin it would be paying
    for the whole reach of a pool whose last few steps cannot be seen, and every
    unit the floor grows comes off the size of the plate on the page.
    """
    on = np.vstack([quad(R, centre, k, s) for k in range(3) for s in (-1, 1)]
                   + [chock(R, centre, k)[0][-1] for k in PROPPED]
                   + [ends[ends[:, 2] < touching]])[:, :2]
    lo, hi = on.min(axis=0) - margin, on.max(axis=0) + margin
    stood = casters(R, centre)
    thrown = np.vstack([cast(P) for P in stood] + [bed(P, BED[0]) for P in stood])
    lo, hi = np.minimum(lo, thrown.min(axis=0)), np.maximum(hi, thrown.max(axis=0))
    return np.array([[hi[0], hi[1], 0.0], [hi[0], lo[1], 0.0],
                     [lo[0], lo[1], 0.0], [lo[0], hi[1], 0.0]])


def ground(ax, floor):
    """That quad, drawn: a near-white sheet with its edges shown.

    The furthest thing in the picture, so the lightest line in it.
    """
    ax.add_collection3d(Poly3DCollection([floor], facecolor=GROUND3,
                                         edgecolor=GROUND_INK, lw=1.4, zorder=0))


def foot(R, centre):
    """Where to draw the floor's own generator once the contact is a whole face.

    Balanced on a corner the body touched the floor at the origin and the arrow
    was drawn there. Lying down it touches over its whole underside, and the
    origin is the middle of that -- under the plate, where nothing can be seen.
    Position is no part of the model, so this is legibility and nothing else:
    the corner of the footprint nearest the reader, which is on the silhouette
    and stands clear of the body in every panel.
    """
    return max(quad(R, centre, LEVEL, -1), key=lambda p: p @ VIEW)


def producible(gens, pitch=ROW_PITCH):
    """The forces these contacts can supply: a direction, and how hard, in a row.

    The rim of the cone, walked at a fixed angular pitch, ordered as a sweep. One
    generator has no rim and is its own answer -- ONE direction, at exactly F,
    which is the whole of what the floor alone can do and the reason panel 1's
    row is a single arrow. Two give the arc between them, ends included. Three
    give the three arcs closing the patch, walked round as a loop, so every
    generator appears once and every pair contributes the arc it spans.

    The ends are kept, unlike the ball's `fanned`, which drops them because the
    generator arrows are already standing at exactly those two places on the same
    shell. Here the row stands somewhere else entirely -- on the plate, in a line
    -- so nothing else in it would say what a push direction on its own is worth,
    and that is the number the rest of the row is to be compared against.

    The magnitude comes from `strongest`, so this row and the pairs figure's fan
    are one calculation, and `spanned` is asserted on every sample in `main` so
    the row cannot drift off the set the rest of the figure counts.
    """
    G = [np.asarray(g, float) for g in gens]
    if len(G) == 1:
        path = [G[0]]
    else:
        loop = G + G[:1] if len(G) > 2 else G            # three arcs close; two do not
        path = []
        for a, b in zip(loop, loop[1:]):
            t = float(np.arccos(np.clip(a @ b, -1.0, 1.0)))
            n = max(int(round(np.degrees(t) / pitch)), 1)
            # the far end of each arc is left to open the next one, and for a
            # pair -- which does not close -- it is put back below
            for phi in np.linspace(0.0, t, n + 1)[:-1]:
                path.append((np.sin(t - phi) * a + np.sin(phi) * b) / np.sin(t))
        if len(G) == 2:
            path.append(loop[-1])
    return [(v, strongest(gens, v)) for v in path]


def rank(v):
    """Where a direction sits in the row: by the angle it is DRAWN at, descending.

    The row is a row and not a starburst, so the arrows have to be laid out in an
    order that reads, and sorting on the page angle makes the row a sweep: each
    arrow leans a little further round than the one to its left, like a clock
    hand strobed across the plate. It is the DRAWN angle that is sorted on and
    not the true one, because the drawn angle is what the reader compares.

    Panel 3's rim is a closed loop, so its six arrows cover a whole turn of the
    page, and no order can lay a whole turn along a line without one end leaning
    back over its neighbours. What this sort does about that is put the leaning
    one where it costs least: the two ends come out at +150 and -150 deg, so the
    left-hand arrow leans up and out of the row, and the right-hand one is a
    push direction at F -- the shortest arrow there is, and so the shortest
    reach back. It also nests the panels: sorted this way panel 2's three arrows
    are the second, third and fourth of panel 3's six, in that order, so the
    reader who has read one row can read the next.
    """
    r = np.cross(UP3, VIEW)
    r /= np.linalg.norm(r)
    return -float(np.arctan2(v @ np.cross(VIEW, r), v @ r))


def perch(R, centre, n):
    """`n` tails in a row across the top face, square to the page.

    Position is no part of the model -- a force enters as a direction and nothing
    else -- so this is legibility only, and it answers to three things. The row
    runs along the PAGE, on the widest chord of the face, so the arrows can be
    compared side by side like a bar chart; that chord is a world direction that
    turns with the camera, which is why it is taken from VIEW and not written
    down. It sits on the BROAD FACE, which is the horizontal plane itself, so a
    horizontal force drawn from it lies in the face and reads as horizontal
    rather than as the down-the-page arrow the projection makes of it. And the
    two small offsets nudge it off what is already there -- the roundel sits
    just under the middle of that chord, which is exactly where panel 1's single
    arrow would otherwise stand, and the blue arrows crowd the rest of the face.
    """
    r = np.cross(UP3, VIEW)
    r /= np.linalg.norm(r)
    back = np.cross(UP3, r)                              # up the page, still horizontal
    off = ROW_AT * back + ROW_OFF * r
    mid = centre + HALF3[LEVEL] * UP3 + off
    # the row must stay ON the face, so what is left over is measured against the
    # face's own half extents rather than assumed; `main` prints it
    b, o = R.T @ r, R.T @ off
    t = (np.arange(n) - (n - 1) / 2) * ROW_STEP
    slack = min(HALF3[k] - abs(o[k]) - abs(t).max() * abs(b[k])
                for k in range(3) if k != LEVEL)
    return [mid + s * r for s in t], float(slack)


def laid(R, centre, gens):
    """The row worked out but not yet drawn: where each arrow starts and ends.

    Separated from the drawing because `solid` has to frame these ends along
    with everything else -- an artist drawn outside the box is an artist cut in
    half -- and because `main` reads the same numbers back to print them. Called
    twice per panel and answering identically both times, which costs nothing at
    this size and is what keeps the framing honest.
    """
    out = sorted(producible(gens), key=lambda vm: rank(vm[0]))
    at, slack = perch(R, centre, len(out))
    seg = np.array([[t, t + ROW_LN * m / onpage(v, VIEW, ROW_SEEN) * v]
                    for t, (v, m) in zip(at, out)])
    return seg, [m for _, m in out], [v for v, _ in out], slack


def row(ax, R, centre, gens, z=13):
    """That row, drawn: each arrow along its own direction, sized by what it takes.

    Length and width both run with the magnitude, off the same number and with no
    curve on either, which is the ball fan's rule and is kept identical here on
    purpose. The two arrow families must not be confusable and four things
    separate them at once. COLOUR: orange is what the contacts supply, blue what
    they survive, and this row is made of the orange arrows' own combinations, so
    it takes their hue and it would be wrong in a third one. SENSE: a blue arrow
    ARRIVES, tip on the face; a row arrow LEAVES, tail on the face, which is
    where a force's arrow starts. PLACE: the blues spread over three faces and
    point inwards from outside the body, the row is a tidy line across the top of
    it, on dots. WEIGHT: ROW_LW is 1.9 at F against the generators' 3.0, so the
    two pushes stay the heaviest orange in the panel -- these are samples of what
    the pushes can do between them and not the pushes themselves.
    """
    seg, mag, _, _ = laid(R, centre, gens)
    for (tail, tip), m in zip(seg, mag):
        # Arrow3D and not quiver, for the reason the ball's fan is: mplot3d
        # builds a quiver head in a plane it picks from the shaft alone, so an
        # arrow leaning towards the reader loses its head, and every arrow in
        # this row leans towards the reader -- the whole cone does
        a = Arrow3D(tail, tip, color=ORANGE, lw=max(ROW_LW * m, ROW_THIN),
                    shrinkA=0.0, shrinkB=0.0, zorder=z,
                    arrowstyle="-|>,head_width=.15,head_length=.42")
        # the head is in points, so it has to be scaled by hand on the same
        # magnitude as the rest, or a short arrow becomes a head with nothing
        # behind it and reads as the largest thing in the row
        a.set_mutation_scale(max(ROW_HEAD * m, ROW_TIP))
        ax.add_artist(a)
    # a dot under every tail, which is what makes the row a row. The arrows
    # point six different ways on the page and a reader given only their shafts
    # has to work out where each one starts before any two can be compared; the
    # dots hand over that baseline in one glance, and they are what tells this
    # family from the blue arrows at a distance -- nothing else in the picture
    # stands in an even line
    ax.scatter(*seg[:, 0].T, s=ROW_DOT, c=ORANGE, edgecolors=PAPER, linewidths=0.8,
               depthshade=False, zorder=z - 0.1)


def paged(p):
    """A point, dropped onto the page: right and up, in world units.

    The row is checked for crowding where crowding happens, which is on the
    page and not in the world. Two arrows a long way apart in space can be drawn
    across one another, and this figure has 21 blue arrows aimed inwards from
    every side of a body the row sits on top of.
    """
    r = np.cross(UP3, VIEW)
    r /= np.linalg.norm(r)
    return np.asarray(p, float) @ np.array([r, np.cross(VIEW, r)]).T


def nearest(p, q, a, b):
    """Least distance between two segments in the plane, zero if they cross."""
    def near(p, a, b):
        v = b - a
        t = np.clip((p - a) @ v / max(v @ v, 1e-12), 0.0, 1.0)
        return float(np.linalg.norm(p - (a + t * v)))
    d = min(near(p, a, b), near(q, a, b), near(a, p, q), near(b, p, q))
    (u1, u2), (v1, v2) = q - p, b - a
    den = u1 * v2 - u2 * v1
    if abs(den) > 1e-12:
        s = ((a - p)[0] * v2 - (a - p)[1] * v1) / den
        t = ((a - p)[0] * u2 - (a - p)[1] * u1) / den
        if 0 <= s <= 1 and 0 <= t <= 1:
            return 0.0
    return d


def crowding(R, centre, gens, props, shots):
    """How close the row comes to everything else drawn, measured on the page.

    Not a decoration: panel 3 already carries 21 blue arrows, two chocks and a
    roundel, and the row lays six more arrows across the middle of it. Two
    numbers come back, because two things can go wrong and they are not equally
    bad.

    NEAR is the general one: the least the row comes to anything that would
    still confuse if it touched -- another row arrow, a generator arrow, an
    arrowHEAD against a blue shaft either way round, or the roundel. Blue
    crossing orange at a steep angle is not counted, because it is two colours
    crossing and the reader separates them without effort.

    FAMILY is the one that decides how WIDE the row may be, and it is the reason
    the widest layouts are not taken although they score better on NEAR. A
    generator arrow is the same hue as a row arrow, so the two merge rather than
    cross -- and at this camera they merge into a straight line, because the
    three pushes project 120 deg apart and therefore sum to nothing on the page,
    which makes every sqrt(2) F direction draw exactly ANTIPARALLEL to the one
    push it does not contain. Spread the row past the plate's corners and its
    outermost arrow comes down along a chock's own push, 0.12 apart instead of
    0.25. ROW_STEP, ROW_LN, ROW_AT and ROW_OFF were picked by maximising NEAR
    subject to FAMILY staying above 0.20, and `main` prints both, so a later
    change that crowds the row says so out loud.
    """
    seg, _, _, slack = laid(R, centre, gens)
    seg = [(paged(a), paged(b)) for a, b in seg]
    blue = [(paged(at - 0.70 * d), paged(at)) for at, d in shots]
    ink = [(paged(at - 0.46 * (-pressed(R, k) * (R @ np.eye(3)[k]))), paged(at))
           for k, at in ((k, chock(R, centre, k)[1]) for k in props)]
    ink.append((paged(foot(R, centre) - 0.46 * UP3), paged(foot(R, centre))))
    mark = paged(centre)
    near, family = (np.inf, ""), (np.inf, "")
    for i, (p, q) in enumerate(seg):
        for j, (a, b) in enumerate(seg[i + 1:]):
            near = min(near, (nearest(p, q, a, b), f"arrow {i} on arrow {i + 1 + j}"))
        for j, (a, b) in enumerate(ink):
            family = min(family, (nearest(p, q, a, b), f"arrow {i} on generator {j}"))
        for a, b in blue:                                # heads only, either way
            near = min(near, (min(nearest(q, q, a, b), nearest(b, b, p, q)),
                              f"arrow {i} head against a blue one"))
        near = min(near, (nearest(mark, mark, p, q) - 0.085, f"arrow {i} on the roundel"))
    near = min(near, family)
    return float(near[0]), near[1], slack, float(family[0]), family[1]


def panel(ax, R, centre, props, shots, gens, floor):
    """One panel: the plate, the chocks it has, and the pushes it survives."""
    faces = facing(R)
    ground(ax, floor)
    shadow(ax, R, centre, props)
    for k in props:
        # every chock stands behind the plate, because `pressed` gives it the
        # face turned away from the reader. That is what lets them all take one
        # zorder: while the body balanced on a corner this had to be decided per
        # support and per azimuth, and from two of the four the chock came
        # forward and covered the contact the argument rested on
        polys, at = chock(R, centre, k)
        seen = lit(polys)
        # standing behind the plate is not the same as looking at it: cooler,
        # paler and shaded over a narrower range than the body, with a lighter
        # line round it
        ax.add_collection3d(Poly3DCollection(
            [P for P, _ in seen], facecolor=shaded([n for _, n in seen], SUPPORT3, *HAZE),
            edgecolor=SUPPORT_INK, lw=1.0, zorder=2))
        generator3(ax, at, -pressed(R, k) * (R @ np.eye(3)[k]), 3)
    # the plane figure could fill the body with one flat colour; three faces of
    # one solid cannot, or the form is left for the edges alone to carry
    ax.add_collection3d(Poly3DCollection(
        [quad(R, centre, k, s) for k, s, _ in faces],
        facecolor=shaded([n for _, _, n in faces], BODY3, *SUN),
        edgecolor=INK, lw=1.1, zorder=4))
    ax.add_collection3d(Line3DCollection(silhouette(R, centre, faces), colors=INK,
                                         lw=2.4, zorder=4.4))
    com3(ax, centre)
    generator3(ax, foot(R, centre), UP3, 12)             # the floor, always free
    for at, d in shots:
        arrow3(ax, at, d)
    # the row of producible forces used to be drawn here, last and on top. It is
    # off, on the human's instruction: a third family of arrows on a panel that
    # already carries 21 blue ones and up to 3 orange generators did not read --
    # nobody could tell what the extra orange arrows meant. `row()` and its
    # constants are kept because they work and were expensive to place, and
    # because the same quantity IS drawn, legibly, on the ball in
    # dimension_pairs.png, which is where it belongs. Turning this back on
    # without also thinning the blue fan will reproduce the same complaint.
    if ROW:
        row(ax, R, centre, gens)


def measure(gens, n):
    """Sweep the sphere and keep what survives, minus the one artefact.

    A disturbance of exactly one body weight straight up leaves the contacts
    nothing at all to do, so it passes every panel, and it passes alone: it is
    an isolated direction each time, it survives on an equality, and it says
    nothing about what a support bought. `spans` drops it in the plane for the
    same reason. Here it is dropped by throwing out the directions that pass
    only because the force they ask for is next to nothing.
    """
    h = np.sqrt(4 * np.pi / n)                           # mean spacing on the sphere
    d, ok = survivable(gens, n, h)
    return d, ok & (np.linalg.norm(UP3 - d, axis=1) > 4 * h)


def clumps(d, sep):
    """The surviving directions grouped into the separate pieces they form."""
    left, out = list(range(len(d))), []
    while left:
        grp, i = [left.pop()], 0
        while i < len(grp):
            near = [j for j in left if d[j] @ d[grp[i]] > np.cos(sep)]
            left = [j for j in left if j not in near]
            grp, i = grp + near, i + 1
        out.append(np.array(grp))
    return out


def spread(d):
    """The angular length of a set that is a curve: its two ends, as an angle."""
    return np.degrees(np.arccos(np.clip((d @ d.T).min(), -1, 1)))


def refine(gens, ns):
    """The three dimensions themselves, read off two grid densities.

    With the tolerance tied to the grid spacing the counts have to behave. A set
    with no length keeps the same handful of samples however fine the grid; one
    with length collects them like the square root of the density; only a set
    with area holds a fixed share of them. Nine times the samples is therefore
    the test: 1, 3 and 9 times the hits, and 0, 1 and 2 dimensions.
    """
    rows = []
    for g in gens:
        rows.append([])
        for n in ns:
            ok = measure(g, n)[1]
            rows[-1].append((int(ok.sum()), 100 * ok.sum() / n))
    return rows


def solid(out):
    """The three panels, and the three numbers that are the argument."""
    R, centre = slab()
    u = [-pressed(R, k) * (R @ np.eye(3)[k]) for k in range(3)]
    props = [(), PROPPED[:1], PROPPED]
    gens = [[UP3], [UP3, u[PROPPED[0]]], [UP3, *(u[k] for k in PROPPED)]]
    # 6 rather than 4 in the last panel, and it is not a taste. The set there is
    # a quarter of the sphere and the reader looks straight into it, so one
    # sampled direction always runs close to the view axis and projects to
    # almost nothing (see `pressed`). Of the densities that give a workable
    # count, 6 leaves that shortest arrow 26 % of its length and 5 leaves it 6 %
    drawn = [reachable(g, m) for g, m in zip(gens, (1, 9, 6))]

    faces = facing(R)
    land = [[landing(R, d, faces) for d in D] for D in drawn]
    shots = [list(zip(A, D)) for A, D in zip(anchors(R, centre, land), drawn)]

    ends = np.array([[at, at - 0.70 * d] for S in shots for at, d in S])
    floor = tile(R, centre, ends[:, 1])
    pts = [quad(R, centre, k, s) for k in range(3) for s in (-1, 1)]
    pts += [chock(R, centre, k)[0][-1] for k in PROPPED]
    pts += [floor]                                       # its corners are the widest thing
    pts += [ends.reshape(-1, 3)]
    # and the row of producible forces, which has to be framed like anything
    # else that is drawn. It reaches nowhere near the arrow tails -- it lives on
    # top of the plate -- so it does not in fact move the box; `main` prints the
    # margin, and if that ever goes negative the frame is what has changed
    pts += [laid(R, centre, g)[0].reshape(-1, 3) for g in gens]

    # frame all three panels on one cube, sized by what the drawing covers on
    # the page rather than by its extent along the world axes: those axes run
    # diagonally across the view, so a box drawn tight around the geometry
    # projects to a hexagon far larger than the drawing, and every bit of the
    # difference comes out as empty floor. The cube is not negotiable -- see
    # `trim`, which is what pays for it
    r = np.cross(UP3, VIEW)
    r /= np.linalg.norm(r)
    page = np.array([r, np.cross(VIEW, r)])              # right and up on the page
    cube = np.abs(page).sum(axis=1)                      # a unit cube's page size
    flat = np.vstack(pts) @ page.T
    span = flat.max(axis=0) - flat.min(axis=0)
    side = 1.04 * max(span / cube)
    mid = (flat.max(axis=0) + flat.min(axis=0)) / 2 @ page
    lo, hi = mid - side / 2, mid + side / 2

    # computed_zorder=False is what makes the panels drawable at all. Left to
    # itself mplot3d sorts every collection and patch by one averaged depth, so
    # an arrow that lands on the body counts as nearer or further than the whole
    # of it, and the answer flips as the arrow moves. Turned off, the zorders
    # set here decide, exactly as they do in the plane figure
    #
    # a 3d axes stretches its box to whatever rectangle it is given, so the
    # panel has to be cut to the shape the box projects to or the elevation the
    # camera is set to is not the elevation the reader sees
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.5 * cube[1] / cube[0]), dpi=210,
                             facecolor=PAPER,
                             subplot_kw=dict(projection="3d", computed_zorder=False))
    for ax, prop, shot, g in zip(axes, props, shots, gens):
        ax.set_facecolor(PAPER)
        iso(ax, lo, hi)
        panel(ax, R, centre, prop, shot, g, floor)
    fig.subplots_adjust(0.0, 0.0, 1.0, 1.0, 0.0)
    fig.savefig(out, facecolor=PAPER)
    plt.close(fig)
    trim(out)
    # the row against everything else, panel by panel, and the tightest of the
    # three is the figure's number -- once for anything at all, once for the
    # other orange arrows on their own. Measured on the panels that were just
    # drawn, so it cannot describe a layout other than the one saved
    tight = [crowding(R, centre, g, p, s) + (n + 1,)
             for n, (g, p, s) in enumerate(zip(gens, props, shots))]
    return gens, u, (min(tight, key=lambda t: t[0]), min(tight, key=lambda t: t[3]))


# ------------------------------------------------- the same thing, on the ball

# the shell, the region the arrows close on, and the weight through the
# middle. The region takes the generators' own hue washed out, because it is
# what they span and not a second thing: pale orange between orange arrows
SHELL, FAN, SPOKE = "#c9ccc6", "#F2C88E", "#111110"
LON, LAT = 96, 48                                        # the ball's own tessellation
# where an arrow leaving the ball starts, and how long it is drawn when the
# force along it is exactly F -- one support's whole strength. The generator
# arrows are drawn at precisely these two numbers, which is what anchors the
# capped-magnitude fan below: the reader has the unit in front of them, so a fan
# arrow LONGER than a generator arrow is a direction the pair can push harder
# than either push on its own, and a shorter one is a direction they cannot even
# manage one support's worth in
STAND, UNIT_LN = 1.03, 0.44
# the fan: how hard each direction can be pushed, sampled every FAN_PITCH deg
# along the region. Both cues carry the same number, length AND width, because
# the weak end of a 175 deg arc is where either cue alone gives out -- a hairline
# that is also a stub is unmistakably a stub, where a hairline of full length
# reads as a drawing error and a short arrow of full width reads as a near miss.
# FAN_LW is under the generators' own 3.2 on purpose: these are samples of what
# the pair can do, not the two pushes themselves, and the two ends of the fan ARE
# the generator arrows and must stay the heaviest things on the ball.
#
# FAN_THIN is a floor and it costs something, so it is set as low as printing
# allows. It binds below 0.341 F, which is 16 of the 18 arrows in the 175 deg
# column and none at all in the 90 deg one -- the middle of that arc is 0.087 F
# and asks for 0.19 pt, and a 0.19 pt line does not print. So across the weak
# middle the width stops carrying the number and only says "small", and the
# LENGTH is left un-floored to carry it instead: those arrows run 9.9 px to
# 24.8 px, all of them visible, none of them softened. Neither cue would do on
# its own and that is the whole reason there are two
FAN_PITCH, FAN_LW, FAN_THIN = 9.0, 2.2, 0.75
# the arrowhead, in mutation points, at F and at its smallest. A ball radius is
# 88 pt on this page, so an arrow standing for F is 0.44 * 88 = 38.8 pt long, and
# 28 puts a head of 0.42 * 28 = 11.8 pt on it -- a little under a third of the
# shaft, against the 0.42 of its own that mplot3d gives the generator arrows
# beside it. Leaner on purpose, and the same reason FAN_LW is: a fan arrow is a
# sample, the two generators are the pushes. FAN_TIP keeps the head of the
# 0.087 F stub at 1.7 pt, half the length of the stub, which is as small as a
# thing can be and still be an arrow rather than a tick
FAN_HEAD, FAN_TIP = 28.0, 4.0
# the least of a length the projection may leave on the page before `onpage`
# stops undoing it, so no arrow is ever stretched by more than 1/0.55 = 1.8x. It
# is a guard and nothing else: the worst foreshortening anywhere in this figure
# is 0.669, at the bisector of the 90 deg pair, so the clamp is never reached.
# It is here so that moving the camera degrades the figure instead of exploding it
FAN_SEEN = 0.55


def shell(n_lon=LON, n_lat=LAT):
    """The unit sphere as quads, with the outward direction of each.

    One mesh, drawn as ONE collection with a colour per quad, is what lets a
    translucent ball sort against itself: handed two collections mplot3d gives
    each a single averaged depth and the near half of one can end up behind the
    far half of the other. Fine enough that a boundary running through it is a
    boundary and not a staircase -- the sets here have edges that are exact
    great circles, and a coarse ball turns them into bunting.
    """
    lon = np.linspace(0, 2 * np.pi, n_lon + 1)
    lat = np.linspace(0, np.pi, n_lat + 1)
    quads = []
    for i in range(n_lon):
        for j in range(n_lat):
            a, t = np.array([lon[i], lon[i + 1], lon[i + 1], lon[i]]), \
                   np.array([lat[j], lat[j], lat[j + 1], lat[j + 1]])
            quads.append(np.stack([np.sin(t) * np.cos(a),
                                   np.sin(t) * np.sin(a), np.cos(t)], axis=1))
    quads = np.array(quads)
    mid = quads.mean(axis=1)
    return quads, mid / np.linalg.norm(mid, axis=1, keepdims=True)


def spanned(gens, v, tol=1e-9):
    """Which directions the contacts can actually push in.

    A contact can pull nothing, so what the set of them can supply is the
    NON-NEGATIVE combinations of their push directions -- a cone, and its trace
    on the ball is the region between the arrows. Read off `gap`, the same
    solver the counts and the scene figure use, so the painted region and every
    number quoted about it are one claim and not two.
    """
    return gap(np.array(gens, float).T, np.atleast_2d(v)) <= tol


def between(a, b, n=241):
    """The great-circle arc from one direction to another, walked evenly.

    Two generators span a wedge, and the wedge meets the ball along exactly this
    arc: it IS the two-force region, and it is also an edge of the three-force
    one. Drawn as a curve and not painted, because it has no area -- nothing on
    a tessellated ball can carry it.

    None of that survives the two being OPPOSITE, and the failure is not a
    rounding one. Non-negative combinations of `u` and `-u` are the multiples of
    `u`, so the cone is the LINE and it meets the ball in two points, not in an
    arc -- the conic hull is not continuous in its generators, and 179 deg apart
    gives an arc 179 deg long while 180 deg apart gives nothing but the two
    ends. Swept at n = 200000 and clustered, `{x,-x}` comes back as 2 separate
    pieces 0.6 deg across; `{x, 179 deg away}` comes back as 1 piece 179.4 deg
    across. So there is nothing to draw here and this hands back an empty curve;
    the caller must not read the absence as a thin arc.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a @ b < -1 + 1e-12:
        return np.empty((0, 3))
    t = np.linspace(0.0, 1.0, n)[:, None]
    g = (1 - t) * a + t * b
    return g / np.linalg.norm(g, axis=1, keepdims=True)


def strongest(gens, v, cap=1.0, tol=1e-9):
    """The largest force these contacts can put along `v`, each capped at `cap`.

    `spanned` answers WHICH directions the cone reaches and says nothing about
    how far, and on its own that is a misleading half of the answer: it scores
    175 deg apart as nearly a half circle of capability when the pair can barely
    push at all across the middle of it. This is the other half. Every support is
    the same physical thing and so takes the same cap F. A direction inside the
    cone is `v = sum c_i u_i` with every `c_i >= 0`, and scaling the whole
    combination scales every coefficient together, so the cap binds on the
    LARGEST of them and the answer is

        m(v) = F / max_i c_i

    Which decomposition, when there is more than one? The one that leaves the
    largest coefficient smallest -- that is what the contacts would actually do,
    and it is the only choice that does not read a redundancy as a weakness. It
    matters exactly once here and the case is the figure's first column: `u` with
    `-u` spans the LINE, and `u` itself is `1*u + 0*(-u)` but equally
    `(1+s)*u + s*(-u)` for any `s >= 0`. Taking the minimum returns F, which is
    right -- one support pushing on its own -- where a careless solve could
    return anything down to nothing.

    Built the same way `gap` is, and for the same reason: with at most three
    generators the feasible decompositions live in eight subsets, so this is a
    handful of ordinary solves and no solver. The residual test is what rejects a
    subset too small to represent `v` at all; `gap` does not need it because it
    is measuring that residual rather than requiring it to vanish.
    """
    G = np.array(gens, float).T
    v = np.asarray(v, float)
    best = np.inf
    for r in range(1, G.shape[1] + 1):
        for S in itertools.combinations(range(G.shape[1]), r):
            c = np.linalg.lstsq(G[:, S], v, rcond=None)[0]
            if (c >= -tol).all() and np.linalg.norm(G[:, S] @ c - v) <= tol:
                best = min(best, float(c.max()))
    return cap / best if np.isfinite(best) else 0.0


def fanned(gens, pitch=FAN_PITCH):
    """Directions to hang the fan on: the region's edges, walked by ANGLE.

    Walked by angle rather than by a fixed count per edge, so the pitch is the
    same in every panel and a longer arc holds more arrows -- the fan then says
    how far the region runs as well as how hard it can be pushed, and the two
    columns can be compared arrow for arrow. The two ends of each arc are left
    out because they are the generators, already drawn, heavier, at exactly the
    length F earns.

    An opposed pair returns nothing at all, and that is the answer rather than an
    omission: the cone is the LINE and meets the ball in the two generators
    themselves, so the fan there IS the two arrows already standing on it, each
    at the full F. Three generators would give the three edges of the patch and
    not its inside; nothing asks for that today.
    """
    out = []
    for a, b in itertools.combinations(np.array(gens, float), 2):
        t = float(np.arccos(np.clip(a @ b, -1.0, 1.0)))
        if not 1e-9 < t < np.pi - 1e-9:
            continue
        n = max(int(round(np.degrees(t) / pitch)), 2)
        for phi in np.linspace(0.0, t, n + 1)[1:-1]:
            # the unit direction phi along from `a`, which is also the
            # decomposition `strongest` reads back: c = (sin(t-phi), sin phi)/sin t
            out.append((np.sin(t - phi) * a + np.sin(phi) * b) / np.sin(t))
    return out


def onpage(d, v, floor=FAN_SEEN):
    """What fraction of a length along `d` survives the projection to the page.

    The reason this is needed at all is the reason the ball exists. An arrow's
    3-D length cannot be read off the page -- the projection scales it by
    `sqrt(1 - (d.v)^2)`, which for these three pairs runs from 1.000 at a push
    lying across the view to 0.669 at the bisector of the 90 deg pair, and that
    bisector is exactly the direction the figure most needs drawn long. Left
    uncorrected the 90 deg fan swells 41 % in truth and 11 % on the page, and the
    two ends of the 90 deg column are drawn 15 % shorter than the two ends of the
    175 deg column although both stand for the same force F.

    So the arrows are given the 3-D length whose PAGE length is the magnitude,
    and the page is where the reader is. Nothing is lost by it: a radial arrow on
    a ball has no readable 3-D length in the first place -- there is no
    foreshortening cue on it to contradict, which is not true of anything drawn
    in the scenes. What is NOT corrected is the tail, which stands at 1.03 on the
    ball because that is a POSITION and positions are what this figure projects
    honestly; only the length carries a number.
    """
    return max(float(np.sqrt(max(1.0 - (np.asarray(d, float) @ v) ** 2, 0.0))), floor)


def axis(gens):
    """Where to stand to see a cone: straight down its own middle.

    Every generator then leans the same amount towards the reader and none of
    them is drawn foreshortened more than another, which is what the figure asks
    for -- all three arrows on the near half of the ball, and the region they
    close on facing the reader square. For the three pushes here the axis comes
    out at azimuth -45 and elevation 35.26, the isometric direction, because the
    three stand mutually square: the floor pushes up and each chock pushes along
    a horizontal it shares with no one.
    """
    v = np.sum(np.asarray(gens, float), axis=0)
    v /= np.linalg.norm(v)
    return v, float(np.degrees(np.arcsin(v[2]))), float(np.degrees(np.arctan2(v[1], v[0])))


def ball(ax, gens, view, title, fan=False):
    """One panel: the ball of directions, painted where the contacts can push.

    The scene figure draws these as arrows standing on the body, which an
    orthographic view cannot keep straight -- inclination and azimuth are two
    numbers and the angle an arrow is drawn at is one, so a push 16.6 deg BELOW
    horizontal comes out drawn above one exactly in it. Here nothing is
    projected twice: a direction is a place on the ball, and the set is whatever
    shape it makes there.

    What it makes is exact rather than measured, and it is the argument in one
    line. One force reaches ONE DIRECTION, a point. Two reach the ARC between
    them -- 90 deg here, the two standing square. Three reach the PATCH they
    close on, and since all three stand mutually square that patch is an octant,
    an eighth of the ball, 12.500 %. Point, arc, patch; nothing, nothing, an
    eighth. The second panel buys no area whatever and still buys everything,
    which is the whole of what this figure is for.

    `fan` adds the other half of the answer -- HOW HARD each of those directions
    can be pushed, as a fan of capped-magnitude arrows along the region -- and it
    is OPT-IN and off by default because this function is shared. Turning it on
    for everyone would rewrite `dimension_ball.png`, whose whole subject is the
    point / arc / patch progression and which must not start arguing a second
    thing in the middle of it. `dimension_pairs.png` asks for it because there
    the dimension count alone is actively misleading: 175 deg apart scores as an
    arc all but a half circle long, and the pair cannot push across the middle of
    it at all.
    """
    v, elev, azim = view
    quads, mid = shell()
    inside = spanned(gens, mid)
    # shaded, and not flat. A ball filled with one colour projects to a disc and
    # the reader has to take the third dimension on trust from the equator
    # alone; the same Lambert term the solids take costs nothing here. The paint
    # keeps its own hue under the same law, so a lit part of the region and a
    # lit part of the shell agree about where the light is
    face = np.empty((len(quads), 4))
    face[~inside] = np.hstack([shaded(mid[~inside], SHELL, 0.62, 0.38, 0.0),
                               np.full((int((~inside).sum()), 1), 0.30)])
    face[inside] = np.hstack([shaded(mid[inside], FAN, 0.70, 0.30, 0.0),
                              np.full((int(inside.sum()), 1), 0.88)])
    ax.add_collection3d(Poly3DCollection(quads, facecolors=face, edgecolors="none",
                                         zsort="average", zorder=1))
    # every edge of the region, which for two generators IS the region. Cut at
    # the silhouette and drawn twice, because mplot3d gives a whole line one
    # depth and would put all of it in front: the far half then reads as a loop
    # floating over the ball rather than a curve lying on it
    for a, b in itertools.combinations(gens, 2):
        arc = between(a, b)
        for near, w, al in ((True, 3.4, 1.0), (False, 2.4, 0.40)):
            keep = (arc @ v > 0) == near
            ax.plot(*np.where(keep[:, None], 1.004 * arc, np.nan).T, color=ORANGE,
                    lw=w, alpha=al, solid_capstyle="round", zorder=2 if near else 0.4)
    # the fan, when it is asked for: one arrow per sampled direction, pointing
    # out along ITS OWN direction and standing off the shell where the arc it
    # belongs to lies, so the fan and the curve cannot come apart. The curve
    # stays under it and earns its keep twice over -- it is what carries the
    # region across the middle of the 175 deg arc, where the arrows have shrunk
    # to stubs and would otherwise leave a gap the reader would read as a break
    # in the SET rather than as a collapse in what it is worth.
    #
    # Length and width both run in proportion to the magnitude, off the same
    # number and with no curve applied to either: a square-root would flatter the
    # weak middle of a 175 deg arc, and flattering it is the one thing this fan
    # exists not to do. `arrow_length_ratio` is the generators' own, so the head
    # shrinks with the shaft and a stub stays an arrow rather than turning into a
    # head with nothing behind it
    if fan:
        for d in fanned(gens):
            m = strongest(gens, d)
            near = d @ v > 0
            tail = STAND * d
            # Arrow3D and not `quiver`, and this one is not a preference.
            # mplot3d builds a quiver head out of two barbs in a plane it picks
            # from the shaft alone, so an arrow aimed at the reader has its head
            # projected edge on and loses it -- and the arrow most nearly aimed
            # at the reader here is the bisector of the 90 deg pair, the
            # strongest direction in the figure, which came out a bar with a
            # square end. A FancyArrowPatch re-projects both ends and draws its
            # head on the PAGE, so the head is there whichever way the arrow
            # points. `shrink` has to be zeroed with it: FancyArrowPatch pulls
            # 2 pt off each end by default, and the weakest arrows here are
            # 3.4 pt long altogether
            a = Arrow3D(tail, tail + UNIT_LN * m / onpage(d, v) * d, color=ORANGE,
                        lw=max(FAN_LW * m, FAN_THIN), shrinkA=0.0, shrinkB=0.0,
                        alpha=1.0 if near else 0.40, zorder=2.6 if near else 0.35,
                        # slimmer than the scenes' arrows at .22, because it is
                        # cut to the same taper mplot3d gives the generator
                        # arrows beside it -- a 15 deg cone, half-width over
                        # length 0.27 -- so the fan and the two arrows it hangs
                        # between read as one family of arrows
                        arrowstyle="-|>,head_width=.115,head_length=.42")
            # the head is in points and so has to be scaled by hand, on the same
            # magnitude as everything else: a stub whose head stayed full size
            # would be a head with nothing behind it, and would read as the
            # largest thing in the fan rather than the smallest
            a.set_mutation_scale(max(FAN_HEAD * m, FAN_TIP))
            ax.add_artist(a)
    # gravity down the middle -- inside the ball, so under the shell and seen
    # through it -- and the generators standing out of it, each at the direction
    # it pushes. The corner dots are the one-force region drawn as itself
    ax.quiver(0, 0, 0.58, 0, 0, -1.16, color=SPOKE, lw=2.0, arrow_length_ratio=0.20,
              zorder=0.5)
    for g in gens:
        g = np.asarray(g, float)
        # the two ends of the fan, and they have to be measured on the same rule
        # as the rest of it or the anchor is a lie: with the fan on, a generator
        # arrow is the length F earns ON THE PAGE, not in the world
        ln = UNIT_LN / onpage(g, v) if fan else UNIT_LN
        ax.quiver(*(STAND * g), *(ln * g), color=ORANGE, lw=3.2,
                  arrow_length_ratio=0.42, zorder=3)
        ax.scatter(*(1.02 * g), s=80, c=ORANGE, edgecolors="white", linewidths=1.1,
                   depthshade=False, zorder=3)
    th = np.linspace(0, 2 * np.pi, 240)
    ring = np.stack([1.02 * np.cos(th), 1.02 * np.sin(th), 0 * th], axis=1)
    for near, al in ((True, 0.75), (False, 0.30)):
        keep = (ring @ v > 0) == near
        ax.plot(*np.where(keep[:, None], ring, np.nan).T, color=MUTED, lw=0.9,
                alpha=al, zorder=2 if near else 0.4)
    # the box is the same 1.5 with the fan as without it, and that is worth a
    # line because it looks like luck. On the page an arrow reaches
    # 1.03 * seen + 0.44 * m from the middle, and the two terms trade against
    # each other: the arrows that are drawn longest are the ones pointing most
    # nearly at the reader, whose tails the same foreshortening has pulled in.
    # Measured over all three columns the largest is 1.47, which is the
    # generator arrows lying across the view -- exactly what the figure already
    # framed before the fan existed
    if fan:
        # and the projection has to become orthographic with it. mplot3d's
        # default is a PERSPECTIVE one at focal length 1, which is a strong
        # perspective -- the near side of a ball of radius 1 is drawn visibly
        # larger than the far side. Nothing was wrong with that while every
        # arrow was the same length and only its DIRECTION was being read; it is
        # fatal the moment a length carries a number, because the same magnitude
        # then draws longer for pointing at the reader than for pointing across.
        # The worst case is not hypothetical: at 90 deg apart the strongest
        # direction on the arc is the bisector, and `apart` has put the bisector
        # nearly down the camera's own axis. So the fan is drawn flat, the way
        # the scenes are (`iso`), and the ball figure keeps the default because
        # nothing there is measured off a length
        ax.set_proj_type("ortho")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    ax.set_box_aspect((1, 1, 1), zoom=1.30)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, color=INK, fontsize=13, pad=2, linespacing=1.5)


def balls(out):
    """The three balls, and the areas that are the whole argument."""
    R, _ = slab()
    u = [-pressed(R, k) * (R @ np.eye(3)[k]) for k in range(3)]
    gens = [[UP3], [UP3, u[PROPPED[0]]], [UP3, *(u[k] for k in PROPPED)]]
    # one camera for all three panels, taken from the widest of them, so the
    # three are comparable and every arrow stands on the near half of the ball
    view = axis(gens[-1])
    # measured off the same tessellation that is painted, so the caption cannot
    # drift from the picture
    _, mid = shell()
    share = [100 * spanned(g, mid).mean() for g in gens]
    said = ["one force  \u00b7  the floor\nit reaches one direction\nno area at all",
            "two forces\nthe arc between them, 90 deg\nstill no area at all",
            "three forces\nthe patch they close on\nan eighth of the ball, 12.5 %"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.0), dpi=210, facecolor=PAPER,
                             subplot_kw=dict(projection="3d", computed_zorder=False))
    for ax, g, t in zip(axes, gens, said):
        ax.set_facecolor(PAPER)
        ball(ax, g, view, t)
    fig.subplots_adjust(0.0, 0.0, 1.0, 0.80, 0.0)
    fig.savefig(out, facecolor=PAPER)
    plt.close(fig)
    trim(out)
    return share, view[1], view[2]


# ---------------------------------------- what the angle between two pushes buys

# 180, 175 and 90 degrees apart, drawn twice each: the scene, and the ball. The
# whole point is that the first two are a degree apart in setup and nothing
# alike in result -- 175 gives an arc all but closed, 180 gives two isolated
# points, because non-negative combinations of u and -u are the multiples of u
PAIRS = (180.0, 175.0, 90.0)
# a horizontal push is at least ELEV off the view axis whatever its azimuth, so
# it keeps at least sin(ELEV) of its length on the page: 50 % at 30 deg, 71 % at
# 45. Every push in this figure is horizontal, and the 90 deg pair puts one of
# them right down the camera's own azimuth, so this row is raised until that
# worst case still reads -- 42 deg keeps 67 % of it
PAIR_ELEV, PAIR_AZIM = 42.0, -45.0
# ------------------------------------------- the two blue shoves on the top row
#
# A chock is a block like any other and can be leaned on only so hard. The cap
# is one body weight, F, the same unit the fan on the balls below is drawn in.
# The FLOOR is left UNCAPPED, and that is deliberate rather than overlooked: it
# is already carrying the weight, it is the one generator no arrangement has to
# earn, and `dimension.md` records why it has to stay a free generator. So the
# question the whole top row asks is
#
#     up - d  =  c0*up + c1*u1 + c2*u2,   every c >= 0,   c1, c2 <= F
#
# and the answer to it is what separates these three columns.
PAIR_CAP = 1.0
# swept over this many directions for the two percentages the captions carry.
# 20000 rather than the N = 40000 the rest of the script uses, and the reason is
# that it is the density which reproduces every closed form to the two places
# those captions print. With the force free the survivable share is exactly
# t/3.6 % -- twice the spherical triangle (t + 90 + 90 - 180)/720 that the cone
# itself covers, checked at 40, 60, 90, 120, 150 and 175 deg and coming back at
# 2.0000 every time -- so 175 deg is 48.611 % and 90 deg is 25.000 %. At n =
# 20000 the sweep gives 48.610 and 25.005, which print as 48.61 and 25.00; at n
# = 40000 it gives 48.615, which rounds AWAY from its own closed form, to 48.62.
# At 180 deg the closed form runs smoothly on to 50 % and the true answer is 0,
# which is the arc-versus-two-points discontinuity over again in a second place
PAIR_N = 20000
# the two disturbances the top row is drawn around, as an angle either side of
# straight away from the reader. TWO of them, the same two in all three panels,
# and everything about that is deliberate. An earlier pass swept the whole
# survivable set and thinned it to a fixed 34 deg spacing -- 10, 21 and 13
# directions, re-derived by re-running that sampler -- and the human's verdict
# was 有点儿丑，算了吧, it is ugly, drop it. It was arguing the wrong thing as well
# as looking wrong: a count of arrows says something only where the panels differ
# in DIMENSION, which is `solid`'s claim, and all three panels here have three
# generators. What is left is the narrow claim, which is worth more: HERE ARE TWO
# SHOVES, and 180 and 175 cannot hold them.
#
# Straight away from the reader is where they have to sit. The 90 deg pair's own
# chocks are turned by `apart`, which puts their bisector facing the reader, so
# the directions that pair survives -- {d.u1 <= 0, d.u2 <= 0} -- are the quarter
# sphere centred on the direction pointing AWAY. That is the `pressed` cost this
# figure has always paid, and it is why these two arrows keep 0.739 of their
# length on the page rather than all of it. 25 deg either side is the trade: any
# wider and one chock goes idle as the shove lines up with the other, any
# narrower and the two arrows stop being told apart on the page. At 25 they are
# 50 deg apart in the world and 70 deg apart as drawn, they land on the two
# different faces the reader can see, and the 90 deg pair holds each of them
# with its two chocks at 0.34 F and 0.94 F
SHOVE = 25.0
# where the question mark goes on a shove the pair cannot hold: on the arrow's
# own line, past its tail, so it reads as a query about that arrow and not as a
# mark on the floor. 1.00 against `arrow3`'s own 0.70 leaves 0.30 of clear air
QUERY_LN, QUERY_PT = 1.00, 26.0


def carried(gens, T, cap=PAIR_CAP, tol=1e-9):
    """Can these contacts produce `T` with every CHOCK held under `cap`?

    The floor is `gens[0]` and stays free; every other generator is capped. Hand
    it `cap=np.inf` and it answers the question `gap` answers, which is whether
    `T` is in the cone at all.

    **This must not be a plain solve, and the reason is the figure's own first
    column.** At 180 deg the two pushes are exactly opposite, the three
    generators are rank 2, and `np.linalg.solve` on that matrix returns
    coefficients of order 1e15 and no warning -- it will happily report a
    direction as held that the contacts cannot touch. So the rank is measured
    and the deficient case is answered properly: `lstsq` gives one solution, the
    null space gives the line of all the others, and each coefficient's box
    constraint cuts an interval out of that line. The contacts can do it exactly
    when the intervals overlap. One free direction is all this figure ever has;
    more than one is an assertion rather than a case.

    A linear program would answer the same question and was used to check this
    one: `scipy.optimize.linprog` with `A_eq = G.T` and
    `bounds = [(0,None),(0,F),(0,F)]`, run over all 20000 directions of all
    three columns both capped and free -- 120000 programs, and not one
    disagreement. The closed form is kept because it is vectorised and this
    script sweeps the sphere several times a run.
    """
    G = np.array(gens, float).T
    T = np.atleast_2d(np.asarray(T, float))
    top = np.array([np.inf] + [cap] * (G.shape[1] - 1))  # the floor is free, chocks are not
    sv, Vt = np.linalg.svd(G)[1:]
    c = np.linalg.lstsq(G, T.T, rcond=None)[0].T
    ok = np.linalg.norm(T - c @ G.T, axis=1) <= tol      # is T in the span at all
    if int((sv > sv[0] * 1e-9).sum()) == G.shape[1]:
        return ok & (c >= -tol).all(axis=1) & (c <= top + tol).all(axis=1), c
    assert int((sv > sv[0] * 1e-9).sum()) == G.shape[1] - 1, "more than one free direction"
    k = Vt[-1]                                           # the one direction c is free in
    lo, hi = np.full(len(T), -np.inf), np.full(len(T), np.inf)
    for i in range(G.shape[1]):
        if abs(k[i]) <= 1e-12:                           # that coefficient cannot move
            ok &= (c[:, i] >= -tol) & (c[:, i] <= top[i] + tol)
            continue
        a, b = (0.0 - c[:, i]) / k[i], (top[i] - c[:, i]) / k[i]
        lo, hi = np.maximum(lo, np.minimum(a, b)), np.minimum(hi, np.maximum(a, b))
    ok &= lo <= hi + tol
    # and one representative point of that overlap, for the caller that wants the
    # numbers rather than the verdict. The infinities are only ever the untouched
    # ends of an interval nothing bounded, so any finite stand-in serves
    lo, hi = np.where(np.isfinite(lo), lo, -1.0), np.where(np.isfinite(hi), hi, 1.0)
    return ok, c + np.clip(0.0, lo, hi)[:, None] * k


def shoves():
    """The two disturbances the top row is drawn around, and they are the row.

    Both are HORIZONTAL, which is what a shove against a plate on a frictionless
    floor is and what the chocks are there for, and both point away from the
    reader because that is where the 90 deg pair's survivable quarter sits (see
    SHOVE). Derived from the camera and from nothing else, so they follow it.
    """
    return np.array([[np.cos(a), np.sin(a), 0.0] for a in
                     np.radians(PAIR_AZIM + 180.0 + SHOVE * np.array([-1.0, 1.0]))])


def query(ax, at, v, ln=QUERY_LN, z=11.5):
    """A question mark on a shove this pair cannot answer.

    It stands on the arrow's own line, past the tail, so the reader reads it as
    a query about that arrow rather than as a mark on the floor. Blue, because
    it belongs to the blue family and not to the contacts.
    """
    ax.text(*(at - ln * np.asarray(v, float)), "?", color=BLUE, fontsize=QUERY_PT,
            fontweight="bold", ha="center", va="center", zorder=z)


def apart(deg):
    """Two horizontal pushes that many degrees apart, turned to where they read.

    Spinning the pair about the vertical changes nothing about the angle between
    them, so it is free, and two things want it. An arrow along the view axis
    has no length on the page -- and worse than none, a HORIZONTAL push drawn
    near it comes out pointing down the page, which is the one thing it is not.
    And the arc between them lives round the bisector, so if that bisector faces
    away the arc is drawn through the ball rather than across it.

    So the pair is turned to make the shorter of the two arrows as long as it
    can be, subject to the bisector still facing the reader. Derived, not
    chosen: change the camera and the three pairs follow it. For a pair 180 deg
    apart there is no bisector to constrain and the rule puts both ends on the
    silhouette, where an arrow keeps all of its length.
    """
    h = np.radians(deg) / 2
    best = None
    for phi in np.radians(np.arange(0.0, 360.0, 0.5)):
        u = [np.array([np.cos(phi + s * h), np.sin(phi + s * h), 0.0]) for s in (1, -1)]
        mid = u[0] + u[1]
        if np.linalg.norm(mid) > 1e-9 and mid @ VIEW / np.linalg.norm(mid) < 0.15:
            continue                                     # the arc would go round the back
        seen = min(np.sqrt(max(1.0 - (g @ VIEW) ** 2, 0.0)) for g in u)
        if best is None or seen > best[0] + 1e-12:
            best = (seen, u)
    return best[1]


def touching(R, centre, n):
    """Where a push `n` meets the body: its support point in the direction -n.

    Handed a face normal this averages the four vertices that tie and gives the
    face centre; handed anything else it gives the edge or the corner the plane
    square to `n` first touches. That is the whole of the physics of it, and it
    is also the answer to why 175 deg is harder to build than 180: two opposite
    pushes are two FACE contacts on a plain box, and nothing between 180 and the
    corner cone is, so 175 has to be taken on an EDGE -- or on a tapered part,
    or through friction. A box with parallel sides offers 180 and nothing near
    it.
    """
    v = np.array([[sx * HALF3[0], sy * HALF3[1], sz * HALF3[2]]
                  for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) @ R.T + centre
    d = v @ -n
    return v[d > d.max() - 1e-9].mean(axis=0)


def wall(q, n, half=0.34, depth=0.46, crest=0.56):
    """A block on the floor, its inner wall square to `n` and touching at `q`.

    Same cross-section as `chock` and the same guarantee -- every point of it
    lies on the far side of that wall from the body, so it touches and reaches
    into nothing -- but aimed by a direction rather than by a face index, which
    is what lets it take a push that no face of the body is square to.
    """
    t = np.cross(UP3, n)
    t /= np.linalg.norm(t)
    ends = [(q + s * half * t) * [1, 1, 0] for s in (-1, 1)]
    sect = [(0.0, 0.0), (0.0, crest), (0.30 * depth, crest), (depth, 0.0)]
    ring = [[e - a * n + b * UP3 for a, b in sect] for e in ends]
    polys = [np.array([ring[0][i], ring[1][i], ring[1][i + 1], ring[0][i + 1]])
             for i in range(len(sect) - 1)]
    polys += [np.array(ring[0]), np.array(ring[1][::-1]),
              np.array([ring[0][-1], ring[1][-1], ring[1][0], ring[0][0]])]
    at = max(ends, key=lambda e: e @ VIEW) + HALF3[LEVEL] * UP3
    return polys, at


def stage(ax, R, centre, pushes, floor, shots=()):
    """One scene: the plate lying there, and the two blocks holding it.

    `shots` are the two blue shoves, `(anchor, direction, held)` apiece, and they
    arrive with the FLOOR already counted among the generators -- see `pairs`.
    Without it there is nothing to ask: two horizontal pushes alone would have to
    produce `up - d` out of a purely horizontal wedge, which forces `d` straight
    up, and that is the one direction `measure` throws away. Swept, the pair
    alone leaves 20, 157 and 87 of 40000 directions, all of them tolerance
    residue on an empty set. The floor is what makes the question well posed, and
    it is standing right there in the picture anyway.

    The same two arrows are drawn in all three panels, and the only thing that
    changes is whether they carry a question mark. That is the whole top row.
    """
    solids = [wall(touching(R, centre, n), n) for n in pushes]
    faces = facing(R)
    ground(ax, floor)
    shadow(ax, R, centre, (), stood=[np.vstack([quad(R, centre, k, s)
                                                for k in range(3) for s in (-1, 1)])]
           + [np.vstack(p) for p, _ in solids])
    for (polys, at), n in zip(solids, pushes):
        # one of an opposed pair must come between the reader and the body; the
        # rest stand behind it. Which is which is just which way its wall looks
        near = 5 if -n @ VIEW > 0 else 2
        seen = lit(polys)
        ax.add_collection3d(Poly3DCollection(
            [P for P, _ in seen], facecolor=shaded([m for _, m in seen], SUPPORT3, *HAZE),
            edgecolor=SUPPORT_INK, lw=1.0, zorder=near))
        generator3(ax, at, n, near + 1)
    ax.add_collection3d(Poly3DCollection(
        [quad(R, centre, k, s) for k, s, _ in faces],
        facecolor=shaded([m for _, _, m in faces], BODY3, *SUN),
        edgecolor=INK, lw=1.1, zorder=4))
    ax.add_collection3d(Line3DCollection(silhouette(R, centre, faces), colors=INK,
                                         lw=2.4, zorder=4.4))
    com3(ax, centre)
    if len(shots):
        generator3(ax, foot(R, centre), UP3, 12)          # the floor, always free
        for at, d, ok in shots:
            arrow3(ax, at, d)
            if not ok:
                query(ax, at, d)


def pairs(out):
    """Three angles, each drawn as the scene and then as the ball."""
    global ELEV
    keep = ELEV
    ELEV = PAIR_ELEV
    look(PAIR_AZIM)
    R, centre = slab()
    sets = [apart(a) for a in PAIRS]
    # the blue family, and the one decision it forces: the FLOOR goes back into
    # the generator set. The balls below stay about the two pushes alone -- that
    # pairwise geometry is what the figure is for -- so the two rows answer
    # different questions and the captions have to say which
    gensets = [[UP3, *u] for u in sets]
    # TWO arrows, the same two in every panel, and the panels differ only in
    # whether the pair can hold them. An earlier pass swept the whole survivable
    # set and thinned it to a fixed angular spacing, giving 9, 16 and 21 arrows;
    # the human's verdict on that was 有点儿丑，算了吧 -- it is ugly, drop it -- and
    # it was arguing badly as well as looking bad. A count of arrows says
    # something only when the panels differ in DIMENSION, which is `solid`'s
    # claim and not this figure's: all three panels here have three generators.
    #
    # So the row makes the narrow claim instead, which is the better one. Two
    # shoves, drawn identically in all three panels because `anchors` ranges over
    # all three at once and each of the two lands alone on its own visible face.
    # The 90 deg pair holds both; the other two cannot, and each of their arrows
    # carries a question mark saying so
    drawn = [shoves()] * len(gensets)
    held = [carried(g, UP3 - d)[0] for g, d in zip(gensets, drawn)]
    # asserted rather than trusted, because the whole top row is this one line:
    # move the camera, the angles or SHOVE and the arrows follow, and if they
    # ever stop making the claim the figure is drawn to make, the run stops here
    assert [list(h) for h in held] == [[False, False], [False, False], [True, True]]
    faces = facing(R)
    land = [[landing(R, d, faces) for d in D] for D in drawn]
    shots = [list(zip(A, D, H)) for A, D, H in zip(anchors(R, centre, land), drawn, held)]
    ends = np.array([[at, at - 0.70 * d] for S in shots for at, d, _ in S])
    marks = np.array([at - QUERY_LN * d for S in shots for at, d, ok in S if not ok])

    # one floor and one frame for all three scenes, so the only thing that
    # changes across the row is the angle between the two pushes
    pts = [quad(R, centre, k, s) for k in range(3) for s in (-1, 1)]
    for pushes in sets:
        for n in pushes:
            pts.append(np.vstack(wall(touching(R, centre, n), n)[0]))
    # the floor has to reach under the arrow tails that dip below it, exactly as
    # `tile` does for the main scene, or they hang over bare paper -- and under
    # the question marks past them, which stand at the same height and would
    # otherwise be the one thing in the picture floating over the paper
    pts.append(ends[:, 1][ends[:, 1][:, 2] < 0.20])
    pts.append(marks[marks[:, 2] < 0.20])
    on = np.vstack(pts)[:, :2]
    lo, hi = on.min(axis=0) - 0.15, on.max(axis=0) + 0.15
    floor = np.array([[hi[0], hi[1], 0.0], [hi[0], lo[1], 0.0],
                      [lo[0], lo[1], 0.0], [lo[0], hi[1], 0.0]])
    r = np.cross(UP3, VIEW)
    r /= np.linalg.norm(r)
    page = np.array([r, np.cross(VIEW, r)])
    cube = np.abs(page).sum(axis=1)
    flat = np.vstack(pts + [floor, ends.reshape(-1, 3)]) @ page.T
    side = 1.06 * max((flat.max(axis=0) - flat.min(axis=0)) / cube)
    mid = (flat.max(axis=0) + flat.min(axis=0)) / 2 @ page

    # what the pair can put into the MIDDLE of its own arc, which is where the
    # 175 deg column collapses and the 90 deg one peaks. Measured with
    # `strongest` and written into the caption from that measurement, so the
    # words under the picture cannot drift from the arrows in it. An opposed pair
    # has no middle -- the bisector is the zero vector, there is no arc for it to
    # sit on -- and says so instead of quoting a number
    def middle(u):
        m = np.sum(u, axis=0)
        n = np.linalg.norm(m)
        return None if n < 1e-9 else strongest(u, m / n)

    # the caption carries BOTH rows, because they answer different questions: the
    # ball is the two pushes on their own, the scene has the floor with them.
    #
    # And it carries the capped number under the free one, which is the whole of
    # what the top row now says. An earlier pass quoted 1.7 % for the 180 deg
    # column, and that was a tolerance band reported as a result: at 180 the
    # three generators are rank 2, the survivable set is a CURVE, and the true
    # answer is 0.00 % -- what `measure` was counting was the width of the ring
    # it draws around a set of measure zero to make one comparable with an area.
    # Nothing here goes through that band any more; `carried` is exact, and the
    # 180 column comes back empty because it is empty
    swept = UP3 - sweep(3, PAIR_N)
    free = [100 * carried(g, swept, cap=np.inf)[0].mean() for g in gensets]
    capped = [100 * carried(g, swept)[0].mean() for g in gensets]
    told = [f"{a:.0f} deg apart\n"
            + ("two points, not an arc\nand each of them at the full F"
               if middle(u) is None else
               f"one arc, {a:.0f} deg of it\n"
               f"F at each end, {middle(u):.2f} F in the middle")
            + f"\nwith the floor, force free: {f:.2f} % held"
            + f"\neach chock capped at F: {c:.2f} %"
            for a, u, f, c in zip(PAIRS, sets, free, capped)]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.8), dpi=210, facecolor=PAPER,
                             subplot_kw=dict(projection="3d", computed_zorder=False))
    for col, (pushes, said, shot) in enumerate(zip(sets, told, shots)):
        top = axes[0][col]
        top.set_facecolor(PAPER)
        iso(top, mid - side / 2, mid + side / 2)
        # mplot3d leaves about two fifths of an axes empty around whatever it is
        # given; the balls already ask for it back through `zoom` and the scenes
        # have to as well, or the two rows are drawn at different scales
        top.set_box_aspect((side, side, side), zoom=1.62)
        stage(top, R, centre, pushes, floor, shot)
        top.set_title(said, color=INK, fontsize=13, pad=2, linespacing=1.5)
        # the fan is on here and nowhere else. Which directions the pair reaches
        # is only half of what these three columns differ in, and on its own it
        # is the misleading half: 175 deg apart reaches an arc all but a half
        # circle long and cannot push across the middle of it at all
        ball(axes[1][col], pushes, (VIEW, PAIR_ELEV, PAIR_AZIM), "", fan=True)
        axes[1][col].set_facecolor(PAPER)
        # the band between the two rows is the padding mplot3d leaves inside each
        # axes, not the space between them, so it closes with `zoom` and not with
        # `hspace`. The ball figure keeps its own 1.30; here it can be pushed
        # further because nothing is written under it
        axes[1][col].set_box_aspect((1, 1, 1), zoom=1.62)
    # 0.81 and not the 0.83 this had while the caption was four lines: the fifth
    # line grows UPWARDS from the axes, a line is 13 pt at 1.5 spacing = 57 px at
    # this dpi, and at 0.83 the top line of the 180 deg column came within 4 px of
    # the edge of the raster. `trim` cannot rescue that -- it crops to the ink and
    # cannot put back what fell off the paper. At 0.81 the top line clears by 20
    fig.subplots_adjust(0.0, 0.01, 1.0, 0.81, 0.0, -0.18)
    fig.savefig(out, facecolor=PAPER)
    plt.close(fig)
    trim(out)
    ELEV = keep
    look(BEST)
    _, mid3 = shell()
    # the area, and the two magnitudes the fan is drawn from. The end is measured
    # rather than assumed to be F: it is F only because a push direction can be
    # supplied by that one support working alone, and the 180 deg column is the
    # case where the decomposition is not unique and a careless solve would say
    # otherwise. The last three are the top row: what fraction survives with the
    # force free, what survives once each chock is capped at F, and the two blue
    # shoves' own coefficients in that panel
    return [(a, 100 * spanned(g, mid3).mean(), strongest(g, g[0]), middle(g), f, c,
             [(v, carried(G, UP3 - v, cap=np.inf), carried(G, UP3 - v)) for v in drawn[0]])
            for a, g, G, f, c in zip(PAIRS, sets, gensets, free, capped)]


def main():
    corners, centre, R = block()
    face = R @ np.array([0.0, -1.0])                     # the down-right face, outward
    push = -face                                         # a support there pushes up-left
    edge0, edge1 = corners[0], corners[1]
    a, b = edge0 + 0.42 * (edge1 - edge0), edge0 + 0.58 * (edge1 - edge0)
    pad = [a, b, [1.16, b[1]], [1.16, 0.0], [0.74, 0.0]]
    # the upper-left face: `push` is its outward normal, so the admissible arc is
    # symmetric about it and every surviving disturbance arrives from outside
    edge = (corners[3], corners[2])

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 6.2), dpi=210, facecolor=PAPER)
    for ax in axes:
        ax.set_facecolor(PAPER)
    w0 = draw(axes[0], corners, centre, edge, None, [UP])
    w1 = draw(axes[1], corners, centre, edge, (pad, (a + b) / 2, push), [UP, push])
    fig.subplots_adjust(0.01, 0.02, 0.99, 0.98, 0.02)
    out = figure_path('dimension_flat.png')
    fig.savefig(out, facecolor=PAPER)
    plt.close(fig)
    print(f"support push direction ({push[0]:+.3f},{push[1]:+.3f})")
    print(f"floor only          survivable disturbances span {w0:6.1f} deg of the circle")
    print(f"floor + 1 support   survivable disturbances span {w1:6.1f} deg of the circle")
    print(out)

    out2 = figure_path('normals.png')
    u = normals(corners, centre, R, out2)
    print(f"\nflat face   every contact pushes ({u[0]:+.3f},{u[1]:+.3f}) -- one direction, "
          f"parallel, wherever you touch it")
    print(f"fillet      the pushes fan into the centre of curvature, which is not "
          f"the centre of mass")
    print(out2)

    # the same scene from each of its four sides -- the camera is the only thing
    # that moves between them, so every number is the number for all four alike.
    # OFF: they were scaffolding for choosing BEST by looking, and that choice is
    # settled and argued in the handoff (only azim -45 leaves all three visible
    # faces separated in lightness; +45 makes two of them identical). Four files
    # that nothing reads any more. Set VARIANTS = True to get them back
    stem = figure_path('dimension_iso_%s.png')
    for tag, azim in zip("abcd", AZIMS) if VARIANTS else ():
        look(azim)
        # the row is laid out against the camera, so its crowding has to be read
        # at every azimuth and not only at the one dimension.png takes
        near, kin = solid(stem % tag)[2]
        print(f"{stem % tag}   azim {azim:+.1f}  elev {ELEV:.3f}   row clears "
              f"{near[0]:.3f} (panel {near[5]}, {near[1]}), orange {kin[3]:.3f}")

    out3 = figure_path('dimension.png')
    look(BEST)
    gens, u, (near, kin) = solid(out3)

    # what the contacts can PRODUCE, which is the row drawn on the plate. Every
    # support is capped at F and the three push directions here stand mutually
    # square, so a direction in the cone decomposes as c_i = v.u_i and reaches
    # F / max(c) -- exact, and printed against `strongest`, `spanned` and the
    # closed form together so the row cannot drift from the set the rest of the
    # figure counts
    print("\nthe row on the body -- how hard the contacts can push, each capped at F")
    G = np.array(gens[-1], float)
    print(f"      the three generators are mutually square: pairwise dots "
          + ", ".join(f"{G[i] @ G[j]:+.3f}" for i, j in itertools.combinations(range(3), 2)))
    R0, c0 = slab()
    said = ("one force   ", "two forces  ", "three forces")
    for tag, g in zip(said, gens):
        _, mag, dirs, _ = laid(R0, c0, g)
        # the row against the solver the rest of the figure is counted with, on
        # every arrow drawn: each one is IN the cone `spanned` paints, and the
        # magnitude `strongest` hands back is F / max(c) with c read off the
        # generators directly -- which it is only because they stand square
        assert all(spanned(g, v)[0] for v in dirs)
        assert all(abs(m - 1 / max(v @ np.array(g, float).T)) < 1e-9
                   for v, m in zip(dirs, mag))
        seen = [onpage(v, VIEW, 0.0) for v in dirs]
        print(f"  {tag}   {len(mag)} arrow(s)   "
              + "  ".join(f"{m:.4f}" for m in mag) + " F")
        print(f"                   drawn at   "
              + "  ".join(f"{ROW_LN * m:.3f}" for m in mag) + " on the page; the "
              f"projection leaves at least {min(seen):.3f} of a length")
    print(f"      exactly: F at every push, sqrt(2) F = {np.sqrt(2):.4f} between any two "
          f"of them,\n               sqrt(3) F = {np.sqrt(3):.4f} down the cone's own axis -- "
          f"which is {np.degrees(np.arccos(min(1.0, G.sum(axis=0) @ VIEW / np.sqrt(3)))):.1f} deg "
          f"off the camera\n               and keeps "
          f"{onpage(G.sum(axis=0) / np.sqrt(3), VIEW, 0.0):.3f} of its length, so the rim is "
          f"what is drawn")
    print(f"      the row clears {near[0]:.3f} on the page (panel {near[5]}, {near[1]}), "
          f"and {kin[3]:.3f} from the other ORANGE arrows (panel {kin[5]}, {kin[4]});\n"
          f"      its outermost tail leaves {near[2]:.3f} of the face's half extent over")

    out4 = figure_path('dimension_ball.png')
    share, elev, azim = balls(out4)
    print(f"\nball  seen down the cone's own axis, elev {elev:.2f} azim {azim:.2f}")
    print(f"      the arrows reach   one force {share[0]:6.3f} %   two {share[1]:6.3f} %   "
          f"three {share[2]:6.3f} %   (exactly 0, 0 and 12.5)")
    print(out4)

    out5 = figure_path('dimension_pairs.png')
    print()
    # the area is the same 0.000 % three times over, which is the row's first
    # point; what the fan draws is the second, and it is the one that separates
    # them. Each support is capped at F, a direction in the cone is a unique
    # non-negative combination of the two pushes, and scaling it scales both
    # coefficients together -- so the cap binds on the larger and the direction
    # reaches F / max(c). At a push itself that is F, the WEAKEST place on any of
    # these arcs; at the bisector it is 2 cos(t/2) F, which is sqrt(2) at 90 deg
    # and next to nothing at 175
    for deg, pct, end, mid, free, capped, shot in pairs(out5):
        print(f"two pushes {deg:5.1f} deg apart   area {pct:.3f} %   "
              + ("two points -- the cone is a LINE" if deg >= 180
                 else f"an arc {deg:.0f} deg long"))
        print(f"                             capped at F, it reaches {end:.4f} F "
              f"at each end and "
              + ("has no middle to reach" if mid is None else
                 f"{mid:.4f} F across the middle\n"
                 f"                             "
                 f"(2 cos(t/2) = {2 * np.cos(np.radians(deg) / 2):.4f})"))
        # and the top row: the whole sphere, then the two arrows drawn on it. The
        # generators are rank 2 at 180 deg, where a plain solve returns
        # coefficients of order 1e15 and reports a direction as held that the
        # contacts cannot touch, so `carried` measures the rank first
        print(f"      with the floor   {free:6.2f} % of disturbances held with the "
              f"force free, {capped:5.2f} % with each chock capped at F"
              + (f"  ({free / capped:.1f}x)" if capped > 1e-9 else ""))
        for v, (okf, cf), (okc, cc) in shot:
            print(f"      shove ({v[0]:+.3f},{v[1]:+.3f},{v[2]:+.3f})   "
                  + ("not in the cone at all -- no solution, capped or free"
                     if not okf[0] else
                     "c = (" + ", ".join(f"{x:5.2f}" for x in cf[0]) + ")   "
                     + ("HELD" if okc[0] else
                        f"over the cap by {max(cf[0][1:]):.1f}x -- NOT held")))
    print(out5)
    # lying down, the three answers are exact and can be written out in closed
    # form, which is new: every push is horizontal, so `d` survives exactly when
    # it leans away from each of them. The floor alone leaves one direction, one
    # chock leaves a half great circle, two leave the quarter of the sphere that
    # runs away from both. The sweep has to reproduce 180.0 and 25.000, less the
    # band `measure` cuts out around straight up
    print()
    h = np.degrees(np.sqrt(4 * np.pi / N))
    print(f"exactly           the pushes lie in the floor, so the arc is a half great "
          f"circle, 180.0 deg,\n                  and the patch is the quarter sphere "
          f"{{d.u1 <= 0, d.u2 <= 0}}, 25.000 % --\n                  against which the "
          f"sweep loses the 4h = {4 * h:.2f} deg cap around straight up")

    d, ok = measure(gens[0], N)
    pieces = clumps(d[ok], np.deg2rad(6))
    where = [v / np.linalg.norm(v) for v in (d[ok][c].mean(axis=0) for c in pieces)]
    print(f"\nfloor only          {len(pieces)} direction survives, "
          + " ".join("(%+.3f,%+.3f,%+.3f)" % tuple(v) for v in where)
          + f" -- straight down; {ok.sum()} of {N} samples fall within a step of it")
    d, ok = measure(gens[1], N)
    # the arc has a length that can be written down without sweeping anything.
    # Its two ends are the two edges of the wedge: the floor's own ray gives
    # straight down, the chock's ray gives the direction straight down REFLECTED
    # in the push, and a reflection turns it through twice the angle between
    # them. So the arc is twice the push's tilt from vertical -- 180 exactly,
    # since the push is horizontal -- and that far end is straight up, which is
    # the one direction `measure` throws away. The sweep should therefore come
    # back SHORT by the 4h cap and not long by a step, which is the opposite of
    # what it did while the plate stood on a corner
    lean = np.degrees(np.arccos(u[PROPPED[0]][2]))
    print(f"floor + 1 support   an arc {spread(d[ok]):.1f} deg long, and no area at all "
          f"({ok.sum()} of {N} samples); twice the push's {lean:.1f} deg lean "
          f"predicts {2 * lean:.1f}, less the {4 * h:.2f} deg cap = "
          f"{2 * lean - 4 * h:.1f}")
    d, ok = measure(gens[2], N)
    print(f"floor + 2 supports  a patch, {100 * ok.sum() / N:.1f} % of the sphere "
          f"({ok.sum()} of {N} samples)")

    print(f"\nnine times the samples   {'n = %d' % N:>20}{'n = %d' % (9 * N):>22}")
    for tag, row in zip(("point", "arc  ", "patch"), refine(gens, (N, 9 * N))):
        print(f"  {tag}                 "
              + "".join(f"{h:9d} hits {p:7.3f} %" for h, p in row))
    print(out3)


if __name__ == "__main__":
    main()
