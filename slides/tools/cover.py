"""Adding supports one at a time, and watching the sphere go white.

Translation only, so a contact contributes nothing but the direction it pushes,
and the floor is a free generator pointing straight up. What the supports plus
the floor have to produce, for one push `d` at full strength, is

    T = (0,0,1) - K*d          one weight up, minus the push

so a push is answered exactly when its `T` lies in the cone spanned by the
directions chosen so far together with straight-up. Choosing supports is
therefore covering a patch of the sphere with a cone, and it is worth watching
happen: each new direction whitens the part of the patch it brings inside the
cone, and the job is done when nothing red is left.

Growing beats shrinking here. The answer is two or three directions out of a few
hundred, so growth stops in three steps where removal would walk the whole list.

    python slides/tools/cover.py A1-f
    python slides/tools/cover.py A1-f --poses 0 3
"""
from __future__ import annotations

import argparse
import io
import itertools
import shutil

import matplotlib
import mujoco
from yup_render import Renderer as YUpRenderer
import numpy as np
import coordinates as COORD
import trimesh
from PIL import Image, ImageDraw
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402
from matplotlib.lines import Line2D                        # noqa: E402
from mpl_toolkits.mplot3d import proj3d                    # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection    # noqa: E402

from common import mat_to_quat_wxyz, obj_path, read_json, write_json  # noqa: E402
from work_regions import SCENE                             # noqa: E402
from disturbances import _font                             # noqa: E402
from opposing_supports import tiling                       # noqa: E402
from shrink_support import angled_pushes, paint
PUSH_RGBA = (0.13, 0.35, 0.92, 1.0)
HOLD_RGBA = (0.92, 0.35, 0.05, 1.0)    # noqa: E402
from supports import CONTACT_EPS, region_mask              # noqa: E402
from work_regions import BOUNDARY_EDGE, gun_directions, spot_region   # noqa: E402

INK, MUTED, PAPER, EMPTY = "#1b1b1a", "#6b6b66", "#ffffff", "#e6e6e0"
NEED, DONE = "#B02A26", "#cfd6cf"
HAVE = "#E08A24"
HAZE = "#f3ddc0"
PUSH_BLUE = "#2563EB"
ARROW = (0.85, 0.30, 0.05, 1.0)
UP = np.array([0.0, 1.0, 0.0])
TOL = 1e-7
RAY_DOT = 0.02       # a ray shorter than this is drawn as a dot: matplotlib's own
                     # arrow head is a fixed FRACTION of the shaft, so a short one
                     # is not a small arrow, it is an invisible one
ELEV, AZIM = 14.0, -62.0      # one viewpoint for the whole figure, spheres and part alike
TRIAD_RGBA = (0.24, 0.24, 0.22, 1.0)
TRIAD_INK = "#3d3d3a"
AMBIENT = 0.55       # how dark a face turned fully away from the light is allowed to
                     # get. A relief on a sphere turns through 180 degrees, so a
                     # Lambert term used raw takes half the ball to black and takes
                     # the colour -- which is the panel's IDENTITY, blue or red or
                     # orange -- with it. The shading here only has to make a step
                     # legible, so it runs from .55 to 1 and never loses the hue.


def covered(G: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Which targets lie in the cone spanned by the columns of G.

    Carathéodory says three generators are enough for any point of a cone in
    three dimensions, and there are only ever a handful of generators here, so
    every subset of up to three can be tried outright -- one batched solve each,
    no linear program.
    """
    hit = np.zeros(len(T), bool)
    m = G.shape[1]
    for size in (1, 2, 3):
        if size > m:
            break
        for pick in itertools.combinations(range(m), size):
            A = G[:, list(pick)]
            lam = np.linalg.pinv(A) @ T.T                  # size x n
            resid = np.linalg.norm(A @ lam - T.T, axis=0)
            scale = np.maximum(np.linalg.norm(T, axis=1), 1e-12)
            hit |= (resid <= TOL * 1e3 * scale) & (lam >= -TOL).all(axis=0)
            if hit.all():
                return hit
    return hit


def grow(targets: np.ndarray, avail: np.ndarray, limit: int = 8):
    """Add the direction that answers the most that are still unanswered."""
    chosen, order = [], []
    G = UP.reshape(3, 1)
    done = covered(G, targets)
    order.append(done.copy())
    while not done.all() and len(chosen) < limit:
        best, gain = None, 0
        for j in range(len(avail)):
            if j in chosen:
                continue
            got = covered(np.hstack([G, avail[j].reshape(3, 1)]), targets)
            n = int((got & ~done).sum())
            if n > gain:
                best, gain = j, n
        if best is None:
            break
        chosen.append(best)
        G = np.hstack([G, avail[best].reshape(3, 1)])
        done = covered(G, targets)
        order.append(done.copy())
    return chosen, order, done


# ------------------------------------------------------------------ drawing ---

def shot(name, T, parts, px, pushes, holds_, triad=True):
    """The part, the pushes on it, and the supports placed so far.

    One camera for every pose and every step, and the same one the spheres use,
    so a direction that points up-and-left on a sphere points up-and-left on the
    part too. The price is that a support on the far side is sometimes behind
    the part; a second view would break the correspondence, which is worse.
    """
    obj = obj_path(name)
    xml = SCENE.replace(
        '<material name="work" rgba="0.16 0.68 0.40 1" specular="0.15" shininess="0.2"/>',
        '<material name="work" rgba="0.55 0.82 0.62 1" specular="0.1"/>').replace(
        '<material name="rest" rgba="0.85 0.79 0.68 1" specular="0.1"/>',
        '<material name="rest" rgba="0.90 0.89 0.85 1" specular="0.1"/>\n'
        '    <material name="spare" rgba="0.90 0.89 0.85 1" specular="0.1"/>').format(
        assets="\n".join(f'    <mesh name="p{i}" file="{f}"/>' for i, (f, _) in enumerate(parts)),
        geoms="\n".join(f'      <geom type="mesh" mesh="p{i}" material="{m}"/>'
                        for i, (_, m) in enumerate(parts)),
        pos=" ".join(f"{v:.9g}" for v in T[:3, 3]),
        quat=" ".join(f"{v:.9g}" for v in mat_to_quat_wxyz(T[:3, :3])))
    tmpx = obj / ".cover.xml"
    tmpx.write_text(xml)
    try:
        model = mujoco.MjModel.from_xml_path(str(tmpx))
    finally:
        tmpx.unlink(missing_ok=True)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mesh = trimesh.load(obj / "mesh.stl", force="mesh")
    V = np.asarray(mesh.vertices) @ T[:3, :3].T + T[:3, 3]
    lo, hi = V.min(axis=0), V.max(axis=0)
    size = float(np.linalg.norm(hi - lo))
    cam = mujoco.MjvCamera()
    # mujoco's angles name the direction the camera looks ALONG, matplotlib's name
    # where the eye stands, so the same view is the elevation negated and the
    # azimuth turned half a circle -- checked by drawing world +X in both.
    cam.azimuth, cam.elevation = AZIM + 180.0, -ELEV
    cam.lookat[:] = (lo + hi) / 2
    fov = np.deg2rad(model.vis.global_.fovy / 2)
    cam.distance = 1.25 * size / 2 / np.tan(fov)
    a, e = np.deg2rad(cam.azimuth), np.deg2rad(cam.elevation)
    fwd = np.array([np.cos(e) * np.cos(a), np.sin(e), np.cos(e) * np.sin(a)])
    right = np.array([np.sin(a), 0.0, -np.cos(a)])
    up = -np.cross(right, fwd)
    eye = np.asarray(cam.lookat) - cam.distance * fwd
    with YUpRenderer(model, px, px, max_geom=3000) as r:
        r.update_scene(data, camera=cam)
        scn = r.scene

        def arrow(q, u, L, w, rgba, ball=0.0):
            # mjv_connector fills only HALF the segment it is handed when the geom
            # is an arrow, so passing (q - u*L, q) drew every push and every
            # support stopping short of the surface it is meant to touch. Handing
            # it a segment centred on q puts the tip exactly on the contact.
            q, u = np.asarray(q), np.asarray(u)
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                                np.zeros(9), np.array(rgba, np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, w, q - u * L, q + u * L)
            scn.ngeom += 1
            if ball:
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, ball),
                                    np.asarray(q), np.eye(3).ravel(),
                                    np.array(rgba, np.float32))
                scn.ngeom += 1

        for q, u in pushes:
            arrow(q, u, 0.13 * size, 1.8e-3, PUSH_RGBA)
        for q, u in holds_:
            arrow(q, u, 0.34 * size, 6.5e-3, HOLD_RGBA, ball=7e-3)

        # the world frame, held near the camera in the bottom-left corner. Near,
        # because out at the part's own depth that corner is under the floor
        # plane; bottom-left, because in the two top corners the perspective
        # bends x and y until they lie on top of each other. On screen it is a
        # fixed size wherever it is put, so only those two things decide.
        if not triad:
            return Image.fromarray(r.render())
        depth = 0.45 * cam.distance
        frame = depth * np.tan(fov)                  # half the frame, out there
        org = eye + depth * fwd - frame * (0.80 * right + 0.62 * up)
        L = 0.20 * frame
        for u in np.eye(3):
            arrow(org + L * u, u, L, 0.05 * L, TRIAD_RGBA)
        im = Image.fromarray(r.render())

    # mujoco geoms carry no text, so the three letters are projected by hand
    dr = ImageDraw.Draw(im)
    for u, lab in zip(np.eye(3), "xyz"):
        v = org + 1.32 * L * u - eye
        s = np.array([v @ right, v @ up]) / (v @ fwd) / np.tan(fov)
        dr.text(((0.5 + 0.5 * s[0]) * px, (0.5 - 0.5 * s[1]) * px), lab,
                fill=tuple(int(255 * c) for c in TRIAD_RGBA[:3]),
                font=_font(int(px * 0.028)), anchor="mm")
    return im



def neighbours(ico):
    """Tile adjacency on the icosphere, as a sparse operator."""
    fa = np.asarray(ico.face_adjacency)
    n = len(ico.faces)
    r, c = np.r_[fa[:, 0], fa[:, 1]], np.r_[fa[:, 1], fa[:, 0]]
    return coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n)).tocsr()


def sheet(A, idx, rounds=3):
    """The tiles a continuous set really covers, given a finite sample of it.

    A few thousand samples spread over a few thousand tiles miss a third of the
    tiles they are sitting inside -- coupon collector, not geometry -- so a
    sheet drawn straight from the sample comes out as speckle. Dilating and
    then eroding the same number of times fills those pinholes and puts the
    boundary back where the sample left it, which is the only part of the
    outline that carries information.
    """
    m = np.zeros(A.shape[0], bool)
    m[idx] = True
    for _ in range(rounds):
        m |= (A @ m) > 0
    for _ in range(rounds):
        m &= ~((A @ ~m) > 0)
    return m


def screen_axes():
    """The camera's own three vectors: eye, screen right, screen up.

    One camera serves the whole figure (`ELEV`, `AZIM`), so every hand-projected
    thing on it -- a label offset, the triad, a light -- is some combination of
    these three, and they are worth having in one place rather than re-derived at
    each site with the sign of `cross` guessed again.
    """
    a, e = np.deg2rad(AZIM), np.deg2rad(ELEV)
    rt = np.array([-np.sin(a), 0.0, np.cos(a)])
    ey = np.array([np.cos(e)*np.cos(a), np.sin(e), np.cos(e)*np.sin(a)])
    return COORD.polar(ey), COORD.polar(rt), COORD.polar(np.cross(ey, rt))


def _light():
    """Where the light on a relief comes from: over the viewer's LEFT SHOULDER.

    Not a taste. A reader reads a shaded surface by assuming the light is above,
    and reads it as a HOLLOW if it is not -- the convex/concave flip is the oldest
    illusion in relief drawing. Above and to the left is the assumption every
    engraver has used since the light came from a window, so the bumps read as
    bumps. It is put slightly toward the eye as well, so that nothing facing the
    reader goes fully dark and the panel's colour survives everywhere.
    """
    ey, rt, up = screen_axes()
    v = ey - 0.65 * rt + 0.75 * up
    return v / np.linalg.norm(v)


LIGHT = _light()


def relief_shell(ico, h, colour, bare=EMPTY, ambient=AMBIENT):
    """The sphere with each tile pushed OUT by its own value: a height map, not paint.

    `h` is one number per tile, in SPHERE RADII, and NaN where the tile carries no
    value at all. A tile with a value sits at `1 + h`; a tile without one sits at
    exactly 1 in `bare`. That difference is the point of the element: "no push can
    go that way" and "a short arm" are different facts (METHOD s3.0), and a colour
    ramp says them with two neighbouring shades where relief says them with a cliff.

    PER FACE, NOT PER VERTEX. The data is one number a tile, so each face is
    displaced as a whole and the shell comes out stepped. Averaging to the vertices
    would draw a smooth surface through numbers that were never measured, and worse,
    it would drag the edge of the no-value region inward by half a tile -- the one
    boundary on the panel that is exact.

    WHICH MEANS THE WALLS ARE THE PICTURE. Displacing a face radially is a scaling
    about the origin, so the displaced face has the SAME normal as the tile it came
    from: the tops of a relief shade exactly like a plain sphere and carry no shape
    information at all. Everything the eye can use is in the RISERS between
    neighbours -- one quad per adjacent pair whose heights differ, closing the shell
    -- so they are built, lit by their own normals, and given the taller neighbour's
    colour, the way a terrace step belongs to the terrace above it. Left out, they
    would be cracks, and the cliff at the edge of the no-value region -- the biggest
    step on the panel -- would be a hole.

    Returns the polygons as one `(n, 4, 3)` array and one lit rgba a polygon, ready
    for a single `Poly3DCollection`.
    """
    v = ico.vertices / np.linalg.norm(ico.vertices, axis=1, keepdims=True)
    h = np.asarray(h, float)
    has = np.isfinite(h)
    r = np.where(has, 1.0 + h, 1.0)
    # `colour` may be ONE colour for the whole shell (every existing caller), or
    # one rgba PER TILE -- a relief whose height is one field and whose colour is
    # another (e.g. the demand's shape coloured by whether it is covered)
    if not isinstance(colour, str) and np.ndim(colour) == 2:
        rgba = np.asarray(colour, float).copy()
    else:
        rgba = np.tile(matplotlib.colors.to_rgba(colour), (len(r), 1))
    rgba[~has] = matplotlib.colors.to_rgba(bare)

    def lit(poly, col):
        """Lambert on a polygon's OWN normal, oriented away from the origin."""
        n = np.cross(poly[:, 1] - poly[:, 0], poly[:, 2] - poly[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-15)
        out = np.asarray(col, float).copy()
        out[:, :3] *= (ambient + (1 - ambient)
                       * np.clip(n @ LIGHT, 0.0, 1.0))[:, None]
        return out

    top = v[ico.faces] * r[:, None, None]
    n = np.cross(top[:, 1] - top[:, 0], top[:, 2] - top[:, 0])
    flip = np.einsum("ij,ij->i", n, top.mean(axis=1)) < 0
    top[flip] = top[flip][:, ::-1]                 # outward-facing, for the light
    cols = [lit(top, rgba)]
    # the tile is a TRIANGLE and the riser below is a QUAD, and they have to go into
    # one collection to sort against each other -- so the triangle is written as a quad
    # with its last vertex repeated, which draws the same triangle. Not a nicety:
    # handed a RAGGED list, `Poly3DCollection.get_vector` pads it into a rectangular
    # array with `np.empty`, and `add_collection3d` then autoscales over the raw array,
    # UNINITIALISED PADDING AND ALL. Whatever is in that memory is usually a harmless
    # number and once in about thirty panels it is a NaN, and then the axes refuses its
    # own limits. A regular array is passed straight through untouched.
    top = np.concatenate([top, top[:, 2:3]], axis=1)

    fa, fe = np.asarray(ico.face_adjacency), np.asarray(ico.face_adjacency_edges)
    step = r[fa[:, 0]] != r[fa[:, 1]]
    i, j = fa[step, 0], fa[step, 1]
    a, b = v[fe[step, 0]], v[fe[step, 1]]
    wall = np.stack([r[i, None] * a, r[i, None] * b,
                     r[j, None] * b, r[j, None] * a], axis=1)
    # A riser lies in the plane through the origin and its edge, so "outward" cannot
    # orient it -- its normal is square to the radius. It faces its SHORTER
    # neighbour, which is the side the drop is on and the side a reader sees.
    c = v[ico.faces].mean(axis=1)
    tall = np.where(r[i] >= r[j], i, j)
    n = np.cross(wall[:, 1] - wall[:, 0], wall[:, 2] - wall[:, 0])
    flip = np.einsum("ij,ij->i", n, c[np.where(r[i] >= r[j], j, i)] - c[tall]) < 0
    wall[flip] = wall[flip][:, ::-1]
    cols.append(lit(wall, rgba[tall]))
    # ONE collection, because matplotlib depth-sorts WITHIN a collection and not
    # between them: tops and risers in two would sort as two shells and the far side
    # of the ball would come out in front of the near side of the relief
    return np.concatenate([top, wall]), np.vstack(cols)


def globe(ax, ico, fills, dots, quills, marks, title, triad=False, bars=(),
          reach=1.45, rays=(), rays_front=False, relief=None, rings_front=False,
          weight=True):
    """The ball of directions, painted where a set lives, dotted where it doesn't.

    Which of the two a set gets is not a style choice, it is what the set is.
    A continuous set -- every direction the process can push, its reverse, what
    the supports still owe -- really is a sheet on the sphere, so it is painted
    as one. A flat face is the other thing: every point of it carries the SAME
    normal, so the whole face lands on a single direction and stays a single
    dot no matter how large the face is. Only the fillets sweep, and they get a
    thin painted band.

    Paint goes on as one collection with a colour per tile, so the sphere sorts
    against itself correctly; the arrows ride outside it and only say which way
    the forces point, so a couple of dozen is plenty.

    `bars` is the one element that carries a LENGTH. A quill says which way; a bar
    says which way and how far, as a radial segment from `1 + r0` to `1 + r1` in
    sphere radii with its head at `r1`, so a set whose members differ only in
    magnitude -- the moment a given push direction can make, which depends on where
    it is applied -- can be drawn without inventing a second sphere for it. `r1`
    may be the smaller of the two, and usually should be: it is the end the head
    lands on, so it is the end the reader looks at, and it should be the end that
    carries the information. It is the only element that
    can leave the frame, which is why `reach` exists: it is the half-extent of the
    axes, and its default is the value every existing panel was drawn at, so
    passing bars is the only thing that ever changes the framing. Pass one `reach`
    for a whole page or the spheres on it come out different sizes.

    `rays` is the other end of that: not one element per direction but ONE vector
    of the space itself, `(vec, colour, label)`, drawn from the ORIGIN outward in
    sphere radii. A fill, a dot, a quill and a bar all hang at a direction, so
    none of them can say "this single vector, here, with this length" -- which is
    what a constant term of a demand is. A ray whose length is under `RAY_DOT` is
    drawn as a dot on the origin instead, because an arrow of no length draws
    nothing at all and nothing reads as "the panel forgot", not as "this is zero".
    That dot is put IN FRONT of the sphere by hand; a ray with real length is an
    ordinary 3d arrow and the paint sorts in front of it, so most of one will be
    buried in the ball -- which is where it honestly is.

    `rays_front` gives the SHORT-BUT-NOT-ZERO ray the same treatment as the zero
    one, projecting every ray by hand and drawing it flat in front of the paint.
    It exists because burying is honest only while the ball means something to the
    ray. When the panel's whole subject IS one vector -- a moment of a few tenths of
    a radius, on a ball whose directions index pushes and not moments -- the ray does
    not belong to the sphere at all, the sphere is there as a ruler, and sorting the
    subject behind the ruler loses it: a 0.22-radius arrow inside a 30%-opaque
    hemisphere reads as a smudge. Off by default, so nothing already drawn moves.

    `relief` is the third element that carries a LENGTH, and unlike a bar or a ray
    it carries one at EVERY direction at once: `(h, colour)`, or `(h, colour, bare)`,
    or `(h, colour, bare, rings)`, where `h` is a value per tile in sphere radii and
    the shell is pushed out by it (`relief_shell`, which says why per face and why
    the risers are drawn). It REPLACES the flat shell rather than joining it: paint
    and relief are two ways to draw one field, the sphere can only be at one radius
    at a time, and a panel that did both would be saying the same number twice. So
    `fills` and `relief` are alternatives, and a panel passing neither is unchanged.
    `rings` are radii to draw a circle at, in the same units as `h` and square to the
    eye, which is what makes them a RULER: a relief is radial, so a ring at `h` is
    the silhouette of the sphere of that height, and the field stands above it in
    the picture exactly where it stands above it in the number.

    `rings_front` is to a ring what `rays_front` is to a ray, and it is there for the
    same reason. A ring drawn as an ordinary 3d circle is a circle THROUGH the ball,
    so the ball sorts in front of all of it and a ruler that is entirely buried is
    not a ruler -- which is what happens the moment the relief stands taller than the
    ring, i.e. exactly when the reader wants to read it. Projected by hand and drawn
    flat in front, it is a ruler laid over the picture, and the relief's silhouette
    crosses it where the number crosses it. Off by default, so nothing already drawn
    moves.

    `weight` is the black arrow through the middle, the one thing on the panel that
    is not about the sphere's own directions: it says which way the workpiece's own
    weight pulls. Every force-space ball wants it. A ball whose points are TURNING
    AXES does not -- a force arrow in a space of moments is a category error, and
    gravity's moment about the centre of mass is zero, which is the whole reason the
    centre of mass is where these moments are taken. So it can be turned off, and it
    is on by default.
    """
    v = ico.vertices / np.linalg.norm(ico.vertices, axis=1, keepdims=True)
    ring_r = (relief[3] if relief is not None and len(relief) > 3 else ())
    if relief is not None:
        polys, cols = relief_shell(ico, relief[0], relief[1], *relief[2:3])
        _, rt, up = screen_axes()
        th = np.linspace(0, 2 * np.pi, 361)
        for rh in (() if rings_front else ring_r):
            P = (1.0 + rh) * (np.cos(th)[:, None] * rt + np.sin(th)[:, None] * up)
            ax.plot(P[:, 0], P[:, 1], P[:, 2], color=MUTED, lw=.8, alpha=.55,
                    ls=(0, (5, 4)))
        ax.add_collection3d(Poly3DCollection(polys, facecolors=cols,
                                             edgecolors="none", zsort="average"))
    else:
        face_rgba = np.tile(matplotlib.colors.to_rgba(EMPTY, .30), (len(ico.faces), 1))
        for idx, col, alpha in fills:
            if len(idx):
                face_rgba[np.asarray(idx)] = matplotlib.colors.to_rgba(col, alpha)
        ax.add_collection3d(Poly3DCollection(0.99 * v[ico.faces],
                                             facecolors=face_rgba,
                                             edgecolors="none", zsort="average"))
    for dot in dots:
        pts, col, size, outward = dot[:4]
        if not len(pts):
            continue
        # an optional fifth item is the dot's OUTLINE. White is right while the dot's
        # colour is the panel's one orange; it disappears when the dot's colour is a
        # step of a ramp and the step is a pale one, and a marker that cannot be seen
        # cannot say which step it is
        q = (1.015 if outward else -1.015) * np.asarray(pts)
        ax.scatter(q[:, 0], q[:, 1], q[:, 2], s=size, c=col, alpha=.95,
                   edgecolors=dot[4] if len(dot) > 4 else "white",
                   linewidths=.9, depthshade=False)
    for dirs, col, outward in quills:
        if not len(dirs):
            continue
        d = np.asarray(dirs)
        tail, vec = ((1.03 * d, 0.40 * d) if outward else (-1.43 * d, 0.40 * d))
        ax.quiver(tail[:, 0], tail[:, 1], tail[:, 2], vec[:, 0], vec[:, 1], vec[:, 2],
                  color=col, linewidth=1.5, arrow_length_ratio=.40, alpha=.95)
    for dirs, r0, r1, col in bars:
        if not len(dirs):
            continue
        d = np.asarray(dirs, float)
        tail = (1.0 + np.asarray(r0, float))[:, None] * d
        vec = (np.asarray(r1, float) - np.asarray(r0, float))[:, None] * d
        ax.quiver(tail[:, 0], tail[:, 1], tail[:, 2], vec[:, 0], vec[:, 1], vec[:, 2],
                  color=col, linewidth=1.1, arrow_length_ratio=.34, alpha=.92)
    if weight:
        ax.quiver(0, .55, 0, 0, -1.1, 0, color="#111110", linewidth=2.6,
                  arrow_length_ratio=.24, zorder=70)
    for u, col, lab in marks:
        ax.scatter(*(1.05 * np.asarray(u)), s=70, c=col, edgecolors="white",
                   linewidths=1.0, depthshade=False, zorder=60)
        ax.text(*(1.55 * np.asarray(u)), lab, color=col, fontsize=12,
                ha="center", va="center", zorder=61, fontweight="bold")
    if triad:
        # the world is Y-up: the floor is y=0 and the weight through the centre
        # runs down -y. Parked below and left of the ball, where neither the
        # sheet nor the weight goes, and tipped toward the eye so the sphere
        # cannot sort itself in front of it.
        a, e = np.deg2rad(AZIM), np.deg2rad(ELEV)
        rt = np.array([-np.sin(a), 0.0, np.cos(a)])          # screen right,
        ey = np.array([np.cos(e)*np.cos(a), np.sin(e), np.cos(e)*np.sin(a)])
        o = 0.50 * ey - 1.34 * rt + 1.12 * np.cross(ey, rt)  # eye, and screen up
        for u, lab in zip(np.eye(3), "xyz"):
            ax.quiver(*o, *(0.34 * u), color=TRIAD_INK, linewidth=1.3,
                      arrow_length_ratio=.35, zorder=80)
            ax.text(*(o + 0.48 * u), lab, color=TRIAD_INK, fontsize=10,
                    ha="center", va="center", zorder=81)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(1.02 * np.cos(th), np.zeros_like(th), 1.02 * np.sin(th), color=MUTED,
            lw=.7, alpha=.5)
    ax.set_xlim(-reach, reach)
    ax.set_zlim(-reach, reach)
    ax.set_ylim(-reach - 0.05, reach + 0.15)
    ax.set_box_aspect((1, 1.06, 1), zoom=1.42)
    COORD.matplotlib_view(ax, elev=ELEV, azim=AZIM)
    ax.set_axis_off()
    if rings_front:
        # the ruler, laid OVER the picture instead of buried in it. Square to the eye
        # and centred on the origin, so it projects to the silhouette of the sphere
        # of that height and the relief crosses it in the picture exactly where the
        # number crosses it. After the framing, like a ray, because the projection is
        # not known until the limits and the view are.
        _, rt, up = screen_axes()
        th = np.linspace(0, 2 * np.pi, 361)
        for rh in ring_r:
            P = (1.0 + rh) * (np.cos(th)[:, None] * rt + np.sin(th)[:, None] * up)
            xy = np.array([proj3d.proj_transform(*p, ax.get_proj())[:2] for p in P])
            ax.add_artist(Line2D(xy[:, 0], xy[:, 1], color=MUTED, lw=1.0, alpha=.65,
                                 ls=(0, (6, 5)), transform=ax.transData, zorder=102))
    # after the framing, because a ray is projected by hand and the projection is
    # not known until the limits and the view are
    for vec, col, lab in rays:
        v3 = np.asarray(vec, float)
        if np.linalg.norm(v3) > RAY_DOT and rays_front:
            # the same hand projection the zero ray gets, for the same reason, and
            # built out of Line2D for a duller one: a 3d axes gathers every Patch
            # and every Collection added to it and demands `do_3d_projection` of
            # them, so a flat arrow cannot BE a patch here. Shaft and head are two
            # lines, the head a triangle marker turned to face along the shaft.
            x0, y0, _ = proj3d.proj_transform(0.0, 0.0, 0.0, ax.get_proj())
            x2, y2, _ = proj3d.proj_transform(*v3, ax.get_proj())
            deg = np.degrees(np.arctan2(y2 - y0, x2 - x0)) - 90.0
            # RAY_DOT's problem, mirrored. A quiver's head is a FRACTION of its
            # shaft, so a short one vanishes; a marker's head is a fixed size, so a
            # short one is all head and a ray of a fiftieth of a radius draws the
            # same blob as a ray of a fifth. Scaling the head with the shaft keeps
            # the drawn LENGTHS in proportion, which is the only thing these rays
            # are on the page to say; the floor keeps a small one visible.
            ms = float(np.clip(11.0 * np.linalg.norm(v3) / 0.30, 3.5, 11.0))
            ax.add_artist(Line2D([x0, x2], [y0, y2], color=col, lw=2.4,
                                 solid_capstyle="round", transform=ax.transData,
                                 zorder=100))
            ax.add_artist(Line2D([x2], [y2], marker=(3, 0, deg), markersize=ms,
                                 color=col, markeredgecolor=col,
                                 transform=ax.transData, zorder=101))
        elif np.linalg.norm(v3) > RAY_DOT:
            ax.quiver(0, 0, 0, *v3, color=col, linewidth=2.2,
                      arrow_length_ratio=.30, alpha=.95, zorder=71)
        else:
            # The origin is INSIDE the ball, and a 3d axes sorts its collections
            # by computed depth and throws `zorder` away, so a dot drawn there
            # goes behind the near face of the paint and is simply not there. It
            # is projected by hand and added FLAT instead -- the one thing on the
            # panel allowed in front of the sphere, and it has to be, because
            # "this is zero" is the only thing such a panel has to say.
            x2, y2, _ = proj3d.proj_transform(*v3, ax.get_proj())
            ax.add_artist(Line2D([x2], [y2], marker="o", markersize=9, color=col,
                                 markeredgecolor="white", markeredgewidth=1.2,
                                 transform=ax.transData, zorder=100))
        if lab:
            # straight UP THE SCREEN off the tip: the one offset that cannot land
            # back on the ray, or on the weight arrow the origin already carries
            a, e = np.deg2rad(AZIM), np.deg2rad(ELEV)
            rt = np.array([-np.sin(a), 0.0, np.cos(a)])
            ey = np.array([np.cos(e)*np.cos(a), np.sin(e), np.cos(e)*np.sin(a)])
            ax.text(*(v3 + 0.26 * np.cross(ey, rt)), lab, color=col, fontsize=12,
                    ha="center", va="center", zorder=101, fontweight="bold")
    ax.set_title(title, color=INK, fontsize=13, pad=10, linespacing=1.35)


def globe_png(ico, panels, px, triad=False, reach=1.45, rays_front=False, zoom=1.0,
              rings_front=False, weight=True):
    """A row of globes as one image. A panel is the five drawing lists and a title,
    optionally followed by a sixth, `bars`, a seventh, `rays`, and an eighth,
    `relief`; older callers pass five and are unaffected.

    `zoom` makes the BALL bigger without making the TEXT bigger, which `px` cannot
    do: `px` sets the dpi, and dpi scales a sphere drawn in axes coordinates and a
    title set in points by exactly the same factor, so it buys resolution and never
    size. Inches at a fixed dpi scale only the first of the two. A page whose subject
    is a shape in a silhouette wants the shape twice as wide and the words where they
    were. The title band is held at its pixel height for the same reason -- it is
    sized by the type in it, not by the panel under it -- so `zoom` does not open a
    strip of white above every globe. `zoom = 1` is the figure every existing caller
    already gets, to the pixel.
    """
    fig = plt.figure(figsize=(3.0 * zoom * len(panels), 3.5 * zoom), dpi=px / 3.5,
                     facecolor=PAPER)
    for i, panel in enumerate(panels):
        fills, dots, quills, marks, title = panel[:5]
        globe(fig.add_subplot(1, len(panels), i + 1, projection="3d", facecolor=PAPER),
              ico, fills, dots, quills, marks, title, triad=triad and i == 0,
              bars=panel[5] if len(panel) > 5 else (), reach=reach,
              rays=panel[6] if len(panel) > 6 else (), rays_front=rays_front,
              relief=panel[7] if len(panel) > 7 else None,
              rings_front=rings_front, weight=weight)
    fig.subplots_adjust(0, .01 / zoom, 1, 1 - .18 / zoom, 0, 0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=PAPER)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("object")
    ap.add_argument("--poses", type=int, nargs="+", default=None)
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--push-points", type=int, default=90)
    ap.add_argument("--push-dirs", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=840)
    # A spray pass is unbounded, so on a part whose faces are a sixth of it each
    # the region comes out as two or three whole faces. 'spot' gives the gun a
    # finite beam instead, which is what makes a region of a stated size whose
    # boundary runs across faces rather than around them.
    ap.add_argument("--region", choices=("spray", "spot"), default="spray")
    ap.add_argument("--region-area", type=float, nargs=2, default=(0.17, 0.33),
                    metavar=("LO", "HI"), help="spot only: how much surface the "
                    "process asks for, as a fraction, drawn per pose")
    args = ap.parse_args()

    d = obj_path(args.object)
    mesh = trimesh.load(d / "mesh.stl", force="mesh")
    examples = read_json(d / "tips" / "tips.json")["examples"]
    poses = args.poses if args.poses is not None else range(len(examples))
    ico, tiles = tiling(4)
    tree = cKDTree(tiles)
    adjacency = neighbours(ico)
    px = args.size
    out_dir = d / "cover"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = d / "_render_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)

    records = []
    for pose in poses:
        ex = examples[pose]
        T = np.asarray(ex["T_world_mesh"])
        R, t = T[:3, :3], T[:3, 3]
        rng = np.random.default_rng(args.seed + 1000 * pose)
        # the work region is the only thing the two paths disagree about; `part`
        # is the mesh it is expressed on, refined along the region boundary when
        # the region has to cut across a face rather than stop at its edge
        extra = {}
        if args.region == "spot":
            reg = spot_region(mesh, T, rng, float(rng.uniform(*args.region_area)),
                              BOUNDARY_EDGE * float(np.linalg.norm(mesh.extents)))
            part, inside = reg.pop("mesh"), reg.pop("mask")
            n_passes, extra = reg["n_passes"], reg
        else:
            n_passes = int(rng.integers(1, 4))
            dirs = gun_directions(rng, n_passes)
            part = mesh
            inside = region_mask(mesh, T, dirs, mesh.triangles_center, mesh.face_normals)
        area = float(part.area_faces[inside].sum() / part.area)

        pw, pu = angled_pushes(part, T, inside, args.push_points, args.push_dirs,
                               args.seed + pose)
        targets = UP - args.k * pu                       # what the supports owe
        keep = np.linalg.norm(targets, axis=1) > 1e-9
        targets, pw, pu = targets[keep], pw[keep], pu[keep]
        unit = targets / np.linalg.norm(targets, axis=1, keepdims=True)
        tile_of = tree.query(unit)[1]

        # a support may touch anywhere off the work region; its direction is the
        # inward normal there, and one direction per tile is plenty
        on_floor = (part.triangles_center @ R.T + t)[:, 1] <= CONTACT_EPS
        off = np.flatnonzero(~inside & ~on_floor)
        push = -(part.face_normals[off] @ R.T)
        push /= np.linalg.norm(push, axis=1, keepdims=True)
        # the tiles only deduplicate; the direction that goes into the covering
        # test is the representative face's OWN normal, so that what is verified
        # is exactly what gets built. Using the tile centre instead is off by up
        # to 2.7 degrees, which was enough to make two of ten poses report 100%
        # when the real normals held barely half.
        hit = tree.query(push)[1]
        first = {}
        for k, h in enumerate(hit):
            first.setdefault(h, k)
        rep = np.array([first[h] for h in sorted(first)])
        avail, avail_face = push[rep], off[rep]

        chosen, steps, done = grow(targets, avail)
        faces = avail_face[chosen]
        # how much real surface stands behind each direction
        back = np.zeros(len(tiles))
        np.add.at(back, hit, part.area_faces[off])
        weight = back[tree.query(avail)[1]]
        order = np.argsort(-weight)
        share = np.cumsum(weight[order]) / weight.sum()
        solid = order[:max(int(np.searchsorted(share, .95)) + 1, 3)]   # the real faces
        haze = np.random.default_rng(13).choice(
            order[len(solid):], min(200, max(len(order) - len(solid), 0)), replace=False)
        orange_solid, orange_haze = avail[solid], avail[haze]
        pts = part.triangles_center[faces] @ R.T + t
        us = avail[chosen]

        # ---- the picture: spheres on top, the part underneath, almost no words ----
        chosen_dirs = avail[chosen]
        # ---- the panels: an accurate point cloud, a couple of dozen arrows ----
        few = np.random.default_rng(11).choice(len(pu), min(26, len(pu)), replace=False)
        many = np.random.default_rng(12).choice(len(pu), min(1400, len(pu)), replace=False)

        # what the part can offer: one point per DISTINCT normal, sized by the
        # surface behind it, so a flat face is one fat dot and a fillet a streak
        uq, inv = np.unique(np.round(push, 3), axis=0, return_inverse=True)
        uq /= np.linalg.norm(uq, axis=1, keepdims=True)
        back = np.zeros(len(uq))
        np.add.at(back, inv, part.area_faces[off])
        # a flat face puts a tenth of the surface on ONE direction; a fillet
        # spreads a hundredth over a thousand of them. Split them by that.
        flat = np.flatnonzero(back >= .01 * back.sum())
        band = np.flatnonzero(back < .01 * back.sum())
        band_share = float(back[band].sum() / back.sum())
        if len(band) > 1200:
            band = np.random.default_rng(15).choice(band, 1200, replace=False)
        real = flat[np.argsort(-back[flat])]

        # continuous sets get painted onto the tiles they cover; the flat faces
        # stay dots, because a flat face IS one direction however big it is
        job_tiles = np.flatnonzero(sheet(adjacency, tree.query(-pu)[1]))
        owed = sheet(adjacency, tile_of)                 # the shifted patch
        nearest = cKDTree(unit).query(tiles)[1]          # fill each tile from its sample
        band_tiles = (np.flatnonzero(sheet(adjacency, tree.query(uq[band])[1]))
                      if len(band) else np.array([], int))
        # one orange for both: a fillet really does sweep a band and a flat face
        # really is a single direction, and that difference is the shape, not a
        # shade. Dimming the band was smuggling "how much surface stands behind
        # this direction" into the same channel, which is a different quantity.
        BLUE_F = [(job_tiles, PUSH_BLUE, .95)]
        ORANGE_F = [(band_tiles, HAVE, .95)]
        ORANGE_D = [(uq[flat], HAVE, 70, True)]
        chosen_dirs = avail[chosen]
        panels = [(BLUE_F, [], [(pu[few], PUSH_BLUE, False)], [], "the job"),
                  ([(job_tiles, NEED, .95)], [],
                   [(pu[few], PUSH_BLUE, False), (-pu[few], NEED, True)], [], "reverse it")]
        for i, st in enumerate(steps):
            marks = [(UP, "#3d3d3a", "G")]
            marks += [(chosen_dirs[j], "#d2450f", str(j + 1)) for j in range(i)]
            fewt = np.random.default_rng(14).choice(len(unit), min(26, len(unit)),
                                                    replace=False)
            red = [(np.flatnonzero(owed & st[nearest]), DONE, .95),
                   (np.flatnonzero(owed & ~st[nearest]), NEED, .95)]
            arr = [(unit[fewt][~st[fewt]], NEED, True)]
            if i == 0:
                panels.append((red, [], arr, [(UP, "#3d3d3a", "G")],
                               f"+ gravity\n{100 * st.mean():.1f}% held"))
                # arrows sample the set, the same couple of dozen the blue and red
                # balls get. A continuous piece survives sampling; the flat faces
                # are isolated points, and any one dropped is simply gone, so they
                # all keep an arrow and the band gets the rest.
                spray = np.random.default_rng(16).choice(
                    band, min(26 - len(flat), len(band)), replace=False)
                # a sharp-edged part has no band, and saying it has one under a
                # sphere that is bare between the dots is the one thing this
                # panel must not do. What decides is how much surface the
                # non-flat directions hold, not how many of them there are: a
                # face all but eaten by the work region leaves a shaving that
                # counts as a direction and is not a band.
                panels.append((ORANGE_F, ORANGE_D,
                               [(np.vstack([uq[real], uq[spray]]), HAVE, True)], [],
                               f"what we have\n{len(flat)} faces + a fillet band"
                               if band_share > .01 else
                               f"what we have\n{len(flat)} faces, and no band"))
                # no panel laying orange over red: whether the two overlap is not
                # the test. A direction outside the red patch still helps cover
                # it in combination, because what has to contain the red is the
                # CONE the chosen directions span. The +1 +2 +3 panels show that
                # happening; a superposition would only invite reading it wrong.
            else:
                panels.append((red, [], arr, marks,
                               f"+{i}\n{100 * st.mean():.1f}% held"))

        parts = paint(part, T, inside, set(), tmp, f"cov_p{pose}", rel="_render_tmp")
        show = np.random.default_rng(5).choice(len(pw), min(26, len(pw)), replace=False)
        blue = [(pw[j], pu[j]) for j in show]
        bare = shot(args.object, T, parts, px, [], [])    # no forces yet
        forces = shot(args.object, T, parts, px, blue, [])
        # one render per panel, in panel order: 'reverse it', '+ gravity' and
        # 'what we have' all sit over the same part with the pushes and no
        # supports, and panel '+i' sits over the part carrying i of them
        shots = [bare, forces, forces, forces]
        shots += [shot(args.object, T, parts, px, blue,
                       [(pts[j], us[j]) for j in range(i)]) for i in range(1, len(steps))]

        # eight panels in one strip would print at postage-stamp size, so they go
        # in two rows of four, each with its own row of the part underneath
        per = (len(panels) + 1) // 2
        rows_ = [(panels[:per], shots[:per]), (panels[per:], shots[per:])]
        head = int(px * 0.16)
        blocks = []
        for k, (ps, sh_) in enumerate(rows_):
            g = globe_png(ico, ps, px, triad=k == 0)     # the triad on 'the job' only
            gw = g.size[0] // max(len(ps), 1)
            gh = int(g.size[1] * 0.99)
            blk = Image.new("RGB", (gw * len(ps), gh + gw), "white")
            blk.paste(g.crop((0, 0, g.size[0], gh)), (0, 0))
            for i, im in enumerate(sh_):
                blk.paste(im.resize((gw, gw)), (i * gw, gh))   # the render is square
            blocks.append(blk)
        caption = (
            f"{args.object}  pose {pose}   ·   BLUE = every direction the process can "
            f"push in   ·   RED = what the supports still owe, GREY = held\n"
            f"ORANGE = every direction this workpiece can actually supply   ·   "
            f"black = the weight   ·   G = the floor, 1 2 3 = the supports")
        font = _font(int(px * 0.042))
        # a pose that takes few supports makes few panels and so a narrow page,
        # while the caption is the same length whatever the page is. It therefore
        # sets the width whenever the panels do not, or it runs off the canvas.
        wide = max(max(b_.size[0] for b_ in blocks),
                   ImageDraw.Draw(Image.new("RGB", (1, 1))).multiline_textbbox(
                       (16, 0), caption, font=font)[2] + 16)
        page = Image.new("RGB", (wide, head + sum(b_.size[1] for b_ in blocks)), "white")
        y = head
        for b_ in blocks:
            page.paste(b_, (0, y))
            y += b_.size[1]
        ImageDraw.Draw(page).text((16, int(head * .22)), caption,
                                  fill=(90, 90, 90), font=font)
        out = out_dir / f"pose{pose}.png"
        page.save(out)

        if extra:
            # what a direction is worth is not the same as whether it exists: the
            # covering test asks only whether some face carries this normal, and a
            # face that is all but swallowed by the work region still carries it.
            # On a part with six faces that difference is the whole story, so the
            # surface left behind each candidate is recorded alongside it.
            extra["area_behind_available"] = sorted(
                float(w / part.area) for w in weight)
            extra["n_available_directions"] = int(len(avail))
        records.append({"pose": pose, "n_passes": n_passes, **extra,
                        "region_area_fraction": area,
                        "n_targets": int(len(targets)),
                        "n_supports": len(chosen),
                        "answered_fraction": float(done.mean()),
                        "answered_by_floor_alone": float(steps[0].mean()),
                        "supports": [{"p": [float(x) for x in pts[i]],
                                      "push": [float(x) for x in us[i]]}
                                     for i in range(len(chosen))]})
        print(f"pose {pose:2d}: {len(flat):2d} flat faces + a band of "
              f"{len(uq) - len(flat):5d} fillet directions, {len(targets):4d} pushes, "
              f"floor alone answers "
              f"{100 * steps[0].mean():4.1f}%  ->  {len(chosen)} supports, "
              f"{100 * done.mean():5.1f}% answered   {out}", flush=True)

    shutil.rmtree(tmp, ignore_errors=True)
    write_json(out_dir / f"cover_k{args.k:g}.json",
               {"object": args.object, "k": args.k, "poses": records})


if __name__ == "__main__":
    main()
