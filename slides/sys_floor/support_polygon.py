"""Row (3)'s two closed forms, kept in one place so `on_the_floor.py` imports
them instead of re-deriving them.

Not a script.  There was a page here -- a true plan view of the floor, ten
panels an object, the `p_cop` region drawn as a star hull about the plumb point
with the workpiece's shadow and its contact bar -- and it is gone, along with
`coverage.py`, which drew the same set as a region with its convex hull and the
ring of feet that answers it.  `on_the_floor.py` puts the same quantity on the
setup slide's own pose and camera, and that is the page that is kept.  What
survives here is the algebra those pages and this one share.

WHAT THE TWO FUNCTIONS ARE

Take the workpiece AND its supports as ONE free body.  The supports' forces on
the workpiece are then internal and cancel, so the right-hand side of the
balance depends only on gravity and on the process push -- NOT on the pressure
field underneath.  That is the whole trick: the floor's demand is computable
with no design in hand.

A floor point `p = (x, y, 0)` carrying `f` contributes the moment

    p x f = (y f_z,  -x f_z,  x f_y - y f_x)

so the VERTICAL (normal) forces answer only the two horizontal torque rows and
the HORIZONTAL (friction) forces answer only yaw.  The horizontal rows therefore
fix one point of the floor, the CENTRE OF PRESSURE, whatever the pressure field
is -- and that is `cop`:

    R    = mg - F_push (d_push . z)          the total normal force
    M    = c x (-mg z) + q x (F_push d_push)
    R y* = -M_x ,   R x* = M_y ,   p_cop = (x*, y*, 0)

Because the floor can only PUSH (`N >= 0`), the assembly stands only while
`p_cop` lies inside the CONVEX HULL of everything of it that touches the floor
-- the support feet, plus the workpiece's own pivot.  A floor that could pull
would accept any `p_cop` at all and there would be no condition here; that is
what makes row (3) a hull and not an equation.  And friction does not close the
other half: friction is bounded by `mu N` and tipping is the rotation in which
the lifting side's `N -> 0`, so its friction goes to zero with it (METHOD s5).

`exact_R_min` bounds the normal reaction over the full, unoccluded process
cones without sampling. For a face whose cone contains straight up, dz_max is
1; otherwise it is the largest vertical component on the cone's rim:

    dz_max = -n_z cos(alpha) + sin(alpha) sqrt(1 - n_z^2)
    R_min = 1 - t max_faces(dz_max)

Both functions take `t` in body weights, defaulting to the slides' `K = 0.5`.
Consequently `R >= 0.5` for every direction, including straight up. Visibility
can remove directions, so the full-cone minimum is a lower bound after the
line-of-sight filter. METHOD's historical `K = 1` results are not defaults here.

`CONE_HALF_DEG` IS IMPORTED, never re-implemented, because step 1 draws the
demand these pages measure and a second copy of the cone would let the two drift.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # slides/sys_floor
ROOT = HERE.parent.parent                       # the repository root
sys.path.insert(0, str(ROOT / "slides/tools"))

from cone_model import CONE_HALF_DEG                              # noqa: E402

K = 0.5                          # process magnitude, in body weights

DOWN = np.array([0.0, 0.0, -1.0])


def cop(q, d, com, t=K):
    """Where the load lands: the assembly's centre of pressure, and the floor's
    total normal force. `t` is the push magnitude in body weights, not a
    fraction of K. Lengths are in metres.

        R    = mg - F (d . z)          M = c x (-mg z) + q x (F d)
        R x* = M_y                     R y* = -M_x
    """
    q = np.atleast_2d(np.asarray(q, float))
    d = np.atleast_2d(np.asarray(d, float))
    R = 1.0 - t * d[:, 2]
    M = np.cross(com, DOWN) + np.cross(q, t * d)
    return np.stack([M[:, 1] / R, -M[:, 0] / R], axis=1), R


def exact_R_min(mesh, T, faces, t=K):
    """Return (minimum R, minimum outward n_z, can lose positive R).

    The minimum is over the full cones, before visibility filtering. A cone
    containing straight up attains d_z = 1 in its interior; other cones attain
    their maximum on the rim. At the default t = K = 0.5, R_min >= 0.5.
    """
    nz = np.clip((mesh.face_normals[faces] @ T[:3, :3].T)[:, 2], -1.0, 1.0)
    c, s = np.cos(np.radians(CONE_HALF_DEG)), np.sin(np.radians(CONE_HALF_DEG))
    dz = np.where(nz <= -c, 1.0,
                  -nz * c + s * np.sqrt(np.clip(1 - nz ** 2, 0, None)))
    r_min = float(1.0 - t * dz.max())
    return r_min, float(nz.min()), bool(r_min <= 0.0)
