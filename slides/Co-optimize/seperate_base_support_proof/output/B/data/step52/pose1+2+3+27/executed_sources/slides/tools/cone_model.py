"""THE PROCESS CONE: one half-angle, one sampler, one definition for the chain.

Imported, never copied.  `step1_demand` and `step3_ceiling` both draw the
process disturbance and they MUST draw the same one -- the demand fields step 1
saves and the invoices step 3 pays are the same table read twice, so a cone that
differed between them would make the gate compare a demand against a supply
priced for a different process.  Hence one module, one constant.

    CONE_HALF_DEG = 30.0

**Why 30 deg and not `GRAZE`.**  The model as it stood kept every direction with
`v . n > GRAZE = 0.08` -- `shrink_support.angled_pushes`, a cone of half-angle
85.4 deg about the local inward normal, which is "a force from any direction in
the air".  Under it the moment demand covered essentially the whole turning
sphere (100.0 % of the 5120 axes on A1-f pose 1 and B pose 0, 96.7 % on cuboid
pose 3), so WHICH axes were demanded carried no information and only the arm was
left saying anything.  `cone_compare` priced the narrowing and `ratio_sweep_B`
swept it as a friction ratio over all ten of B's poses; 15 deg is the prior
baseline and 30 deg is the current declared input.  Like PRESSURE and K, it is
chosen rather than solved, so it lives here as a named
constant rather than as a number repeated at each call site.

**Why the sampler is not `angled_pushes` with a smaller number.**  This is the
load-bearing distinction and it is the reason this module exists at all.
`angled_pushes` draws `3 * n_dirs` directions uniformly on the WHOLE SPHERE and
keeps the first `n_dirs` that clear the cone.  At 85.4 deg that accepts 46 % and
the truncation binds; at 15 deg it accepts **1.7 %**, so it would hand back
about 20 directions a point instead of `n_dirs` -- a twentyfold thinner sample
dressed up as a narrower cone, and METHOD s7's coupon-collector trap would then
credit the narrowing with its own convergence lag.  `cone_pushes` samples the
spherical cap DIRECTLY (cos psi uniform on `[cos half, 1]`, azimuth uniform),
which is the same distribution restricted to the cap and yields exactly
`n_dirs` a point at every half-angle.

It reproduces the wide model when asked for it: at 85.4 deg its ladder on cuboid
pose 3 reads 30.6 / 68.7 / 92.0 / 95.8 / 96.6 % against `angled_pushes`' own
recorded 31.6 / 68.3 / 92.1 / 95.9 / 96.7 %, so the two samplers agree where
they can be compared and differ only where `angled_pushes` cannot follow.

Everything else about a push is unchanged: the point stream is `angled_pushes`'
call for call (area-weighted face choice, barycentric point, the LOCAL face
normal), and the reachability ray test is the same -- the tool still arrives
along a straight line out of the air, and self-occluded angles still drop out on
their own rather than being masked.
"""
from __future__ import annotations

import numpy as np

CONE_HALF_DEG = 30.0        # THE process cone: half-angle about the local
                            # inward normal, degrees.  Updated 2026-09-07;
                            # the whole chain reads it from here.


def frame(n):
    """An orthonormal pair square to each unit normal, branch-free (Duff et al.)."""
    n = np.asarray(n)
    sg = np.copysign(1.0, n[:, 2])
    a = -1.0 / (sg + n[:, 2])
    b = n[:, 0] * n[:, 1] * a
    e1 = np.stack([1.0 + sg * n[:, 0] ** 2 * a, sg * b, -sg * n[:, 0]], 1)
    e2 = np.stack([b, sg + n[:, 1] ** 2 * a, -n[:, 1]], 1)
    return e1, e2


def cone_pushes(part, T, inside, n_points, n_dirs, seed, half_deg):
    """`shrink_support.angled_pushes` with the cone's half-angle made a parameter.

    Point stream identical, call for call, so a sweep's cones stand on the same
    points; directions drawn straight from the spherical cap instead of by
    rejection, so the count a point gets does not fall with the half-angle; the
    reachability test unchanged -- the tool still arrives along a straight line
    out of the air, and self-occluded angles still drop out on their own.

    Returns `(q_world, d_world)`: where each push lands and which way it acts.
    """
    R, t = T[:3, :3], T[:3, 3]
    rng = np.random.default_rng(seed)
    where = np.flatnonzero(inside)
    w = part.area_faces[where]
    face = rng.choice(where, size=n_points, p=w / w.sum())
    a, b = rng.random(n_points), rng.random(n_points)
    flip = a + b > 1
    a[flip], b[flip] = 1 - a[flip], 1 - b[flip]
    tri = part.triangles[face]
    pts = (tri[:, 0] + a[:, None] * (tri[:, 1] - tri[:, 0])
           + b[:, None] * (tri[:, 2] - tri[:, 0]))
    nrm = part.face_normals[face]                 # the LOCAL normal, per point
    c0 = float(np.cos(np.radians(half_deg)))
    u = c0 + (1.0 - c0) * rng.random((n_points, n_dirs))       # cos psi
    ph = 2.0 * np.pi * rng.random((n_points, n_dirs))
    s = np.sqrt(np.clip(1.0 - u * u, 0.0, None))
    e1, e2 = frame(nrm)
    V = u[..., None] * nrm[:, None, :]
    V += (s * np.cos(ph))[..., None] * e1[:, None, :]
    V += (s * np.sin(ph))[..., None] * e2[:, None, :]
    del u, ph, s
    V = V.reshape(-1, 3)
    O = np.repeat(pts + nrm * 1e-5, n_dirs, axis=0)
    assert np.abs(np.linalg.norm(V, axis=1) - 1.0).max() < 1e-12
    assert (V * np.repeat(nrm, n_dirs, axis=0)).sum(1).min() >= c0 - 1e-12
    clear = ~part.ray.intersects_any(ray_origins=O, ray_directions=V)
    return O[clear] @ R.T + t, -(V[clear] @ R.T)   # the push runs against the escape
