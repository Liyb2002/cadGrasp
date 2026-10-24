"""The actual contact polygons for the gradient's guidance, as step3/independent_physics.contacts builds them,
vectorized (slides/Co-optimize/helper_func/co_common.contact_boundary loops in Python over every source triangle
and every shell triangle in its box: 0.7 s a task).

The same pairs -- a source triangle and a shell triangle whose boxes meet (padded by 1e-10 m; one bulk query of
the same tree, which returns them in the same order) and whose vertices lie within 1e-10 m of the source plane --
clipped by the source triangle's edge planes as the kernel's clip_plane does (Sutherland-Hodgman: per vertex, the
crossing point from the previous vertex, then the vertex if inside), all pairs at once; then the same
orientation, the same fan and the same sliver test, in the kernel's order.  The arithmetic is the same but for
the order of a few sums, so where a shell vertex lies on an edge plane (a shell triangle on its own source
triangle) the sign of rounding may add or drop a sliver of no area there; the contact region, its wrench cone
and the guidance's judgement are the kernel's.  The oracle's full audit keeps the kernel's own loop.
"""
import numpy as np
import rtree
from whole_task_2.step3.legacy import kernel as K

TOL = 1e-10                      # the kernel's coplanarity tolerance (m)
M = 6                            # a triangle clipped by three planes keeps at most six vertices


def clip(poly, count, plane):
    """clip_plane on many polygons: poly (P, M, 3) with `count` vertices each, plane (P, 4)."""
    j = np.arange(M)[None]; valid = j < count[:, None]
    values = np.einsum('pmc,pc->pm', poly, plane[:, :3])+plane[:, 3:]
    inside = (values <= 0) & valid
    i = (j-1) % np.maximum(count, 1)[:, None]                       # the previous vertex, around each polygon
    vi = np.take_along_axis(values, i, 1); pi = np.take_along_axis(poly, i[..., None], 1)
    crossing = valid & (np.take_along_axis(inside, i, 1) != inside)
    with np.errstate(divide='ignore', invalid='ignore'):
        point = pi+np.where(crossing, vi/(vi-values), 0.)[..., None]*(poly-pi)
    keep = np.stack([crossing, inside], 2).reshape(len(poly), 2*M)
    order = np.argsort(~keep, axis=1, kind='stable')[:, :M]
    out = np.take_along_axis(np.stack([point, poly], 2).reshape(len(poly), 2*M, 3), order[..., None], 1)
    return out, np.minimum(keep.sum(1), M)


def contact_boundary(mesh, shell, allowed_faces):
    """K.contact_boundary(mesh, shell, allowed_faces): (triangles (n, 3, 3), their source faces)."""
    allowed = np.asarray(allowed_faces, int); empty = (np.zeros((0, 3, 3)), np.zeros(0, int))
    if not len(allowed) or not len(shell.faces): return empty
    O = mesh.triangles[allowed]; N = mesh.face_normals[allowed]; T = shell.triangles
    # shell.triangles_tree's tree, bulk-loaded from arrays instead of a Python stream: the same tree, the same
    # answers in the same order, 5x faster to build.
    tree = rtree.index.Index((np.arange(len(T), dtype=np.int64), T.min(1), T.max(1)), properties=rtree.index.Property(dimension=3))
    ids, counts = tree.intersection_v(O.min(1)-TOL, O.max(1)+TOL)
    f = np.repeat(np.arange(len(allowed)), np.asarray(counts, int)); t = np.asarray(ids, int)
    flat = np.abs(np.einsum('pkc,pc->pk', T[t]-O[f, :1], N[f])).max(1) <= TOL; f, t = f[flat], t[flat]
    if not len(f): return empty
    # Each source triangle's edge planes [outward, -outward . a], edges (a, b) = (0, 1), (1, 2), (2, 0).
    out = np.cross(np.roll(O, -1, axis=1)-O, N[:, None]); out /= np.linalg.norm(out, axis=2, keepdims=True)
    planes = np.concatenate([out, -np.einsum('fkc,fkc->fk', out, O)[..., None]], axis=2)
    poly = np.zeros((len(f), M, 3)); poly[:, :3] = T[t]; count = np.full(len(f), 3)
    for k in range(3):
        poly, count = clip(poly, count, planes[f, k]); count[count < 3] = 0
    live = count >= 3; poly, count, n, face = poly[live], count[live], N[f[live]], allowed[f[live]]
    flip = np.einsum('pc,pc->p', np.cross(poly[:, 1]-poly[:, 0], poly[:, 2]-poly[:, 0]), n) < 0
    j = np.arange(M)[None]; reverse = np.where(j < count[:, None], count[:, None]-1-j, j)
    poly[flip] = np.take_along_axis(poly[flip], reverse[flip][..., None], 1)
    k = np.arange(1, M-1)[None]                                     # the fan: (0, k, k + 1)
    fan = np.stack([np.broadcast_to(poly[:, :1], (len(poly), M-2, 3)), poly[:, 1:M-1], poly[:, 2:M]], 2)
    real = (k+1 < count[:, None]) & (np.linalg.norm(np.cross(fan[:, :, 1]-fan[:, :, 0], fan[:, :, 2]-fan[:, :, 0]), axis=2) > 1e-18)
    return fan[real].reshape(-1, 3, 3), np.broadcast_to(face[:, None], real.shape)[real].astype(int)


def contacts(body, support, work_ids, com):
    """step3.independent_physics.contacts on contact_boundary() above."""
    allowed = np.setdiff1d(np.arange(len(body.faces)), work_ids)
    triangles, sources = contact_boundary(body, support, allowed)
    if not len(triangles): raise RuntimeError('No actual contact polygons remain')
    q = triangles.reshape(-1, 3); n = np.repeat(-body.face_normals[sources], 3, axis=0)
    scale = np.r_[np.ones(3), np.ones(3)/body.extents.max()]
    full = K.I.merge_columns(K.U.floor(np.empty((0, 6)), scale), K.U.heads(np.c_[n, np.cross(q-com, n)], scale))
    return triangles, sources, full, scale
