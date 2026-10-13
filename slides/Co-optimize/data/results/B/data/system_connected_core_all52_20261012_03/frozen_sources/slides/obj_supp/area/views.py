"""Object overview cameras and unobstructed views of every contact region."""
import itertools
import numpy as np
import patch as P


def visible(S, ids, cam):
    """Use finite rays: geometry beyond a close-up camera is not an occluder."""
    delta = cam.eye-S.skin.cs[ids]
    distance = np.linalg.norm(delta, axis=1)
    direction = delta/distance[:, None]
    facing = np.einsum('ij,ij->i', S.skin.ns[ids], direction)
    front = np.flatnonzero(facing > .08)
    seen = np.zeros(len(ids), bool)
    if len(front):
        starts = S.skin.cm[ids[front]]+P.RAY_EPS*S.skin.nm[ids[front]]
        loc, ray, _ = S.mesh.ray.intersects_location(
            ray_origins=starts, ray_directions=direction[front]@S.R, multiple_hits=False)
        blocked = np.zeros(len(front), bool)
        blocked[ray] = np.linalg.norm(loc-starts[ray], axis=1) < distance[front][ray]-2*P.RAY_EPS
        seen[front] = ~blocked
    return seen, facing


def overview_cameras(S, masks):
    """Choose detail views that together expose the selected contact area."""
    ids = np.flatnonzero(np.logical_or.reduce(masks))
    pts = S.skin.cs[ids]
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = max(float((hi-lo).max())*.22, 2*S.radius)
    box = np.array(list(itertools.product(*zip(lo-pad, hi+pad))))
    # Keep the object recognisable and fully inside every panel while comparing
    # the same four designs from fixed views.
    box = np.vstack([box, np.asarray(S.mesh.vertices)@S.R.T+S.t])
    candidates = []
    for elev in (-60, -40, -20, 0, 20, 40, 60):
        for azim in range(0, 360, 30):
            cam = P.fit(elev, azim, box)
            if cam.eye[2] < .001 or S.mesh.contains(((cam.eye-S.t) @ S.R)[None])[0]:
                continue
            seen, facing = visible(S, ids, cam)
            visibility = np.array([S.skin.area[ids][seen & m[ids]].sum()/S.skin.area[m].sum()
                                   for m in masks])
            projected = np.array([(S.skin.area[ids]*np.maximum(facing, 0))[seen & m[ids]].sum()/S.skin.area[m].sum()
                                  for m in masks])
            candidates.append((cam, seen, visibility, projected))
    def score(indices):
        selected = [candidates[i] for i in indices]
        seen = np.logical_or.reduce([c[1] for c in selected])
        visible = np.array([S.skin.area[ids][seen & m[ids]].sum()/S.skin.area[m].sum()
                            for m in masks])
        projected = np.maximum.reduce([c[3] for c in selected])
        return (round(float(visible.min()), 3), round(float(projected.min()), 3),
                float(projected.mean()))
    best = max(itertools.combinations(range(len(candidates)), 2), key=score)
    selected = [candidates[i] for i in best]
    return ids, selected


def detail_camera(S, mask, seed):
    """See the whole region, using a closer cavity view or a distant crop."""
    ids = np.flatnonzero(mask)
    vertices = S.skin.sub.vertices[np.unique(S.skin.sub.faces[ids])]@S.R.T+S.t
    candidates = []
    def evaluate(elev, azim):
        base = P.fit(elev, azim, vertices)
        for factor in (1., 3.):
            cam = base._replace(dist=base.dist*factor, eye=base.lookat-base.dist*factor*base.fwd)
            if cam.eye[2] < .001 or S.mesh.contains(((cam.eye-S.t)@S.R)[None])[0]:
                continue
            seen, facing = visible(S, ids, cam)
            fraction = float(S.skin.area[ids][seen].sum()/S.skin.area[ids].sum())
            projected = float((S.skin.area[ids]*np.maximum(facing, 0)*seen).sum()/S.skin.area[ids].sum())
            candidates.append((cam, fraction, projected))
    for elev in range(-80, 81, 20):
        for azim in range(0, 360, 20):
            evaluate(elev, azim)
    for normal in (S.skin.ns[seed], np.average(S.skin.ns[ids], axis=0, weights=S.skin.area[ids])):
        normal = normal/np.linalg.norm(normal)
        evaluate(float(np.degrees(np.arcsin(normal[2]))),
                 float(np.degrees(np.arctan2(normal[1], normal[0]))))
    key = lambda v: (round(v[1], 3), v[2])
    best = max(candidates, key=key)
    if best[1] < .995:
        cam = best[0]
        for de in range(-15, 16, 5):
            for da in range(-15, 16, 5):
                evaluate(float(np.clip(cam.elev+de, -89, 89)), (cam.azim+da)%360)
        best = max(candidates, key=key)
    cam, fraction, projected = best
    assert fraction >= .95, ('Contact region needs another detail view', seed, fraction)
    xy = P.TS.screen(vertices, cam)
    low, high = xy.min(axis=0), xy.max(axis=0)
    centre, side = (low+high)/2, min(float((high-low).max())*1.12, P.PX)
    low = np.clip(centre-side/2, 0, P.PX-side)
    high = low+side
    crop = tuple(int(v) for v in np.r_[np.floor(low), np.ceil(high)])
    return cam, crop
