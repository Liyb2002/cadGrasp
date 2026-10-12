"""Geometry and drawing shared by the contact-area demos in area.py.

Every subdivided contact face contributes its inward normal and the moment
of that normal about the object's COM. Contact magnitudes are nonnegative
and unbounded. This module contains no patch pruning or coverage objective.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import trimesh
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in ("slides/setup/poses", "slides/tools"):
    sys.path.insert(0, str(ROOT / p))
from supports import CONTACT_EPS
from cover import NEED, UP, globe_png, neighbours, sheet, tiling
from mesh_export import export
import big_tip as G
import tip_sequence as TS
from tip_sequence import PX, fit, render
import inputs as I

OBJECTS = ('A1-f', 'B', 'C5')
CUT = .01
LIFT = 2e-4
ORANGE = '#E08A24'
BARE = '#cbc6ba'
FLOOR_MARK = '#3d3d3a'
PAPER, BALL, GAP = '#ffffff', 840, 18
AGREE, RAY_EPS = 5, 1e-5

def hexf(h):
    return tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))


def norm(v):
    return np.linalg.norm(v, axis=-1)


def stripcut(tri, cut, levels=6):
    """Slice every triangle whose longest edge exceeds `cut` into strips
    parallel to its shortest edge, and again until no edge is over `cut`.

    With `(A, B, C)` a rotation of the face so that `AC` is the shortest edge
    and `B` the vertex opposite it, `k = ceil(max(|AB|, |CB|) / cut)`, and
    `P_i`, `Q_i` the points at fraction `i / k` along `AB` and `CB`: strip `i`
    is the quad `P_i P_i+1 Q_i+1 Q_i`, drawn as `(P_i, P_i+1, Q_i+1)` and
    `(P_i, Q_i+1, Q_i)`, and the last strip is the triangle `(P_k-1, B, Q_k-1)`.
    Every piece keeps the face's winding (its normal), the pieces tile the
    face exactly, and a sliver of length `l` becomes `2k - 1 ~ 2 l / cut`
    pieces rather than `3^rounds`.  Returns the pieces and, per piece, the
    index of the input triangle it came from.
    """
    tri = np.asarray(tri, float)
    src = np.arange(len(tri))
    done_t, done_s = [], []
    for _ in range(levels + 1):
        e = norm(tri[:, [1, 2, 0]] - tri)                # e0=(0,1) e1=(1,2) e2=(2,0)
        long = e.max(axis=1) > cut
        done_t.append(tri[~long])
        done_s.append(src[~long])
        if not long.any():
            break
        t, el, s = tri[long], e[long], src[long]
        shift = np.array([1, 2, 0])[el.argmin(axis=1)]   # AC shortest, B opposite
        idx = (np.arange(3)[None, :] + shift[:, None]) % 3
        abc = np.take_along_axis(t, idx[:, :, None], axis=1)
        A_, B_, C_ = abc[:, 0], abc[:, 1], abc[:, 2]
        k = np.ceil(np.maximum(norm(B_ - A_), norm(B_ - C_)) / cut).astype(int)
        assert (k >= 2).all()
        rep = np.repeat(np.arange(len(t)), k)
        i = np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k)
        kk = k[rep]
        s0, s1 = (i / kk)[:, None], ((i + 1) / kk)[:, None]
        Ar, Br, Cr = A_[rep], B_[rep], C_[rep]
        P0, P1 = Ar + (Br - Ar) * s0, Ar + (Br - Ar) * s1
        Q0, Q1 = Cr + (Br - Cr) * s0, Cr + (Br - Cr) * s1
        last = i == kk - 1
        t1 = np.stack([P0, P1, Q1], axis=1)
        t1[last] = np.stack([P0[last], Br[last], Q0[last]], axis=1)
        t2 = np.stack([P0, Q1, Q0], axis=1)[~last]
        tri = np.vstack([t1, t2])
        src = np.r_[s[rep], s[rep][~last]]
    else:
        raise AssertionError("stripcut: still long edges after every level")
    return np.vstack(done_t), np.concatenate(done_s)

class Skin:
    """The touchable skin, strip-cut, in the WORLD frame at the tip pose,
    with source-face identities preserved."""

    def __init__(self, mesh, touch, R, t, cut, log):
        t0 = time.time()
        idx = np.flatnonzero(touch)
        tri, src = stripcut(mesh.triangles[idx], cut)
        sub = trimesh.Trimesh(tri.reshape(-1, 3), np.arange(tri.size // 3).reshape(-1, 3),
                              process=True)
        assert len(sub.faces) == len(tri), "process() dropped a face"
        assert abs(sub.area - mesh.area_faces[idx].sum()) <= 1e-9 * mesh.area
        assert ((sub.face_normals * mesh.face_normals[idx[src]]).sum(axis=1) > 0.999).all()
        self.sub, self.src, self.mesh = sub, idx[src], mesh
        self.cm = sub.triangles_center                    # mesh frame, for rays
        self.nm = sub.face_normals
        self.cs = self.cm @ R.T + t                       # world frame
        self.ns = self.nm @ R.T
        self.area = sub.area_faces
        self.tree = cKDTree(self.cs)
        log(f'  skin: {len(idx)} source faces -> {len(sub.faces)} contact samples; '
            f'{time.time()-t0:.1f} s')

    def gens(self, sel, com):
        """The wrenches at faces `sel`: push direction, torque about `c`."""
        u = -self.ns[sel]
        return np.c_[u, np.cross(self.cs[sel] - com, u)]

def fps(P, start, n):
    """`n` farthest-point-spread indices into `P`, from `start`."""
    out = [int(start)]
    dmin = norm(P - P[start])
    while len(out) < min(n, len(P)):
        nxt = int(np.argmax(dmin))
        out.append(nxt)
        dmin = np.minimum(dmin, norm(P - P[nxt]))
    return np.array(out), float(dmin.max())

def seal(piece):
    """T-junctions closed: a face with another face's vertex on one of its
    edges is fanned from its own centroid through every such vertex."""
    V, F = np.asarray(piece.vertices), np.asarray(piece.faces)
    tree = cKDTree(V)
    eps = 1e-7 * float(piece.scale)
    verts, faces = [V], []
    n_new = len(V)
    for f in F:
        ring = []
        for e in range(3):
            a, b = V[f[e]], V[f[(e + 1) % 3]]
            ring.append(int(f[e]))
            ab = b - a
            L2 = float(ab @ ab)
            on = []
            for c in tree.query_ball_point((a + b) / 2, 0.5 * np.sqrt(L2) + eps):
                if c in f:
                    continue
                s = float((V[c] - a) @ ab) / L2
                if 1e-6 < s < 1 - 1e-6 and norm(V[c] - (a + s * ab)) < eps:
                    on.append((s, int(c)))
            ring += [c for _, c in sorted(on)]
        if len(ring) == 3:
            faces.append(list(f))
            continue
        verts.append(V[f].mean(axis=0)[None])
        for i in range(len(ring)):
            faces.append([n_new, ring[i], ring[(i + 1) % len(ring)]])
        n_new += 1
    return trimesh.Trimesh(np.vstack(verts), np.array(faces), process=False)

def sticker(skin, masks, tmp, tag):
    """The discs' faces, sealed, `LIFT` proud along the vertex normals, as
    one piece."""
    faces = np.flatnonzero(np.logical_or.reduce(masks))
    piece = seal(skin.sub.submesh([faces], append=True))
    piece.vertices = piece.vertices + LIFT * piece.vertex_normals
    export(piece, tmp / f"{tag}.obj")
    return (f"{G.TMP}/{tag}.obj", "patch")

def shot(name, Tm, parts, cam, colour=ORANGE):
    """`tip_sequence.render` with the patch's material taught to the scene
    for the one call (`skin.shot`'s string patch, put back in a `finally`),
    rendered until two renders agree byte for byte.

    The patch material uses the same orange as the sphere coverage."""
    r, g, b = hexf(colour)
    mat = f'<material name="patch" rgba="{r:.3f} {g:.3f} {b:.3f} 1" specular="0.1"/>'
    anchor = '<material name="work"'
    assert anchor in TS.SCENE
    was = TS.SCENE
    TS.SCENE = was.replace(anchor, mat + "\n    " + anchor, 1)
    try:
        seen = []
        for _ in range(AGREE):
            im = render(name, Tm, parts, cam)
            raw = np.asarray(im)
            if any(np.array_equal(done, raw) for done in seen):
                break
            seen.append(raw)
        else:
            raise AssertionError(f"{AGREE} renders and no two agree")
    finally:
        TS.SCENE = was
    assert im.size == (PX, PX)
    return im

def demand_sheet(tiles, tree, A, want, ok, drawn_at=+1):
    """Conservative colouring of the sampled demand directions.

    Keep the same three smoothing rounds and let any failed sample mark a
    shared tile as failed. This only supplies the area's existing tile masks.
    """
    paint = sheet(A, tree.query(drawn_at * want)[1], 3)
    bad = (sheet(A, tree.query(drawn_at * want[~ok])[1], 3) & paint
           if (~ok).any() else np.zeros(len(tiles), bool))
    return ([(np.flatnonzero(paint & ~bad), "#cfd6cf", 1.0),
             (np.flatnonzero(bad), "#B02A26", 1.0)],
            int(paint.sum()), int(bad.sum()))

