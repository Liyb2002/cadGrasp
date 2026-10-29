"""Twenty random large work areas on bunny B, plus the whole shape.

An area is one to three pieces on the surface, together 50-70% of the surface area; every piece is
connected, larger than 2% of the surface, and kept apart from the other pieces.  A piece grows from one
to three nearby source faces in order of geodesic distance (Dijkstra over adjacent face centres), so it
is a compact blob; a small hole it encloses joins it, and a few rounds of majority voting over the three
edge neighbours smooth its saw-toothed triangle boundary.

Per area, in this folder:
  <name>.json  face_ids (what step1's --target reads), the pieces and their area fractions
  <name>.png   the area light pink on the sitting bunny (original +Y up), front and back
index.json lists them all.

  PYTHONPATH=. .venv/bin/python -B objects/B/big_tasks/generate.py
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from whole_task_2.step1.common import ROOT, load_shape, sha256

HERE = Path(__file__).resolve().parent
MESH = ROOT/'objects/B/mesh.stl'
COUNT = 20
LOW, HIGH = .50, .70            # total area fraction of an area
MIN_PIECE = .02                 # area fraction of the smallest piece
GAP = .03                       # geodesic gap between pieces, fraction of the AABB diagonal
SMOOTHING = 4                   # rounds of boundary majority voting
SEED = 20261024
LIGHT_PINK = (1., .74, .84)
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


class Surface:
    def __init__(self, mesh):
        self.mesh = mesh; self.area = mesh.area_faces; self.total = float(self.area.sum())
        pairs = mesh.face_adjacency; F = len(mesh.faces)
        w = np.linalg.norm(np.diff(mesh.triangles_center[pairs], axis=1)[:, 0], axis=1)
        rows, cols = np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]]
        self.graph = csr_matrix((np.r_[w, w], (rows, cols)), shape=(F, F))
        self.edges = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(F, F))
        self.neighbours = self.edges.tolil().rows

    def distance(self, sources, allowed=None):
        graph = self.graph
        if allowed is not None:
            keep = allowed.astype(float); graph = graph.multiply(keep[:, None]).multiply(keep[None, :]).tocsr()
        return dijkstra(graph, indices=np.atleast_1d(sources), min_only=True)

    def components(self, mask):
        ids = np.flatnonzero(mask)
        if not len(ids): return []
        n, label = connected_components(self.edges[ids][:, ids], directed=False)
        return [ids[label == k] for k in range(n)]

    def fraction(self, ids):
        return float(self.area[ids].sum()/self.total)


def grow(surface, rng, share, allowed):
    """One compact piece of `share` of the surface from one to three nearby sources, inside `allowed`."""
    first = int(rng.choice(np.flatnonzero(allowed), p=surface.area[allowed]/surface.area[allowed].sum()))
    radius = np.sqrt(share*surface.total/np.pi)
    d = surface.distance(first, allowed)
    sources = [first]
    for _ in range(int(rng.integers(0, 3))):
        ring = np.flatnonzero((d > .3*radius) & (d < .8*radius))
        if len(ring): sources.append(int(rng.choice(ring)))
    d = surface.distance(sources, allowed)
    order = np.argsort(d, kind='stable'); order = order[np.isfinite(d[order])]
    reached = np.cumsum(surface.area[order]) >= share*surface.total
    if not reached.any(): return None
    piece = np.zeros(len(d), bool); piece[order[:np.argmax(reached)+1]] = True
    return piece


def fill_holes(surface, pieces):
    """An unselected patch below MIN_PIECE whose neighbours all belong to one piece joins it."""
    owner = np.full(len(surface.area), -1)
    for k, p in enumerate(pieces): owner[p] = k
    for hole in surface.components(owner < 0):
        if surface.fraction(hole) >= MIN_PIECE: continue
        neighbours = np.unique(surface.edges[hole].indices)
        around = np.unique(owner[neighbours]); around = around[around >= 0]
        if len(around) == 1: pieces[around[0]][hole] = True
    return pieces


def smooth(surface, pieces, rounds=SMOOTHING):
    """A face takes the label (a piece or none) of at least two of its three neighbours; it never joins a
    piece while it touches another one, so pieces stay apart."""
    owner = np.full(len(surface.area), -1)
    for k, p in enumerate(pieces): owner[p] = k
    for _ in range(rounds):
        new = owner.copy()
        for f, nb in enumerate(surface.neighbours):
            labels = owner[nb]; values, counts = np.unique(labels, return_counts=True)
            top = values[np.argmax(counts)]
            if counts.max() < 2 or top == owner[f]: continue
            if top >= 0 and np.any((labels >= 0) & (labels != top)): continue
            new[f] = top
        if np.array_equal(new, owner): break
        owner = new
    return [owner == k for k in range(len(pieces))]


def area(surface, rng, gap, count):
    total = rng.uniform(LOW+.02, HIGH-.02)
    while True:
        shares = total*rng.dirichlet(np.full(count, 2.))
        if shares.min() >= 1.5*MIN_PIECE: break
    pieces = []; allowed = np.ones(len(surface.area), bool)
    for share in shares:
        piece = grow(surface, rng, share, allowed)
        if piece is None or len(surface.components(piece)) != 1: return None
        pieces.append(piece)
        allowed &= surface.distance(np.flatnonzero(piece)) > gap
        if not allowed.any(): return None
    pieces = fill_holes(surface, smooth(surface, pieces))
    if any(len(surface.components(p)) != 1 for p in pieces): return None
    selected = np.any(pieces, axis=0)
    fractions = [surface.fraction(p) for p in pieces]
    ok = (len(surface.components(selected)) == count and min(fractions) > MIN_PIECE and
          LOW <= surface.fraction(selected) <= HIGH)
    return pieces if ok else None


def record(name, surface, pieces, seed, digest):
    selected = np.any(pieces, axis=0)
    return dict(name=name, object='B', mesh_sha256=digest, mesh_face_count=len(surface.area),
                area_fraction=surface.fraction(selected), face_ids=np.flatnonzero(selected).tolist(),
                pieces=[dict(area_fraction=surface.fraction(p), face_ids=np.flatnonzero(p).tolist()) for p in pieces],
                seed=seed)


class Figure:
    """The bunny sitting (original +Y up) on one studio context, front and back side by side."""
    def __init__(self, mesh, size=(800, 860)):
        from whole_task_2.step1.reach_video import Scene
        setup = dict(needle_length_m=.1, head_radius_m=.02, needle_radius_m=.002, max_tilt_deg=25.,
                     working_direction_world=[0., 0., -1.])
        self.mesh = mesh; self.scene = Scene(mesh, setup, *size)
        W = np.eye(4); W[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
        v = mesh.vertices@W[:3, :3].T
        W[:3, 3] = [-(v[:, 0].min()+v[:, 0].max())/2, -(v[:, 1].min()+v[:, 1].max())/2, -v[:, 2].min()]
        self.W = W; self.points = mesh.vertices@W[:3, :3].T+W[:3, 3]

    def panel(self, colors, view):
        s = self.scene; s.view = np.asarray(view, float)/np.linalg.norm(view); s.fit(self.points, 1.12)
        data = s.draw(self.W, colors, [])
        return Image.frombytes('RGB', s.size, data).transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    def save(self, selected, label, path):
        from whole_task_2.step5.video import raster as r
        colors = np.tile(r.GRAY, (len(self.mesh.faces), 1)); colors[selected] = LIGHT_PINK
        panels = [self.panel(colors, view) for view in ([.45, -1., .5], [-.45, 1., .5])]
        w, h = panels[0].size
        image = Image.new('RGB', (2*w, h+70), 'white')
        for k, p in enumerate(panels): image.paste(p, (k*w, 0))
        draw = ImageDraw.Draw(image)
        draw.text((w, h+35), label, font=ImageFont.truetype(FONT, 34), fill='#27313b', anchor='mm')
        image.save(path)


def main():
    mesh = load_shape(MESH); surface = Surface(mesh); digest = sha256(MESH)
    gap = GAP*float(np.linalg.norm(mesh.extents))
    figure = Figure(mesh); index = []
    jobs = [('whole_shape', None)]+[(f'big_{i:02d}', SEED+i) for i in range(1, COUNT+1)]
    for name, seed in jobs:
        if seed is None:
            pieces = [np.ones(len(mesh.faces), bool)]
        else:
            # The piece count is drawn once, so retries do not drift toward the easier single piece.
            rng = np.random.default_rng(seed); count = int(rng.choice([1, 2, 3], p=[.4, .35, .25])); pieces = None
            for _ in range(5000):
                pieces = area(surface, rng, gap, count)
                if pieces is not None: break
            else:
                raise RuntimeError(f'{name}: no {count}-piece area in 5000 tries')
        row = record(name, surface, pieces, seed, digest)
        (HERE/f'{name}.json').write_text(json.dumps(row)+'\n')
        figure.save(np.any(pieces, axis=0), f'{name}   {100*row["area_fraction"]:.1f}%', HERE/f'{name}.png')
        index.append(dict(name=name, file=f'{name}.json', figure=f'{name}.png', area_fraction=row['area_fraction'],
                          piece_count=len(pieces), piece_area_fractions=[p['area_fraction'] for p in row['pieces']]))
        print(name, f'{100*row["area_fraction"]:.1f}%', 'pieces', [round(100*p['area_fraction'], 1) for p in row['pieces']], flush=True)
    (HERE/'index.json').write_text(json.dumps(dict(object='B', mesh_sha256=digest, areas=index), indent=1)+'\n')


if __name__ == '__main__':
    main()
