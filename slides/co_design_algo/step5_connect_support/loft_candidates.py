"""Whole, reference-style lofts as optional shared-material candidates."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pickle
import time

import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import MultiPoint

from step5_connect_support import build_coupled_saddle as S, material_graph as M


def volume(solid):
    return float(solid.volume())*S.SCALE**3*1e6


def carve(solid, context):
    return M.trim_floors(solid, context['bases'], context['offsets'])-context['forbidden']


def component_with(solid, required):
    for part in sorted(solid.decompose(), key=lambda q: -q.volume()):
        if all(volume(item-part) <= 8e-8 for item in required): return part
    return None


def floor_menu(demand):
    """Automatically generated choices, never a fixed assignment to heads."""
    hull = MultiPoint(demand).convex_hull
    vertices = np.asarray(hull.exterior.coords)[:-1]
    centers = []
    for margin in (.004, .018):
        rim = MultiPoint(vertices).convex_hull.buffer(margin, join_style=2)
        p = np.asarray(rim.exterior.coords)[:-1]
        for angle in np.arange(12)*2*np.pi/12:
            u = np.array([np.cos(angle), np.sin(angle)])
            centers.append(p[np.argmax(p@u)])
    rectangle = np.asarray(hull.buffer(.008, join_style=2).minimum_rotated_rectangle.exterior.coords)[:-1]
    centers.extend(rectangle)
    lo, hi = vertices.min(0), vertices.max(0)
    middle = (lo+hi)/2
    for margin in (.020, .035, .055):
        centers.extend([[lo[0]-margin,middle[1]], [hi[0]+margin,middle[1]],
                        [middle[0],lo[1]-margin], [middle[0],hi[1]+margin]])
    return np.unique(np.round(centers, 9), axis=0)


def terminal_section(target, demand):
    """Short radial depth and a broad tangential landing face."""
    center = (np.min(demand,axis=0)+np.max(demand,axis=0))/2
    normal = np.asarray(target)-center
    normal /= max(np.linalg.norm(normal),1e-12)
    tangent = np.array([-normal[1],normal[0]])
    return np.array([-.004*normal-.014*tangent, .004*normal-.014*tangent,
                     .004*normal+.014*tangent, -.004*normal+.014*tangent])


def build(case, context, work, minimum_retained_fraction=.5):
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    key = dict(context=context['key'], minimum_retained_fraction=minimum_retained_fraction,
        source=S.I.sha256(Path(__file__)), shared_geometry=S.I.sha256(Path(M.__file__)))
    cache = work/'loft_candidates.pickle'
    if cache.exists():
        with cache.open('rb') as f: saved = pickle.load(f)
        if saved['key'] == key:
            saved['solids'] = [S.solid(S.trimesh.Trimesh(v, f, process=False)) for v, f in zip(saved['vertices'], saved['faces'])]
            print('Reusing loft library', saved['stats'], flush=True)
            return saved
    began = time.monotonic(); heads = context['heads']; hcount = len(heads)
    vertices = [S.unpack(h).vertices for h in heads]
    roots = []
    for v, row in zip(vertices, context['records']):
        k = case.poses.index(row['head_pose'])
        roots.append(v+.008*context['directions'][k]@context['bases'][k])
    menus = [floor_menu(xy) for xy in case.demands]
    specs = []
    for h, v in enumerate(vertices):
        for k, (b, o) in enumerate(zip(context['bases'], context['offsets'])):
            xy = (v-o)@b[:2].T; xy = xy[ConvexHull(xy).vertices]
            center = xy.mean(0); projected = .7*(xy-center)
            for j, target in enumerate(np.vstack([center, menus[k]])):
                # The projected head section is the reference primitive. Small
                # rectangular terminals allow its additional oblique feet too.
                shapes = [('projected_head', projected)] if j == 0 else [
                    ('tangential_terminal', terminal_section(target,case.demands[k])),
                    ('axis_terminal',np.array([[-.006,-.012],[.006,-.012],[.006,.012],[-.006,.012]]))]
                for kind, section in shapes:
                    specs.append(dict(kind='head_floor_loft', head=h, floor=k,
                        terminal_family=kind, polygon_xy_m=(section+target).tolist()))
    for a in range(hcount):
        for b in range(a): specs.append(dict(kind='head_head_loft', head=a, other_head=b))

    def make(spec):
        h = spec['head']; original = heads[h]
        if spec['kind'] == 'head_floor_loft':
            k = spec['floor']; b, o = context['bases'][k], context['offsets'][k]
            xy = np.asarray(spec['polygon_xy_m'])
            pad = np.c_[xy, np.zeros(len(xy))]@b+o
            terminal = np.vstack([pad, pad+.003*b[2]])
            required_foot = carve(S.solid(S.G.hull_mesh(terminal)), context)
            if required_foot.is_empty(): return None
            fm = S.unpack(required_foot)
            if not len(M.floor_vertices(fm, context['bases'], context['offsets'])[k]): return None
            hull = S.solid(S.G.hull_mesh(np.vstack([vertices[h], roots[h], terminal])))
            restored = original
            required = [original, required_foot]
        else:
            other = spec['other_head']; restored = original+heads[other]
            hull = S.solid(S.G.hull_mesh(np.vstack([vertices[h], roots[h], vertices[other], roots[other]])))
            required = [original, heads[other]]
        floor_trimmed = M.trim_floors(hull, context['bases'], context['offsets'])
        solid = component_with(carve(hull, context)+restored, required)
        if solid is None: return None
        retained = volume(solid)/max(volume(floor_trimmed), 1e-12)
        if retained < minimum_retained_fraction: return None
        return solid, dict(spec, retained_fraction=retained, volume_cm3=volume(solid))

    solids = list(heads); records = [dict(kind='immutable_head', head=i) for i in range(hcount)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for j, made in enumerate(pool.map(make, specs)):
            if made is not None:
                s, r = made; solids.append(s); records.append(r)
            if j % 50 == 0: print('Loft candidates', j, '/', len(specs), 'retained', len(solids)-hcount, flush=True)
    meshes = [S.unpack(s) for s in solids]
    bounds = np.array([m.bounds for m in meshes])
    required = M.union(heads)
    costs = np.array([max(0., volume(s-required)) for s in solids]); costs[:hcount] = 0.
    pairs = []
    for i in range(len(solids)):
        for j in range(i):
            # Two alternatives attached to the same physical head already have
            # a real zero-cost connection through that immutable head node.
            if i >= hcount and j >= hcount and records[i]['head'] == records[j]['head']: continue
            if np.all(np.minimum(bounds[i,1], bounds[j,1])-np.maximum(bounds[i,0], bounds[j,0]) > 1e-7):
                pairs.append((i,j))
    def intersection(pair):
        i,j = pair
        return pair if volume(solids[i]^solids[j]) > .0001 else None
    with ThreadPoolExecutor(max_workers=4) as pool:
        edges = [edge for edge in pool.map(intersection, pairs) if edge is not None]
    floors = [M.floor_vertices(m, context['bases'], context['offsets']) for m in meshes]
    saved = dict(key=key, vertices=[m.vertices for m in meshes], faces=[m.faces for m in meshes],
        edges=np.asarray(edges, int).reshape(-1,2), floors=floors, cost=costs,
        heads=list(range(hcount)), records=records,
        stats=dict(nodes=len(solids), edges=len(edges), candidate_specs=len(specs),
            minimum_retained_fraction=minimum_retained_fraction, head_back_extension_m=.008,
            foot_thickness_m=.003, minimum_overlap_cm3=.0001,
            material_cost='Exact Boolean union increments; individual volumes only propose paths',
            candidate_construction='reference_style_head_floor_and_head_head_lofts',
            handcrafted_reference_terminals_used=False, build_seconds=time.monotonic()-began))
    with cache.open('wb') as f: pickle.dump(saved, f, protocol=5)
    saved['solids'] = solids
    print('Loft library ready', saved['stats'], flush=True)
    return saved
