"""Build compact, rounded, solid legs; route around obstacles instead of boxes.

The certified Step5.2 contact core is mandatory. Each new leg is tested as
actual material, not as a centre line. Routing voxels propose paths only;
all published geometry is clipped and checked using the original solids.
"""
import heapq
import time
import numpy as np
import manifold3d as md
from scipy.ndimage import distance_transform_edt
from scipy.spatial import ConvexHull, cKDTree
from matplotlib.path import Path as PolygonPath
from shapely.geometry import MultiPoint, Point

from deterministic_base import solid, unpack, volume, union, moved, transform, SCALE, enclosure
from prune_support import material_components


def capsule(a, b, radius=.004, root_radius=None):
    ball = md.Manifold.sphere(radius/SCALE, 24)
    first = ball.translate(tuple(np.asarray(a)/SCALE))
    last = (md.Manifold.sphere((root_radius or radius)/SCALE, 24)
            .translate(tuple(np.asarray(b)/SCALE)))
    return union([first, last]).hull()


def no_tunnels(value):
    """A single solid rod may have concave boundaries, but no tunnel or cavity."""
    if value.is_empty() or len(material_components(value)) != 1:
        return False
    mesh = unpack(value)
    mesh.merge_vertices(digits_vertex=12)
    return bool(mesh.is_watertight and mesh.euler_number == 2)


def bounded_domain(witness, body, rows, bounds):
    value = solid(witness)
    frames = list({np.round(r['fixture_transform'], 11).tobytes():
                   r['fixture_transform'] for r in rows}.values())
    for T in frames:
        for axis in range(3):
            value = value.trim_by_plane(tuple(T[axis, :3]),
                float((bounds[0, axis]-T[axis, 3])/SCALE))
            value = value.trim_by_plane(tuple(-T[axis, :3]),
                float((T[axis, 3]-bounds[1, axis])/SCALE))
    from whole_search.common import transform_mesh
    for row in rows:
        Q = np.linalg.inv(row['fixture_transform']) @ row['object_transform']
        value = value-solid(transform_mesh(body, Q))
    return value


def raster(value, origin, shape, pitch):
    """Even/odd polygon slices preserve genuine voids in the routing guide."""
    grid = np.zeros(shape, dtype=bool)
    xy = np.stack(np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
                             indexing='ij'), -1).reshape(-1, 2)*pitch+origin[:2]
    for k in range(shape[2]):
        contours = value.slice(float((origin[2]+k*pitch)/SCALE)).to_polygons()
        layer = np.zeros(len(xy), dtype=bool)
        for contour in contours:
            points = np.asarray(contour)*SCALE
            if len(points) < 3:
                continue
            low, high = points.min(0), points.max(0)
            ids = np.flatnonzero(np.all((xy >= low)&(xy <= high), axis=1))
            layer[ids] ^= PolygonPath(points).contains_points(xy[ids], radius=1e-12)
        grid[:, :, k] = layer.reshape(shape[:2])
    return grid


class RouteGuide:
    def __init__(self, domain, core, pitch=.0015):
        bounds = unpack(domain).bounds
        self.pitch = max(pitch, float(np.prod(bounds[1]-bounds[0])/12_000_000)**(1/3))
        self.origin = bounds[0]-self.pitch
        self.shape = tuple(np.ceil((bounds[1]-self.origin)/self.pitch).astype(int)+2)
        self.valid = raster(domain, self.origin, self.shape, self.pitch)
        self.roots = raster(core, self.origin, self.shape, self.pitch)&self.valid
        if not self.roots.any():
            raise ValueError('Routing resolution has no interior core seed')
        self.clearance = distance_transform_edt(self.valid, sampling=self.pitch).astype(np.float32)
        roots = np.argwhere(self.roots)
        self.root_tree = cKDTree(self.origin+roots[::max(1, len(roots)//8000)]*self.pitch)
        self.directions = np.array([(x,y,z) for x in [-1,0,1] for y in [-1,0,1]
                                   for z in [-1,0,1] if x or y or z], dtype=int)
        self.lengths = np.linalg.norm(self.directions, axis=1)*self.pitch

    def route(self, target, limit=250000):
        center = np.rint((target-self.origin)/self.pitch).astype(int)
        offsets = np.array([(x,y,z) for x in range(-3,4) for y in range(-3,4)
                            for z in range(-3,4)])
        ids = center+offsets
        ids = ids[np.all((ids >= 0)&(ids < self.shape), axis=1)]
        ids = ids[self.valid[tuple(ids.T)]]
        if not len(ids):
            raise ValueError('Target has no legal routing seed')
        distances = np.linalg.norm(self.origin+ids*self.pitch-target, axis=1)
        start = tuple(ids[np.argmin(distances)])
        heap = [(0., 0., start)]
        best = {start: 0.}
        previous = {}
        end = None
        explored = 0
        while heap and explored < limit:
            _, cost, key = heapq.heappop(heap)
            if cost != best.get(key):
                continue
            explored += 1
            if self.roots[key]:
                end = key
                break
            neighbors = np.asarray(key)+self.directions
            valid = np.all((neighbors >= 0)&(neighbors < self.shape), axis=1)
            indices = np.flatnonzero(valid)
            neighbors = neighbors[indices]
            valid = self.valid[tuple(neighbors.T)]
            neighbors, indices = neighbors[valid], indices[valid]
            clear = self.clearance[tuple(neighbors.T)]
            costs = cost+self.lengths[indices]*(1.+(.0025/np.maximum(clear,.0008))**2)
            xyz = self.origin+neighbors*self.pitch
            hints = self.root_tree.query(xyz)[0]
            for point, next_cost, hint in zip(neighbors, costs, hints):
                neighbor = tuple(point)
                if next_cost < best.get(neighbor, float('inf')):
                    best[neighbor] = float(next_cost)
                    previous[neighbor] = key
                    heapq.heappush(heap, (float(next_cost+1.05*hint), float(next_cost), neighbor))
        if end is None:
            raise ValueError('No bounded route to the certified core')
        path = [end]
        while path[-1] != start:
            path.append(previous[path[-1]])
        path = np.asarray(path[::-1])
        # Simplify only when the sampled shortcut stays within the route guide.
        selected = [path[0]]
        k = 0
        while k < len(path)-1:
            stop = min(len(path)-1, k+24)
            for j in range(stop, k, -1):
                n = max(2, int(np.linalg.norm(path[j]-path[k])*2)+1)
                samples = np.rint(np.linspace(path[k], path[j], n)).astype(int)
                if self.valid[tuple(samples.T)].all():
                    selected.append(path[j]); k = j; break
        xyz = self.origin+np.asarray(selected)*self.pitch
        return np.vstack([target, xyz]), dict(explored_voxels=explored,
            guide_pitch_m=self.pitch, path_length_m=float(np.linalg.norm(np.diff(xyz,axis=0),axis=1).sum()))


class Builder:
    def __init__(self, core, previous, witness, body, rows, bounds):
        self.core = solid(core)
        self.body = self.core
        self.previous = solid(previous)
        self.rows, self.object = rows, body
        self.bounds = bounds
        self.domain = bounded_domain(witness, body, rows, bounds)
        self.guide = None
        self.history = []
        self.targets = []
        self.started = time.monotonic()
        self.old_vertices = previous.vertices

    def required_targets(self):
        # First retain the original global BBX extrema with small connected ends.
        for axis in range(3):
            for side in [0,1]:
                edge = self.bounds[side,axis]
                objects = np.vstack([transform(self.object.vertices,r['object_transform']) for r in self.rows])
                if abs((objects.min(0) if side==0 else objects.max(0))[axis]-edge) <= 1e-9:
                    continue
                current = np.vstack([transform(unpack(self.core).vertices,r['fixture_transform']) for r in self.rows])
                if abs((current.min(0) if side==0 else current.max(0))[axis]-edge) <= 1e-9:
                    continue
                choices=[]
                for row in self.rows:
                    world=transform(self.old_vertices,row['fixture_transform'])
                    ids=np.flatnonzero(abs(world[:,axis]-edge)<=1e-10)
                    if len(ids):
                        centers=self.old_vertices[ids]
                        root=unpack(self.core).vertices
                        distances=cKDTree(root).query(centers)[0]
                        k=int(np.argmin(distances))
                        choices.append((float(distances[k]),centers[k]))
                if not choices: raise ValueError('Original BBX extremum has no real source vertex')
                self.targets.append(dict(point=min(choices,key=lambda x:x[0])[1],kind='bbx_extremum'))
        distinct={np.round(r['fixture_transform'],11).tobytes():r['fixture_transform'] for r in self.rows}
        for T in distinct.values():
            relevant=[r for r in self.rows if np.allclose(T,r['fixture_transform'],atol=1e-11,rtol=0)]
            original=transform(self.old_vertices,T)
            ground=np.unique(np.round(original[abs(original[:,2])<=1e-8,:2],12),axis=0)
            current=transform(unpack(self.core).vertices,T)
            existing=current[abs(current[:,2])<=1e-8,:2]
            if len(ground)<3: continue
            points=ground[ConvexHull(ground).vertices]
            active=set(range(len(points)))
            order=sorted(active,key=lambda k:(-float(cKDTree(current[:,:2]).query(points[k])[0]),k))
            for k in order:
                remaining=np.array([points[j] for j in sorted(active-{k})]).reshape(-1,2)
                ok=True
                for row in relevant:
                    cloud=np.vstack([remaining,existing,row['object_ground']])
                    hull=MultiPoint(cloud).convex_hull
                    if hull.is_empty or any(hull.distance(Point(p[:2]))>1e-8 for p in row['hull']):
                        ok=False;break
                if ok: active.remove(k)
            for k in sorted(active):
                point=transform([[*points[k],0.]],np.linalg.inv(T))[0]
                self.targets.append(dict(point=point,kind='ground_terminal',pose=relevant[0]['pose']))
        unique={}
        for target in self.targets:
            key=tuple(np.round(target['point'],9))
            unique.setdefault(key,target)
        self.targets=list(unique.values())

    def attach(self, target):
        point=target['point']
        current=unpack(self.body)
        distances,ids=cKDTree(current.vertices).query(point,k=min(160,len(current.vertices)))
        roots=[]
        for index in np.atleast_1d(ids):
            root=current.vertices[index]
            if not roots or min(np.linalg.norm(root-r) for r in roots)>.002:
                roots.append(root)
            if len(roots)>=8: break
        alternatives=[]
        for radius in [.004,.003,.005]:
            for root in roots:
                proposed=capsule(point,root,radius,root_radius=radius*1.35)^self.domain
                for piece in material_components(proposed):
                    mesh=unpack(piece)
                    if cKDTree(mesh.vertices).query(point)[0]>.000001:
                        # Distance to vertices is conservative; use a tiny real
                        # intersection instead when the point lies on a face.
                        probe=md.Manifold.sphere(.0002/SCALE,12).translate(tuple(point/SCALE))
                        if volume(piece^probe)<=1e-14: continue
                    if volume(piece^self.body)<=1e-13 or not no_tunnels(piece): continue
                    candidate=self.body+piece
                    if len(material_components(candidate))!=1: continue
                    added=volume(candidate)-volume(self.body)
                    alternatives.append((added,candidate,dict(operation='solid_rounded_leg',
                        kind=target['kind'],tip_m=point.tolist(),root_m=root.tolist(),
                        radius_m=radius,added_cm3=added*1e6,leg_tunnels=0)))
                if alternatives: break
            if alternatives: break
        if not alternatives:
            if self.guide is None:
                print('BUILD ROUTE GUIDE',flush=True)
                self.guide=RouteGuide(self.domain,self.core)
            path,description=self.guide.route(point)
            for radius in [.004,.003,.005]:
                proposed=union([capsule(a,b,radius) for a,b in zip(path[:-1],path[1:])])^self.domain
                for piece in material_components(proposed):
                    probe=md.Manifold.sphere(.0002/SCALE,12).translate(tuple(point/SCALE))
                    if volume(piece^probe)<=1e-14 or volume(piece^self.body)<=1e-13 or not no_tunnels(piece):continue
                    candidate=self.body+piece
                    if len(material_components(candidate))!=1:continue
                    added=volume(candidate)-volume(self.body)
                    alternatives.append((added,candidate,dict(operation='routed_solid_rounded_leg',
                        kind=target['kind'],tip_m=point.tolist(),path_m=path.tolist(),
                        radius_m=radius,added_cm3=added*1e6,leg_tunnels=0,**description)))
                if alternatives:break
        if not alternatives:
            raise ValueError('No continuous solid leg for '+str(target))
        _,candidate,record=min(alternatives,key=lambda x:x[0])
        if volume(candidate)>volume(self.previous)+1e-10:
            raise ValueError('Solid-leg material budget exceeded')
        self.body=candidate
        self.history.append(record)
        print('CONTINUOUS LEG',len(self.history),target['kind'],round(record['added_cm3'],3),'cm3',flush=True)

    def run(self):
        self.required_targets()
        for target in self.targets:
            self.attach(target)
        mesh=unpack(self.body)
        cover=enclosure(mesh,self.rows)
        from stage5_shared import occupied_bounds
        actual=occupied_bounds(self.object,mesh,self.rows)
        if not all(r['passed']for r in cover) or not np.allclose(actual,self.bounds,atol=1e-8,rtol=0):
            raise ValueError('Continuous legs did not preserve floor coverage and BBX')
        return mesh,dict(passed=True,connected=True,watertight=bool(mesh.is_watertight),
            material_cm3=volume(self.body)*1e6,added_material_cm3=volume(self.body-self.core)*1e6,
            retained_legal_step4_material_loss_cm3=volume(self.core-self.body)*1e6,
            per_pose=cover,history=self.history,leg_profile='continuous_rounded_solid',
            contact_core_preserved=True,large_box_bridge_used=False,
            target_count=len(self.targets),seconds=time.monotonic()-self.started)
