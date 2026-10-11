"""Deterministic growth with exact floor certificates and an immutable envelope."""
from __future__ import annotations

from itertools import permutations, product

import manifold3d as md
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import MultiPoint, Point
import trimesh

import stage5_common as C

TOL = 1e-9
SOLID_TOL_M3 = 1e-14


def polygon(points):
    points = np.asarray(points).reshape(-1,2)
    if not len(points): return np.empty((0,2))
    shape = MultiPoint(points).convex_hull
    return np.asarray(shape.exterior.coords)[:-1] if shape.geom_type == 'Polygon' else np.asarray(shape.coords)


def clip_polygon(vertices, a, offset, tol=1e-12):
    """Keep a dot xy + offset >= 0, including the polygon boundary."""
    vertices = np.asarray(vertices).reshape(-1,2)
    if not len(vertices): return vertices
    output = []
    for first, last in zip(vertices, np.roll(vertices,-1,axis=0)):
        f, g = float(first@a+offset), float(last@a+offset)
        inside_f, inside_g = f >= -tol, g >= -tol
        if inside_f: output.append(first)
        if inside_f != inside_g:
            output.append(first+np.clip(f/(f-g),0.,1.)*(last-first))
    return np.asarray(output).reshape(-1,2)


def floor_region(states, index, bounds):
    """Generous zero-thickness ground region allowed by every floor and BBX.

    Object/work/withdrawal obstacles are deliberately omitted here. Failure
    with this larger region is a necessary geometric rejection, not a search
    failure or a proof based on a finite candidate set.
    """
    low, high = np.asarray(bounds)
    vertices = np.array([[low[0],low[1]],[high[0],low[1]],
                         [high[0],high[1]],[low[0],high[1]]])
    inverse = np.linalg.inv(states[index])
    for state in states:
        transform = state@inverse
        vertices = clip_polygon(vertices, transform[2,:2], transform[2,3])
        for axis in range(3):
            vertices = clip_polygon(vertices, transform[axis,:2], transform[axis,3]-low[axis])
            vertices = clip_polygon(vertices, -transform[axis,:2], high[axis]-transform[axis,3])
    return vertices


def floor_certificates(rows, tolerance=TOL):
    """Prove fixed-layout impossibility with one separating floor inequality.

    Any fixture ground point for task i must have nonnegative height in task j.
    Object-ground points belong only to task i and are included separately.
    Every attainable system pressure center is a convex combination of these
    two types of point. A demand below their common lower bound is impossible,
    even for an arbitrarily large fixture and without any other obstacles.
    """
    certificates = []
    for i, row in enumerate(rows):
        inverse = np.linalg.inv(row['fixture_transform'])
        required = row['hull'][:,:2]
        own_ground = row['object_ground']
        for j, other in enumerate(rows):
            transform = other['fixture_transform']@inverse
            a, b = transform[2,:2], float(transform[2,3])
            minimum = min(0.,float(np.min(own_ground@a+b))) if len(own_ground) else 0.
            values = required@a+b
            k = int(np.argmin(values))
            gap = minimum-float(values[k])
            if gap > tolerance:
                certificates.append(dict(task_index=i, other_index=j, task=row['pose'],
                    other_task=other['pose'], demand_vertex_index=k,
                    demand_xy_m=required[k].tolist(), inequality_a_xy=a.tolist(),
                    inequality_offset_m=b, attainable_lower_bound_m=minimum,
                    demand_value_m=float(values[k]), separation_gap_m=gap,
                    violating_hull_vertex_count=int(np.sum(values<minimum-tolerance)),
                    object_ground_points_included=len(own_ground), bounding_box_independent=True))
    return sorted(certificates,key=lambda c:(-c['separation_gap_m'],c['task_index'],c['other_index']))


def ground_points(mesh, state):
    world = C.F.transform_points(mesh.vertices,state)
    points = world[np.abs(world[:,2])<=TOL,:2]
    return np.unique(points,axis=0)


def coverage(mesh, rows):
    floors, counts, worst = [], [], []
    for row in rows:
        points = np.vstack([ground_points(mesh,row['fixture_transform']),row['object_ground']])
        floors.append(points)
        shape = MultiPoint(points).convex_hull if len(points) else None
        distances = np.array([Point(q).distance(shape) if shape is not None else np.inf for q in row['hull'][:,:2]])
        counts.append(int(np.sum(distances<=TOL)))
        worst.append(float(np.max(distances)))
    return floors, counts, worst


def solid(mesh):
    result = md.Manifold(md.Mesh64(np.ascontiguousarray(mesh.vertices*1000,dtype=np.float64),
                                  np.ascontiguousarray(mesh.faces,dtype=np.uint32)))
    if result.status() != md.Error.NoError:
        raise ValueError('A valid closed mesh is required for geometric construction')
    return result


def unpack(value):
    data = value.to_mesh64()
    return trimesh.Trimesh(np.asarray(data.vert_properties[:,:3])/1000,
                           np.asarray(data.tri_verts),process=False)


def material_volume(value):
    if value.status() != md.Error.NoError: raise ValueError('Unresolved construction Boolean')
    return abs(float(value.volume()))*1e-9


def convex(vertices):
    return trimesh.convex.convex_hull(np.asarray(vertices))


class Grow:
    """One greedy state; straight attachments then six orthogonal detours.

    Candidate ordering is envelope expansion, all-pose coverage reuse, added
    material, path length, and fixed geometric ties. Exact added-solid checks
    precede commitment. Exhaustion is reported as finite-search exhaustion.
    """
    def __init__(self, mesh, rows, bounds, legal=None, width=.005, max_actions=256):
        self.original = mesh.copy()
        self.rows, self.states = rows, [r['fixture_transform'] for r in rows]
        self.bounds = np.asarray(bounds)
        self.legal_callback = legal
        self.width, self.max_actions = width, max_actions
        self.kernel = np.array(list(product([-width/2,width/2],repeat=3)))
        self.old = solid(mesh)
        self.body = self.old
        self.actions = []
        self.partial = mesh.copy()

    def legal(self, candidate):
        vertices = unpack(candidate).vertices
        if any(np.min(C.F.transform_points(vertices,t)[:,2]) < -TOL for t in self.states): return False
        added = candidate-self.old
        return self.legal_callback(added) if self.legal_callback is not None else True

    def foot(self, xy, state):
        half = self.width/2
        points = np.array([[xy[0]+x,xy[1]+y,z] for x,y,z in product([-half,half],[-half,half],[0.,self.width])])
        mesh = convex(C.F.transform_points(points,np.linalg.inv(state)))
        normal = state[2,:3]
        height = -float(np.min(self.kernel@normal))+1e-6
        start = C.F.transform_points([*xy,height],np.linalg.inv(state))
        return solid(mesh), start

    def rods(self, path):
        value = md.Manifold()
        for a,b in zip(path,path[1:]):
            if np.linalg.norm(b-a)>1e-12:
                value = value+solid(convex(np.vstack([a+self.kernel,b+self.kernel])))
        return value

    def routes(self, start, end):
        yield [start,end]
        for order in permutations(range(3)):
            path, point = [start], start.copy()
            for axis in order:
                point=point.copy(); point[axis]=end[axis]
                if np.linalg.norm(point-path[-1])>1e-12: path.append(point)
            yield path

    def proposal(self, addition, path, tag, required_counts):
        candidate = self.body+addition
        if not self.legal(candidate): return None
        mesh = unpack(candidate)
        _,counts,worst = coverage(mesh,self.rows)
        gain = sum(counts)-sum(required_counts)
        if tag['kind']=='ground' and gain<=0: return None
        bounds=C.add_bounds(self.bounds,mesh.vertices,self.states)
        increment=max(0.,C.volume(bounds)-C.volume(self.bounds))
        length=sum(float(np.linalg.norm(b-a)) for a,b in zip(path,path[1:]))
        added=material_volume(candidate-self.body)
        key=(round(increment,14),-gain,round(added,14),length,
             tuple(np.round(path[0],12)),tuple(np.round(path[-1],12)),tag['kind'])
        return dict(key=key,addition=addition,body=candidate,mesh=mesh,tag=tag,
                    path=path,coverage_counts=counts,coverage_gap=worst,bounds=bounds,
                    added_material_m3=added,box_expansion_m3=increment)

    def ground_actions(self, mesh, floors, counts):
        pool = []
        for i,row in enumerate(self.rows):
            shape=MultiPoint(floors[i]).convex_hull if len(floors[i]) else None
            for vertex in row['hull'][:,:2]:
                if shape is not None and Point(vertex).distance(shape)<=TOL: continue
                for dx,dy in product([-self.width/2,0.,self.width/2],repeat=2):
                    foot,start=self.foot(vertex+np.array([dx,dy]),row['fixture_transform'])
                    if not self.legal(foot): continue
                    # The old surface is always an attachment option. New
                    # branches become options too, allowing structural reuse.
                    endpoints=[]
                    for surface in [self.original,mesh]:
                        point=trimesh.proximity.closest_point(surface,start[None])[0][0]
                        endpoints.append(point)
                        indices=np.argsort(np.linalg.norm(surface.vertices-start,axis=1),kind='stable')[:4]
                        endpoints.extend(surface.vertices[indices])
                    endpoints=np.unique(np.asarray(endpoints),axis=0)
                    endpoints=sorted(endpoints,key=lambda p:(float(np.linalg.norm(p-start)),tuple(p)))
                    for end in endpoints:
                        for path in self.routes(start,end):
                            addition=foot+self.rods(path)
                            action=self.proposal(addition,path,dict(kind='ground',task=row['pose']),counts)
                            if action is not None:
                                pool.append(action)
                                break
        return pool

    def connect_components(self):
        for _ in range(self.max_actions-len(self.actions)):
            mesh=unpack(self.body)
            components=mesh.split(only_watertight=False)
            if len(components)<=1: return True
            _,counts,_=coverage(mesh,self.rows)
            candidates=[]
            for j,part in enumerate(components[1:],1):
                other=trimesh.util.concatenate(components[:j])
                point=part.vertices[np.argmin(np.linalg.norm(part.vertices-other.center_mass,axis=1))]
                end=trimesh.proximity.closest_point(other,point[None])[0][0]
                for path in self.routes(point,end):
                    action=self.proposal(self.rods(path),path,dict(kind='component_join',task=None),counts)
                    if action is not None and len(action['mesh'].split(only_watertight=False))<len(components):
                        candidates.append(action)
                        break
            if not candidates: return False
            action=min(candidates,key=lambda a:a['key'])
            self.body=action['body']; self.actions.append(action)
        return False

    def run(self):
        for _ in range(self.max_actions):
            mesh=unpack(self.body)
            floors,counts,worst=coverage(mesh,self.rows)
            if max(worst)<=TOL: break
            choices=self.ground_actions(mesh,floors,counts)
            if not choices: return self.result(False,'deterministic_attachment_search_exhausted')
            action=min(choices,key=lambda a:a['key'])
            self.body=action['body']; self.actions.append(action)
            if len(self.actions)==1: self.partial=action['mesh'].copy()
        else: return self.result(False,'growth_action_limit')
        if not self.connect_components(): return self.result(False,'component_attachment_search_exhausted')
        # Prune in reverse creation order, keeping every all-pose coverage and
        # connectivity constraint. Never remove any original support material.
        for action in self.actions[::-1]:
            trial=self.old
            for other in self.actions:
                if other is not action: trial=trial+other['addition']
            mesh=unpack(trial)
            if len(mesh.split(only_watertight=False))==1 and max(coverage(mesh,self.rows)[2])<=TOL:
                self.actions=[a for a in self.actions if a is not action]; self.body=trial
        return self.result(True,'geometric_ground_coverage_complete')

    def result(self, passed, status):
        mesh=unpack(self.body)
        floors,counts,worst=coverage(mesh,self.rows)
        final_bounds=C.add_bounds(self.bounds,mesh.vertices,self.states)
        return dict(passed=passed,status=status,mesh=mesh,partial=self.partial,
            initial_bounds_m=self.bounds.tolist(),final_bounds_m=final_bounds.tolist(),
            box_volume_increase_cm3=max(0.,C.volume(final_bounds)-C.volume(self.bounds))*1e6,
            added_material_cm3=material_volume(self.body-self.old)*1e6,
            original_material_removed_cm3=material_volume(self.old-self.body)*1e6,
            actions=[dict(a['tag'],path_m=[p.tolist() for p in a['path']],
                     added_material_cm3=a['added_material_m3']*1e6,
                     envelope_expansion_cm3=a['box_expansion_m3']*1e6) for a in self.actions],
            covered_hull_vertex_counts=counts,
            maximum_floor_hull_gap_m=[v if np.isfinite(v) else None for v in worst],
            geometric_coverage_only=True,full_ground_force_equilibrium_solved=False)


def preflight(data):
    certificates=floor_certificates(data['rows'])
    heights=[float(np.min(C.F.transform_points(data['fixture'].vertices,t)[:,2])) for t in data['states']]
    return dict(fixed_layout_floor_certificates=certificates,
                original_support_minimum_floor_height_m=heights,
                original_support_penetrates_a_floor=bool(min(heights)<-TOL),
                can_attempt_growth=not certificates and min(heights)>=-TOL)
