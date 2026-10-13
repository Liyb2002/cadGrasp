"""Layout-dependent ground enclosure and common-component oracle.

Only installed fixture states constrain the floor. The cheap phase contains
2D polygon operations, not solid Booleans or force solves. A positive final
answer additionally supplies a common connected free-space component.
Finite windows/conservative exit envelopes produce UNKNOWN, not global NO.
"""
from collections import OrderedDict
from pathlib import Path
import time
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from shapely.geometry import Polygon, MultiPoint
from shapely.ops import unary_union
from whole_search.common import C, transform_mesh, transform_solid, unpack_solid, RayMeshIntersector
from system_floor import system_floor_demands, transform_points
from pose_oracle import PoseOracleConstraints, transform_planes, SCHEMA
from work_access_shadow import shadow_clipped_access
from normal_work_access import normal_work_access
from fixture_state import installed_state
from component_membership import robust_contains


CONTACT_ANCHOR_POLICY='floor_adaptive_deterministic_winding_v3'


def contact_probe_points(points,normals,floor_planes,default_step):
    """Numerical interior probes, without reserving material/floor clearance.

    Use a mesh-containment probe, and shorten the displacement before any
    installed floor. A fixed 1um displacement can
    falsely reject an otherwise legal contact with a smaller positive gap.
    This changes neither the physical contact location nor its reaction ray.
    """
    points=np.asarray(points,float);normals=np.asarray(normals,float)
    planes=np.asarray(floor_planes,float).reshape(-1,4)
    steps=np.full(len(points),float(default_step))
    if len(planes) and len(points):
        gaps=points@planes[:,:3].T+planes[:,3]
        slopes=normals@planes[:,:3].T
        limits=np.full_like(gaps,np.inf)
        closing=slopes < -1e-14
        limits[closing]=.5*np.maximum(0.,gaps[closing])/(-slopes[closing])
        steps=np.minimum(steps,limits.min(1))
    return points+steps[:,None]*normals


def clip(points, a, b):
    points = np.asarray(points).reshape(-1, 2)
    if not len(points): return points
    values = points @ a + b
    if values.min() >= -1e-12: return points
    if values.max() < -1e-12: return np.empty((0, 2))
    output = []
    for p, q, f, g in zip(points, np.roll(points, -1, axis=0), values, np.roll(values, -1)):
        if f >= -1e-12: output.append(p)
        if (f >= -1e-12) != (g >= -1e-12):
            # Tolerant inclusion can classify a tiny negative f as inside.
            # Its intersection must stay on the edge, never extrapolate.
            output.append(p + np.clip(f/(f-g),0.,1.)*(q-p))
    return np.asarray(output).reshape(-1, 2)


def slice_polygon(points):
    """A convex halfspace slice, without buffering or geometric clearance.

    Coincident boundaries can leave tiny reversed edges after clipping. The
    slice is mathematically convex; order its existing points only if GEOS
    reports an invalid ring. No new points or width are introduced.
    """
    shape = Polygon(points) if len(points) >= 3 else Polygon()
    return shape if shape.is_valid else MultiPoint(points).convex_hull


def vertices(shape):
    if shape.is_empty: return np.empty((0, 2))
    if shape.geom_type == 'Polygon': return np.asarray(shape.exterior.coords)[:-1]
    if shape.geom_type in ('Point', 'LineString'): return np.asarray(shape.coords)
    pieces = [vertices(p) for p in shape.geoms]
    return np.vstack(pieces) if pieces else np.empty((0, 2))


def cover(required, points):
    """Pressure-centre coverage; holes in material are allowed."""
    polygon = MultiPoint(points).convex_hull if len(points) else Polygon()
    if polygon.is_empty: return False, 1., np.empty((0, 2))
    if polygon.geom_type == 'Polygon':
        # GEOS already returned an ordered convex boundary. Direct edge
        # inequalities handle arbitrarily thin polygons without Qhull jitter.
        v=vertices(polygon);edge=np.roll(v,-1,axis=0)-v
        length=np.linalg.norm(edge,axis=1);keep=length>1e-13
        signed=float(np.sum(v[:,0]*np.roll(v[:,1],-1)-v[:,1]*np.roll(v[:,0],-1)))
        normals=np.c_[edge[:,1],-edge[:,0]]* (1. if signed>=0 else -1.)
        normals=normals[keep]/length[keep,None]
        offset=np.einsum('ij,ij->i',normals,v[keep])
        gap=max(0.,float((required@normals.T-offset).max())) if len(normals) else 1.
    else:
        from shapely.geometry import Point
        gap = max(polygon.distance(Point(p)) for p in required)
    return gap <= 1e-8, float(gap), vertices(polygon)


def change_planes(planes, local):
    return transform_planes(planes, local)


def foot_obstacle_offsets(planes, width):
    """Conservative intersection with a cube centred width/2 above ground.

    Each obstacle face has its own cube support value. Summing coefficients
    across faces would make the clearance depend on mesh tessellation.
    """
    return planes[:, 3]+.5*width*planes[:, 2]-.5*width*np.abs(planes[:, :3]).sum(axis=1)


class BaseOracle:
    def __init__(self, model, padding=None, foot_width=0., work_mode='normal'):
        self.model = model
        if work_mode not in ('normal', 'angular'):
            raise ValueError('Unknown Oracle working-volume mode')
        self.work_mode = work_mode
        self.padding = 2*model.extent if padding is None else float(padding)
        self.foot_width = foot_width
        self.loads = []
        for task, pose in zip(model.tasks, model.poses):
            path = C.ROOT/'objects'/model.name/'poses'/pose/f'angle_{int(task.domain.data["load"]["cone_half_deg"])}'/'samples.npz'
            with np.load(path) as z: self.loads.append(z['need_wrench'].copy())
            if path not in model.inputs: model.inputs.append(path)
        self.cache = OrderedDict(); self.hulls = OrderedDict(); self.family_cache = OrderedDict()
        self.seconds = 0.; self.queries = 0; self.component_queries = 0
        self.last_geometry = None
        self.constraint_cache = OrderedDict()
        self.work_access=[normal_work_access(model.mesh, work) if work_mode=='normal' else
            shadow_clipped_access(model.mesh,work,
                task.domain.data.get('reachability',{}).get('ray_offset_m',1e-5))
            for work,task in zip(model.work_rays,model.tasks)]

    def rows(self, layout):
        result = []
        for k in layout.active:
            T = installed_state(self.model.native, layout, layout.hosts[k]); U = T @ layout.placements[k]
            key = k, np.round(U[:3, 3], 11).tobytes()
            if key not in self.hulls:
                com = transform_points(self.model.mesh.center_mass, U)
                xy = system_floor_demands(self.loads[k], com)['floor_demands_xy_m']
                hull = xy[ConvexHull(xy).vertices]
                self.hulls[key] = hull
                while len(self.hulls) > 256: self.hulls.popitem(last=False)
            body = transform_points(self.model.mesh.vertices, U)
            own = body[np.abs(body[:, 2]) <= 1e-9, :2]
            result.append(dict(index=k, pose=self.model.poses[k], T=T, U=U,
                               hull=self.hulls[key], own=own))
        return result

    def pose_constraints(self, layout):
        key = (tuple(layout.active),tuple(layout.hosts[list(layout.active)]),
               np.round(layout.placements[list(layout.active)],11).tobytes(),
               np.round(layout.state_offsets[np.unique(layout.hosts[list(layout.active)])],11).tobytes())
        if key not in self.constraint_cache:
            self.constraint_cache[key] = PoseOracleConstraints(self.model, layout,
                work_access=getattr(self,'work_access',None))
            while len(self.constraint_cache) > 256: self.constraint_cache.popitem(last=False)
        return self.constraint_cache[key]

    def families_by_pose(self, layout, *, bounds=None):
        """Keep work and exit constraints tagged by the task that owns them."""
        output = []
        for task in self.pose_constraints(layout).poses:
            k=task.index; Q = task.Q_fixture_object; inverse = np.linalg.inv(Q)
            d = Q[:3, :3].T @ layout.directions[k]
            length = self.model.length
            if bounds is not None:
                local = transform_points(bounds, inverse)
                length = max(length, float((local@d).max()-(self.model.mesh.vertices@d).min())+.01)
            key = k, np.round(Q, 11).tobytes(), np.round(d, 11).tobytes(), round(length, 11)
            if key not in self.family_cache:
                families = list(task.work_common)
                # A containing convex sweep is a safe fast screen. A failure
                # is UNKNOWN; exact union sweeps are used in component checks.
                body = self.model.mesh.vertices
                sweep = ConvexHull(np.vstack([body, body+length*d]))
                families.append(change_planes(sweep.equations, inverse))
                self.family_cache[key] = families
                while len(self.family_cache) > 256: self.family_cache.popitem(last=False)
            output.append(dict(index=int(k),work=self.family_cache[key][:-1],exit=self.family_cache[key][-1]))
        return output

    def families(self, layout, *, work_only=False, bounds=None):
        # Compatibility for existing consumers. This union is in permanent
        # fixture coordinates, never an overlay of unrelated world cones.
        output=[]
        for task in self.families_by_pose(layout,bounds=bounds):
            output.extend(task['work'])
            if not work_only:output.append(task['exit'])
        return output

    def query(self, layout, *, connected=False):
        key = layout.key(), connected
        if key in self.cache: return self.cache[key]
        began = time.monotonic(); self.queries += 1
        constraints = self.pose_constraints(layout)
        rows = self.rows(layout)
        # Exact, unbounded upper-bound rejection, including object contacts.
        certificates = []
        for row in rows:
            inv = np.linalg.inv(row['T'])
            for other in rows:
                t = other['T'] @ inv; a, b = t[2, :2], float(t[2, 3])
                lower = min(0., float((row['own'] @ a+b).min())) if len(row['own']) else 0.
                values = row['hull'] @ a+b; i = int(values.argmin())
                if values[i] < lower-1e-8:
                    certificates.append(dict(pose=row['pose'], other_pose=other['pose'],
                        gap_m=lower-float(values[i]), normal_xy=a.tolist(), offset_m=b,
                        demand_xy_m=row['hull'][i].tolist()))
        if certificates:
            result = dict(passed=False, enclosure_passed=False, connected_passed=False,
                status='certified_floor_incompatible', certificates=certificates,
                gap_m=max(c['gap_m'] for c in certificates), per_pose=[],
                query_seconds=time.monotonic()-began)
        else:
            points = [transform_points(self.model.mesh.vertices, layout.placements[k]) for k in layout.active]
            points += [transform_points(np.c_[r['hull'], np.zeros(len(r['hull']))], np.linalg.inv(r['T'])) for r in rows]
            joined = np.vstack(points); low, high = joined.min(0)-self.padding, joined.max(0)+self.padding
            bounds = np.array([[x,y,z] for x in [low[0],high[0]] for y in [low[1],high[1]] for z in [low[2],high[2]]])
            task_families = self.families_by_pose(layout, bounds=bounds); records = []; geometry = []
            floor_cache = {}
            for row in rows:
                floor_key = np.round(row['T'],11).tobytes()
                if floor_key in floor_cache:
                    legal, forbidden_shape, contacts, work_contacts = floor_cache[floor_key]
                    passed, gap, attainable = cover(row['hull'], np.vstack([contacts,row['own']]))
                    records.append(dict(pose=row['pose'], passed=passed, gap_m=gap,
                        legal_ground_xy_m=contacts.tolist(), attainable_xy_m=attainable.tolist(),
                        demand_hull_xy_m=row['hull'].tolist(), object_ground_xy_m=row['own'].tolist()))
                    records[-1]['work_only_enclosure_passed']=cover(row['hull'],np.vstack([work_contacts,row['own']]))[0]
                    geometry.append((row,legal,forbidden_shape)); continue
                inv = np.linalg.inv(row['T']); world = transform_points(joined, row['T'])
                lo, hi = world[:, :2].min(0)-self.padding, world[:, :2].max(0)+self.padding
                polygon = np.array([[*lo], [hi[0],lo[1]], [*hi], [lo[0],hi[1]]])
                for state in [r['T'] for r in rows]:
                    t = state @ inv
                    # Default zero inflation: arbitrarily narrow legal space
                    # remains available. A constructor can clip feet into it.
                    margin = .5*self.foot_width*(np.abs(t[2,:3]).sum()-t[2,2])
                    polygon = clip(polygon, t[2,:2], t[2,3]-margin)
                for axis in range(3):
                    polygon = clip(polygon, inv[axis,:2], inv[axis,3]-low[axis])
                    polygon = clip(polygon, -inv[axis,:2], high[axis]-inv[axis,3])
                legal = slice_polygon(polygon)
                forbidden = []; work_forbidden=[]; work_legal=legal
                if not legal.is_empty:
                    tagged = [(kind,family) for task in task_families
                              for kind,families in [('work',task['work']),('exit',[task['exit']])]
                              for family in families]
                    for kind,family in tagged:
                        p = change_planes(family, inv)
                        # Default width=0 leaves the original exclusions intact.
                        b = foot_obstacle_offsets(p, self.foot_width)
                        cut = polygon.copy()
                        for a, offset in zip(p[:,:2], b):
                            cut = clip(cut, -a, -offset)
                            if len(cut) < 3: break
                        if len(cut) >= 3:
                            forbidden.append(slice_polygon(cut))
                            if kind=='work':work_forbidden.append(forbidden[-1])
                    work_legal=legal.difference(unary_union(work_forbidden))
                    legal = legal.difference(unary_union(forbidden))
                contacts = vertices(legal)
                forbidden_shape = unary_union(forbidden)
                work_contacts=vertices(work_legal)
                floor_cache[floor_key] = legal, forbidden_shape, contacts, work_contacts
                all_points = np.vstack([contacts, row['own']])
                passed, gap, attainable = cover(row['hull'], all_points)
                records.append(dict(pose=row['pose'], passed=passed, gap_m=gap,
                    legal_ground_xy_m=contacts.tolist(), attainable_xy_m=attainable.tolist(),
                    demand_hull_xy_m=row['hull'].tolist(), object_ground_xy_m=row['own'].tolist()))
                records[-1]['work_only_enclosure_passed']=cover(row['hull'],np.vstack([work_contacts,row['own']]))[0]
                geometry.append((row, legal, forbidden_shape))
            enclosure = all(r['passed'] for r in records)
            result = dict(passed=enclosure and not connected, enclosure_passed=enclosure,
                connected_passed=False, status='enclosure_witness' if enclosure else 'not_found_in_window',
                gap_m=max(r['gap_m'] for r in records), per_pose=records, certificates=[],
                bounds_common_m=[low.tolist(), high.tolist()], foot_width_m=self.foot_width,
                conservative_sweep_screen=True, installed_states_only=True,
                work_only_enclosure_passed=all(r['work_only_enclosure_passed'] for r in records),
                query_seconds=time.monotonic()-began)
            self.last_geometry = (layout.key(), geometry, low, high)
            if connected and enclosure:
                result.update(self.components(layout, rows, low, high))
        result.update(constraint_schema=SCHEMA,work_constraints_pose_specific=True,
            contact_anchor_policy=('actual_connected_contact_core_containment'
                if getattr(self.model,'require_connected_core',False)else CONTACT_ANCHOR_POLICY),
            work_access_mode=getattr(self,'work_mode','angular'),
            work_access_definition='outward normal-ray union' if getattr(self,'work_mode','angular')=='normal'
                                   else 'outward angular cones with certified self-occlusion clipping',
            force_demand_angle_changed=False,contact_work_exclusion_changed=False,
            shared_material_checked_in_fixture_coordinates=True,
            other_task_world_cones_superimposed=False,
            support_contact_clearance_m=constraints.support_contact_clearance_m,
            base_feet_ground_height_m=0.)
        self.seconds += time.monotonic()-began
        self.cache[key] = result
        while len(self.cache) > 256: self.cache.popitem(last=False)
        return result

    def components(self, layout, rows, low, high):
        """One construction query on a finalist, not on gradient probes.

        A maximal legal component witnesses possible connection. Its volume
        is not the optimized base volume and it is never a minimum claim.
        """
        self.component_queries += 1; start = time.monotonic(); m = self.model
        self.core_clip_proposal=None
        stock_mesh = C.trimesh.creation.box(high-low)
        stock_mesh.apply_translation((high+low)/2)
        stock = C.S.solid(stock_mesh)
        for row in rows:
            T = row['T']; stock = stock.trim_by_plane(tuple(T[2,:3]), float(-T[2,3]/C.S.SCALE))
        bounds = np.array([[x,y,z] for x in [low[0],high[0]] for y in [low[1],high[1]] for z in [low[2],high[2]]])
        obstacles = []
        for task in self.pose_constraints(layout).poses:
            k=task.index; Q = task.Q_fixture_object; inv = np.linalg.inv(Q)
            work = task.work_access; cap = work.required_length(transform_points(bounds, inv), minimum=.001)
            obstacles.append(transform_solid(work.solid(cap), Q))
            direction = Q[:3,:3].T @ layout.directions[k]
            # Cap the sweep beyond the whole candidate stock, independently
            # of the historical contact guidance ray length.
            local = transform_points(bounds, inv)
            length = max(m.length, float((local@direction).max()-(m.mesh.vertices@direction).min())+.01)
            obstacles.append(transform_solid(m.clearance.sweep(length*direction, padded=False), Q))
        free = stock - C.union(obstacles)
        answers = []; winner = None
        flags = m.contact_delta.state(layout).available
        anchors = []
        floor_planes=np.array([row['T'][2] for row in rows])
        for k in layout.active:
            Q=layout.placements[k]; ids=np.flatnonzero(flags[k])
            points=transform_points(m.points[ids],Q)
            normals=m.point_normals[ids]@Q[:3,:3].T
            anchors.append(contact_probe_points(points,normals,floor_planes,1e-6))
        anchors = np.vstack(anchors) if anchors else np.empty((0,3))
        contact_core=None
        if getattr(m,'require_connected_core',False):
            from connected_core import prepare
            contact_core=C.S.solid(prepare(m,layout)[0])
        repair_loss=float('inf')
        for part in free.decompose():
            if C.material_volume(part) < 1e-12: continue
            mesh = unpack_solid(part); per_pose = []
            for row in rows:
                ground = transform_points(mesh.vertices, row['T'])
                contact = ground[np.abs(ground[:,2]) <= 1e-8,:2]
                passed, gap, hull = cover(row['hull'], np.vstack([contact,row['own']]))
                per_pose.append(dict(pose=row['pose'], passed=passed, gap_m=gap))
            core_outside=C.material_volume(contact_core-part)if contact_core is not None else None
            connected_contacts = (all(r['passed'] for r in per_pose) and
                (core_outside<=1e-10 if contact_core is not None else
                 (not len(anchors) or robust_contains(mesh,anchors).all())))
            if (contact_core is not None and all(r['passed']for r in per_pose)and
                    1e-10<core_outside<repair_loss and
                    core_outside<=C.material_volume(contact_core)*.01):
                self.core_clip_proposal=dict(layout_key=layout.key(),core=unpack_solid(contact_core^part),
                    witness=mesh,per_pose=per_pose,outside_before_m3=core_outside)
                repair_loss=core_outside
            if all(r['passed'] for r in per_pose) and connected_contacts:
                winner = mesh; answers = per_pose; break
            if sum(r['passed'] for r in per_pose) > sum(r['passed'] for r in answers): answers = per_pose
        if winner is not None: self.witness = (layout.key(), winner)
        return dict(passed=winner is not None, connected_passed=winner is not None,
            status='common_connected_enclosure_witness' if winner is not None else 'no_common_component_in_window',
            component_per_pose=answers, component_seconds=time.monotonic()-start,
            component_queries=self.component_queries, normal_pressure_enclosure_only=True,
            force_contacts_replayed=False, final_base_optimized=False)
