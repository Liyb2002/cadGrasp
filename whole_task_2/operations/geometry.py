"""Actual solid construction and full acceptance shared by all operations.

The accepted Step3.2 is the initial state. Seat translations and SO(3)
installations change fixture geometry while leaving the working head fixed.
World gravity and paired demands are rebuilt after every object pose change.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import shapely
import trimesh
from scipy.spatial import ConvexHull

from whole_task_2.step1.common import ROOT, DEFAULT_OUTPUT, sha256
from whole_task_2.common.force_cone import tangent_frame
from whole_task_2.step1.needle_geometry import geometric_mask
from whole_task_2.step3 import independent_geometry as G
from whole_task_2.step3.exit_geometry import (
    whole_object_sweep, grounded_material, condition_exit, refine_face_prisms,
    continuous_checks,
)
from whole_task_2.step3.run_independent import make_domain
from whole_task_2.step3.independent_physics import demands, certificate
from .state import Layout, transform


def moved_mesh(mesh, matrix):
    result = mesh.copy()
    result.apply_transform(matrix)
    return result


def contact_surface(body, support):
    """Unchanged actual boundary triangles near this body, for contact queries.

    The legacy contact kernel queries source-triangle AABBs with 1e-10 m
    padding. A triangle outside the entire body's padded AABB cannot be
    returned by any such query. No triangle is clipped or approximated here.
    """
    triangles=support.triangles
    low,high=body.bounds
    keep=(triangles.max(1)>=low-2e-10).all(1)&(triangles.min(1)<=high+2e-10).all(1)
    if not keep.any():raise RuntimeError('No actual support boundary near the task object')
    return trimesh.Trimesh(support.vertices,support.faces[keep],process=False)


class OperationModel:
    def note_repair_targets(self,name,kind,indices):
        """Remember actual failed constraints for subsequent gradient guidance."""
        attribute='repair_'+kind+'_indices'
        cache=getattr(self,attribute,None)
        if cache is None:setattr(self,attribute,{});cache=getattr(self,attribute)
        indices=np.asarray(indices,dtype=int)
        if len(indices)>32:indices=indices[np.unique(np.linspace(0,len(indices)-1,32).astype(int))]
        values=np.unique(np.r_[cache.get(name,np.empty(0,int)),indices])
        if len(values)>128:values=values[np.unique(np.linspace(0,len(values)-1,128).astype(int))]
        cache[name]=values

    def check_deadline(self):
        deadline=getattr(self,'proposal_deadline',None)
        if deadline is not None and time.monotonic()>deadline:
            raise RuntimeError('Proposal exceeded its evaluation budget; certified parent retained')

    def extend_exits_to_actual_support(self,candidate):
        """Use the saved complete fixture, including remote turned material.

        A stock-derived construction bound may be smaller than retained
        material. Length is a derived travel distance, not a pose variable.
        Extending it strengthens the subsequent original-prism check.
        """
        vertices=candidate['support'].vertices
        for task in candidate['tasks']:
            M=task['material'];n=M[:3,2]
            top=float(((vertices-M[:3,3])@n).max())
            direction=candidate['layout'].exits[task['index']]
            required=max(.01,(top+.015-task['body_world'].bounds[0,2])/direction[2])
            if required>task['length']:
                task['length']=float(required)
                task['exit_length_derived_from_actual_complete_fixture']=True

    def __init__(self, output_root=DEFAULT_OUTPUT, name='B', task_ids=None):
        self.output_root = Path(output_root).resolve()
        self.folder = self.output_root / name / 'step3/step3.2'
        report = json.loads((self.folder / 'report.json').read_text())
        validation = json.loads((self.folder / 'validation.json').read_text())
        if validation['status'] != 'passed' or validation['report_sha256'] != sha256(self.folder / 'report.json'):
            raise ValueError('Step3.2 must have an independently accepted saved initialization')
        for path, digest in report['input_sha256'].items():
            if sha256(path) != digest:
                raise ValueError('Changed Step3 input: ' + path)
        for path, digest in report['artifact_sha256'].items():
            if sha256(self.folder / path) != digest:
                raise ValueError('Changed Step3 artifact: ' + path)
        self.report = report
        self.name = name
        self.policy = deepcopy(report['policy'])
        self.tolerance = self.policy['boolean_volume_tolerance_m3']
        self.physical = report['physical_properties']
        self.records = [r for r in report['tasks'] if task_ids is None or r['id'] in task_ids]
        self.ids = [r['id'] for r in self.records]
        if task_ids is not None and len(self.ids) != len(task_ids):
            raise ValueError('Unknown task requested')
        self.workspace = deepcopy(report['source_workspace'])
        self.head = np.asarray(self.workspace['reference_head_front_world_m'], float)
        self.head[2] = report['common_tool_head_height_m']
        self.workspace['reference_head_front_world_m'] = self.head.tolist()
        self.axis = np.asarray(self.workspace['nominal_axis_world'], float)
        with np.load(self.folder / 'initialization.npz') as z:
            self.source = trimesh.Trimesh(z['source_vertices_object'], z['source_faces'], process=False)
            self.initial_support = trimesh.Trimesh(z['support_vertices_m'], z['support_faces'], process=False)
        self.inputs = [self.folder / p for p in ('report.json', 'validation.json', 'initialization.npz', 'support.obj')]
        outer_path = ROOT / 'whole_task_2/data' / name / 'step3_mold_plate/fitted_outer.npz'
        with np.load(outer_path) as z:
            self.outer = trimesh.Trimesh(z['vertices'], z['faces'], process=False)
        self.inputs.append(outer_path)
        self.initial = Layout(np.zeros((len(self.ids), 4, 4)),
                              np.tile(np.eye(4), (len(self.ids), 1, 1)),
                              np.asarray([r['exit_direction_world'] for r in self.records]))
        for i, record in enumerate(self.records):
            pose = np.asarray(record['world_from_object']).copy()
            offset = np.asarray(record['workspace']['reference_head_front_world_m']) - self.head
            pose[:3, 3] -= offset
            self.initial.poses[i] = pose
            self.initial.material[i, :3, 3] = offset
        self.pose_cache = {}
        self.geometry_cache = {}
        self.path_cache = {}
        self.world_sweep_cache = {}
        self.evaluations = 0

    def area_axis(self, layout, owner):
        ids = np.asarray(self.records[owner]['working_face_ids'])
        rotation = layout.poses[owner, :3, :3]
        center = np.average(self.source.triangles_center[ids], weights=self.source.area_faces[ids], axis=0)
        normal = np.average(self.source.face_normals[ids], weights=self.source.area_faces[ids], axis=0)
        normal = rotation @ normal
        if np.linalg.norm(normal)<1e-12:
            nearest=ids[np.linalg.norm(self.source.triangles_center[ids]-center,axis=1).argmin()]
            normal=rotation@self.source.face_normals[nearest]
        return normal / np.linalg.norm(normal), transform(center, layout.poses[owner])

    def geometry_key(self, pose, material, direction):
        return (pose.tobytes(), material.tobytes(), direction.tobytes(), getattr(self, 'safe_top', None))

    def path_checks(self, solid, paths):
        """Reuse a continuous proof only for the byte-identical saved solid/path.

        The independent disk replay creates a fresh empty cache. This avoids
        repeating an expensive primitive check during the same acceptance.
        """
        self.check_deadline()
        mesh = G.mesh(solid)
        digest = hashlib.sha256()
        for array in (mesh.vertices,mesh.faces,np.asarray([self.tolerance])):
            digest.update(np.ascontiguousarray(array).tobytes())
        for name,body,direction,length in paths:
            self.check_deadline()
            digest.update(name.encode())
            for array in (body.vertices,body.faces,np.asarray(direction),np.asarray([length])):
                digest.update(np.ascontiguousarray(array).tobytes())
        key = digest.hexdigest()
        if key in self.path_cache:
            return self.path_cache[key],[]
        rows,bad = [],[]
        for name,body,direction,length in paths:
            self.check_deadline()
            # Every original face prism lies inside the AABB of the body's
            # two endpoint vertex sets. Crop only the audit's temporary
            # obstacle to that complete envelope; distant seats cannot
            # intersect any point of this continuous path. All original
            # prisms and the original volume tolerance are retained.
            points=np.vstack([body.vertices,body.vertices+length*np.asarray(direction)])
            low,high=points.min(0)-1e-6,points.max(0)+1e-6
            # A box aligned with the extraction direction contains the same
            # complete path, without the large empty corners of a diagonal
            # world AABB. All original prisms remain inside with 1 um margin.
            u,v=tangent_frame(np.asarray(direction))
            frame=np.column_stack([u,v,direction])
            projected=body.vertices@frame
            local_low=projected.min(0)-1e-6
            local_high=projected.max(0)+1e-6
            local_high[2]+=length
            envelope=G.mesh(G.box(local_low,local_high))
            envelope.vertices=envelope.vertices@frame.T
            nearby=solid^G.solid(envelope)
            local,failures=continuous_checks(nearby,[(name,body,direction,length)],self.tolerance)
            local[0]['full_sweep_bounding_crop_m']=[low.tolist(),high.tolist()]
            local[0]['exit_aligned_crop']=dict(frame=frame.tolist(),
                low_m=local_low.tolist(),high_m=local_high.tolist(),all_original_prisms_retained=True)
            rows.extend(local)
            bad.extend(failures)
        if all(r['passed'] for r in rows):
            self.path_cache[key] = rows
            if len(self.path_cache)>64:
                self.path_cache.pop(next(iter(self.path_cache)))
        return rows,bad

    def material_component_count(self,support,solid):
        """Reuse topology only for byte-identical physical boundary arrays."""
        key=hashlib.sha256()
        for array in (support.vertices,support.faces):
            a=np.ascontiguousarray(array)
            key.update(str((a.dtype.str,a.shape)).encode());key.update(a.tobytes())
        key=key.hexdigest()
        cache=getattr(self,'component_cache',None)
        if cache is None:self.component_cache=cache={}
        if key not in cache:
            cache[key]=sum(p.volume()*G.SCALE**3>1e-15 for p in solid.decompose())
            if len(cache)>16:cache.pop(next(iter(cache)))
        return cache[key]

    def world_data(self, record, pose):
        self.check_deadline()
        key = (record['id'], pose.tobytes())
        if key not in self.pose_cache:
            body = moved_mesh(self.source, pose)
            if body.bounds[0, 2] < -1e-12:
                raise ValueError(record['id'] + ': object crosses the world floor')
            ids = np.asarray(record['working_face_ids'])
            domain = make_domain(self.source, pose, ids, record['id'], self.workspace, self.physical, self.policy)
            endpoints, samples = demands(domain, self.policy)
            outer_world = moved_mesh(self.outer, pose)
            stock_world = G.casting_stock(outer_world, self.policy['plate_thickness_m'])
            bounds = G.mesh(stock_world).bounds
            if bounds[0, 2] < 0:
                stock_world = stock_world ^ G.box([*(bounds[0, :2]-.01), 0.], bounds[1]+.01)
            self.pose_cache[key] = dict(body_world=body, domain=domain, endpoints=endpoints, samples=samples,
                stock_world=G.mesh(stock_world))
        return self.pose_cache[key]

    def prepare(self, layout):
        prepared = []
        for i, record in enumerate(self.records):
            pose, material, direction = layout.poses[i], layout.material[i], layout.exits[i]
            if not all(np.isfinite(a).all() for a in (pose, material, direction)):
                raise ValueError('Nonfinite operation state')
            for matrix in (pose, material):
                if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-12, rtol=0)
                        or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-12, rtol=0)
                        or abs(np.linalg.det(matrix[:3, :3]) - 1) > 1e-12):
                    raise ValueError('Expected a proper rigid transform')
            if abs(np.linalg.norm(direction) - 1) > 1e-12 or direction[2] < .05:
                raise ValueError('Invalid upper-hemisphere extraction direction')
            ids = np.asarray(record['working_face_ids'])
            head_object = (self.head - pose[:3, 3]) @ pose[:3, :3]
            forward_object = pose[:3, :3].T @ self.axis
            if not geometric_mask(self.source, head_object, forward_object, self.workspace)[ids].all():
                raise ValueError(record['id'] + ': an original working triangle leaves the needle sector')
            if self.source.contains(head_object[None])[0]:
                raise ValueError(record['id'] + ': the working head center is inside the object')
            row = dict(self.world_data(record, pose), id=record['id'], index=i, material=material,
                       pose=pose, working_face_ids=ids, workspace=deepcopy(self.workspace))
            geometric_key = self.geometry_key(pose, material, direction)
            if geometric_key not in self.geometry_cache:
                body = moved_mesh(self.source, material @ pose)
                # Casting follows gravity in the WORKING WORLD, then moves
                # rigidly into the common material coordinates. In particular
                # a sideways installation has a sideways backing and feet.
                stock = G.solid(moved_mesh(row['stock_world'], material))
                ws = deepcopy(self.workspace)
                ws['reference_head_front_world_m'] = transform(self.head, material).tolist()
                ws['nominal_axis_world'] = (material[:3, :3] @ self.axis).tolist()
                work, _ = G.work_sector(ws, self.policy)
                d = material[:3, :3] @ direction
                # A fixed safe top bound contains every admissible seat at the
                # fixed head height and original 100mm work radius.
                safe_top = getattr(self, 'safe_top', max(.6, self.head[2] + 3 * self.source.extents.max()))
                minimum = max(.01, (safe_top + .015 - row['body_world'].bounds[0, 2]) / direction[2])
                length = minimum
                sweep_length = float(np.ceil(minimum/.1)*.1)
                # A rigid installation must not rebuild thousands of sweep
                # cells in a different floating point frame. Construct the
                # same exact nonconvex world sweep once and rigidly move it.
                # Final original-face checks still certify the actual solid.
                sweep_key=(record['id'],pose.tobytes(),direction.tobytes(),sweep_length)
                if sweep_key not in self.world_sweep_cache:
                    from .sweep_cache import sweep_mesh,implementation_key
                    if not hasattr(self,'sweep_cache_implementation'):
                        self.sweep_cache_implementation=implementation_key()
                    def construct_sweep():
                        world_sweep,_=whole_object_sweep(row['body_world'],direction,sweep_length)
                        return G.mesh(world_sweep)
                    self.world_sweep_cache[sweep_key]=sweep_mesh(
                        ROOT/'whole_task_2/data'/self.name/'step4_world_sweeps',
                        row['body_world'],direction,sweep_length,construct_sweep,
                        self.sweep_cache_implementation)
                    if len(self.world_sweep_cache)>48:
                        self.world_sweep_cache.pop(next(iter(self.world_sweep_cache)))
                sweep=G.solid(moved_mesh(self.world_sweep_cache[sweep_key],material))
                self.geometry_cache[geometric_key] = dict(body=body, stock=stock, work=work,
                    sweep=sweep, direction=d, length=float(length),
                    sweep_construction_length=float(sweep_length),workspace_material=ws)
                if len(self.geometry_cache) > 96:
                    self.geometry_cache.pop(next(iter(self.geometry_cache)))
            row.update(self.geometry_cache[geometric_key])
            pressure = G.pressure_points(row['endpoints']['endpoint_need_wrench'], row['domain'].com)
            row['pressure_world'] = pressure
            row['pressure_3d'] = transform(np.c_[pressure, np.zeros(len(pressure))], material)
            row['pressure'] = row['pressure_3d'][:, :2]
            prepared.append(row)
        return prepared

    def build(self, layout, *, facewise=False):
        if not all(np.allclose(M[2],[0,0,1,0],atol=1e-12,rtol=0)for M in layout.material):
            # The original yaw-only constructor remains for the existing
            # demonstrations. Spatial proposals use the floor-aware builder.
            from copy import copy
            from .construction import CompactConstructionModel
            spatial=copy(self)
            spatial.__class__=CompactConstructionModel
            return spatial.build(layout,facewise=facewise)
        began = time.monotonic()
        tasks = self.prepare(layout)
        # One material plate; its convex footprint includes ACTUAL pressure
        # envelopes and the projected casting stocks. No reference floor ray
        # is substituted for contact with the saved physical bottom.
        footprint = np.vstack([np.vstack([G.mesh(t['stock']).vertices[:, :2], t['pressure']]) for t in tasks])
        hull = footprint[ConvexHull(footprint).vertices]
        margin = self.policy['plate_edge_margin_m']
        offsets = np.array([[x, y] for x in (-margin, margin) for y in (-margin, margin)])
        hull = (hull[:, None] + offsets).reshape(-1, 2)
        top = self.policy['plate_thickness_m']
        import manifold3d as md
        plate = md.Manifold.hull_points(np.vstack([np.c_[hull, np.zeros(len(hull))],
                                                    np.c_[hull, np.full(len(hull), top)]]) / G.SCALE)
        parent = G.union([plate, *[t['stock'] for t in tasks]])
        forbidden = [t[k] for t in tasks for k in ('sweep', 'work')]
        cut = parent - G.union(forbidden)
        cut = grounded_material(cut)
        # The union is a construction acceleration. Its near-coincident
        # internal boundaries are not the exit certificate: replay every
        # original leading-face prism below. This avoids trusting a serialized
        # union whose Boolean boundary can disagree with its primitive cells.
        actual_forbidden = [p for t in tasks for p in (G.solid(t['body']), t['work'])]
        support, conditioning = condition_exit(cut, actual_forbidden, parent, self.tolerance)
        paths = [(t['id'], t['body'], t['direction'], t['length']) for t in tasks]
        if facewise:
            support, refinement = refine_face_prisms(G.solid(support), paths, parent,
                                                     actual_forbidden, self.tolerance)
            conditioning['source_face_prisms'] = refinement
        return dict(layout=layout.copy(), support=support, tasks=tasks,
                    volume_cm3=float(support.volume * 1e6), construction=conditioning,
                    build_seconds=time.monotonic() - began)

    def audit(self, candidate, *, full=True, continuous=True, all_outer=False):
        """Rebuild each load domain and use only contacts of the actual solid.

        Reduced loads can reject proposals, but only full=True may be accepted.
        Continuous source-face-prism checks use the original volume tolerance.
        """
        self.check_deadline()
        began = time.monotonic()
        self.evaluations += 1
        support, tasks = candidate['support'], candidate['tasks']
        self.extend_exits_to_actual_support(candidate)
        if not support.is_watertight or not support.is_winding_consistent or support.volume <= 0:
            raise ValueError('Fixture is not a closed consistently oriented solid')
        solid = G.solid(support)
        if self.material_component_count(support,solid) != 1:
            raise ValueError('Fixture has several material components')
        paths = [(t['id'], t['body'], t['direction'], t['length']) for t in tasks]
        if continuous:
            checks, _ = self.path_checks(solid, paths)
            if not all(r['passed'] for r in checks):
                raise ValueError('Continuous original-face-prism exit failed: ' + str(checks))
        else:
            checks = []
        records = []
        for t in tasks:
            self.check_deadline()
            overlaps = dict(body=G.volume(solid ^ G.solid(t['body'])), work=G.volume(solid ^ t['work']))
            if max(overlaps.values()) > self.tolerance:
                raise ValueError(t['id'] + ': object or work-volume interference')
            world_support = moved_mesh(support, np.linalg.inv(t['material']))
            if world_support.bounds[0, 2] < -1e-9:
                raise ValueError(t['id'] + ': installed support crosses physical floor')
            ground = world_support.triangles[np.abs(world_support.triangles[:, :, 2]).max(1) < 1e-9, :, :2]
            footprint = shapely.union_all(shapely.polygons(ground))
            pressure = G.pressure_points(t['endpoints']['endpoint_need_wrench'], t['domain'].com)
            pressure_covered=shapely.covers(footprint, shapely.points(pressure))
            if not pressure_covered.all():
                self.note_repair_targets(t['id'],'pressure',np.flatnonzero(~pressure_covered))
                raise ValueError(t['id'] + ': actual floor does not contain every pressure center')
            if full:
                from .contact_cache import cached_certificate_contacts
                cache=getattr(self,'actual_contact_cache',None)
                if cache is None:self.actual_contact_cache=cache={}
                with cached_certificate_contacts(cache):
                    proof, data = certificate(t['body_world'], contact_surface(t['body_world'],world_support), t['domain'],
                                              t['endpoints'], t['samples'], search=True)
                if not proof['passed']:
                    if 'unpassed_outer_index' in proof:
                        self.note_repair_targets(t['id'],'endpoint',[proof['unpassed_outer_index']])
                    for key,kind in (('endpoint_passed','endpoint'),('sample_passed','sample')):
                        if key in data:self.note_repair_targets(t['id'],kind,np.flatnonzero(~data[key]))
                    raise ValueError(t['id'] + ': paired force/moment certificate failed ' + str(proof))
                t['contact_data'] = data
            else:
                from whole_task_2.step3.independent_physics import contacts
                from whole_task_2.step3.legacy import kernel as K
                from .contact_cache import query_contacts
                cache=getattr(self,'actual_contact_cache',None)
                if cache is None:self.actual_contact_cache=cache={}
                _, _, columns, scale = query_contacts(t['body_world'], contact_surface(t['body_world'],world_support),
                    t['domain'].work_ids,t['domain'].com,cache,contacts)
                endpoints = t['endpoints']['endpoint_need_wrench']
                # One failed original endpoint suffices to reject a trial.
                # Do not classify thousands of already-infeasible endpoints
                # merely to discover that first failure. Complete original
                # loads remain mandatory in the final full certificate.
                known=getattr(self,'repair_endpoint_indices',{}).get(t['id'],np.empty(0,int))
                known=known[(known>=0)&(known<len(endpoints))]
                probes=np.unique(np.r_[np.linspace(0,len(endpoints)-1,min(32,len(endpoints))).astype(int),known])
                probe_targets=K.J.C.U.target(endpoints[probes]*scale,columns.shape[1])
                for index,target in zip(probes,probe_targets):
                    if K.J.C.W.solve(columns,target) is None:
                        self.note_repair_targets(t['id'],'endpoint',[index])
                        raise ValueError(t['id']+': reduced endpoint rejection')
                indices = np.arange(len(endpoints))if all_outer else np.unique(np.linspace(0, len(endpoints) - 1, 48).astype(int))
                passed, _ = K.J.classify(columns, endpoints[indices] * scale)
                if not passed.all():
                    self.note_repair_targets(t['id'],'endpoint',indices[~passed])
                    raise ValueError(t['id'] + ': reduced endpoint rejection')
                proof = dict(passed=True, preflight_only=True)
            end = t['body_world'].vertices + t['length'] * candidate['layout'].exits[t['index']]
            gap = float(end[:, 2].min() - world_support.bounds[1, 2])
            if gap < .015 - 1e-12:
                raise ValueError(t['id'] + ': full object exit endpoint is not clear')
            records.append(dict(id=t['id'], force=proof, overlap_m3=overlaps, final_vertical_gap_m=gap,
                floor=dict(passed=True,minimum_world_z_m=float(world_support.bounds[0,2]),
                    actual_contact_area_m2=float(footprint.area),pressure_center_count=len(pressure),
                    all_pressure_centers_on_actual_bottom=True,
                    world_up_in_material=t['material'][:3,2].tolist())))
        result = dict(passed=True, full_loads=full, source_face_prisms_checked=continuous,
                      tasks=records, continuous_paths=checks, seconds=time.monotonic() - began)
        candidate['audit'] = result
        return result

    def accept(self, layout):
        candidate = self.build(layout, facewise=True)
        self.audit(candidate)
        return candidate
