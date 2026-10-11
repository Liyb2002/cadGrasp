"""Construct a connected base in all moved states, retaining legal contacts.

The finite stock window is recorded; all cones and withdrawal sweeps are
capped strictly beyond it. Floor planes are exact halfspace intersections.
One construction acceptance occurs before export. No prior model is edited.
"""
from __future__ import annotations

from itertools import product

import manifold3d as md
import numpy as np
import trimesh

import stage5_common as C
import co_common as K
from exit_clearance import ExitClearance
from work_access import WorkAccess

EPS_VOLUME = 1e-14


def solid(mesh):
    result=md.Manifold(md.Mesh64(np.ascontiguousarray(mesh.vertices*1000, dtype=np.float64),
                                np.ascontiguousarray(mesh.faces, dtype=np.uint32)))
    if result.status()!=md.Error.NoError:
        raise RuntimeError(f'Input closed solid unresolved: {result.status()}')
    return result


def unpack(value):
    data=value.to_mesh64()
    return trimesh.Trimesh(np.asarray(data.vert_properties[:, :3])/1000,
                           np.asarray(data.tri_verts), process=False)


def volume(value):
    if value.status()!=md.Error.NoError: raise RuntimeError(f'Construction Boolean unresolved: {value.status()}')
    return abs(float(value.volume()))*1e-9


def transform(value, matrix):
    affine=np.asarray(matrix)[:3].copy(); affine[:, 3]*=1000
    return value.transform(affine.tolist())


def from_legacy(value):
    return value.scale((1000*K.S.SCALE,)*3)


def combine(parts):
    if not parts: return md.Manifold()
    result=md.Manifold.batch_boolean(parts, md.OpType.Add)
    volume(result)
    return result


def closed_prism(triangle, normal, depth=.005):
    faces=np.array([[2,1,0],[3,4,5],[0,1,4],[0,4,3],[1,2,5],[1,5,4],[2,0,3],[2,3,5]])
    mesh=trimesh.Trimesh(np.vstack([triangle, triangle+normal*depth]), faces, process=False)
    if mesh.volume<0: mesh.invert()
    return solid(mesh)


class Construction:
    def __init__(self, data, tasks, body, padding=.025):
        self.data, self.tasks, self.body=data, tasks, body
        self.padding=padding
        self.placements=[np.linalg.inv(row['fixture_transform'])@row['object_transform'] for row in data['rows']]
        points=[data['fixture'].vertices]
        for row, placement in zip(data['rows'], self.placements):
            points.append(C.F.transform_points(body.vertices, placement))
            points.append(C.F.transform_points(row['hull'], np.linalg.inv(row['fixture_transform'])))
        points=np.vstack(points)
        self.low,self.high=points.min(0)-padding, points.max(0)+padding
        self.window=np.array(list(product(*zip(self.low,self.high))))
        stock=md.Manifold.cube(tuple((self.high-self.low)*1000)).translate(tuple(self.low*1000))
        for state in data['states']:
            stock=stock.trim_by_plane(tuple(state[2, :3]), float(-state[2, 3]*1000))
        self.stock=stock
        self.old=solid(data['fixture'])
        self.nominal, self.padded, self.work, self.allowed=[], [], [], []
        clearance=ExitClearance(body)
        self.exit_metadata=clearance.metadata
        self.exclusions=[]
        for task, row, placement in zip(tasks, data['rows'], self.placements):
            local_window=C.F.transform_points(self.window, np.linalg.inv(placement))
            direction=placement[:3, :3].T@row['common_exit_direction']
            direction/=np.linalg.norm(direction)
            length=max(.001, float((local_window@direction).max()-(body.vertices@direction).min())
                       +2*clearance.radius+.01)
            self.nominal.append(transform(from_legacy(clearance.sweep(length*direction, padded=False)), placement))
            self.padded.append(transform(from_legacy(clearance.sweep(length*direction, padded=True)), placement))
            work=WorkAccess(body, task.domain.work_ids, task.domain.data['load']['cone_half_deg'])
            work_length=work.required_length(local_window, minimum=.001)
            self.work.append(transform(from_legacy(work.solid(work_length)), placement))
            ids=np.setdiff1d(np.arange(len(body.faces)), task.domain.work_ids)
            ids=ids[body.face_normals[ids]@direction<=1e-9]
            self.allowed.append(ids)
            self.exclusions.append(dict(pose=row['pose'], sweep_cap_m=length, work_cap_m=work_length,
                                        all_work_triangles_reserved=True, cap_beyond_entire_stock=True))
        self.nominal_cut=combine(self.nominal)
        self.work_cut=combine(self.work)
        nominal=stock-self.nominal_cut-self.work_cut
        nominal_mesh=unpack(nominal)
        self.contact_cores=[]
        self.initial_contact_areas=[]
        for placement, allowed in zip(self.placements, self.allowed):
            local=nominal_mesh.copy(); local.apply_transform(np.linalg.inv(placement))
            triangles, sources=(K.contact_boundary(body, local, allowed) if len(local.faces)
                                else (np.empty((0,3,3)),np.empty(0,int)))
            parts=[transform(closed_prism(tri, body.face_normals[src]), placement)
                   for tri,src in zip(triangles, sources)]
            core=combine(parts)^nominal if parts else md.Manifold()
            self.contact_cores.append(core)
            self.initial_contact_areas.append(float(np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],
                triangles[:,2]-triangles[:,0]), axis=1).sum()/2))
        # Each task has its own physically present contact-core exception.
        # A core from another pose cannot excuse interference with this pose.
        padded_forbidden=combine([sweep-core for sweep,core in zip(self.padded,self.contact_cores)])
        self.free=nominal-padded_forbidden
        self.forbidden_clearance=padded_forbidden

    def components(self, hollow=False, wall=.005):
        free=self.free
        if hollow:
            inner=md.Manifold.cube(tuple((self.high-self.low-2*wall)*1000)).translate(tuple((self.low+wall)*1000))
            for state in self.data['states']:
                inner=inner.trim_by_plane(tuple(state[2, :3]), float((-state[2,3]+wall)*1000))
            # Preserve the fitted object-adjacent contact cores and original
            # material where available while hollowing the excess stock.
            preserve=combine([self.old]+self.contact_cores)
            free=free-(inner-preserve)
        return sorted(free.decompose(), key=lambda value:-volume(value))

    def checks(self, value):
        mesh=unpack(value)
        minimum=[float(C.F.transform_points(mesh.vertices, state)[:,2].min()) for state in self.data['states']]
        nominal_overlap=volume(value^self.nominal_cut)
        work_overlap=volume(value^self.work_cut)
        clearance_overlap=volume(value^self.forbidden_clearance)
        components=len(value.decompose())
        return dict(passed=min(minimum)>=-1e-9 and max(nominal_overlap, work_overlap, clearance_overlap)<=EPS_VOLUME and components==1,
                    minimum_floor_heights_m=minimum, component_count=components,
                    nominal_exit_and_body_overlap_cm3=nominal_overlap*1e6,
                    full_work_overlap_cm3=work_overlap*1e6,
                    padded_exit_overlap_outside_own_contact_cores_cm3=clearance_overlap*1e6,
                    stock_common_bounds_m=[self.low.tolist(),self.high.tolist()],
                    stock_padding_m=self.padding, clearance=self.exit_metadata,
                    exclusions=self.exclusions, material_volume_cm3=volume(value)*1e6,
                    retained_original_material_cm3=volume(value^self.old)*1e6,
                    removed_original_material_in_derived_candidate_cm3=volume(self.old-value)*1e6,
                    added_material_cm3=volume(value-self.old)*1e6,
                    original_step4_mesh_modified=False)
