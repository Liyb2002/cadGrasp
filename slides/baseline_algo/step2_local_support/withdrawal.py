"""Shared finite 3-D withdrawal catalogue and continuous whole-head sweeps.

Direction IDs refer to one immutable catalogue per pose. Clear IDs certify
individual full translation rays, never an angular neighbourhood or all S².
"""
from step1.needs import COORD
from pathlib import Path
import manifold3d as manifold
import numpy as np
from step2_local_support import insertion as H
from step3_scheculer import contacts as I

MODE = 'common_rigid_withdrawal_3d'
NORMAL_TOL = 1e-10
VOLUME_TOL = 1e-11
MOTION = dict(direction='support withdrawal +d; insertion reverses the same line',
              frame='Step1 world coordinates', rotation_allowed=False,
              floor_checked=True, connectors_checked=False,
              representation='finite shared 3-D direction IDs; no continuous-sphere completeness claim')
signature = H.signature


def code_hashes():
    return {**H.code_hashes(), **I.hashes([Path(__file__)])}


def normalize(ids=()):
    return dict(ids=sorted(set(map(int, ids))))


def intersect(a, b):
    return normalize(set(a['ids']) & set(b['ids']))


def nonempty(value):
    return bool(value['ids'])


def common(records, initial):
    result = initial
    for row in records:
        result = intersect(result, row['certified_directions'])
    return result


def preferred(mesh, work_ids):
    v = -(mesh.face_normals[work_ids]*mesh.area_faces[work_ids, None]).sum(axis=0)
    return v/np.linalg.norm(v) if np.linalg.norm(v) > mesh.area*1e-12 else None


def make_catalogue(mesh, work_ids, contacts):
    vectors = []
    def add(v):
        v = np.asarray(v, float).copy(); length = np.linalg.norm(v)
        if length < 1e-10: return
        v /= length; v[np.abs(v)<1e-12] = 0.; v /= np.linalg.norm(v)
        if all(np.linalg.norm(v-q)>1e-7 for q in vectors): vectors.append(v)
    back = preferred(mesh, work_ids)
    if back is not None:
        add(back); add(COORD.lift_floor(COORD.floor(back))); add(-back)
    for elevation in (-80,-60,-40,-20,0,20,40,60,80):
        e = np.deg2rad(elevation)
        for azimuth in range(0,360,15):
            a = np.deg2rad(azimuth); add([np.cos(e)*np.cos(a),np.sin(e),np.cos(e)*np.sin(a)])
    add([0,1,0]); add([0,-1,0])
    # Geometry-specific seeds preserve useful tangent directions on planar parts.
    for c in contacts:
        n = np.average(mesh.face_normals[c['source_faces']], axis=0, weights=c['triangle_areas_m2'])
        add(n); add(COORD.lift_floor(COORD.floor(n))); add(-np.cross(n,[0,1,0])); add(-np.cross([0,1,0],n))
    array = np.asarray(vectors)
    floor = array[:,1] < -NORMAL_TOL
    work = array@back < -NORMAL_TOL if back is not None else np.zeros(len(array),bool)
    allowed = np.flatnonzero(~floor & ~work)
    return dict(vectors=array.tolist(), global_allowed_directions=normalize(allowed),
        floor_locked_ids=np.flatnonzero(floor).tolist(), work_face_locked_ids=np.flatnonzero(work).tolist(),
        preferred_withdrawal_direction=back.tolist() if back is not None else None,
        work_face_rule='Exclude outward hemisphere of area-weighted work-face normal; tangent allowed',
        work_face_normal_degenerate=back is None,
        resolution='15 degree azimuth, 20 degree elevation, poles and contact-normal/tangent seeds',
        finite_direction_set=True, entire_sphere_infeasibility_claimed=False)


def representative(value, catalogue):
    if not nonempty(value): return None
    ids = value['ids']; vectors = np.asarray(catalogue['vectors'])
    back = catalogue['preferred_withdrawal_direction']
    index = max(ids,key=lambda i: float(vectors[i]@back)) if back is not None else ids[0]
    return dict(direction_id=index, vector=vectors[index].tolist())


def solid(mesh, origin, scale):
    result = manifold.Manifold(manifold.Mesh64(np.asarray((mesh.vertices-origin)/scale,np.float64),
                                               np.asarray(mesh.faces,np.uint64)))
    if result.status() != manifold.Error.NoError: raise ValueError(str(result.status()))
    return result


class Analyzer(H.Analyzer):
    def __init__(self, mesh, depth, catalogue):
        super().__init__(mesh, depth)
        self.catalogue = catalogue; self.vectors = np.asarray(catalogue['vectors'])
        self.scale = float(mesh.extents.max()); self.origin = mesh.bounds.mean(axis=0)
        self.obstacle = solid(mesh,self.origin,self.scale)
        self.clearance = H.G.Clearance(mesh, self.scale*1e-12)

    def test(self, heads, direction):
        """Convex head cells swept continuously until an AABB separates forever."""
        d = np.asarray(direction)
        if d[1] < -NORMAL_TOL: return dict(clear=False,reason='floor_direction')
        points = np.vstack([h.vertices for h in heads])
        if points[:,1].min() < -self.scale*1e-10: return dict(clear=False,reason='installed_floor_collision')
        distances=[]
        for axis,v in enumerate(d):
            if v>1e-10: distances.append((self.mesh.bounds[1,axis]+.035*self.scale-points[:,axis].min())/v)
            elif v< -1e-10: distances.append((points[:,axis].max()-self.mesh.bounds[0,axis]+.035*self.scale)/-v)
        length=max(.01*self.scale,min(distances))
        for k,h in enumerate(heads):
            points=np.vstack([h.vertices,h.vertices+length*d])
            # A separating coordinate plane certifies the complete ray segment.
            if np.any(points.min(axis=0)>=self.mesh.bounds[1]) or np.any(points.max(axis=0)<=self.mesh.bounds[0]):
                continue
            # Every object boundary triangle is clipped against the full convex
            # sweep. The segment reaches a separated endpoint, so an intersecting
            # closed solid cannot be hidden by full containment without a boundary.
            # This geometric tolerance is much tighter than the Boolean volume
            # allowance. Near-boundary/unresolved pairs still use float64 solids.
            if self.clearance.obstruction(points)<0:
                continue
            swept=H.engine.hull_mesh(np.vstack([h.vertices,h.vertices+length*d]))
            try: volume=abs(float((self.obstacle ^ solid(swept,self.origin,self.scale)).volume()))*self.scale**3
            except ValueError: return dict(clear=False,reason='unresolved_boolean',head_cell=k,length_m=length)
            if volume>VOLUME_TOL*self.scale**3:
                return dict(clear=False,reason='swept_object_collision',head_cell=k,length_m=length,intersection_volume_m3=volume)
        return dict(clear=True,reason='continuous_head_ray_clear',length_m=length)

    def analyze(self, contact, allowed=None, stop_after_first=False):
        ids=(allowed or self.catalogue['global_allowed_directions'])['ids']
        normals=np.unique(self.mesh.face_normals[contact['source_faces']],axis=0)
        heads=None; checks=[]; clear=[]; locked=[]; unresolved=[]
        for i in ids:
            if np.min(normals@self.vectors[i]) < -NORMAL_TOL:
                row=dict(clear=False,reason='contact_normal_blocks_withdrawal')
            else:
                if heads is None: heads=self.heads(contact)
                row=self.test(heads,self.vectors[i])
            checks.append(dict(direction_id=i,**row))
            (clear if row['clear'] else unresolved if row['reason']=='unresolved_boolean' else locked).append(i)
            if clear and stop_after_first: break
        directions=normalize(clear)
        return dict(candidate_id=contact['candidate_id'],candidate_index=int(contact['candidate_index']),
            geometry_signature=signature(contact,self.depth),radius_m=float(contact['radius_m']),
            center_m=contact['center_m'].tolist(),area_m2=I.area([contact]),
            certified_directions=directions,locked_direction_ids=locked,unresolved_direction_ids=unresolved,
            local_constraint_normals=normals.tolist(),local_rule='n_out dot withdrawal >= 0 for every contact face',
            has_certified_direction=bool(clear),representative=representative(directions,self.catalogue),
            analysis=dict(checks=checks,length_m=max((r.get('length_m',0.) for r in checks),default=0.),
                          full_translation_sweeps=True,angular_neighbourhoods_certified=False,
                          surface_clip_tolerance_m=self.scale*1e-12,volume_tolerance_m3=VOLUME_TOL*self.scale**3,
                          acceleration='separating bounds / full boundary clipping before float64 Boolean'),
            global_locks_in_shared_catalogue=True)
