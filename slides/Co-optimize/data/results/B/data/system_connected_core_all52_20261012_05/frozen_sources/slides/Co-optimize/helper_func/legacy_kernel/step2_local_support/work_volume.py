"""Step 2 owns visibility-aware bounds on the unbounded process access volume.

Later stages read these saved families. Their operation here is querying whether a
solid intersects them; neither visibility decomposition nor directions are rebuilt.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step1.needs import frame,ContinuousNeeds,OUTPUTS,OBJECTS
from step1.cases import pose_name
from step3_scheculer import contacts as I
from step2_local_support import visibility as V

SIDES=32
CLEARANCE_FRACTION=1e-6
LP_TOLERANCE=1e-9
STAGE='step2_local_support'


class WorkVolume:
    def __init__(self,triangles,normals,face_ids,half_angle_deg,scale,origin=None):
        self.original_triangles=np.asarray(triangles,float)
        self.original_normals=np.asarray(normals,float)
        self.original_face_ids=np.asarray(face_ids,int)
        self.half_angle_deg=float(half_angle_deg);self.scale=float(scale)
        self.origin=np.asarray(origin if origin is not None else self.original_triangles.mean(axis=(0,1)),float)
        if not 0<self.half_angle_deg<90 or self.scale<=0:raise ValueError('Invalid access domain')
        if self.original_triangles.shape!=(len(self.original_face_ids),3,3) or not len(self.original_face_ids):raise ValueError('Expected complete working triangles')
        if not np.isfinite(self.original_triangles).all():raise ValueError('Nonfinite geometry')
        np.testing.assert_allclose(np.linalg.norm(self.original_normals,axis=1),1.,atol=1e-12)
        self.tangent=np.tan(np.deg2rad(self.half_angle_deg))
        angles=(np.arange(SIDES)+.5)*2*np.pi/SIDES
        self.full_slopes=self.tangent/np.cos(np.pi/SIDES)*np.c_[np.cos(angles),np.sin(angles)]
        count=len(self.original_face_ids)
        self._set_cells(self.original_triangles,np.arange(count),np.tile(self.full_slopes,(count,1,1)),
                        np.full(count,SIDES),np.ones(count,bool),np.zeros(count,int))
        self.object_mesh=None;self.ray_offset=0.
        self.visibility=dict(method='No occluding geometry supplied',cell_count=count,
                             certified_visible_cells=count,transition_cells=0,fully_occluded_cells_removed=0)

    def _set_cells(self,triangles,indices,slopes,counts,visible,depth):
        self.triangles=np.asarray(triangles,float).reshape(-1,3,3)
        self.work_indices=np.asarray(indices,int)
        self.face_ids=self.original_face_ids[self.work_indices]
        self.normals=self.original_normals[self.work_indices]
        self.e1,self.e2=frame(self.normals)
        self.slopes=np.asarray(slopes,float).reshape(-1,SIDES,2)
        self.slope_counts=np.asarray(counts,int);self.certified_visible=np.asarray(visible,bool);self.depth=np.asarray(depth,int)
        self.outer=np.zeros((len(self.triangles),SIDES+1,3))
        self.outer[:,-1]=-self.normals
        for i,(polygon,count) in enumerate(zip(self.slopes,self.slope_counts)):
            planes=ConvexHull(polygon[:count]).equations
            self.outer[i,:len(planes)]=planes[:,0,None]*self.e1[i]+planes[:,1,None]*self.e2[i]+planes[:,2,None]*self.normals[i]
        azimuth=np.arange(SIDES)*2*np.pi/SIDES
        lateral=np.cos(azimuth)[None,:,None]*self.e1[:,None]+np.sin(azimuth)[None,:,None]*self.e2[:,None]
        circle_inner=lateral-self.tangent*np.cos(np.pi/SIDES)*self.normals[:,None]
        self.inner=np.concatenate([self.outer,circle_inner],axis=1)
        self.local=(self.triangles-self.origin)/self.scale
        self.support_bounds=np.einsum('fkj,fvj->fkv',self.outer,self.local).max(axis=2)
        self.cache={}

    @classmethod
    def from_domain(cls,domain):
        result=cls(domain.mesh.triangles[domain.work_ids],-domain.normals,domain.work_ids,
                   domain.data['load']['cone_half_deg'],domain.mesh.extents.max(),domain.mesh.bounds.mean(axis=0))
        e1,e2=frame(result.original_normals)
        cells,report=V.build_cells(domain.mesh,result.original_triangles,result.original_normals,e1,e2,
                                  result.scale,domain.ray_offset,result.full_slopes)
        result._set_cells([c[0] for c in cells],[c[1] for c in cells],
            [np.pad(c[2],((0,SIDES-len(c[2])),(0,0))) for c in cells],
            [len(c[2]) for c in cells],[c[3] for c in cells],[c[4] for c in cells])
        result.object_mesh=domain.mesh;result.ray_offset=domain.ray_offset;result.visibility=report
        return result

    def arrays(self):
        return dict(work_triangles_m=self.original_triangles,outward_normals=self.original_normals,
            work_face_ids=self.original_face_ids,normalization_origin_m=self.origin,normalization_scale_m=self.scale,
            cell_triangles_m=self.triangles,cell_work_indices=self.work_indices,cell_slopes=self.slopes,
            cell_slope_counts=self.slope_counts,cell_certified_visible=self.certified_visible,cell_depth=self.depth,
            outer_cone_planes=self.outer,ray_origin_offset_m=self.ray_offset,
            object_vertices_m=np.empty((0,3)) if self.object_mesh is None else self.object_mesh.vertices,
            object_faces=np.empty((0,3),int) if self.object_mesh is None else self.object_mesh.faces)

    def definition(self):
        return dict(equation='V_access = union {pt + s*u: pt in S_work, s >= 0, ||u||=1, u dot n_out(pt) >= cos(alpha), reachable(pt,u)}',
            reachable='No object intersection on the full half-ray from pt + ray_origin_offset*n_out, exactly the Step 1 numerical predicate',
            pt='Every point of every saved working triangle, in world metres',
            u='Outward access direction, opposite the inward push',s='Unbounded distance in metres',
            alpha_deg=self.half_angle_deg,ray_origin_offset_m=self.ray_offset,
            force_magnitude_determines_ray_length=False,self_occluded_directions='Always-occluded source/direction families removed; partially resolved boundaries recorded separately',
            owner_stage=STAGE,depends_on_support_or_insertion=False,
            support_scope='Entire final installed solid',insertion_path_constrained=False,
            boundary_contact_allowed=False,unbounded_rays_used_for_acceptance=True,
            representation='Adaptive continuous source polygons and convex direction cells, with explicit visibility bounds',
            cone_sides=SIDES,maximum_radial_overestimate_fraction=float(1/np.cos(np.pi/SIDES)-1),
            numerical_clearance_m=CLEARANCE_FRACTION*self.scale,lp_tolerance_in_normalized_coordinates=LP_TOLERANCE,
            transition_policy='A hit in an unresolved outer family cannot be accepted as clear; an occluded single witness cannot prove a collision',
            collision_certificate='Actual circular-cone intersection with a Step 1 reachable ray, or a reachable work-surface contact',
            preview_scope='Finite geometry for display only; transition band is recorded, never silently counted as certified visible')

    def export(self,out,needs_path):
        out=Path(out);out.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out/'work_volume.npz',**self.arrays())
        report=dict(complete=True,definition=self.definition(),geometry_file='work_volume.npz',
            work_face_count=len(self.original_face_ids),visibility=self.visibility,
            artifacts={'work_volume.npz':I.sha256(out/'work_volume.npz')},
            provenance=dict(inputs=I.hashes([needs_path]),code=I.hashes([Path(__file__),Path(V.__file__)])))
        I.save(out/'work_volume.json',report);return out/'work_volume.json'

    @classmethod
    def read(cls,path):
        path=Path(path);report=I.check_report(path);a=I.load_npz(path.parent/report['geometry_file'])
        result=cls(a['work_triangles_m'],a['outward_normals'],a['work_face_ids'],report['definition']['alpha_deg'],
                   float(a['normalization_scale_m']),a['normalization_origin_m'])
        result._set_cells(a['cell_triangles_m'],a['cell_work_indices'],a['cell_slopes'],a['cell_slope_counts'],a['cell_certified_visible'],a['cell_depth'])
        result.ray_offset=float(a['ray_origin_offset_m']);result.visibility=report['visibility']
        if len(a['object_faces']):result.object_mesh=trimesh.Trimesh(a['object_vertices_m'],a['object_faces'],process=False)
        np.testing.assert_array_equal(result.outer,a['outer_cone_planes'])
        assert result.definition()==report['definition']
        return result

    def _intersection(self, vertices, equations, face, planes, margin):
        a, b, c = self.local[face]
        edge = np.array([b-a, c-a]).T
        constraints = np.vstack([np.c_[equations[:, :3], np.zeros((len(equations), 2))],
                                 np.c_[planes, -planes@edge], [0., 0., 0., 1., 1.]])
        rhs = np.r_[-equations[:, 3]+margin, planes@a, 1.]
        bounds = list(zip(vertices.min(axis=0)-margin, vertices.max(axis=0)+margin)) + [(0., 1.), (0., 1.)]
        result = linprog(np.zeros(5), A_ub=constraints, b_ub=rhs, bounds=bounds,
                         method='highs', options={'primal_feasibility_tolerance': LP_TOLERANCE,
                                                  'dual_feasibility_tolerance': LP_TOLERANCE})
        if result.status == 2:
            return None
        if not result.success:
            return dict(classification='solver_unresolved', solver_status=int(result.status),
                        solver_message=result.message)
        point = result.x[:3]
        pt = a + edge@result.x[3:]
        return self._witness(point, pt, face, float(np.max(constraints@result.x-rhs)))

    def _witness(self, point, pt, face, residual):
        v = point-pt
        height = float(v@self.normals[face])
        radius = float(np.linalg.norm(v-height*self.normals[face]))
        length = float(np.linalg.norm(v))
        at_surface = length <= 4*LP_TOLERANCE
        witness = dict(point_m=(point*self.scale+self.origin).tolist(),
                    pt_m=(pt*self.scale+self.origin).tolist(), work_face_id=int(self.face_ids[face]),
                    ray_length_m=length*self.scale, at_work_surface=at_surface,
                    angle_from_outward_normal_deg=None if at_surface else float(np.degrees(np.arccos(np.clip(height/length,-1,1)))),
                    outward_height_m=height*self.scale, transverse_radius_m=radius*self.scale,
                    circular_cone_residual_m=(radius-self.tangent*height)*self.scale,
                    maximum_constraint_residual=residual,
                    classification='outer_envelope_intersection')
        witness['visibility_cell_index']=int(face)
        witness['visibility_cell_certified']=bool(self.certified_visible[face])
        if at_surface:
            slopes=self.slopes[face,:self.slope_counts[face]]
            trials=np.vstack([slopes.mean(axis=0),.5*(slopes+slopes.mean(axis=0))])
            vectors=self.normals[face]+trials[:,:1]*self.e1[face]+trials[:,1:]*self.e2[face]
            vectors/=np.linalg.norm(vectors,axis=1)[:,None]
        else:vectors=(v/length)[None]
        allowed=(vectors@self.normals[face])>=np.cos(np.deg2rad(self.half_angle_deg))-1e-10
        vectors=vectors[allowed]
        clear=np.ones(len(vectors),bool)
        if self.object_mesh is not None and len(vectors):
            origin=np.asarray(witness['pt_m'])+self.ray_offset*self.normals[face]
            clear=~self.object_mesh.ray.intersects_any(np.repeat(origin[None],len(vectors),axis=0),vectors)
        found=np.flatnonzero(clear)
        witness['reachable_ray_witness_found']=bool(len(found))
        witness['step1_self_occlusion_ray_clear']=bool(len(found))
        witness['access_direction']=vectors[found[0]].tolist() if len(found) else None
        witness['self_occluded_ray_is_a_collision_certificate']=False
        return witness

    def check_piece(self,piece):
        key=hashlib.sha256(np.asarray(piece.vertices,np.float64).tobytes()).hexdigest()
        if key in self.cache:return self.cache[key]
        vertices=(np.asarray(piece.vertices)-self.origin)/self.scale
        minima=np.einsum('fkj,vj->fkv',self.outer,vertices).min(axis=2)
        allowance=CLEARANCE_FRACTION*np.linalg.norm(self.outer,axis=2)
        candidates=np.flatnonzero(~np.any(minima>self.support_bounds+allowance,axis=1))
        # Prefer certified-visible families for an immediate valid collision witness.
        candidates=sorted(candidates,key=lambda i:not self.certified_visible[i])
        result=dict(passed=True,broad_phase_candidate_faces=len(candidates),lp_faces_checked=0,classification='clear',witness=None)
        pending=None
        if candidates:
            equations=ConvexHull(vertices).equations
            for face in candidates:
                result['lp_faces_checked']+=1
                outer=self._intersection(vertices,equations,face,self.outer[face],CLEARANCE_FRACTION)
                if outer is None:continue
                inner=self._intersection(vertices,equations,face,self.inner[face],0.)
                if inner is not None and inner.get('reachable_ray_witness_found'):
                    inner['classification']='circular_cone_intersection'
                    result.update(passed=False,classification=inner['classification'],witness=inner)
                    break
                if pending is None:
                    classification='solver_unresolved' if outer.get('classification')=='solver_unresolved' else 'visibility_or_envelope_unresolved'
                    pending=dict(passed=False,classification=classification,witness=inner or outer,
                                 reason='Outer family intersects, but no actual reachable circular-cone intersection was established')
            else:
                if pending is not None:result.update(pending)
        self.cache[key]=result
        return result

    def check_surface(self, triangles, source_faces=None):
        """Check the actual zero-thickness interface, including triangle interiors.

        Four barycentric unknowns describe the contact point and work point.
        This separates contact conflicts from collisions of an arbitrary backing.
        Boundary contact still rejects, consistently with check_piece.
        """
        checked = 0
        pending = None
        for index, triangle in enumerate(np.asarray(triangles, float)):
            vertices = (triangle-self.origin)/self.scale
            minima = np.einsum('fkj,vj->fkv', self.outer, vertices).min(axis=2)
            allowance = CLEARANCE_FRACTION*np.linalg.norm(self.outer, axis=2)
            candidates = np.flatnonzero(~np.any(minima > self.support_bounds+allowance, axis=1))
            t = vertices[0]
            edge = (vertices[1:]-t).T
            for face in candidates:
                checked += 1
                a, b, c = self.local[face]
                work_edge = np.array([b-a, c-a]).T
                witness = None
                for planes, margin in [(self.outer[face], CLEARANCE_FRACTION), (self.inner[face], 0.)]:
                    matrix = np.vstack([np.c_[planes@edge, -planes@work_edge],
                                        [1., 1., 0., 0.], [0., 0., 1., 1.]])
                    rhs = np.r_[planes@(a-t)+margin*np.linalg.norm(planes, axis=1), 1., 1.]
                    # Prefer a positive ray to an arbitrary apex witness when available.
                    objective = np.r_[-self.normals[face]@edge, self.normals[face]@work_edge]
                    result = linprog(objective, A_ub=matrix, b_ub=rhs, bounds=[(0., 1.)]*4,
                        method='highs', options={'primal_feasibility_tolerance': LP_TOLERANCE,
                                                 'dual_feasibility_tolerance': LP_TOLERANCE})
                    if result.status == 2:
                        break
                    if not result.success:
                        if witness is None:
                            witness = dict(classification='solver_unresolved', solver_status=int(result.status),
                                           solver_message=result.message)
                        break
                    point = t+edge@result.x[:2]
                    pt = a+work_edge@result.x[2:]
                    witness = self._witness(point, pt, face, float(np.max(matrix@result.x-rhs)))
                    if margin == 0.:
                        witness['classification'] = 'circular_cone_intersection'
                if witness is not None:
                    record=dict(passed=False,classification=witness['classification'],witness=witness,
                        contact_triangle_index=index,
                        contact_source_face=None if source_faces is None else int(source_faces[index]),
                        lp_pairs_checked=checked,scope='Actual contact triangles against saved accessible-ray families')
                    if witness['classification']=='circular_cone_intersection' and witness.get('reachable_ray_witness_found'):
                        return record
                    if pending is None:
                        record['classification']='solver_unresolved' if witness['classification']=='solver_unresolved' else 'visibility_or_envelope_unresolved'
                        pending=record
        if pending is not None:return {**pending,'lp_pairs_checked':checked}
        return dict(passed=True, classification='clear', witness=None, lp_pairs_checked=checked,
                    scope='Actual contact triangles, with boundary contact forbidden')

    def check_parts(self,parts,labels=None):
        pending=None
        for index,part in enumerate(parts):
            result=self.check_piece(part)
            if not result['passed']:
                record=dict(part_index=index,part_label=str(labels[index]) if labels is not None else None,**result)
                if result['classification']=='circular_cone_intersection':return record
                if pending is None:pending=record
        return pending or dict(passed=True,part_count=len(parts),classification='clear',witness=None)


def build(name):
    started=time.monotonic();root=OUTPUTS/name/pose_name();source=root/'step_1_needs/needs.json'
    domain=ContinuousNeeds.read(source);work=WorkVolume.from_domain(domain)
    out=root/STAGE;path=work.export(out,source)
    print(name,pose_name(),'Step 2 work volume:',len(work.triangles),'families;',
          work.visibility['transition_cells'],'transition cells;',round(time.monotonic()-started,2),'seconds',flush=True)
    return path


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('objects',nargs='*')
    for name in parser.parse_args().objects or OBJECTS:build(name)
