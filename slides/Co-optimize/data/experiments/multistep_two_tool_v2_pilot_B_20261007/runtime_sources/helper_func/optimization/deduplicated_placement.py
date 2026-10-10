"""Sample direction changes and floor-tangent pose translations together.

Cheap contact-lock screening never constructs solids. Exact finalists rebuild
translated seeds and complete exits, with current contact-core clearance policy.
"""
import time
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from trimesh.ray.ray_pyembree import RayMeshIntersector
from co_common import (HERE,ROOT,state,save,union,material_volume,contact_boundary,
    supply,transform_points,wrap_offsets,provenance,code_sources,np,trimesh,G,S,D,F,U,J)
from exit_clearance import CONTACT_DEPTH_M,_PRISM_FACES
from physics_guided_padded_sweep import ConservativeExitClearance
from physics_guided_geometry import tangent_frames,retract
from physics_guided_cone_iterative import cone_projection
from contact_recovery import preserves_loads
from worst_wrench_descent import farthest_load


def shifted(solid,offset):return solid.translate(np.asarray(offset)/S.SCALE)

def fixture_transform(T,offset):
    result=T.copy();result[:3,3]-=T[:3,:3]@offset
    return result


def colocated_allowed(normals,allowed,owner,directions,offsets):
    coincident=np.linalg.norm(offsets-offsets[owner],axis=1)<1e-10
    blocked=np.max(normals[allowed]@directions[coincident].T,axis=1)>1e-9
    return allowed[~blocked]


def separation_layout(normals,spacing):
    frames=tangent_frames(normals);offsets=np.zeros_like(normals)
    for k in range(1,len(normals)):
        for attempt in range(1000):
            radius=spacing*(1+attempt//24)
            angle=2*np.pi*((attempt%24)/24+k*.38196601125)
            candidate=frames[k]@(radius*np.array([np.cos(angle),np.sin(angle)]))
            if np.min(np.linalg.norm(offsets[:k]-candidate,axis=1))>=spacing:
                offsets[k]=candidate;break
        else:raise RuntimeError('separation layout exhausted')
    return offsets



from placement_sampling import colocated_allowed

class DeduplicatedPlacement:
    def exact(self,directions,offsets):
        if np.min(np.sum(directions*self.normals,axis=1))<-1e-12:raise ValueError('illegal exit direction')
        if np.max(np.abs(np.sum(offsets*self.normals,axis=1)))>1e-10:raise ValueError('translation leaves native floor plane')
        self.exact_calls+=1;began=time.monotonic()
        seed=union([shifted(s,o) for s,o in zip(self.seed_parts,offsets)])
        work=union([shifted(self.work_obstacle,o) for o in offsets])
        seed=seed-work
        nominal_sweeps=[shifted(self.clearance.sweep(self.length*d,padded=False),o) for d,o in zip(directions,offsets)]
        padded_sweeps=[shifted(self.clearance.sweep(self.length*d,padded=True),o) for d,o in zip(directions,offsets)]
        nominal_cut=union(nominal_sweeps);padded_cut=union(padded_sweeps);nominal=seed-nominal_cut
        nominal_mesh=S.unpack(nominal);meshes=[];allowed=[];core_groups=[]
        for k,o in enumerate(offsets):
            mesh=self.mesh.copy();mesh.apply_translation(o);meshes.append(mesh)
            ids=colocated_allowed(self.mesh.face_normals,self.allowed,k,directions,offsets);allowed.append(ids)
            if k and np.any(np.linalg.norm(offsets[:k]-o,axis=1)<1e-10):continue
            owner_parts=[]
            tris,src=contact_boundary(mesh,nominal_mesh,ids)
            for tri,source in zip(tris,src):
                prism=trimesh.Trimesh(np.vstack([tri,tri+CONTACT_DEPTH_M*mesh.face_normals[source]]),_PRISM_FACES,process=False)
                if prism.volume<0:prism.invert()
                part=S.solid(prism)
                if material_volume(part^nominal)>1e-16:owner_parts.append(part)
            if owner_parts:core_groups.append(union(owner_parts)^nominal)
        core=union(core_groups)
        remaining=((seed-padded_cut)+core)-nominal_cut
        remaining=remaining-(padded_cut-core)
        # Remove any Boolean residual that actually violates the core exception.
        # This changes material conservatively rather than relaxing tolerances.
        for _ in range(1):
            residual=(remaining-core)^padded_cut
            if material_volume(residual)<1e-10:break
            remaining=remaining-residual
        diagnostics=dict(nominal_overlap_m3=max(material_volume(remaining^s) for s in nominal_sweeps),
            padded_overlap_outside_contact_cores_m3=material_volume((remaining-core)^padded_cut),
            working_region_overlap_m3=material_volume(remaining^work),
            partition_error_m3=abs(material_volume(seed)-material_volume(remaining)-material_volume(seed-remaining)))
        if max(diagnostics.values())>=1e-10:raise RuntimeError('exit/clearance/work/partition unresolved: '+str(diagnostics))
        parts=[S.unpack(p) for p in remaining.decompose() if material_volume(p)>=1e-12]
        if not parts:raise RuntimeError('no positive material')
        masks=[];supplies=[];patches=[];transforms=[]
        for k,((task,T),o) in enumerate(zip(self.states,offsets)):
            triangles=[];sources=[]
            for part in parts:
                tri,src=contact_boundary(meshes[k],part,allowed[k]);triangles.append(tri);sources.append(src)
            tri=np.concatenate(triangles);src=np.concatenate(sources);Tf=fixture_transform(T,o);transforms.append(Tf)
            full=supply(task,Tf,tri,src);mask,info=J.classify(full,task.targets)
            masks.append(mask);supplies.append(full);patches.append(dict(pose=self.group['poses'][k],triangles=len(tri),
                area_m2=float(.5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1).sum())))
        endpoint=max(material_volume(shifted(S.solid(self.mesh),o+self.length*d)^seed) for o,d in zip(offsets,directions))
        if endpoint>=1e-10:raise RuntimeError('complete exit endpoint unresolved')
        result=dict(serial=self.exact_calls,directions=directions.copy(),offsets=offsets.copy(),masks=masks,supplies=supplies,
            counts=[int(m.sum()) for m in masks],remaining=remaining,diagnostics=diagnostics,endpoint_overlap_m3=endpoint,
            contact_patches=patches,transforms=transforms,seconds=time.monotonic()-began)
        print('PLACEMENT EXACT',self.exact_calls,result['counts'],'offset max',float(np.linalg.norm(offsets,axis=1).max()),flush=True)
        return result

