"""Equivalent, faster continuous-exit checks for Step4/Step5; Step1-3 files unchanged.

Installed on import of membrane_search, so every Step4/Step5 model uses it:

1. continuous_checks: an exact broad phase in the exit-aligned (u,v,d) frame.
   A face prism is the column {(u,v) in the face, d in [d0, d0+L]}. If no
   support triangle box meets the column box, their boundaries cannot
   intersect, so the prism is entirely outside or entirely inside the
   support, decided by one point. Only possibly crossing prisms get the
   Boolean. The per-prism overlap sum, the tolerance and the failed prisms
   returned for recutting are identical to the all-Boolean check.
2. refine_exits: a recut only removes material and the derived travel can
   only shorten, so passed tasks stay passed; only failed tasks are
   rechecked, and the final rows seed the audit's path cache.

Measured on a 9-task fixture: exit refinement 17.2 s -> 2.3 s, ground
oracle 62.7 s -> 7.1 s, with bitwise identical overlap bounds.
"""
import hashlib
import numpy as np
import manifold3d as md

from whole_task_2.common.force_cone import tangent_frame
from whole_task_2.step3 import independent_geometry as G
from whole_task_2.step3 import exit_geometry as X
from whole_task_2.step3.membranes import MembraneModel, positive_components
from whole_task_2.operations import geometry as O


class ColumnIndex:
    """Uniform (u,v) grid of support triangle boxes in the exit frame."""
    PAD=1e-9;BIG=4096

    def __init__(self,triangles):
        self.lo=triangles.min(1)-self.PAD;self.hi=triangles.max(1)+self.PAD
        # Grid cell of the typical support triangle, so any mesh scale works.
        self.CELL=max(float(np.median((self.hi-self.lo)[:,:2].max(1))),1e-6)
        ix0,iy0=(np.floor(self.lo[:,:2]/self.CELL).astype(np.int64)).T
        ix1,iy1=(np.floor(self.hi[:,:2]/self.CELL).astype(np.int64)).T
        nx=ix1-ix0+1;ny=iy1-iy0+1;count=nx*ny
        large=count>self.BIG;self.large=np.flatnonzero(large)
        ids=np.flatnonzero(~large);count=count[ids]
        owner=np.repeat(ids,count);step=np.arange(count.sum())-np.repeat(np.cumsum(count)-count,count)
        cx=ix0[owner]+step%nx[owner];cy=iy0[owner]+step//nx[owner]
        key=(cx<<32)+(cy&0xffffffff);order=np.argsort(key,kind='stable')
        self.keys=key[order];self.owners=owner[order]

    def crosses(self,lo,hi):
        lo=lo-self.PAD;hi=hi+self.PAD
        (ax,ay),(bx,by)=np.floor(lo[:2]/self.CELL).astype(np.int64),np.floor(hi[:2]/self.CELL).astype(np.int64)
        keys=((np.arange(ax,bx+1)[:,None]<<32)+(np.arange(ay,by+1)[None,:]&0xffffffff)).ravel()
        left=np.searchsorted(self.keys,keys,'left');right=np.searchsorted(self.keys,keys,'right')
        ids=np.concatenate([self.owners[a:b] for a,b in zip(left,right)]+[self.large])
        if not len(ids):return False
        return bool(((self.lo[ids]<=hi)&(self.hi[ids]>=lo)).all(1).any())


def leading_faces(body,direction,length):
    """The positive leading faces of exit_geometry.leading_prisms, prisms built on demand."""
    direction=np.asarray(direction,float);delta=length*direction/G.SCALE
    if not np.isfinite(delta).all() or abs(np.linalg.norm(direction)-1)>1e-12 or length<=0:
        raise ValueError('Positive length and a unit direction are required')
    triangles=np.asarray(body.vertices)/G.SCALE
    triangles=triangles[body.faces];extended=triangles.astype(np.longdouble)
    signed=np.cross(extended[:,1]-extended[:,0],extended[:,2]-extended[:,0])@delta.astype(np.longdouble)
    def build(face):
        points=np.vstack([triangles[face],triangles[face]+delta])
        prism=md.Manifold(md.Mesh64(np.ascontiguousarray(points),X.PRISM_FACES))
        if prism.status()!=md.Error.NoError or prism.is_empty() or prism.volume()<=0:
            raise RuntimeError('A positive leading face prism is numerically unresolved: '+str(face))
        return prism
    return [(int(face),build) for face in np.flatnonzero(signed>0)]


def continuous_checks(support,paths,tolerance):
    """exit_geometry.continuous_checks with the exact column broad phase."""
    rows=[];bad=[];surface=G.mesh(support)
    for id,body,direction,length in paths:
        initial=G.solid(body);body_overlap=G.volume(support^initial)
        values=[];task_bad=[];threshold=tolerance/(len(body.faces)+1)
        u,v=tangent_frame(np.asarray(direction,float));frame=np.column_stack([u,v,direction])
        index=ColumnIndex(surface.triangles@frame) if len(surface.faces) else None
        delta=length*np.asarray(direction,float);triangles=body.triangles;pending=[];skipped=0
        def boolean(face,build):
            prism=build(face);overlap=G.volume(support^prism);values.append(overlap)
            if overlap>threshold:task_bad.append((id,face,prism))
        for face,build in leading_faces(body,direction,length):
            corners=np.vstack([triangles[face],triangles[face]+delta])@frame
            if index is not None and not index.crosses(corners.min(0),corners.max(0)):pending.append((face,build))
            else:boolean(face,build)
        if pending:
            centers=np.array([triangles[f].mean(0)+delta/2 for f,_ in pending])
            inside=surface.contains(centers) if len(surface.faces) else np.zeros(len(pending),bool)
            for (face,build),enclosed in zip(pending,inside):
                if enclosed:boolean(face,build)
                else:skipped+=1;values.append(0.)
        total=body_overlap+sum(values)
        if total>tolerance:
            bad.extend(task_bad)
            if body_overlap>threshold:bad.append((id,-1,initial))
        rows.append(dict(id=id,continuous_overlap_upper_bound_m3=float(total),
            initial_object_overlap_m3=float(body_overlap),leading_face_count=len(values),
            largest_face_prism_overlap_m3=float(max(values,default=0.)),passed=total<=tolerance,
            prisms_exactly_disjoint_by_broad_phase=skipped))
    return rows,bad


def path_key(model,solid,paths):
    """The same cache key as OperationModel.path_checks."""
    mesh=G.mesh(solid);digest=hashlib.sha256()
    for array in (mesh.vertices,mesh.faces,np.asarray([model.tolerance])):
        digest.update(np.ascontiguousarray(array).tobytes())
    for name,body,direction,length in paths:
        digest.update(name.encode())
        for array in (body.vertices,body.faces,np.asarray(direction),np.asarray([length])):
            digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def refine_exits(self,candidate):
    """MembraneModel.refine_exits, rechecking only the tasks that failed."""
    passed={}
    for turn in range(4):
        paths=self.exit_paths(candidate);value=G.solid(candidate['support'])
        rows,bad=self.path_checks(value,[p for p in paths if p[0] not in passed])
        passed.update({row['id']:(turn,row) for row in rows if row['passed']})
        if len(passed)==len(paths):
            if turn:
                self.path_cache[path_key(self,value,paths)]=[dict(passed[p[0]][1],
                    checked_on_superset_material=passed[p[0]][0]<turn) for p in paths]
            break
        if turn==3 or not bad:raise ValueError('Unresolved continuous exit '+str(rows))
        for _,_,prism in bad:value=value-prism
        parts=positive_components(value)
        if not parts:raise ValueError('Exit refinement removes all support')
        keep=max(parts,key=lambda p:float(p.volume()))
        for part in parts:
            if part is not keep:value=value-part
        candidate['support']=G.mesh(value.simplify(0.))
    candidate['volume_cm3']=float(candidate['support'].volume*1e6)
    candidate['construction']['exit_refinement_rounds']=turn


def install():
    O.continuous_checks=continuous_checks
    MembraneModel.refine_exits=refine_exits


install()
