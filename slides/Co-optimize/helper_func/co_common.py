"""Shared immutable inputs, exact rigid registration and surface wrapping."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
import os,sys,json
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parent.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE.parent/'baseline_algo'))
import numpy as np
import trimesh
from step3_scheculer import contacts as I,passive_support as U,floor_support as FLOOR
from current_task import current_task
from step3_scheculer.pair_scoring import J,C
from step2_local_support import geometry as G
from step4_connect_support import boxed_support as F,build_coupled_saddle as S,deterministic_space as D,working_surface as WORK


def save(path,value):I.save(path,value)

def state(name,pose):
    task=current_task(name,pose)
    transform=np.asarray(task.domain.data['frame']['T_world_mesh'],float)
    mesh=task.domain.mesh.copy();mesh.apply_transform(np.linalg.inv(transform))
    return task,transform,mesh

def transform_points(points,T):return np.asarray(points)@T[:3,:3].T+T[:3,3]

def provenance(inputs,code):return dict(inputs=I.hashes(inputs),code=I.hashes(code))

def union(parts):
    if not parts:return F.md.Manifold()
    result=F.union(parts)
    if result.status()!=F.md.Error.NoError:raise RuntimeError('Unresolved wrap Boolean')
    return result

def material_volume(solid):
    if solid.status()!=F.md.Error.NoError:raise RuntimeError('Unresolved material Boolean')
    return abs(float(solid.volume()))*S.SCALE**3

def wrap_offsets(mesh,thickness):
    offsets,valid=G.vertex_offsets(mesh,thickness)
    if not valid.all():raise RuntimeError('Full surface offset unresolved; cannot label this as force failure')
    # The shared head kernel guarantees normal depth by arbitrarily long
    # vertex offsets at near-tangent sharp corners. A full shell instead uses
    # bounded displacement: preserve its outward directions, cap length.
    lengths=np.linalg.norm(offsets,axis=1)
    offsets*=np.minimum(1.,thickness/lengths)[:,None]
    if np.any(np.einsum('fvc,fc->fv',offsets[mesh.faces],mesh.face_normals)<=0):raise RuntimeError('Non-outward wrap offset')
    return offsets

def contact_boundary(mesh,shell,allowed_faces):
    """Intersect actual inner shell triangles with each original source triangle."""
    triangles=[];sources=[];tol=1e-10
    tree=shell.triangles_tree;all_triangles=shell.triangles
    for face in allowed_faces:
        original=mesh.triangles[face];normal=mesh.face_normals[face]
        candidates=list(tree.intersection(np.r_[original.min(0)-tol,original.max(0)+tol]))
        for index in candidates:
            poly=all_triangles[index].copy()
            if np.max(np.abs((poly-original[0])@normal))>tol:continue
            for a,b in zip(original,np.roll(original,-1,axis=0)):
                outward=np.cross(b-a,normal);outward/=np.linalg.norm(outward)
                poly=G.clip_plane(poly,np.r_[outward,-outward@a])
                if len(poly)<3:break
            if len(poly)<3:continue
            if np.cross(poly[1]-poly[0],poly[2]-poly[0])@normal<0:poly=poly[::-1]
            for k in range(1,len(poly)-1):
                tri=np.array([poly[0],poly[k],poly[k+1]])
                if np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0]))>1e-18:
                    triangles.append(tri);sources.append(face)
    return np.asarray(triangles,float).reshape(-1,3,3),np.asarray(sources,int)

def supply(task,T,triangles,sources):
    points=transform_points(triangles.reshape(-1,3),T)
    normals=np.repeat(-task.domain.mesh.face_normals[sources],3,axis=0)
    raw=np.c_[normals,np.cross(points-task.domain.com,normals)]
    floor=U.floor(FLOOR.columns(task.floor,task.domain.com),task.scale)
    return I.merge_columns(floor,U.heads(raw,task.scale))

def code_sources():
    return [HERE/'helper_func/co_common.py',HERE/'helper_func/current_task.py',HERE/'step3.1/step31.py',HERE/'step3.2/step32.py',HERE/'helper_func/run_all.py',Path(G.__file__),Path(J.__file__),Path(C.__file__),Path(C.W.__file__),Path(U.__file__),Path(FLOOR.__file__),Path(WORK.__file__),Path(S.__file__),Path(F.__file__)]
