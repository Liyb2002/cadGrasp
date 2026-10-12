"""Straight-line antipodal jaw candidates from mesh surface ray intersections."""
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def candidates(mesh,initial,limit=300,max_width=.15,depths=(.10,.08,.112),rolls=None,min_width=.015):
    sampled,faces=trimesh.sample.sample_surface(mesh,1800,seed=20260918)
    ids=np.arange(len(mesh.faces))
    if len(ids)>800:
        ids=np.random.default_rng(20260918).choice(ids,800,replace=False,p=mesh.area_faces/mesh.area)
    points=np.concatenate([mesh.triangles_center[ids],sampled])
    face_ids=np.r_[ids,faces]
    normals=mesh.face_normals[face_ids]
    locations,rays,opposite=mesh.ray.intersects_location(points-1e-6*normals,-normals,multiple_hits=True)
    widths=np.linalg.norm(locations-points[rays],axis=1)
    aligned=np.einsum('ij,ij->i',mesh.face_normals[opposite],-normals[rays])>.985
    valid=np.flatnonzero(aligned & (widths>min_width) & (widths<max_width))
    middle=(points[rays]+locations)/2
    score=np.linalg.norm(middle-mesh.center_mass,axis=1)
    seen=set();count=0
    for index in valid[np.argsort(score[valid])]:
        p=trimesh.transform_points(np.array([points[rays[index]],locations[index]]),initial)
        center=p.mean(axis=0);closing=(p[1]-p[0])/widths[index]
        key=tuple(np.round(middle[index]/.004).astype(int))+tuple(np.round(normals[rays[index]]/.08).astype(int))
        if key in seen:continue
        seen.add(key)
        base=np.array([0.,0.,-1.]);base-=closing*(base@closing)
        if np.linalg.norm(base)<.1:continue
        base/=np.linalg.norm(base)
        for roll in ((0,-15,15,-30,30,-45,45,-60,60,-75,75) if rolls is None else rolls):
            approach=Rotation.from_rotvec(closing*np.radians(roll)).apply(base)
            R=np.column_stack([np.cross(closing,approach),closing,approach])
            for depth in depths:
                hand=np.eye(4);hand[:3,:3]=R;hand[:3,3]=center-R@[0,0,depth]
                yield dict(hand=hand,width=float(widths[index]),contacts=p,opening=max_width/2+.005,
                    face_ids=[int(face_ids[rays[index]]),int(opposite[index])],roll_deg=roll,
                    com_distance_m=float(score[index]),candidate_method='normal-ray antipodal pair')
        count+=1
        if count>=limit:return
