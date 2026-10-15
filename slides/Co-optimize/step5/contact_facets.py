"""Expose necessary original planar contact normals lost by a mesh union."""
import numpy as np
import manifold3d as md
from periphery import PlaneForceFilter
from deterministic_base import solid,unpack,union,SCALE
from whole_search.common import transform_mesh


def restore(checker,reference,candidate,radius=.00003,depth=.00005):
    before=PlaneForceFilter(checker,reference).contacts
    after=PlaneForceFilter(checker,candidate).contacts
    records=[];cuts=[];pads=[]
    for k,(triangles,sources)in before.items():
        posed=transform_mesh(checker.mesh,checker.layout.placements[k])
        for face in np.setdiff1d(sources,after[k][1]):
            selected=triangles[sources==face]
            area=float(np.linalg.norm(np.cross(selected[:,1]-selected[:,0],selected[:,2]-selected[:,0]),axis=1).sum()/2)
            if area>1e-10:raise ValueError('A substantial contact face needs explicit geometric recovery')
            normal=posed.face_normals[face];origin=posed.triangles[face,0]
            center=selected.reshape(-1,3).mean(0);center-=normal*((center-origin)@normal)
            axis=np.eye(3)[int(np.argmin(abs(normal)))];u=np.cross(normal,axis);u/=np.linalg.norm(u);v=np.cross(normal,u)
            def prism(lo,hi):
                vertices=np.array([center+x*u+y*v+z*normal for x in [-radius,radius]
                    for y in [-radius,radius]for z in [lo,hi]])
                return md.Manifold.hull_points(np.ascontiguousarray(vertices/SCALE))
            cuts.append(prism(-depth,0.));pads.append(prism(0.,depth))
            records.append(dict(pose_index=int(k),source_face=int(face),
                original_contact_area_m2=area,center_m=center.tolist(),normal_m=normal.tolist(),
                radius_m=radius,depth_m=depth))
    if not records:return candidate,records
    # Preserve the new exact functional planes: no post-Boolean simplification.
    return unpack((solid(candidate)-union(cuts))+union(pads)),records
