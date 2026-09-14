"""Small exact contact patches shared by geometry regression tests."""
import numpy as np
from step2_local_support.surface import areas


def contacts(mesh):
    ids = np.flatnonzero(mesh.face_normals[:, 0] < -.9)
    result = []
    for i, face in enumerate(ids):
        tri = mesh.triangles[face]; center = tri.mean(axis=0)
        tri = (center+.3*(tri-center))[None]
        result.append(dict(candidate_id=f'C{i}', candidate_index=i, center_m=center,
            center_face=int(face), radius_m=float(np.linalg.norm(tri-center, axis=2).max()),
            source_faces=np.array([face]), triangles_m=tri, triangle_areas_m2=areas(tri)))
    return result

def contact(mesh,normal,index):
    face=int(np.argmax(mesh.face_normals@normal));tri=mesh.triangles[face];center=tri.mean(axis=0)
    tri=(center+.3*(tri-center))[None]
    return dict(candidate_id=f'C{index}',candidate_index=index,center_m=center,center_face=face,
        radius_m=float(np.linalg.norm(tri-center,axis=2).max()),source_faces=np.array([face]),
        triangles_m=tri,triangle_areas_m2=areas(tri))
