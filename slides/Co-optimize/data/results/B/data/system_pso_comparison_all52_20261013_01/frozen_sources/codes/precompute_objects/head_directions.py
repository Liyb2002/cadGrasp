"""Finite native-pose direction menu shared with DSL initialization."""
import numpy as np
from codes.precompute_objects.head_geometry import horizontal_catalogue
from step2_local_support import withdrawal as W

def catalogue(mesh):
    vectors=list(np.asarray(horizontal_catalogue(mesh)['vectors']))
    def add(v):
        v=np.asarray(v,float);v=v/np.linalg.norm(v)
        if v[2]>1e-10 or any(np.linalg.norm(v-q)<1e-7 for q in vectors):return
        vectors.append(v)
    for e in (-15,-30,-45,-60,-75):
        for a in range(0,360,30):
            el,az=np.deg2rad([e,a]);add([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
    add([0,0,-1])
    for n in mesh.face_normals[np.argsort(-mesh.area_faces)[:24]]:
        if n[2]<=0:add(n)
    return dict(vectors=np.asarray(vectors).tolist(),global_allowed_directions=W.normalize(range(len(vectors))),
        preferred_withdrawal_direction=None,installation=dict(floor_plane=[0.,0.,0.,0.]),
        motion='Object moves -d in native saved world; fixed heads, not virtual-head floor',
        finite_direction_menu=True,forced_upward_exit=False)
