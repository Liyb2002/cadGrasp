"""Unbounded working-face access cones shared by contacts, material and meshes.

Each whole triangle is Minkowski-added to a circumscribed polygonal cone.
No ray length or directional sampling truncates the search exclusion.
"""
import os
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/pose_set_search_numba')
import numpy as np
from numba import njit
from scipy.spatial import ConvexHull
from common import C,unpack_solid,tangent_frame

SIDES=32


@njit(cache=True)
def _contains(points,planes,counts):
    result=np.zeros(len(points),np.bool_)
    for i in range(len(points)):
        x,y,z=points[i]
        for face in range(len(counts)):
            inside=True
            for p in range(counts[face]):
                a,b,c,d=planes[face,p]
                if a*x+b*y+c*z+d>1e-10:
                    inside=False
                    break
            if inside:
                result[i]=True
                break
    return result


class WorkAccess:
    def __init__(self,mesh,face_ids,half_angle_deg):
        self.face_ids=np.asarray(face_ids,int)
        self.triangles=mesh.triangles[self.face_ids].copy()
        self.normals=mesh.face_normals[self.face_ids].copy()
        self.half_angle_deg=float(half_angle_deg)
        if not 0<self.half_angle_deg<90:raise ValueError('Invalid saved work cone angle')
        self.clearance=float(mesh.extents.max())*1e-6
        self.radial_slope=np.tan(np.radians(self.half_angle_deg))/np.cos(np.pi/SIDES)
        angles=(np.arange(SIDES)+.5)*2*np.pi/SIDES
        self.rings=[];families=[]
        for triangle,normal in zip(self.triangles,self.normals):
            frame=tangent_frame(normal)
            ring=self.radial_slope*(np.cos(angles)[:,None]*frame[:,0]+np.sin(angles)[:,None]*frame[:,1])
            self.rings.append(ring)
            hull=ConvexHull(self.cell_points(triangle,normal,ring,1.))
            equations=hull.equations[hull.equations[:,:3] @ normal<1e-7]
            # Remove only the far cap: the remaining halfspaces are unbounded.
            _,unique=np.unique(np.round(equations,10),axis=0,return_index=True)
            equations=equations[np.sort(unique)]
            base=int(np.argmin(equations[:,:3] @ normal))
            equations=np.concatenate([equations[base:base+1],np.delete(equations,base,axis=0)])
            families.append(equations)
        self.rings=np.asarray(self.rings)
        self.counts=np.array([len(p) for p in families],np.int64)
        self.planes=np.zeros((len(families),int(self.counts.max()),4))
        for i,p in enumerate(families):self.planes[i,:len(p)]=p
        self._solids={}

    def cell_points(self,triangle,normal,ring,length):
        base=triangle-self.clearance*normal
        top=(base[:,None,:]+length*(normal+ring)[None,:,:]).reshape(-1,3)
        return np.vstack([base,top])

    def contains_points(self,points):
        return _contains(np.ascontiguousarray(points,dtype=np.float64),self.planes,self.counts)

    def required_length(self,points,minimum=.5):
        # All possible support is inside the fitted-wrap seed. Cap geometry
        # beyond that seed, never at a physical/tool or force-dependent length.
        height=np.asarray(points)[None]-self.triangles[:,0,None]
        maximum=float(np.einsum('fpc,fc->fp',height,self.normals).max())
        return max(float(minimum),maximum+2*self.clearance+.001)

    def solid(self,length):
        key=round(float(length),9)
        if key not in self._solids:
            parts=[C.S.solid(C.G.hull_mesh(self.cell_points(t,n,r,key)))
                   for t,n,r in zip(self.triangles,self.normals,self.rings)]
            self._solids[key]=C.union(parts)
        return self._solids[key]

    def mesh(self,length):
        return unpack_solid(self.solid(length))

    def definition(self):
        return dict(source='original task.domain.work_ids and task.domain.data.load.cone_half_deg',
            half_angle_deg=self.half_angle_deg,total_opening_deg=2*self.half_angle_deg,
            equation='union over ALL work-triangle points p of p + outward cone(alpha)',
            work_face_count=len(self.face_ids),unbounded_during_search=True,
            whole_work_triangles=True,full_angle_range=True,cone_sides=SIDES,
            circumscribed=True,maximum_radial_overestimate_fraction=float(1/np.cos(np.pi/SIDES)-1),
            clearance_m=self.clearance,self_occluded_directions_also_reserved=True,
            finite_mesh_cap_only_beyond_entire_candidate_seed=True)
