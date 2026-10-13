"""Deterministic membership of the Oracle's closed legal components.

Ray parity is a fast filter. Ambiguous parity uses signed solid angles on
the SAME triangles and SAME probe point; no exclusion or point is moved.
"""
import numpy as np
from numba import njit
from trimesh.ray.ray_pyembree import RayMeshIntersector
from trimesh.ray.ray_util import contains_points


@njit(cache=True)
def winding_numbers(triangles, points):
    values=np.zeros(len(points))
    for i in range(len(points)):
        total=0.;compensation=0.
        for triangle in triangles:
            a=triangle[0]-points[i];b=triangle[1]-points[i];c=triangle[2]-points[i]
            la=np.sqrt(a@a);lb=np.sqrt(b@b);lc=np.sqrt(c@c)
            if min(la,lb,lc)<=1e-15:
                total=np.nan;break
            a=a/la;b=b/lb;c=c/lc
            determinant=a[0]*(b[1]*c[2]-b[2]*c[1])+a[1]*(b[2]*c[0]-b[0]*c[2])+a[2]*(b[0]*c[1]-b[1]*c[0])
            angle=np.arctan2(determinant,1+a@b+b@c+c@a)
            increment=angle-compensation;new=total+increment
            compensation=(new-total)-increment;total=new
        values[i]=total/(2*np.pi)
    return values


def robust_contains(mesh,points,intersector=None):
    points=np.asarray(points,float).reshape(-1,3)
    ray=RayMeshIntersector(mesh) if intersector is None else intersector
    # Specifying the direction disables Trimesh's random fallback.
    inside=contains_points(ray,points,check_direction=np.array([1.,np.sqrt(2.),np.sqrt(3.)]))
    candidate=(~inside)&np.all(points>mesh.bounds[0],axis=1)&np.all(points<mesh.bounds[1],axis=1)
    indices=np.flatnonzero(candidate)
    if len(indices) and mesh.is_watertight:
        winding=winding_numbers(np.asarray(mesh.triangles),points[indices])
        inside[indices]=np.isfinite(winding)&(np.abs(np.abs(winding)-1.)<=1e-6)
    return inside


class StableRayMeshIntersector(RayMeshIntersector):
    def contains_points(self,points):
        return robust_contains(self.mesh,points,self)
