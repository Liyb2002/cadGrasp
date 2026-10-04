"""Optional compiled version of the same convex-plane triangle clipping test."""
import numpy as np
try:
    from numba import njit
except ImportError:
    njit=None

def first_obstruction(triangles,ids,planes,tol):
    capacity=3+len(planes)+8
    a=np.empty((capacity,3));b=np.empty((capacity,3));values=np.empty(capacity)
    for face in ids:
        possible=True
        for plane in planes:
            outside=True
            for j in range(3):
                v=plane[3]
                for axis in range(3):v+=triangles[face,j,axis]*plane[axis]
                if v < -tol:outside=False;break
            if outside:possible=False;break
        if not possible:continue
        a[:3]=triangles[face];n=3
        for plane in planes:
            inside_count=0
            for j in range(n):
                v=plane[3]+tol
                for axis in range(3):v+=a[j,axis]*plane[axis]
                values[j]=v
                if v<=0:inside_count+=1
            if inside_count==n:continue
            if inside_count==0:n=0;break
            k=0
            for j in range(n):
                previous=(j-1)%n
                vi,vj=values[previous],values[j]
                if (vi<=0)!=(vj<=0):
                    alpha=vi/(vi-vj)
                    for axis in range(3):b[k,axis]=a[previous,axis]+alpha*(a[j,axis]-a[previous,axis])
                    k+=1
                if vj<=0:
                    b[k]=a[j];k+=1
            a,b=b,a;n=k
            if n<3:break
        if n>=3:
            area=0.
            for j in range(1,n-1):
                u=a[j]-a[0];v=a[j+1]-a[0]
                x=u[1]*v[2]-u[2]*v[1];y=u[2]*v[0]-u[0]*v[2];z=u[0]*v[1]-u[1]*v[0]
                area+=np.sqrt(x*x+y*y+z*z)/2
            if area>tol*tol:return int(face)
    return -1

compiled=njit(cache=True)(first_obstruction) if njit else None

def edge_bounds(poly,normal,segment,tolerance):
    lo=0.;hi=1.
    delta=segment[1]-segment[0]
    for j in range(len(poly)):
        edge=poly[(j+1)%len(poly)]-poly[j]
        start=segment[0]-poly[j]
        value=((edge[1]*start[2]-edge[2]*start[1])*normal[0]+(edge[2]*start[0]-edge[0]*start[2])*normal[1]+(edge[0]*start[1]-edge[1]*start[0])*normal[2])
        slope=((edge[1]*delta[2]-edge[2]*delta[1])*normal[0]+(edge[2]*delta[0]-edge[0]*delta[2])*normal[1]+(edge[0]*delta[1]-edge[1]*delta[0])*normal[2])
        threshold=tolerance*np.sqrt((edge*edge).sum())
        if abs(slope)<=threshold*1e-3:
            if value < -threshold:return 1.,0.
        elif slope>0:lo=max(lo,-value/slope)
        else:hi=min(hi,-value/slope)
    return lo,hi
compiled_edge=njit(cache=True)(edge_bounds) if njit else None
