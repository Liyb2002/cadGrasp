"""Independent surface-distance and winding-number checks for exported solids."""
import numpy as np
import trimesh

def surface_distances(mesh,points):
    """Distance queries that preserve nanoscale triangles in metre coordinates.

    Plane projections and three edge projections avoid a fixed absolute
    tolerance on squared/fourth-power edge products in the generic query.
    The bounding tree only selects candidate faces; it does not decide distance.
    """
    points=np.asarray(points,float)
    result=[]
    for start in range(0,len(points),128):
        chunk=points[start:start+128]
        candidates=trimesh.proximity.nearby_faces(mesh,chunk)
        count=np.array([len(ids) for ids in candidates])
        tri=np.asarray(mesh.triangles[np.concatenate(candidates)],np.longdouble)
        p=np.repeat(np.asarray(chunk,np.longdouble),count,axis=0)
        edges=np.roll(tri,-1,axis=1)-tri
        n=np.cross(edges[:,0],-edges[:,2]);n2=np.sum(n*n,axis=1)
        valid=n2>0
        signed=np.sum((p-tri[:,0])*n,axis=1)
        projected=p-n*np.divide(signed,n2,out=np.zeros_like(signed),where=valid)[:,None]
        side=np.sum(np.cross(edges,projected[:,None,:]-tri)*n[:,None,:],axis=2)
        inside=valid & np.all(side>=0,axis=1)
        e2=np.sum(edges*edges,axis=2)
        fraction=np.divide(np.sum((p[:,None,:]-tri)*edges,axis=2),e2,
                           out=np.zeros_like(e2),where=e2>0)
        closest=tri+np.clip(fraction,0,1)[:,:,None]*edges
        d2=np.min(np.sum((p[:,None,:]-closest)**2,axis=2),axis=1)
        d2[inside]=signed[inside]**2/n2[inside]
        offsets=np.r_[0,np.cumsum(count)]
        result.extend(float(np.sqrt(d2[lo:hi].min())) for lo,hi in zip(offsets[:-1],offsets[1:]))
    return np.asarray(result)

def winding_number(mesh,point):
    """Oriented solid-angle containment, independent of the generator's ray test."""
    vectors=mesh.triangles-np.asarray(point)
    a,b,c=vectors[:,0],vectors[:,1],vectors[:,2]
    la,lb,lc=np.linalg.norm(a,axis=1),np.linalg.norm(b,axis=1),np.linalg.norm(c,axis=1)
    numerator=np.einsum('ij,ij->i',a,np.cross(b,c))
    denominator=la*lb*lc+np.einsum('ij,ij->i',a,b)*lc+np.einsum('ij,ij->i',b,c)*la+np.einsum('ij,ij->i',c,a)*lb
    return float(np.sum(2*np.arctan2(numerator,denominator))/(4*np.pi))
