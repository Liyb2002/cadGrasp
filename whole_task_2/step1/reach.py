"""Faces a given head can paint: a needle (length L, radius r), then a head of radius R.

The arm translates freely. A point p is reached along tool axis d (the needle
points along d, into the surface) when
  - the needle line {p - s d : s >= eps} misses the object;
  - the needle body clears it by r beyond the tip contact, s >= r(1 + tan phi),
    where phi is the needle's incidence on the face: a flat-ended needle of
    radius r touching a plane at incidence phi has its rim at r tan phi;
  - no object point lies within R of the line at s >= L (the head and all behind).
The face's outward normal must also face the tool, within the process's maximum
incidence when one is given (the angle between needle and inward normal). A face
is reached when its three vertices and centroid are.

Screening uses one orthographic depth map per axis; every plan is then certified
with the exact clipped-triangle distance below, never with the depth maps.
"""
import numpy as np
from scipy import ndimage

CROSS=np.array([[0,1,0],[1,1,1],[0,1,0]],bool)
MAX_INCIDENCE=np.radians(80.)


def facing(max_incidence_deg):
    """Smallest allowed cos(needle, inward normal), exclusive: facing the tool, within the incidence limit."""
    return 0. if max_incidence_deg is None else float(np.cos(np.radians(max_incidence_deg)))-1e-12


def tip_contact(radius,cos_incidence):
    """Distance behind the tip where a flat needle end of this radius stops touching its face."""
    phi=np.minimum(np.arccos(np.clip(cos_incidence,-1,1)),MAX_INCIDENCE)
    return radius*(1+np.tan(phi))


def sphere_axes(spacing_deg):
    """Near-uniform unit vectors whose typical neighbour spacing is spacing_deg."""
    n=int(np.ceil(4*np.pi/np.radians(spacing_deg)**2))
    i=np.arange(n)+.5;phi=np.arccos(1-2*i/n);theta=np.pi*(1+5**.5)*i
    return np.c_[np.cos(theta)*np.sin(phi),np.sin(theta)*np.sin(phi),np.cos(phi)]


def frame(d):
    u=np.cross(d,[1.,0,0] if abs(d[0])<.9 else [0,1.,0]);u/=np.linalg.norm(u)
    return u,np.cross(d,u)


def test_points(mesh):
    """Per face: three vertices and the centroid, shape (faces, 4, 3)."""
    return np.concatenate([mesh.triangles,mesh.triangles_center[:,None]],axis=1)


class Screen:
    """Depth-map screening of every face against one tool axis."""
    def __init__(self,mesh,length,radius,needle_radius,pixel,samples,max_incidence_deg=None):
        import trimesh
        self.mesh=mesh;self.length=length;self.radius=radius;self.needle_radius=needle_radius;self.pixel=pixel
        self.facing=facing(max_incidence_deg)
        points,_=trimesh.sample.sample_surface(mesh,samples,seed=1)
        self.points=np.vstack([points,mesh.vertices]);self.tests=test_points(mesh)

    def erode(self,D,r):
        """Minimum over an octagon of radius ~r: alternate 3x3 square and cross steps."""
        for i in range(int(round(r/self.pixel))):
            D=ndimage.minimum_filter(D,size=3,mode='constant',cval=np.inf) if i%2==0 else \
              ndimage.minimum_filter(D,footprint=CROSS,mode='constant',cval=np.inf)
        return D

    def __call__(self,d):
        h=self.pixel;u,v=frame(d);P=np.c_[self.points@u,self.points@v];z=self.points@d
        lo=P.min(0)-self.radius-2*h;ij=np.floor((P-lo)/h).astype(int)
        D=np.full(ij.max(0)+int(self.radius/h)+3,np.inf);np.minimum.at(D,(ij[:,0],ij[:,1]),z)
        needle=ndimage.minimum_filter(D,size=3,mode='constant',cval=np.inf)
        body=self.erode(D,self.needle_radius);head=self.erode(D,self.radius)
        T=self.tests.reshape(-1,3);q=np.floor((np.c_[T@u,T@v]-lo)/h).astype(int);zt=T@d
        cos=-(self.mesh.face_normals@d);tip=np.repeat(tip_contact(self.needle_radius,cos),4)
        ok=((needle[q[:,0],q[:,1]]>=zt-3*h)&(body[q[:,0],q[:,1]]>=zt-tip)
            &(head[q[:,0],q[:,1]]>=zt-self.length))
        return ok.reshape(-1,4).all(1)&(cos>self.facing)


def clearance(triangles,points,d,tau):
    """Exact distance from each line {p - s d : s >= tau} to the triangles.

    Each triangle is clipped to s >= tau; its image in the plane normal to d is
    convex, so the distance is zero inside it and otherwise the nearest edge.
    """
    u,v=frame(d);w=-np.asarray(d);result=np.empty(len(points))
    for k,p in enumerate(points):
        X=triangles-p;s=X@w;q=np.stack([X@u,X@v],axis=-1)
        keep=s>=tau;slots=np.zeros((len(X),6,2));valid=np.zeros((len(X),6),bool)
        for e,(i,j) in enumerate([(0,1),(1,2),(2,0)]):
            slots[:,2*e]=q[:,i];valid[:,2*e]=keep[:,i]
            cross=keep[:,i]!=keep[:,j];t=np.where(cross,(tau-s[:,i])/np.where(cross,s[:,j]-s[:,i],1.),0.)
            slots[:,2*e+1]=q[:,i]+t[:,None]*(q[:,j]-q[:,i]);valid[:,2*e+1]=cross
        order=np.argsort(~valid,axis=1,kind='stable');slots=np.take_along_axis(slots,order[...,None],1)
        n=valid.sum(1);live=n>0
        if not live.any():result[k]=np.inf;continue
        slots=slots[live];n=n[live];index=np.arange(6)[None];used=index<n[:,None]
        a=slots;b=np.take_along_axis(slots,np.where(index+1<n[:,None],index+1,0)[...,None],1)
        e=b-a;t=np.clip(-(a*e).sum(-1)/np.maximum((e*e).sum(-1),1e-300),0,1)
        dist=np.linalg.norm(a+t[...,None]*e,axis=-1);dist[~used]=np.inf;best=dist.min(1)
        c=e[...,0]*(-a[...,1])-e[...,1]*(-a[...,0]);edge=used&(n[:,None]>=3)
        # A triangle parallel to d projects to a segment: only the edge distance applies.
        twice_area=np.abs(np.where(used,a[...,0]*b[...,1]-a[...,1]*b[...,0],0.).sum(1))
        solid=twice_area>1e-12*np.where(used,(e*e).sum(-1),0.).max(1)
        inside=(n>=3)&solid&(np.all(np.where(edge,c>=0,True),1)|np.all(np.where(edge,c<=0,True),1))
        best[inside]=0.;result[k]=best.min()
    return result


def certify(mesh,face,d,length,radius,needle_radius,eps,max_incidence_deg=None):
    """Exact reach of one face along axis d; returns (passed, needle_body_gap_m, head_gap_m)."""
    cos=-float(mesh.face_normals[face]@d)
    if cos<=facing(max_incidence_deg):return False,0.,0.
    P=test_points(mesh)[face];T=mesh.triangles
    line=float(clearance(T,P,d,eps).min())
    body=float(clearance(T,P,d,float(tip_contact(needle_radius,cos))).min())
    head=float(clearance(T,P,d,length).min())
    return bool(line>0 and body>=needle_radius and head>=radius),body,head
