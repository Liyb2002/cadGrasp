"""A single closed rounded leg mesh, avoiding coincident capsule unions."""
import numpy as np
import trimesh
import manifold3d as md
from deterministic_base import SCALE
from deterministic_base import solid


def flared_path(path,shaft_radius,root_radius,flare_length=.020,minimum_envelope=None):
    path=np.asarray(path,dtype=float)
    keep=np.r_[True,np.linalg.norm(np.diff(path,axis=0),axis=1)>1e-8]
    path=path[keep]
    if len(path)<2:return md.Manifold.sphere(root_radius/SCALE,32).translate(tuple(path[0]/SCALE))
    lengths=np.linalg.norm(np.diff(path,axis=0),axis=1)
    cumulative=np.r_[0.,np.cumsum(lengths)];total=float(cumulative[-1])
    reach=min(flare_length,total*.55)
    directions=np.diff(path,axis=0)/lengths[:,None]
    stops=np.unique(np.r_[cumulative,np.linspace(0.,reach,9)])
    centers=np.array([np.interp(stops,cumulative,path[:,k])for k in range(3)]).T
    u=np.clip(stops/reach,0.,1.);radii=shaft_radius+(root_radius-shaft_radius)*(1.-u*u*(3.-2.*u))
    if minimum_envelope is not None:
        old_tip,old_root=minimum_envelope
        slope=min(.85,abs(old_tip-old_root)/total)
        envelope=(old_root+(old_tip-old_root)*stops/total)/np.sqrt(1.-slope*slope)
        radii=np.maximum(radii,envelope*1.007)
    tangents=[]
    for distance in stops:
        index=min(len(lengths)-1,int(np.searchsorted(cumulative,distance,side='right')-1))
        tangent=directions[index]
        if index and abs(distance-cumulative[index])<1e-10:
            combined=directions[index-1]+directions[index]
            if np.linalg.norm(combined)>1e-8:tangent=combined/np.linalg.norm(combined)
        tangents.append(tangent)
    angles=np.linspace(0.,2*np.pi,32,endpoint=False)
    rings=[];previous=None
    def ring(center,radius,tangent):
        nonlocal previous
        if previous is None:
            axis=np.eye(3)[int(np.argmin(abs(tangent)))];across=np.cross(tangent,axis)
        else:
            across=previous-tangent*float(previous@tangent)
            if np.linalg.norm(across)<1e-8:
                axis=np.eye(3)[int(np.argmin(abs(tangent)))];across=np.cross(tangent,axis)
        across/=np.linalg.norm(across);previous=across
        other=np.cross(tangent,across)
        rings.append(center+radius*(np.cos(angles)[:,None]*across+np.sin(angles)[:,None]*other))
    effective_root=float(radii[0]);effective_tip=float(radii[-1])
    for angle in [-np.pi/3,-np.pi/6]:
        ring(path[0]+effective_root*np.sin(angle)*directions[0],effective_root*np.cos(angle),directions[0])
    for center,radius,tangent in zip(centers,radii,tangents):ring(center,radius,tangent)
    for angle in [np.pi/6,np.pi/3]:
        ring(path[-1]+effective_tip*np.sin(angle)*directions[-1],effective_tip*np.cos(angle),directions[-1])
    vertices=np.vstack([path[0]-effective_root*directions[0],*rings,path[-1]+effective_tip*directions[-1]])
    faces=[];n=len(angles)
    for j in range(n):faces.append([0,1+(j+1)%n,1+j])
    for k in range(len(rings)-1):
        first=1+k*n;second=first+n
        for j in range(n):
            next_j=(j+1)%n
            faces.extend([[first+j,first+next_j,second+next_j],[first+j,second+next_j,second+j]])
    last=1+(len(rings)-1)*n;pole=len(vertices)-1
    for j in range(n):faces.append([last+j,last+(j+1)%n,pole])
    mesh=trimesh.Trimesh(vertices,faces,process=False)
    if mesh.volume<0:mesh.invert()
    if not mesh.is_watertight or not mesh.is_winding_consistent:raise ValueError('Invalid rounded loft topology')
    return solid(mesh)
