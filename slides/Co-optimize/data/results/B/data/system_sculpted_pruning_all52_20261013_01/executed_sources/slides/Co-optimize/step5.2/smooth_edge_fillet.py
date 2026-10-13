"""An inward edge fillet whose radius smoothly vanishes at both endpoints."""
import numpy as np
import manifold3d as md
from deterministic_base import SCALE


def fillet_knife(a,b,n1,n2,radius):
    direction=np.asarray(b)-a;length=float(np.linalg.norm(direction));direction/=length
    bisector=n1+n2;bisector/=np.linalg.norm(bisector)
    cosine=float(n1@bisector)
    if cosine<.15:return md.Manifold()
    across=np.cross(direction,bisector);across/=np.linalg.norm(across)
    angle=float(np.arccos(np.clip(n1@n2,-1,1)));s=np.linspace(0.,1.,25)
    arc=(np.sin((1-s)*angle)[:,None]*n1+np.sin(s*angle)[:,None]*n2)/np.sin(angle)
    template=np.vstack([np.zeros(2),np.c_[arc@bisector-1/cosine,arc@across]])
    area=np.sum(template[:,0]*np.roll(template[:,1],-1)-template[:,1]*np.roll(template[:,0],-1))
    if area<0:template=template[::-1]
    # Smoothstep shoulders meet the untouched edge with zero radius slope.
    blend=min(length*.2,radius*2)
    stations=np.unique(np.r_[np.linspace(0,blend/length,7),np.linspace(blend/length,1-blend/length,5),np.linspace(1-blend/length,1,7)])
    vertices=[np.asarray(a)];rings=[]
    for t in stations[1:-1]:
        u=min(t*length/blend,(1-t)*length/blend,1.)
        r=radius*u*u*(3-2*u)
        ring=np.asarray(a)+t*length*direction+r*(template[:,:1]*bisector+template[:,1:]*across)
        rings.append(np.arange(len(vertices),len(vertices)+len(template)))
        vertices.extend(ring)
    top=len(vertices);vertices.append(np.asarray(b));faces=[]
    for j in range(len(template)):
        q=(j+1)%len(template);faces.append([0,rings[0][q],rings[0][j]])
        for low,high in zip(rings[:-1],rings[1:]):faces.extend([[low[j],low[q],high[q]],[low[j],high[q],high[j]]])
        faces.append([rings[-1][j],rings[-1][q],top])
    result=md.Manifold(md.Mesh64(np.ascontiguousarray(np.asarray(vertices)/SCALE),np.ascontiguousarray(faces,dtype=np.uint32)))
    if result.status()!=md.Error.NoError:raise ValueError('Invalid smooth-end fillet knife: '+str(result.status()))
    return result
