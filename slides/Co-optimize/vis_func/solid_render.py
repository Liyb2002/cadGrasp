"""Solid arrows and depth-buffered orthographic mesh presentation."""
import numpy as np
import trimesh
from mpl_toolkits.mplot3d import proj3d


def arrow_mesh(origin,direction,length):
    direction=np.asarray(direction,dtype=float);direction/=np.linalg.norm(direction)
    head=length*.24;shaft=length-head
    body=trimesh.creation.cylinder(radius=length*.024,height=shaft,sections=24)
    body.apply_translation([0,0,shaft/2])
    tip=trimesh.creation.cone(radius=length*.075,height=head,sections=24)
    tip.apply_translation([0,0,shaft])
    mesh=trimesh.util.concatenate([body,tip])
    mesh.apply_transform(trimesh.geometry.align_vectors([0,0,1],direction))
    mesh.apply_translation(origin)
    return mesh


def depth_render(ax,triangles,colors,transparent=None,tint=None):
    """Rasterize true per-pixel depth, avoiding mplot3d average-face sorting."""
    fig=ax.figure;w,h=fig.canvas.get_width_height();rgba=np.zeros((h,w,4),dtype=float);depth=np.full((h,w),np.inf)
    projection=ax.get_proj()
    def projected(ts):
        xyz=np.array(proj3d.proj_transform(*ts.reshape(-1,3).T,projection)).T
        xy=ax.transData.transform(xyz[:,:2]);xy[:,1]=h-xy[:,1]
        return np.c_[xy,xyz[:,2]].reshape(-1,3,3)
    def paint(t,color,opaque):
        xmin=max(0,int(np.floor(t[:,0].min())));xmax=min(w-1,int(np.ceil(t[:,0].max())))
        ymin=max(0,int(np.floor(t[:,1].min())));ymax=min(h-1,int(np.ceil(t[:,1].max())))
        if xmin>xmax or ymin>ymax:return
        x,y=np.meshgrid(np.arange(xmin,xmax+1)+.5,np.arange(ymin,ymax+1)+.5)
        a,b,c=t;den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(den)<1e-12:return
        u=((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/den
        v=((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/den;z=u*a[2]+v*b[2]+(1-u-v)*c[2]
        # Coplanar Boolean interfaces keep the first material rather than flicker.
        # This tolerance applies only to display depth, never to solid geometry.
        dst=depth[ymin:ymax+1,xmin:xmax+1];mask=(u>=0)&(v>=0)&(u+v<=1)&(z<dst-1e-5)
        pixels=rgba[ymin:ymax+1,xmin:xmax+1]
        if opaque:dst[mask]=z[mask];pixels[mask]=color
        else:
            alpha=color[3];old=pixels[mask];outalpha=alpha+old[:,3]*(1-alpha)
            pixels[mask,:3]=(np.asarray(color[:3])*alpha+old[:,:3]*old[:,3,None]*(1-alpha))/outalpha[:,None]
            pixels[mask,3]=outalpha
    e,a=np.radians([ax.elev,ax.azim]);camera=np.array([np.cos(e)*np.cos(a),np.cos(e)*np.sin(a),np.sin(e)])
    normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]);visible=normals@camera>0
    for t,color in zip(projected(triangles[visible]),colors[visible]):paint(t,color,True)
    if transparent is not None:
        ts=projected(transparent)
        order=np.argsort(ts[:,:,2].mean(1))[::-1]
        tints=np.tile(tint,(len(ts),1)) if np.asarray(tint).ndim==1 else np.asarray(tint)
        for index in order:paint(ts[index],tints[index],False)
    fig.figimage((np.clip(rgba,0,1)*255).astype(np.uint8),xo=0,yo=0,origin='upper',zorder=.5)
