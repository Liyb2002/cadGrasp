"""Shared mesh pieces, cameras and CPU rasterization for Co-optimize figures.

Generic helpers relocated from the retired slides/reuse renderer. Geometry,
colors, lighting and rasterization retain the original behavior.
"""
from pathlib import Path
import ctypes
import shutil
import subprocess
import numpy as np
from PIL import Image, ImageColor
import trimesh

HERE = Path(__file__).resolve().parent


def rgb(color):
    return np.array(ImageColor.getrgb(color) if isinstance(color,str) else np.array(color)*255.,float)


def piece(data,color,smooth=False,bias=0):
    v=np.asarray(data['v']).reshape(-1,3);f=np.asarray(data['f']).reshape(-1,3)
    mesh=trimesh.Trimesh(v,f,process=False)
    normals=mesh.vertex_normals[f] if smooth else np.repeat(mesh.face_normals[:,None,:],3,axis=1)
    colors=np.tile(rgb(color),(len(f),3,1)) if not isinstance(color,np.ndarray) else color
    return dict(triangles=v[f],normals=normals,colors=colors,bias=bias)


def box(dimensions,center,color='#414c54'):
    mesh=trimesh.creation.box(dimensions);mesh.apply_translation(center)
    return piece(dict(v=mesh.vertices.ravel(),f=mesh.faces.ravel()),color)


def placed(mesh,rotation=np.eye(3),point=np.zeros(3)):
    return (mesh,rotation,point)


class Renderer:
    def __init__(self):
        source=HERE/'cpu_mesh_raster.cpp';library=HERE/'cpu_mesh_raster.so'
        if not library.exists() or library.stat().st_mtime < source.stat().st_mtime:
            compiler = shutil.which('clang++') or shutil.which('g++')
            if compiler is None:
                raise RuntimeError('CPU rasterization requires clang++ or g++')
            subprocess.run([compiler,'-O3','-std=c++17','-shared','-fPIC',str(source),'-o',str(library)],check=True)
        self.raster=ctypes.CDLL(str(library)).raster
        self.raster.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
        self.raster.restype=None

    def render(self,parts,camera,width,height,scale=1.0):
        focus,basis,span=camera
        w,h=round(width*scale),round(height*scale)
        tris=[];colors=[]
        light=basis[2]*.72+basis[1]*.5-basis[0]*.22;light/=np.linalg.norm(light)
        fill=basis[2]*.25+basis[1]*.3+basis[0]*.65;fill/=np.linalg.norm(fill)
        for mesh,rotation,point in parts:
            world=mesh['triangles']@rotation.T+point
            p=(world-focus)@basis.T
            p[...,0]=w/2+p[...,0]*h/span
            p[...,1]=h/2-p[...,1]*h/span
            p[...,2]+=mesh['bias']
            n=mesh['normals']@rotation.T
            shade=.64+.27*np.maximum(0,n@light)+.12*np.maximum(0,n@fill)
            if mesh.get('unlit'):shade[:]=1
            tris.append(p.reshape(-1,9));colors.append((mesh['colors']*shade[...,None]).reshape(-1,9))
        p=np.ascontiguousarray(np.concatenate(tris),dtype=np.float32)
        c=np.ascontiguousarray(np.concatenate(colors),dtype=np.float32)
        output=np.empty((h,w,3),dtype=np.uint8)
        self.raster(p.ctypes.data,c.ctypes.data,len(p),w,h,output.ctypes.data)
        im=Image.fromarray(output)
        return im.resize((width,height),Image.Resampling.LANCZOS) if scale!=1 else im


def axes(direction):
    view=np.asarray(direction,float);view/=np.linalg.norm(view)
    right=np.cross([0,0,1],view);right/=np.linalg.norm(right)
    return np.array([right,np.cross(view,right),view])


def corners(parts):
    result=[]
    for mesh,rotation,point in parts:
        lo=mesh['triangles'].min(axis=(0,1));hi=mesh['triangles'].max(axis=(0,1))
        cube=np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
        result.append(cube@rotation.T+point)
    return np.concatenate(result)


def fit(points,direction,aspect,padding=1.10):
    basis=axes(direction);p=points@basis.T;lo,hi=p.min(0),p.max(0)
    span=max(hi[1]-lo[1],(hi[0]-lo[0])/aspect)*padding
    return ((lo+hi)/2)@basis,basis,span
