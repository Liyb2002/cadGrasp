from pathlib import Path
import ctypes,shutil,subprocess
import numpy as np
from PIL import Image
class Renderer:
    def __init__(self):
        folder=Path(__file__).resolve().parent;library=folder/'transparency.dylib'
        if not library.exists():subprocess.run([shutil.which('clang++') or shutil.which('g++'),'-O3','-std=c++17','-shared','-fPIC',str(folder/'transparency.cpp'),'-o',str(library)],check=True)
        self.raster=ctypes.CDLL(str(library)).raster_layers
        self.raster.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p];self.raster.restype=None
    def render(self,parts,camera,size=(1600,900)):
        width,height=size;w,h=width*2,height*2;focus,basis,span=camera;triangles=[];colors=[];alphas=[]
        light=basis[2]*.72+basis[1]*.5-basis[0]*.22;light/=np.linalg.norm(light)
        fill=basis[2]*.25+basis[1]*.3+basis[0]*.65;fill/=np.linalg.norm(fill)
        for mesh,rotation,point in parts:
            p=(mesh['triangles']@rotation.T+point-focus)@basis.T
            p[...,0]=w/2+p[...,0]*h/span;p[...,1]=h/2-p[...,1]*h/span;p[...,2]+=mesh['bias']
            normal=mesh['normals']@rotation.T;shade=.64+.27*np.maximum(0,normal@light)+.12*np.maximum(0,normal@fill)
            if mesh.get('unlit'):shade[:]=1
            color=mesh['colors']*shade[...,None];alpha=mesh.get('opacity',1.)
            if np.allclose(mesh['colors'][0,0],[42,145,210],atol=.1):color=color*alpha+255*(1-alpha);alpha=1.
            triangles.append(p.reshape(-1,9));colors.append(color.reshape(-1,9));alphas.extend([alpha]*len(p))
        p=np.ascontiguousarray(np.concatenate(triangles),dtype=np.float32);c=np.ascontiguousarray(np.concatenate(colors),dtype=np.float32)
        a=np.ascontiguousarray(alphas,dtype=np.float32);out=np.empty((h,w,3),dtype=np.uint8)
        self.raster(p.ctypes.data,c.ctypes.data,len(p),w,h,a.ctypes.data,out.ctypes.data)
        return Image.fromarray(out).resize(size,Image.Resampling.LANCZOS)
