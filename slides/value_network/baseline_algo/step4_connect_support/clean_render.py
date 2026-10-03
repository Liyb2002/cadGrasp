"""Unlabelled CAD pictures: grey solid, translucent object, distinct heads."""
import colorsys
import ctypes
from pathlib import Path
import subprocess
import shutil
import tempfile

import numpy as np
from PIL import Image
import trimesh

from step3_scheculer import contacts as I
from step4_connect_support.publish_compact import CPU, piece

SUPPORT = '#bfc2c0'
OBJECT = '#929da4'
OPACITY = .22
PALETTE = ['#dc9d47', '#ac7098', '#7196c0', '#50a59b', '#77a76a',
           '#cf796c', '#9c86c5', '#d5bb51', '#4cabb3', '#dc8cb5',
           '#a59668', '#81b5d2', '#b67547', '#8c9e4b', '#897ca3',
           '#bb6d88', '#609889', '#bbba75', '#7481b9', '#d69b78']


def head_colors(identifiers):
    return {ident: PALETTE[k] if k < len(PALETTE) else
            '#%02x%02x%02x' % tuple(int(255*x) for x in colorsys.hsv_to_rgb(k*.61803398875 % 1, .5, .8))
            for k,ident in enumerate(identifiers)}


def collar(mesh, low, high):
    """Co-design's exact clipping of exterior triangles to a display-only box."""
    lo,hi=np.asarray(low),np.asarray(high)
    triangles=mesh.triangles
    triangles=triangles[np.all(triangles.max(1)>=lo,axis=1)&np.all(triangles.min(1)<=hi,axis=1)]
    result=[]
    for tri in triangles:
        polygon=list(tri)
        for axis in range(3):
            for bound,sign in ((lo[axis],1),(hi[axis],-1)):
                clipped=[]
                for a,b in zip(polygon,polygon[1:]+polygon[:1]):
                    da,db=sign*(a[axis]-bound),sign*(b[axis]-bound)
                    if da>=0:clipped.append(a)
                    if (da>=0)!=(db>=0):clipped.append(a+da/(da-db)*(b-a))
                polygon=clipped
        result.extend([[polygon[0],polygon[j],polygon[j+1]] for j in range(1,len(polygon)-1)])
    points=np.asarray(result).reshape(-1,3)
    return trimesh.Trimesh(points,np.arange(len(points)).reshape(-1,3),process=False)


class Renderer:
    def __init__(self):
        source=Path(__file__).with_name('translucent_raster.cpp')
        library=Path(tempfile.gettempdir())/('cadgrasp-translucent-'+I.sha256(source)[:16]+'.dylib')
        if not library.exists():
            compiler = shutil.which('clang++') or shutil.which('g++')
            if compiler is None:
                raise RuntimeError('CAD renderer requires clang++ or g++')
            subprocess.run([compiler,'-O3','-std=c++17','-shared','-fPIC',str(source),'-o',str(library)],check=True)
        self.raster=ctypes.CDLL(str(library)).raster
        self.raster.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]
        self.raster.restype=None

    def render(self, parts, camera, size=1000):
        # Supersampling preserves the thin exact head surfaces without text.
        w=h=size*2; focus,basis,span=camera
        tris=[];colors=[];alpha=[]
        light=basis[2]*.72+basis[1]*.5-basis[0]*.22;light/=np.linalg.norm(light)
        fill=basis[2]*.25+basis[1]*.3+basis[0]*.65;fill/=np.linalg.norm(fill)
        for mesh,rotation,point in parts:
            p=(mesh['triangles']@rotation.T+point-focus)@basis.T
            p[...,0]=w/2+p[...,0]*h/span
            p[...,1]=h/2-p[...,1]*h/span
            p[...,2]+=mesh['bias']
            n=mesh['normals']@rotation.T
            shade=.64+.27*np.maximum(0,n@light)+.12*np.maximum(0,n@fill)
            if mesh.get('unlit'):shade[:]=1
            tris.append(p.reshape(-1,9))
            colors.append((mesh['colors']*shade[...,None]).reshape(-1,9))
            alpha.extend([mesh.get('opacity',1.)]*len(p))
        p=np.ascontiguousarray(np.concatenate(tris),dtype=np.float32)
        c=np.ascontiguousarray(np.concatenate(colors),dtype=np.float32)
        a=np.ascontiguousarray(alpha,dtype=np.float32)
        output=np.empty((h,w,3),dtype=np.uint8)
        self.raster(p.ctypes.data,c.ctypes.data,len(p),w,h,a.ctypes.data,output.ctypes.data)
        return Image.fromarray(output).resize((size,size),Image.Resampling.LANCZOS)
