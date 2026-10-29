"""Transparent overlays above opaque CAD material, on the CPU: the renderer and pieces of
slides/Co-optimize/seperate_base_support_proof (videos/01_hulls_and_support.mp4: transparency.py, transparency.cpp,
vis_func/step51_render.py, step5.1/hull_render.py), copied.

A piece is a dict: triangles, normals and colors (n x 3 x 3; colors 0-255), bias (moved towards the camera), opacity
(1: opaque), unlit.  A camera is (focus, basis: rows right, up, towards the camera; span: the view's width, m).  The
rasterizer keeps the four nearest transparent layers over the nearest opaque one, per pixel, at twice the size.
"""
import ctypes
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageColor

from whole_task_2.step1.common import ROOT

BUILD = ROOT/'whole_task_2/data/.build'
BLUE = '#2A91D2'
GRAY = '#A5ADB5'
COLORS = ['#df8236', '#895eab', '#299584', '#c75278', '#ab912c', '#5875bf', '#bc6456', '#649142', '#af64a0', '#398fba']
SOURCE = r'''
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

struct Layer {float z,a,r,g,b;};
extern "C" void raster_layers(const float* xyz,const float* rgb,int count,
    int width,int height,const float* alpha,uint8_t* pixels){
    constexpr int K=4;
    const int size=width*height;
    const float far=-std::numeric_limits<float>::infinity();
    std::fill(pixels,pixels+size*3,uint8_t(255));
    std::vector<float> depth(size,far);
    std::vector<Layer> layers(size*K,Layer{far,0,255,255,255});
    for(int pass=0;pass<2;++pass)for(int f=0;f<count;++f){
        if((alpha[f]<1.f)!=(pass==1)||alpha[f]<=0)continue;
        const float *a=xyz+9*f,*b=a+3,*c=a+6,*col=rgb+9*f;
        float det=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1]);
        if(std::abs(det)<1e-8f)continue;
        int x0=std::max(0,int(std::floor(std::min({a[0],b[0],c[0]}))));
        int x1=std::min(width-1,int(std::ceil(std::max({a[0],b[0],c[0]}))));
        int y0=std::max(0,int(std::floor(std::min({a[1],b[1],c[1]}))));
        int y1=std::min(height-1,int(std::ceil(std::max({a[1],b[1],c[1]}))));
        float ux=(b[1]-c[1])/det,uy=(c[0]-b[0])/det;
        float vx=(c[1]-a[1])/det,vy=(a[0]-c[0])/det;
        for(int y=y0;y<=y1;++y){
            float u=ux*(x0+.5f-c[0])+uy*(y+.5f-c[1]);
            float v=vx*(x0+.5f-c[0])+vy*(y+.5f-c[1]);
            for(int x=x0;x<=x1;++x,u+=ux,v+=vx){
                float w=1-u-v;if(u< -1e-5f||v< -1e-5f||w< -1e-5f)continue;
                int at=y*width+x;float z=u*a[2]+v*b[2]+w*c[2];
                if(z<=depth[at])continue;
                float color[3];for(int j=0;j<3;++j)color[j]=std::clamp(u*col[j]+v*col[j+3]+w*col[j+6],0.f,255.f);
                if(pass==0){depth[at]=z;for(int j=0;j<3;++j)pixels[3*at+j]=uint8_t(color[j]);}
                else{
                    Layer* p=&layers[K*at];int target=K;bool duplicate=false;
                    for(int j=0;j<K;++j){
                        if(std::abs(z-p[j].z)<2e-7f){duplicate=true;break;}
                        if(target==K&&z>p[j].z)target=j;
                    }
                    if(duplicate||target==K)continue;
                    for(int j=K-1;j>target;--j)p[j]=p[j-1];
                    p[target]=Layer{z,alpha[f],color[0],color[1],color[2]};
                }
            }
        }
    }
    for(int at=0;at<size;++at){
        float color[3]={float(pixels[3*at]),float(pixels[3*at+1]),float(pixels[3*at+2])};
        for(int j=K-1;j>=0;--j){const Layer& p=layers[K*at+j];
            color[0]=p.a*p.r+(1-p.a)*color[0];color[1]=p.a*p.g+(1-p.a)*color[1];color[2]=p.a*p.b+(1-p.a)*color[2];}
        for(int j=0;j<3;++j)pixels[3*at+j]=uint8_t(std::clamp(color[j],0.f,255.f));
    }
}
'''


def rgb(color):
    return np.array(ImageColor.getrgb(color) if isinstance(color, str) else np.array(color)*255., float)


def piece(mesh, color, opacity=1.):
    """A mesh's flat-shaded triangles in one colour."""
    f = np.asarray(mesh.faces); n = np.repeat(np.asarray(mesh.face_normals)[:, None, :], 3, axis=1)
    return dict(triangles=np.asarray(mesh.vertices)[f], normals=n, colors=np.tile(rgb(color), (len(f), 3, 1)), bias=0, opacity=opacity)


def placed(p, T):
    return p, np.asarray(T)[:3, :3], np.asarray(T)[:3, 3]


def camera(points, view, margin=1.16):
    """The camera looking along -view (towards the camera: view), framing points."""
    view = np.asarray(view, float)/np.linalg.norm(view); right = np.cross([0., 0., 1.], view); right /= np.linalg.norm(right)
    basis = np.array([right, np.cross(view, right), view]); p = np.asarray(points)@basis.T; low, high = p.min(0), p.max(0)
    return ((low+high)/2)@basis, basis, max(float(np.max(high[:2]-low[:2]))*margin, .001)


def points_piece(points, color, cam, size, half_size_pixels=.6):
    """Every point as a camera-facing square marker."""
    _, basis, span = cam; corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    squares = np.asarray(points)[:, None, :]+corners@basis[:2]*half_size_pixels*span/size
    triangles = squares[:, [[0, 1, 2], [0, 2, 3]], :].reshape(-1, 3, 3)
    return dict(triangles=triangles, normals=np.broadcast_to(basis[2], triangles.shape),
                colors=np.broadcast_to(rgb(color), triangles.shape), bias=2e-6, unlit=True)


def outline_piece(vertices, color, cam, size, half_width_pixels=1.35):
    """A closed polygon's straight edges as camera-facing ribbons of constant screen width."""
    vertices = np.asarray(vertices, float); _, basis, span = cam; ends = np.roll(vertices, -1, axis=0)
    projected = (ends-vertices)@basis[:2].T; length = np.linalg.norm(projected, axis=1); perpendicular = np.zeros_like(projected)
    visible = length > np.finfo(float).eps*span
    perpendicular[visible] = np.c_[-projected[visible, 1], projected[visible, 0]]/length[visible, None]
    offsets = perpendicular@basis[:2]*half_width_pixels*span/size
    corners = np.stack([vertices+offsets, ends+offsets, ends-offsets, vertices-offsets], axis=1)
    previous = np.roll(offsets, 1, axis=0)
    triangles = np.concatenate([corners[:, [[0, 1, 2], [0, 2, 3]], :].reshape(-1, 3, 3),
                                np.stack([vertices, vertices+previous, vertices+offsets], axis=1),
                                np.stack([vertices, vertices-previous, vertices-offsets], axis=1)])
    return dict(triangles=triangles, normals=np.broadcast_to(basis[2], triangles.shape),
                colors=np.broadcast_to(rgb(color)*.62, triangles.shape), bias=4e-6, unlit=True)


class Renderer:
    def __init__(self):
        BUILD.mkdir(parents=True, exist_ok=True); source = BUILD/'transparent.cpp'; library = BUILD/'transparent.so'
        if not source.exists() or source.read_text() != SOURCE: source.write_text(SOURCE)
        if not library.exists() or library.stat().st_mtime < source.stat().st_mtime:
            compiler = shutil.which('clang++') or shutil.which('g++')
            if compiler is None: raise RuntimeError('the transparent renderer needs clang++ or g++')
            subprocess.run([compiler, '-O3', '-std=c++17', '-shared', '-fPIC', str(source), '-o', str(library)], check=True)
        self.raster = ctypes.CDLL(str(library)).raster_layers
        self.raster.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        self.raster.restype = None

    def render(self, parts, cam, size=1000):
        w = h = size*2; focus, basis, span = cam; triangles, colors, alphas = [], [], []
        light = basis[2]*.72+basis[1]*.5-basis[0]*.22; light /= np.linalg.norm(light)
        fill = basis[2]*.25+basis[1]*.3+basis[0]*.65; fill /= np.linalg.norm(fill)
        opaque_palette = np.array([[42, 145, 210], [205, 88, 97], [204, 83, 96], [199, 68, 82]], float)
        for mesh, rotation, point in parts:
            p = (mesh['triangles']@rotation.T+point-focus)@basis.T
            p[..., 0] = w/2+p[..., 0]*h/span; p[..., 1] = h/2-p[..., 1]*h/span; p[..., 2] += mesh['bias']
            normal = mesh['normals']@rotation.T
            shade = .64+.27*np.maximum(0, normal@light)+.12*np.maximum(0, normal@fill)
            if mesh.get('unlit'): shade[:] = 1
            color = mesh['colors']*shade[..., None]; alpha = mesh.get('opacity', 1.)
            # The pale blue (and red) material stays solid: its opacity only lightens it.
            if any(np.allclose(mesh['colors'][0, 0], v, atol=.1) for v in opaque_palette):
                color = color*alpha+255*(1-alpha); alpha = 1.
            triangles.append(p.reshape(-1, 9)); colors.append(color.reshape(-1, 9)); alphas.extend([alpha]*len(p))
        p = np.ascontiguousarray(np.concatenate(triangles), dtype=np.float32)
        c = np.ascontiguousarray(np.concatenate(colors), dtype=np.float32)
        a = np.ascontiguousarray(alphas, dtype=np.float32); output = np.empty((h, w, 3), dtype=np.uint8)
        self.raster(p.ctypes.data, c.ctypes.data, len(p), w, h, a.ctypes.data, output.ctypes.data)
        return Image.fromarray(output).resize((size, size), Image.Resampling.LANCZOS)
