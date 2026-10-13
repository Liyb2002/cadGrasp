"""Static illustrations of surface centers and their circular contacts."""
from __future__ import annotations
from step1.needs import COORD
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

PAPER = (255, 255, 255)
GREY = np.array([177., 185., 191.])
GREEN = np.array([137., 181., 129.])
ORANGE = np.array([245., 139., 37.])
FLOOR = np.array([226., 228., 229.])
INK = '#202b30'


def axes(view):
    view = np.array(view, dtype=float, copy=True)
    view /= np.linalg.norm(view)
    up = np.array([0., 0., 1.])
    if abs(view@up) > .95:
        up = np.array([0., 1., 0.])
    right = np.cross(up, view)
    right /= np.linalg.norm(right)
    return np.array([right, np.cross(view, right), view])


def project(points, focus, basis, width, size):
    p = (points-focus)@basis.T
    return np.stack([size/2+p[..., 0]*size/width,
                     size/2-p[..., 1]*size/width, p[..., 2]], axis=-1)


def raster(triangles, colors, focus, basis, width, size, overlay=None, unlit=()):
    """Depth-tested actual triangles; overlay gets only a small depth bias."""
    pixels = np.full((size, size, 3), PAPER, dtype=np.uint8)
    depth = np.full((size, size), -np.inf)
    face_id = np.full((size, size), -1, dtype=np.int32)
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-30)
    light = .7*basis[2]+.3*basis[1]-.25*basis[0]
    light /= np.linalg.norm(light)
    shade = .64+.36*np.maximum(0, normals@light)
    lit = np.clip(colors*shade[:, None], 0, 255).astype(np.uint8)
    lit[list(unlit)] = colors[list(unlit)].astype(np.uint8)
    points = project(triangles, focus, basis, width, size)
    if overlay is not None:
        points[overlay, :, 2] += width*1e-5
    low = np.maximum(0, np.floor(points[:, :, :2].min(axis=1)).astype(int))
    high = np.minimum(size-1, np.ceil(points[:, :, :2].max(axis=1)).astype(int))
    candidates = np.flatnonzero((high >= low).all(axis=1))
    for k in candidates:
        xmin, ymin = low[k]
        xmax, ymax = high[k]
        a, b, c = points[k]
        det = (b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(det) < 1e-12:
            continue
        yy, xx = np.mgrid[ymin:ymax+1, xmin:xmax+1]
        xx, yy = xx+.5, yy+.5
        u = ((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/det
        v = ((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/det
        w = 1-u-v
        z = u*a[2]+v*b[2]+w*c[2]
        old = depth[ymin:ymax+1, xmin:xmax+1]
        hit = (u >= -1e-10) & (v >= -1e-10) & (w >= -1e-10) & (z > old)
        old[hit] = z[hit]
        pixels[ymin:ymax+1, xmin:xmax+1][hit] = lit[k]
        face_id[ymin:ymax+1, xmin:xmax+1][hit] = k
    return Image.fromarray(pixels), face_id


def font(size):
    return ImageFont.truetype(font_manager.findfont('DejaVu Sans'), size)


def floor_triangles(domain):
    low,high=domain.mesh.bounds
    margin=.13*domain.mesh.extents.max()
    x0,y0=COORD.floor(low)-margin
    x1,y1=COORD.floor(high)+margin
    corners=np.array([[x0,y0,0.],[x1,y0,0.],[x1,y1,0.],[x0,y1,0.]])
    return corners[[[0,1,2],[0,2,3]]]


def overall_camera(domain,basis):
    points=np.concatenate([domain.mesh.vertices,floor_triangles(domain).reshape(-1,3)])
    coordinates=points@basis.T
    low,high=coordinates.min(axis=0),coordinates.max(axis=0)
    return ((low+high)/2)@basis,1.10*np.max(high[:2]-low[:2])


def scene(domain,data,basis,focus,width,size,selected=None,ground=True,show_patches=True,show_hidden=False,indices=None):
    if selected is not None and indices is not None:raise ValueError('Specify one contact or a set, not both')
    active=np.array([selected],int) if selected is not None else np.asarray(
        np.flatnonzero(data.valid) if indices is None else indices,int)
    active=active[data.valid[active]]
    base=domain.mesh.triangles
    colors=np.tile(GREY,(len(base),1))
    colors[domain.work_ids]=GREEN
    pieces=[data.triangles[data.offsets[i]:data.offsets[i+1]] for i in active]
    patches=np.concatenate(pieces) if pieces else np.empty((0,3,3))
    if not show_patches:patches=np.empty((0,3,3))
    floor=floor_triangles(domain) if ground else np.empty((0,3,3))
    triangles=np.concatenate([floor,base,patches])
    colors=np.concatenate([np.tile(FLOOR,(len(floor),1)),colors,np.tile(ORANGE,(len(patches),1))])
    overlay=np.arange(len(floor)+len(base),len(triangles))
    picture,ids=raster(triangles,colors,focus,basis,width,size,overlay,unlit=range(len(floor)))
    ink=ImageDraw.Draw(picture)
    centers=data.centers_m[active]
    face_ids=data.center_faces[active]
    normals=domain.mesh.face_normals[face_ids]
    clear=~domain.mesh.ray.intersects_any(centers+1e-6*normals,
                                         np.tile(basis[2],(len(centers),1))) if len(active) else np.empty(0,bool)
    xy=project(centers,focus,basis,width,size)[:,:2]
    shown=0
    for p,visible in zip(xy,clear):
        if not visible and show_hidden and ((p>=4)&(p<size-4)).all():
            r=3.5
            ink.ellipse((p[0]-r,p[1]-r,p[0]+r,p[1]+r),outline='#649cbf',width=1)
        if visible and ((p>=4)&(p<size-4)).all():
            r=3.5 if selected is None else 5.
            ink.ellipse((p[0]-r,p[1]-r,p[0]+r,p[1]+r),fill='#267fcb',outline='#fff9e7',width=1)
            shown+=1
    return picture,shown
