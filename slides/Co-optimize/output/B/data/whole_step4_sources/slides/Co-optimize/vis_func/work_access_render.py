"""Depth-rendered round work-volume views, ported from the pose-set test wheel."""
import sys,importlib.util
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
from co_common import *
from collections import OrderedDict
from types import SimpleNamespace
from PIL import Image,ImageDraw
from scipy.ndimage import binary_erosion
CO=HERE
REGION_RASTER_CACHE=OrderedDict()
def original_raster():
    # Import the existing video's renderer without loading/running its local
    # geometry example. Its rasterizer is independent of that geometry module.
    path = CO/'operation_demo/juxtapose/code/render.py'
    # Keep the module name used by the original demo's Numba disk cache.
    spec = importlib.util.spec_from_file_location('juxtapose_demo_render', path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get('geometry')
    sys.modules['geometry'] = SimpleNamespace()
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            del sys.modules['geometry']
        else:
            sys.modules['geometry'] = previous
    return module.raster

RASTER=original_raster()

class Panel:
    def __init__(self, bounds, size=(360,320), elevation=35.26438968, azimuth=-45, points=None):
        self.width,self.height = size
        elevation,azimuth = np.radians([elevation,azimuth])
        self.camera = np.array([np.cos(elevation)*np.cos(azimuth),
                               np.cos(elevation)*np.sin(azimuth),np.sin(elevation)])
        self.right = np.array([-np.sin(azimuth),np.cos(azimuth),0.])
        self.up = np.cross(self.camera,self.right)
        self.light = np.array([1.,-1.,1.8]); self.light /= np.linalg.norm(self.light)
        if points is None:
            points = np.array([[x,y,z] for x in bounds[:,0] for y in bounds[:,1] for z in bounds[:,2]])
        basis = np.column_stack([self.right,self.up,self.camera])
        projected = points @ basis
        self.center = ((projected.min(0)+projected.max(0))/2) @ basis.T
        self.scale = .91*min(self.width/np.ptp(projected[:,0]),
                             self.height/np.ptp(projected[:,1]))

    def render(self, meshes):
        image = np.full((self.height,self.width,3),255,np.uint8)
        depth = np.full((self.height,self.width),-np.inf)
        triangles,colors = [],[]
        for mesh,color in meshes:
            if not len(mesh.faces):
                continue
            normals = mesh.face_normals
            keep = normals @ self.camera > 0
            ts,ns = mesh.triangles[keep],normals[keep]
            q = ts-self.center
            projected = np.stack([q @ self.right*self.scale+self.width/2,
                                   -q @ self.up*self.scale+self.height/2,
                                   ts @ self.camera],axis=2)
            shade = .60+.40*np.maximum(ns @ self.light,0)
            rgb = np.array(ImageDraw.ImageColor.getrgb(color))
            triangles.append(projected);colors.append((shade[:,None]*rgb).astype(np.uint8))
        if triangles:
            RASTER(np.concatenate(triangles),np.concatenate(colors),image,depth,0,0,self.width,self.height)
        return Image.fromarray(image)

def render_scene(panel,layers):
    def raster(parts):
        pixels=np.full((panel.height,panel.width,3),255,np.uint8)
        depth=np.full((panel.height,panel.width),-np.inf)
        triangles=[];colors=[]
        for mesh,color in parts:
            if not len(mesh.faces):continue
            normals=mesh.face_normals;keep=normals @ panel.camera>0
            ts,ns=mesh.triangles[keep],normals[keep];q=ts-panel.center
            projected=np.stack([q @ panel.right*panel.scale+panel.width/2,
                -q @ panel.up*panel.scale+panel.height/2,ts @ panel.camera],axis=2)
            shade=.60+.40*np.maximum(ns @ panel.light,0)
            if mesh.metadata.get('uniform_display_color'):shade=np.ones(len(ns))
            rgb=np.array(ImageDraw.ImageColor.getrgb(color))
            triangles.append(projected);colors.append((shade[:,None]*rgb).astype(np.uint8))
        if triangles:RASTER(np.concatenate(triangles),np.concatenate(colors),pixels,depth,0,0,panel.width,panel.height)
        return pixels,depth
    image,depth=raster([(layer[0],layer[1]) for layer in layers if len(layer)==2])
    for mesh,color,alpha in [layer for layer in layers if len(layer)==3]:
        key=(mesh.metadata.get('work_region_key'),color,panel.width,panel.height,
             panel.camera.tobytes(),panel.center.tobytes(),float(panel.scale))
        if key not in REGION_RASTER_CACHE:
            REGION_RASTER_CACHE[key]=raster([(mesh,color)])
            while len(REGION_RASTER_CACHE)>8:REGION_RASTER_CACHE.popitem(last=False)
        REGION_RASTER_CACHE.move_to_end(key)
        overlay,z=REGION_RASTER_CACHE[key]
        visible=np.isfinite(z)&(z>=depth-1e-9)
        image[visible]=np.rint((1-alpha)*image[visible]+alpha*overlay[visible]).astype(np.uint8)
        outline=visible&~binary_erosion(visible)
        image[outline]=np.rint(.60*image[outline]+.40*np.array(ImageDraw.ImageColor.getrgb(color))).astype(np.uint8)
    return Image.fromarray(image)
