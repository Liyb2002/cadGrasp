"""Immutable object distance grid for continuous direction optimization.

The grid is guidance, never collision or force acceptance. Trilinear extension
makes a fixed geometry field continuous; exact original meshes remain intact.
"""
import sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import _bootstrap
from co_common import np,trimesh,HERE
from scipy.ndimage import map_coordinates

_cache={}


class ObjectDistanceField:
    def __new__(cls,mesh,resolution=96):
        digest=hashlib.sha256()
        digest.update(np.asarray(mesh.vertices,dtype=np.float64).tobytes())
        digest.update(np.asarray(mesh.faces,dtype=np.int64).tobytes())
        digest.update(str(resolution).encode())
        key=digest.hexdigest()
        if key not in _cache:
            instance=super().__new__(cls);instance.key=key;_cache[key]=instance
        return _cache[key]

    def __init__(self,mesh,resolution=96):
        if hasattr(self,'values'):return
        extent=float(mesh.extents.max())
        self.spacing=extent/resolution
        padding=extent*.04
        self.lower=mesh.bounds[0]-padding
        self.shape=np.ceil((mesh.extents+2*padding)/self.spacing).astype(int)+1
        self.upper=self.lower+self.spacing*(self.shape-1)
        folder=HERE/'data/cache/physics_guided';folder.mkdir(parents=True,exist_ok=True)
        self.path=folder/(self.key+'.npz')
        if self.path.exists():
            with np.load(self.path) as z:
                if not np.array_equal(z['shape'],self.shape):raise RuntimeError('distance cache shape mismatch')
                self.values=z['values'].copy()
            return
        total=int(np.prod(self.shape));values=np.empty(total)
        print('BUILD OBJECT FIELD',self.shape.tolist(),'points',total,flush=True)
        for first in range(0,total,2048):
            indices=np.array(np.unravel_index(np.arange(first,min(first+2048,total)),self.shape)).T
            points=self.lower+self.spacing*indices
            values[first:first+len(points)]=trimesh.proximity.signed_distance(mesh,points)
        if not np.isfinite(values).all():raise RuntimeError('unresolved object distance grid')
        self.values=values.reshape(self.shape)
        # Write an immutable cache atomically, including multi-process use.
        import os
        temporary=folder/(self.key+'.'+str(os.getpid())+'.npz')
        np.savez_compressed(temporary,values=self.values,shape=self.shape,lower=self.lower,spacing=self.spacing)
        temporary.replace(self.path)

    def sample(self,points):
        points=np.asarray(points)
        clipped=np.clip(points,self.lower,self.upper)
        coordinates=((clipped-self.lower)/self.spacing).T
        values=map_coordinates(self.values,coordinates,order=1,mode='nearest',prefilter=False)
        # Continuous negative extension outside the grid bounding box.
        return values-np.linalg.norm(points-clipped,axis=1)
