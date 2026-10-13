"""Compiled triangle-distance guidance; distinct caches and exact acceptance unchanged."""
import hashlib
from importlib.metadata import version
import os
from pathlib import Path
import igl
from physics_guided_field import ObjectDistanceField as ReferenceField
from co_common import np,HERE

_cache={}
BACKEND='libigl pseudonormal signed triangle distance'
LIBRARY_VERSION=version('libigl')


class FastObjectDistanceField(ReferenceField):
    def __new__(cls,mesh,resolution=96):
        digest=hashlib.sha256()
        digest.update(np.asarray(mesh.vertices,dtype=np.float64).tobytes())
        digest.update(np.asarray(mesh.faces,dtype=np.int64).tobytes())
        digest.update(f'libigl-{LIBRARY_VERSION}-pseudonormal-v1-{resolution}'.encode())
        key=digest.hexdigest()
        if key not in _cache:
            instance=object.__new__(cls);instance.key=key;_cache[key]=instance
        return _cache[key]

    def __init__(self,mesh,resolution=96):
        if hasattr(self,'values'):return
        if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume<=0:
            raise RuntimeError('Compiled signed field requires an outward oriented closed mesh')
        extent=float(mesh.extents.max());self.spacing=extent/resolution
        padding=extent*.04;self.lower=mesh.bounds[0]-padding
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
        print('BUILD COMPILED OBJECT FIELD',self.shape.tolist(),'points',total,flush=True)
        vertices=np.asarray(mesh.vertices,dtype=np.float64);faces=np.asarray(mesh.faces,dtype=np.int64)
        for first in range(0,total,32768):
            indices=np.array(np.unravel_index(np.arange(first,min(first+32768,total)),self.shape)).T
            points=self.lower+self.spacing*indices
            distances,_,_,_=igl.signed_distance(points,vertices,faces,igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)
            values[first:first+len(points)]=-distances  # Same positive-interior convention.
        if not np.isfinite(values).all():raise RuntimeError('unresolved compiled object field')
        self.values=values.reshape(self.shape)
        temporary=folder/(self.key+'.'+str(os.getpid())+'.npz')
        np.savez_compressed(temporary,values=self.values,shape=self.shape,lower=self.lower,spacing=self.spacing)
        temporary.replace(self.path)
