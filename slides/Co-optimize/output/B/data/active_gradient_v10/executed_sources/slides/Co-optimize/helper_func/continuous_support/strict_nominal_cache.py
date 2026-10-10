"""Match the physical sweep's leading-face sign on coincident body surfaces.

An epsilon-offset ray misses a positive leading extrusion thinner than that
epsilon. On shared placements, determine its sign with the SAME normalized
long-double area arithmetic used by the original nominal sweep. No physical
source or tolerance changes. Work and other-geometry predicates stay intact.
"""
from collections import OrderedDict
import numpy as np
from .geometry_cache import GeometryCache


class StrictNominalCache(GeometryCache):
    def __init__(self,model):
        super().__init__(model);self.leading_cache=OrderedDict();self.leading_points_removed=0

    def leading(self,direction):
        key=np.asarray(direction).tobytes()
        if key not in self.leading_cache:
            m=self.model;displacement=m.length*np.asarray(direction)
            origin=np.asarray(m.mesh.bounds).mean(axis=0)
            scale=max(float(m.mesh.extents.max()),float(np.linalg.norm(displacement)))
            vertices=(np.asarray(m.mesh.vertices)-origin)/scale;delta=displacement/scale
            triangles=vertices[np.asarray(m.mesh.faces)].astype(np.longdouble)
            area_normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
            self.leading_cache[key]=(area_normals@delta.astype(np.longdouble)>0)[m.sources]
            self.bound(self.leading_cache,512)
        return self.leading_cache[key]

    def locks(self,layout,owner,blocker):
        result=super().locks(layout,owner,blocker)
        if np.array_equal(layout.placements[owner],layout.placements[blocker]):
            direction=layout.placements[blocker,:3,:3].T@layout.directions[blocker]
            leading=self.leading(direction)
            self.leading_points_removed+=int(np.count_nonzero(leading&~result))
            return result|leading
        return result
