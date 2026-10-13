"""Share body/sweep primitives across identical relative placements.

Working cones remain pose-specific. Lazy work predicates query only points
not already removed by body/exit, with explicit known masks for later changes.
Returned lock vectors equal the original full predicates; no excluded region
or reaction rule changes. This module is not used by frozen v3 experiments.
"""
from collections import OrderedDict
import time
import numpy as np
from whole_search.common import C


class GeometryCache:
    def __init__(self,model):
        self.model=model;self.bodies=OrderedDict();self.sweeps=OrderedDict();self.work=OrderedDict()
        self.results=OrderedDict();self.seconds=0.;self.work_points=0;self.sweep_queries=0

    @staticmethod
    def bound(cache,size):
        while len(cache)>size:cache.popitem(last=False)

    def locks(self,layout,owner,blocker):
        start=time.monotonic();m=self.model
        relative=np.linalg.inv(layout.placements[blocker])@layout.placements[owner]
        direction=layout.placements[blocker,:3,:3].T@layout.directions[blocker]
        placement_key=np.round(relative,11).tobytes();sweep_key=placement_key,np.round(direction,11).tobytes()
        key=blocker,sweep_key
        if key in self.results:
            self.results.move_to_end(key);return self.results[key]
        if placement_key not in self.bodies:
            surface=C.transform_points(m.points,relative)
            origins=surface+m.epsilon*(m.point_normals@relative[:3,:3].T)
            self.bodies[placement_key]=surface,origins,m.ray.contains_points(origins)
            self.bound(self.bodies,64)
        else:self.bodies.move_to_end(placement_key)
        surface,origins,body=self.bodies[placement_key]
        if sweep_key not in self.sweeps:
            swept=body.copy();indices=np.flatnonzero(~body)
            if len(indices):
                points,rays,_=m.ray.intersects_location(origins[indices],np.tile(-direction,(len(indices),1)),multiple_hits=False)
                distance=(origins[indices[rays]]-points)@direction
                swept[indices[rays[(distance>=-m.epsilon)&(distance<=m.length)]]]=True
            self.sweeps[sweep_key]=swept;self.sweep_queries+=1;self.bound(self.sweeps,512)
        else:self.sweeps.move_to_end(sweep_key)
        swept=self.sweeps[sweep_key]
        work_key=blocker,placement_key
        if work_key not in self.work:
            self.work[work_key]=np.zeros(len(m.points),bool),np.zeros(len(m.points),bool)
            self.bound(self.work,512)
        else:self.work.move_to_end(work_key)
        values,known=self.work[work_key];indices=np.flatnonzero(~swept&~known)
        if len(indices):
            values[indices]=m.work_rays[blocker].contains_points(surface[indices]);known[indices]=True
            self.work_points+=len(indices)
        result=swept|values;self.results[key]=result;self.bound(self.results,512)
        self.seconds+=time.monotonic()-start
        return result
