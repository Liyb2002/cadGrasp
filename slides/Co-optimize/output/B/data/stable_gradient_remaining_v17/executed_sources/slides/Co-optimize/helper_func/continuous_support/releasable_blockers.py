"""Rank external blockers using rays already permitted by the owner's exit.

Relocating another pose cannot release an owner's own body/work/sweep lock.
The polar score is the fixed-cone, one-ray NNLS improvement lower bound, not
an area pressure capacity or a claim of the complete discrete jump's benefit.
"""
from collections import OrderedDict
import time
import numpy as np
from .expanded_gradient import ExpandedGradientModel
from .projection import project_demands


class ReleasableBlockerModel(ExpandedGradientModel):
    def useful_coordinates(self,current):
        layout=current['layout'];key=layout.key(),self.quadrature_epoch
        if key in self.coordinate_cache:
            ordered,scores,weights=self.coordinate_cache[key]
            self.jump_ray_weights=weights
            return ordered,scores
        start=time.monotonic();state=self.contact_delta.state(layout)
        scores={k:0. for k in layout.active};weights_by_owner={}
        for owner in layout.active:
            full,_=self.supply_at_points(owner,state.available[owner])
            needs,w,scale=self.demand_screen.demands[owner]
            nodes=[(needs,.5*w/scale**2)]
            if owner in self.adaptive_demands:
                targets,w,_=self.adaptive_demands[owner]
                nodes.append((targets,.5*w/self.original_scales[owner]**2))
            gain=np.zeros(len(self.points));norm2=np.sum(self.point_rays[owner]**2,axis=1)
            for targets,w in nodes:
                residual=project_demands(full,targets)['residuals']
                polar=np.maximum(-residual@self.point_rays[owner].T,0.)
                gain+=((polar**2)/norm2).T@w
            material=(state.coverage_counts[owner]>0)&np.isin(self.sources,self.allowed[owner])
            material &= ~state.locks[owner,owner]
            weights_by_owner[owner]=gain*self.point_areas*material
            for blocker in layout.active:
                relevant=state.locks[owner,blocker]&material
                scores[blocker]+=float((weights_by_owner[owner][relevant]/
                                       state.lock_counts[owner][relevant]).sum())
        value=sorted(layout.active,key=lambda k:-scores[k])
        self.coordinate_cache[key]=value,scores,weights_by_owner
        self.jump_ray_weights=weights_by_owner
        while len(self.coordinate_cache)>32:self.coordinate_cache.popitem(last=False)
        self.coordinate_seconds+=time.monotonic()-start
        return value,scores
