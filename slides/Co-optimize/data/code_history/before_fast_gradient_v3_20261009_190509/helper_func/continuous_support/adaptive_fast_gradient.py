"""Positive all-demand strata repair quadrature blind spots near feasibility.

Strata are fixed during each local finite-difference/line-search block. They
approximate the original demand measure, retaining BOTH currently covered
and uncovered regions and their original masses; not a hardest-load loss.
"""
import hashlib
import time
from collections import OrderedDict
import numpy as np
from whole_search.common import C
from .projection import project_demands
from .fast_gradient import GradientModel,GradientSearch


def demand_strata(targets,mask,passed_nodes=32,failed_nodes=64):
    ids=[];weights=[]
    for region,budget in [(np.flatnonzero(mask),passed_nodes),(np.flatnonzero(~mask),failed_nodes)]:
        if not len(region):continue
        # Resolve a rare failure region completely. A large region remains a
        # deterministic equal-mass quadrature over the original Sobol demands.
        count=len(region) if len(region)<=256 and budget==failed_nodes else min(budget,len(region))
        selected=region[np.minimum(len(region)-1,((np.arange(count)+.5)*len(region)/count).astype(int))]
        ids.extend(selected);weights.extend([len(region)/len(mask)/count]*count)
    return np.asarray(targets)[np.asarray(ids,int)],np.asarray(weights)


class AdaptiveGradientModel(GradientModel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.adaptive_demands={};self.adaptive_cache=OrderedDict();self.quadrature_epoch=0
        self.adaptive_seconds=0.;self.original_scales={k:max(1.,float(np.linalg.norm(t.targets,axis=1).max()))
                                                      for k,t in enumerate(self.tasks)}

    def set_working_measure(self,current):
        self.quadrature_epoch+=1;self.adaptive_cache.clear()
        for k,mask in current['masks'].items():
            targets,weights=demand_strata(C.U.target(self.tasks[k].targets),mask)
            self.adaptive_demands[k]=targets,weights

    def proxy(self,layout,targets=()):
        fixed=super().proxy(layout,targets)
        if not self.adaptive_demands:return fixed
        began=time.monotonic();flags=self.contact_delta.state(layout).available;losses=[]
        for k in layout.active:
            key=k,hashlib.sha256(flags[k].tobytes()).digest()
            if key not in self.adaptive_cache:
                rays,_=self.supply_at_points(k,flags[k]);needs,weights=self.adaptive_demands[k]
                projection=project_demands(rays,needs)
                self.adaptive_cache[key]=float(weights@projection['losses']/self.original_scales[k]**2)
                while len(self.adaptive_cache)>512:self.adaptive_cache.popitem(last=False)
            losses.append(self.adaptive_cache[key])
        combined=(np.asarray(fixed['residual_loss'])+np.asarray(losses))/2
        # Roundoff below the checked projection residual cannot distinguish
        # geometrically identical zero-distance cones or choose a good host.
        combined[combined<1e-22]=0.
        self.adaptive_seconds+=time.monotonic()-began
        return dict(fixed,loss=float(combined.mean()),sum_loss=float(combined.sum()),
                    residual_loss=combined.tolist(),quadrature_epoch=self.quadrature_epoch,
                    quadrature='half fixed positive boundary rule; half original measure strata',
                    adaptive_nodes={str(k):len(self.adaptive_demands[k][0]) for k in layout.active},
                    covered_and_uncovered_measure_preserved=True)


class AdaptiveGradientSearch(GradientSearch):
    def targets(self,current):
        self.model.set_working_measure(current)
        return super().targets(current)

    def polish_volume(self,current,rounds=2):
        self.model.set_working_measure(current)
        current=super().polish_volume(current,rounds)
        self.model.timing['adaptive_demand_projection_s']=self.model.adaptive_seconds
        return current
