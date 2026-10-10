"""All-pose coverage objective and deterministic finite-difference descent."""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import time
import numpy as np
from whole_search.common import tangent_frame,legal_direction
from whole_search.reuse_first import registered
from .coverage import DemandGrid,covered_measure
from .contacts import ContactBoundaries


@dataclass
class Evaluation:
    layout: object
    coverage: np.ndarray
    details: list
    volume_cm3: float
    volume_epoch: int


class CoverageObjective:
    def __init__(self,model,initial,quadrature_level=1,contact_depth=1):
        self.model=model;self.grids=[DemandGrid.from_task(t,quadrature_level) for t in model.tasks]
        self.geometry=ContactBoundaries(model,contact_depth);self.cache=OrderedDict();self.capacity_cache=OrderedDict()
        self.initial=initial.copy();self.evaluations=0;self.lp_calls=0;self.seconds=0.
        model.contact_delta.commit(initial)
        model.volume_guidance([initial])
        self.reference_volume=max(model.volume_delta.estimate(initial),1.)

    def evaluate(self,layout):
        key=layout.key()
        if key in self.cache:
            result=self.cache[key];self.refresh(result);return result
        started=time.monotonic();model=self.model;values=[];details=[]
        for owner in layout.active:
            placed=model.native[layout.hosts[owner]]@layout.placements[owner]
            if not np.allclose(placed[:3,:3],model.native[owner,:3,:3],atol=1e-10,rtol=0) or abs(placed[2,3]-model.native[owner,2,3])>1e-9:
                raise ValueError('Continuous operation changed task orientation/height')
            if layout.directions[owner]@model.floor_normal(layout,owner)<-1e-12:
                raise ValueError('Continuous direction crosses the ground')
            rays,geometry=self.geometry.supply(layout,owner)
            signature=(owner,hashlib.sha256(rays.tobytes()).digest())
            if signature not in self.capacity_cache:
                measured=covered_measure(rays,self.grids[owner]);self.lp_calls+=measured['lp_calls']
                self.capacity_cache[signature]=measured
                if len(self.capacity_cache)>128:self.capacity_cache.popitem(last=False)
            measured=self.capacity_cache[signature]
            values.append(measured['coverage'])
            details.append(dict(pose=model.poses[owner],coverage=measured['coverage'],
                interval_lp_calls=measured['lp_calls'],maximum_interval_uncertainty=measured['maximum_interval_uncertainty'],
                gravity_supported=measured['gravity_supported'],quadrature=measured['quadrature'],**geometry))
        guidance=model.volume_guidance([layout]);volume=guidance.estimate(layout)
        result=Evaluation(layout.copy(),np.asarray(values),details,volume,guidance.epoch)
        self.cache[key]=result
        if len(self.cache)>96:self.cache.popitem(last=False)
        self.evaluations+=1;self.seconds+=time.monotonic()-started
        return result

    def refresh(self,result):
        guidance=self.model.volume_guidance([result.layout])
        if result.volume_epoch!=guidance.epoch:
            result.volume_cm3=guidance.estimate(result.layout);result.volume_epoch=guidance.epoch

    @staticmethod
    def coverage_loss(result):return float(np.mean(1-result.coverage))

    @staticmethod
    def covered(result):return bool(np.min(result.coverage)>=1-2e-6)

    def value(self,result,phase):
        self.refresh(result)
        if phase=='coverage':
            return self.coverage_loss(result)+1e-5*result.volume_cm3/self.reference_volume
        return result.volume_cm3/self.reference_volume

    def acceptable(self,current,trial,phase):
        self.refresh(current);self.refresh(trial)
        if phase=='volume' and not self.covered(trial):return False
        return self.value(trial,phase)<self.value(current,phase)-1e-9

    def better(self,a,b):
        self.refresh(a);self.refresh(b)
        if self.covered(a)!=self.covered(b):return self.covered(a)
        if self.covered(a):return a.volume_cm3<b.volume_cm3-1e-5
        return self.value(a,'coverage')<self.value(b,'coverage')-1e-9


def descent(objective,current,kind,difference,step,backtracks=4):
    """Projected block gradient; backtracking evaluates ONE descent direction.

    All original poses contribute to every difference. There is no hardest
    demand selection, operation-candidate pool or random design sampling.
    Translation coordinates are scaled by object extent.
    """
    model=objective.model;layout=current.layout
    phase='volume' if objective.covered(current) else 'coverage'
    coordinates=[]
    for owner in layout.active:
        if kind=='translation' and registered(layout,owner):continue
        normal=(layout.directions[owner] if kind=='direction' else model.floor_normal(layout,owner))
        frame=tangent_frame(normal)
        for axis in range(2):coordinates.append((owner,frame[:,axis]))

    def perturb(vector):
        trial=layout.copy()
        for value,(owner,tangent) in zip(vector,coordinates):
            if kind=='direction':
                trial.directions[owner]+=value*tangent
            else:trial.placements[owner,:3,3]+=model.extent*value*tangent
        if kind=='direction':
            for owner in trial.active:
                trial.directions[owner]=legal_direction(trial.directions[owner],model.floor_normal(trial,owner))
        return trial

    gradient=np.zeros(len(coordinates));probes=[]
    for index,(owner,axis) in enumerate(coordinates):
        unit=np.zeros(len(coordinates));unit[index]=difference
        left=objective.evaluate(perturb(-unit));right=objective.evaluate(perturb(unit))
        gradient[index]=(objective.value(right,phase)-objective.value(left,phase))/(2*difference)
        probes.append(dict(pose=model.poses[owner],coordinate=index,
                           minus_coverage=left.coverage.tolist(),plus_coverage=right.coverage.tolist()))
    norm=float(np.linalg.norm(gradient));trials=[];best=current
    if norm>1e-12:
        direction=-gradient/norm
        for attempt in range(backtracks):
            amount=step/(2**attempt)
            trial=objective.evaluate(perturb(amount*direction))
            accepted=objective.acceptable(current,trial,phase)
            trials.append(dict(step=amount,accepted=accepted,coverage=trial.coverage.tolist(),
                               estimated_material_cm3=trial.volume_cm3))
            if accepted:best=trial;break
    return best,dict(operation=kind,accepted=best is not current,phase=phase,
                     gradient_method='central finite differences of integrated all-pose coverage / material',
                     gradient_norm=norm,gradient=gradient.tolist(),difference=difference,
                     all_pose_gain_and_loss_included=True,probes=probes,line_search=trials,
                     random_design_sampling=False)
