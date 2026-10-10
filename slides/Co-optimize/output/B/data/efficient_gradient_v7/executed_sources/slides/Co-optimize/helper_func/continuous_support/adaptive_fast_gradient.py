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
from whole_search.reuse_first import registered,juxtaposed_count
from whole_search.search import failed_count
from whole_search.model import Model
from whole_search.common import legal_direction


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
        for k,(targets,weights,scale) in enumerate(self.demand_screen.demands):
            scale=max(scale,self.original_scales[k]);self.original_scales[k]=scale
            self.demand_screen.demands[k]=(targets,weights,scale)

    def set_working_measure(self,current):
        changed=False
        for k,mask in current['masks'].items():
            signature=hashlib.sha256(mask.tobytes()).digest()
            if k in self.adaptive_demands and self.adaptive_demands[k][2]==signature:continue
            targets,weights=demand_strata(C.U.target(self.tasks[k].targets),mask)
            self.adaptive_demands[k]=targets,weights,signature;changed=True
        self.quadrature_epoch+=int(changed)

    def proxy(self,layout,targets=()):
        fixed=super().proxy(layout,targets)
        if not self.adaptive_demands:return fixed
        began=time.monotonic();flags=self.contact_delta.state(layout).available;losses=[]
        for k in layout.active:
            needs,weights,signature=self.adaptive_demands[k]
            key=k,signature,hashlib.sha256(flags[k].tobytes()).digest()
            if key not in self.adaptive_cache:
                rays,_=self.supply_at_points(k,flags[k])
                projection=project_demands(rays,needs)
                self.adaptive_cache[key]=float(weights@projection['losses']/self.original_scales[k]**2)
                while len(self.adaptive_cache)>1024:self.adaptive_cache.popitem(last=False)
            losses.append(self.adaptive_cache[key])
        combined=(np.asarray(fixed['residual_loss'])+np.asarray(losses))/2
        # Roundoff below the checked projection residual cannot distinguish
        # geometrically identical zero-distance cones or choose a good host.
        combined[combined<1e-28]=0.
        self.adaptive_seconds+=time.monotonic()-began
        return dict(fixed,loss=float(combined.mean()),sum_loss=float(combined.sum()),
                    residual_loss=combined.tolist(),quadrature_epoch=self.quadrature_epoch,
                    quadrature='half fixed positive boundary rule; half original measure strata',
                    adaptive_nodes={str(k):len(self.adaptive_demands[k][0]) for k in layout.active},
                    covered_and_uncovered_measure_preserved=True)


class AdaptiveGradientSearch(GradientSearch):
    def juxtapose_proposals(self,current,guests,targets):
        rows=super().juxtapose_proposals(current,guests,targets)
        selected=[]
        for kind,layout,detail in rows:
            k=detail['guest_index'];host=int(layout.hosts[k])
            signature=(k,host,np.round(layout.placements[k]/self.model.extent,2).tobytes(),
                       detail['direction_host_weight'],
                       tuple(current['layout'].hosts),
                       np.round(self.world_direction(current['layout'],k),2).tobytes(),
                       np.round(self.world_direction(current['layout'],self.model.poses.index(detail['host'])),2).tobytes())
            detail=dict(detail,seat_signature=hashlib.sha256(repr(signature).encode()).hexdigest())
            selected.append((kind,layout,detail))
        # Keep keys serializable in trace; actual evaluated branches alone
        # become tabu, never candidates merely omitted by the screen budget.
        return [r for r in selected if r[2]['seat_signature'] not in getattr(self,'seat_hashes',set())]

    def record(self,row):
        if row.get('phase') in ['selective_juxtapose','blocking_pose_juxtapose']:
            if not hasattr(self,'seat_hashes'):self.seat_hashes=set()
            for trial in row.get('trials',[]):
                key=trial.get('detail',{}).get('seat_signature')
                if key:self.seat_hashes.add(key)
        super().record(row)

    def targets(self,current):
        self.model.set_working_measure(current)
        return super().targets(current)

    def solve_frontier(self,current,anchor=None,insertion_guest=None):
        current=self.refine(current,3,anchor=None,phase='rotate_reuse_direction')
        for iteration in range(self.iterations):
            if failed_count(current)==0:break
            if failed_count(current)<=128:
                # A conservative contact grid can miss very small lever-arm
                # regions. Stop repeated seats; fine descent, then ORIGINAL
                # actual-contact acceptance resolves this diagnostic gap.
                current=self.fine_refine(current,rounds=2);break
            before=current;worst=self.targets(current)[1]
            guests=sorted([k for k,mask in current['masks'].items() if not mask.all()],
                key=lambda k:(k!=worst['pose_index'],int(current['masks'][k].sum())))
            current=self.rescue(current,guests[:3],anchor=None)
            if current is before:
                blockers=self.blocking_poses(current)
                if blockers:
                    chosen=[k for _,k,_ in blockers[:3]]
                    current=self.rescue(current,chosen,anchor=None,blocking_guests=chosen)
                if current is before:break
            if failed_count(current)<=128:continue
            current=self.refine(current,2,anchor=None,phase='direction_after_commit')
        if 128<failed_count(current)<=512:current=self.fine_refine(current,rounds=3)
        return current

    def polish_volume(self,current,rounds=2):
        self.model.set_working_measure(current)
        current=super().polish_volume(current,rounds)
        self.model.timing['adaptive_demand_projection_s']=self.model.adaptive_seconds
        return current

    def save(self,current,mode,began,initialization,**extra):
        # Save the complete cheap trace first. Final checks change neither
        # physical sources nor tolerances. Small legal upward direction
        # candidates avoid exact tangencies in exported geometry.
        from co_common import save
        m=self.model;self.search_only=True
        sampled=super().save(current,mode,began,initialization,**extra)
        candidates=[];screen=[]
        for degrees in [.03125,.125,.5]:
            layout=current['layout'].copy()
            for k in layout.active:
                layout.directions[k]=legal_direction(layout.directions[k]+np.tan(np.radians(degrees))*m.floor_normal(layout,k),m.floor_normal(layout,k))
            trial=m.evaluate(layout)
            screen.append(dict(degrees=degrees,counts=trial['counts']))
            if failed_count(trial)<=max(128,failed_count(current)):
                candidates.append((trial,dict(operation='direction-numerical-separation',degrees=degrees)))
        candidates.sort(key=lambda row:(failed_count(row[0]),row[0]['volume_cm3']))
        candidates=candidates[:2]+[(current,dict(operation='unperturbed'))]
        for state in reversed(list(m.complete_sampled_states.values())):
            if not any(state['layout'].key()==r[0]['layout'].key() for r in candidates):
                candidates.append((state,dict(operation='earlier_feasible_state')))
        save(self.out/'final_direction_screen.json',screen)
        attempts=[];actual=None;selected=None;start=time.monotonic()
        for candidate,detail in candidates[:4]:
            record=dict(sampled_serial=candidate['serial'],passed=False,detail=detail);clock=time.monotonic()
            try:
                checked=m.exact(candidate['layout']);work=checked.get('actual_work_surface_checks',[])
                record.update(counts=checked['counts'],volume_cm3=checked['volume_cm3'])
                if failed_count(checked)==0 and len(work)==len(candidate['layout'].active) and all(r['passed'] for r in work):
                    actual=checked;selected=candidate;record['passed']=True
            except (RuntimeError,ValueError,AssertionError) as error:record['error']=str(error)
            record['seconds']=time.monotonic()-clock;attempts.append(record)
            save(self.out/'final_validation_attempts.json',attempts)
            if actual is not None:break
        if actual is None:raise RuntimeError('Final gradient candidates unresolved; original acceptance unchanged')
        self.process_snapshot(actual,'final_verified_direction',attempts[-1])
        if getattr(self,'capture_process',False):
            self.process_rows[-1]['geometry_verified']=True
            self.process_rows[-1]['actual_counts_verified']=actual['counts']
            save(self.out/'process.json',self.process_rows)
        extra=dict(extra);extra['passed']=True
        return Model.save(m,actual,self.out,dict(strategy=mode,policy='all-demand-gradient-delta-final-original-checks',
            initialization=initialization,search_seconds=sampled['search_seconds'],validation_seconds=time.monotonic()-start,
            seconds=time.monotonic()-began,sample_evaluations=m.sample_calls,exact_evaluations_during_search=0,
            timing=m.timing,events=self.events,final_validation_attempts=attempts,
            rotating_reuse_pose_count=sum(registered(actual['layout'],k) for k in actual['layout'].active),
            juxtaposed_pose_count=juxtaposed_count(actual['layout']),
            conservative_contact_grid_unmet_loads=failed_count(selected),
            final_acceptance_run=True,**extra))
