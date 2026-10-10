"""Loss-aware plateau repair on shared contact geometry.

The original all-load count remains the first feasibility ranking, but an
equal count now chooses the lower all-demand distance before material. Thus
progress toward a currently uncovered region is retained before its last
original demand becomes feasible. Material descent starts after feasibility.
"""
import time
import numpy as np
from whole_search.common import legal_direction
from whole_search.model import Model
from whole_search.fast_search import FastReuseSearch
from whole_search.reuse_first import registered,juxtaposed_count
from whole_search.search import failed_count
from .cached_fast_gradient import CachedGradientModel,CachedGradientSearch


class DescentGradientSearch(CachedGradientSearch):
    def score(self,result,anchor=None):
        score=list(super().score(result,anchor))
        score.insert(2,self.model.proxy(result['layout'])['loss'] if failed_count(result) else 0.)
        return tuple(score)

    def solve_frontier(self,current,anchor=None,insertion_guest=None):
        # Do not repeatedly launch another seat merely because the contact
        # grid has a tiny uncovered boundary. First move every explicit seat.
        if 0<failed_count(current)<=512:
            movable=[k for k in current['layout'].active if not registered(current['layout'],k)]
            current=self.refine(current,4,translation_guests=movable,phase='all_movable_gradient_repair')
            if failed_count(current):current=self.fine_refine(current,rounds=4)
            if failed_count(current)==0:return current
        return super().solve_frontier(current,anchor,insertion_guest)

    def save(self,current,mode,began,initialization,**extra):
        # Save the complete cheap trace first. Final checks change neither
        # physical sources nor tolerances. Small legal upward direction
        # candidates avoid exact tangencies in exported geometry.
        from co_common import save
        m=self.model;self.search_only=True
        sampled=FastReuseSearch.save(self,current,mode,began,initialization,**extra)
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
        candidates=[(current,dict(operation='unperturbed'))]+candidates[:2]
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
