"""Bounded legal translations for unresolved exported mesh conditioning.

Only explicit Juxtapose seats move. Every proposal first replays the original
ALL sampled loads, then uses the unchanged actual worker. These translations
are marked samples, never credited as a descent gradient when the loss is zero.
"""
import time
import numpy as np
from whole_search.model import Model
from whole_search.fast_search import FastReuseSearch
from whole_search.reuse_first import registered,juxtaposed_count
from whole_search.search import failed_count
from .descent_fast_gradient import CachedGradientModel,DescentGradientSearch


class SeparationGradientSearch(DescentGradientSearch):
    def save(self,current,mode,began,initialization,**extra):
        from co_common import save
        m=self.model;self.search_only=True
        sampled=FastReuseSearch.save(self,current,mode,began,initialization,**extra)
        layout=current['layout'];movable=[k for k in layout.active if not registered(layout,k)]
        proposals=[]
        for fraction,kind in [(1/1024,'staggered'),(1/1024,'common_x'),(1/512,'staggered'),(1/512,'common_y'),
                              (1/256,'staggered'),(1/2048,'staggered')]:
            trial=layout.copy();shifts={}
            for rank,k in enumerate(movable):
                if kind=='staggered':
                    angle=2*np.pi*(rank+1)*(np.sqrt(5)-1)/2
                    world=m.extent*fraction*np.array([np.cos(angle),np.sin(angle),0.])
                else:world=m.extent*fraction*np.array([float(kind=='common_x'),float(kind=='common_y'),0.])
                trial.placements[k,:3,3]+=m.native[layout.hosts[k],:3,:3].T@world
                shifts[m.poses[k]]=world.tolist()
            detail=dict(operation='translation-conditioning-sample',style=kind,body_fraction=fraction,
                        world_horizontal_shifts_m=shifts,gradient_step=False)
            checked=m.evaluate(trial)
            proposals.append((checked,detail))
        save(self.out/'translation_conditioning_screen.json',
             [dict(detail=detail,counts=state['counts'],original_sampled_passed=failed_count(state)==0)
              for state,detail in proposals])
        candidates=[r for r in proposals if failed_count(r[0])==0]
        attempts=[];actual=None;selected=None;start=time.monotonic()
        for state,detail in candidates[:4]:
            row=dict(sampled_serial=state['serial'],passed=False,detail=detail);clock=time.monotonic()
            try:
                result=m.exact(state['layout']);work=result.get('actual_work_surface_checks',[])
                row.update(counts=result['counts'],volume_cm3=result['volume_cm3'])
                if failed_count(result)==0 and len(work)==len(layout.active) and all(r['passed'] for r in work):
                    actual=result;selected=state;row['passed']=True
            except (RuntimeError,ValueError,AssertionError) as error:row['error']=str(error)
            row['seconds']=time.monotonic()-clock;attempts.append(row);save(self.out/'final_validation_attempts.json',attempts)
            if actual is not None:break
        if actual is None:raise RuntimeError('Legal translation conditioning unresolved; physical acceptance unchanged')
        self.process_snapshot(actual,'final_verified_translation',attempts[-1])
        self.process_rows[-1]['geometry_verified']=True
        self.process_rows[-1]['actual_counts_verified']=actual['counts'];save(self.out/'process.json',self.process_rows)
        extra=dict(extra);extra['passed']=True
        return Model.save(m,actual,self.out,dict(strategy=mode,policy='bounded-original-load-preserving-translation-conditioning',
            initialization=initialization,search_seconds=sampled['search_seconds'],conditioning_screen_seconds=time.monotonic()-began-sampled['search_seconds']-(time.monotonic()-start),
            validation_seconds=time.monotonic()-start,seconds=time.monotonic()-began,
            sample_evaluations=m.sample_calls,exact_evaluations_during_search=0,timing=m.timing,events=self.events,
            final_validation_attempts=attempts,translation_conditioning_gradient_claimed=False,
            rotating_reuse_pose_count=sum(registered(actual['layout'],k) for k in layout.active),
            juxtaposed_pose_count=juxtaposed_count(actual['layout']),conservative_contact_grid_unmet_loads=failed_count(selected),
            final_acceptance_run=True,**extra))
