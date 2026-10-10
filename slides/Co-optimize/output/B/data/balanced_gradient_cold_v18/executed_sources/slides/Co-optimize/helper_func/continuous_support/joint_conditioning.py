"""Small combined exit/seat samples for failed final geometry conditioning.

The original all sampled demands and unchanged actual checks remain gates.
These final zero-loss perturbations are samples, not gradient descent steps.
"""
import time
import numpy as np
from whole_search.common import legal_direction
from whole_search.search import failed_count
from whole_search.reuse_first import registered, juxtaposed_count
from whole_search.fast_search import FastReuseSearch
from whole_search.model import Model
from .active_fast_gradient import ActiveGradientModel
from .progressive_gradient import ProgressiveGradientSearch


def joint_perturbation(model, layout, degrees, fraction):
    trial=layout.copy();movable=[k for k in layout.active if not registered(layout,k)]
    shifts={}
    for rank,k in enumerate(movable):
        angle=2*np.pi*(rank+1)*(np.sqrt(5)-1)/2
        world=model.extent*fraction*np.array([np.cos(angle),np.sin(angle),0.])
        trial.placements[k,:3,3]+=model.native[trial.hosts[k],:3,:3].T@world
        shifts[model.poses[k]]=world.tolist()
    for k in layout.active:
        n=model.floor_normal(trial,k)
        trial.directions[k]=legal_direction(trial.directions[k]+np.tan(np.radians(degrees))*n,n)
    return trial, dict(operation='joint-final-conditioning-sample',degrees=degrees,body_fraction=fraction,
                       world_horizontal_shifts_m=shifts,gradient_step=False)


class JointConditionedGradientSearch(ProgressiveGradientSearch):
    def save(self,current,mode,began,initialization,**extra):
        from co_common import save
        model=self.model;self.search_only=True
        sampled=FastReuseSearch.save(self,current,mode,began,initialization,**extra)
        candidates=[];screen=[];screen_start=time.monotonic()
        # A positive angular margin avoids repeatedly exporting almost exactly
        # coincident planes. Lateral shifts affect explicit Juxtapose seats only.
        for degrees,fraction in [(.125,1/512),(.5,1/512),(.125,1/256),(.5,1/256),(1.,1/128),(.5,0.)]:
            layout,detail=joint_perturbation(model,current['layout'],degrees,fraction)
            state=model.evaluate(layout)
            screen.append(dict(detail=detail,counts=state['counts'],passed=failed_count(state)==0))
            if failed_count(state)==0:candidates.append((state,detail))
        screen_seconds=time.monotonic()-screen_start
        save(self.out/'joint_conditioning_screen.json',screen)
        start=time.monotonic();attempts=[];actual=None
        for state,detail in candidates[:4]:
            row=dict(sampled_serial=state['serial'],passed=False,detail=detail);clock=time.monotonic()
            try:
                result=model.exact(state['layout']);work=result.get('actual_work_surface_checks',[])
                row.update(counts=result['counts'],volume_cm3=result['volume_cm3'])
                if failed_count(result)==0 and len(work)==len(state['layout'].active) and all(r['passed'] for r in work):
                    actual=result;row['passed']=True
            except (RuntimeError,ValueError,AssertionError) as error:row['error']=str(error)
            row['seconds']=time.monotonic()-clock;attempts.append(row)
            save(self.out/'final_validation_attempts.json',attempts)
            if actual is not None:break
        if actual is None:raise RuntimeError('Joint final conditioning unresolved; physical checks unchanged')
        self.process_snapshot(actual,'final_verified_joint_conditioning',attempts[-1])
        if getattr(self,'capture_process',False):
            self.process_rows[-1]['geometry_verified']=True
            self.process_rows[-1]['actual_counts_verified']=actual['counts'];save(self.out/'process.json',self.process_rows)
        extra=dict(extra);extra['passed']=True
        return Model.save(model,actual,self.out,dict(strategy=mode,policy='joint-original-load-preserving-final-conditioning',
            initialization=initialization,search_seconds=sampled['search_seconds'],conditioning_screen_seconds=screen_seconds,
            validation_seconds=time.monotonic()-start,seconds=time.monotonic()-began,
            sample_evaluations=model.sample_calls,exact_evaluations_during_search=0,timing=model.timing,events=self.events,
            final_validation_attempts=attempts,final_conditioning_gradient_claimed=False,
            rotating_reuse_pose_count=sum(registered(actual['layout'],k) for k in actual['layout'].active),
            juxtaposed_pose_count=juxtaposed_count(actual['layout']),
            conservative_contact_grid_unmet_loads=0,final_acceptance_run=True,**extra))
