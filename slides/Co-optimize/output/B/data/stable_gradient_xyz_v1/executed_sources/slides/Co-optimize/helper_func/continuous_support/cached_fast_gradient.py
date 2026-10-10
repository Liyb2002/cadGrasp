"""Shared exact contact predicates and joint world-horizontal descent.

This revision is independent of the frozen v3 source set. Candidate geometry
uses the original predicates, sharing identical body/sweep queries. The final
physical worker and its acceptance conditions are unchanged.
"""
import time
import numpy as np
from whole_search.common import tangent_frame, legal_direction
from whole_search.model import Model
from whole_search.reuse_first import juxtaposed_count
from whole_search.reuse_first import registered
from whole_search.search import failed_count
from .adaptive_fast_gradient import AdaptiveGradientModel, AdaptiveGradientSearch
from .fast_gradient import turn
from .geometry_cache import GeometryCache
from translation.layout_update import shift as move, derivative


class CachedGradientModel(AdaptiveGradientModel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.geometry_cache=GeometryCache(self)

    def locks(self,layout,owner,blocker):
        return self.geometry_cache.locks(layout,owner,blocker)


class CachedGradientSearch(AdaptiveGradientSearch):
    def local_proposals(self,current,targets,worst,translation_guests=()):
        rows,base=super().local_proposals(current,targets,worst,translation_guests)
        m,layout=self.model,current['layout'];start=time.monotonic()
        movable=[k for k in translation_guests if not registered(layout,k)]
        fine=getattr(self,'fine_resolution',False)
        if fine:
            # The failing owner's loss can be controlled by another owner's
            # exit column. Preserve fine coordinate probes for every blocker.
            for k in layout.active:
                frame=tangent_frame(layout.directions[k])
                for degrees in [.03125,.125]:
                    for axis in range(2):
                        for sign in [-1,1]:
                            rows.append(('direction-sample',turn(m,layout,k,sign*np.radians(degrees)*frame[:,axis]),
                                         dict(pose=m.poses[k],step_degrees=sign*degrees,axis=axis)))
        if len(movable)>1:
            delta=m.extent*(1/512 if fine else 1/100)
            fractions=[1/1024,1/512,1/256,1/128] if fine else [1/128,1/64,1/32,1/16]
            def shift(v):
                trial=layout.copy()
                for k in movable:
                    world=np.r_[v,0.] if len(v)==2 else v
                    trial=move(m,trial,k,m.native[layout.hosts[k],:3,:3].T@world)
                return trial
            dimensions=3 if getattr(m,'allow_z_translation',False) else 2
            gradient=np.array([derivative(m,layout,movable,axis,delta) for axis in range(dimensions)])
            if dimensions==3 and min(m.workpiece_height(layout,k) for k in movable)<=1e-9:
                gradient[2]=min(gradient[2],0.)
            length=float(np.linalg.norm(gradient))
            if length>1e-28:
                for fraction in fractions:
                    rows.append(('translation-gradient-coherent',shift(-m.extent*fraction*gradient/length),
                                 dict(poses=[m.poses[k] for k in movable],step_m=m.extent*fraction,
                                      gradient=gradient.tolist(),finite_difference_m=delta,
                                      all_poses_in_objective=True,coordinate='shared_world_xyz' if dimensions==3 else 'shared_world_horizontal')))
            for fraction in fractions[:3]:
                for axis in np.eye(dimensions):
                    for sign in [-1,1]:
                        rows.append(('translation-coherent-sample',shift(sign*m.extent*fraction*axis),
                                     dict(step_m=m.extent*fraction,axis=axis.tolist(),sign=sign)))
        m.gradient_seconds+=time.monotonic()-start
        return rows,base

    def solve_frontier(self,current,anchor=None,insertion_guest=None):
        # Restore the original search's final all-movable repair, while every
        # derivative remains the ALL-owner integrated-distance objective.
        current=super().solve_frontier(current,anchor,insertion_guest)
        if failed_count(current):
            movable=[k for k in current['layout'].active if not registered(current['layout'],k)]
            current=self.refine(current,3,anchor=None,translation_guests=movable,
                                phase='all_movable_gradient_repair')
        if 0<failed_count(current)<=512:
            current=self.fine_refine(current,rounds=6)
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
