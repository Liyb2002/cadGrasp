"""Strict nominal contacts and active coordinates of the all-demand loss.

Coordinate selection integrates all quadrature residuals against potentially
released reaction rays. It never selects one hardest force/torque demand.
Actual differences and line steps still evaluate every owner's gains/losses.
"""
from collections import OrderedDict
import time
import numpy as np
from whole_search.common import tangent_frame,legal_direction
from whole_search.reuse_first import registered
from whole_search.search import failed_count
from whole_search.fast_search import FastReuseSearch
from whole_search.model import Model
from whole_search.reuse_first import juxtaposed_count
from .cached_fast_gradient import CachedGradientModel
from .efficient_fast_gradient import EfficientGradientSearch
from .projection import project_demands
from .strict_nominal_cache import StrictNominalCache
from .fast_gradient import turn


class ActiveGradientModel(CachedGradientModel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.geometry_cache=StrictNominalCache(self)
        self.coordinate_cache=OrderedDict();self.coordinate_seconds=0.

    def useful_coordinates(self,current):
        layout=current['layout'];key=layout.key(),self.quadrature_epoch
        if key in self.coordinate_cache:return self.coordinate_cache[key]
        start=time.monotonic();state=self.contact_delta.state(layout)
        scores={k:0. for k in layout.active}
        for owner in layout.active:
            full,_=self.supply_at_points(owner,state.available[owner])
            needs,weights,scale=self.demand_screen.demands[owner]
            nodes=[(needs,.5*weights/scale**2)]
            if owner in self.adaptive_demands:
                targets,w,_=self.adaptive_demands[owner]
                nodes.append((targets,.5*w/self.original_scales[owner]**2))
            gain=np.zeros(len(self.points))
            for targets,w in nodes:
                residual=project_demands(full,targets)['residuals']
                polar=np.maximum(-residual@self.point_rays[owner].T,0.)
                gain+=(polar**2).T@w
            material=(state.coverage_counts[owner]>0)&np.isin(self.sources,self.allowed[owner])
            for blocker in layout.active:
                relevant=state.locks[owner,blocker]&material
                # Multiple blockers share the release score; coherent sphere
                # and world-horizontal coordinates can move them together.
                scores[blocker]+=float((gain[relevant]*self.point_areas[relevant]/
                                       state.lock_counts[owner][relevant]).sum())
        value=sorted(layout.active,key=lambda k:-scores[k])
        self.coordinate_cache[key]=value,scores
        while len(self.coordinate_cache)>32:self.coordinate_cache.popitem(last=False)
        self.coordinate_seconds+=time.monotonic()-start
        return value,scores


class ActiveGradientSearch(EfficientGradientSearch):
    def coordinate_subset(self,current,base):
        ordered,scores=self.model.useful_coordinates(current)
        active=list(current['layout'].active)
        owners=sorted(active,key=lambda k:-base['residual_loss'][active.index(k)])
        chosen=list(dict.fromkeys([owners[0]]+ordered))[:min(4,len(active))]
        return chosen,ordered,scores

    def local_proposals(self, current, targets, worst, translation_guests=()):
        m, layout = self.model, current['layout']
        began = time.monotonic(); base = m.proxy(layout)
        coordinates,ordered,scores=self.coordinate_subset(current,base)
        all_movable=[k for k in translation_guests if not registered(layout,k)]
        chosen_movable=sorted(all_movable,key=lambda k:ordered.index(k))[:2]
        fine = getattr(self, 'fine_resolution', False)
        h = np.radians(.125 if fine else .75)
        rows = []; gradients = {}; measurements = []
        for k in coordinates:
            frame = tangent_frame(layout.directions[k]); g = np.zeros(2)
            for axis in range(2):
                minus = m.proxy(turn(m, layout, k, -h*frame[:, axis]))['loss']
                plus = m.proxy(turn(m, layout, k, h*frame[:, axis]))['loss']
                g[axis] = (plus-minus)/(2*h)
                measurements.append(dict(pose=m.poses[k], axis=axis,
                                         minus=minus, plus=plus, derivative=float(g[axis])))
            gradients[k] = (frame, g)
        norm = np.sqrt(sum(float(g@g) for _, g in gradients.values()))
        degrees = [.03125, .125, .25] if fine else [.25, .5, 1., 2., 4., 8.]
        if norm > 1e-28:
            for step in degrees:
                trial = layout.copy()
                for k, (frame, g) in gradients.items():
                    trial.directions[k] = turn(m, layout, k, -np.radians(step)*frame@g/norm).directions[k]
                rows.append(('direction-gradient-joint', trial,
                    dict(step_degrees=step, finite_difference_degrees=float(np.degrees(h)),
                         gradient_norm=float(norm), gradient=measurements,
                         derivative_kind='finite_resolution_contact_secant', all_poses_in_objective=True)))
        # Common rotation moves several overlapping blockers simultaneously.
        # Differentiate this shared coordinate instead of summing independently
        # flat contact-cell derivatives.
        coherent = np.zeros(3)
        def rotate(axis, angle):
            trial = layout.copy()
            for k in layout.active:
                trial.directions[k] = turn(m, layout, k, angle*np.cross(axis, layout.directions[k])).directions[k]
            return trial
        for axis in range(3):
            unit = np.eye(3)[axis]
            coherent[axis] = (m.proxy(rotate(unit,h))['loss']-m.proxy(rotate(unit,-h))['loss'])/(2*h)
        length = np.linalg.norm(coherent)
        if length > 1e-28:
            for step in degrees:
                rows.append(('direction-gradient-coherent', rotate(-coherent/length,np.radians(step)),
                    dict(step_degrees=step, gradient=coherent.tolist(), gradient_norm=float(length),
                         finite_difference_degrees=float(np.degrees(h)), all_poses_in_objective=True)))
        owners = sorted(coordinates, key=lambda k: -base['residual_loss'][list(layout.active).index(k)])[:3]
        for k in owners:
            frame, g = gradients[k]; length = np.linalg.norm(g)
            if length > 1e-28:
                for step in degrees:
                    rows.append(('direction-gradient', turn(m,layout,k,-np.radians(step)*frame@g/length),
                                 dict(pose=m.poses[k],step_degrees=step,all_poses_in_objective=True)))
            # Limited sampling, explicitly separate from gradient steps.
            for step in ([.03125,.125] if fine else [.5,2.,5.]):
                for axis in range(2):
                    for sign in [-1,1]:
                        rows.append(('direction-sample', turn(m,layout,k,sign*np.radians(step)*frame[:,axis]),
                                     dict(pose=m.poses[k],step_degrees=sign*step,axis=axis)))
        for axis in np.eye(3):
            for step in ([-.125,.125] if fine else [-2.,2.,-5.,5.]):
                rows.append(('direction-coherent-sample',rotate(axis,np.radians(step)),
                             dict(step_degrees=step,axis=axis.tolist())))
        movable = chosen_movable
        for k in movable:
            frame=tangent_frame(m.floor_normal(layout,k)); g=np.zeros(2)
            delta=m.extent*(1/512 if fine else 1/100)
            for axis in range(2):
                values=[]
                for sign in [-1,1]:
                    trial=layout.copy();trial.placements[k,:3,3]+=sign*delta*frame[:,axis]
                    values.append(m.proxy(trial)['loss'])
                g[axis]=(values[1]-values[0])/(2*delta)
            vectors=[(np.eye(2)[axis]*sign,'sample') for axis in range(2) for sign in [-1,1]]
            if np.linalg.norm(g)>1e-28: vectors.insert(0,(-g/np.linalg.norm(g),'gradient'))
            for fraction in ([1/1024,1/512,1/256] if fine else [1/128,1/64,1/32,1/16,1/8]):
                for v,kind in vectors:
                    trial=layout.copy();trial.placements[k,:3,3]+=m.extent*fraction*frame@v
                    rows.append(('translation-'+kind,trial,dict(pose=m.poses[k],step_m=m.extent*fraction,
                        finite_difference_m=delta,gradient=g.tolist(),all_poses_in_objective=True)))
        if len(all_movable)>1:
            for axis in np.eye(3)[:2]:
                for sign in [-1,1]:
                    trial=layout.copy()
                    for k in all_movable:
                        trial.placements[k,:3,3]+=m.extent/256*sign*m.native[layout.hosts[k],:3,:3].T@axis
                    rows.append(('translation-coherent-sample',trial,dict(step_m=m.extent/256)))
        if len(all_movable)>1:
            delta=m.extent*(1/512 if fine else 1/100)
            fractions=[1/1024,1/512,1/256,1/128] if fine else [1/128,1/64,1/32,1/16]
            def shift(v):
                trial=layout.copy()
                for k in all_movable:
                    trial.placements[k,:3,3]+=m.native[layout.hosts[k],:3,:3].T@np.r_[v,0.]
                return trial
            gradient=np.array([(m.proxy(shift(delta*axis))['loss']-m.proxy(shift(-delta*axis))['loss'])/(2*delta)
                               for axis in np.eye(2)])
            length=float(np.linalg.norm(gradient))
            if length>1e-28:
                for fraction in fractions:
                    rows.append(('translation-gradient-coherent',shift(-m.extent*fraction*gradient/length),
                                 dict(poses=[m.poses[k] for k in all_movable],step_m=m.extent*fraction,
                                      gradient=gradient.tolist(),finite_difference_m=delta,
                                      all_poses_in_objective=True,coordinate='shared_world_horizontal')))
            for fraction in fractions[:3]:
                for axis in np.eye(2):
                    for sign in [-1,1]:
                        rows.append(('translation-coherent-sample',shift(sign*m.extent*fraction*axis),
                                     dict(step_m=m.extent*fraction,axis=axis.tolist(),sign=sign)))
        self.last_coordinate_selection=dict(direction_poses=[m.poses[k] for k in coordinates],
            translation_poses=[m.poses[k] for k in chosen_movable],all_owner_loss=True,
            release_scores={m.poses[k]:scores[k] for k in layout.active})
        m.gradient_seconds += time.monotonic()-began
        return rows,base


    def save(self,current,mode,began,initialization,**extra):
        # Save the complete cheap trace first. Final checks change neither
        # physical sources nor tolerances. Small legal upward direction
        # candidates avoid exact tangencies in exported geometry.
        from co_common import save
        m=self.model;self.search_only=True
        sampled=FastReuseSearch.save(self,current,mode,began,initialization,**extra)
        candidates=[];screen=[]
        movable=[k for k in current['layout'].active if not registered(current['layout'],k)]
        separation=None
        if movable:
            q=current['layout'].copy();shifts={}
            for rank,k in enumerate(movable):
                angle=2*np.pi*(rank+1)*(np.sqrt(5)-1)/2
                world=m.extent/512*np.array([np.cos(angle),np.sin(angle),0.])
                q.placements[k,:3,3]+=m.native[q.hosts[k],:3,:3].T@world
                shifts[m.poses[k]]=world.tolist()
            trial=m.evaluate(q)
            screen.append(dict(operation='translation-conditioning-sample',counts=trial['counts'],shifts=shifts))
            if failed_count(trial)==0:
                separation=(trial,dict(operation='translation-conditioning-sample',body_fraction=1/512,world_horizontal_shifts_m=shifts,gradient_step=False))
        for degrees in [.03125,.125,.5]:
            layout=current['layout'].copy()
            for k in layout.active:
                layout.directions[k]=legal_direction(layout.directions[k]+np.tan(np.radians(degrees))*m.floor_normal(layout,k),m.floor_normal(layout,k))
            trial=m.evaluate(layout)
            screen.append(dict(degrees=degrees,counts=trial['counts']))
            if failed_count(trial)<=max(128,failed_count(current)):
                candidates.append((trial,dict(operation='direction-numerical-separation',degrees=degrees)))
        candidates.sort(key=lambda row:(failed_count(row[0]),row[0]['volume_cm3']))
        if separation is not None:
            candidates=[separation]+candidates[:1]+[(current,dict(operation='unperturbed'))]
        else:
            candidates=candidates[:1]+[(current,dict(operation='unperturbed'))]+candidates[1:2]
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
        m.timing.update(active_coordinate_selection_s=m.coordinate_seconds,strict_leading_corrections=m.geometry_cache.leading_points_removed)
        return Model.save(m,actual,self.out,dict(strategy=mode,policy='all-demand-gradient-delta-final-original-checks',
            initialization=initialization,search_seconds=sampled['search_seconds'],validation_seconds=time.monotonic()-start,
            seconds=time.monotonic()-began,sample_evaluations=m.sample_calls,exact_evaluations_during_search=0,
            timing=m.timing,events=self.events,final_validation_attempts=attempts,
            rotating_reuse_pose_count=sum(registered(actual['layout'],k) for k in actual['layout'].active),
            juxtaposed_pose_count=juxtaposed_count(actual['layout']),
            conservative_contact_grid_unmet_loads=failed_count(selected),
            final_acceptance_run=True,**extra))

    def record(self,row):
        if hasattr(self,'last_coordinate_selection'):row['active_coordinate_selection']=self.last_coordinate_selection
        super().record(row)
