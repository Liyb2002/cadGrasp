"""Sample adjustment blocks; calculate small increments from contact boundaries."""
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from co_common import save,D,S,U,provenance,code_sources
from placement_sampling import PlacementSearch
from contact_boundary_model import ContactBoundaryModel
from physics_guided_geometry import tangent_frames,retract
from contact_recovery import preserves_loads
from worst_wrench_descent import farthest_load


from local_placement_sampling import LocalPlacementSearch
from contact_event_steps import event_candidates
from co_common import J

from event_placement_sampling import EventPlacementSearch
from joint_placement_target import JointPlacementTarget

class JointGuidedPlacementSearch(EventPlacementSearch):
    def propose(self,directions,offsets,targets,kind,indices,base):
        return self.guide.propose(directions,offsets,kind,indices,targets,base)

    def run(self,iterations=12,candidates=6,finalists=2):
        began=time.monotonic();directions=self.initial_directions.copy();offsets=np.zeros_like(directions)
        current=None;errors=[];events=[];stop='iteration_limit';accepted_steps=0;self.guide=None;self.guidance_avoided={};self.event_avoid=set();event_plateaus={};stalled_rounds=0
        try:current=self.exact(directions,offsets)
        except (RuntimeError,ValueError) as error:errors.append(dict(stage='initial',error=str(error)))
        initial_counts=[int(m.sum()) for m in self.saved_masks];exact_initial=current['counts'] if current else None
        for iteration in range(iterations):
            if current is not None and all(m.all() for m in current['masks']):stop='force_exit_feasible';break
            targets=self.refresh_targets(current)
            if self.guide is None:self.guide=JointPlacementTarget(self,current,directions,offsets)
            base=self.boundary.evaluate(directions,offsets,targets);began_proxy=time.monotonic()
            proposed=[];stats=[]
            for kind,indices in self.blocks(current,candidates):
                try:
                    options,info=self.propose(directions,offsets,targets,kind,indices,base)
                    proposed.extend(options);stats.append(info)
                except (RuntimeError,ValueError) as error:stats.append(dict(kind=kind,indices=indices,error=str(error)))
            events_proposed=[]
            for event in event_candidates(self,current,directions,offsets,limit=8):
                value=self.boundary.evaluate(event['directions'],event['offsets'],targets)
                info=dict(event=True,owner=event['owner'],contact_index=event['contact_index'],benefit=event['benefit'],**event['step'])
                events_proposed.append(((value['loss'],value['sum_loss'],float(np.linalg.norm(event['offsets']))),event['kind'],event['directions'],event['offsets'],value,info))
            proposed.sort(key=lambda row:row[0]);row=dict(iteration=iteration+1,targets=[(k,i) for k,i,t in targets],
                proxy_loss=base['loss'],proposal_stats=stats,proposals=len(proposed),proxy_seconds=time.monotonic()-began_proxy,trials=[])
            accepted=False
            finalists_list=proposed[:finalists]+events_proposed[:3]
            for score,kind,d,o,value,step in finalists_list:
                record=dict(kind=kind,proxy_score=score,step=step,directions=d.tolist(),offsets_m=o.tolist(),
                    **self.boundary.changes(base['geometry'],value['geometry']))
                try:
                    geometry=self.boundary.geometry(d,o)
                    cheap_masks=[J.classify(rays,task.targets)[0] for rays,(task,T) in zip(geometry['rays'],self.states)]
                    record['screen_counts']=[int(m.sum()) for m in cheap_masks]
                    if not preserves_loads(current if current is not None else dict(masks=self.saved_masks),dict(masks=cheap_masks)):
                        record['screened_out']=True;row['trials'].append(record)
                        if step.get('event'):self.event_avoid.add((step['owner'],step['contact_index']))
                        continue
                    trial=self.exact(d,o);record['counts']=trial['counts']
                    protected=preserves_loads(current if current is not None else dict(masks=self.saved_masks),trial)
                    accepted=protected and (current is None or self.real_score(trial)>self.real_score(current))
                    if protected and not accepted:
                        before,_=farthest_load(self,current);after,_=farthest_load(self,trial)
                        accepted=(0. if after is None else after['loss'])<before['loss']-1e-8
                    event_key=('guidance',step.get('target_key')) if step.get('reaction_guidance') else (step.get('owner'),step.get('contact_index'))
                    plateau=False
                    if protected and not accepted and step.get('reaction_guidance'):
                        plateau=(step['guidance_after']<step['guidance_before']-1e-8 and event_plateaus.get(event_key,0)<8)
                        accepted=plateau
                    if protected and not accepted and step.get('event'):
                        plateau=(step['violation_after']<step['violation_before']-1e-14 and event_plateaus.get(event_key,0)<8)
                        accepted=plateau
                    if accepted:
                        current=trial;directions=d;offsets=o;accepted_steps+=1
                        if plateau:event_plateaus[event_key]=event_plateaus.get(event_key,0)+1
                        else:self.event_avoid.clear();event_plateaus.clear();self.guide=None;self.guidance_avoided.clear()
                    elif step.get('event'):self.event_avoid.add(event_key)
                    record['contact_event_plateau']=plateau
                    record.update(accepted=accepted,protected=protected)
                except (RuntimeError,ValueError) as error:record.update(error=str(error),accepted=False)
                row['trials'].append(record)
                if accepted:break
            row.update(accepted=accepted,counts=current['counts'] if current else None);events.append(row)
            save(self.out/'sampling_trace.json',events)
            print('LOCAL PLACEMENT ROUND',iteration+1,'accepted',accepted,row['counts'],'seconds',row['proxy_seconds'],flush=True)
            if not accepted:
                stalled_rounds+=1
                if stalled_rounds%2==0:
                    for k,ids,w in self.guide.selected:self.guidance_avoided.setdefault(k,[]).extend(map(int,ids))
                    self.guide=None
                if stalled_rounds>=6:stop='local_boundary_search_stalled';break
            else:stalled_rounds=0
        passed=current is not None and all(m.all() for m in current['masks'])
        if current is not None:D.export_exact_obj(S.unpack(current['remaining']),self.out/'remaining_support.obj')
        np.savez_compressed(self.out/'layout.npz',directions=directions,offsets_m=offsets,
            T_fixture_to_world=np.asarray(current['transforms']) if current else np.empty((0,4,4)))
        report=dict(complete=True,algorithm='sample pose/tool blocks; joint force-balanced contact-boundary guidance; small computed steps and exact acceptance',
            pose_set=self.group['id'],poses=self.group['poses'],initial_counts=initial_counts,exact_initial_counts=exact_initial,
            final_counts=current['counts'] if current else None,force_exit_passed=bool(passed),force_passed=bool(passed),
            geometry_constructed=current is not None,clearance_certified=current is not None,full_fixture_accepted=False,
            baseline_used=False,baseline_passed=False,stop_reason=stop,accepted_sampling_steps=accepted_steps,iterations=events,errors=errors,
            offsets_m=offsets.tolist(),directions=directions.tolist(),maximum_translation_m=float(np.linalg.norm(offsets,axis=1).max()),
            maximum_translation_step_m=.001,maximum_direction_step_degrees=1.,fixed_reference_pose=self.group['poses'][0],
            seconds=time.monotonic()-began,exact_evaluations=self.exact_calls,original_loads_reused=True,
            load_count_per_pose=[len(t.targets) for t,T in self.states],component_count=len(current['remaining'].decompose()) if current else None,
            diagnostics=current['diagnostics'] if current else None,contact_patches=current['contact_patches'] if current else None,
            derivative='central differences of continuously clipped contact polygons, not fixture solids; signed-distance interpolation for initial cross-object overlap is guidance only',
            acceptance='all original loads/no-uplift, continuous nominal exits, 1% clearance and work exclusions; protect every previously passing load',
            deferred='connectivity, installed whole-fixture floor legality/support, strength; illegal group remains a diagnostic',
            provenance=provenance(self.inputs,[Path(__file__),Path(__file__).with_name('placement_sampling.py'),Path(__file__).with_name('contact_boundary_model.py'),Path(__file__).with_name('local_placement_sampling.py'),Path(__file__).with_name('contact_event_steps.py'),Path(__file__).with_name('joint_placement_target.py'),Path(__file__).with_name('event_placement_sampling.py')]+code_sources()))
        save(self.out/'report.json',report);print('LOCAL PLACEMENT FINAL',self.group['id'],report['final_counts'],stop,flush=True)
        return report

