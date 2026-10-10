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

from joint_guided_placement import JointGuidedPlacementSearch
from deduplicated_placement import DeduplicatedPlacement
from current_reaction_target import CurrentReactionTarget

class GlobalPlacementSearch(JointGuidedPlacementSearch):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        supplies=[]
        for pose in self.group['poses']:
            with np.load(self.base/'step4/step4.1/data'/f'{pose}.npz') as z:supplies.append(z['supply_7d'].copy())
        self.reference=dict(serial=-1,masks=self.saved_masks,supplies=supplies,counts=[int(m.sum()) for m in self.saved_masks])

    def exact(self,directions,offsets):
        return DeduplicatedPlacement.exact(self,directions,offsets)

    def run(self,iterations=12,candidates=6,finalists=2):
        began=time.monotonic();directions=self.initial_directions.copy();offsets=np.zeros_like(directions)
        current=None;errors=[];events=[];stop='iteration_limit';accepted_steps=0;self.guide=None;self.guidance_avoided={};self.event_avoid=set();event_plateaus={};stalled_rounds=0
        try:current=self.exact(directions,offsets)
        except (RuntimeError,ValueError) as error:errors.append(dict(stage='initial',error=str(error)))
        initial_counts=[int(m.sum()) for m in self.saved_masks];exact_initial=current['counts'] if current else None
        for iteration in range(iterations):
            if current is not None and all(m.all() for m in current['masks']):stop='force_exit_feasible';break
            targets=self.refresh_targets(current)
            if self.guide is None:self.guide=CurrentReactionTarget(self,current,directions,offsets)
            reference=current if current is not None else self.reference
            before_worst,_=farthest_load(self,reference)
            before_loss=before_worst['loss'] if before_worst else 0.
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
                    trial=self.exact(d,o);record['counts']=trial['counts']
                    protected=preserves_loads(reference,trial)
                    after_worst,_=farthest_load(self,trial)
                    after_loss=after_worst['loss'] if after_worst else 0.
                    tolerance=max(1e-10,before_loss*1e-5)
                    actual_nonregressing=after_loss<=before_loss+max(1e-10,before_loss*1e-6)
                    accepted=after_loss<before_loss-tolerance or (actual_nonregressing and self.real_score(trial)>self.real_score(reference))
                    if current is None and actual_nonregressing:accepted=True
                    record.update(worst_loss_before=before_loss,worst_loss_after=after_loss)
                    event_key=('guidance',step.get('target_key')) if step.get('reaction_guidance') else (step.get('owner'),step.get('contact_index'))
                    plateau=False
                    if actual_nonregressing and not accepted and step.get('reaction_guidance'):
                        plateau=(step['guidance_after']<step['guidance_before']-1e-8 and event_plateaus.get(event_key,0)<8)
                        accepted=plateau
                    if actual_nonregressing and not accepted and step.get('event'):
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
        report=dict(complete=True,algorithm='sample pose/tool blocks; small force-guided placement steps; global worst original wrench acceptance',
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
            acceptance='all original loads/no-uplift, continuous nominal exits, 1% clearance and work exclusions; intermediate states reduce global worst wrench distance or use bounded nonregressing guidance steps',
            deferred='connectivity, installed whole-fixture floor legality/support, strength; illegal group remains a diagnostic',
            provenance=provenance(self.inputs,[Path(__file__),Path(__file__).with_name('placement_sampling.py'),Path(__file__).with_name('contact_boundary_model.py'),Path(__file__).with_name('local_placement_sampling.py'),Path(__file__).with_name('contact_event_steps.py'),Path(__file__).with_name('joint_placement_target.py'),Path(__file__).with_name('event_placement_sampling.py'),Path(__file__).with_name('current_reaction_target.py'),Path(__file__).with_name('deduplicated_placement.py')]+code_sources()))
        save(self.out/'report.json',report);print('LOCAL PLACEMENT FINAL',self.group['id'],report['final_counts'],stop,flush=True)
        return report

