"""One incumbent chain; compare single-pose proposals after joint descent."""
import time
import numpy as np
from single_pose_chain import sample_one
from physics_guided import save


def candidate_chain_search(self, iterations=1):
    began=time.monotonic()
    rng=np.random.default_rng(self.chain_seed)
    current=self.exact(self.normals.copy())
    initial_counts=current['counts']
    np.savez_compressed(self.out/'initial_directions.npz',directions=current['directions'])
    self.remember(current)
    events=[dict(stage='initial',directions=current['directions'].tolist(),counts=initial_counts)]
    round_index=0
    while self.proposals<self.max_proposals and not all(m.all() for m in current['masks']):
        round_index+=1
        origin=current
        candidates=[]
        rows=[]
        count=min(self.candidates_per_round,self.max_proposals-self.proposals)
        # Balanced pose exposure; every candidate starts from the SAME incumbent.
        poses=[]
        while len(poses)<count: poses.extend(rng.permutation(len(self.normals)).tolist())
        for index,pose in enumerate(poses[:count]):
            self.proposals+=1
            start=len(self.trace); evaluation_start=self.exact_calls
            row=dict(round=round_index,candidate=index+1,stage='direction_choice',
                     pose=self.group['poses'][pose],before=origin['directions'].tolist())
            try:
                directions,axis,angle=sample_one(origin['directions'],self.normals,pose,rng,self.sample_angle)
                row.update(axis=axis,angle_degrees=angle,directions=directions.tolist())
                proposed=self.exact(directions)
                row['direction_choice_counts']=proposed['counts']
                self.remember(proposed)
                result=self.refine(proposed,iterations,f'round {round_index} candidate {index+1} gradient_descent')
                self.remember(result)
                candidates.append((result,row))
                row.update(gradient_descent_counts=result['counts'],after=result['directions'].tolist())
            except (RuntimeError,ValueError) as error:
                row['error']=str(error)
            row.update(gradient_descent_attempts=len(self.trace)-start,
                       gradient_descent_accepted=sum(bool(t.get('accepted')) for t in self.trace[start:]),
                       exact_evaluations=self.exact_calls-evaluation_start)
            rows.append(row)
            save(self.out/'candidate_round_in_progress.json',dict(round=round_index,candidates=rows))
            print('CANDIDATE',round_index,index+1,'/',count,row.get('gradient_descent_counts',row.get('error')),flush=True)
            if candidates and all(m.all() for m in candidates[-1][0]['masks']):break
        # All contenders use the same accumulated original-load working set.
        loads=sorted(self.load_bank)
        before,_=self.actual_loss(origin,loads)
        winner=None; winner_loss=before
        for result,row in candidates:
            try:
                after,_=self.actual_loss(result,loads)
                protected=all(not old.all() or new.all() for old,new in zip(origin['masks'],result['masks']))
                feasible=all(m.all() for m in result['masks'])
                improved=after<before-max(1e-12,before*1e-4)
                eligible=protected and (feasible or improved)
                row.update(comparison_loss=after,incumbent_loss=before,eligible=bool(eligible),selected=False)
                if eligible and (winner is None or after<winner_loss or feasible):
                    winner=(result,row);winner_loss=after
                    if feasible:break
            except (RuntimeError,ValueError) as error:row['comparison_error']=str(error)
        if winner is not None:
            current=winner[0];winner[1]['selected']=True
        events.extend(rows)
        events.append(dict(stage='round_result',round=round_index,accepted=winner is not None,
                           directions=current['directions'].tolist(),counts=current['counts'],
                           incumbent_loss=before,selected_loss=winner_loss,comparison_load_count=len(loads)))
        save(self.out/'chain_trajectory.json',events)
        print('ROUND',round_index,'accepted',winner is not None,current['counts'],flush=True)
    passed=all(m.all() for m in current['masks'])
    np.savez_compressed(self.out/'continuation_directions.npz',directions=current['directions'])
    save(self.out/'chain_trajectory.json',events)
    save(self.out/'global_proposals.json',events)
    save(self.out/'critical_load_selection.json',self.selection_trace)
    self.report_extra=dict(algorithm='one chain; single-pose direction_choice candidates then joint gradient_descent',
        candidate_count_per_round=self.candidates_per_round,round_count=round_index,
        proposal_count=self.proposals,proposal_budget=self.max_proposals,random_seed=self.chain_seed,
        gradient_descent_steps_per_candidate=iterations,sampling_angle_degrees=[5.,self.sample_angle],
        selection_policy='same accumulated original-load working set; strict real max cone-deficit improvement; preserve fully feasible poses; no geometry-only incumbent acceptance',
        connectivity_required=False,component_pruning_deferred=True,
        passed=bool(passed and current['overlap']<1e-10 and current['partition']<1e-10 and max(current['endpoint_overlap'])<1e-10))
    return self.finish(current,initial_counts,began,'aggregate_force_feasible' if passed else 'bounded_candidate_chain_unresolved',
                       returned_best=False,continuation_counts=current['counts'])
