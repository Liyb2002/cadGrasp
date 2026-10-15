"""Juxtapose-only controller: original gates, ranking and discrete proposals.

The solve loop is copied from PopulationSoftSearch, with the continuous branch
replaced by a discrete branch. Local refinement and LP height moves are disabled
in the default strict baseline; an explicit adjusted mode isolates that choice.
"""
import numpy as np
from population_soft import (PopulationSoftSearch, physical_rank, weighted_pose,
    discrete_move, differential_move, pso_move, preserves_poses, native_slide)
from population32 import registered_population, centered_seat
from material_search import freeze_population_measure, sample_id
from whole_search.reuse_first import registered
from whole_search.search import failed_count
from whole_search.common import legal_direction
from fixture_state import installed_state

def baseline_joint_discrete_move(model,layout,weights,rng,owner,native_family=False,apply_floor_heights=False):
    """Choose a complete role change during optimization, using physical floor certificates."""
    active=list(layout.active);q=layout.copy()
    for k in active:
        world=installed_state(model.native,layout,int(layout.hosts[k]))[:3,:3]@layout.directions[k]
        q.hosts[k]=k;q.placements[k]=np.eye(4);q.state_offsets[k]=0.
        q.directions[k]=legal_direction(model.native[k,:3,:3].T@world,model.floor_normal(q,k))
    number=min(len(active)-1,max(1 if all(registered(layout,k)for k in active)else 0,int(rng.binomial(len(active),1/3))))
    guests=[owner]if number else[]
    available=[k for k in active if k!=owner]
    maximum=max(weights.values(),default=0.)
    for _ in range(max(0,number-1)):
        extra=weighted_pose(available,{k:weights[k]+.1*max(maximum,1e-12)for k in available},rng)
        guests.append(extra);available.remove(extra)
    reuse=[k for k in active if k not in guests];prior=getattr(model,'registered_floor_prior',None)
    assessment={};removed=[]
    if prior is not None:
        while len(reuse)>1:
            assessment=prior.assess(reuse)
            if assessment.get('status')!='certified_infeasible':break
            rows=assessment['certificate']['rows']
            conflict=list(set(r['owner']for r in rows)|set(r['floor']for r in rows))
            choices=[k for k in reuse if k in conflict]or reuse
            selected=weighted_pose(choices,{k:weights[k]+.1*max(maximum,1e-12)for k in choices},rng)
            reuse.remove(selected);guests.append(selected);removed.append(selected)
        assessment=prior.assess(reuse)
        if apply_floor_heights and assessment.get('status')=='necessary_feasible':
            q.state_offsets[reuse,2]=assessment['heights_m']
    for k in guests:
        anchor=int(rng.choice(reuse))
        q=native_slide(model,q,k,anchor,rng)if native_family else centered_seat(model,q,k,anchor)
    for k in active:model.validate_placement(q,k)
    return q,dict(action='joint-reseat',selected_pose=owner,poses=guests,registered_indices=reuse,
        floor_certificate_removed=removed,initial_sampled_juxtaposed_count=number,native_state_sliding=native_family,
        initialization='world_xy_center_of_mass_alignment',contact_matching_run=False, floor_prior_height_adjustment_enabled=apply_floor_heights)

class PopulationNoPSOSearch(PopulationSoftSearch):
    local_adjustment = False

    def __init__(self, search):
        super().__init__(search)
        self.continuous_policy = 'none'
        if not self.local_adjustment:
            self.allow_height_repair = False
        self.allow_native_sliding = False

    def checked_adjustment(self, raw, iteration, slot, parent=None):
        if self.local_adjustment:
            return super().checked_adjustment(raw, iteration, slot, parent)
        return raw

    def solve(self, current):
        from co_common import save
        s, m = self.search, self.model
        start = self.preview_cached(current['layout'], result=current)
        self.evaluated[start.layout.key()] = start
        identical=bool(getattr(m,'require_connected_core',False))
        layouts, common = registered_population(m, current['layout'], s.rng,self.population_size,identical=identical)
        initialization = s.out.parent.parent/'step4.1'/s.out.name
        initialization.mkdir(parents=True, exist_ok=False)
        for slot, q in enumerate(layouts):
            np.savez_compressed(initialization/f'slot_{slot:02d}.npz', **q.arrays(m.native))
        save(initialization/'population.json', dict(population_size=self.population_size, common_exit_directions=common,
            identical_registered_initial_directions=identical,
            source='saved original registered Step4.1', warm_start=False,
            predefined_juxtaposed_pose_ids=[], numerical_inputs_changed=False))
        with m.hold_measure(current):
            self.slots=[]
            for i,q in enumerate(layouts):
                try:
                    item=self.checked(self.preview_cached(q,'registered-initialize',detail=dict(slot=i)))
                except (ValueError,RuntimeError)as error:
                    if not getattr(m,'require_connected_core',False):raise
                    # A tiny optional direction perturbation can give an
                    # unresolved LP. Keep the already checked registered seed;
                    # never label the unresolved candidate as force-passed.
                    item=self.preview_cached(current['layout'],'registered-initialize',
                        detail=dict(slot=i,numerical_direction_fallback=True,error=str(error)),result=current)
                    np.savez_compressed(initialization/f'slot_{i:02d}.npz',**item.layout.arrays(m.native))
                    self.errors.append(dict(stage='registered-direction-fallback',slot=i,error=str(error)))
                self.slots.append(item)
            self.certify(self.slots); self.propagate_evidence()
        save(s.out/'initial_population_summary.json',[dict(slot=i,**self.describe(item))for i,item in enumerate(self.slots)])
        save(s.out/'population32_trace.json',self.audit)
        self.pso_best = list(self.slots)
        for iteration in range(s.iterations):
            if self.stop_when_feasible and self.incumbent is not None:break
            reference = min(self.slots, key=lambda x: failed_count(x.result)).result
            with m.hold_measure(reference):
                freeze_population_measure(m, self.slots)
                for item in self.slots:
                    item.loss = float(m.proxy(item.layout)['loss'])
                snapshot = [item.layout.copy() for item in self.slots]
                if self.continuous_policy == 'pso':
                    for item in self.pso_best:
                        item.loss = float(m.proxy(item.layout)['loss'])
                    for slot, item in enumerate(self.slots):
                        self.update_pso_best(slot, item)
                    personal_snapshot = [item.layout.copy() for item in self.pso_best]
                    social_slots = [min(range(self.population_size),
                        key=lambda index: physical_rank(self.pso_best[index], preference)) for preference in (0,1)]
                    social_snapshot = [personal_snapshot[index] for index in social_slots]
                events = []
                for slot, parent in enumerate(list(self.slots)):
                    self.active_slot=slot
                    weights, source = self.weights(parent)
                    coupled = self.allow_coupled_adjustment and not parent.completed and (self.allow_group_seating or not parent.oracle.get('passed'))
                    self.protected = () if coupled else tuple(k for k in parent.layout.active if weights[k] == 0)
                    total = sum(weights.values())
                    probabilities = {str(k): weights[k]/total if total else 0. for k in weights}
                    owner = weighted_pose(parent.layout.active, weights, s.rng)
                    branch = 'hold' if owner is None else 'discrete'
                    self.branch_counts[branch] += 1
                    event = dict(slot=slot, branch=branch, accepted=False, adjustment_passes=0,
                        pose_probabilities=probabilities, weight_source=source,
                        selected_pose=owner,
                        coupled_adjustment_enabled=coupled,
                        protected_pose_indices=list(self.protected), before=self.describe(parent))
                    try:
                        if branch == 'hold':
                            event['move'] = dict(action='hold', reason=source)
                        else:
                            if branch == 'discrete':
                                if self.allow_joint_reseating and s.rng.random()<.25:
                                    native_family=self.allow_native_sliding and s.rng.random()<.5
                                    q,detail=baseline_joint_discrete_move(m,parent.layout,weights,s.rng,owner,
                                        native_family=native_family,apply_floor_heights=self.local_adjustment)
                                else:q, detail = discrete_move(m, parent.layout, weights, s.rng, owner=owner,carry_group=self.allow_group_seating,allow_native_slide=self.allow_native_sliding)
                            elif branch == 'pso':
                                q, detail = pso_move(m, parent.layout, personal_snapshot[slot],
                                    social_snapshot[slot % 2], self.pso_velocity[slot], s.rng, owner)
                                detail.update(personal_best_slot=slot, social_best_slot=social_slots[slot % 2],
                                    ranking_preference='oracle' if slot % 2 else 'force')
                            else:
                                q, detail = differential_move(m, parent.layout, snapshot, slot, weights, s.rng, iteration, owner=owner)
                            if self.allow_height_repair:q=self.repair_height(q)
                            if q is None or not preserves_poses(parent.layout, q, self.protected):
                                raise ValueError('Mutation changed a protected pose or has no eligible owner')
                            raw = self.checked(self.preview_cached(q, branch, sample_id(parent.layout), detail))
                            event['adjustment_passes'] = int(self.local_adjustment)
                            trial = self.checked_adjustment(raw,iteration,slot,parent=parent)
                            if not preserves_poses(parent.layout, trial.layout, self.protected):
                                raise ValueError('Adjustment changed a protected pose')
                            trial = self.checked(trial)
                            better = physical_rank(trial, slot) < physical_rank(parent, slot)
                            exploratory = (branch == 'discrete' and not parent.completed and
                                trial.layout.key() != parent.layout.key() and s.rng.random() < .1)
                            if better or exploratory:
                                self.slots[slot] = trial; self.accepted += 1
                                self.exploration_accepted += int(not better)
                                event.update(accepted=True, reason='physical-rank' if better else 'structural-exploration')
                                if self.continuous_policy == 'pso' and branch == 'discrete':
                                    for k in parent.layout.active:
                                        if (parent.layout.hosts[k] != trial.layout.hosts[k] or not np.array_equal(
                                                parent.layout.placements[k], trial.layout.placements[k])):
                                            self.pso_velocity[slot, k] = 0.
                            if self.continuous_policy == 'pso':
                                self.update_pso_best(slot, self.slots[slot])
                            event.update(move=detail,raw=self.describe(raw), trial=self.describe(trial),protected_pose_parameters_unchanged=True,
                                successful_pose_parameters_unchanged=preserves_poses(parent.layout,trial.layout,
                                    [k for k in parent.layout.active if parent.result['masks'][k].all()]))
                        event['after'] = self.describe(self.slots[slot])
                    except (ValueError, RuntimeError, AssertionError) as error:
                        event['error'] = str(error)
                        self.errors.append(dict(stage='population-soft-slot', slot=slot, error=str(error)))
                    events.append(event)
                    if slot % 8 == 7:
                        print('SOFT SLOTS', s.out, iteration+1, slot+1, '/', self.population_size, flush=True)
                self.protected = ()
                self.certify(self.slots); self.propagate_evidence()
                selected = self.best_checked(start)
                row = dict(phase='population_soft', iteration=iteration+1, population_size=len(self.slots),
                    generated_trials=self.population_size, maximum_trials_per_slot=1, count_penalty_used=False,
                    extra_role_seeds=0, events=events, branches=dict(self.branch_counts),
                    statistics=dict(self.evaluator.stats), retained=[dict(slot=i, **self.describe(x))
                    for i, x in enumerate(self.slots)], incumbent=self.describe(self.incumbent) if self.incumbent else None)
                self.audit.append(row); save(s.out/'population32_trace.json', self.audit); s.record(row)
                m.commit(selected.result)
                signature=(selected.layout.key(),failed_count(selected.result),bool(selected.completed))
                if not self.allow_placement_guidance or signature!=getattr(self,'snapshot_signature',None):
                    s.process_snapshot(selected.result, 'population_soft', dict(iteration=iteration+1))
                    self.snapshot_signature=signature
                print('SOFT32 ROUND', s.out, iteration+1, self.describe(selected), flush=True)
        self.certify(self.slots); selected = self.best_checked(start)
        if self.continuous_policy == 'pso':
            for slot, item in enumerate(self.slots):
                self.update_pso_best(slot, item)
            np.savez_compressed(s.out/'pso_memory.npz', velocity=self.pso_velocity,
                personal_best_placements=np.array([item.layout.placements for item in self.pso_best]),
                personal_best_hosts=np.array([item.layout.hosts for item in self.pso_best]))
        s.frontier_witnesses = self.witnesses; s.frontier_states = [x.result for x in self.slots]
        s.frontier_statistics = dict(self.evaluator.stats, width=self.population_size, rounds=len(self.audit),configured_rounds=s.iterations,
            policy='juxtapose_only' if not self.local_adjustment else 'juxtapose_with_local_adjustment', maximum_retained=self.population_size, population_size=self.population_size,
            generated_trials=len(self.audit)*self.population_size, maximum_trials_per_slot=1, initialization_role_seeds=0,
            warm_start=False, count_penalty_used=False, sampling_count_center=len(current['layout'].active)/3.,
            oracle_fallback_when_all_force_passed=self.allow_oracle_fallback,
            coupled_adjustment_for_oracle_failure=self.allow_coupled_adjustment,
            group_seating_and_global_direction_adjustment=self.allow_group_seating,
            joint_reseating_during_optimization=self.allow_joint_reseating,
            placement_guidance_after_dynamic_role_selection=self.allow_placement_guidance,
            height_repair_after_proposals=self.allow_height_repair,stop_when_feasible=self.stop_when_feasible,
            native_state_sliding_during_optimization=self.allow_native_sliding,
            actual_geometry_validation_beyond_point_screen=self.allow_geometry_validation,
            continuous_proposal_policy=self.continuous_policy,
            local_adjustment_enabled=self.local_adjustment,
            pso_coefficients=dict(w=.25,c1=.25,c2=.25) if self.continuous_policy=='pso' else None,
            pso_personal_best_updates=self.pso_best_updates,
            pso_social_rank_preserves_existing_even_force_odd_oracle_preferences=True,
            branch_counts=self.branch_counts, accepted_updates=self.accepted,
            structural_exploration_updates=self.exploration_accepted, errors=self.errors[-30:])
        s.material_selected = selected
        if selected.witness is not None:
            m.base_oracle.witness = selected.layout.key(), selected.witness
        selected.result['base_oracle'] = selected.oracle; m.commit(selected.result)
        return selected.result
