"""Continue the cold population with moving particles and certified incumbents.

The established cold search remains reproducible. Extra refinement is an
explicit, separately budgeted phase; temporary particle states may fail the
original force gates, but only fully constructed certified solids can win.
"""
import numpy as np

from population_soft import (PopulationSoftSearch, physical_rank, weighted_pose,
    discrete_move, differential_move)
from material_search import freeze_population_measure, sample_id
from translation.layout_update import shift
from continuous_support.fast_gradient import turn
from whole_search.reuse_first import registered
from whole_search.search import failed_count

PHI = 4.1
CHI = 2. / (PHI - 2. + np.sqrt(PHI * PHI - 4. * PHI))


def center(model, layout, owner):
    q = layout.placements[owner]
    return q[:3, :3] @ np.asarray(model.mesh.center_mass) + q[:3, 3]


def trajectory_move(model, layout, personal, social, velocity, rng, owner, limit):
    before = center(model, layout, owner)
    inertia = CHI * velocity[owner]
    cognitive = CHI * PHI / 2. * rng.random(3) * (center(model, personal, owner) - before)
    collective = CHI * PHI / 2. * rng.random(3) * (center(model, social, owner) - before)
    proposed = inertia + cognitive + collective
    proposed *= min(1., limit / max(float(np.linalg.norm(proposed)), 1e-30))
    q = shift(model, layout, owner, proposed)
    # Persist actual displacement after the existing legal-floor projection.
    velocity[owner] = center(model, q, owner) - before
    if rng.random() < .35:
        axis = rng.normal(size=3)
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        q.directions[owner] = turn(model, q, owner, np.radians(12.) * axis).directions[owner]
    for k in q.active:
        model.validate_placement(q, k)
    return q, dict(selected_pose=owner, coordinate_frame='common_material',
        inertia_component_m=inertia.tolist(), cognitive_component_m=cognitive.tolist(),
        social_component_m=collective.tolist(), proposed_velocity_m=proposed.tolist(),
        velocity_m=velocity[owner].tolist(), velocity_limit_m=limit,
        coefficients=dict(w=float(CHI), c1=float(CHI*PHI/2.), c2=float(CHI*PHI/2.)),
        translation_noise_added=False, role_and_host_changed=False)


class PopulationPSORefine(PopulationSoftSearch):
    def __init__(self, search):
        super().__init__(search)
        self.refinement_rounds = int(search.pso_refinement_rounds)
        self.refinement_policy = getattr(search, 'pso_refinement_policy', 'pso')
        self.refinement_seed = int(getattr(search, 'pso_refinement_seed', 42))
        if self.refinement_rounds < 1 or self.refinement_policy not in ('pso', 'differential'):
            raise ValueError('Invalid explicitly budgeted population refinement')

    def keep_motion(self, slot, raw, adjusted):
        """Moving state and best observed state have independent lifetimes."""
        self.slots[slot] = raw
        self.update_pso_best(slot, raw)
        self.update_pso_best(slot, adjusted)

    def refinement_weights(self, item):
        if item.completed or (all(mask.all() for mask in item.result['masks'].values())
                              and item.oracle.get('enclosure_passed')):
            return dict.fromkeys(item.layout.active, 1.), 'force_feasible_volume_refinement'
        return self.weights(item)

    def final_candidate(self, cold_selected):
        if self.incumbent is not None:
            return self.incumbent
        # Failed current positions must not erase a better observed diagnostic
        # candidate. Feasibility still requires full construction and all gates.
        return min(self.slots+self.pso_best+[cold_selected], key=lambda x: (
            failed_count(x.result)!=0, not x.oracle.get('enclosure_passed'),
            float(x.oracle.get('gap_m',0.)), failed_count(x.result), x.loss, x.lower_litres))

    def solve(self, current):
        from co_common import save
        from search_checkpoint import save_checkpoint
        from search_runtime import runtime
        s, m = self.search, self.model
        original = super().solve(current)
        store=runtime(m)
        resumed=store.restored_checkpoint if store else None
        continuation=resumed is not None and resumed['phase']=='refine'
        cold_selected = s.material_selected
        first = self.incumbent
        cold = dict(s.frontier_statistics)
        if first is None:
            # Extra PSO rounds also apply to unfinished cold candidates. No
            # saved historical answer is used to manufacture a feasible seed.
            baseline = None
        else:
            baseline = dict(id=sample_id(first.layout), bbx_l=float(first.cost[0]),
                material_cm3=float(first.cost[1]))
        if not continuation:
            save(s.out/'before_refinement.json', dict(complete=True, cold_search=cold,
                certified_incumbent=baseline, warm_start=False, extra_rounds=self.refinement_rounds))
        rng = np.random.default_rng(self.refinement_seed)
        velocities = rng.uniform(-m.extent/32., m.extent/32., self.pso_velocity.shape)
        if not continuation:self.pso_best = list(self.slots)
        audit = []
        next_iteration=0
        if continuation:
            extra=resumed['extra']
            cold_selected=extra['cold_selected'];first=extra['first'];cold=extra['cold']
            baseline=extra['baseline'];velocities=extra['velocities'];audit=extra['audit']
            rng.bit_generator.state=extra['refinement_rng']
            next_iteration=resumed['next_iteration']
        for iteration in range(next_iteration,self.refinement_rounds):
            reference = self.best_checked(self.slots[0]).result
            with m.hold_measure(reference):
                freeze_population_measure(m, self.slots)
                for item in self.slots + self.pso_best:
                    item.loss = float(m.proxy(item.layout)['loss'])
                for slot, item in enumerate(self.slots):
                    self.update_pso_best(slot, item)
                personal = [item.layout.copy() for item in self.pso_best]
                exemplars = self.pso_best + ([self.incumbent] if self.incumbent else [])
                social = [min(exemplars, key=lambda x: physical_rank(x, parity)).layout.copy()
                          for parity in (0, 1)]
                snapshot = [item.layout.copy() for item in self.slots]
                candidates = list(self.pso_best)
                events = []
                # This is a proposal bound, never a physical acceptance limit.
                fraction = 1/8. * (1/4.)**(iteration/max(self.refinement_rounds-1, 1))
                for slot, parent in enumerate(list(self.slots)):
                    self.active_slot = slot
                    self.protected = ()
                    weights, source = self.refinement_weights(parent)
                    owner = weighted_pose(parent.layout.active, weights, rng)
                    branch = ('hold' if owner is None else 'discrete' if registered(parent.layout, owner)
                              else self.refinement_policy)
                    event = dict(slot=slot, branch=branch, selected_pose=owner,
                        weight_source=source, before=self.describe(parent), accepted=False)
                    try:
                        if branch == 'hold':
                            events.append(event)
                            continue
                        if branch == 'discrete':
                            q, detail = discrete_move(m, parent.layout, weights, rng, owner=owner,
                                carry_group=self.allow_group_seating, allow_native_slide=self.allow_native_sliding)
                        elif branch == 'pso':
                            q, detail = trajectory_move(m, parent.layout, personal[slot], social[slot%2],
                                velocities[slot], rng, owner, m.extent*fraction)
                        else:
                            q, detail = differential_move(m, parent.layout, snapshot, slot,
                                weights, rng, iteration, owner=owner)
                        if self.allow_height_repair:
                            q = self.repair_height(q)
                        raw = self.checked(self.preview_cached(q, branch, sample_id(parent.layout), detail))
                        adjusted = self.checked_adjustment(raw, cold['rounds']+iteration, slot, parent=parent)
                        candidates.extend([raw, adjusted])
                        if branch in ('pso', 'differential'):
                            # The same moving-state rule applies to the matched
                            # differential control; infeasible states remain
                            # strictly search states and can never become PASS.
                            self.keep_motion(slot, raw, adjusted)
                            if branch == 'pso':
                                velocities[slot, owner] = center(m, raw.layout, owner)-center(m, parent.layout, owner)
                            event.update(accepted=True, reason='particle-motion')
                        else:
                            # Existing reseating proposals can change topology
                            # during refinement even when the incumbent is
                            # already complete. Otherwise all-registered
                            # populations never acquire a continuous variable.
                            self.keep_motion(slot, adjusted, raw)
                            for k in parent.layout.active:
                                if parent.layout.hosts[k] != adjusted.layout.hosts[k] or not np.array_equal(
                                        parent.layout.placements[k], adjusted.layout.placements[k]):
                                    velocities[slot, k] = rng.uniform(-m.extent/32., m.extent/32., 3)
                            event.update(accepted=True, reason='reseating-motion')
                        event.update(move=detail, raw=self.describe(raw), trial=self.describe(adjusted),
                            after=self.describe(self.slots[slot]))
                    except (ValueError, RuntimeError, AssertionError) as error:
                        event['error'] = str(error)
                        self.errors.append(dict(stage='pso-refinement', error=str(error)))
                    events.append(event)
                    if slot%8 == 7:
                        print('REFINEMENT SLOTS', s.out, iteration+1, slot+1, flush=True)
                self.certify(candidates)
                self.propagate_evidence()
                for slot, item in enumerate(self.slots):
                    self.update_pso_best(slot, item)
                row = dict(phase='population_refinement', iteration=iteration+1,
                    policy=self.refinement_policy, events=events, retained=[self.describe(x) for x in self.slots],
                    incumbent=self.describe(self.incumbent) if self.incumbent else None)
                audit.append(row)
                save(s.out/'pso_refinement_trace.json', audit)
                s.record(row)
                selected = self.best_checked(self.slots[0])
                m.commit(selected.result)
                print('REFINEMENT ROUND', s.out, iteration+1, self.describe(selected), flush=True)
            save_checkpoint(self,'refine',iteration+1,cold_result=original,
                cold_selected=cold_selected,first=first,cold=cold,baseline=baseline,
                velocities=velocities,audit=audit,refinement_rng=rng.bit_generator.state)
        self.certify(self.pso_best)
        selected = self.final_candidate(cold_selected)
        if first is not None:
            assert selected.completed and tuple(selected.cost) <= tuple(first.cost)
        s.frontier_statistics = dict(cold, refinement_rounds=len(audit),
            refinement_policy=self.refinement_policy, refinement_seed=self.refinement_seed,
            refinement_generated_trials=len(audit)*self.population_size,
            refinement_baseline=baseline, refinement_selected_cost=list(selected.cost) if selected.completed else None,
            refinement_coefficients=dict(w=float(CHI), c1=float(CHI*PHI/2.), c2=float(CHI*PHI/2.)),
            refinement_current_position_can_be_worse=True,
            refinement_statistics={k:v-cold.get(k,0) for k,v in self.evaluator.stats.items()},
            total_search_statistics=dict(self.evaluator.stats),
            refinement_uses_only_current_cold_search_candidates=True)
        np.savez_compressed(s.out/'pso_refinement_memory.npz', velocity=velocities,
            personal_best_placements=np.array([x.layout.placements for x in self.pso_best]))
        s.frontier_witnesses = self.witnesses
        s.frontier_states = [x.result for x in self.slots]
        s.material_selected = selected
        if selected.witness is not None:
            m.base_oracle.witness = selected.layout.key(), selected.witness
        selected.result['base_oracle'] = selected.oracle
        m.commit(selected.result)
        return selected.result
