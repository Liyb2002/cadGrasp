"""Failure-weighted poses and a soft count bias in exactly 32 slots."""
import numpy as np

from population32 import registered_population, centered_seat, POPULATION_SIZE
from material_search import MaterialSearch, freeze_population_measure, sample_id
from whole_search.reuse_first import registered
from whole_search.search import failed_count
from whole_search.common import legal_direction
from continuous_support.fast_gradient import turn
from translation.layout_update import shift
from fixture_state import installed_state


def failure_weights(layout, result):
    if result is None or result['layout'].key() != layout.key():
        raise ValueError('Pose probabilities require full masks for this exact layout')
    return {k: float(np.count_nonzero(~np.asarray(result['masks'][k], dtype=bool))) /
            len(result['masks'][k]) for k in layout.active}


def weighted_pose(indices, weights, rng):
    choices = [int(k) for k in indices if weights.get(k, 0.) > 0.]
    if not choices:
        return None
    probabilities = np.array([weights[k] for k in choices], dtype=float)
    probabilities /= probabilities.sum()
    return int(rng.choice(choices, p=probabilities))


def problem_weights(model, item, allow_oracle_fallback=False):
    weights = failure_weights(item.layout, item.result)
    force_failed = any(weights.values())
    if force_failed and not allow_oracle_fallback:
        return weights, 'original_force_failure_fraction'
    oracle = item.oracle
    if item.completed or not allow_oracle_fallback or oracle.get('passed'):
        return weights, 'original_force_failure_fraction' if force_failed else 'no_failed_pose'
    if oracle.get('enclosure_passed') and oracle.get('status') == 'enclosure_witness':
        return weights, 'original_force_failure_fraction' if force_failed else 'awaiting_connection_certification'
    names = {model.poses[k]: k for k in item.layout.active}
    oracle_weights = dict.fromkeys(item.layout.active, 0.)
    for row in oracle.get('certificates', []):
        for field in ['pose', 'other_pose']:
            if row.get(field) in names:
                oracle_weights[names[row[field]]] += max(0., float(row.get('gap_m', 0.)))
    for row in oracle.get('component_per_pose', []) or oracle.get('per_pose', []):
        if not row.get('passed') and row.get('pose') in names:
            oracle_weights[names[row['pose']]] += max(0., float(row.get('gap_m', 0.)))
    if any(oracle_weights.values()):
        if force_failed:
            scale=max(float(model.extent),1e-12)
            return {k:weights[k]+oracle_weights[k]/scale for k in weights}, 'original_force_plus_oracle_failure_gap'
        return oracle_weights, 'oracle_failure_gap_after_all_force_passed'
    if force_failed:
        return weights, 'original_force_failure_fraction'
    return {k: 1. for k in item.layout.active}, 'unattributed_oracle_failure_after_all_force_passed'


def action_probabilities(layout, owner, weights):
    guests = [k for k in layout.active if not registered(layout, k)]
    reuse = [k for k in layout.active if registered(layout, k)]
    deficit = np.clip(len(layout.active)/3. - len(guests), -3., 3.)
    if owner in reuse:
        values = dict(create=float(np.exp(deficit)))
        if any(weights.get(k, 0.) > 0 for k in guests):
            values['role-exchange'] = 2.
    else:
        values = dict(restore=float(np.exp(-deficit)), reseat=1.)
        if any(weights.get(k, 0.) > 0 for k in reuse):
            values['role-exchange'] = 2.
    total = sum(values.values())
    return {name: value/total for name, value in values.items()}


def replace_guest(model, layout, outgoing, incoming):
    """Try the old seat first, with native rotations; never invent a guest."""
    old_host = int(layout.hosts[outgoing])
    T = installed_state(model.native, layout, old_host)
    old_U = T @ layout.placements[outgoing]
    incoming_T = installed_state(model.native, layout, int(layout.hosts[incoming]))
    incoming_U = incoming_T @ layout.placements[incoming]
    c = np.asarray(model.mesh.center_mass)
    target = incoming_U[:3, :3] @ c + incoming_U[:3, 3]
    target[:2] = (old_U[:3, :3] @ c + old_U[:3, 3])[:2]
    world_direction = incoming_T[:3, :3] @ layout.directions[incoming]
    q = model.restore_registered(layout, outgoing)
    T = installed_state(model.native, q, old_host)
    R = T[:3, :3].T @ model.native[incoming, :3, :3]
    q.hosts[incoming] = old_host
    Q = np.eye(4); Q[:3, :3] = R
    Q[:3, 3] = T[:3, :3].T @ (target-T[:3, 3]) - R @ c
    q.placements[incoming] = Q
    q.directions[incoming] = legal_direction(T[:3, :3].T @ world_direction,
                                             model.floor_normal(q, incoming))
    q = shift(model, q, incoming, np.zeros(3))
    retained_seat = not registered(q, incoming)
    if not retained_seat:
        # Replacing the old anchor in its own registered seat is not a swap.
        q = centered_seat(model, model.restore_registered(layout, outgoing), incoming, outgoing)
    for k in q.active:
        model.validate_placement(q, k)
    return q, retained_seat


def discrete_move(model, layout, weights, rng, owner=None):
    owner = weighted_pose(layout.active, weights, rng) if owner is None else owner
    if owner is None:
        return None, dict(action='hold', reason='no_failed_pose')
    probabilities = action_probabilities(layout, owner, weights)
    names = list(probabilities)
    action = str(rng.choice(names, p=[probabilities[name] for name in names]))
    detail = dict(action=action, selected_pose=owner, action_probabilities=probabilities)
    if action == 'restore':
        return model.restore_registered(layout, owner), dict(detail, pose=owner)
    if action == 'role-exchange':
        if registered(layout, owner):
            incoming = owner
            outgoing = weighted_pose([k for k in layout.active if not registered(layout, k)], weights, rng)
        else:
            outgoing = owner
            incoming = weighted_pose([k for k in layout.active if registered(layout, k)], weights, rng)
        q, kept = replace_guest(model, layout, outgoing, incoming)
        return q, dict(detail, pose=incoming, restored=outgoing, retained_previous_seat=kept)
    anchor = int(rng.choice([k for k in layout.active if k != owner]))
    return centered_seat(model, layout, owner, anchor), dict(
        detail, pose=owner, anchor=anchor, initialization='world_xy_center_of_mass_alignment',
        contact_matching_run=False)


def differential_move(model, layout, population, slot, weights, rng, iteration, owner=None):
    owner = (weighted_pose([k for k in layout.active if not registered(layout, k)], weights, rng)
             if owner is None else owner)
    if owner is None:
        return None, dict(action='hold', reason='no_failed_guest')
    donors = [i for i in range(len(population)) if i != slot]
    a, b = map(int, rng.choice(donors, size=2, replace=False))
    first, second = population[a], population[b]
    c = np.asarray(model.mesh.center_mass)
    ca = first.placements[owner, :3, :3] @ c + first.placements[owner, :3, 3]
    cb = second.placements[owner, :3, :3] @ c + second.placements[owner, :3, 3]
    noise = model.extent*(1/16 if iteration < 8 else 1/64)*rng.normal(size=3)
    q = shift(model, layout, owner, (.6 if iteration < 8 else .3)*(ca-cb)+noise)
    if rng.random() < .35:
        axis = rng.normal(size=3); axis /= max(float(np.linalg.norm(axis)), 1e-12)
        q.directions[owner] = turn(model, q, owner, np.radians(12.)*axis).directions[owner]
    for k in q.active:
        model.validate_placement(q, k)
    return q, dict(selected_pose=owner, poses=[owner], donor_slots=[a, b],
                   coordinate_frame='common_material', better_donors_required=False)


def preserves_poses(before, after, protected):
    for k in protected:
        if int(before.hosts[k]) != int(after.hosts[k]):
            return False
        for old, new in [(before.placements[k], after.placements[k]),
                         (before.directions[k], after.directions[k]),
                         (before.state_offsets[int(before.hosts[k])], after.state_offsets[int(after.hosts[k])])]:
            if not np.allclose(old, new, atol=1e-12, rtol=0.):
                return False
    return True


def physical_rank(item, slot=0):
    if item.completed:
        return (0, 0., 0., 0., 0., *item.cost)
    failures = failed_count(item.result) if item.result is not None else float('inf')
    space = (int(not item.oracle.get('enclosure_passed')), float(item.oracle.get('gap_m', 0.)))
    if slot % 2:
        return (1, *space, float(item.loss), failures, item.lower_litres)
    return (1, float(item.loss), failures, *space, item.lower_litres)


class PopulationSoftSearch(MaterialSearch):
    def __init__(self, search):
        if search.frontier_width != POPULATION_SIZE:
            raise ValueError('population_soft requires 32 slots')
        if (getattr(search, 'initialization_seeds', None) or
                getattr(search, 'material_warm_layouts', []) or
                getattr(search, 'warm_checkpoint_experiment', None)):
            raise ValueError('population_soft is a registered cold initialization')
        super().__init__(search)
        self.slots = []; self.audit = []; self.protected = ()
        self.allow_oracle_fallback = bool(getattr(search, 'allow_oracle_fallback', False))
        self.branch_counts = dict(discrete=0, differential=0, hold=0)
        self.accepted = 0; self.exploration_accepted = 0

    def goal_gap(self, layout):
        return 0

    def choose_pool(self, items, width):
        selected = []; seen = set()
        for slot in range(width):
            available = [x for x in items if x.layout.key() not in seen]
            if not available:
                break
            item = min(available, key=lambda x: physical_rank(x, slot))
            selected.append(item); seen.add(item.layout.key())
        return selected

    def weights(self, item):
        return problem_weights(self.model, item, self.allow_oracle_fallback)

    def adjustable_poses(self, parent):
        weights, _ = self.weights(parent)
        return [k for k in parent.layout.active if weights[k] > 0 and k not in self.protected]

    def adjustable_hosts(self, parent):
        active = set(self.adjustable_poses(parent)); q = parent.layout
        return [h for h in sorted(set(int(q.hosts[k]) for k in q.active))
                if all(k in active for k in q.active if int(q.hosts[k]) == h)]

    def continuation_rows(self, parent, round_index):
        # Keep the original multiple directions, signs and step sizes.
        return [row for row in super().continuation_rows(parent, round_index)
                if preserves_poses(parent.layout, row[1], self.protected)]

    def preview_cached(self, layout, operation='parent', parent='', detail=None, result=None):
        saved = self.evaluated.get(layout.key())
        item = self.preview(layout, operation, parent, detail,
                            result=saved.result if saved else result)
        if saved:
            for name in ['witness', 'nominal', 'solid', 'construction', 'evaluation']:
                setattr(item, name, getattr(saved, name))
            if saved.witness is not None:
                item.oracle = saved.oracle
        return item

    def checked(self, item):
        rows = self.classify([item], 1)
        if not rows:
            raise RuntimeError('Original full-demand classification failed')
        return rows[0]

    def certify(self, items, connections=3, constructions=2):
        super().certify(items, connections, constructions)
        for item in items:
            saved = self.evaluated.get(item.layout.key())
            if saved is None or (item.completed, item.witness is not None) > (saved.completed, saved.witness is not None):
                self.evaluated[item.layout.key()] = item

    def propagate_evidence(self):
        for i, item in enumerate(self.slots):
            self.slots[i] = self.preview_cached(item.layout, item.operation, item.parent, item.detail, item.result)

    def describe(self, item):
        row = super().describe(item)
        row.update(sampling_count_center=len(item.layout.active)/3., count_penalty_used=False)
        if item.result is not None:
            row['per_pose_failed_demands'] = {str(k): int(np.count_nonzero(~item.result['masks'][k]))
                                            for k in item.layout.active}
        return row

    def best_checked(self, start):
        if self.incumbent is not None:
            return self.incumbent
        return min(self.slots+[start], key=lambda x: (
            failed_count(x.result) != 0, not x.oracle.get('enclosure_passed'),
            float(x.oracle.get('gap_m', 0.)), failed_count(x.result), x.loss, x.lower_litres))

    def solve(self, current):
        from co_common import save
        s, m = self.search, self.model
        start = self.preview_cached(current['layout'], result=current)
        self.evaluated[start.layout.key()] = start
        layouts, common = registered_population(m, current['layout'], s.rng)
        initialization = s.out.parent.parent/'step4.1'/s.out.name
        initialization.mkdir(parents=True, exist_ok=False)
        for slot, q in enumerate(layouts):
            np.savez_compressed(initialization/f'slot_{slot:02d}.npz', **q.arrays(m.native))
        save(initialization/'population.json', dict(population_size=32, common_exit_directions=common,
            source='saved original registered Step4.1', warm_start=False,
            predefined_juxtaposed_pose_ids=[], numerical_inputs_changed=False))
        with m.hold_measure(current):
            self.slots = [self.checked(self.preview_cached(q, 'registered-initialize', detail=dict(slot=i)))
                          for i, q in enumerate(layouts)]
            self.certify(self.slots); self.propagate_evidence()
        for iteration in range(s.iterations):
            reference = min(self.slots, key=lambda x: failed_count(x.result)).result
            with m.hold_measure(reference):
                freeze_population_measure(m, self.slots)
                for item in self.slots:
                    item.loss = float(m.proxy(item.layout)['loss'])
                snapshot = [item.layout.copy() for item in self.slots]
                events = []
                for slot, parent in enumerate(list(self.slots)):
                    weights, source = self.weights(parent)
                    self.protected = tuple(k for k in parent.layout.active if weights[k] == 0)
                    total = sum(weights.values())
                    probabilities = {str(k): weights[k]/total if total else 0. for k in weights}
                    owner = weighted_pose(parent.layout.active, weights, s.rng)
                    branch = ('hold' if owner is None else 'discrete' if registered(parent.layout, owner) or s.rng.random() < .5
                              else 'differential')
                    self.branch_counts[branch] += 1
                    event = dict(slot=slot, branch=branch, accepted=False, adjustment_passes=0,
                        pose_probabilities=probabilities, weight_source=source,
                        selected_pose=owner,
                        protected_pose_indices=list(self.protected), before=self.describe(parent))
                    try:
                        if branch == 'hold':
                            event['move'] = dict(action='hold', reason=source)
                        else:
                            if branch == 'discrete':
                                q, detail = discrete_move(m, parent.layout, weights, s.rng, owner=owner)
                            else:
                                q, detail = differential_move(m, parent.layout, snapshot, slot, weights, s.rng, iteration, owner=owner)
                            if q is None or not preserves_poses(parent.layout, q, self.protected):
                                raise ValueError('Mutation changed a protected pose or has no eligible owner')
                            raw = self.checked(self.preview_cached(q, branch, sample_id(parent.layout), detail))
                            event['adjustment_passes'] = 1
                            trial = self.refine_preview(raw, iteration, maximum=1)
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
                            event.update(move=detail, trial=self.describe(trial), successful_pose_parameters_unchanged=True)
                        event['after'] = self.describe(self.slots[slot])
                    except (ValueError, RuntimeError, AssertionError) as error:
                        event['error'] = str(error)
                        self.errors.append(dict(stage='population-soft-slot', slot=slot, error=str(error)))
                    events.append(event)
                    if slot % 8 == 7:
                        print('SOFT32 SLOTS', s.out, iteration+1, slot+1, '/', 32, flush=True)
                self.protected = ()
                self.certify(self.slots); self.propagate_evidence()
                selected = self.best_checked(start)
                row = dict(phase='population_soft', iteration=iteration+1, population_size=len(self.slots),
                    generated_trials=32, maximum_trials_per_slot=1, count_penalty_used=False,
                    extra_role_seeds=0, events=events, branches=dict(self.branch_counts),
                    statistics=dict(self.evaluator.stats), retained=[dict(slot=i, **self.describe(x))
                    for i, x in enumerate(self.slots)], incumbent=self.describe(self.incumbent) if self.incumbent else None)
                self.audit.append(row); save(s.out/'population32_trace.json', self.audit); s.record(row)
                m.commit(selected.result); s.process_snapshot(selected.result, 'population_soft', dict(iteration=iteration+1))
                print('SOFT32 ROUND', s.out, iteration+1, self.describe(selected), flush=True)
        self.certify(self.slots); selected = self.best_checked(start)
        s.frontier_witnesses = self.witnesses; s.frontier_states = [x.result for x in self.slots]
        s.frontier_statistics = dict(self.evaluator.stats, width=32, rounds=s.iterations,
            policy='failure_weighted_soft_count32', maximum_retained=32, population_size=32,
            generated_trials=s.iterations*32, maximum_trials_per_slot=1, initialization_role_seeds=0,
            warm_start=False, count_penalty_used=False, sampling_count_center=len(current['layout'].active)/3.,
            oracle_fallback_when_all_force_passed=self.allow_oracle_fallback,
            branch_counts=self.branch_counts, accepted_updates=self.accepted,
            structural_exploration_updates=self.exploration_accepted, errors=self.errors[-30:])
        s.material_selected = selected
        if selected.witness is not None:
            m.base_oracle.witness = selected.layout.key(), selected.witness
        selected.result['base_oracle'] = selected.oracle; m.commit(selected.result)
        return selected.result
