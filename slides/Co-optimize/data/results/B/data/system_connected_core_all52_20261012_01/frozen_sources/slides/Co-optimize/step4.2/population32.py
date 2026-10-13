"""Exactly 32 slots: one stochastic move, one bounded adjustment per slot."""
from dataclasses import replace
import numpy as np

from material_search import MaterialSearch, freeze_population_measure, sample_id
from material_evaluator import membership, target_gap
from whole_search.reuse_first import registered
from whole_search.common import legal_direction
from whole_search.search import failed_count
from continuous_support.fast_gradient import turn
from translation.layout_update import shift
from fixture_state import installed_state

POPULATION_SIZE = 32


def registered_population(model, initial, rng, size=POPULATION_SIZE):
    if any(not registered(initial, k) for k in initial.active):
        raise ValueError('Population must start with every task registered')
    common = bool(np.max(np.linalg.norm(initial.directions[list(initial.active)] -
                                   initial.directions[initial.active[0]], axis=1)) < 1e-8)
    rows = [initial.copy()]
    for slot in range(1, size):
        q = initial.copy()
        if not common:
            for k in q.active:
                axis = rng.normal(size=3)
                axis /= max(float(np.linalg.norm(axis)), 1e-12)
                q.directions[k] = turn(model, q, k,
                    np.radians(float(rng.uniform(-.125, .125))) * axis).directions[k]
        rows.append(q)
    return rows, common


def centered_seat(model, layout, guest, anchor):
    """Keep native task rotation, match WORLD XY COM, preserve guest height."""
    q = layout.copy()
    host = int(layout.hosts[anchor])
    T = installed_state(model.native, layout, host)
    old_T = installed_state(model.native, layout, int(layout.hosts[guest]))
    c = np.asarray(model.mesh.center_mass)
    anchor_U = T @ layout.placements[anchor]
    guest_U = old_T @ layout.placements[guest]
    target = guest_U[:3, :3] @ c + guest_U[:3, 3]
    target[:2] = (anchor_U[:3, :3] @ c + anchor_U[:3, 3])[:2]
    R = T[:3, :3].T @ model.native[guest, :3, :3]
    Q = np.eye(4)
    Q[:3, :3] = R
    Q[:3, 3] = T[:3, :3].T @ (target - T[:3, 3]) - R @ c
    q.hosts[guest] = host
    q.placements[guest] = Q
    world_direction = old_T[:3, :3] @ layout.directions[guest]
    q.directions[guest] = legal_direction(T[:3, :3].T @ world_direction,
                                         model.floor_normal(q, guest))
    q = shift(model, q, guest, np.zeros(3))
    for k in q.active:
        model.validate_placement(q, k)
    return q


def discrete_move(model, layout, rng):
    """One internal operation; identities are chosen during optimization."""
    owner = int(rng.choice(layout.active))
    guests = [k for k in layout.active if not registered(layout, k)]
    reuse = [k for k in layout.active if registered(layout, k)]
    choice = float(rng.random())
    if owner in guests and choice < .2:
        return model.restore_registered(layout, owner), dict(action='restore', pose=owner)
    if owner in guests and reuse and choice < .3:
        other = int(rng.choice(reuse))
        q = model.restore_registered(layout, owner)
        return centered_seat(model, q, other, owner), dict(
            action='role-exchange', restored=owner, pose=other, anchor=owner)
    anchors = [k for k in layout.active if k != owner]
    anchor = int(rng.choice(anchors))
    return centered_seat(model, layout, owner, anchor), dict(
        action='create' if owner in reuse else 'reseat', pose=owner, anchor=anchor,
        initialization='world_xy_center_of_mass_alignment', contact_matching_run=False)


def differential_move(model, layout, population, slot, rng, iteration):
    guests = [k for k in layout.active if not registered(layout, k)]
    donors = [i for i in range(len(population)) if i != slot]
    if not guests or len(donors) < 2:
        return None, dict(reason='no_juxtaposed_pose_or_donors')
    a, b = map(int, rng.choice(donors, size=2, replace=False))
    first, second = population[a], population[b]
    q = layout.copy()
    c = np.asarray(model.mesh.center_mass)
    forced = int(rng.choice(guests)); moved = []
    for k in guests:
        if k != forced and rng.random() > .65:
            continue
        ca = first.placements[k, :3, :3] @ c + first.placements[k, :3, 3]
        cb = second.placements[k, :3, :3] @ c + second.placements[k, :3, 3]
        noise = model.extent * (1/16 if iteration < 8 else 1/64) * rng.normal(size=3)
        q = shift(model, q, k, (.6 if iteration < 8 else .3) * (ca-cb) + noise)
        moved.append(k)
    for k in q.active:
        if rng.random() < .35:
            axis = rng.normal(size=3); axis /= max(float(np.linalg.norm(axis)), 1e-12)
            q.directions[k] = turn(model, q, k, np.radians(12.) * axis).directions[k]
        model.validate_placement(q, k)
    return q, dict(donor_slots=[a, b], poses=moved,
                   coordinate_frame='common_material', better_donors_required=False)


def choose_branch(layout, rng, discrete_probability=.5):
    if all(registered(layout, k) for k in layout.active):
        return 'discrete'
    return 'discrete' if rng.random() < discrete_probability else 'differential'


def slot_rank(item, slot):
    goal = target_gap(item.layout)
    if item.completed:
        return (goal, 0, 0., 0., *item.cost)
    gap = float(item.oracle.get('gap_m', 0.))
    space = (int(not item.oracle.get('enclosure_passed')), gap)
    force = float(item.loss)
    # Slot preferences preserve force/space alternatives without another pool.
    if slot % 2:
        return (goal, 1, *space, force, item.lower_litres)
    return (goal, 1, force, *space, item.lower_litres)


class Population32Search(MaterialSearch):
    def __init__(self, search):
        if search.frontier_width != POPULATION_SIZE:
            raise ValueError('population32 requires exactly 32 persistent slots')
        if (getattr(search, 'initialization_seeds', None) or
                getattr(search, 'material_warm_layouts', []) or
                getattr(search, 'warm_checkpoint_experiment', None)):
            raise ValueError('population32 uses registered initialization, not warm layouts')
        super().__init__(search)
        self.slots = []
        self.branch_counts = dict(discrete=0, differential=0)
        self.accepted = 0

    def preview_cached(self, q, operation, parent='', detail=None):
        saved = self.evaluated.get(q.key())
        item = self.preview(q, operation, parent, detail, result=saved.result if saved else None)
        if saved:
            for name in ['witness', 'nominal', 'solid', 'construction', 'evaluation']:
                setattr(item, name, getattr(saved, name))
            if saved.witness is not None:
                item.oracle = saved.oracle
        return item

    def continuation_rows(self, parent, round_index):
        rows = super().continuation_rows(parent, round_index)
        # One local pass with a bounded mix of existing physical proposals.
        selected = []; seen = set()
        for family in ['material-floor-projection', 'installation-floor-program',
                       'base-opening-installation', 'direction-coherent-gradient',
                       'direction-coherent-event', 'seat-gradient', 'seat-event',
                       'installation-gradient', 'direction-gradient', 'direction-event']:
            choices = [row for row in rows if row[0] == family]
            if choices:
                row = choices[round_index % len(choices)]
                if row[1].key() not in seen:
                    selected.append(row); seen.add(row[1].key())
        return selected[:10]

    def checked_slots(self, budget=8, final=False):
        pending = [item for item in self.slots if item.result is None]
        checked = self.classify(pending, budget)
        for item in checked:
            self.evaluated[item.layout.key()] = item
        # Identical slots share only evidence for that exact layout key.
        for item in self.slots:
            saved = self.evaluated.get(item.layout.key())
            if saved and item.result is None:
                item.result = saved.result
        self.certify([x for x in self.slots if x.result is not None])
        return checked

    def best_checked(self, fallback):
        if self.incumbent is not None:
            return self.incumbent
        checked = [x for x in self.slots if x.result is not None]
        return min(checked, key=lambda x: (target_gap(x.layout),
            not x.oracle.get('enclosure_passed'), failed_count(x.result),
            x.loss, x.lower_litres)) if checked else fallback

    def solve(self, current):
        from co_common import save
        s, m = self.search, self.model
        start = self.preview(current['layout'], result=current)
        self.evaluated[start.layout.key()] = start
        layouts, common = registered_population(m, current['layout'], s.rng)
        initialization = s.out.parent.parent / 'step4.1' / s.out.name
        initialization.mkdir(parents=True, exist_ok=False)
        for slot, q in enumerate(layouts):
            np.savez_compressed(initialization/f'slot_{slot:02d}.npz', **q.arrays(m.native))
        save(initialization/'population.json', dict(population_size=POPULATION_SIZE,
            source='saved original registered Step4.1', common_exit_directions=common,
            registered_pose_count=len(current['layout'].active), warm_start=False,
            predefined_juxtaposed_pose_ids=[], numerical_inputs_changed=False))
        with m.hold_measure(current):
            self.slots = [self.preview_cached(q, 'registered-initialize', detail=dict(slot=i))
                          for i, q in enumerate(layouts)]
            self.checked_slots()
        print('POPULATION32 INITIAL', s.out, 'slots', len(self.slots), flush=True)
        self.audit = []
        for iteration in range(s.iterations):
            actual = [x for x in self.slots if x.result is not None] or [start]
            reference = min(actual, key=lambda x: (failed_count(x.result), x.loss)).result
            with m.hold_measure(reference):
                freeze_population_measure(m, actual)
                for item in self.slots:
                    item.loss = float(m.proxy(item.layout)['loss'])
                snapshot = [x.layout.copy() for x in self.slots]
                events = []
                for slot in range(POPULATION_SIZE):
                    parent = self.slots[slot]
                    branch = choose_branch(parent.layout, s.rng)
                    self.branch_counts[branch] += 1
                    event = dict(slot=slot, branch=branch, accepted=False, adjustment_passes=1)
                    try:
                        if branch == 'discrete':
                            q, detail = discrete_move(m, parent.layout, s.rng)
                        else:
                            q, detail = differential_move(m, parent.layout, snapshot, slot, s.rng, iteration)
                        if q is None:
                            raise ValueError('Selected branch has no applicable move')
                        raw = self.preview_cached(q, branch, sample_id(parent.layout), detail)
                        trial = self.refine_preview(raw, iteration, maximum=1)
                        better = slot_rank(trial, slot) < slot_rank(parent, slot)
                        exploratory = (branch == 'discrete' and not parent.completed and
                            trial.layout.key() != parent.layout.key() and
                            target_gap(trial.layout) <= target_gap(parent.layout) and s.rng.random() < .1)
                        if better or exploratory:
                            self.slots[slot] = trial
                            self.accepted += 1
                            event.update(accepted=True, reason='rank' if better else 'structural-exploration')
                        event.update(move=detail, before=self.describe(parent),
                                     after=self.describe(self.slots[slot]))
                    except (ValueError, RuntimeError, AssertionError) as error:
                        event['error'] = str(error)
                        self.errors.append(dict(stage='population32-slot', slot=slot, error=str(error)))
                    events.append(event)
                    if slot % 8 == 7:
                        print('POPULATION32 SLOTS', s.out.parent.parent.parent.name, iteration+1,
                              slot+1, '/', POPULATION_SIZE, flush=True)
                self.checked_slots()
                selected = self.best_checked(start)
                row = dict(phase='population32', iteration=iteration+1,
                    population_size=len(self.slots), generated_trials=POPULATION_SIZE,
                    maximum_trials_per_slot=1, extra_role_seeds=0, events=events,
                    branches=dict(self.branch_counts), statistics=dict(self.evaluator.stats),
                    retained=[dict(slot=i, **self.describe(x)) for i,x in enumerate(self.slots)],
                    incumbent=self.describe(self.incumbent) if self.incumbent else None)
                self.audit.append(row)
                save(s.out/'population32_trace.json', self.audit)
                s.record(row)
                m.commit(selected.result)
                s.process_snapshot(selected.result, 'population32', dict(iteration=iteration+1))
                print('POPULATION32 ROUND', s.out, iteration+1, self.describe(selected), flush=True)
        print('POPULATION32 FINAL CHECK', s.out, flush=True)
        self.checked_slots(POPULATION_SIZE, final=True)
        selected = self.best_checked(start)
        if selected.result is None:
            raise ValueError('No final layout has original full-demand evidence')
        s.frontier_witnesses = self.witnesses
        s.frontier_states = [x.result for x in self.slots if x.result is not None]
        s.frontier_statistics = dict(self.evaluator.stats, width=POPULATION_SIZE,
            rounds=s.iterations, policy='fixed_population32', maximum_retained=POPULATION_SIZE,
            population_size=len(self.slots), generated_trials=s.iterations*POPULATION_SIZE,
            maximum_trials_per_slot=1, initialization_role_seeds=0, warm_start=False,
            branch_counts=self.branch_counts, accepted_updates=self.accepted,
            target_satisfied=not target_gap(selected.layout), errors=self.errors[-30:])
        s.material_selected = selected
        if selected.witness is not None:
            m.base_oracle.witness = selected.layout.key(), selected.witness
        selected.result['base_oracle'] = selected.oracle
        m.commit(selected.result)
        return selected.result
