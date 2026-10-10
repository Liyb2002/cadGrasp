"""Reuse the successful overlapping-seat family; refine with whole-set gradients.

Discrete seeds include native/aligned placement, overlapping offsets and
guest/host direction blends. They select a basin for continuous refinement.
Screening is sampled guidance, never a physical acceptance result.
"""
from collections import OrderedDict
from pathlib import Path
import hashlib
import time
import numpy as np
from continuous_support.projection import project_demands
from whole_search.common import legal_direction
from whole_search.reuse_first import ReuseFirstSearch, registered
from whole_search.search import failed_count


def operation_key(layout, trial, detail, extent):
    guest = detail['guest_index']
    return (tuple(layout.hosts), guest, int(trial.hosts[guest]),
            np.round(trial.placements[guest, :3, :3], 8).tobytes(),
            np.round(trial.placements[guest, :3, 3] / extent, 2).tobytes(),
            detail.get('direction_host_weight', 'native'))


def proposals(objective, current):
    model, layout = objective.model, current.layout
    losses = dict(zip(layout.active, current.residual_loss))
    failing = sorted((k for k in layout.active
                      if current.maximum_residual[layout.active.index(k)] > 2e-8),
                     key=lambda k: -losses[k])
    state = model.contact_delta.state(layout)
    blockers = sorted(layout.active, key=lambda blocker: -sum(
        losses[owner] * float(model.point_areas[
            state.locks[owner, blocker] & (state.coverage_counts[owner] > 0)].sum())
        for owner in failing if owner != blocker))
    guests = list(dict.fromkeys(failing[:3] + blockers[:2]))
    masks = {k: np.asarray([current.maximum_residual[layout.active.index(k)] <= 2e-8])
             for k in layout.active}
    generator = ReuseFirstSearch(model, Path('.'))
    # This is the same proposal generator as the previous successful whole
    # version, not the previous saved answer or its optimized directions.
    rows = generator.juxtapose_proposals(dict(layout=layout, masks=masks), guests, [])
    for guest in layout.active:
        if registered(layout, guest): continue
        trial = layout.copy(); trial.hosts[guest] = guest; trial.placements[guest] = np.eye(4)
        trial.directions[guest] = legal_direction(
            layout.placements[guest, :3, :3].T @ layout.directions[guest],
            model.floor_normal(trial, guest))
        rows.append(('juxtapose-restore', trial, dict(guest=model.poses[guest],
            guest_index=guest, host=model.poses[guest], direction_host_weight='native',
            horizontal_offset_fixture_m=[0., 0., 0.], placement='restore native rotating reuse')))
    return rows


def balanced_subset(rows, budget):
    """Cover hosts, direction seeds and offset scales deterministically."""
    groups = OrderedDict()
    for row in rows:
        detail = row[2]
        groups.setdefault((detail['guest_index'], detail['host']), []).append(row)
    ordered = []
    for items in groups.values():
        # Preserve all three native direction seeds, then visit different
        # offset scales. Six native/aligned seeds per group can exhaust a
        # 96-slot screen before it ever reaches an overlapping offset.
        front = list(range(min(3, len(items))))
        rest = list(range(3, len(items)))
        if rest:
            # Largest/smallest interval bisection gives prefix diversity.
            intervals = [(0, len(rest)-1)]; tail = []
            while intervals:
                a, b = intervals.pop(0)
                mid = (a+b)//2; tail.append(rest[mid])
                if a < mid: intervals.append((a, mid-1))
                if mid < b: intervals.append((mid+1, b))
            ordered.append([items[j] for j in front+tail])
        else: ordered.append([items[j] for j in front])
    return [items[j] for j in range(max((len(x) for x in ordered), default=0))
            for items in ordered if j < len(items)][:budget]


class JumpScreen:
    """Fast fixed all-pose demand quadrature on cached contact availability."""
    def __init__(self, objective, nodes_per_pose=32):
        self.model = objective.model; self.demands = []; self.cache = OrderedDict()
        for distance in objective.distances:
            weights = distance.unique_weights
            # Stratify the original positive measure; include gravity even
            # when a small screen budget would otherwise omit that endpoint.
            samples = np.searchsorted(np.cumsum(weights),
                                      (np.arange(nodes_per_pose)+.5)/nodes_per_pose)
            ids, counts = np.unique(samples, return_counts=True)
            selected = dict(zip(map(int, ids), counts/nodes_per_pose))
            for index in distance.gravity_indices:
                selected.setdefault(int(index), float(weights[index]))
            ids = np.asarray(sorted(selected), int)
            w = np.asarray([selected[int(i)] for i in ids]); w /= w.sum()
            scale = max(1., float(np.linalg.norm(distance.unique_targets, axis=1).max()))
            self.demands.append((distance.unique_targets[ids], w, scale))
        self.evaluations = 0; self.seconds = 0.

    def evaluate(self, layout):
        started = time.monotonic(); flags = self.model.contact_delta.state(layout).available
        losses = []; residuals = []
        for owner in layout.active:
            key = owner, hashlib.sha256(flags[owner].tobytes()).digest()
            if key not in self.cache:
                rays, _ = self.model.supply_at_points(owner, flags[owner])
                targets, weights, scale = self.demands[owner]
                projection = project_demands(rays, targets)
                self.cache[key] = (float(weights @ projection['losses'] / scale**2),
                                   float(np.max(np.abs(projection['residuals']))))
                if len(self.cache) > 256: self.cache.popitem(last=False)
            loss, residual = self.cache[key]; losses.append(loss); residuals.append(residual)
        self.evaluations += 1; self.seconds += time.monotonic()-started
        return dict(loss=float(np.mean(losses)), residual_loss=losses,
                    maximum_residual=residuals, sampled_guidance_only=True)


def juxtapose(objective, current, *, budget=96, full_budget=6, refine=None,
              excluded=None, max_refined=3):
    model, layout = objective.model, current.layout
    excluded = excluded if excluded is not None else set()
    model.contact_delta.commit(layout)
    rows = []; seen = set()
    for row in proposals(objective, current):
        key = operation_key(layout, row[1], row[2], model.extent)
        if key in excluded or row[1].key() in seen or row[1].key() == layout.key(): continue
        seen.add(row[1].key()); rows.append(row)
    selected = balanced_subset(rows, budget)
    screen = getattr(objective, '_jump_screen', None)
    if screen is None: screen = objective._jump_screen = JumpScreen(objective)
    screened = []; failures = []
    for _, candidate, detail in selected:
        key = operation_key(layout, candidate, detail, model.extent)
        excluded.add(key)
        try: screened.append((screen.evaluate(candidate), candidate, detail))
        except (RuntimeError, ValueError) as error:
            failures.append(dict(**detail, error=str(error), accepted=False))
    screened.sort(key=lambda row: row[0]['loss'])
    # Reserve distinct guest/host branches, then fill with other good seeds.
    finalists = []; used = set(); chosen = set()
    for row in screened:
        pair = row[2]['guest_index'], row[2]['host']
        if pair in used: continue
        finalists.append(row); used.add(pair); chosen.add(row[1].key())
        if len(finalists) >= min(3, full_budget): break
    for row in screened:
        if len(finalists) >= full_budget: break
        if row[1].key() not in chosen:
            finalists.append(row); chosen.add(row[1].key())
    trials = []; evaluated = []
    for preview, candidate, detail in finalists:
        result = objective.evaluate(candidate); evaluated.append((result, detail))
        check = None
        if objective.covered(result):
            checked = model.evaluate(candidate)
            check = dict(counts=checked['counts'], passed=failed_count(checked)==0,
                         contact_geometry='sampled', actual_mesh_acceptance=False)
        trials.append(dict(**detail, screen=preview, original_load_screen=check,
                           coverage=result.coverage.tolist(),
                           residual_loss=result.residual_loss.tolist(),
                           estimated_material_cm3=result.volume_cm3))
    load_checks = {candidate.layout.key(): trial['original_load_screen']
                   for (candidate, _), trial in zip(evaluated, trials)}
    evaluated.sort(key=lambda row: (not objective.covered(row[0]) or
                                    (load_checks[row[0].layout.key()] is not None and
                                     not load_checks[row[0].layout.key()]['passed']),
                                    objective.coverage_loss(row[0]), row[0].volume_cm3))
    branches = []
    for result, detail in evaluated[:max_refined]:
        raw = result.layout
        jump = dict(placements=raw.placements.tolist(), directions=raw.directions.tolist(),
                    hosts=raw.hosts.tolist(), active=list(raw.active))
        records = []
        if refine is not None and not objective.covered(result): result, records = refine(result, None)
        branches.append((result, detail, jump, records))
    accepted = False; best = current; decision = None
    for branch in branches:
        check = load_checks.get(branch[0].layout.key())
        if objective.covered(branch[0]):
            if check is None:
                checked = model.evaluate(branch[0].layout)
                check = dict(passed=failed_count(checked)==0)
            if not check['passed']: continue
        if objective.better(branch[0], best): best = branch[0]; decision = branch; accepted = True
    model.contact_delta.commit(best.layout)
    report = dict(operation='juxtapose', accepted=accepted, all_poses_active=True,
                  proposal_family='previous successful rotating-reuse whole search',
                  deterministic_discrete_seeds=True, random_design_sampling=False,
                  generated_discrete_placements=len(rows), screened_discrete_placements=len(selected),
                  screen_seconds=screen.seconds, screening_is_acceptance=False,
                  discrete_trials=trials, screening_failures=failures,
                  refined_branches=[dict(selected=d, loss=objective.coverage_loss(r),
                                         coverage=r.coverage.tolist(), operations=records)
                                    for r, d, _, records in branches])
    if decision is not None:
        _, detail, jump, records = decision
        report.update(selected=detail, jump_layout=jump, continuous_refinement=records)
    return best, report
