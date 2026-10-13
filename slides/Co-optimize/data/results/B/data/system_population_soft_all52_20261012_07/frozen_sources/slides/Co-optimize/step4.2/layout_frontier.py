"""Persistent layout alternatives scored on force loss AND base feasibility.

Only the bounded shortlist runs the original full-demand classification. All
parents and children in a decision share the same positive demand quadrature.
Oracle UNKNOWN states remain explorable; a connected feasible witness is kept.
"""
from dataclasses import dataclass, field
import hashlib
import numpy as np

from whole_search.search import failed_count, lost_protected
from whole_search.reuse_first import registered
from translation.installation import installation_proposals


@dataclass(eq=False)
class Sample:
    layout: object
    loss: float
    oracle: dict
    operation: str = 'parent'
    detail: dict = field(default_factory=dict)
    result: object = None
    parent: str = ''

    @property
    def feasible(self):
        return (self.result is not None and failed_count(self.result) == 0
                and bool(self.oracle.get('connected_passed', False))
                and bool(self.oracle.get('passed', False)))

    @property
    def metrics(self):
        return (self.loss, int(not self.oracle.get('enclosure_passed', False)),
                float(self.oracle.get('gap_m', 0.)))


def identity(layout):
    return hashlib.sha256(repr(layout.key()).encode()).hexdigest()[:16]


def geometry_features(sample, extent):
    """Directions, seats, installations and attainable foot *positions*.

    Ground support functions distinguish different extreme foot locations even
    when both enclosures pass. No ground area is used as bearing capacity.
    """
    q = sample.layout; ids = list(q.active); scale = max(float(extent), 1e-12)
    values = [q.directions[ids].ravel(), q.placements[ids, :3, 3].ravel()/scale,
              q.state_offsets[q.hosts[ids]].ravel()/scale, q.hosts[ids].astype(float)]
    # Use every task's installation offset so feature sizes never depend on
    # how many fixture states the layout currently has.
    angles = np.arange(8)*np.pi/4
    axes = np.c_[np.cos(angles), np.sin(angles)]
    rows = {row.get('index', row.get('pose')): row
            for row in sample.oracle.get('per_pose', [])}
    ordered_rows = sample.oracle.get('per_pose', [])
    for order, k in enumerate(ids):
        row = rows.get(k, ordered_rows[order] if order < len(ordered_rows) else {})
        points = np.asarray(row.get('legal_ground_xy_m', []), float).reshape(-1, 2)
        required = np.asarray(row.get('demand_hull_xy_m', []), float).reshape(-1, 2)
        center = required.mean(0) if len(required) else np.zeros(2)
        values.append(((points-center)@axes.T).max(0)/scale if len(points)
                      else np.full(8, -10.))
    feature = np.concatenate(values)
    # Bound large cohort offsets without imposing any physical space limit.
    return feature/(1.+np.abs(feature))


def select_samples(samples, width, extent):
    """Keep force/base tradeoffs, a feasible incumbent, and distinct geometry."""
    if width < 1:
        raise ValueError('Frontier width must be positive')
    unique = {}
    for item in samples:
        key = item.layout.key()
        old = unique.get(key)
        if old is None or (item.feasible, item.result is not None) > (old.feasible, old.result is not None):
            unique[key] = item
    pool = list(unique.values())
    if not pool:
        return []
    chosen = []

    def keep(item):
        if len(chosen) < width and all(item is not old for old in chosen):
            chosen.append(item)

    feasible = [item for item in pool if item.feasible]
    if feasible:
        keep(min(feasible, key=lambda item: item.result['volume_cm3']))
    keep(min(pool, key=lambda item: item.metrics))
    keep(min(pool, key=lambda item: (item.metrics[1], item.metrics[2], item.loss)))
    features = {id(item): geometry_features(item, extent) for item in pool}
    while len(chosen) < min(width, len(pool)):
        remaining = [item for item in pool if all(item is not old for old in chosen)]
        nondominated = [item for item in remaining if not any(
            all(a <= b for a, b in zip(other.metrics, item.metrics))
            and any(a < b for a, b in zip(other.metrics, item.metrics))
            for other in pool if other is not item)]
        candidates = nondominated or remaining
        def priority(item):
            distance = min(float(np.linalg.norm(features[id(item)]-features[id(old)]))
                           for old in chosen)
            return (-distance, *item.metrics)
        keep(min(candidates, key=priority))
    return chosen


def bounded_proposals(rows, budget, rng):
    """Share one budget across operation/pose/host groups; keep paired moves."""
    groups = {}; seen = set()
    for kind, layout, detail in rows:
        key = layout.key()
        if key in seen:
            continue
        seen.add(key)
        subject = detail.get('pose', detail.get('guest_index', tuple(detail.get('poses', ()))))
        group = kind, str(subject), str(detail.get('host', detail.get('hosts', '')))
        groups.setdefault(group, []).append((kind, layout, detail))
    families = {}
    for key, group in groups.items():
        rest = list(rng.permutation(len(group)-1)+1) if len(group) > 1 else []
        kind = key[0]
        family = ('oracle-cohort' if kind.startswith('oracle-cohort') else
                  'oracle-fixture' if kind.startswith('oracle-fixture') else kind.split('-')[0])
        families.setdefault(family, []).append([group[0]]+[group[int(i)] for i in rest])
    # Interleave operation families before subjects. A pool of independent
    # direction candidates must not starve cohort/base-repair alternatives.
    ordered = []
    for family in families.values():
        interleaved = []; depth = 0
        while any(depth < len(group) for group in family):
            interleaved.extend(group[depth] for group in family if depth < len(group))
            depth += 1
        ordered.append(interleaved)
    output = []; depth = 0
    while len(output) < budget:
        available = False
        for group in ordered:
            if depth < len(group):
                output.append(group[depth]); available = True
                if len(output) == budget:
                    break
        if not available:
            break
        depth += 1
    return output


class LayoutFrontier:
    def __init__(self, search):
        self.search = search; self.model = search.model
        self.width = search.frontier_width
        self.witnesses = {}; self.connected_attempts = set()
        self.stats = dict(width=self.width, rounds=0, maximum_retained=1,
                          screened_candidates=0, full_demand_candidates=0,
                          connection_candidates=0, shared_quadrature=True,
                          parents_compete_with_children=True,
                          ranking='force_loss_and_base_enclosure_with_geometry_diversity')

    def sample(self, result, operation='parent', detail=None, parent=''):
        oracle = result.get('base_oracle')
        if oracle is None:
            oracle = self.model.base_oracle.query(result['layout'])
            result['base_oracle'] = oracle
        loss = float(self.model.proxy(result['layout'])['loss'])
        return Sample(result['layout'], loss, oracle, operation, detail or {}, result, parent)

    def connected(self, samples):
        eligible = [item for item in samples if item.result is not None
                    and failed_count(item.result) == 0
                    and item.oracle.get('enclosure_passed', False)]
        eligible.sort(key=lambda item: item.result['volume_cm3'])
        for item in eligible:
            if item.feasible:
                return item
        pending = [item for item in eligible if item.layout.key() not in self.connected_attempts]
        for item in pending[:self.search.finalists]:
            key = item.layout.key()
            if key not in self.connected_attempts:
                self.connected_attempts.add(key); self.stats['connection_candidates'] += 1
                item.oracle = self.model.base_oracle.query(item.layout, connected=True)
                item.result['base_oracle'] = item.oracle
                witness = getattr(self.model.base_oracle, 'witness', (None, None))
                if item.feasible and witness[0] == key:
                    self.witnesses[key] = witness[1]
            if item.feasible:
                return item
        return None

    def proposals(self, parent, iteration, jumps):
        s, m = self.search, self.model
        m.commit(parent.result)
        targets, worst, _ = s.targets(parent.result)
        movable = [k for k in parent.layout.active if not registered(parent.layout, k)]
        rows, _ = s.local_proposals(parent.result, targets, worst, movable)
        if not parent.oracle.get('enclosure_passed', False):
            objective = lambda q: m.base_oracle.query(q)['gap_m']
            for kind, layout, detail in installation_proposals(m, parent.layout, objective=objective):
                rows.append((kind.replace('fixture-', 'oracle-fixture-'), layout, detail))
        if jumps:
            guests, _, _ = s.jump_guests(parent.result)
            rows.extend(s.juxtapose_proposals(parent.result, guests, targets))
            rows.extend(getattr(s, 'pending_pair_rows', [])); s.pending_pair_rows = []
        if (not parent.oracle.get('work_only_enclosure_passed', True)
                or (jumps and not parent.oracle.get('enclosure_passed', False))):
            rows.extend(s.oracle_layout_proposals(parent.result, int(iteration >= s.iterations)))
        return rows

    def finish(self, samples, selected):
        s, m = self.search, self.model
        s.frontier_states = [item.result for item in samples]
        s.frontier_statistics = dict(self.stats)
        s.frontier_witnesses = self.witnesses
        if selected.layout.key() in self.witnesses:
            m.base_oracle.witness = selected.layout.key(), self.witnesses[selected.layout.key()]
        m.commit(selected.result)
        return selected.result

    def solve(self, current, anchor=None):
        s, m = self.search, self.model
        frontier = [self.sample(current)]; stalled = False
        passed = self.connected(frontier)
        if passed is not None:
            return self.finish(frontier, passed)
        # Preserve the existing two bounded Oracle-rescue rounds after local
        # search; their alternatives now compete in the persistent frontier.
        for iteration in range(s.iterations+2):
            reference = min(frontier, key=lambda item: item.metrics).result
            with m.hold_measure(reference):
                frontier = [self.sample(item.result, item.operation, item.detail, item.parent)
                            for item in frontier]
                before_force = min(item.loss for item in frontier)
                before_base = min((item.metrics[1], item.metrics[2]) for item in frontier)
                descriptors = []; errors = []
                budget = getattr(s, 'screen_budget', 96)
                quota, extra = divmod(budget, len(frontier))
                for order, parent in enumerate(frontier):
                    jumps = stalled or iteration % 2 == 1 or iteration >= s.iterations
                    rows = self.proposals(parent, iteration, jumps)
                    bounded = bounded_proposals(rows, quota+int(order < extra), s.rng)
                    for kind, layout, detail in bounded:
                        try:
                            proxy = m.proxy(layout)
                            loss = float(proxy['loss'])
                            if not np.isfinite(loss):
                                raise ValueError('Nonfinite whole-demand projection loss')
                            oracle = m.base_oracle.query(layout)
                            descriptors.append(Sample(layout, loss, oracle, kind, detail,
                                                      parent=identity(parent.layout)))
                        except (RuntimeError, ValueError, AssertionError) as error:
                            errors.append(dict(operation=kind, error=str(error)))
                self.stats['screened_candidates'] += len(descriptors)+len(errors)
                screened_count = len(descriptors)+len(errors)
                m.commit(reference)
                shortlist = select_samples(frontier+descriptors, 2*self.width, m.extent)
                pool = list(frontier)
                for item in shortlist:
                    if item.result is not None:
                        continue
                    try:
                        self.stats['full_demand_candidates'] += 1
                        result = m.evaluate(item.layout)
                        if lost_protected(anchor, result):
                            continue
                        item.result = result; item.oracle = result['base_oracle']; pool.append(item)
                        if 'seat_signature' in item.detail:
                            if not hasattr(s, 'seat_hashes'):
                                s.seat_hashes = set()
                            s.seat_hashes.add(item.detail['seat_signature'])
                    except (RuntimeError, ValueError, AssertionError) as error:
                        errors.append(dict(operation=item.operation, error=str(error)))
                # Full force feasibility must get the connection query before
                # geometric diversity can remove it from the retained pool.
                passed = self.connected(pool)
                frontier = select_samples(pool, self.width, m.extent)
                self.stats['rounds'] += 1
                self.stats['maximum_retained'] = max(self.stats['maximum_retained'], len(frontier))
                stalled = (min(item.loss for item in frontier) >= before_force
                           and min((item.metrics[1], item.metrics[2]) for item in frontier) >= before_base)
                s.record(dict(phase='persistent_layout_frontier', iteration=iteration+1,
                              quadrature_epoch=getattr(m, 'quadrature_epoch', 0),
                              screen_budget=budget, screened=screened_count,
                              retained=[dict(id=identity(item.layout), parent=item.parent,
                                             operation=item.operation, force_loss=item.loss,
                                             failed_demands=failed_count(item.result),
                                             base_status=item.oracle.get('status'),
                                             base_gap_m=item.metrics[2], feasible=item.feasible)
                                        for item in frontier], errors=errors))
                selected = passed or min(frontier, key=lambda item:
                                        (item.metrics[1], item.metrics[2], item.loss))
                m.commit(selected.result)
                s.process_snapshot(selected.result, 'persistent_layout_frontier', dict(iteration=iteration+1))
                print('LAYOUT FRONTIER', iteration+1, 'retained', len(frontier),
                      'screened', len(descriptors), 'force loss', selected.loss,
                      'oracle', selected.oracle.get('status'), flush=True)
                if passed is not None:
                    return self.finish(frontier, passed)
        return self.finish(frontier, min(frontier, key=lambda item:
                           (item.metrics[1], item.metrics[2], item.loss, failed_count(item.result))))
