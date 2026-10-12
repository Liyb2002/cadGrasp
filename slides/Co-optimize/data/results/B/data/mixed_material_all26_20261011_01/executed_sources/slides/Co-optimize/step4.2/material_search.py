"""Bounded mixed-layout search, preserving complete solids and their true cost."""
from dataclasses import replace
from collections import Counter
import hashlib
import numpy as np

from material_evaluator import (MaterialEvaluator, membership, target_gap,
                               align_world_xy, floor_height_proposal)
from reuse_proposals import (initial_topologies, bounded_seat_proposals, structure)
from whole_search.search import failed_count
from whole_search.reuse_first import registered
from whole_search.common import tangent_frame
from continuous_support.fast_gradient import turn
from translation.layout_update import shift, translation_frame, translation_gradient
from translation.installation import shift_installation, installation_gradient


def sample_id(layout):
    return hashlib.sha256(repr(layout.key()).encode()).hexdigest()[:16]


def select_pool(samples, width):
    """Force/base/cost alternatives with at most two of one sharing structure."""
    unique = {}
    for item in samples:
        key = item.layout.key(); old = unique.get(key)
        priority = (item.completed, item.witness is not None, item.result is not None)
        if old is None or priority > (old.completed, old.witness is not None, old.result is not None):
            unique[key] = item
    items = list(unique.values()); selected = []; counts = Counter()
    def keep(item, compulsory=False):
        if item in selected or len(selected) >= width:
            return
        group = structure(item.layout)
        if compulsory or counts[group] < 2:
            selected.append(item); counts[group] += 1
    completed = [x for x in items if x.completed]
    if completed:
        keep(min(completed, key=lambda x: (target_gap(x.layout), *x.cost)), True)
    mixed = [x for x in items if not target_gap(x.layout)] or items
    if mixed:
        keep(min(mixed, key=lambda x: (x.loss, x.metrics[2], x.metrics[3], x.lower_litres)), True)
        keep(min(mixed, key=lambda x: (x.metrics[2], x.metrics[3], x.loss, x.lower_litres)), True)
        for count in sorted(set(membership(x.layout)[0] for x in mixed)):
            group=[x for x in mixed if membership(x.layout)[0]==count]
            keep(min(group,key=lambda x:(x.metrics[2],
                failed_count(x.result) if x.result is not None else 0,
                x.loss,x.metrics[3],x.lower_litres)),True)
    remaining = sorted(items, key=lambda x: (target_gap(x.layout),
        not x.oracle.get('enclosure_passed'), x.loss, x.metrics[3], x.lower_litres))
    # Keep distinct registered subsets before another height/direction variant.
    for item in remaining:
        if counts[structure(item.layout)] == 0:
            keep(item)
    for item in remaining:
        keep(item)
    return selected


class MaterialSearch:
    def __init__(self, search):
        self.search = search; self.model = search.model
        self.evaluator = MaterialEvaluator(search)
        self.width = search.frontier_width
        self.witnesses = {}; self.incumbent = None; self.errors = []
        self.evaluated = {}; self.seed_cursor = 0

    def preview(self, layout, operation='parent', parent='', detail=None, result=None):
        return self.evaluator.preview(layout, operation, parent, detail, result)

    def repair_height(self, layout):
        answer = self.model.base_oracle.query(layout)
        if answer.get('status') != 'certified_floor_incompatible':
            return layout
        trial = floor_height_proposal(self.model, layout)
        return trial if trial is not None else layout

    def classify(self, items, budget=8):
        selected = select_pool(items, budget) if len(items)>budget else items
        checked = []
        for item in selected:
            try:
                key = item.layout.key()
                if key in self.evaluated:
                    saved = self.evaluated[key]
                    item.result = saved.result
                    if saved.witness is not None:
                        item.witness = saved.witness; item.oracle = saved.oracle
                self.evaluator.classify(item)
                self.evaluated[key] = item; checked.append(item)
            except (ValueError, RuntimeError, AssertionError) as error:
                self.errors.append(dict(stage='force', operation=item.operation, error=str(error)))
        return checked

    def certify(self, items, connections=3, constructions=2):
        pending = [x for x in items if x.result is not None and not failed_count(x.result)
                   and x.oracle.get('enclosure_passed')]
        pending.sort(key=lambda x: (target_gap(x.layout), x.lower_litres,
                                   x.result['volume_cm3']))
        fresh = [x for x in pending if x.layout.key() not in self.evaluator.connection_attempts]
        chosen=fresh[:connections]
        opening=[x for x in fresh if x.operation.startswith('base-opening')]
        if opening and chosen and opening[0] not in chosen:
            chosen=chosen[:connections-1]+opening[:1]
        for item in chosen:
            try:
                self.evaluator.connect(item)
                if item.witness is not None:
                    self.witnesses[item.layout.key()] = item.witness
            except (ValueError, RuntimeError, AssertionError) as error:
                self.errors.append(dict(stage='connection', error=str(error)))
        buildable = [x for x in pending if x.witness is not None and
                     x.layout.key() not in self.evaluator.build_attempts]
        for item in buildable[:constructions]:
            self.evaluator.construct(item)
            if item.completed and (self.incumbent is None or
                    (target_gap(item.layout), *item.cost) <
                    (target_gap(self.incumbent.layout), *self.incumbent.cost)):
                self.incumbent = item

    def continuation_rows(self, parent, round_index):
        """Original sphere/XYZ secants, with ALL owners in every objective."""
        m = self.model; q = parent.layout
        # A preview has no certified masks. Commit geometry caches directly;
        # Model.commit would otherwise remember borrowed masks as a real pass.
        if parent.result is not None:
            m.commit(parent.result)
        else:
            m.contact_delta.commit(q); m.volume_guidance([q]).commit(q)
        base = m.proxy(q); active = list(q.active)
        order = sorted(active, key=lambda k: -base['residual_loss'][active.index(k)])
        owners = list(dict.fromkeys([order[0], active[round_index%len(active)]]))[:2]
        if not parent.oracle.get('enclosure_passed'):
            objective = lambda x: float(m.base_oracle.query(x)['gap_m'])
        elif parent.result is not None and failed_count(parent.result):
            objective = lambda x: float(m.proxy(x)['loss'])
        else:
            objective = lambda x: float(m.proxy(x)['loss'])
        angular = []; h = np.radians(.75 if round_index < 4 else .125)
        for k in owners:
            frame = tangent_frame(q.directions[k])
            g = np.array([(objective(turn(m, q, k, h*frame[:, a]))-
                           objective(turn(m, q, k, -h*frame[:, a])))/(2*h)
                          for a in range(2)])
            angular.append((k, frame, g))
        norm = np.sqrt(sum(float(g@g) for _, _, g in angular))
        rows = []
        if norm > 1e-28:
            for degrees in [.25, 1., 4., 8.]:
                trial = q.copy()
                for k, frame, g in angular:
                    trial.directions[k] = turn(m, q, k, -np.radians(degrees)*frame@g/norm).directions[k]
                rows.append(('direction-gradient', trial, dict(owners=owners, degrees=degrees)))
        # Explicit event probes remain available when a contact-cell derivative is flat.
        k = owners[-1]; frame = tangent_frame(q.directions[k])
        for a in range(2):
            for sign in [-1., 1.]:
                rows.append(('direction-event', turn(m, q, k, sign*np.radians(3.)*frame[:, a]),
                             dict(pose=k, event_probe=True)))
        movable = [k for k in order if not registered(q, k)]
        delta = m.extent*(1/100 if round_index < 4 else 1/512)
        if movable:
            k = movable[round_index%len(movable)]
            gradient = translation_gradient(m, q, [k], delta, objective)
            length = np.linalg.norm(gradient)
            if length > 1e-28:
                for fraction in [1/64, 1/16, 1/4]:
                    trial = shift(m, q, k, -m.extent*fraction*translation_frame(m, q, k)@gradient/length)
                    if not target_gap(trial):
                        rows.append(('seat-gradient', trial, dict(pose=k, fraction=fraction)))
            else:
                axis = round_index%3
                for sign in [-1., 1.]:
                    trial = shift(m, q, k, sign*m.extent/32*translation_frame(m, q, k)[:, axis])
                    if not target_gap(trial):
                        rows.append(('seat-event', trial, dict(pose=k, axis=axis)))
        hosts = sorted(set(int(q.hosts[k]) for k in q.active))
        host = hosts[round_index%len(hosts)]
        gradient = installation_gradient(m, q, (host,), delta, objective)
        length = np.linalg.norm(gradient)
        if length > 1e-28:
            for fraction in [1/64, 1/8]:
                trial = shift_installation(m, q, (host,), -m.extent*fraction*gradient/length)
                rows.append(('installation-gradient', trial, dict(host=host, fraction=fraction)))
        trial = floor_height_proposal(m, q)
        if trial is not None and trial.key() != q.key():
            rows.append(('installation-floor-program', trial, dict(original_demands=True)))
        if parent.result is not None and not failed_count(parent.result) and not parent.oracle.get('connected_passed'):
            opening=[]
            for fraction in [1/64,1/16,1/4]:
                trial=shift_installation(m,q,tuple(hosts),np.array([0.,0.,m.extent*fraction]))
                opening.append(('base-opening-installation',trial,dict(fraction=fraction)))
            for axis in np.eye(3):
                for sign in [-1.,1.]:
                    trial=q.copy()
                    for k in q.active:
                        trial.directions[k]=turn(m,q,k,sign*np.radians(12.)*np.cross(axis,q.directions[k])).directions[k]
                    opening.append(('base-opening-direction',trial,dict(axis=axis.tolist(),sign=sign)))
            rows=opening+rows[:7]
        return rows[:16]

    def refine_preview(self, parent, round_index, maximum=2):
        current = parent
        for step in range(maximum):
            options = [current]
            try:
                for kind, q, detail in self.continuation_rows(current, round_index+step):
                    if target_gap(q) > target_gap(current.layout):
                        continue
                    options.append(self.preview(q, kind, sample_id(current.layout), detail))
            except (ValueError, RuntimeError, AssertionError) as error:
                self.errors.append(dict(stage='refine', error=str(error)))
            # Force and base tradeoffs remain in the outer pool; each local
            # continuation uses one coherent feasibility ordering.
            best = min(options, key=lambda x: (target_gap(x.layout),
                not x.oracle.get('enclosure_passed'), x.loss, x.metrics[3], x.lower_litres))
            opening=[x for x in options if x.operation.startswith('base-opening')
                     and x.loss<=1e-28 and x.oracle.get('enclosure_passed')]
            if (current.result is not None and not failed_count(current.result)
                    and not current.oracle.get('connected_passed') and opening):
                best=min(opening,key=lambda x:x.lower_litres)
            if best is current:
                break
            current = best
        return current

    def describe(self, item):
        reuse, juxtaposed = membership(item.layout)
        row = dict(id=sample_id(item.layout), operation=item.operation, loss=item.loss,
            reuse=reuse, juxtaposed=juxtaposed, target_satisfied=not target_gap(item.layout),
            oracle=item.oracle.get('status'), lower_bbx_litres=item.lower_litres,
            completed=item.completed, parent=item.parent)
        if item.result is not None:
            row['failed_demands'] = failed_count(item.result)
        if item.completed:
            row['full_bbx_litres'], row['material_cm3'] = item.cost
        if item.construction is not None and not item.construction.get('passed'):
            row['construction_error'] = item.construction.get('error', 'construction_checks_failed')
        return row

    def solve(self, current):
        s, m = self.search, self.model
        start = self.preview(current['layout'], result=current)
        pool = [start]
        self.certify(pool)
        warm = getattr(s, 'material_warm_layouts', [])
        seed_rows = initial_topologies(m, current['layout'], s.rng,
                                      getattr(s, 'initialization_budget', 96))
        # Existing layouts are explicitly recorded warm seeds, not cold results.
        seed_rows = [('saved-warm-layout', q, dict(source=source)) for source, q in warm]+seed_rows
        with m.hold_measure(current):
            previews = []
            for kind, q, detail in seed_rows:
                try:
                    q = self.repair_height(q)
                    q = align_world_xy(m, q)
                    previews.append(self.preview(q, kind, detail=detail))
                except (ValueError, RuntimeError, AssertionError) as error:
                    self.errors.append(dict(stage='initialize', operation=kind, error=str(error)))
            initial = self.classify(previews, 8)
            pool = select_pool(pool+initial, self.width)
            self.certify(pool)
        print('MATERIAL INITIAL', len(previews), 'pool', len(pool),
              [self.describe(x) for x in pool], flush=True)
        for iteration in range(s.iterations):
            reference = min(pool, key=lambda x: (failed_count(x.result), x.loss)).result
            reference = dict(reference, masks={k: np.logical_and.reduce(
                [x.result['masks'][k] for x in pool]) for k in reference['layout'].active})
            with m.hold_measure(reference):
                for item in pool:
                    item.loss = float(m.proxy(item.layout)['loss'])
                previews = []
                for parent in pool:
                    candidate = self.refine_preview(parent, iteration, maximum=1)
                    if candidate.layout.key() != parent.layout.key():
                        previews.append(candidate)
                    for kind, q, detail in bounded_seat_proposals(m, parent.layout, s.rng, iteration):
                        try:
                            q = self.repair_height(q)
                            previews.append(self.preview(q, kind, sample_id(parent.layout), detail))
                        except (ValueError, RuntimeError, AssertionError) as error:
                            self.errors.append(dict(stage='proposal', error=str(error)))
                # Fixed pool/proposal count; injected fresh structures replace
                # missing proposal slots, and never create a nested search tree.
                fresh = seed_rows[self.seed_cursor:self.seed_cursor+4]; self.seed_cursor += 4
                if len(previews) < 40:
                    for kind, q, detail in fresh[:40-len(previews)]:
                        if not target_gap(q):
                            previews.append(self.preview(self.repair_height(q), kind, detail=detail))
                shortlisted = select_pool(previews, 8)
                opening=[x for x in previews if x.operation.startswith('base-opening')
                         and x.loss<=1e-28 and x.oracle.get('enclosure_passed')]
                if opening and not any(x.operation.startswith('base-opening') for x in shortlisted):
                    shortlisted=shortlisted[:7]+[min(opening,key=lambda x:x.lower_litres)]
                refined = []
                for item in shortlisted:
                    child = item if item.operation.startswith('base-opening') else self.refine_preview(item, iteration, maximum=1)
                    refined.append(child)
                children = self.classify(refined, 8)
                self.certify(pool+children)
                pool = select_pool(pool+children+([self.incumbent] if self.incumbent else []), self.width)
                selected = self.incumbent or min(pool, key=lambda x: (target_gap(x.layout),
                    not x.oracle.get('enclosure_passed'), failed_count(x.result), x.loss, x.lower_litres))
                s.record(dict(phase='shared_material_search', iteration=iteration+1,
                    statistics=dict(self.evaluator.stats), retained=[self.describe(x) for x in pool],
                    incumbent=self.describe(self.incumbent) if self.incumbent else None,
                    errors=self.errors[-12:]))
                m.commit(selected.result)
                s.process_snapshot(selected.result, 'shared_material_search', dict(iteration=iteration+1))
                print('MATERIAL ROUND', iteration+1, self.describe(selected), flush=True)
        selected = self.incumbent or min(pool, key=lambda x: (target_gap(x.layout),
            not x.oracle.get('enclosure_passed'), failed_count(x.result), x.loss, x.lower_litres))
        s.frontier_witnesses = self.witnesses
        s.frontier_states = [x.result for x in pool]
        s.frontier_statistics = dict(self.evaluator.stats, width=self.width, rounds=s.iterations,
            maximum_retained=len(pool), policy='shared_material_mixed_search',
            target_satisfied=not target_gap(selected.layout), errors=self.errors[-30:])
        s.material_selected = selected
        if selected.witness is not None:
            m.base_oracle.witness = selected.layout.key(), selected.witness
        selected.result['base_oracle'] = selected.oracle
        m.commit(selected.result)
        return selected.result
