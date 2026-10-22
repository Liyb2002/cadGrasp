"""Incremental juxtaposition with continuous roll and compact seat placement."""
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize_scalar

from .angular_model import roll_to_match
from .family_guidance import FamilyGuidance
from .family_gate import FamilyEvaluator
from .station_seed import planar_offset
from .search import context
from whole_task_2.step1.common import write_json
from fixture_state import installed_state
from population32 import centered_seat
from translation.layout_update import shift


def centered_material_origin(model, layout):
    """Use the initial COM as material origin, preserving every world task."""
    q = layout.copy(); c = model.mesh.center_mass
    for k in range(len(model.poses)):
        q.state_offsets[k] += model.native[k, :3, :3] @ c
        q.placements[k, :3, 3] -= c
    return q


def insertion_offsets(extent, maximum_rings=8):
    yield np.zeros(2)
    for ring in range(1, maximum_rings+1):
        radius = extent*ring/3
        count = max(8, 4*ring)
        for j in range(count):
            theta = 2*math.pi*j/count
            yield radius*np.array([math.cos(theta), math.sin(theta)])


class FamilySearch:
    def __init__(self, model, folder):
        self.model = model; self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.guide = FamilyGuidance(model)
        self.events = []
        self.context = context(model, self.folder/'search', iterations=1, population=3)
        self.evaluator = FamilyEvaluator(self.context)
        self.accepted = None
        self.rejected_proposals = 0
        self.lift_attempts = set()

    def try_score(self, layout):
        """An unresolved proposal is unusable, not a reason to stop search."""
        try:
            return self.guide.score(layout)
        except (ValueError, RuntimeError, AssertionError):
            self.rejected_proposals += 1
            return None

    def save(self, q, phase, **detail):
        self.model.activate(q)
        np.savez_compressed(self.folder/'latest_layout.npz', **q.arrays(self.model.native))
        row = dict(phase=phase, active_count=len(q.active), angles_deg=q.angles_deg.tolist(),
                   rejected_proposals=self.rejected_proposals, **detail)
        self.events.append(row)
        write_json(self.folder/'search_trace.json', self.events)
        print('FAMILY', phase, len(q.active), detail, flush=True)

    def insert(self, current, owner, height, maximum_rings=8):
        m = self.model; m.activate(current)
        q = current.copy(); q.active = tuple([*current.active, owner])
        q.state_offsets[owner, 2] += height
        anchor = current.active[0]
        match = roll_to_match(m.native[owner, :3, :3], m.native[anchor, :3, :3], m.rotation_axis)
        angles = list(dict.fromkeys([round(float(match % 360.), 8), float(q.angles_deg[owner]),
                                    float((match+90) % 360.), float((match+180) % 360.)]))
        bases = []
        for angle in angles:
            trial = m.spin_guest(q, owner, angle)
            trial = centered_seat(m, trial, owner, anchor)
            # Favor straightforward upward removal as a proposal; the original
            # direction remains another candidate when that cuts useful seats.
            for vertical in [False, True]:
                candidate = trial.copy()
                if vertical:
                    T = installed_state(m.native, candidate, int(candidate.hosts[owner]))
                    candidate.directions[owner] = T[:3, :3].T @ [0., 0., 1.]
                bases.append(candidate)
                if height > 0:
                    installation = installed_state(m.native, candidate, int(candidate.hosts[owner]))
                    grounded = shift(m, candidate, owner, installation[:3, :3].T @ [0., 0., -height])
                    bases.append(grounded)
        best, best_score = None, None
        successful_radius = None
        for offset in insertion_offsets(m.extent, maximum_rings):
            radius = float(np.linalg.norm(offset))
            if successful_radius is not None and radius > successful_radius+1e-8:
                break
            for base in bases:
                m.activate(base)
                trial = planar_offset(m, base, owner, offset)
                try:
                    score = self.guide.score(trial)
                except (ValueError, RuntimeError, AssertionError):
                    self.rejected_proposals += 1
                    continue
                if best_score is None or score['rank'] < best_score['rank']:
                    best, best_score = trial, score
                if score['rank'][0] == 0:
                    successful_radius = radius
            if successful_radius is not None:
                # At this radius sample a bounded number of alternatives;
                # continuous refinement below supplies non-grid angles.
                break
        if best is None:
            raise RuntimeError('No geometrically valid insertion candidate')
        m.activate(best)
        base_angle = float(best.angles_deg[owner])

        def objective(angle):
            nonlocal best, best_score
            try:
                trial = m.spin_guest(best, owner, angle)
            except (ValueError, RuntimeError, AssertionError):
                self.rejected_proposals += 1
                return 1e12
            value = self.try_score(trial)
            if value is None:
                return 1e12
            if value['rank'] < best_score['rank']:
                best, best_score = trial, value
            rank = value['rank']
            return 1000*rank[0]+10*rank[1]+min(1., rank[2]/(10*m.extent**3))

        minimize_scalar(objective, bounds=(base_angle-30., base_angle+30.), method='bounded',
                        options=dict(maxiter=9, xatol=.05))
        self.guide.commit(best)
        self.save(best, 'inserted', owner=m.poses[owner], guide_rank=list(best_score['rank']),
                  guidance_failures={m.poses[k]: v for k, v in best_score['failures'].items()})
        return best

    def certify(self, q, label):
        m = self.model; m.activate(q)
        try:
            item = self.evaluator.preview(q, operation=label)
            item = self.evaluator.classify(item)
            counts = {m.poses[k]: int(v.sum()) for k, v in item.result['masks'].items()}
            self.save(item.layout, 'full-load-check', counts=counts,
                      oracle=item.oracle.get('status'))
            if self.evaluator.space_candidate(item):
                item = self.evaluator.connect(item)
                item = self.evaluator.construct(item)
            if item.completed and item.construction.get('family_finite_access_gate_passed'):
                self.accepted = item
                self.save(item.layout, 'complete-fixture-passed', counts=counts,
                          material_cm3=item.evaluation['material_full_cm3'])
                return item
            if not all(mask.all() for mask in item.result['masks'].values()):
                self.guide.add_failures(item.result)
            failed_endpoints = m.family_failed_endpoints.get(item.layout.key(), {})
            if failed_endpoints:
                self.guide.add_endpoint_failures(failed_endpoints)
                self.save(item.layout, 'continuous-boundary-counterexamples',
                    failed_counts={m.poses[k]: len(ids) for k, ids in failed_endpoints.items()})
            self.save(item.layout, 'not-yet-complete', construction=item.construction,
                      oracle=item.oracle.get('status'), component_trials=item.oracle.get('component_trials', []))
            return item
        except (ValueError, RuntimeError, AssertionError) as error:
            self.save(q, 'certificate-error', error=f'{type(error).__name__}: {error}')
            return None

    def try_boundary_lifts(self, q):
        """React to boundary counterexamples whenever they become available."""
        m = self.model
        focused = [k for k in q.active if self.guide.extra_endpoint_indices[k]]
        for owner in focused:
            candidates = []
            for lift in [.005, .015, .03]:
                m.activate(q)
                t = installed_state(m.native, q, int(q.hosts[owner]))
                trial = shift(m, q, owner, t[:3, :3].T @ [0., 0., lift])
                vertical = trial.copy()
                vertical.directions[owner] = t[:3, :3].T @ [0., 0., 1.]
                for candidate in [trial, vertical]:
                    score = self.try_score(candidate)
                    if score is not None and candidate.key() not in self.lift_attempts:
                        candidates.append((score['rank'], candidate))
            for _, trial in sorted(candidates, key=lambda row: row[0])[:3]:
                self.lift_attempts.add(trial.key())
                item = self.certify(trial, 'continuous-boundary-lift')
                if item is not None and item.completed and self.accepted is not None:
                    return item
        return None

    def solve(self, height=.015, maximum_rings=8, resume=None):
        began = time.monotonic(); m = self.model
        # Individual contact pieces may later join through the complete base.
        # The full fixture still has the original one-component acceptance.
        m.require_connected_core = False
        if resume is not None:
            from .angular_model import AngularLayout
            with np.load(resume) as archive:
                q = AngularLayout.from_arrays(archive)
            m.activate(q); self.guide.commit(q)
            self.save(q, 'resumed_generated_layout', source=str(Path(resume).resolve()))
        else:
            q, _ = m.initial(); q = centered_material_origin(m, q)
            q.active = (0,)
            anchor_trials = []
            for gap in [height, 0.]:
                for direction in [[0., 0., 1.], -m.rotation_axis]:
                    trial = q.copy(); trial.state_offsets[0, 2] += gap
                    T = installed_state(m.native, trial, 0)
                    trial.directions[0] = T[:3, :3].T @ direction
                    score = self.try_score(trial)
                    if score is not None:
                        anchor_trials.append((score['rank'], trial))
            if not anchor_trials:
                raise RuntimeError('No numerically resolved anchor proposal')
            _, q = min(anchor_trials, key=lambda pair: pair[0])
            self.guide.commit(q)
            self.save(q, 'single-registered-task', guide_rank=list(self.guide.score(q)['rank']))
        for owner in range(len(q.active), len(m.poses)):
            q = self.insert(q, owner, height, maximum_rings)
        item = self.certify(q, 'incremental-continuous-juxtaposition')
        if item is not None and item.completed and self.accepted is not None:
            return item
        # A few extreme continuous loads can need slightly more bearing or a
        # different object-floor state than the random sample family. Try a
        # bounded physical lift, with floor reactions recomputed, before broad
        # coordinate search. Every proposal uses the same final certificate.
        lifted = self.try_boundary_lifts(q)
        if lifted is not None:
            return lifted
        # Load counterexamples refine the same guidance. Relocate the worst
        # seats while preserving all regions and all original demand samples.
        for sweep in range(5):
            unresolved = dict(rank=(len(q.active), 10**9, float('inf'), 0.),
                              failures={k: 1 for k in q.active})
            current_score = self.try_score(q) or unresolved
            owners = sorted(q.active, key=lambda k: -current_score['failures'][k])
            for owner in owners:
                best, score = q, self.try_score(q) or unresolved
                for delta_angle in [0., -45., 45., -90., 90., 180.]:
                    try:
                        base = m.spin_guest(q, owner, q.angles_deg[owner]+delta_angle)
                    except (ValueError, RuntimeError, AssertionError):
                        self.rejected_proposals += 1
                        continue
                    for direction in [(0., 0.), (1., 0.), (-1., 0.), (0., 1.), (0., -1.)]:
                        m.activate(base)
                        trial = planar_offset(m, base, owner, m.extent*(.125 if sweep < 2 else .25)*np.asarray(direction))
                        value = self.try_score(trial)
                        if value is not None and value['rank'] < score['rank']:
                            best, score = trial, value
                q = best; self.guide.commit(q)
                self.save(q, 'repair', owner=m.poses[owner], sweep=sweep, guide_rank=list(score['rank']))
                if self.guide.extra_endpoint_indices[owner] and score['rank'][0] == 0:
                    item = self.certify(q, f'continuous-boundary-seat-repair-{sweep}')
                    if item is not None and item.completed and self.accepted is not None:
                        return item
                    lifted = self.try_boundary_lifts(q)
                    if lifted is not None:
                        return lifted
            item = self.certify(q, f'counterexample-repair-{sweep}')
            if item is not None and item.completed and self.accepted is not None:
                return item
            lifted = self.try_boundary_lifts(q)
            if lifted is not None:
                return lifted
        raise RuntimeError(f'Continuous juxtaposition not yet certified after {time.monotonic()-began:.1f}s')
