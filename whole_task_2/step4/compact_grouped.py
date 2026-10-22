"""Rank real multi-installation occupancy and repair connected-material failures."""
from itertools import product
from pathlib import Path
import json
import numpy as np

from .grouped_search import GroupedSearch, memberships
from .family_guidance import FamilyGuidance
from .family_search import FamilySearch
from .grouped_placement import fit_groups
from .angular_model import roll_to_match
from fixture_state import installed_state
from translation.layout_update import shift
from whole_task_2.evaluation.metrics import static_bounds as original_bounds


def execution_box(model, layout):
    """Body envelope in every execution of the common prospective fixture.

    This remains guidance: the final support and feet get their own actual box.
    Unlike a material-frame box it cannot hide expansion by rotating the host.
    """
    model.activate(layout)
    clouds = np.vstack([model.mesh.vertices @ layout.placements[k, :3, :3].T+layout.placements[k, :3, 3]
                        for k in layout.active])
    points = []
    for h, ids in memberships(layout).items():
        t = np.asarray(model.execution(layout, ids[0])['world_from_support'])
        points.append(clouds @ t[:3, :3].T+t[:3, 3])
    low = np.min([p.min(0) for p in points], axis=0)
    high = np.max([p.max(0) for p in points], axis=0)
    return float(np.prod(high-low)), (high-low).tolist()


class OccupancyGuidance(FamilyGuidance):
    def score(self, layout):
        result = dict(super().score(layout))
        occupied, dimensions = execution_box(self.model, layout)
        result['rank'] = (*result['rank'][:2], occupied, result['rank'][3])
        result['execution_envelope_dimensions_m'] = dimensions
        return result


def shared_static_bounds(body, layout, step4_vertices=None, full_vertices=None, pose_names=None, execution=None):
    """Canonicalize only floating signed-zero/roundoff for membership counting.

    Geometric transforms and their input files remain unchanged. Matrices in
    one group differ by <1e-10 m; the representative avoids byte-key splits.
    """
    representatives = []
    canonical = []
    for row in execution:
        t = np.asarray(row['world_from_support'])
        for representative in representatives:
            if np.allclose(t, representative, atol=1e-10, rtol=0):
                break
        else:
            representative = t.copy(); representatives.append(representative)
        canonical.append(dict(row, world_from_support=representative.tolist()))
    result = original_bounds(body, layout, step4_vertices, full_vertices, pose_names, canonical)
    result['support_pose_metric_version'] = 'actual_shared_installations_tolerance_1e-10_v1'
    result['membership_roundoff_canonicalized_only'] = True
    return result


def actual_failure_rank(item):
    if item is None:
        return (10**9, 10**9)
    scores = []
    for row in item.oracle.get('component_trials', []):
        if not row.get('all_ground_enclosures_passed') or not row.get('sample_counts'):
            continue
        samples = sum(32768-count for count in row['sample_counts'].values())
        endpoints = row.get('continuous_endpoint_counts')
        missing = sum(r['total']-r['passed'] for r in endpoints.values()) if endpoints else 10**9
        scores.append((samples, missing))
    return min(scores) if scores else (10**9, 10**9)


class CompactGroupedSearch(GroupedSearch):
    def __init__(self, model, folder, group_count=5):
        super().__init__(model, folder, group_count)
        self.guide = OccupancyGuidance(model)
        self.actual_repairs = set()

    def solve(self, height=0., maximum_rings=8, resume=None):
        if resume is not None:
            trace = Path(resume).with_suffix('.trace.json')
            if trace.exists():
                self.events = json.loads(trace.read_text())
                self.model.inputs.append(trace)
                self.save_checkpoint_trace = str(trace.resolve())
        return super().solve(height, maximum_rings, resume)

    def certify(self, q, label, _existing=None):
        item = _existing if _existing is not None else FamilySearch.certify(self, q, label)
        if self.accepted is not None or item is None:
            return item
        usable = [r for r in item.oracle.get('component_trials', [])
                  if r.get('all_ground_enclosures_passed') and r.get('sample_counts')]
        if not usable:
            return item
        m = self.model
        baseline_failure = actual_failure_rank(item)
        best = min(usable, key=lambda r: sum(len(m.tasks[k].targets)-r['sample_counts'][m.poses[k]] for k in q.active))
        owners = [k for k in q.active if best['sample_counts'][m.poses[k]] < len(m.tasks[k].targets)]
        if not owners:
            owners = [k for k in q.active if k in m.family_failed_endpoints.get(q.key(), {})]
        # A failed task can be blocked by the OTHER object poses sharing its
        # installation. Release those contacts without changing the failed
        # task's world pose, gravity or working region.
        for owner in owners:
            peers = [k for k in memberships(q)[int(q.hosts[owner])] if k != owner]
            if not peers:
                continue
            for affected in [peers] + [[k] for k in reversed(peers)]:
                for degrees in [2., 1., .5, .25, -.25, -.5, -1., -2., 5., -5.]:
                    trial = q.copy()
                    for member in affected:
                        trial = m.spin_guest(trial, member, q.angles_deg[member]+degrees)
                    try:
                        trial, _ = fit_groups(m, trial, preserve=True)
                        occupied, _ = execution_box(m, trial)
                    except (ValueError, RuntimeError, AssertionError):
                        continue
                    if trial.key() in self.actual_repairs:
                        continue
                    self.actual_repairs.add(trial.key()); self.group_invariant(trial)
                    self.save(trial, 'co-served-object-pose-repair', failed_region=m.poses[owner],
                        rotated_regions=[m.poses[k] for k in affected], degrees=degrees,
                        failed_task_world_pose_preserved=True, surrogate_execution_litres=1000*occupied)
                    tested = FamilySearch.certify(self, trial, 'co-served-object-pose-repair')
                    if self.accepted is not None:
                        return tested
                    if actual_failure_rank(tested) < baseline_failure:
                        self.save(trial, 'accepted-partial-actual-force-improvement',
                            before=list(baseline_failure), after=list(actual_failure_rank(tested)))
                        return self.certify(trial, 'actual-force-coordinate-progress', _existing=tested)
        # Withdrawal is an original Step4 variable, independent of the fixed
        # painting axis. Do not assume world-Z removal is always the best one.
        # These changes preserve every actual object and support installation.
        tool_side = -m.rotation_axis
        vertical = np.array([0., 0., 1.])
        for scope in ['failed', 'all']:
            for blend in [0., 1., .5]:
                world_direction = (1.-blend)*vertical+blend*tool_side
                world_direction /= np.linalg.norm(world_direction)
                trial = q.copy(); m.activate(trial)
                affected = owners if scope == 'failed' else trial.active
                for owner in affected:
                    t = installed_state(m.native, trial, int(trial.hosts[owner]))
                    trial.directions[owner] = t[:3, :3].T @ world_direction
                if trial.key() == q.key() or trial.key() in self.actual_repairs:
                    continue
                self.actual_repairs.add(trial.key()); self.group_invariant(trial)
                self.save(trial, 'actual-component-withdrawal-repair', affected_regions=[m.poses[k] for k in affected],
                          tool_side_blend=blend, object_poses_changed=False, scope=scope)
                proposal = FamilySearch.certify(self, trial, 'actual-component-withdrawal-repair')
                if self.accepted is not None:
                    return proposal
                if actual_failure_rank(proposal) < baseline_failure:
                    self.save(trial, 'accepted-partial-actual-force-improvement',
                        before=list(baseline_failure), after=list(actual_failure_rank(proposal)))
                    return self.certify(trial, 'actual-force-coordinate-progress', _existing=proposal)
                residual = m.family_failed_endpoints.get(trial.key(), {})
                if scope == 'failed' and blend == 0. and residual:
                    # Keep a partial physical improvement (e.g. one task now
                    # passed) and refine only the remaining failures.
                    horizontal = tool_side-tool_side.dot(vertical)*vertical
                    horizontal /= np.linalg.norm(horizontal)
                    side = np.cross(vertical, horizontal)
                    for axis in [horizontal, side]:
                        for degrees in [.25, -.25, .5, -.5, 1., -1., 2., -2.]:
                            fine = trial.copy(); m.activate(fine)
                            angle = np.radians(degrees)
                            direction = np.cos(angle)*vertical+np.sin(angle)*axis
                            for owner in residual:
                                t = installed_state(m.native, fine, int(fine.hosts[owner]))
                                fine.directions[owner] = t[:3, :3].T @ direction
                            if fine.key() in self.actual_repairs:
                                continue
                            self.actual_repairs.add(fine.key()); self.group_invariant(fine)
                            self.save(fine, 'small-withdrawal-angle-repair',
                                remaining_regions=[m.poses[k] for k in residual], degrees=degrees,
                                direction_world=direction.tolist(), partial_improvement_preserved=True)
                            tested = FamilySearch.certify(self, fine, 'small-withdrawal-angle-repair')
                            if self.accepted is not None:
                                return tested
        # Keep useful ground reactions. Mixed corrections let a grounded
        # task roll while an already suspended task changes height; lifting
        # every failed task together can destroy an otherwise useful support.
        grounded = []
        suspended = []
        m.activate(q)
        for owner in owners:
            (grounded if m.floor_signature(q, owner) else suspended).append(owner)
        if grounded and suspended:
            mixed = []
            for degrees in [-2., 2., -5., 5., -10., 10.]:
                for lift in [.005, -.005]:
                    trial = q.copy()
                    for owner in grounded:
                        trial = m.spin_guest(trial, owner, q.angles_deg[owner]+degrees)
                    for owner in suspended:
                        m.activate(trial); t = installed_state(m.native, trial, int(trial.hosts[owner]))
                        trial = shift(m, trial, owner, t[:3, :3].T @ [0., 0., lift])
                    try:
                        trial, _ = fit_groups(m, trial, preserve=True)
                        occupied, _ = execution_box(m, trial)
                    except (ValueError, RuntimeError, AssertionError):
                        continue
                    mixed.append((occupied, trial, degrees, lift))
            for occupied, trial, degrees, lift in sorted(mixed, key=lambda row: row[0]):
                if trial.key() in self.actual_repairs:
                    continue
                self.actual_repairs.add(trial.key()); self.group_invariant(trial)
                self.save(trial, 'ground-aware-joint-pose-repair',
                    grounded_rotating_regions=[m.poses[k] for k in grounded],
                    suspended_translating_regions=[m.poses[k] for k in suspended],
                    delta_angle_deg=degrees, lift_m=lift, surrogate_execution_litres=1000*occupied)
                proposal = FamilySearch.certify(self, trial, 'ground-aware-joint-pose-repair')
                if self.accepted is not None:
                    return proposal
        # Continuous target failures need actual-material probes even when
        # the centroid guide dislikes them. Try small simultaneous physical
        # lifts first; no membership is changed and no force test is skipped.
        if owners:
            for amount in [.005, .001, .003, .01]:
                trial = q.copy()
                for owner in owners:
                    m.activate(trial); t = installed_state(m.native, trial, int(trial.hosts[owner]))
                    trial = shift(m, trial, owner, t[:3, :3].T @ [0., 0., amount])
                try:
                    trial, _ = fit_groups(m, trial, preserve=True)
                except (ValueError, RuntimeError, AssertionError):
                    continue
                if trial.key() in self.actual_repairs:
                    continue
                self.actual_repairs.add(trial.key()); self.group_invariant(trial)
                occupied, _ = execution_box(m, trial)
                self.save(trial, 'actual-component-joint-lift', owners=[m.poses[k] for k in owners],
                          lift_m=amount, surrogate_execution_litres=1000*occupied)
                proposal = FamilySearch.certify(self, trial, 'actual-component-joint-lift')
                if self.accepted is not None:
                    return proposal
        proposals = []
        for owner in owners:
            for lift in [.001, .002, .003, .004, .005, .0075, .01, .015]:
                m.activate(q); t = installed_state(m.native, q, int(q.hosts[owner]))
                trial = shift(m, q, owner, t[:3, :3].T @ [0., 0., lift])
                proposals.append((trial, dict(owner=m.poses[owner], lift_m=lift)))
            # Search the failed task and its co-served objects; either one's
            # rotation can release material that was trapped in an island.
            for member in memberships(q)[int(q.hosts[owner])]:
                for delta in [-2., 2., -5., 5., -10., 10., -15., 15., -30., 30.]:
                    try:
                        trial = m.spin_guest(q, member, q.angles_deg[member]+delta)
                    except (ValueError, RuntimeError, AssertionError):
                        continue
                    proposals.append((trial, dict(owner=m.poses[member], failed_region=m.poses[owner], delta_angle_deg=delta)))
        # Several continuous boundary failures may need a JOINT correction.
        # Trying one owner at a time cannot pass while the other is unchanged.
        if 1 < len(owners) <= 3:
            for lifts in product([.001, .003, .005, .01], repeat=len(owners)):
                trial = q.copy()
                for owner, lift in zip(owners, lifts):
                    m.activate(trial); t = installed_state(m.native, trial, int(trial.hosts[owner]))
                    trial = shift(m, trial, owner, t[:3, :3].T @ [0., 0., lift])
                proposals.append((trial, dict(owners=[m.poses[k] for k in owners], joint_lifts_m=list(lifts))))
            if len(owners) == 2:
                for deltas in product([-5., -2., 2., 5.], repeat=2):
                    trial = q.copy()
                    for owner, delta in zip(owners, deltas):
                        trial = m.spin_guest(trial, owner, q.angles_deg[owner]+delta)
                    proposals.append((trial, dict(owners=[m.poses[k] for k in owners], joint_delta_angles_deg=list(deltas))))
        ranked = []
        for trial, detail in proposals:
            try:
                trial, _ = fit_groups(m, trial, preserve=True)
                occupied, _ = execution_box(m, trial)
            except (ValueError, RuntimeError, AssertionError):
                continue
            if trial.key() in self.actual_repairs:
                continue
            ranked.append((0 if 'owners' in detail else 1, occupied, trial, detail))
        for _, occupied, trial, detail in sorted(ranked, key=lambda row: row[:2])[:12]:
            self.actual_repairs.add(trial.key())
            self.group_invariant(trial)
            self.save(trial, 'actual-component-targeted-pose-repair', surrogate_execution_litres=1000*occupied, **detail)
            proposal = FamilySearch.certify(self, trial, 'actual-component-targeted-pose-repair')
            if self.accepted is not None:
                return proposal
        return item
