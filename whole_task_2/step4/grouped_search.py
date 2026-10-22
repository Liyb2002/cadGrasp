"""Balanced shared installations: pose adjustment first, group merges second.

A group shares the full *centered execution* support transform, not just a
host label. Within a group only axial object roll, height and withdrawal are
changed. Independent in-plane guest translations are deliberately excluded.
"""
from itertools import combinations
import math
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp, minimize_scalar

from .angular_model import roll_to_match
from .family_search import FamilySearch, centered_material_origin
from fixture_state import installed_state
from population32 import centered_seat
from translation.layout_update import shift


def memberships(layout):
    return {int(h): [k for k in layout.active if layout.hosts[k] == h]
            for h in sorted(set(layout.hosts[list(layout.active)]))}


def balanced_partition(rotations, axis, count=5):
    """Compare every pair/triple; exact-cover groups without anatomical IDs."""
    directions = np.einsum('kji,j->ki', rotations, axis)
    n = len(rotations)
    minimum, maximum = n//count, math.ceil(n/count)
    candidates = [ids for size in range(minimum, maximum+1)
                  for ids in combinations(range(n), size)]
    costs = np.array([sum(1.-float(directions[a] @ directions[b])
                         for a, b in combinations(ids, 2))/len(ids) for ids in candidates])
    incidence = np.zeros((n+1, len(candidates)))
    for j, ids in enumerate(candidates):
        incidence[list(ids), j] = 1.; incidence[-1, j] = 1.
    target = np.r_[np.ones(n), count]
    result = milp(costs, integrality=np.ones(len(costs)), bounds=Bounds(0., 1.),
                  constraints=LinearConstraint(incidence, target, target),
                  options={'mip_rel_gap': 0.})
    if not result.success:
        raise RuntimeError('Balanced installation partition did not solve')
    selected = [list(candidates[j]) for j in np.flatnonzero(result.x > .5)]
    # The retained host can be ANY member. Use the orientation medoid.
    def medoid(ids):
        host = min(ids, key=lambda a: (sum(1.-directions[a] @ directions[b] for b in ids), a))
        return [host]+[k for k in ids if k != host]
    return sorted([medoid(ids) for ids in selected], key=lambda ids: min(ids))


def move_group_material(model, layout, host, delta):
    """Move an entire seat group in fixture coordinates, preserving world tasks."""
    model.activate(layout)
    q = layout.copy(); t = installed_state(model.native, q, host)
    q.state_offsets[host] -= t[:3, :3] @ delta
    for k in memberships(q)[host]:
        q.placements[k, :3, 3] += delta
    return q


def installation_summary(model, layout):
    model.activate(layout)
    rows = [model.execution(layout, k) for k in layout.active]
    groups = []
    for row in rows:
        matrix = np.asarray(row['world_from_support'])
        for group in groups:
            if np.allclose(matrix, group['world_from_support'], atol=1e-10, rtol=0):
                group['working_areas'].append(row['pose']); break
        else:
            groups.append(dict(working_areas=[row['pose']], world_from_support=matrix.tolist()))
    rotations = []
    for group in groups:
        r = np.asarray(group['world_from_support'])[:3, :3]
        if not any(np.allclose(r, old, atol=1e-10, rtol=0) for old in rotations):
            rotations.append(r)
    return dict(actual_support_pose_count=len(groups), support_orientation_count=len(rotations),
                group_sizes=sorted([len(g['working_areas']) for g in groups], reverse=True),
                groups=groups)


class GroupedSearch(FamilySearch):
    def __init__(self, model, folder, group_count=5):
        super().__init__(model, folder)
        self.group_count = group_count
        self.partition = None

    def save(self, q, phase, **detail):
        super().save(q, phase, support_group_count=len(memberships(q)), **detail)

    def group_invariant(self, q):
        summary = installation_summary(self.model, q)
        if summary['actual_support_pose_count'] != len(memberships(q)):
            raise ValueError('A nominal group split into several actual support installations')
        return summary

    def seed(self):
        m = self.model; q, _ = m.initial(); q = centered_material_origin(m, q)
        self.save(q, 'thirteen-independent-installations')
        self.partition = balanced_partition(m.reference_source_native[:, :3, :3], m.rotation_axis,
                                            self.group_count)
        self.save(q, 'balanced-partition-proposal', groups=[[m.poses[k] for k in ids] for ids in self.partition],
                  candidate_hosts='every member of every remaining installation')
        # Rotate the independent object BEFORE deciding its shared seat.
        for ids in self.partition:
            h = ids[0]
            for k in ids[1:]:
                m.activate(q)
                angle = roll_to_match(m.reference_source_native[k, :3, :3], m.native[h, :3, :3], m.rotation_axis)
                q = m.spin_guest(q, k, angle)
        self.save(q, 'independent-object-poses-aligned')
        for ids in self.partition:
            h = ids[0]
            for k in ids[1:]:
                q = centered_seat(m, q, k, h)
                self.save(q, 'merge-installations', owner=m.poses[k], retained_host=m.poses[h])
        for k in q.active:
            t = installed_state(m.native, q, int(q.hosts[k]))
            q.directions[k] = t[:3, :3].T @ np.array([0., 0., 1.])
        self.group_invariant(q)
        return q

    def spread(self, base, radius):
        m = self.model; q = base.copy(); m.activate(q)
        for h in memberships(q):
            t = installed_state(m.native, q, h)
            normal = t[:3, :3].T @ np.array([0., 0., 1.])
            q = move_group_material(m, q, h, -radius*normal)
        self.group_invariant(q)
        return q

    def pose_repair(self, q, sweeps=2):
        """Fixed groups: try continuous angle and height, never an XY escape."""
        m = self.model
        from .grouped_placement import fit_groups
        def normalise(trial):
            return fit_groups(m, trial, preserve=True)[0]
        unresolved = dict(rank=(13, 10**9, 1., 0.), failures={k: 1 for k in q.active})
        for sweep in range(sweeps):
            value = self.try_score(q) or unresolved
            owners = sorted(q.active, key=lambda k: (-value['failures'][k], k))
            for owner in owners:
                best, score = q, self.try_score(q) or unresolved
                angle = float(q.angles_deg[owner])
                for degrees in [-30., 30., -60., 60., -120., 120., 180.]:
                    try:
                        trial = normalise(m.spin_guest(q, owner, angle+degrees))
                    except (ValueError, RuntimeError, AssertionError):
                        continue
                    s = self.try_score(trial)
                    if s is not None and s['rank'] < score['rank']:
                        best, score = trial, s
                # Refine this object's spin continuously about the current best.
                if best.key() != q.key():
                    center = float(best.angles_deg[owner]); reference = best.copy()
                    def objective(degrees):
                        nonlocal best, score
                        try:
                            trial = normalise(m.spin_guest(reference, owner, degrees))
                            s = self.try_score(trial)
                        except (ValueError, RuntimeError, AssertionError):
                            return 1e9
                        if s is None:
                            return 1e9
                        if s['rank'] < score['rank']:
                            best, score = trial, s
                        return 1000*s['rank'][0]+10*s['rank'][1]+s['rank'][2]
                    minimize_scalar(objective, bounds=(center-25., center+25.), method='bounded',
                                    options=dict(maxiter=7, xatol=.1))
                for height in [0., .005, .015, .03]:
                    m.activate(best); t = installed_state(m.native, best, int(best.hosts[owner]))
                    delta = height-m.workpiece_height(best, owner)
                    trial = normalise(shift(m, best, owner, t[:3, :3].T @ [0., 0., delta]))
                    s = self.try_score(trial)
                    if s is not None and s['rank'] < score['rank']:
                        best, score = trial, s
                q = best; self.guide.commit(q); self.group_invariant(q)
                self.save(q, 'fixed-groups-pose-repair', owner=m.poses[owner], sweep=sweep,
                          guide_rank=list(score['rank']))
            if (self.try_score(q) or unresolved)['rank'][0] == 0:
                break
        return q

    def solve(self, height=0., maximum_rings=8, resume=None):
        m = self.model; m.require_connected_core = False
        if resume is not None:
            from .angular_model import AngularLayout
            with np.load(resume) as archive:
                q = AngularLayout.from_arrays(archive)
            self.group_invariant(q); self.guide.commit(q)
            item = self.certify(q, 'resumed-balanced-installations')
            if self.accepted is not None:
                return item
            q = self.pose_repair(q, sweeps=3)
            item = self.certify(q, 'repaired-balanced-installations')
            if self.accepted is not None:
                return item
            raise RuntimeError('Resumed groups need further geometric repair')
        from .grouped_placement import fit_groups
        q = self.seed()
        q, placement = fit_groups(m, q)
        self.save(q, 'joint-group-floor-placement', placement=placement)
        # Brief whole-group clearance proposals, then prioritize object poses.
        # Equal outward displacement preserves every pair of floor halfspaces:
        # its slack increases by r*(1 - dot(n_i,n_j)) >= 0.
        candidates = []
        for radius in [0., .015, .03, .045, .06]:
            trial = self.spread(q, radius); score = self.try_score(trial)
            if score is not None:
                candidates.append((score['rank'], trial))
            self.save(trial, 'whole-group-clearance-proposal', radius_m=radius,
                      guide_rank=list(score['rank']) if score else None)
        if not candidates:
            raise RuntimeError('No resolved shared-installation proposal')
        _, q = min(candidates, key=lambda row: row[0]); self.guide.commit(q)
        for attempt in range(8):
            score = self.try_score(q)
            if score is None or score['rank'][0] > 0 or attempt > 0:
                q = self.pose_repair(q, sweeps=1)
            self.group_invariant(q)
            score = self.try_score(q)
            if score is not None and score['rank'][0] <= 1:
                item = self.certify(q, 'balanced-shared-installations')
                if self.accepted is not None:
                    self.group_invariant(q)
                    return item
            # Only after object-angle/height repair: move entire groups by a
            # small amount, retaining all five installations and all members.
            candidates = []
            for radius in [.005, .015, .03]:
                trial = self.spread(q, radius); value = self.try_score(trial)
                if value is not None:
                    candidates.append((value['rank'], trial))
            if candidates:
                rank, q = min([(score['rank'], q)] + candidates, key=lambda row: row[0])
                self.guide.commit(q)
                self.save(q, 'post-pose-repair-group-clearance', attempt=attempt, guide_rank=list(rank))
        raise RuntimeError('Balanced installation search has not certified a complete fixture')
