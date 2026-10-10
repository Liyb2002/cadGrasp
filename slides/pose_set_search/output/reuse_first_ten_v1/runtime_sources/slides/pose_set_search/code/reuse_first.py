"""Rotate the same fitted fixture first; selectively Juxtapose stalled tasks.

Q=I and host=k preserve the same object-to-fixture seating relation. The
fixture then follows native task k into the world. Host count is NOT a cost.
Only originally unsatisfied tasks may start Juxtapose branches, and successful
registered tasks cannot be translated/rehosted inside those branches.
"""
import time
from common import *
from search import Search, failed_count, lost_protected


def registered(layout, k):
    return int(layout.hosts[k]) == k and np.allclose(layout.placements[k], np.eye(4), atol=1e-10, rtol=0)


def juxtaposed_count(layout):
    return sum(not registered(layout, k) for k in layout.active)


class ReuseFirstSearch(Search):
    def score(self, result, anchor=None):
        return (lost_protected(anchor, result), failed_count(result),
                juxtaposed_count(result['layout']), result['volume_cm3'],
                result['maximum_projected_footprint_m2'])

    def local_proposals(self, current, targets, worst, translation_guests=()):
        # A still-registered pose remains at Q=I, even when a branch damages
        # it. Direction can repair it; the branch cannot silently migrate it.
        movable = tuple(k for k in translation_guests if not registered(current['layout'], k))
        return super().local_proposals(current, targets, worst, movable)

    def refine(self, current, rounds, anchor=None, translation_guests=(), phase='direction'):
        for iteration in range(rounds):
            if failed_count(current) == 0:
                break
            targets, worst, scan = self.targets(current)
            proposals, base = self.local_proposals(current, targets, worst, translation_guests)
            selected, screened = self.shortlist(proposals, targets)
            row = dict(phase=phase, iteration=iteration+1, before_counts=current['counts'],
                       hardest=worst, hardest_scan=scan, proxy_before=base,
                       screened_candidates=screened, trials=[])
            best = current
            for _, _, _, kind, layout, detail, proxy in selected:
                record = dict(operation=kind, detail=detail, proxy=proxy, accepted=False)
                try:
                    trial = self.model.exact(layout)
                    record.update(counts=trial['counts'], rank=self.score(trial, anchor),
                                  lost_protected_loads=lost_protected(anchor, trial), serial=trial['serial'])
                    if self.score(trial, anchor) < self.score(best, anchor):
                        best = trial
                        record['eligible'] = True
                except (RuntimeError, ValueError, AssertionError) as error:
                    record['error'] = str(error)
                row['trials'].append(record)
                if failed_count(best) == 0 and lost_protected(anchor, best) == 0:
                    break
            accepted = best is not current
            if accepted:
                current = best
                self.checkpoint(current, phase)
                for record in row['trials']:
                    record['accepted'] = record.get('serial') == best['serial']
            row.update(accepted=accepted, after_counts=current['counts'])
            self.record(row)
            print('REUSE REFINE', phase, iteration+1, 'accepted', accepted,
                  'failed', failed_count(current), flush=True)
            if not accepted:
                break
        return current

    def world_direction(self, layout, k):
        return self.model.native[layout.hosts[k], :3, :3] @ layout.directions[k]

    def juxtapose_proposals(self, current, guests, targets):
        m, layout = self.model, current['layout']
        rows = []
        for guest in guests:
            original = self.world_direction(layout, guest)
            hosts = [k for k in layout.active if k != guest]
            hosts.sort(key=lambda k: (not current['masks'][k].all(),
                       -float(original @ self.world_direction(layout, k))))
            for host in hosts[:4]:
                host_world = self.world_direction(layout, host)
                angle = float(np.degrees(np.arccos(np.clip(original @ host_world, -1, 1))))
                base = m.juxtapose(layout, guest, host)
                if base.key() == layout.key():
                    continue
                frame = tangent_frame(m.floor_normal(base, guest))
                cg = C.transform_points(m.mesh.center_mass[None], base.placements[guest])[0]
                ch = C.transform_points(m.mesh.center_mass[None], layout.placements[host])[0]
                alignment = frame @ (frame.T @ (ch-cg))
                # Begin with small offsets; a larger overlapping offset remains
                # a candidate, never a far-separated packing fallback.
                shifts = [np.zeros(3), alignment]
                for fraction in [1/32, 1/16, 1/8, 1/3]:
                    for azimuth in np.arange(0, 2*np.pi, np.pi/2):
                        shifts.append(alignment + m.extent*fraction*frame @ [np.cos(azimuth), np.sin(azimuth)])
                host_rotation = m.native[base.hosts[guest], :3, :3]
                for shift in shifts:
                    trial = base.copy()
                    trial.placements[guest, :3, 3] += shift
                    a = transform_mesh(m.mesh, trial.placements[guest]).bounds
                    b = transform_mesh(m.mesh, layout.placements[host]).bounds
                    intersection = np.maximum(0, np.minimum(a[1], b[1])-np.maximum(a[0], b[0]))
                    overlap = float(np.prod(intersection)/min(np.prod(a[1]-a[0]), np.prod(b[1]-b[0])))
                    if overlap < .03:
                        continue
                    for weight in [0., .5, 1.]:
                        # 0 preserves the guest's previous WORLD path; 1 uses
                        # the nearby host path. No forced common-world +Z.
                        world = (1-weight)*original + weight*host_world
                        if np.linalg.norm(world) < 1e-10:
                            continue
                        alternative = trial.copy()
                        alternative.directions[guest] = legal_direction(
                            host_rotation.T @ world, m.floor_normal(alternative, guest))
                        rows.append(('juxtapose', alternative, dict(
                            guest=m.poses[guest], host=m.poses[host], guest_index=guest,
                            host_direction_angle_degrees=angle, direction_host_weight=weight,
                            horizontal_offset_fixture_m=shift.tolist(), body_bbox_overlap_fraction=overlap)))
        return rows

    def rescue(self, current, guests, anchor=None, secondary=False):
        # Every guest failed BEFORE this branch. A passing registered task may
        # not be sacrificed and then rehosted by an implicit secondary rescue.
        guests = [k for k in guests if not current['masks'][k].all()]
        targets, worst, scan = self.targets(current)
        proposals = self.juxtapose_proposals(current, guests, targets)
        selected, screened = self.shortlist(proposals, targets)
        row = dict(phase='selective_juxtapose', guests=[self.model.poses[k] for k in guests],
                   hardest=worst, hardest_scan=scan, before_counts=current['counts'],
                   screened_candidates=screened, trials=[])
        best = current
        for _, _, _, kind, layout, detail, proxy in selected:
            record = dict(operation=kind, detail=detail, proxy=proxy, accepted=False)
            try:
                branch = self.model.exact(layout)
                record['juxtapose_counts'] = branch['counts']
                if failed_count(branch):
                    movable = tuple(k for k in branch['layout'].active
                                    if not registered(branch['layout'], k) and
                                    (k == detail['guest_index'] or not branch['masks'][k].all()))
                    branch = self.refine(branch, self.branch_rounds, anchor=anchor,
                                         translation_guests=movable, phase='post_selective_juxtapose')
                record.update(counts=branch['counts'], rank=self.score(branch, anchor),
                              serial=branch['serial'], lost_protected_loads=lost_protected(anchor, branch))
                if lost_protected(anchor, branch) == 0 and self.score(branch, anchor) < self.score(best, anchor):
                    best = branch
                    record['eligible'] = True
            except (RuntimeError, ValueError, AssertionError) as error:
                record['error'] = str(error)
            row['trials'].append(record)
            if failed_count(best) == 0 and lost_protected(anchor, best) == 0:
                break
        if best is not current:
            self.checkpoint(best, 'selective_juxtapose')
            for record in row['trials']:
                record['accepted'] = record.get('serial') == best['serial']
        row.update(accepted=best is not current, after_counts=best['counts'])
        self.record(row)
        print('SELECTIVE JUXTAPOSE', 'accepted', best is not current,
              'failed', failed_count(best), 'juxtaposed', juxtaposed_count(best['layout']), flush=True)
        return best

    def solve_frontier(self, current, anchor=None, insertion_guest=None):
        # Spend the direction-only budget before any re-seating operation.
        if not current.get('pending_geometry'):
            current = self.refine(current, max(3, self.branch_rounds), anchor=anchor, phase='rotate_reuse_direction')
        for iteration in range(self.iterations):
            if failed_count(current) == 0 and lost_protected(anchor, current) == 0:
                break
            before = current
            worst = self.targets(current)[1]
            guests = sorted([k for k, mask in current['masks'].items() if not mask.all()],
                            key=lambda k: (k != worst['pose_index'], int(current['masks'][k].sum())))
            if insertion_guest in guests:
                guests.remove(insertion_guest)
                guests.insert(0, insertion_guest)
            # A single failing task changes its seating per committed branch.
            current = self.rescue(current, guests[:3], anchor=anchor)
            if current is before:
                break
            current = self.refine(current, 2, anchor=anchor, phase='direction_after_commit')
        return current

    def restore_reuse(self, current):
        if failed_count(current):
            return current
        initial, _ = self.model.initial(active=current['layout'].active)
        for k in current['layout'].active:
            if registered(current['layout'], k):
                continue
            layout = current['layout'].copy()
            layout.hosts[k] = k
            layout.placements[k] = np.eye(4)
            candidates = [initial.directions[k]]
            kept = [j for j in layout.active if j != k and registered(layout, j)]
            if kept:
                candidates.insert(0, legal_direction(layout.directions[kept].sum(axis=0), self.model.floor_normal(layout, k)))
            restored = None
            row = dict(phase='restore_rotating_reuse', pose=self.model.poses[k], trials=[])
            for direction in candidates:
                layout.directions[k] = direction
                record = dict(accepted=False)
                try:
                    branch = self.model.exact(layout)
                    if failed_count(branch):
                        branch = self.refine(branch, 1, anchor=current, phase='restore_reuse_direction')
                    record.update(counts=branch['counts'], serial=branch['serial'])
                    if failed_count(branch) == 0 and self.score(branch) < self.score(current):
                        restored = branch
                        record['accepted'] = True
                except (RuntimeError, ValueError, AssertionError) as error:
                    record['error'] = str(error)
                row['trials'].append(record)
                if restored is not None:
                    break
            row['accepted'] = restored is not None
            self.record(row)
            if restored is not None:
                current = restored
                self.checkpoint(current, 'restored_rotating_reuse')
        return current

    def save(self, current, mode, began, initialization, **extra):
        layout = current['layout']
        modes = {self.model.poses[k]: ('single_use_rotating_fixture' if registered(layout, k) else 'juxtaposed')
                 for k in layout.active}
        return self.model.save(current, self.out, dict(
            strategy=mode, policy='rotate-first-selective-juxtapose', initialization=initialization,
            seconds=time.monotonic()-began, events=self.events, reuse_modes=modes,
            rotating_reuse_pose_count=sum(registered(layout, k) for k in layout.active),
            juxtaposed_pose_count=juxtaposed_count(layout),
            fixture_placement_count_is_not_cost=True, all_pose_juxtapose_disabled=True,
            separated_fallback_used=False, failure_scope='Finite search failure is not an infeasibility proof', **extra))

    def joint(self):
        began = time.monotonic()
        layout, initialization = self.model.initial()
        current = self.initialize_exact(layout, 'registered_initial')
        initial_counts = current['counts']
        self.baseline(current)
        current = self.solve_frontier(current, anchor=current)
        current = self.restore_reuse(current)
        return self.save(current, 'joint', began, initialization, initial_counts=initial_counts)

    def incremental(self):
        began = time.monotonic()
        layout, initialization = self.model.initial(active=(0,))
        current = self.initialize_exact(layout, 'incremental_registered_initial')
        self.baseline(current)
        current = self.solve_frontier(current)
        committed = current if failed_count(current) == 0 else None
        stages = [dict(added_pose=self.model.poses[0], passed=committed is not None, counts=current['counts'])]
        for guest in range(1, len(self.model.poses)):
            anchor = committed
            layout = current['layout'].copy()
            layout.active = tuple(range(guest+1))
            # Only REGISTERED paths are averaged in common object/fixture
            # coordinates. Juxtaposed paths have different object frames.
            native_members = [k for k in current['layout'].active if registered(layout, k)]
            mean = layout.directions[native_members].sum(axis=0) if native_members else self.model.floor_normal(layout, guest)
            layout.directions[guest] = legal_direction(mean, self.model.floor_normal(layout, guest))
            try:
                # A numerical timeout does not justify abandoning rotating
                # reuse: try the same registered layout's nearby directions.
                trial = self.initialize_exact(layout, f'insert_{self.model.poses[guest]}_registered')
            except (RuntimeError, ValueError, AssertionError) as error:
                self.record(dict(phase='insert_registered_unresolved', added_pose=self.model.poses[guest], error=str(error)))
                trial = dict(current, layout=layout, pending_geometry=True, pending_pose=guest,
                    masks={**current['masks'], guest:np.zeros(len(self.model.tasks[guest].targets), bool)},
                    supplies={**current['supplies'], guest:self.model.floors[guest]},
                    counts={**current['counts'], str(guest):None})
            current = self.solve_frontier(trial, anchor=anchor, insertion_guest=guest)
            passed = failed_count(current) == 0 and lost_protected(anchor, current) == 0
            if passed:
                committed = current
            stages.append(dict(added_pose=self.model.poses[guest], counts=current['counts'], passed=passed,
                               lost_existing_passed_loads=lost_protected(anchor, current),
                               geometry_evaluated=not current.get('pending_geometry', False),
                               rotating_reuse_pose_count=sum(registered(current['layout'], k) for k in current['layout'].active),
                               juxtaposed_pose_count=juxtaposed_count(current['layout'])))
            C.save(self.out/'insertion_stages.json', stages)
            print('REUSE INSERT', self.model.poses[guest], 'passed', passed,
                  'juxtaposed', juxtaposed_count(current['layout']), flush=True)
            if not passed:
                # Do not build an insertion success on a failed prefix.
                break
        current = self.restore_reuse(current)
        return self.save(current, 'incremental', began, initialization, insertion_stages=stages,
                         requested_pose_count=len(self.model.poses), inserted_pose_count=len(stages),
                         every_insertion_passed=all(stage['passed'] for stage in stages))
