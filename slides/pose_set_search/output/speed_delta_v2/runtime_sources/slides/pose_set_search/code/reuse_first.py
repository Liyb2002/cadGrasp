"""Rotate the same fitted fixture first; selectively Juxtapose stalled tasks.

Q=I and host=k preserve the same object-to-fixture seating relation. The
fixture then follows native task k into the world. Host count is NOT a cost.
Stalled tasks and not-yet-accepted insertions may start Juxtapose branches;
successful registered tasks cannot be translated inside those branches.
"""
import time
from common import *
from search import Search, failed_count, lost_protected
from volume_guidance import VolumeGuidance


def registered(layout, k):
    return int(layout.hosts[k]) == k and np.allclose(layout.placements[k], np.eye(4), atol=1e-10, rtol=0)


def juxtaposed_count(layout):
    return sum(not registered(layout, k) for k in layout.active)


class ReuseFirstSearch(Search):
    def score(self, result, anchor=None):
        return (lost_protected(anchor, result), failed_count(result),
                result['volume_cm3'], result['maximum_projected_footprint_m2'],
                juxtaposed_count(result['layout']))

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
                    trial = self.model.evaluate(layout)
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

    def rescue(self, current, guests, anchor=None, secondary=False, uncommitted_guests=()):
        # Every guest failed BEFORE this branch. A passing registered task may
        # not be sacrificed and then rehosted by an implicit secondary rescue.
        # An uncommitted insertion is stalled if its cuts damage an old task,
        # even when its own loads pass. It is not a previously accepted task.
        guests = [k for k in guests if not current['masks'][k].all() or k in uncommitted_guests]
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
                branch = self.model.evaluate(layout)
                record['juxtapose_counts'] = branch['counts']
                if not branch['masks'][detail['guest_index']].all() or lost_protected(anchor, branch):
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
            if lost_protected(anchor, current) == 0:
                anchor = current
        for iteration in range(self.iterations):
            if failed_count(current) == 0 and lost_protected(anchor, current) == 0:
                break
            before = current
            worst = self.targets(current)[1]
            guests = sorted([k for k, mask in current['masks'].items() if not mask.all()],
                            key=lambda k: (k != worst['pose_index'], int(current['masks'][k].sum())))
            if insertion_guest is not None:
                if insertion_guest in guests:
                    guests.remove(insertion_guest)
                guests.insert(0, insertion_guest)
            # A single failing task changes its seating per committed branch.
            current = self.rescue(current, guests[:3], anchor=anchor,
                                  uncommitted_guests=(() if insertion_guest is None else (insertion_guest,)))
            if current is before:
                break
            anchor = current
            current = self.refine(current, 1, anchor=anchor, phase='direction_after_commit')
            anchor = current
        return current

    def restore_reuse(self, current):
        if failed_count(current):
            return current
        initial, _ = self.model.initial(active=current['layout'].active)
        targets = self.guard_targets(current['layout'])
        reference = self.model.proxy(current['layout'], targets)
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
            # Returning to Q=I is a sampled candidate, not a mandatory full
            # Boolean/gradient search for every unsuccessful return. Screen
            # its effects on ALL tasks before spending exact evaluations.
            proposals = [];seen = set()
            for direction in candidates:
                frame = tangent_frame(direction)
                for degrees, vector in [(0., frame[:, 0])] + [
                        (degrees, sign*frame[:, axis]) for degrees in [.5, 2., 5.]
                        for axis in range(2) for sign in [-1., 1.]]:
                    trial = layout.copy()
                    trial.directions[k] = legal_direction(direction+np.tan(np.radians(degrees))*vector,
                                                         self.model.floor_normal(trial, k))
                    if trial.key() in seen:
                        continue
                    seen.add(trial.key())
                    try:
                        proxy = self.model.proxy(trial, targets)
                        if proxy['loss'] <= reference['loss']+1e-9:
                            proposals.append((proxy['loss'], proxy['sum_loss'], degrees, trial, proxy))
                    except (RuntimeError, ValueError):
                        pass
            proposals.sort(key=lambda item:item[:3])
            restored = None
            row = dict(phase='restore_rotating_reuse', pose=self.model.poses[k], trials=[],
                       sampled_candidates=len(seen), proxy_feasible_candidates=len(proposals),
                       screening_is_not_infeasibility_proof=True)
            for _, _, _, trial, proxy in proposals[:min(2, self.finalists)]:
                record = dict(accepted=False, force_guidance=proxy)
                try:
                    branch = self.model.evaluate(trial)
                    record.update(counts=branch['counts'], serial=branch['serial'])
                    # Rotating reuse is accepted only if it does not increase
                    # actual material, within the existing geometry tolerance.
                    if failed_count(branch) == 0 and branch['volume_cm3'] <= current['volume_cm3']+TOL*1e6:
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

    def guard_targets(self, layout):
        targets = []
        for k in layout.active:
            demands = self.model.tasks[k].targets
            for axis in range(6):
                for index in [int(np.argmax(demands[:, axis])), int(np.argmin(demands[:, axis]))]:
                    if (k, index) not in targets:
                        targets.append((k, index))
        return targets

    def polish_volume(self, current, rounds=2):
        if failed_count(current):
            return current
        m = self.model
        before_volume = current['volume_cm3']
        for iteration in range(rounds):
            layout = current['layout']
            proposals = []
            for k in layout.active:
                frame = tangent_frame(layout.directions[k])
                for degrees in [.5, 2., 5.]:
                    for vector in [frame[:, 0], -frame[:, 0], frame[:, 1], -frame[:, 1]]:
                        trial = layout.copy()
                        trial.directions[k] = legal_direction(layout.directions[k]+np.tan(np.radians(degrees))*vector,
                                                             m.floor_normal(layout, k))
                        proposals.append(('direction', trial, dict(pose=m.poses[k], step_degrees=degrees)))
            for axis in np.eye(3):
                for degrees in [-2., 2., -5., 5.]:
                    trial = layout.copy()
                    for k in layout.active:
                        trial.directions[k] = legal_direction(layout.directions[k]+np.radians(degrees)*np.cross(axis, layout.directions[k]),
                                                             m.floor_normal(layout, k))
                    proposals.append(('direction-coherent', trial, dict(step_degrees=degrees, axis=axis.tolist())))
            for k in layout.active:
                if registered(layout, k):
                    continue
                frame = tangent_frame(m.floor_normal(layout, k))
                for fraction in [1/128, 1/32, 1/8]:
                    for vector in [frame[:, 0], -frame[:, 0], frame[:, 1], -frame[:, 1]]:
                        trial = layout.copy()
                        trial.placements[k, :3, 3] += m.extent*fraction*vector
                        proposals.append(('translation', trial, dict(pose=m.poses[k], step_m=m.extent*fraction)))
            targets = self.guard_targets(layout)
            layouts = [layout]+[p[1] for p in proposals]
            guidance = (m.volume_guidance(layouts) if hasattr(m, 'volume_guidance')
                        else VolumeGuidance(m, layouts))
            if hasattr(m, 'refresh_volume'):
                m.refresh_volume(current)
            estimates = [];seen = set()
            for kind, trial, detail in proposals:
                key = trial.key()
                if key in seen or key in m.unresolved_layouts:
                    continue
                seen.add(key)
                try:
                    estimates.append((guidance.estimate(trial), kind, trial, detail))
                except (RuntimeError, ValueError):
                    continue
            estimates.sort(key=lambda row: row[0])
            pool = estimates[:self.finalists*6]
            # Preserve a slot for both direction and translation if available.
            for tool in ['direction', 'translation']:
                alternative = next((row for row in estimates if row[1].split('-')[0] == tool), None)
                if alternative is not None and not any(row[2].key() == alternative[2].key() for row in pool):
                    pool.append(alternative)
            screened = []
            for estimate, kind, trial, detail in pool:
                try:
                    proxy = m.proxy(trial, targets)
                    screened.append((proxy['loss'], estimate, kind, trial, detail, proxy))
                except (RuntimeError, ValueError):
                    continue
            screened.sort(key=lambda row: row[:2])
            selected = [];tools = set()
            for row in screened:
                tool = row[2].split('-')[0]
                if tool not in tools:
                    selected.append(row);tools.add(tool)
                if len(selected) >= self.finalists:
                    break
            for row in screened:
                if len(selected) >= self.finalists:
                    break
                if any(row[3].key() == other[3].key() for other in selected):
                    continue
                selected.append(row)
            event = dict(phase='feasible_volume_descent', iteration=iteration+1,
                         before_volume_cm3=current['volume_cm3'], sampled_layouts=len(estimates),
                         guidance='32768 fixed Sobol points in candidate bbox; nominal geometry only', trials=[])
            best = current
            for _, estimate, kind, trial, detail, proxy in selected:
                record = dict(operation=kind, detail=detail, estimated_volume_cm3=estimate,
                              force_guidance=proxy, accepted=False)
                try:
                    result = m.evaluate(trial)
                    record.update(counts=result['counts'], volume_cm3=result['volume_cm3'], serial=result['serial'])
                    if failed_count(result) == 0 and result['volume_cm3'] < best['volume_cm3']-TOL*1e6:
                        best = result
                        record['eligible'] = True
                except (RuntimeError, ValueError, AssertionError) as error:
                    record['error'] = str(error)
                event['trials'].append(record)
            if best is not current:
                for record in event['trials']:
                    record['accepted'] = record.get('serial') == best['serial']
                self.checkpoint(best, 'feasible_volume_descent')
            event.update(accepted=best is not current, after_volume_cm3=best['volume_cm3'])
            self.record(event)
            print('VOLUME DESCENT', iteration+1, current['volume_cm3'], '->', best['volume_cm3'], flush=True)
            if best is current:
                break
            current = best
        self.volume_descent = dict(before_cm3=before_volume, after_cm3=current['volume_cm3'])
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
            objective='all original force/torque/exit/work constraints first, then actual material volume',
            volume_descent=getattr(self, 'volume_descent', None),
            separated_fallback_used=False, failure_scope='Finite search failure is not an infeasibility proof', **extra))

    def joint(self):
        began = time.monotonic()
        layout, initialization = self.model.initial()
        current = self.initialize_exact(layout, 'registered_initial')
        initial_counts = current['counts']
        self.baseline(current)
        current = self.solve_frontier(current, anchor=current)
        current = self.restore_reuse(current)
        current = self.polish_volume(current, getattr(self, 'volume_rounds', 2))
        return self.save(current, 'joint', began, initialization, initial_counts=initial_counts)

    def incremental(self, start_prefix=None):
        began = time.monotonic()
        prefix_extra = {}
        if start_prefix is None:
            layout, initialization = self.model.initial(active=(0,))
            current = self.initialize_exact(layout, 'incremental_registered_initial')
            self.baseline(current)
            current = self.solve_frontier(current)
            stages = [dict(added_pose=self.model.poses[0], passed=failed_count(current) == 0, counts=current['counts'])]
        else:
            import hashlib
            import json
            from model import Layout
            source = start_prefix.resolve()
            with np.load(source) as saved:
                layout = Layout(saved['placements'].copy(), saved['directions'].copy(), saved['hosts'].copy(),
                                tuple(map(int, saved['active'])))
                assert layout.active == tuple(range(len(layout.active)))
                assert len(layout.placements) == len(self.model.poses)
            stages_path = source.parent.parent/'incremental/insertion_stages.json'
            stages = json.loads(stages_path.read_text())[:len(layout.active)]
            assert len(stages) == len(layout.active) and all(row['passed'] for row in stages)
            assert [row['added_pose'] for row in stages] == self.model.poses[:len(stages)]
            _, initialization = self.model.initial(active=layout.active)
            current = self.initialize_exact(layout, 'accepted_prefix_regenerated')
            assert failed_count(current) == 0
            self.baseline(current)
            prefix_extra = dict(source_accepted_prefix=str(source.relative_to(ROOT)),
                source_accepted_prefix_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                prefix_pose_count=len(stages), prefix_fully_regenerated=True,
                source_insertion_stages=str(stages_path.relative_to(ROOT)))
            self.record(dict(phase='resume_accepted_incremental_prefix', **prefix_extra))
        committed = current if failed_count(current) == 0 else None
        for guest in range(len(stages), len(self.model.poses)):
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
        current = self.polish_volume(current, getattr(self, 'volume_rounds', 2))
        return self.save(current, 'incremental', began, initialization, insertion_stages=stages,
                         requested_pose_count=len(self.model.poses), inserted_pose_count=len(stages),
                         every_insertion_passed=all(stage['passed'] for stage in stages), **prefix_extra)

    def optimize_existing(self, source, mode):
        """Resume a saved construction with full exact regeneration, then reduce volume."""
        import json
        import hashlib
        source = source.resolve()
        began = time.monotonic()
        old = json.loads((source/'report.json').read_text())
        assert old['poses'] == self.model.poses
        assert all(C.I.sha256(ROOT/name) == expected for name, expected in old['provenance']['inputs'].items())
        with np.load(source/'layout.npz') as data:
            from model import Layout
            layout = Layout(data['placements'].copy(), data['directions'].copy(), data['hosts'].copy(),
                            tuple(map(int, data['active'])))
        current = self.initialize_exact(layout, 'resumed_construction_regenerated')
        self.baseline(current)
        if not getattr(self, 'recheck_only', False):
            current = self.solve_frontier(current, anchor=current) if failed_count(current) else current
            current = self.restore_reuse(current)
            current = self.polish_volume(current, getattr(self, 'volume_rounds', 2))
        else:
            assert failed_count(current) == 0, 'Final layout no longer satisfies the exact constraints'
        extra = dict(source_construction=str(source.relative_to(ROOT)),
                     source_report_sha256=hashlib.sha256((source/'report.json').read_bytes()).hexdigest(),
                     source_policy=old.get('policy'), construction_history=old.get('events', []),
                     final_layout_regenerated_without_further_search=getattr(self, 'recheck_only', False),
                     source_volume_descent=old.get('volume_descent_events', old.get('volume_descent')))
        if mode == 'incremental':
            extra.update(insertion_stages=old['insertion_stages'], every_insertion_passed=all(
                stage['passed'] for stage in old['insertion_stages']), requested_pose_count=len(self.model.poses))
        return self.save(current, mode, began, old['initialization'], **extra)

    def resume_layout(self, source, mode):
        import hashlib
        from model import Layout
        source = source.resolve()
        began = time.monotonic()
        with np.load(source) as data:
            layout = Layout(data['placements'].copy(), data['directions'].copy(), data['hosts'].copy(),
                            tuple(map(int, data['active'])))
        assert layout.active == tuple(range(len(self.model.poses)))
        _, initialization = self.model.initial()
        current = self.initialize_exact(layout, 'resumed_exact_layout')
        self.baseline(current)
        initial_counts = current['counts']
        current = self.solve_frontier(current, anchor=current)
        current = self.restore_reuse(current)
        current = self.polish_volume(current, getattr(self, 'volume_rounds', 2))
        return self.save(current, mode, began, initialization, initial_counts=initial_counts,
                         resumed_layout=str(source.relative_to(ROOT)),
                         resumed_layout_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                         resumed_layout_fully_regenerated=True)
