"""Delta-based search; exact geometry is reserved for final export validation."""
import time
from collections import OrderedDict
from common import *
from model import Model
from classify import classify
from projection import cone_projection
from delta_guidance import ContactDelta, VolumeDelta
from reuse_first import ReuseFirstSearch, registered, juxtaposed_count
from search import failed_count


class FastModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Force/torque capacity depends on lever-arm extremes, not just area.
        # Keep the original interior quadrature and add points close to each
        # face corner; otherwise a good narrow bearing region may be missed.
        bary = np.array([[2/3,1/6,1/6], [1/6,2/3,1/6], [1/6,1/6,2/3], [1/3]*3,
                         [.998,.001,.001], [.001,.998,.001], [.001,.001,.998]])
        self.points = np.einsum('av,tvc->tac',bary,self.mesh.triangles).reshape(-1,3)
        self.sources = np.repeat(np.arange(len(self.mesh.faces)),len(bary))
        self.point_normals = self.mesh.face_normals[self.sources]
        self.point_areas = np.repeat(self.mesh.area_faces/len(bary),len(bary))
        self.point_rays = []
        for task, transform, _ in self.states:
            points = C.transform_points(self.points,transform)
            normal = -task.domain.mesh.face_normals[self.sources]
            self.point_rays.append(C.U.heads(np.c_[normal,np.cross(points-task.domain.com,normal)],task.scale))
        self.contact_delta = ContactDelta(self)
        self.volume_delta = None
        self.sample_cache = OrderedDict()
        self.force_cache = OrderedDict()
        self.proxy_force_cache = OrderedDict()
        self.worst_cache = OrderedDict()
        self.basis_caches = {k:[] for k in range(len(self.poses))}
        self.sample_calls = 0
        self.complete_sampled_states = OrderedDict()
        self.timing = dict(contact_updates_s=0., occupancy_updates_s=0.,
                           sampled_full_load_checks_s=0., proxy_force_checks_s=0.,
                           hardest_load_checks_s=0.)

    def contact_flags(self, layout):
        began = time.monotonic()
        flags = self.contact_delta.state(layout).available
        self.timing['contact_updates_s'] += time.monotonic()-began
        return flags

    def volume_guidance(self, layouts):
        if self.volume_delta is None:
            self.volume_delta = VolumeDelta(self, layouts)
        else:
            reference = self.contact_delta.base.layout
            self.volume_delta.ensure_bounds([reference]+list(layouts))
        return self.volume_delta

    def supply_at_points(self, k, flags):
        ids = np.flatnonzero(flags)
        full = np.vstack([self.floors[k], self.point_rays[k][ids]])
        column_ids = np.r_[np.arange(len(self.floors[k])), len(self.floors[k])+ids]
        return full, column_ids

    def proxy(self, layout, targets):
        self.proxy_calls += 1
        active = self.contact_flags(layout)
        by_owner = {}
        for k, index in targets:
            by_owner.setdefault(k, []).append(index)
        losses = []
        began = time.monotonic()
        for k, indices in by_owner.items():
            key = k, active[k].tobytes(), tuple(indices)
            if key in self.proxy_force_cache:
                values = self.proxy_force_cache[key]
            else:
                full, column_ids = self.supply_at_points(k, active[k])
                mask, _ = classify(full, self.tasks[k].targets[indices],
                                   basis_cache=self.basis_caches[k], column_ids=column_ids)
                values = [0.]*int(mask.sum())
                for p in np.flatnonzero(~mask):
                    projection = cone_projection(full, C.U.target(self.tasks[k].targets[indices[p]]))
                    if projection['kkt_max_violation'] > 1e-7:
                        raise RuntimeError('Sampled projection KKT unresolved')
                    values.append(float(projection['loss']))
                self.proxy_force_cache[key] = values
                while len(self.proxy_force_cache) > 256:
                    self.proxy_force_cache.popitem(last=False)
            losses.extend(values)
        self.timing['proxy_force_checks_s'] += time.monotonic()-began
        return dict(loss=max(losses, default=0.), sum_loss=sum(losses), span_m=self.layout_span(layout),
                    available_areas_m2={str(k):float(self.point_areas[active[k]].sum()) for k in layout.active})

    def evaluate(self, layout):
        began = time.monotonic()
        flags = self.contact_flags(layout)
        guidance = self.volume_guidance([layout])
        epoch = guidance.epoch
        key = layout.key(), epoch
        if key in self.sample_cache:
            return self.sample_cache[key]
        self.sample_calls += 1
        for k in layout.active:
            if layout.directions[k] @ self.floor_normal(layout,k) < -1e-12:
                raise ValueError('Direction crosses its world ground plane')
            relative = self.native[layout.hosts[k]] @ layout.placements[k]
            if not np.allclose(relative[:3,:3], self.native[k,:3,:3], atol=1e-10) or abs(relative[2,3]-self.native[k,2,3]) > 1e-9:
                raise ValueError('Sampled candidate changes original orientation/height')
        clock = time.monotonic()
        volume = guidance.estimate(layout)
        delta = guidance.delta(layout)
        self.timing['occupancy_updates_s'] += time.monotonic()-clock
        masks, supplies, infos = {}, {}, {}
        clock = time.monotonic()
        for k in layout.active:
            full, column_ids = self.supply_at_points(k, flags[k])
            force_key = k, flags[k].tobytes()
            if force_key in self.force_cache:
                mask, info = self.force_cache[force_key]
                self.force_cache.move_to_end(force_key)
            else:
                mask, info = classify(full, self.tasks[k].targets,
                                      basis_cache=self.basis_caches[k], column_ids=column_ids)
                self.force_cache[force_key] = mask, info
                while len(self.force_cache) > 64:
                    self.force_cache.popitem(last=False)
            masks[k], supplies[k], infos[k] = mask, full, info
        self.timing['sampled_full_load_checks_s'] += time.monotonic()-clock
        # Only a tie breaker. Material ranking is explicitly sampled here.
        extent = np.ptp(np.vstack([C.transform_points(self.mesh.vertices,layout.placements[k])
                                  for k in layout.active]), axis=0)
        result = dict(layout=layout.copy(), serial=self.sample_calls,
                      masks=masks, supplies=supplies, classifiers=infos,
                      counts={str(k):int(m.sum()) for k,m in masks.items()}, volume_cm3=volume,
                      volume_epoch=epoch, material_delta=delta,
                      maximum_projected_footprint_m2=float(np.prod(np.sort(extent)[-2:])),
                      evaluation='sampled_contacts_and_material_occupancy',
                      work_access_enforced_during_candidate_evaluation=True,
                      geometry_verified=False, seconds=time.monotonic()-began)
        self.sample_cache[key] = result
        while len(self.sample_cache) > 24:
            self.sample_cache.popitem(last=False)
        print('SAMPLED', self.sample_calls, 'counts', list(result['counts'].values()),
              'seconds', round(result['seconds'],2), flush=True)
        return result

    def commit(self, result):
        self.contact_delta.commit(result['layout'])
        self.volume_guidance([result['layout']]).commit(result['layout'])
        if len(result['layout'].active) == len(self.poses) and failed_count(result) == 0:
            key = result['layout'].key()
            self.complete_sampled_states[key] = result
            self.complete_sampled_states.move_to_end(key)
            while len(self.complete_sampled_states) > 4:
                self.complete_sampled_states.popitem(last=False)

    def hardest(self, result):
        key = tuple((k, result['supplies'][k].tobytes()) for k in result['layout'].active)
        if key not in self.worst_cache:
            began = time.monotonic()
            self.worst_cache[key] = super().hardest(result)
            self.timing['hardest_load_checks_s'] += time.monotonic()-began
            while len(self.worst_cache) > 8:
                self.worst_cache.popitem(last=False)
        return self.worst_cache[key]

    def refresh_volume(self, result):
        result['volume_cm3'] = self.volume_delta.estimate(result['layout'])
        result['volume_epoch'] = self.volume_delta.epoch


class FastReuseSearch(ReuseFirstSearch):
    def blocking_poses(self,current):
        """Find passing configurations that obstruct failing configurations."""
        m=self.model;layout=current['layout'];state=m.contact_delta.state(layout)
        failing=[k for k,mask in current['masks'].items() if not mask.all()]
        ranked=[]
        for blocker in layout.active:
            if blocker in failing:continue
            area=0.;points=0
            for owner in failing:
                locked=(state.locks[owner,blocker]&(state.coverage_counts[owner]>0)
                        &m.contact_allowed(layout,owner))
                points+=int(locked.sum())
                area+=float((m.point_areas[locked]/state.lock_counts[owner][locked]).sum())
            if points:ranked.append((area,blocker,points))
        ranked.sort(reverse=True)
        return ranked

    def solve_frontier(self,current,anchor=None,insertion_guest=None):
        # Work cones can remove many old contacts at once. A provisional
        # all-pose branch may lose individual loads while recovering the full
        # set; only a fully feasible insertion is committed by incremental().
        # Freezing every partially passed load prevents this recovery.
        current=self.refine(current,max(3,self.branch_rounds),anchor=None,phase='rotate_reuse_direction')
        from search import failed_count
        for iteration in range(self.iterations):
            if failed_count(current)==0:break
            before=current;worst=self.targets(current)[1]
            guests=sorted([k for k,mask in current['masks'].items() if not mask.all()],
                key=lambda k:(k!=worst['pose_index'],int(current['masks'][k].sum())))
            if insertion_guest is not None:
                if insertion_guest in guests:guests.remove(insertion_guest)
                guests.insert(0,insertion_guest)
            current=self.rescue(current,guests[:3],anchor=None,
                uncommitted_guests=(() if insertion_guest is None else (insertion_guest,)))
            if current is before:
                blockers=self.blocking_poses(current)
                if blockers:
                    chosen=[k for _,k,_ in blockers[:3]]
                    self.record(dict(phase='choose_blocking_configurations',
                        candidates=[dict(pose=self.model.poses[k],potential_released_area_m2=area,
                                         blocked_contact_samples=points) for area,k,points in blockers],
                        selected=[self.model.poses[k] for k in chosen],
                        reason='passing configuration obstructs a failing configuration'))
                    current=self.rescue(current,chosen,anchor=None,blocking_guests=chosen)
                if current is before:break
            current=self.refine(current,2,anchor=None,phase='direction_after_commit')
        if failed_count(current):
            # A selected Juxtapose may need one more small lateral step.
            # Do not rehost distant poses or freeze partial load coverage.
            movable=[k for k in current['layout'].active if not registered(current['layout'],k)]
            current=self.refine(current,3,anchor=None,translation_guests=movable,
                                phase='work_cone_final_local_repair')
        # When all active original loads pass, the original full prefix does
        # too. A failed provisional insertion is reported as failure, never
        # turned into an accepted prefix by relaxing the load requirements.
        return current

    def process_snapshot(self, current, phase, decision=None):
        if not getattr(self, 'capture_process', False):
            return
        if not hasattr(self, 'process_rows'):
            self.process_rows, self.process_layouts = [], []
            self.process_started = time.monotonic()
        layout = current['layout']
        previous = self.process_layouts[-1] if self.process_layouts else None
        if previous is not None and previous.key() == layout.key():
            row = self.process_rows[-1]
            if decision is not None:
                row.setdefault('annotations', []).append(dict(phase=phase, decision=decision))
        else:
            index = len(self.process_rows)
            directory = self.out/'process_states'
            directory.mkdir(exist_ok=True)
            path = directory/f'{index:03d}.npz'
            np.savez_compressed(path, placements=layout.placements, directions=layout.directions,
                hosts=layout.hosts, active=np.array(layout.active), native_world=self.model.native)
            changes = []
            if previous is not None:
                for k in layout.active:
                    if k not in previous.active:
                        changes.append(dict(pose=self.model.poses[k], inserted=True))
                        continue
                    angle = float(np.degrees(np.arccos(np.clip(
                        layout.directions[k] @ previous.directions[k], -1., 1.))))
                    shift = float(np.linalg.norm(layout.placements[k,:3,3]-previous.placements[k,:3,3])*1000)
                    rehost = int(layout.hosts[k]) != int(previous.hosts[k])
                    if angle > 1e-5 or shift > 1e-6 or rehost:
                        changes.append(dict(pose=self.model.poses[k], direction_change_degrees=angle,
                            translation_change_mm=shift, old_host=self.model.poses[previous.hosts[k]],
                            host=self.model.poses[layout.hosts[k]], rehosted=rehost))
            row = dict(index=index, phase=phase, layout=str(path.relative_to(self.out)),
                serial=current['serial'], active_poses=[self.model.poses[k] for k in layout.active],
                counts={self.model.poses[k]:int(current['masks'][k].sum()) for k in layout.active},
                failed_load_count=failed_count(current),
                estimated_volume_cm3=current['volume_cm3'],
                volume_epoch=current.get('volume_epoch'), changes=changes,
                rotating_reuse_pose_count=sum(registered(layout,k) for k in layout.active),
                juxtaposed_pose_count=juxtaposed_count(layout), geometry_verified=False,
                elapsed_seconds=time.monotonic()-self.process_started)
            if decision is not None:
                row['decision'] = decision
            self.process_rows.append(row)
            self.process_layouts.append(layout.copy())
        C.save(self.out/'process.json', self.process_rows)

    def process_stage(self, current, stage):
        self.process_snapshot(current, 'insertion_commit' if stage['passed'] else 'insertion_failed', stage)

    def record(self, row):
        row['evaluation'] = 'sampled_contacts_and_material_occupancy'
        row['geometry_verified'] = False
        super().record(row)
        # A branch refinement is provisional. Its shape is shown only if the
        # enclosing Juxtapose wins and preserves the protected demands.
        if row.get('accepted') and row.get('phase') != 'post_selective_juxtapose':
            winners = [trial for trial in row.get('trials', []) if trial.get('accepted')]
            for state in getattr(self, 'process_rows', []):
                if any(trial.get('serial') == state['serial'] for trial in winners):
                    state['decision'] = row
            if getattr(self, 'capture_process', False):
                C.save(self.out/'process.json', self.process_rows)

    def initialize_exact(self, layout, phase):
        result = self.model.evaluate(layout)
        self.model.commit(result)
        self.record(dict(phase=phase, evaluation=result['evaluation'], counts=result['counts']))
        self.process_snapshot(result, phase)
        return result

    def baseline(self, current):
        # No initial OBJ: sampled occupancy is never presented as an exact solid.
        C.save(self.out/'initial_metrics.json', dict(counts=current['counts'],
            estimated_volume_cm3=current['volume_cm3'], geometry_verified=False,
            sampled_feasible=failed_count(current)==0))

    def checkpoint(self, current, label):
        super().checkpoint(current, label)
        C.save(self.out/'accepted_layout.json', dict(label=label,counts=current['counts'],
            sampled_serial=current['serial'],geometry_verified=False,failed_load_count=failed_count(current)))
        self.model.commit(current)
        directory = self.out/'sampled_states'
        directory.mkdir(exist_ok=True)
        layout = current['layout']
        path = directory/f'{len(self.events):04d}_{current["serial"]:04d}'
        np.savez_compressed(path.with_suffix('.npz'), placements=layout.placements,
            directions=layout.directions, hosts=layout.hosts, active=np.array(layout.active))
        C.save(path.with_suffix('.json'), dict(label=label, counts=current['counts'],
            estimated_volume_cm3=current['volume_cm3'], geometry_verified=False,
            material_delta=current['material_delta']))
        if label != 'post_selective_juxtapose':
            # restore_reuse records its choice before checkpoint; other
            # choices are attached when the following event is recorded.
            event = self.events[-1] if self.events and self.events[-1].get('phase') == 'restore_rotating_reuse' else None
            self.process_snapshot(current, label, event)

    def score(self, result, anchor=None):
        # All competing volumes use the current shared sample coordinate frame.
        if result.get('evaluation') and self.model.volume_delta is not None:
            self.model.refresh_volume(result)
        return super().score(result, anchor)

    def shortlist(self, proposals, targets):
        # Juxtapose's Cartesian product can contain hundreds of placements.
        # Keep deterministic coverage of guests/hosts and sample step sizes,
        # rather than solving force guidance for the entire product.
        budget = getattr(self, 'screen_budget', 96)
        if len(proposals) > budget:
            groups = {}
            for row in proposals:
                detail = row[2]
                key = row[0].split('-')[0], detail.get('guest_index'), detail.get('host')
                groups.setdefault(key, []).append(row)
            sampled = []
            quota = max(1, budget//len(groups))
            for rows in groups.values():
                # Retain original/aligned placements; sample the remaining
                # offsets and direction blends without favoring one scale.
                keep = list(range(min(3, quota, len(rows))))
                rest = [j for j in range(len(rows)) if j not in keep]
                if rest and len(keep) < quota:
                    keep.extend(map(int,self.rng.choice(rest,min(quota-len(keep),len(rest)),replace=False)))
                sampled.extend(rows[j] for j in keep)
            proposals = sampled[:budget]
        self.model.volume_guidance([self.model.contact_delta.base.layout]+[p[1] for p in proposals])
        return super().shortlist(proposals, targets)

    def polish_volume(self, current, rounds=2):
        # Parent constructs a candidate pool before obtaining the shared volume
        # sampler. Refresh the baseline after pool bounding-box changes too.
        self.model.refresh_volume(current)
        return super().polish_volume(current, rounds)

    def refine(self, current, *args, **kwargs):
        result = super().refine(current, *args, **kwargs)
        self.model.commit(result)
        return result

    def rescue(self, current, *args, **kwargs):
        result = super().rescue(current, *args, **kwargs)
        # A rejected provisional refinement must not become the baseline for
        # the next material-delta report or the next sampled candidate pool.
        self.model.commit(result)
        return result

    def save(self, current, mode, began, initialization, **extra):
        m = self.model
        search_seconds = time.monotonic()-began
        m.refresh_volume(current)
        self.process_snapshot(current, 'final_sampled_result')
        m.timing['occupancy_updates_s'] = m.volume_delta.seconds
        layout = current['layout']
        sample_report = dict(complete=True, search_complete=True, final_acceptance_run=False,
            force_exit_work_passed=False, geometry_verified=False,
            work_access_enforced_during_search=True,
            work_access_definition=[work.definition() for work in m.work_rays],
            strategy=mode, poses=[m.poses[k] for k in layout.active], pose_count=len(layout.active),
            counts={m.poses[k]:int(current['masks'][k].sum()) for k in layout.active},
            sampled_force_passed=failed_count(current)==0, estimated_volume_cm3=current['volume_cm3'],
            search_seconds=search_seconds, sample_evaluations=m.sample_calls,
            exact_evaluations_during_search=m.exact_calls, proxy_evaluations=m.proxy_calls,
            timing=m.timing, contact_pair_updates=m.contact_delta.pair_updates,
            rotating_reuse_pose_count=sum(registered(layout,k) for k in layout.active),
            juxtaposed_pose_count=juxtaposed_count(layout), events=self.events,
            placements=layout.placements.tolist(), directions=layout.directions.tolist(),
            hosts=layout.hosts.tolist(), **extra)
        if mode == 'incremental':
            sample_report['every_sampled_insertion_passed'] = extra.get('every_insertion_passed',False)
            sample_report['every_insertion_passed'] = None
            sample_report['insertion_prefixes_exactly_verified'] = False
            sample_report['insertion_stage_evaluation'] = 'sampled_contacts'
        C.save(self.out/'search_report.json', sample_report)
        np.savez_compressed(self.out/'sampled_layout.npz', placements=layout.placements,
            directions=layout.directions, hosts=layout.hosts, active=np.array(layout.active))
        if getattr(self, 'search_only', False):
            return sample_report
        # Sampling can be optimistic. A failed or timed-out export never gets a
        # successful acceptance report; there is no Boolean in candidate search.
        validation_began = time.monotonic()
        candidates = [current]+[result for result in reversed(list(m.complete_sampled_states.values()))
                                if result['layout'].key() != layout.key()]
        attempts = []; actual = None; chosen = None
        for candidate in candidates[:getattr(self,'final_candidates',3)]:
            clock = time.monotonic()
            record = dict(sampled_serial=candidate['serial'],passed=False)
            try:
                checked = m.exact(candidate['layout'])
                record.update(counts=checked['counts'],volume_cm3=checked['volume_cm3'],
                              timings=checked.get('timings'),exact_serial=checked['serial'])
                work = checked.get('actual_work_surface_checks',[])
                passed = (failed_count(checked)==0 and len(work)==len(candidate['layout'].active)
                          and all(row['passed'] for row in work))
                record['passed'] = passed
                if passed:
                    actual,chosen = checked,candidate
            except (RuntimeError,ValueError,AssertionError) as error:
                record['error'] = str(error)
            record['seconds'] = time.monotonic()-clock
            attempts.append(record)
            C.save(self.out/'final_validation_attempts.json',attempts)
            if actual is not None:
                break
        if actual is None:
            raise RuntimeError('Final sampled layouts did not obtain full exact acceptance; see final_validation_attempts.json')
        layout = actual['layout']
        if mode == 'incremental':
            extra['every_sampled_insertion_passed'] = extra.get('every_insertion_passed',False)
            extra['every_insertion_passed'] = None
            extra['insertion_prefixes_exactly_verified'] = False
        return Model.save(m, actual, self.out, dict(strategy=mode,
            policy='rotate-first-delta-search-final-exact-validation', initialization=initialization,
            search_seconds=search_seconds, validation_seconds=time.monotonic()-validation_began,
            seconds=time.monotonic()-began, sample_evaluations=m.sample_calls,
            exact_evaluations_during_search=0, timing=m.timing,
            rotating_reuse_pose_count=sum(registered(layout,k) for k in layout.active),
            juxtaposed_pose_count=juxtaposed_count(layout), events=self.events,
            final_validation_attempts=attempts,
            selected_earlier_sampled_state=chosen['layout'].key()!=current['layout'].key(),
            final_acceptance_run=True, **extra))
