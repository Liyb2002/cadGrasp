"""Strictly volume-decreasing, fully certified design operations from Step3.2."""
from copy import copy
from pathlib import Path
import json
import math
import time

import numpy as np
import trimesh
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from whole_task.step1.common import ROOT, DEFAULT_OUTPUT, sha256, write_json
from whole_task.common.force_cone import tangent_frame
from whole_task.step3 import independent_geometry as G
from whole_task.operations.clearance import refine_candidate
from whole_task.operations import Layout, translation, pose_change, exit_direction_change
from whole_task.operations.state import transform
from whole_task.operations.construction import CompactConstructionModel
from whole_task.operations.installations import standing_pairs


def subset(model, layout, owners):
    sub = copy(model)
    sub.ids = [model.ids[k] for k in owners]
    sub.records = [model.records[k] for k in owners]
    q = Layout(*(array[owners].copy() for array in (layout.poses, layout.material, layout.exits)))
    return sub, q


def legal_improvement(previous_volume, candidate):
    """An approximate force screen can never authorize a state transition."""
    audit = candidate.get('audit', {})
    tasks=candidate.get('tasks',[])
    rows=audit.get('tasks',[])
    paths=audit.get('continuous_paths',[])
    if (not tasks or len(rows)!=len(tasks)
            or len({r['id']for r in rows})!=len(tasks)
            or {r['id']for r in rows}!={t['id']for t in tasks}):
        return False
    if (len(paths)!=len(tasks) or {r['id']for r in paths}!={t['id']for t in tasks}
            or not all(r.get('passed')for r in paths)
            or not all(r.get('floor',{}).get('passed') and
                       r['floor'].get('all_pressure_centers_on_actual_bottom') and
                       r['floor'].get('actual_contact_area_m2',0)>0 for r in rows)):
        return False
    return bool(audit.get('passed') and audit.get('full_loads') and audit.get('source_face_prisms_checked')
                and all(r['force']['passed'] and not r['force'].get('preflight_only', False)
                        and r['force']['original_sample_count']==r['force']['passed_original_sample_count']>0
                        and r['force']['continuous_outer_target_count']==r['force']['passed_continuous_outer_target_count']>0
                        for r in audit.get('tasks', []))
                and candidate['volume_cm3'] < previous_volume - 1e-5)


def record(path,value):
    path=Path(path)
    temporary=path.with_name(path.name+'.tmp')
    write_json(temporary,value)
    temporary.replace(path)


class Optimizer:
    def __init__(self, model, folder, resume=False, audit_resume=True):
        self.model, self.folder = model, Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.events = []
        self.started = time.monotonic()
        self.accepted = 0
        requirements=self.folder/'requirements.json'
        self.requirements=json.loads(requirements.read_text())if requirements.exists()else{}
        manifest = self.folder/'current.json'
        if resume and manifest.exists():
            saved = json.loads(manifest.read_text())
            if saved['initial_report_sha256'] != sha256(model.folder/'report.json'):
                raise ValueError('Cannot resume against a different Step3 initialization')
            path = self.folder/saved['state']
            if sha256(path) != saved['sha256']:
                raise ValueError('Changed checkpoint')
            metadata=json.loads(path.with_suffix('.json').read_text())
            ribs=metadata.get('construction',{}).get('ribbed_stock')
            if ribs:model.stock_grid=(ribs['pitch_m'],ribs['post_width_m'])
            with np.load(path) as z:
                self.layout = Layout.read(z)
                support = trimesh.Trimesh(z['vertices'], z['faces'], process=False)
            self.volume = float(support.volume*1e6)
            self.groups = saved['groups']
            self.accepted = saved['accepted']
            self.events = json.loads((self.folder/'trace.json').read_text())
            self.support = support
            if audit_resume:
                candidate=dict(layout=self.layout,support=support,tasks=model.prepare(self.layout),volume_cm3=self.volume)
                record(self.folder/'resume_validation.json',model.audit(candidate,full=True,continuous=True))
                from .pose_neighbors import remember_contacts
                remember_contacts(model,candidate['tasks'])
        else:
            if manifest.exists():
                raise ValueError('Use --resume to preserve existing accepted operation states')
            self.layout = model.initial.copy()
            self.support = model.initial_support.copy()
            self.volume = float(self.support.volume*1e6)
            self.groups = [[i] for i in range(len(model.ids))]
            baseline = dict(layout=self.layout, support=self.support, volume_cm3=self.volume,
                            audit=dict(passed=True, inherited_step32_certificate=True,
                                       validation_sha256=sha256(model.folder/'validation.json')))
            self.save(baseline, 'accepted_step3_initialization', {})
        self.initial_volume = model.report['material_volume_cm3']

    def event(self, status, operation, **detail):
        row = dict(status=status, operation=operation, elapsed_s=time.monotonic()-self.started, **detail)
        self.events.append(row)
        record(self.folder/'trace.json', self.events)
        print('STEP4', status, operation, detail, flush=True)

    def save(self, candidate, operation, detail):
        name = f'state_{self.accepted:03}.npz'
        path = self.folder/name
        temporary=path.with_name(path.stem+'.tmp.npz')
        np.savez_compressed(temporary, **candidate['layout'].arrays(), vertices=candidate['support'].vertices,
                            faces=candidate['support'].faces)
        temporary.replace(path)
        record(path.with_suffix('.json'), dict(operation=operation, detail=detail,
            volume_cm3=candidate['volume_cm3'], audit=candidate['audit'], mesh_sha256=sha256(path),
            construction=candidate.get('construction',{}),base_construction=candidate.get('base_construction')))
        record(self.folder/'current.json', dict(state=name, sha256=sha256(path), groups=self.groups,
            accepted=self.accepted, volume_cm3=candidate['volume_cm3'], initial_report_sha256=sha256(self.model.folder/'report.json')))
        if not (self.folder/'trace.json').exists():
            record(self.folder/'trace.json', [])

    def try_operation(self, q, operation, *, groups=None, prepared_candidate=None,
                      audit_owners=None, **detail):
        m = self.model
        self.last_trial_status='building'
        self.last_trial_error=''
        candidate=None
        began = time.monotonic()
        owners = None if audit_owners is None else sorted(set(audit_owners))
        parent = json.loads((self.folder/'current.json').read_text())
        parent_audit = json.loads((self.folder/parent['state']).with_suffix('.json').read_text())['audit']
        def preserve_independent():
            from .group_acceptance import retained_certificates,RetainedExitError
            if hasattr(m,'extend_exits_to_actual_support'):
                m.extend_exits_to_actual_support(candidate)
            for attempt in range(len(m.ids)+1):
                try:
                    return retained_certificates(m,self.layout,self.support,candidate,owners,parent_audit)
                except RetainedExitError as error:
                    if attempt==len(m.ids):raise
                    task=candidate['tasks'][m.ids.index(error.task)]
                    scoped=dict(candidate,tasks=[task])
                    cleaned,proof=refine_candidate(m,scoped)
                    candidate['support']=cleaned
                    candidate['volume_cm3']=float(cleaned.volume*1e6)
                    candidate['construction'].setdefault('retained_exit_cleanup',[]).append(
                        dict(task=error.task,refinement=proof,original_path_and_tolerance_unchanged=True))
                    self.event('retained_exit_geometry_refined',operation,task=error.task,
                        volume_cm3=candidate['volume_cm3'])
        def audit_candidate(full=False, continuous=False, all_outer=False):
            if owners is not None:
                preserve_independent()
            scoped = candidate if owners is None else dict(candidate,
                tasks=[candidate['tasks'][k] for k in owners])
            audit = m.audit(scoped,full=full,continuous=continuous,all_outer=all_outer)
            if owners is None:
                return audit
            if full and continuous:
                from .group_acceptance import retained_certificates, combine_audit
                inherited, paths, guards = retained_certificates(m,self.layout,self.support,
                    candidate,owners,parent_audit)
                candidate['audit'] = combine_audit(audit,inherited,paths,guards,
                    parent['sha256'],owners,m.ids)
            return audit
        try:
            if owners is not None:
                from .group_acceptance import validate_partition
                validate_partition(groups,len(m.ids))
                families=[g for g in self.groups if any(k in owners for k in g)]
                if (len(families)!=2 or sorted(k for g in families for k in g)!=owners
                        or len(groups)!=len(self.groups)-1
                        or owners not in groups
                        or any(g not in groups for g in self.groups if g not in families)):
                    raise ValueError('A merge must replace exactly its two existing families by their union')
            if self.requirements.get('final_standing_installation_required') and not standing_pairs(q):
                raise ValueError('A flat-only proposal cannot replace the required standing installation')
            candidate = m.build(q)if prepared_candidate is None else prepared_candidate
            if prepared_candidate is not None:
                for name,array in q.arrays().items():
                    np.testing.assert_array_equal(candidate['layout'].arrays()[name],array)
                # Saved trials never supply a load certificate or task data.
                candidate['tasks']=m.prepare(q)
            candidate['volume_cm3']=float(candidate['support'].volume*1e6)
            if candidate['volume_cm3'] >= self.volume-1e-5:
                self.last_trial_status='no_volume_reduction'
                self.event('no_volume_reduction', operation, volume_cm3=candidate['volume_cm3'], **detail)
                return False
            # Cheap rejection only; nothing is committed here.
            try:
                audit_candidate(all_outer=True)
            except ValueError as error:
                if str(error) != 'Fixture has several material components':
                    raise
                from whole_task.operations.floors import connected_material
                previous_trial_volume=candidate['volume_cm3']
                candidate['support']=G.mesh(connected_material(G.solid(candidate['support'])))
                candidate['volume_cm3']=float(candidate['support'].volume*1e6)
                candidate['construction']['discarded_detached_material_cm3']=previous_trial_volume-candidate['volume_cm3']
                # Removing scraps is another proposed shape, not permission
                # to skip connectivity, floor coverage or any coupled load.
                audit_candidate(all_outer=True)
            self.event('preflight_passed',operation,volume_cm3=candidate['volume_cm3'],**detail)
            if self.requirements.get('final_standing_installation_required'):
                np.savez_compressed(self.folder/'last_preflight.npz',**q.arrays(),
                    vertices=candidate['support'].vertices,faces=candidate['support'].faces)
            scoped = candidate if owners is None else dict(candidate,
                tasks=[candidate['tasks'][k] for k in owners])
            support, refinement = refine_candidate(m,scoped)
            candidate['support'] = support
            candidate['volume_cm3'] = float(support.volume*1e6)
            candidate['construction']['source_face_prisms'] = refinement
            self.event('continuous_exit_passed',operation,volume_cm3=candidate['volume_cm3'],
                       exit_refinement=refinement['local_clearance_history'],**detail)
            if self.requirements.get('final_standing_installation_required'):
                np.savez_compressed(self.folder/'last_refined.npz',**q.arrays(),
                    vertices=support.vertices,faces=support.faces)
            audit_candidate(full=True,continuous=True)
            support=candidate['support']
            if not legal_improvement(self.volume, candidate):
                raise ValueError('Candidate lacks a complete volume-reducing certificate')
        except (ValueError, RuntimeError, AssertionError, np.linalg.LinAlgError) as error:
            self.last_trial_status='rejected'
            self.last_trial_error=str(error)
            self.event('rejected', operation, reason=str(error)[:1200], seconds=time.monotonic()-began, **detail)
            return False
        previous = self.volume
        self.last_trial_status='accepted'
        self.accepted += 1
        if groups is not None:
            self.groups = groups
        # Persist only the fully audited result, before changing the incumbent.
        self.save(candidate, operation, detail)
        self.layout, self.support, self.volume = q.copy(), support, candidate['volume_cm3']
        from .pose_neighbors import remember_contacts
        remember_contacts(m,[t for t in candidate['tasks'] if 'contact_data' in t])
        self.event('accepted', operation, previous_volume_cm3=previous, volume_cm3=self.volume,
                   reduction_fraction=1-self.volume/self.initial_volume,
                   seconds=time.monotonic()-began, groups=[[m.ids[k] for k in g] for g in self.groups], **detail)
        return True

    def solo_rank(self, q, owner):
        m, local = subset(self.model, q, [owner])
        # Remove material-frame translation from proposal ranking only.
        local.material[0, :3, 3] = m.head-local.material[0, :3, :3]@m.head
        try:
            candidate = m.build(local)
            m.audit(candidate, full=False, continuous=False)
            return candidate['volume_cm3']
        except (ValueError, RuntimeError, AssertionError, np.linalg.LinAlgError):
            return math.inf

    def pose_sweep(self):
        m = self.model
        proposals = []
        for owner in range(len(m.ids)):
            baseline = self.solo_rank(self.layout, owner)
            best = (baseline, self.layout, dict(axis='unchanged', degrees=0.))
            normal, center = m.area_axis(self.layout, owner)
            axes = [('needle', m.axis, m.head, [-90., -60., -30., -15., 15., 30., 60., 90., 180.]),
                    ('area_normal', normal, center, [-5., -2., 2., 5.])]
            for label, axis, pivot, angles in axes:
                for angle in angles:
                    q = pose_change(self.layout, owner, angle, axis, pivot)
                    volume = self.solo_rank(q, owner)
                    if volume < best[0]-1e-5:
                        best = (volume, q, dict(axis=label, degrees=angle))
            if best[0] < baseline-1e-5:
                proposals.append((baseline-best[0], owner, best[1], best[2]))
            self.event('pose_proposal_ranked', 'pose_change', task=m.ids[owner], solo_volume_cm3=best[0] if np.isfinite(best[0]) else None,
                       proposal=best[2])
        if not proposals:
            return
        q = self.layout.copy()
        for _, owner, proposal, _ in proposals:
            q.poses[owner] = proposal.poses[owner]
        if self.try_operation(q, 'pose_change', proposals=[dict(task=m.ids[k], **d) for _, k, _, d in proposals]):
            return
        for _, owner, proposal, detail in sorted(proposals, reverse=True, key=lambda x:x[0]):
            q = self.layout.copy()
            q.poses[owner] = proposal.poses[owner]
            self.try_operation(q, 'pose_change', task=m.ids[owner], **detail)

    def direction_sweep(self):
        m = self.model
        proposals = []
        for owner in range(len(m.ids)):
            current = self.layout.exits[owner]
            baseline = self.solo_rank(self.layout, owner)
            best = (baseline, current)
            u, v = tangent_frame(current)
            for axis in (u, v):
                for angle in (-10., -5., 5., 10.):
                    direction = Rotation.from_rotvec(axis*np.radians(angle)).apply(current)
                    q = exit_direction_change(self.layout, owner, direction)
                    volume = self.solo_rank(q, owner)
                    if volume < best[0]-1e-5:
                        best = (volume, direction)
            if best[0] < baseline-1e-5:
                proposals.append((baseline-best[0], owner, best[1]))
            self.event('direction_proposal_ranked', 'exit_direction_change', task=m.ids[owner], solo_volume_cm3=best[0] if np.isfinite(best[0]) else None)
        if not proposals:
            return
        q = self.layout.copy()
        for _, owner, direction in proposals:
            q.exits[owner] = direction
        if self.try_operation(q, 'exit_direction_change', tasks=[m.ids[k] for _, k, _ in proposals]):
            return
        for _, owner, direction in sorted(proposals, reverse=True, key=lambda x:x[0])[:5]:
            self.try_operation(exit_direction_change(self.layout, owner, direction), 'exit_direction_change', task=m.ids[owner])

    def alignment(self, owner, host):
        q, m = self.layout, self.model
        target = q.material[host, :3, :3] @ q.poses[host, :3, :3]
        def loss(x):
            roll = Rotation.from_rotvec(m.axis*np.radians(x[0])).as_matrix()
            yaw = Rotation.from_euler('z', x[1], degrees=True).as_matrix()
            value = q.material[host, :3, :3] @ yaw @ roll @ q.poses[owner, :3, :3]
            return float(np.linalg.norm(value-target)**2)
        result = min((minimize(loss, [a, 0.], method='Nelder-Mead', options={'maxiter':140})
                      for a in (0., 90., 180., 270.)), key=lambda x:x.fun)
        return float(result.fun), result.x

    def merge_proposal(self, owner_group, host_group, owner, host, angles, offset):
        q, m = self.layout.copy(), self.model
        q = pose_change(q, owner, angles[0], m.axis, m.head)
        rotation = q.material[host, :3, :3] @ Rotation.from_euler('z', angles[1], degrees=True).as_matrix()
        center = transform(m.source.center_mass, q.material[host] @ q.poses[host])
        desired = np.eye(4)
        desired[:3, :3] = rotation
        normal=q.material[host,:3,2]
        desired[:3,3]=center-rotation@transform(m.source.center_mass,q.poses[owner])+q.material[host,:3,:2]@np.asarray(offset)
        desired[:3,3]+=normal*(normal@(q.material[host,:3,3]-desired[:3,3]))
        change = desired @ np.linalg.inv(q.material[owner])
        for k in owner_group:
            q.material[k] = change @ q.material[k]
        return q

    def merge_sweep(self, maximum_trials=24):
        m = self.model
        attempted = set()
        trials = 0
        while len(self.groups) > 1 and trials < maximum_trials:
            pairs = []
            for a, ga in enumerate(self.groups):
                for b, gb in enumerate(self.groups):
                    if a==b:
                        continue
                    key = (tuple(ga), tuple(gb))
                    if key in attempted:
                        continue
                    choices = []
                    for owner in ga:
                        for host in gb:
                            score, angles = self.alignment(owner, host)
                            # Changing one member's world pose disturbs a
                            # merged cohort. Prefer moving the smaller seat
                            # set when the orientation fits are comparable.
                            score+=.02*(len(ga)-1)*(1-np.cos(np.radians(angles[0])))
                            choices.append((score, owner, host, angles))
                    pairs.append((min(choices, key=lambda x:x[0]), a, b, key))
            if not pairs:
                break
            (score, owner, host, angles), a, b, key = min(pairs, key=lambda row:row[0][0])
            ga, gb = self.groups[a], self.groups[b]
            attempted.add(key)
            trials += 1
            self.event('merge_proposal', 'support_merge', tasks=[m.ids[owner], m.ids[host]],
                       proposal_rank=score, axial_angle_deg=float(angles[0]), installation_yaw_deg=float(angles[1]))
            merged = False
            for offset in ((0.,0.), (.025,0.), (-.025,0.), (0.,.04)):
                q = self.merge_proposal(ga, gb, owner, host, angles, offset)
                groups = [g for j,g in enumerate(self.groups) if j not in (a,b)] + [sorted(ga+gb)]
                from .spatial_search import floor_repair,separate_free_axis
                q=floor_repair(m,q,groups,host)
                if q is None:
                    continue
                local, lq = subset(m, q, ga+gb)
                try:
                    candidate = local.build(lq)
                    local.audit(candidate, full=False, continuous=False)
                except (ValueError, RuntimeError, AssertionError, np.linalg.LinAlgError) as error:
                    self.event('local_merge_rejected', 'support_merge', tasks=[m.ids[owner], m.ids[host]],
                               offset_m=list(offset), reason=str(error)[:300])
                    continue
                if self.try_operation(q, 'support_merge', groups=groups, tasks=[m.ids[owner],m.ids[host]],
                    axial_angle_deg=float(angles[0]), installation_yaw_deg=float(angles[1]), offset_m=list(offset)):
                    merged = True
                    break
                if self.last_trial_status=='rejected':
                    from .placement import alternatives
                    for shifted,delta in alternatives(m,q,ga+gb):
                        if self.try_operation(shifted,'support_merge',groups=groups,tasks=[m.ids[owner],m.ids[host]],
                                axial_angle_deg=float(angles[0]),installation_yaw_deg=float(angles[1]),
                                offset_m=list(offset),merged_group_translation_m=delta.tolist()):
                            merged=True
                            break
                    if merged:break
                packed=separate_free_axis(m,q,groups,host)
                if not np.allclose(packed.material,q.material,atol=1e-10,rtol=0):
                    if self.try_operation(packed,'support_merge',groups=groups,tasks=[m.ids[owner],m.ids[host]],
                            axial_angle_deg=float(angles[0]),installation_yaw_deg=float(angles[1]),offset_m=list(offset),
                            unrelated_seats_separated=True):
                        merged=True
                        break
            if merged:
                attempted.clear()

    def translation_sweep(self):
        m = self.model
        from .spatial_translation import sweep as compact_sweep
        if compact_sweep(self):
            return
        tasks = m.prepare(self.layout)
        boxes = []
        for group in self.groups:
            points = np.vstack([G.mesh(tasks[k]['stock']).vertices for k in group])
            boxes.append(np.array([points.min(0), points.max(0)]))
        dimensions = np.array([b[1,:2]-b[0,:2] for b in boxes])
        columns = math.ceil(math.sqrt(len(boxes)))
        for gap in (.025, .06, .10):
            cell = dimensions.max(0) + gap
            q = self.layout.copy()
            for index, group in enumerate(self.groups):
                center = boxes[index].mean(0)
                target = np.array([index%columns*cell[0], index//columns*cell[1]])
                delta = np.r_[target-center[:2], 0.]
                for owner in group:
                    q = translation(q, owner, delta)
            from .spatial_search import floor_repair
            q=floor_repair(m,q,self.groups,self.groups[0][0])
            if q is None:
                continue
            if self.try_operation(q, 'translation', layout='compact_group_grid', gap_m=gap):
                break

    def solve(self, phases=('sharing',), merge_trials=24, *, rounds=4, samples=32,
              repair_rounds=1, seed=42, workers=4):
        if 'sharing' in phases:
            if tuple(phases) != ('sharing',):
                raise ValueError('Run the sharing/gradient loop separately from legacy sweeps')
            from .population_search import solve
            solve(self, rounds, samples, repair_rounds, seed, workers)
            if self.accepted == 0 or self.volume >= self.initial_volume-1e-5:
                raise RuntimeError('No fully certified smaller support has been found')
            return self.layout, self.support
        if self.accepted == 0:
            # The first genuine translation also rebuilds the fitted material
            # and pressure-covering base for the operation-based representation.
            self.try_operation(translation(self.layout,0,[-.001,0,0]), 'translation',
                               task=self.model.ids[0], delta_m=[-.001,0,0], rebuild_from_step32=True)
        for phase in phases:
            self.event('phase_started', phase, current_volume_cm3=self.volume)
            if phase == 'poses':
                self.pose_sweep()
            elif phase == 'merges':
                self.merge_sweep(merge_trials)
            elif phase == 'exits':
                self.direction_sweep()
            elif phase == 'translations':
                self.translation_sweep()
            elif phase == 'turns':
                from .spatial_search import sweep
                sweep(self,merge_trials)
            elif phase == 'standing':
                from .aligned_side import sweep
                sweep(self,merge_trials)
            elif phase == 'quarters':
                from .quarter_turns import sweep
                sweep(self,merge_trials)
            elif phase == 'task_turns':
                from .quarter_turns import sweep
                sweep(self,merge_trials,single_tasks=True)
        if self.accepted == 0 or self.volume >= self.initial_volume-1e-5:
            raise RuntimeError('No fully certified smaller support has been found')
        return self.layout, self.support
