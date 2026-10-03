"""Run contact grammar descent on saved B cases, never regenerate their loads."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step1.needs import ContinuousNeeds, demand
from step3_scheculer import contacts as I
from step3_scheculer.pair_scoring import TaskProblem, J
from step3_scheculer import passive_support as U
from step3_scheculer import contact_dsl as DSL
from step3_scheculer import exit_options as EXIT
from step3_scheculer.floor_margin import contact_check, transforms_from_owner
from step2_local_support import withdrawal as W
from step3_scheculer.pair_geometry import horizontal_catalogue, FreePaths
from step2_local_support import circles as P
import trimesh

SCHEMA = 'joint_shared_head_descent_v4'


def saved_task(group, pose):
    """Use immutable saved inputs, including explicitly archived pose revisions."""
    candidates = [group/'step4/data/source_inputs'/pose,
        group/'step3_scheculer/independent_poses_floor2mm'/pose/'step_1_needs',
        group/'step_1_needs'/pose,
        group/'step_1_needs']
    if group.name == 'pose1+3copied':
        candidates.insert(0, group.parent/'pose1+3/step4/data/source_inputs'/pose)
    folder = next((p for p in candidates if (p/'needs.json').is_file() and (p/'samples.json').is_file()), None)
    if folder is None:
        raise FileNotFoundError(f'No saved loads: {group}/{pose}')
    domain_path = folder/'needs.json'; samples_path = folder/'samples.json'
    domain = ContinuousNeeds.read(domain_path)
    raw = json.loads(samples_path.read_text())
    if domain.data['pose_id'] != pose or domain.data['object'] != 'B':
        raise ValueError('Saved task identity mismatch')
    if I.sha256(domain_path) != raw['provenance']['physical_domain_sha256']:
        raise ValueError('Saved loads refer to a different physical domain')
    provenance = domain.data['provenance']
    snapshot_options = [I.ROOT/provenance['setup_snapshot'], I.ROOT/'objects/B/tasks'/pose/'setup.npz']
    snapshot_options += sorted((I.ROOT/'objects/B/history').glob(f'*/tasks/{pose}/setup.npz'))
    snapshot = next((p for p in snapshot_options if p.is_file() and
                     I.sha256(p) == provenance['setup_snapshot_sha256']), None)
    if snapshot is None:
        raise ValueError('No byte-identical saved setup snapshot for this pose revision')
    with np.load(snapshot) as z:
        floor = z['floor_contact_m'].copy()
    targets = np.asarray(raw['need_wrench'])
    np.testing.assert_allclose(targets, demand(raw['pt_m'], raw['force_push_mg'], domain.com, domain.gravity), atol=1e-13, rtol=0)
    if len(targets) != 32768:
        raise ValueError('Expected all 32768 original sampled loads')
    scale = np.r_[np.ones(3), np.ones(3)/domain.mesh.extents.max()]
    task = TaskProblem(pose, domain, floor, np.ascontiguousarray(targets*scale), scale)
    task.inputs = [domain_path, samples_path, snapshot]
    task.random_sample_count = len(targets)
    return task


def seed_contacts(group, pose):
    candidates = [group/'step3_scheculer/independent_poses_floor2mm'/pose/'step3_scheculer'/f'final_contacts_{pose}.npz',
                  group/'step3_scheculer'/f'final_contacts_{pose}.npz',
                  group/'step3_scheculer/co_design_reference'/f'contacts_{pose}.npz']
    if group.name == 'pose1+3copied':
        candidates.append(group.parent/'pose1+3/step3_scheculer/co_design_reference'/f'contacts_{pose}.npz')
    source = next((p for p in candidates if p.is_file()), None)
    return (I.read_contacts(source), source) if source else ([], None)


class Search:
    def __init__(self, task, floor_tasks, out, contacts, config):
        self.task, self.out, self.config = task, out, config
        self.compiler = DSL.Compiler(task, floor_tasks, device=config['device'], gpu_iterations=config['gpu_iterations'])
        self.floor_tasks = floor_tasks
        self.floor_transforms = transforms_from_owner(task, floor_tasks)
        self.catalogue = self.compiler.exit_catalogue
        # Constructor-sized finite roots are a geometric probe, not contact area.
        self.depth = P.DEPTH_FRACTION*task.domain.mesh.extents.max()
        self.analyzer = EXIT.PathAnalyzer(task.domain.mesh, self.depth)
        self.paths = FreePaths(task.domain.mesh,np.array([[0.,0.,1.,0.]]))
        self.path_cache = {}
        self.seeds = self.compiler.seeds(config['seeds'])
        patches = tuple(DSL.Patch(i+1, tuple(c['center_m']), float(c['radius_m'])) for i, c in enumerate(contacts))
        if not patches:
            patches = tuple(replace(p, ident=i+1) for i, p in enumerate(self.seeds[:3]))
        self.initial = DSL.Program(patches)
        self.sample_ids = np.unique(np.linspace(0, len(task.targets)-1, config['samples'], dtype=int))
        self.lp_cache = {}
        self.direction_cache = {}
        self.events = []
        self.next_ident = 2000

    def emit(self, **event):
        self.events.append(event)
        I.save(self.out/'progress.json', dict(complete=False, events=self.events))
        print(self.task.pose, event.get('operation'), 'heads', event.get('heads'),
              'covered', event.get('covered'), flush=True)

    def classify(self, program, refine=False):
        contacts = self.compiler.compile(program)
        if contacts is None:
            return dict(covered=0, passed=False, mask=np.zeros(32768, bool), contacts=[])
        digest = hashlib.sha256(b''.join(c['triangles_m'].tobytes()+c['source_faces'].tobytes() for c in contacts)).hexdigest()
        if digest not in self.lp_cache:
            try:
                mask, info = J.classify(self.task.supply(contacts), self.task.targets)
            except RuntimeError as error:
                if 'HiGHS' not in str(error) and 'Unresolved equilibrium residual' not in str(error):
                    raise
                # A numerically undecided proposal cannot replace a certified
                # design. Do not call it physically infeasible or relax LP tolerances.
                mask = np.zeros(32768, bool)
                info = dict(indeterminate=True, error=str(error))
                self.emit(operation='reject_numerically_indeterminate_lp_proposal',
                          heads=len(program.patches), covered=0, numerical_error=str(error))
            self.lp_cache[digest] = dict(covered=int(mask.sum()), passed=bool(mask.all()), mask=mask,
                                        contacts=contacts, info=info, indeterminate=info.get('indeterminate',False))
        # Equilibrium depends on geometry, but head identity belongs to the
        # current shared program. Never reuse an earlier rewrite's metadata.
        row = dict(self.lp_cache[digest], contacts=contacts)
        if refine and not row['passed'] and not row.get('indeterminate',False):
            failed = np.flatnonzero(~row['mask'])
            chosen = failed[np.linspace(0, len(failed)-1, min(24, len(failed)), dtype=int)]
            self.sample_ids = np.unique(np.r_[self.sample_ids, chosen])
        return row

    def connectivity(self, contacts):
        key=hashlib.sha256(b''.join(c['triangles_m'].tobytes()+c['center_m'].tobytes() for c in contacts)).hexdigest()
        if key not in self.path_cache:
            ports=[]
            for c in contacts:
                face=int(c['center_face'])
                center=np.asarray(c['center_m'])
                weights=trimesh.triangles.points_to_barycentric(self.task.domain.mesh.triangles[face][None],center[None])[0]
                root=center+.5*weights@self.analyzer.relative.offsets[self.task.domain.mesh.faces[face]]
                ports.append(self.paths.ports(root))
            common=set.intersection(*(set(v) for v in ports)) if ports else set()
            self.path_cache[key]=dict(passed=bool(common),common_path_components=sorted(common),
                head_port_components=ports,roadmap=self.paths.record,
                scope='baseline thin-root roadmap prefilter; full exported witness is mandatory for final acceptance')
        return self.path_cache[key]

    def exits(self, program, reference=None):
        contacts = self.compiler.compile(program)
        if not contacts:
            return [], []
        if not self.connectivity(contacts)['passed']:
            return [], []
        vectors = np.asarray(self.catalogue['vectors'])
        normals = np.vstack([self.task.domain.mesh.face_normals[np.unique(c['source_faces'])] for c in contacts])
        valid = np.flatnonzero((normals@vectors.T).min(axis=0) >= -W.NORMAL_TOL)
        valid = EXIT.ordered_ids(self.task.domain.mesh,self.catalogue['options'],valid)
        key = hashlib.sha256(b''.join(c['triangles_m'].tobytes() for c in contacts)).hexdigest()
        retained, records = [], []
        cells = self.analyzer.heads(contacts)
        for ident in valid:
            cache_key = (key, int(ident))
            if cache_key not in self.direction_cache:
                plan = self.catalogue['options'][ident]
                check = self.analyzer.test(cells,plan)
                self.direction_cache[cache_key] = dict(option_id=int(ident),passed=check['clear'],plan=plan,
                    local_swept_bbox_proxy_cm3=EXIT.local_cost(self.task.domain.mesh,plan),check=check)
            row = self.direction_cache[cache_key]
            records.append(row)
            if row['passed']:
                candidates=[self.catalogue['options'][i] for i in retained]+[row['plan']]
                retained=[p['id'] for p in EXIT.diverse(candidates,self.config['exit_witnesses'])]
                if len(retained) >= self.config['exit_witnesses']:
                    break
        return retained, records

    def improve(self, program, reference, exits, operation):
        optimized, history = DSL.descend(self.compiler, program, self.sample_ids, reference,
                                         exits, self.config['steps'])
        row = self.classify(optimized, refine=True)
        self.emit(operation=operation, heads=len(optimized.patches), covered=row['covered'],
                  gradient_steps=history, program=asdict(optimized))
        return optimized, row

    def mutations(self, program, reference, exits, kind):
        candidates = []
        for seed in self.seeds:
            patch = replace(seed, ident=self.next_ident)
            self.next_ident += 1
            if kind == 'replace':
                candidates.extend(program.substitute(i, patch) for i in range(len(program.patches)))
            else:
                candidates.append(program.add(patch))
        losses = self.compiler.loss_batch(candidates, self.sample_ids, reference, exits)
        ranked = [p for _, p in sorted(zip([v[0] for v in losses], candidates), key=lambda row: row[0])]
        return ranked[:self.config['rewrite_trials']]

    def force_stage(self):
        program = self.initial
        baseline = self.classify(program, refine=True)
        self.emit(operation='saved_seed_force_only', heads=len(program.patches), covered=baseline['covered'],
                  exit_gate_applied=False)
        candidate, row = self.improve(program, None, False, 'force_descent')
        if row['covered'] >= baseline['covered']:
            program, baseline = candidate, row
        attempts = 0
        while not baseline['passed'] and attempts < self.config['force_rewrites']:
            kind = 'replace' if attempts % 2 == 0 or len(program.patches) >= self.config['max_heads'] else 'add'
            for proposal in self.mutations(program, None, False, kind):
                proposal, tested = self.improve(proposal, None, False, 'force_'+kind)
                if tested['covered'] > baseline['covered']:
                    program, baseline = proposal, tested
                if baseline['passed']:
                    break
            attempts += 1
        return program

    def repair(self, program, reference):
        # Repair an escape opening without choosing one shared arrow or an
        # inter-pose reference. Several verified paths remain available to Step4.
        best = program
        ids, records = self.exits(best, reference)
        if self.classify(best)['passed'] and ids:
            best = replace(best, angle=float(np.arctan2(self.catalogue['vectors'][ids[0]][1], self.catalogue['vectors'][ids[0]][0])))
            self.emit(operation='certified_exit_selection', heads=len(best.patches), covered=32768,
                      direction_id=ids[0])
            return best
        best, score = self.improve(best, None, True, 'exit_opening_descent')
        ids, _ = self.exits(best, reference)
        if score['passed'] and ids:
            return replace(best, angle=float(np.arctan2(self.catalogue['vectors'][ids[0]][1], self.catalogue['vectors'][ids[0]][0])))
        for attempt in range(self.config['exit_rewrites']):
            kind = 'replace' if attempt % 2 == 0 or len(best.patches) >= self.config['max_heads'] else 'add'
            for proposal in self.mutations(best, reference, True, kind):
                proposal, tested = self.improve(proposal, reference, True, 'exit_'+kind)
                ids, _ = self.exits(proposal, reference)
                if tested['passed'] and ids:
                    return replace(proposal, angle=float(np.arctan2(self.catalogue['vectors'][ids[0]][1], self.catalogue['vectors'][ids[0]][0])))
                a = self.compiler.loss(proposal, self.sample_ids, reference, True)[0]
                b = self.compiler.loss(best, self.sample_ids, reference, True)[0]
                if a < b:
                    best = proposal
        return best

    def prune(self, program, reference):
        initial = self.classify(program)
        ids, _ = self.exits(program, reference)
        if not initial['passed'] or not ids:
            return program
        changed = True
        while changed and len(program.patches) > 1:
            changed = False
            for index in range(len(program.patches)):
                proposal = program.delete(index)
                # Check before descent too: an unchanged subset may already pass.
                tested = self.classify(proposal, refine=True)
                ids, _ = self.exits(proposal, reference) if tested['passed'] else ([], [])
                if not (tested['passed'] and ids):
                    proposal, tested = self.improve(proposal, reference, True, 'delete_and_repair')
                    ids, _ = self.exits(proposal, reference) if tested['passed'] else ([], [])
                if tested['passed'] and ids:
                    self.emit(operation='accept_delete', heads=len(proposal.patches), covered=32768)
                    program = proposal; changed = True; break
        return program

    def export(self, program, reference, input_paths):
        score = self.classify(program)
        ids, checks = self.exits(program, reference)
        contacts = score['contacts']
        floors = [contact_check(c, [t.pose for t in self.floor_tasks], self.floor_transforms) for c in contacts]
        passed = bool(score['passed'] and ids and contacts and all(r['passed'] for r in floors))
        if ids:
            d = np.asarray(self.catalogue['vectors'][ids[0]])
            program = replace(program, angle=float(np.arctan2(d[1], d[0])))
        else:
            d = DSL.direction(program.angle)
        self.out.mkdir(parents=True, exist_ok=True)
        I.save_contacts(self.out/f'final_contacts_{self.task.pose}.npz', contacts)
        np.savez_compressed(self.out/'coverage.npz', passed=score['mask'])
        report = dict(complete=True, schema=SCHEMA, pose=self.task.pose, passed=False,local_contact_passed=passed,
            passed_scope='contact_proposal_only; case-level Step3 requires a complete construction witness',
            force_passed=score['passed'], exit_passed=bool(ids), floor_margin_passed=all(r['passed'] for r in floors),
            covered_count=score['covered'], sample_count=32768, head_count=len(contacts),
            seed_head_count=len(self.initial.patches), seed_covered_count=self.events[0]['covered'],
            contact_area_m2=I.area(contacts), head_areas_m2=[float(c['triangle_areas_m2'].sum()) for c in contacts],
            head_radii_m=[float(c['radius_m']) for c in contacts],
            contact_model='surface contact with constructor-matched finite roots',
            local_geometry_probe_depth_m=self.depth,root_connectivity=self.connectivity(contacts),
            minimum_head_count_proved=False, complete_fixture_verified=False,
            fixed_area_constraint=False, pressure_limit_added=False, head_budget=self.config['max_heads'],
            search='force-only descent, structure rewrites, any-opening repair, accepted-head deletion; multiple exit paths retained',
            derivative='central finite differences on recompiled real connected mesh patches; backtracking descent',
            acceleration=self.compiler.backend.info,
            numerical_radius_bounds_m=[self.compiler.min_radius, self.compiler.max_radius],
            reference_object_exit=None if reference is None else reference.tolist(),
            selected_withdrawal_world=d.tolist(), selected_object_exit=DSL.object_direction(d, self.task).tolist(),
            selected_exit_option_id=ids[0] if ids else None,
            exit_options=[self.direction_cache[(hashlib.sha256(b''.join(c['triangles_m'].tobytes() for c in contacts)).hexdigest(),int(i))] for i in ids],
            certified_exit_option_ids=ids, catalogue=self.catalogue, exit_checks=checks,
            inter_pose_angle_penalty=False,rotation_paths_tested=False,
            floor_checks=floors, program=asdict(program), config=self.config, events=self.events,
            provenance=dict(inputs=I.hashes(input_paths), code=I.hashes([Path(__file__)]+DSL.sources())),
            artifacts={p:I.sha256(self.out/p) for p in (f'final_contacts_{self.task.pose}.npz', 'coverage.npz')})
        I.save(self.out/'report.json', report)
        I.save(self.out/'progress.json', dict(complete=True, passed=passed, covered=score['covered'], heads=len(contacts)))
        draw_contacts(self.task, contacts, d, self.out/'contacts.png')
        return report


def draw_contacts(task, contacts, direction, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(6, 6), facecolor='white'); ax = fig.add_subplot(projection='3d')
    ax.add_collection3d(Poly3DCollection(task.domain.mesh.triangles, facecolor='#aabac4', edgecolor='none', alpha=.18))
    for i, contact in enumerate(contacts):
        ax.add_collection3d(Poly3DCollection(contact['triangles_m'], facecolor=plt.get_cmap('tab10')(i%10), edgecolor='none'))
    lo, hi = task.domain.mesh.bounds; center = (lo+hi)/2; span = (hi-lo).max()*.58
    ax.set(xlim=(center[0]-span, center[0]+span), ylim=(center[1]-span, center[1]+span), zlim=(center[2]-span, center[2]+span))
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(elev=25, azim=np.degrees(np.arctan2(-direction[1], -direction[0])))
    fig.tight_layout(); fig.savefig(output, dpi=130); plt.close(fig)


def run_independent_group_v3(group, config):
    started = time.monotonic()
    if '+' in group.name:
        poses = ['pose_'+v for v in group.name.removeprefix('pose').removesuffix('copied').split('+')]
    else:
        poses = [group.name]
    out = group/'step3_scheculer/dsl'
    out.mkdir(parents=True, exist_ok=True)
    I.save(out/'report.json', dict(complete=False, schema=SCHEMA, status='running', passed=False, poses=poses))
    tasks = [saved_task(group, p) for p in poses]
    input_paths = [p for t in tasks for p in t.inputs]
    placement_paths = [group/'step4/data/growing_support/report.json', group/'step4/data/report.json']
    placement_source = next((p for p in placement_paths if p.is_file()), None)
    baseline_vectors = None
    if placement_source:
        saved = json.loads(placement_source.read_text())
        if saved.get('poses') == poses and 'directions' in saved.get('placement', {}):
            baseline_vectors = np.asarray(saved['placement']['directions'])
            input_paths.append(placement_source)
    searches, programs = [], []
    for task in tasks:
        contacts, source = seed_contacts(group, task.pose)
        if source:
            input_paths.append(source)
        search = Search(task, tasks, out/task.pose, contacts, config)
        if baseline_vectors is not None:
            v = baseline_vectors[len(searches)]
            search.initial = replace(search.initial, angle=float(np.arctan2(v[1], v[0])))
        program = search.force_stage()
        force = search.classify(program)
        search.out.mkdir(parents=True, exist_ok=True)
        checkpoint = search.out/'force_only_contacts.npz'
        I.save_contacts(checkpoint, force['contacts'])
        np.savez_compressed(search.out/'force_only_coverage.npz', passed=force['mask'])
        I.save(search.out/'force_only_report.json', dict(complete=True, pose=task.pose,
            force_passed=force['passed'], covered_count=force['covered'], sample_count=32768,
            head_count=len(force['contacts']), exit_required=False, program=asdict(program),
            acceleration=search.compiler.backend.info,
            provenance=dict(inputs=I.hashes(input_paths), code=I.hashes([Path(__file__)]+DSL.sources())),
            artifacts={p:I.sha256(search.out/p) for p in ('force_only_contacts.npz', 'force_only_coverage.npz')}))
        searches.append(search); programs.append(program)
    reference = None
    reports = []
    repaired_programs = []
    pruned_programs = []
    for index, search in enumerate(searches):
        program = search.repair(programs[index], reference)
        if search.classify(programs[index])['passed'] and not search.classify(program)['passed']:
            program = programs[index]
            search.emit(operation='reject_exit_repair_keep_force_checkpoint',
                        heads=len(program.patches), covered=32768)
        repaired_programs.append(program)
        program = search.prune(program, reference)
        pruned_programs.append(program)
    def describe(reports):
        selected = [r['selected_object_exit'] for r in reports]
        pair_angles = [float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1))))
                       for i, a in enumerate(selected) for b in selected[i+1:]]
        baseline_angles = []
        if baseline_vectors is not None:
            baseline_objects = [DSL.object_direction(v,t) for v,t in zip(baseline_vectors,tasks)]
            baseline_angles = [float(np.degrees(np.arccos(np.clip(a@b,-1,1))))
                               for i,a in enumerate(baseline_objects) for b in baseline_objects[i+1:]]
        result = dict(complete=True, schema=SCHEMA, group=group.name, poses=poses,
            passed=False,local_contact_passed=all(r['local_contact_passed'] for r in reports), force_passed=all(r['force_passed'] for r in reports),
            exit_passed=all(r['exit_passed'] for r in reports), total_heads=sum(r['head_count'] for r in reports),
            seed_total_heads=sum(r['seed_head_count'] for r in reports),
            covered_counts=[r['covered_count'] for r in reports], head_counts=[r['head_count'] for r in reports],
            mean_object_exit_angle_deg=float(np.mean(pair_angles)) if pair_angles else 0.,
            maximum_object_exit_angle_deg=max(pair_angles, default=0.),
            baseline_mean_object_exit_angle_deg=float(np.mean(baseline_angles)) if baseline_angles else None,
            angles_are_certified=all(r['exit_passed'] for r in reports),
            inter_pose_angle_penalty=False,exit_option_counts=[len(r['exit_options']) for r in reports],
            path_selection_authority='step3 constructive compiler; Step4 consumes its certified witness',complete_fixture_verified=False,
            seconds=time.monotonic()-started, config=config,
            per_pose_reports=[str((out/p/'report.json').relative_to(I.ROOT)) for p in poses],
            provenance=dict(inputs=I.hashes(input_paths), code=I.hashes([Path(__file__)]+DSL.sources())))
        return result
    from step3_scheculer.construction_contract import certify
    attempts=[]
    candidate_sets=[pruned_programs,repaired_programs,programs]
    seen=set()
    for candidate_set in candidate_sets:
        contact_sets=[search.compiler.compile(program) for search,program in zip(searches,candidate_set)]
        signature=tuple(hashlib.sha256(b''.join(c['triangles_m'].tobytes()+c['center_m'].tobytes()+
            np.asarray([c['radius_m']]).tobytes() for c in (row or []))).hexdigest() for row in contact_sets)
        if signature in seen:continue
        seen.add(signature)
        reports=[search.export(program,None,input_paths) for search,program in zip(searches,candidate_set)]
        proposal=describe(reports)
        result=certify(group,proposal,trials=config['construction_trials'])
        attempts.append(dict(head_counts=result['head_counts'],local_contact_passed=result['local_contact_passed'],
                             construction_passed=result['passed'],status=result['status']))
        if result['passed'] or not proposal['local_contact_passed']:break
    I.save(out/'construction_attempts.json',dict(complete=True,attempts=attempts,
        recovery='reject unconstructible pruning; retry pre-prune repaired contacts, then force checkpoint'))
    result['provenance']['inputs'].update(I.hashes([out/'construction_attempts.json']))
    result['seconds']=time.monotonic()-started
    I.save(out/'report.json',result)
    print('DSL GROUP DONE', group.name, result['passed'], result['head_counts'], result['covered_counts'], round(result['seconds'], 1), flush=True)
    return result


def run_group(group, config):
    """Global physical-head rewrites and all-pose parameter descent."""
    from step3_scheculer.shared_dsl import JointSearch, STAGE
    from step3_scheculer.construction_contract import certify
    started=time.monotonic()
    poses=['pose_'+v for v in group.name.removeprefix('pose').removesuffix('copied').split('+')] if '+' in group.name else [group.name]
    out=group/'step3_scheculer'/STAGE
    I.save(out/'report.json',dict(complete=False,schema=SCHEMA,passed=False,status='joint_search_running',poses=poses))
    tasks=[saved_task(group,p) for p in poses]
    search=JointSearch(tasks,group,config,Search,seed_contacts)
    force=search.force_stage()
    # Remove globally redundant/blocking heads before opening repair, so an
    # independently feasible union need not preserve a blocked escape cone.
    reduced=search.prune(force,require_exits=False) if search.classify(force)['passed'] else force
    repaired=search.repair(reduced)
    pruned=search.prune(repaired,require_exits=True) if search.feasible(repaired) else repaired
    attempts=[];seen=set()
    for p in (pruned,repaired,reduced,force):
        key=tuple((x.ident,x.center,x.radius) for x in p.patches)
        if key in seen:continue
        seen.add(key)
        proposal=search.export(p)
        result=certify(group,proposal,trials=config['construction_trials'])
        attempts.append(dict(physical_heads=len(p.patches),local_contact_passed=proposal['local_contact_passed'],
                             construction_passed=result['passed'],status=result['status']))
        if result['passed']:break
    I.save(out/'construction_attempts.json',dict(complete=True,attempts=attempts,
        recovery='only shared programs; no independent-head fallback'))
    result['provenance']['inputs'].update(I.hashes([out/'construction_attempts.json']))
    result['seconds']=time.monotonic()-started
    I.save(out/'report.json',result)
    I.save(out/'joint_progress.json',dict(complete=True,events=search.events))
    print('JOINT GROUP DONE',group.name,result['passed'],'physical',result['physical_head_count'],
          'shared',result['shared_head_count'],'covered',result['covered_counts'],round(result['seconds'],1),flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--groups', nargs='+')
    parser.add_argument('--include-independent', action='store_true')
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--gpu-iterations', type=int, default=600)
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--proposal-sample-cap', type=int, default=128)
    parser.add_argument('--seeds', type=int, default=32)
    parser.add_argument('--max-heads', type=int, default=8)
    parser.add_argument('--rewrite-trials', type=int, default=2)
    parser.add_argument('--force-rewrites', type=int, default=3)
    parser.add_argument('--exit-rewrites', type=int, default=3)
    parser.add_argument('--construction-trials',type=int,default=3)
    parser.add_argument('--exit-witnesses', type=int, default=8)
    args = parser.parse_args()
    config = {k:v for k,v in vars(args).items() if k not in ('groups', 'include_independent', 'jobs')}
    if any(v < 1 for k, v in config.items() if k != 'device'):
        raise ValueError('Search budgets must be positive')
    root = I.OUTPUTS/'B'
    groups = [root/g for g in args.groups] if args.groups else sorted(p for p in root.iterdir() if p.is_dir() and '+' in p.name)
    if args.include_independent:
        groups += sorted(p for p in (root/'independent_poses').glob('pose_*') if p.is_dir())
    if args.jobs < 1:
        raise ValueError('Positive worker count required')
    if args.jobs == 1:
        rows = [safe_run_group(group, config) for group in groups]
    else:
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = {pool.submit(safe_run_group, group, config):group for group in groups}
            collected = {futures[f]:f.result() for f in as_completed(futures)}
        rows = [collected[group] for group in groups]
    out = root/'pose2+9+13+15+17/step3_scheculer/dsl_shared'
    I.save(out/'batch_report.json', dict(complete=True, case_count=len(rows), passed_count=sum(r['passed'] for r in rows), results=rows, config=config))
    write_summary(out/'batch_report.md', rows)


def safe_run_group(group, config):
    try:
        return run_group(group, config)
    except Exception as error:
        row = dict(complete=True, group=group.name, passed=False, error=str(error), traceback=traceback.format_exc())
        I.save(group/'step3_scheculer/dsl_shared/report.json', row)
        print('DSL CASE ERROR', group.name, traceback.format_exc(), flush=True)
        return row


def write_summary(path, rows):
    lines = ['# Contact DSL descent: saved B cases', '',
        'Numerical real-mesh parameter descent and discrete rewrites. All 32768 original loads per task are checked. Head-count minimality and compactness are not proved.', '',
        '| Case | Passed | Seed → final heads | Per-pose coverage | Verified exit options | Seconds |',
        '|---|---|---|---|---|---|']
    for r in rows:
        if 'error' in r:
            lines.append(f"| {r['group']} | ERROR | — | {r['error']} | — | — |")
        else:
            lines.append(f"| {r['group']} | {r['passed']} | {r['seed_total_heads']} → {r['total_heads']} | {r['covered_counts']} | {r['exit_option_counts']} | {r['seconds']:.1f} |")
    path.write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
