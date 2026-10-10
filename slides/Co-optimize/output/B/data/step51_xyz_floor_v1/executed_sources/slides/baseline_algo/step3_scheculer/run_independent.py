"""Solve poses independently, optionally requiring a group-wide surface/floor margin."""
import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import json
import multiprocessing
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step1.registry import task_poses
from step1.cases import selected_pose, normalize_pose
from step1.needs import build as build_loads
from step3_scheculer import contacts as I
from step3_scheculer.pair_tasks import read_task, task_folder, input_hashes
from step3_scheculer.joint_geometry import JointGeometry
from step3_scheculer.joint_prepared import geometry_sources, scoring_sources
from step3_scheculer.run_joint import JointSearch
from step3_scheculer.run_pairs import brief
from step3_scheculer.run_sequential import numerical_recovery, top5
from step3_scheculer.random_search import chain_seed
from step3_scheculer import floor_margin as F

SCHEMA = 'independent_single_pose_heads_v1'


def root(name):
    return I.OUTPUTS/name/'independent_poses'


def prepare_task(name, pose, output_root=None):
    target = (root(name) if output_root is None else Path(output_root))/pose/'step_1_needs'
    if not (target/'samples.json').exists():
        try:
            source = task_folder(name, pose)
        except FileNotFoundError:
            with selected_pose(pose):
                build_loads(name, output_folder=target)
        else:
            target.mkdir(parents=True, exist_ok=True)
            for filename in ('needs.json','samples.json','setup.npz','setup.json','floor_contact.npz','examples.json'):
                path=source/filename
                if path.is_file():shutil.copy2(path,target/filename)
    return read_task(name, pose, folder=target)


class IndependentSearch(JointSearch):
    def __init__(self, name, pose, seed=20260929, particles=10, count=200, *,
                 floor_poses=None, output_root=None, floor_clearance_m=F.DEFAULT_CLEARANCE_M):
        if particles < 1 or count < 1 or pose not in task_poses(name):
            raise ValueError('Known pose and positive search sizes required')
        self.name, self.poses = name, (pose,)
        self.seed, self.particles = seed, particles
        self.floor_poses = tuple(normalize_pose(p) for p in floor_poses) if floor_poses is not None else None
        if self.floor_poses is not None and (not self.floor_poses or pose not in self.floor_poses or
                len(set(self.floor_poses)) != len(self.floor_poses) or
                any(p not in task_poses(name) for p in self.floor_poses)):
            raise ValueError('Distinct known floor poses including this owner pose required')
        # Group-specific constraints must never overwrite/reuse the old unconstrained cache.
        if self.floor_poses is not None and output_root is None:
            raise ValueError('A separate output_root is required for group floor constraints')
        self.schema = SCHEMA if self.floor_poses is None else F.SCHEMA
        self.out = (root(name) if output_root is None else Path(output_root))/pose/'step3_scheculer'
        self.out.mkdir(parents=True, exist_ok=True)
        I.save(self.out/'schedule.json', dict(schema=self.schema, complete=False,
            object=name, poses=[pose], floor_poses=self.floor_poses, status='preparing'))
        problem = prepare_task(name, pose) if output_root is None else prepare_task(name, pose, output_root)
        self.problems = [problem]
        self.original_targets = [problem.targets.copy()]
        self.inputs = input_hashes(self.problems)
        self.geometry_folder = self.out.parent/'step2_local_support'
        # The force/path/withdrawal geometry has one task. The optional extra
        # context excludes only contact surfaces within the specified floor band.
        if self.floor_poses is None:
            self.geometry = JointGeometry([problem], count=count, global_head_exclusions=False)
        else:
            floor_problems = [problem if p == pose else prepare_task(name, p) for p in self.floor_poses]
            self.inputs.update(input_hashes(floor_problems))
            self.geometry = F.FloorMarginGeometry(problem, floor_problems, count=count,
                                                  clearance_m=floor_clearance_m)
        self.geometry.save(self.geometry_folder, I.save)
        self.inputs.update(I.hashes(list(self.geometry_folder.glob('candidates_*'))))
        self.scores, self.recoveries = OrderedDict(), {}

    def search_particle(self, particle):
        destination = self.out/f'particle_{particle:03d}'
        rng = np.random.default_rng(chain_seed(self.seed, particle))
        entries, rounds = [], []
        while len(entries) < 4:
            base = self.task_score(entries, 0)
            if len(entries) >= 3 and base.all():
                break
            rows = []
            for index, candidate in enumerate(self.geometry.pools[0]):
                row = dict(index=index, id=candidate['contact']['candidate_id'], eligible=False)
                if not candidate['valid']:
                    row['reason'] = candidate['reason']
                elif self.geometry.same_center(candidate, entries):
                    row['reason'] = 'center_already_selected'
                else:
                    check = self.geometry.task_check(entries+[candidate], 0)
                    row['reason'] = check['reason']
                    if check['passed']:
                        mask = self.task_score(entries+[candidate], 0, base)
                        if np.any(base & ~mask):
                            raise RuntimeError('Adding a head reduced fixed-pose coverage')
                        row.update(eligible=True, covered_count=int(mask.sum()))
                rows.append(row)
            ids, probabilities = top5(rows, int(base.sum()))
            draw = float(rng.random()) if ids else None
            number = len(entries)+1
            I.save(destination/f'round_{number:02d}_scores.json', dict(rows=rows,
                base_count=int(base.sum()), top5_indices=ids,
                probabilities=np.asarray(probabilities).tolist(), uniform_draw=draw))
            if not ids:
                break
            pick = min(int(np.searchsorted(np.cumsum(probabilities), draw, side='right')), len(ids)-1)
            entries.append(self.geometry.pools[0][ids[pick]])
            mask = self.task_score(entries, 0, base)
            row = dict(step=number, selected_id=entries[-1]['contact']['candidate_id'],
                active_ids=[e['contact']['candidate_id'] for e in entries],
                covered_count=int(mask.sum()), sample_count=len(mask),
                geometry=self.geometry.task_check(entries, 0))
            rounds.append(row)
            I.save(destination/f'round_{number:02d}.json', row)
            I.save(self.out/'progress.json', dict(complete=False, particle=particle, **row))
            print(self.poses[0], 'chain', particle, 'heads', len(entries), 'covered', int(mask.sum()), flush=True)
        score = self.evaluate(entries)
        geometry = self.geometry.group_check(entries)
        passed = len(entries) >= 3 and score['all_sampled_complete'] and geometry['passed']
        return entries, dict(particle=particle, seed=chain_seed(self.seed, particle),
            status='all_samples_and_local_geometry_passed' if passed else
                ('four_heads_incomplete' if len(entries) == 4 else 'no_eligible_local_head'),
            passed=bool(passed), heads=len(entries),
            selected_ids=[e['contact']['candidate_id'] for e in entries],
            active_ids_by_pose=[[e['contact']['candidate_id'] for e in entries]],
            rounds=rounds, geometry=geometry, **brief(score))

    def run(self):
        results, designs = [], []
        for particle in range(self.particles):
            entries, result = self.search_particle(particle)
            self.export_and_check(entries, result)
            results.append(result); designs.append(entries)
            if result['passed']:
                break
        winner = min(range(len(results)), key=lambda i:(not results[i]['passed'], -results[i]['mean_coverage'], i))
        selected = designs[winner]
        final = self.out/f'final_contacts_{self.poses[0]}.npz'
        I.save_contacts(final, [e['contact'] for e in selected])
        report = dict(schema=self.schema, complete=True, object=self.name, poses=list(self.poses),
            independent=True, shared_heads=False, cross_pose_constraints=self.floor_poses is not None,
            fixture_placement_solved=False, complete_fixture_verified=False,
            min_heads_per_pose=3, max_heads_per_pose=4, candidate_count_per_pose=self.geometry.count,
            search_seed=self.seed, particle_budget=self.particles, particles=len(results),
            winner=winner, result=results[winner], particle_results=results,
            heads=[dict(id=e['contact']['candidate_id'], owner_pose=self.poses[0],
                active_poses=list(self.poses), area_m2=e['area'], radius_m=e['contact']['radius_m'],
                normal_depth_m=self.geometry.local[0].depth) for e in selected],
            numerical_recovery=self.recoveries,
            provenance=dict(inputs=self.inputs, code=I.hashes([Path(__file__),
                Path(F.__file__),
                Path(__file__).with_name('run_joint.py'), Path(__file__).with_name('run_sequential.py')]
                +geometry_sources()+scoring_sources())), artifacts={final.name:I.sha256(final)})
        if self.floor_poses is not None:
            report.update(head_model=F.HEAD_MODEL, floor_poses=list(self.floor_poses),
                floor_clearance_m=self.geometry.floor_clearance_m,
                cross_pose_force_constraints=False, cross_pose_withdrawal_constraints=False,
                cross_pose_work_surface_constraints=False,
                local_geometry_probe='original finite-thickness head in its owner pose; conservative for the surface model',
                local_geometry_probe_depth_m=float(self.geometry.local[0].depth),
                all_pose_contact_floor_margin=self.geometry.floor_check(selected))
            for head in report['heads']:
                head.update(geometry_model=F.HEAD_MODEL,
                    local_geometry_probe_depth_m=head['normal_depth_m'], construction_thickness_m=0.)
        I.save(self.out/'schedule.json', report)
        I.save(self.out/'progress.json', dict(complete=True, passed=report['result']['passed']))
        if self.floor_poses is None:
            draw_pose(self.problems[0], [e['contact'] for e in selected], report, self.out.parent/'heads.png')
        return report


def draw_pose(problem, contacts, report, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    mesh = problem.domain.mesh
    fig = plt.figure(figsize=(7, 7), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(Poly3DCollection(mesh.triangles, facecolors='#b4bbc1', alpha=.18, edgecolors='none'))
    for i, c in enumerate(contacts):
        ax.add_collection3d(Poly3DCollection(c['triangles_m'], facecolors=plt.get_cmap('tab10')(i), edgecolors='none'))
        point = c['center_m']
        ax.text(*point, f' H{i+1}', fontsize=12)
    center = mesh.bounds.mean(axis=0); radius = mesh.extents.max()*.6
    ax.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius),
           zlim=(center[2]-radius, center[2]+radius))
    ax.set_box_aspect([1,1,1]); ax.set_axis_off(); ax.view_init(elev=22, azim=-55)
    result = report['result']
    fig.suptitle(f'{problem.pose} | {len(contacts)} independent heads\n'
                 f'{result["covered_counts"][0]:,} / 32,768 loads | Step3 {"PASS" if result["passed"] else "FAIL"}')
    fig.text(.5, .025, 'Own-pose contacts and withdrawal only. Connecting body is not constructed here.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0,.04,1,.92)); fig.savefig(path, dpi=140); plt.close(fig)


def solve_one(arguments):
    name, pose, seed, particles, count = arguments[:5]
    options = arguments[5] if len(arguments) > 5 else {}
    folder = Path(options.get('output_root') or root(name))/pose
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'pipeline.log').open('w', buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        began = time.monotonic()
        try:
            search = IndependentSearch(name, pose, seed, particles, count, **options)
            with numerical_recovery(search.out/'numerical_retries', search.recoveries):
                report = search.run()
            result = report['result']
            return dict(pose=pose, status=result['status'], passed=result['passed'], heads=result['heads'],
                covered_count=result['covered_counts'][0], particles=report['particles'],
                seconds=time.monotonic()-began, report=str((search.out/'schedule.json').relative_to(I.ROOT)))
        except Exception as error:
            traceback.print_exc()
            return dict(pose=pose, status='pipeline_error', passed=False,
                        error=f'{type(error).__name__}: {error}', seconds=time.monotonic()-began)


def run_batch(name, poses, seed=20260929, particles=10, count=200, jobs=2, *,
              floor_poses=None, output_root=None, floor_clearance_m=F.DEFAULT_CLEARANCE_M):
    if not poses or len(set(poses)) != len(poses) or min(particles, count, jobs) < 1:
        raise ValueError('Distinct poses and positive search sizes required')
    destination = root(name) if output_root is None else Path(output_root)
    if floor_poses is not None and output_root is None:
        raise ValueError('A separate output_root is required for group floor constraints')
    result = dict(schema=SCHEMA if floor_poses is None else F.SCHEMA, object=name, poses=list(poses), complete=False,
        independent=True, shared_heads=False, cross_pose_constraints=floor_poses is not None, results=[])
    options = dict(floor_poses=floor_poses, output_root=output_root, floor_clearance_m=floor_clearance_m)
    if floor_poses is not None:
        result.update(floor_poses=list(floor_poses), floor_clearance_m=floor_clearance_m, head_model=F.HEAD_MODEL)
    path = destination/'batch.json'
    I.save(path, result)
    with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(solve_one, (name, p, seed, particles, count, options)) for p in poses]
        for future in as_completed(futures):
            row = future.result(); result['results'].append(row); I.save(path, result)
            print('POSE COMPLETE', row, flush=True)
    result['results'].sort(key=lambda r: poses.index(r['pose']))
    result.update(complete=True, passed_count=sum(r['passed'] for r in result['results']))
    I.save(path, result)
    lines = ['# 独立 pose 选头结果', '',
        ('各 pose 独立受力和退出、不共享头；完整接触面在组内所有 pose 下至少离地 '
         f'{floor_clearance_m*1000:g} mm。' if floor_poses is not None else
         '各 pose 单独求解，不共享头、不使用其他 pose 的受力、工作面、地面或退出约束。'),
        '每个 pose 固定 32,768 个原始载荷、200 个候选、3–4 个头、最多十条搜索链。',
        '本表是 Step3 接触结果；支撑摆放、连接实体及完整验收属于 Step4。', '',
        '| Pose | Step3 | 头数 | 覆盖 | 搜索链 |', '|---|---|---:|---:|---:|']
    for row in result['results']:
        p = row['pose']
        lines.append(f'| {p} | {"通过" if row["passed"] else row["status"]} | {row.get("heads", "—")} | '
            f'{row.get("covered_count", "—")}/32768 | {row.get("particles", "—")} |')
    (destination/'report.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object', nargs='?', default='B')
    parser.add_argument('--poses', nargs='+')
    parser.add_argument('--seed', type=int, default=20260929)
    parser.add_argument('--particles', type=int, default=10)
    parser.add_argument('--candidates', type=int, default=200)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--floor-poses', nargs='+', help='Group poses for the full contact surface floor filter')
    parser.add_argument('--floor-clearance-mm', type=float, default=2.)
    parser.add_argument('--output-root', type=Path, help='Separate per-group independent result root')
    args = parser.parse_args()
    poses = [normalize_pose(p) for p in args.poses] if args.poses else task_poses(args.object)
    floor_poses = [normalize_pose(p) for p in args.floor_poses] if args.floor_poses else None
    report = run_batch(args.object, poses, args.seed, args.particles, args.candidates, args.jobs,
        floor_poses=floor_poses, output_root=args.output_root, floor_clearance_m=args.floor_clearance_mm/1000.)
    print('ALL POSES COMPLETE', report['passed_count'], '/', len(poses), flush=True)
    raise SystemExit(0 if all(r['status'] != 'pipeline_error' for r in report['results']) else 2)
