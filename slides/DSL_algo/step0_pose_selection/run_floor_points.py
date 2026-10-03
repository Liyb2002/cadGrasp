"""Rebuild retained Step0 groups: original ground points and pairwise floor heights."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step3_scheculer.pair_tasks import read_task, input_hashes
from step3_scheculer.sample_acceptance import SAMPLE_COUNT
from step0_pose_selection import floor_points as C
from step0_pose_selection.draw_floor_points import draw
from step0_pose_selection.floor_points import pressure_centers

SCHEMA = 'step0_floor_points_v1'


def saved_poses(group):
    if not group.name.startswith('pose'):
        raise ValueError('Expected a pose-group directory')
    poses = ['pose_'+p for p in group.name[4:].split('+')]
    if len(set(poses)) != len(poses) or len(poses) < 2:
        raise ValueError('Expected distinct poses')
    for pose in poses:
        if not (group/'step_1_needs'/pose/'samples.json').is_file():
            raise FileNotFoundError(f'Missing saved Step1: {group.name}/{pose}')
    return poses


def compatibility_dirs(group, poses):
    """Only current consumers, never regenerate old searches or deleted groups."""
    active = group/'step3_scheculer/sequential_k_global'/('from_'+'_'.join(p.split('_')[1] for p in poses))
    if (active/'schedule.json').is_file():
        return [active.relative_to(group/'step3_scheculer')]
    old = group/'step3_scheculer/sequential_3plus2'
    return [p.parent.relative_to(group/'step3_scheculer') for p in sorted(old.glob('*/terminal_expansion/schedule.json'))]


def build(group, destination):
    """Build a complete replacement before deleting the previous Step0."""
    name, poses = group.parent.name, saved_poses(group)
    tasks = [read_task(name, p, folder=group/'step_1_needs'/p) for p in poses]
    return build_tasks(name, tasks, destination, compat=compatibility_dirs(group, poses))


def build_tasks(name, tasks, destination, *, compat=()):
    """Export a floor check directly from prepared loads; no Step1 stage required."""
    poses = [task.pose for task in tasks]
    clouds, originals = [], []
    for task in tasks:
        loads = task.targets/task.scale
        if len(loads) != SAMPLE_COUNT:
            raise ValueError('Step0 must use all original samples, without extra loads')
        xy, normal = pressure_centers(loads, task.domain.com)
        # Independent moment substitution, before coordinate transformation.
        world_moment = loads[:, 3:]+np.cross(task.domain.com, loads[:, :3])
        moment = np.cross(np.c_[xy, np.zeros(len(xy))], np.c_[np.zeros((len(xy), 2)), normal])
        np.testing.assert_allclose(moment[:, :2], world_moment[:, :2], atol=1e-12, rtol=1e-12)
        clouds.append(xy)
        originals.append(dict(load_wrenches=loads, floor_demands_xy_m=xy,
            total_floor_normal_mg=normal, original_pivot_m=task.floor, moment_origin_m=task.domain.com))
    frames = C.validate_frames([t.domain.data['frame']['T_world_mesh'] for t in tasks])
    vertices, faces = C.validate_task_geometry(tasks, frames)
    result = C.check_floor_points(clouds, frames, poses)
    data = destination/'data'; data.mkdir(parents=True)
    np.savez_compressed(data/'scene.npz', object_vertices_m=vertices, object_faces=faces,
                        T_world_mesh=frames)
    demand_records = []
    for k, (task, original) in enumerate(zip(tasks, originals)):
        # These five fields keep downstream Step4 inputs byte-compatible.
        path = data/f'floor_contact_{task.pose}.npz'
        np.savez_compressed(path, **original)
        checks = data/f'floor_check_{task.pose}.npz'
        np.savez_compressed(checks, heights_in_target_poses_m=result.heights[k],
                            no_floor_penetration=result.masks[k])
        for relative in compat:
            target = destination/relative/path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            I.save(target.with_suffix('.json'), dict(pose=task.pose, arrays=target.name,
                sha256=I.sha256(target), load_count=len(clouds[k]), pressure_formula_checked=True,
                actual_floor_contacts_designed=False,
                floor_check_report=str(Path(*(['..']*len(relative.parts)))/'report.json')))
        demand_records.append(dict(pose=task.pose, load_count=len(clouds[k]),
            arrays='data/'+path.name, sha256=I.sha256(path),
            checks='data/'+checks.name, pressure_formula_checked=True,
            actual_floor_contacts_designed=False))
    conflict = any(not row['passed'] for row in result.rows)
    report = dict(schema=SCHEMA, object=name, poses=poses, complete=True,
        status='infeasible_cross_pose_floor' if conflict else 'floor_point_check_passed',
        passed=not conflict, necessary_floor_condition_passed=not conflict,
        complete_fixture_verified=False, head_selection_used=False, extra_loads_added=False,
        sample_count_per_pose=SAMPLE_COUNT,
        load_input_folders={task.pose: str(Path(task.inputs[0]).parent.relative_to(I.ROOT))
                            for task in tasks},
        coordinate_frame='Each pose has its own world frame in metres; floor is world z=0',
        step0_1=dict(original_loads_preserved=True, per_pose=demand_records,
            point_definition='Required center of pressure for each original load, on its own z=0 floor',
            formula='M_world = M_origin + origin cross F; p = (-M_world_y/F_z, M_world_x/F_z, 0)',
            actual_feet_designed=False),
        step0_2=dict(per_pose=result.rows, tolerance_m=C.TOL,
            transform='p_target = T_world_mesh[target] @ inverse(T_world_mesh[source]) @ [x, y, 0, 1]',
            criterion='Check every original point in every other pose: target-world z >= -tolerance',
            height_columns=poses, lateral_bounds=None, legal_region_constructed=False,
            convex_hull_constructed=False,
            violating_count='Unique source samples below at least one target floor; not sum over pairs',
            scope='Fixed original task poses and object correspondence; massless fixture; unilateral planar ground normals. '
                  'Passing checks only sampled floor demands, not bodies, connections, thickness, withdrawal, friction or strength.'),
        compatibility_directories=[str(p) for p in compat],
        provenance=dict(inputs=input_hashes(tasks), code=I.hashes([
            Path(__file__), Path(C.__file__), Path(__file__).with_name('draw_floor_points.py'),
            Path(__file__).with_name('floor_points.py')])))
    report['presentation'] = draw(name, poses, vertices, faces, frames, clouds, result, destination)
    report['artifacts'] = {str(p.relative_to(destination)): I.sha256(p)
                           for p in sorted(destination.rglob('*')) if p.is_file()}
    I.save(destination/'report.json', report)
    return report


def run_group(group, replace=False):
    group = Path(group).resolve()
    output = group/'step0_pose_selection'
    if output.is_symlink():
        raise ValueError('Refusing to replace a symlink')
    if output.exists() and not replace:
        raise FileExistsError(f'Use --replace to rebuild {output}')
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='.step0_build_', dir=group) as tmp:
        staging = Path(tmp)/'step0_pose_selection'
        report = build(group, staging)
        # User-authorized replacement is restricted to this single stage.
        if output.exists(): shutil.rmtree(output)
        staging.rename(output)
    print(json.dumps(dict(group=group.name, status=report['status'],
        counts=[r['violating_sample_count'] for r in report['step0_2']['per_pose']],
        elapsed_seconds=round(time.monotonic()-started, 3)), ensure_ascii=False), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object', nargs='?', default='B')
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--existing-groups', action='store_true')
    selection.add_argument('--group', help='Existing group name, e.g. pose4+5+6+8')
    parser.add_argument('--replace', action='store_true', help='Delete and rebuild only the selected Step0 directories')
    args = parser.parse_args(argv)
    root = I.OUTPUTS/args.object
    if args.existing_groups:
        groups = [p.parent for p in sorted(root.glob('pose*+*/step0_pose_selection')) if p.is_dir()]
    else:
        if Path(args.group).name != args.group or not args.group.startswith('pose'):
            parser.error('Use an existing pose-group directory name')
        groups = [root/args.group]
    if not groups: parser.error('No retained Step0 groups')
    for group in groups:
        saved_poses(group)  # validate the entire batch before changing any stage
    records = [run_group(group, args.replace) for group in groups]
    summary = dict(schema=SCHEMA, object=args.object, complete=True,
        cases=[dict(group=g.name, status=r['status'], poses=r['poses'],
                    violating_counts=[p['violating_sample_count'] for p in r['step0_2']['per_pose']])
               for g, r in zip(groups, records)])
    I.save(groups[-1]/'step0_pose_selection/data/batch_summary.json', summary)
    print('STEP0 COMPLETE', len(groups), 'groups', flush=True)


if __name__ == '__main__': main()
