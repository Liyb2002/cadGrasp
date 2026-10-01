"""Measure the extra workstation footprint of a saved multi-pose support.

Use two XY axis-aligned bounding boxes in the saved task-world coordinates:
one encloses every object pose, the other encloses every object + support pose.
No pose is independently recentered/rotated; no construction or force solve runs.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step5_evaluate import render_bbox as VIS


def points(value):
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or not len(array) or not np.isfinite(array).all():
        raise ValueError('Expected nonempty finite N x 3 vertex positions in metres')
    return array


def bbox(clouds):
    clouds = [points(p) for p in clouds]
    lo = np.min([p.min(axis=0) for p in clouds], axis=0)
    hi = np.max([p.max(axis=0) for p in clouds], axis=0)
    extent = hi - lo
    area = float(extent[0] * extent[1])
    if area <= 0:
        raise ValueError('The XY bounding box must have positive area')
    return dict(min_m=lo.tolist(), max_m=hi.tolist(), extents_mm=(extent * 1000).tolist(),
                xy_area_m2=area, xy_area_cm2=area * 10000)


def compare(object_box, supported_box):
    a = object_box['xy_area_m2']; b = supported_box['xy_area_m2']
    return dict(object_poses=object_box, object_and_support_poses=supported_box,
                extra_area_m2=b-a, extra_area_cm2=(b-a)*10000,
                area_multiplier=b/a, extra_area_ratio=(b-a)/a,
                extra_area_percent=100*(b-a)/a)


def measure(objects, fixture_vertices, bases, offsets):
    objects = [points(p) for p in objects]
    fixture_vertices = points(fixture_vertices)
    count = len(objects)
    bases = np.asarray(bases, dtype=float); offsets = np.asarray(offsets, dtype=float)
    if count == 0 or bases.shape != (count, 3, 3) or offsets.shape != (count, 3):
        raise ValueError('Each object pose needs exactly one basis and offset')
    if not np.isfinite(bases).all() or not np.isfinite(offsets).all():
        raise ValueError('Nonfinite placement transform')
    if not np.allclose(bases @ bases.transpose(0, 2, 1), np.eye(3), atol=1e-10, rtol=0):
        raise ValueError('Placement bases must be orthonormal')
    if not np.allclose(np.linalg.det(bases), 1., atol=1e-10, rtol=0):
        raise ValueError('Placement bases must be proper rotations')
    # Step4 stores fixture = task_world @ basis + offset (row vectors).
    supports = [(fixture_vertices-o) @ b.T for b, o in zip(bases, offsets)]
    aggregate = compare(bbox(objects), bbox(objects + supports))
    per_pose = [compare(bbox([obj]), bbox([obj, support]))
                for obj, support in zip(objects, supports)]
    return aggregate, per_pose, supports


def read_needs(group, pose, report):
    """Prefer imported snapshots; never look up current global pose IDs."""
    snapshot = group/'step4/data/source_inputs'/pose/'needs.json'
    independent = group/'step3_scheculer/independent_poses_floor2mm'/pose/'step_1_needs/needs.json'
    candidates = [snapshot, independent, group/'step_1_needs'/pose/'needs.json']
    records = {}
    for key in ('provenance', 'generation_provenance', 'geometry_generation_provenance'):
        for name, digest in report.get(key, {}).get('inputs', {}).items():
            if Path(name).name == 'needs.json' and pose in Path(name).parts:
                records[name] = digest
                candidates.append(I.ROOT/name)
    available = [p for p in candidates if p.is_file()]
    if not available:
        raise FileNotFoundError(f'No saved Step1 geometry for {group.name}/{pose}')
    path = available[0]
    if records and I.sha256(path) not in records.values():
        raise ValueError(f'Saved geometry differs from the Step4 inputs: {path}')
    data = json.loads(path.read_text())
    if data.get('pose_id') != pose or data.get('object') != group.parent.name:
        raise ValueError(f'Wrong object or pose in {path}')
    if data.get('coordinate_system') != 'z_up_xy_floor' or data['units']['position'] != 'm':
        raise ValueError(f'Expected saved Z-up task-world geometry in metres: {path}')
    mesh = trimesh.Trimesh(points(data['geometry']['vertices_m']), data['geometry']['faces'], process=False)
    return path, mesh


def evaluate(group):
    group = Path(group).resolve()
    source = group/'step4/data/report.json'
    report = I.check_report(source)
    if not report.get('constructed'):
        raise ValueError(f'{group.name} has no constructed Step4 support')
    shape_path = group/'step4/shape.obj'
    if report.get('artifacts', {}).get('../shape.obj') != I.sha256(shape_path):
        raise ValueError('Public support differs from the recorded Step4 shape')
    poses = report['poses']
    if not poses or len(set(poses)) != len(poses):
        raise ValueError('Expected a nonempty list of distinct poses')
    loaded = [read_needs(group, pose, report) for pose in poses]
    objects = [mesh for _, mesh in loaded]
    fixture = trimesh.load(shape_path, force='mesh', process=False)
    inputs = [source, shape_path] + [p for p, _ in loaded]
    preserved = [source, shape_path] + list((group/'step4').glob('*.png')) + [p for p, _ in loaded]
    before = I.hashes(preserved)
    aggregate, per_pose, supports = measure([m.vertices for m in objects], fixture.vertices,
                                           report['placement']['bases'], report['placement']['offsets'])
    out = group/'step5_evaluate';out.mkdir(parents=True, exist_ok=True)
    visualization = VIS.draw(out/'bbox.png', objects, supports, fixture.faces, aggregate)
    result = dict(complete=True, schema='multipose_bbox_footprint_v1', object=group.parent.name,
                  poses=poses, pose_count=len(poses), metrics=aggregate,
                  per_pose=[dict(pose=p, **row) for p, row in zip(poses, per_pose)],
                  definition=dict(metric='XY area of axis-aligned bounding boxes',
                      aggregation='One box around the union of all saved poses; not sum/max of pose areas',
                      baseline='Object only in all saved task-world poses',
                      supported='Object and support together in all saved task-world poses',
                      frame='Saved Z-up task-world workstation XY axes and origins',
                      placement_optimization=False, recentered=False, sweeps_included=False,
                      support_to_world='(fixture_vertices - offset[i]) @ basis[i].T',
                      dimension_units='mm', area_units='cm^2', lower_extra_area_is_better=True),
                  source_step4_status=dict(constructed=report['constructed'], passed=report.get('passed'),
                      status=report.get('status'), strict_shared_head_registration_passed=report.get('strict_shared_head_registration_passed')),
                  visualization=visualization,
                  geometry_changed=False, force_or_geometry_acceptance_rerun=False, size_limit_enforced=False,
                  provenance=dict(inputs=I.hashes(inputs), code=I.hashes([Path(__file__)]+VIS.SOURCES)),
                  artifacts={'bbox.png': I.sha256(out/'bbox.png')})
    a = aggregate['object_poses']; b = aggregate['object_and_support_poses']
    (out/'README.md').write_text(
        f'# {group.name}: Step5 占地评价\n\n'
        '将所有静态工作姿态放在已保存的工位坐标系中，分别计算一个总 XY 轴对齐 bounding box。'
        '图中用完整三维盒展示 XYZ 范围，评价数值仍为 XY 占地面积。'
        '物体与支撑保持 Step4 的实际相对摆放；不独立旋转/平移以缩小包围盒，不计退出或换姿态运动。\n\n'
        '| 情况 | 包围盒宽 × 深（mm） | 面积（cm²） |\n|---|---:|---:|\n'
        f'| 只有物体的所有 pose | {a["extents_mm"][0]:.3f} × {a["extents_mm"][1]:.3f} | {a["xy_area_cm2"]:.3f} |\n'
        f'| 物体 + 支撑的所有 pose | {b["extents_mm"][0]:.3f} × {b["extents_mm"][1]:.3f} | {b["xy_area_cm2"]:.3f} |\n\n'
        f'额外占地 **{aggregate["extra_area_cm2"]:.3f} cm²（+{aggregate["extra_area_percent"]:.2f}%）**；'
        f'面积倍率 {aggregate["area_multiplier"]:.4f}。这是占地指标，不是尺寸通过/失败门槛。\n\n'
        '[同尺度三维包围盒对比](bbox.png) · [完整数值与来源](report.json)\n\n'
        '逐 pose 数值作为诊断保存在 report.json；主指标是整组的两个总包围盒，既不求面积之和，也不取单 pose 最大面积。'
        '沿用输入结果的原始验收状态，本阶段不重新认证支撑。\n')
    result['artifacts']['README.md'] = I.sha256(out/'README.md')
    I.save(out/'report.json', result)
    assert before == I.hashes(preserved), 'Evaluation modified upstream results'
    I.check_report(out/'report.json')
    print(json.dumps(dict(group=group.name, object_bbox_mm=a['extents_mm'][:2],
                          supported_bbox_mm=b['extents_mm'][:2], object_area_cm2=a['xy_area_cm2'],
                          supported_area_cm2=b['xy_area_cm2'], extra_area_percent=aggregate['extra_area_percent'],
                          output=str(out)), ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object', default='B')
    parser.add_argument('--groups', nargs='+', default=['pose1+3'])
    args = parser.parse_args()
    for name in args.groups:
        evaluate(I.OUTPUTS/args.object/name)
