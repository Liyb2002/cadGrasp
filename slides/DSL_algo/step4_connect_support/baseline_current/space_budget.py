"""Step4.1: workstation boxes of saved object poses and Step0 demand points.

No support construction, pose optimization or upstream acceptance changes.
The object-plus-demand box is a necessary spatial target, not a feasible solid.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'slides/baseline_algo/output'
TOL_M = 1e-9


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def points(value):
    value = np.asarray(value, float)
    if value.ndim != 2 or value.shape[1] != 3 or not len(value) or not np.isfinite(value).all():
        raise ValueError('Expected nonempty finite Nx3 points in metres')
    return value


def box(clouds):
    arrays = [points(c) for c in clouds]
    lo = np.min([c.min(axis=0) for c in arrays], axis=0)
    hi = np.max([c.max(axis=0) for c in arrays], axis=0)
    extent = hi-lo
    return dict(min_m=lo.tolist(), max_m=hi.tolist(), extents_mm=(extent*1000).tolist(),
                xy_area_cm2=float(extent[0]*extent[1]*10000), box_volume_cm3=float(np.prod(extent)*1e6))


def inclusion(cloud, bounds):
    cloud = points(cloud)
    lo, hi = np.asarray(bounds['min_m']), np.asarray(bounds['max_m'])
    below, above = cloud < lo-TOL_M, cloud > hi+TOL_M
    outside = np.any(below | above, axis=1)
    return dict(sample_count=len(cloud), inside_count=int((~outside).sum()),
        outside_count=int(outside.sum()), all_inside=bool(not outside.any()),
        below_axis_counts=below.sum(axis=0).tolist(), above_axis_counts=above.sum(axis=0).tolist(),
        required_lower_expansion_mm=(np.maximum(lo-cloud.min(axis=0), 0)*1000).tolist(),
        required_upper_expansion_mm=(np.maximum(cloud.max(axis=0)-hi, 0)*1000).tolist())


def analyze(objects, demands, poses, expansion_step_mm=2.):
    if not (len(objects) == len(demands) == len(poses)) or not poses or len(set(poses)) != len(poses):
        raise ValueError('Matching clouds and distinct pose IDs required')
    if not np.isfinite(expansion_step_mm) or expansion_step_mm <= 0:
        raise ValueError('Positive finite expansion step required')
    original = box(objects)
    target = box(list(objects)+list(demands))
    lo, hi = np.asarray(original['min_m']), np.asarray(original['max_m'])
    lower = lo-np.asarray(target['min_m'])
    upper = np.asarray(target['max_m'])-hi
    step = expansion_step_mm/1000
    count = int(np.ceil(max(lower.max(), upper.max())/step))
    budgets = []
    for index in range(count+1):
        # Only expand the sides that need it; never shrink or recenter a pose.
        bounds = box([np.vstack([lo-np.minimum(lower, index*step),
                                 hi+np.minimum(upper, index*step)])])
        checks = [dict(pose=p, **inclusion(d, bounds)) for p, d in zip(poses, demands)]
        budgets.append(dict(index=index, box=bounds, per_pose=checks,
                            all_demands_inside=all(r['all_inside'] for r in checks)))
    return dict(object_box=original, object_and_demands_box=target,
        per_pose=[dict(pose=p, **inclusion(d, original)) for p, d in zip(poses, demands)],
        lower_expansion_mm=(lower*1000).tolist(), upper_expansion_mm=(upper*1000).tolist(),
        extra_xy_area_cm2=target['xy_area_cm2']-original['xy_area_cm2'],
        expansion_step_mm=expansion_step_mm, expansion_budgets=budgets)


def load_group(group):
    report_path = group/'step0_pose_selection/report.json'
    report = json.loads(report_path.read_text())
    if not report['complete'] or report['object'] != group.parent.name:
        raise ValueError('Complete matching Step0 report required')
    # Old generator code hashes can differ; physical input bytes must match.
    inputs = report['provenance']['inputs']
    for relative, expected in inputs.items():
        if digest(ROOT/relative) != expected:
            raise ValueError(f'Stale Step0 input: {relative}')
    poses = report['poses']
    rows = {r['pose']: r for r in report['step0_1']['per_pose']}
    objects, demands, faces, sources = [], [], [], [report_path]
    for pose in poses:
        folder = ROOT/report['load_input_folders'][pose]
        needs_path = folder/'needs.json'
        needs = json.loads(needs_path.read_text())
        if needs['pose_id'] != pose or needs['object'] != group.parent.name:
            raise ValueError('Saved pose geometry does not match Step0')
        if needs['coordinate_system'] != 'z_up_xy_floor' or needs['units']['position'] != 'm':
            raise ValueError('Expected saved Z-up world coordinates in metres')
        floor_path = report_path.parent/rows[pose]['arrays']
        if digest(floor_path) != rows[pose]['sha256']:
            raise ValueError('Changed Step0 demand array')
        with np.load(floor_path) as data:
            xy = np.asarray(data['floor_demands_xy_m'], float)
        if xy.shape != (report['sample_count_per_pose'], 2):
            raise ValueError('Every original Step0 sample is required')
        objects.append(points(needs['geometry']['vertices_m']))
        faces.append(np.asarray(needs['geometry']['faces'], int))
        demands.append(points(np.c_[xy, np.zeros(len(xy))]))
        sources.extend([needs_path, floor_path])
    return poses, objects, faces, demands, sources


def draw(path, objects, faces, demands, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(13, 6), facecolor='white')
    target = result['object_and_demands_box']
    lo, hi = np.asarray(target['min_m'])*1000, np.asarray(target['max_m'])*1000
    center = (lo+hi)/2
    radius = max((hi-lo).max()*.6, 1.)
    for panel, bounds in enumerate((result['object_box'], target), 1):
        ax = fig.add_subplot(1, 2, panel, projection='3d')
        for vertices, triangles in zip(objects, faces):
            ax.add_collection3d(Poly3DCollection(vertices[triangles]*1000,
                facecolors='#619fce', edgecolors='none', alpha=.14))
        for cloud in demands:
            # Rendering subsampling only; all points enter numeric bounds/checks.
            ids = np.unique(np.r_[np.arange(0, len(cloud), max(1, len(cloud)//1000)),
                                  cloud.argmin(axis=0), cloud.argmax(axis=0)])
            subset = cloud[ids]
            outside = np.any((subset < np.asarray(result['object_box']['min_m'])-TOL_M) |
                             (subset > np.asarray(result['object_box']['max_m'])+TOL_M), axis=1)
            ax.scatter(*(subset*1000).T, s=2, c=np.where(outside, '#d15d44', '#4f9966'), alpha=.55)
        corners = np.array(list(itertools.product(*zip(bounds['min_m'], bounds['max_m']))))*1000
        for bit in (1, 2, 4):
            for i in range(8):
                j = i ^ bit
                if i < j:
                    ax.plot(*corners[[i, j]].T, color='#475668', linewidth=1.3)
        ax.set(xlim=(center[0]-radius, center[0]+radius),
               ylim=(center[1]-radius, center[1]+radius), zlim=(center[2]-radius, center[2]+radius))
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=24, azim=-55); ax.set_axis_off()
        extents = bounds['extents_mm']
        title = 'All object poses' if panel == 1 else 'All object poses + Step0 demands'
        ax.set_title(title+'\n'+ ' x '.join(f'{e:.1f}' for e in extents)+' mm')
    fig.text(.5, .035, 'Green: inside object box   Red: outside object box   Same camera and scale', ha='center')
    fig.tight_layout(rect=(0, .05, 1, 1)); fig.savefig(path, dpi=160); plt.close(fig)


def run(group, expansion_step_mm=2.):
    poses, objects, faces, demands, sources = load_group(group)
    before = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    result = analyze(objects, demands, poses, expansion_step_mm)
    result.update(complete=True, schema='step4_1_workstation_space_v1', object=group.parent.name,
        poses=poses, coordinate_frame='Saved task-world axes and origins; no recentering or alignment',
        definition='3D axis-aligned bounds of all object poses and their own Step0 demand points at z=0',
        optimization_metric='XY workstation area; no fixture aspect ratio or material objective',
        scope='Necessary target under the existing ground-hull coverage policy; not a feasible support certificate',
        physical_demand_points_are_material=False, cross_pose_demand_images_included=False,
        contacts_roots_thickness_connectivity_and_withdrawal_checked=False,
        support_constructed=False, upstream_acceptance_changed=False,
        provenance=dict(inputs=before, code={str(Path(__file__).relative_to(ROOT)): digest(__file__)}))
    out = group/'step4/data/space_budget'
    out.mkdir(parents=True, exist_ok=True)
    draw(out/'bbox.png', objects, faces, demands, result)
    result['artifacts'] = {'bbox.png': digest(out/'bbox.png')}
    (out/'report.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    if before != {str(p.relative_to(ROOT)): digest(p) for p in sources}:
        raise RuntimeError('Space analysis changed upstream inputs')
    print(json.dumps(dict(group=group.name, object_box_mm=result['object_box']['extents_mm'],
        target_box_mm=result['object_and_demands_box']['extents_mm'],
        outside_counts=[r['outside_count'] for r in result['per_pose']],
        extra_xy_area_cm2=result['extra_xy_area_cm2'], expansion_steps=len(result['expansion_budgets'])-1),
        ensure_ascii=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object', default='B')
    parser.add_argument('--groups', nargs='+')
    parser.add_argument('--expansion-step-mm', type=float, default=2.)
    args = parser.parse_args()
    base = OUTPUT/args.object
    groups = [base/g for g in args.groups] if args.groups else sorted(
        p.parent.parent for p in base.glob('pose*+*/step0_pose_selection/report.json'))
    if not groups:
        parser.error('No saved Step0 groups found')
    for group in groups:
        run(group, args.expansion_step_mm)
