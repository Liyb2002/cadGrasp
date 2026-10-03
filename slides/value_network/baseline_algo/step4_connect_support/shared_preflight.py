"""Step5 necessary checks for saved sequential 3+2, with fixed patch registration.

This is a rejection test and design diagnostic, not a fixture generator. It uses
only the original Step1 samples, and never changes Step3 contacts or acceptance.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step1.needs import OUTPUTS, ROOT
from step2_local_support import circles as P, geometry as G, withdrawal as W
from step3_scheculer import contacts as I
from step3_scheculer.pair_geometry import transform_contact
from step3_scheculer.pair_tasks import read_task
from step3_scheculer.sample_acceptance import SAMPLE_COUNT
from step0_pose_selection.floor_points import pressure_centers

HEIGHT_TOL_M = 1e-9
SCHEMA = 'shared_fixture_fixed_registration_preflight_v1'


def sibling(folder, stage):
    parts = list(Path(folder).resolve().relative_to(OUTPUTS).parts)
    if parts[2] != 'step3_scheculer':
        raise ValueError('Expected an existing per-case Step3 folder')
    if stage == 'step4_connect_support':
        return OUTPUTS.joinpath(*parts[:2], 'step4')
    parts[2] = stage
    return OUTPUTS.joinpath(*parts)


def ground_bound(loads, origin, pivot, transform, tolerance=HEIGHT_TOL_M):
    """Necessary h(CoP) >= min(0, h(object pivot)), h(q)=(T[q,1])_z.

Every support foot in this pose lies above the other pose's ground after T.
The original object pivot need not obey that support-material constraint.
Nonnegative ground normals make the whole-assembly CoP a convex combination
of feet and pivot. Passing says nothing about friction, yaw torque or routing.
"""
    transform = np.asarray(transform, float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError('Expected a finite rigid 4x4 transform')
    np.testing.assert_allclose(transform[3], [0, 0, 0, 1], atol=1e-12, rtol=0)
    np.testing.assert_allclose(transform[:3, :3].T@transform[:3, :3], np.eye(3), atol=1e-12, rtol=0)
    np.testing.assert_allclose(np.linalg.det(transform[:3, :3]), 1., atol=1e-12, rtol=0)
    pivot = np.asarray(pivot, float)
    if pivot.shape != (3,) or not np.isfinite(pivot).all() or abs(pivot[2]) > tolerance:
        raise ValueError('Expected the original object pivot on z=0')
    cloud, normal = pressure_centers(loads, origin)
    plane = transform[2]
    height = cloud@plane[:2]+plane[3]
    pivot_height = float(pivot@plane[:3]+plane[3])
    lower = min(0., pivot_height)
    slack = height-lower
    invalid = slack < -tolerance
    worst = int(np.argmin(slack))
    return dict(necessary_condition_passed=not bool(invalid.any()), sample_count=len(cloud),
        violating_sample_count=int(invalid.sum()), worst_sample_index=worst,
        other_ground_plane_in_this_pose=plane.tolist(), transform_to_other_pose=transform.tolist(),
        original_pivot_other_height_m=pivot_height, lower_bound_m=lower,
        minimum_slack_m=float(slack[worst]), minimum_total_normal_mg=float(normal.min()),
        worst_sample=dict(index=worst, load_wrench=np.asarray(loads)[worst].tolist(),
            pressure_center_xy_m=cloud[worst].tolist(), other_height_m=float(height[worst]),
            bound_m=lower, violation_m=max(0., -float(slack[worst]))),
        tolerance_m=tolerance), dict(pressure_centers_xy_m=cloud, other_height_m=height,
            slack_m=slack, violates=invalid)


def registration_rank(contact, mesh):
    """Local contact-preserving rigid twist rank; not a global registration search."""
    points = contact['triangles_m'].reshape(-1, 3)
    normals = np.repeat(mesh.face_normals[contact['source_faces']], 3, axis=0)
    arms = (points-contact['center_m'])/mesh.extents.max()
    jacobian = np.c_[normals, np.cross(arms, normals)]
    singular = np.linalg.svd(jacobian, compute_uv=False)
    rank = int(np.count_nonzero(singular > singular[0]*1e-10))
    return dict(contact_preserving_twist_rank=rank, local_twist_nullity=6-rank,
        singular_values=singular.tolist(), length_scale_m=float(mesh.extents.max()),
        source_face_count=len(np.unique(contact['source_faces'])),
        remote_congruent_registrations_searched=False)


def head_checks(folder, result, problems, frames, offsets, catalogues):
    contacts, sources, groups = {}, [], []
    for k, problem in enumerate(problems):
        path = folder/f'contacts_{problem.pose}.npz'
        sources.append(path)
        group = I.read_contacts(path)
        if [c['candidate_id'] for c in group] != result['active_ids_by_pose'][k]:
            raise ValueError('Particle active contact IDs changed')
        groups.append({c['candidate_id']: c for c in group})
        for contact in group:
            contacts.setdefault(contact['candidate_id'], (k, contact))
    shared = result['shared_head']['selected_id']
    if len(contacts) != 5 or set(groups[0]) & set(groups[1]) != {shared}:
        raise ValueError('Expected five heads with exactly one shared patch')
    transformed = transform_contact(groups[0][shared], frames[1]@np.linalg.inv(frames[0]))
    np.testing.assert_allclose(transformed['triangles_m'], groups[1][shared]['triangles_m'], atol=1e-12, rtol=0)
    np.testing.assert_array_equal(transformed['source_faces'], groups[1][shared]['source_faces'])
    rows, solids = [], []
    for k, problem in enumerate(problems):
        mesh = problem.domain.mesh
        heads, normals, material = [], [], []
        for ident, (owner, contact) in contacts.items():
            transform = frames[k]@np.linalg.inv(frames[owner])
            faces = np.unique(contact['source_faces'])
            cells = [G.head_cell(problems[owner].domain.mesh,
                contact['triangles_m'][contact['source_faces'] == f, 1], int(f), offsets[owner])
                @transform[:3, :3].T+transform[:3, 3] for f in faces]
            heads.extend(SimpleNamespace(vertices=cell) for cell in cells)
            normals.append(mesh.face_normals[faces])
            material.append(dict(id=ident, active=ident in groups[k], owner_pose=problems[owner].pose,
                solid_min_z_m=float(np.concatenate(cells)[:, 2].min()),
                work_overlap_face_ids=np.intersect1d(faces, problem.domain.work_ids).tolist()))
        vectors = np.asarray(catalogues[k]['vectors'])
        active = result['geometry']['per_pose'][k]['common_direction_ids']
        allowed = [j for j in active if np.min(np.concatenate(normals)@vectors[j]) >= -W.NORMAL_TOL]
        rows.append(dict(pose=problem.pose, heads=material,
            head_floor_passed=all(h['solid_min_z_m'] >= -mesh.extents.max()*1e-10 for h in material),
            work_surface_passed=all(not h['work_overlap_face_ids'] for h in material),
            active_triple_direction_ids=active, all_heads_normal_allowed_ids=allowed,
            sweep_checks=[], sweep_status='skipped_after_necessary_head_check_failure',
            finite_menu_only=True))
        solids.append(heads)
    # Retain useful head-only diagnostics even if the separate ground bound fails.
    if all(r['head_floor_passed'] and r['work_surface_passed'] and r['all_heads_normal_allowed_ids'] for r in rows):
        for k, row in enumerate(rows):
            mesh = problems[k].domain.mesh
            analyzer = W.Analyzer(mesh, P.DEPTH_FRACTION*mesh.extents.max(), catalogues[k])
            row['sweep_checks'] = [dict(direction_id=j, **analyzer.test(solids[k], catalogues[k]['vectors'][j]))
                                   for j in row['all_heads_normal_allowed_ids']]
            row['sweep_status'] = 'all_normal_allowed_directions_checked'
    for row in rows:
        row['head_only_clear_direction_ids'] = [r['direction_id'] for r in row['sweep_checks'] if r['clear']]
    return dict(particle=result['particle'], selected_ids=result['selected_ids'], shared_id=shared,
        shared_patch_registration=registration_rank(groups[0][shared], problems[0].domain.mesh),
        per_pose=rows, complete_fixture_verified=False), sources


def draw(folder, problems, checks, arrays):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.3), facecolor='white')
    for ax, problem, check in zip(axes, problems, checks):
        pose = problem.pose
        points = arrays[f'{pose}_pressure_centers_xy_m']*1000
        invalid = arrays[f'{pose}_violates']
        meshxy = problem.domain.mesh.vertices[:, :2]*1000
        bounds = np.vstack([points, meshxy, problem.floor[None, :2]*1000])
        low, high = bounds.min(axis=0), bounds.max(axis=0)
        low -= .08*(high-low); high += .08*(high-low)
        x, y = np.meshgrid(np.linspace(low[0], high[0], 140), np.linspace(low[1], high[1], 140))
        a, b, _, d = check['other_ground_plane_in_this_pose']
        height = a*x/1000+b*y/1000+d
        ax.contourf(x, y, (height >= 0).astype(float), levels=[-.5, .5, 1.5], colors=['#f8e8e7', '#e4f0e8'])
        if height.min() < 0 < height.max():
            ax.contour(x, y, height, levels=[0], colors='#667770', linewidths=1.2)
        ax.add_collection(PolyCollection(problem.domain.mesh.triangles[:, :, :2]*1000,
            facecolors='#c9cece', edgecolors='none', alpha=.25))
        for flag, color in [(False, '#276f89'), (True, '#c54545')]:
            ax.scatter(*points[invalid == flag].T, s=2.5, c=color, alpha=.35, rasterized=True)
        ax.scatter(*problem.floor[:2]*1000, marker='x', s=70, linewidths=2, c='#242b33', zorder=4)
        worst = points[check['worst_sample_index']]
        ax.scatter(*worst, facecolors='none', edgecolors='#20252d', s=110, linewidths=1.2, zorder=5)
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), xlabel='x (mm)', ylabel='y (mm)',
            title=f'{pose}: {check["violating_sample_count"]:,} / {SAMPLE_COUNT:,} outside bound\n'
                  f'Minimum transformed-height slack: {check["minimum_slack_m"]*1000:.3f} mm')
        ax.set_aspect('equal')
    fig.suptitle('Step5 / fixed shared-patch registration: ground compatibility', fontsize=16, y=.98)
    fig.legend(handles=[Patch(color='#e4f0e8', label='Possible fixture foot region (necessary only)'),
        Patch(color='#f8e8e7', label='A foot here penetrates the other pose floor'),
        Line2D([], [], marker='.', linestyle='', color='#c54545', label='Original sample violates CoP bound'),
        Line2D([], [], marker='x', linestyle='', color='#242b33', label='Original object pivot')],
        loc='lower center', ncol=2, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, .13, 1, .93))
    fig.savefig(folder/'proposal_ground_check.png', dpi=170)
    plt.close(fig)


def run(schedule):
    schedule = Path(schedule).resolve()
    report = I.check_report(schedule)
    if report['schema'] != 'sequential_three_heads_share_one_add_two_v1':
        raise ValueError('Expected sequential 3+2 results')
    source = schedule.parent
    out = sibling(source, 'step4_connect_support')
    out.mkdir(parents=True, exist_ok=True)
    I.save(out/'proposal.json', dict(schema=SCHEMA, complete=False, status='checking'))
    problems = [read_task(report['object'], pose, report['poses']) for pose in report['poses']]
    frames = [np.asarray(p.domain.data['frame']['T_world_mesh']) for p in problems]
    transform = frames[1]@np.linalg.inv(frames[0])
    np.testing.assert_array_equal(problems[0].domain.mesh.faces, problems[1].domain.mesh.faces)
    np.testing.assert_allclose(problems[0].domain.mesh.vertices@transform[:3, :3].T+transform[:3, 3],
                               problems[1].domain.mesh.vertices, atol=1e-12, rtol=0)
    checks, arrays, sources = [], {}, [schedule]
    for k, problem in enumerate(problems):
        sources.extend(problem.inputs)
        folder = sibling(source, 'step0_pose_selection')
        meta_path = folder/f'floor_contact_{problem.pose}.json'
        path = folder/f'floor_contact_{problem.pose}.npz'
        meta = json.loads(meta_path.read_text())
        if I.sha256(path) != meta['sha256']:
            raise ValueError('Step4 arrays changed')
        data = I.load_npz(path)
        np.testing.assert_allclose(data['load_wrenches'], problem.targets/problem.scale, atol=1e-13, rtol=0)
        np.testing.assert_allclose(data['moment_origin_m'], problem.domain.com, atol=1e-13, rtol=0)
        np.testing.assert_allclose(data['original_pivot_m'], problem.floor, atol=1e-13, rtol=0)
        if len(data['load_wrenches']) != SAMPLE_COUNT:
            raise ValueError('Expected only the original fixed samples')
        transform = frames[1-k]@np.linalg.inv(frames[k])
        check, values = ground_bound(data['load_wrenches'], problem.domain.com, problem.floor, transform)
        np.testing.assert_allclose(values['pressure_centers_xy_m'], data['floor_demands_xy_m'], atol=1e-13, rtol=0)
        np.testing.assert_allclose(data['total_floor_normal_mg'], data['load_wrenches'][:, 2], atol=1e-13, rtol=0)
        # Independent check in the world-origin moment equations.
        supplied = np.cross(np.c_[values['pressure_centers_xy_m'], np.zeros(SAMPLE_COUNT)],
                            np.c_[np.zeros((SAMPLE_COUNT, 2)), data['total_floor_normal_mg']])
        required = data['load_wrenches'][:, 3:]+np.cross(problem.domain.com, data['load_wrenches'][:, :3])
        np.testing.assert_allclose(supplied[:, :2], required[:, :2], atol=1e-12, rtol=1e-12)
        checks.append(dict(pose=problem.pose, other_pose=problems[1-k].pose, **check))
        arrays.update({f'{problem.pose}_{key}': value for key, value in values.items()})
        sources.extend([path, meta_path])
    catalogues = []
    for problem in problems:
        path = sibling(source, 'step2_local_support')/f'candidates_{problem.pose}.json'
        catalogues.append(json.loads(path.read_text())['direction_catalogue'])
        sources.append(path)
    offsets = [G.vertex_offsets(p.domain.mesh, P.DEPTH_FRACTION*p.domain.mesh.extents.max())[0] for p in problems]
    particles = []
    for result in report['particle_results']:
        if not result['passed']:
            continue
        folder = source/f'particle_{result["particle"]:03d}'
        mask_path = folder/'coverage.npz'
        masks = I.load_npz(mask_path)
        if any(masks[p.pose].shape != (SAMPLE_COUNT,) or not masks[p.pose].all() for p in problems):
            raise ValueError('Saved successful particle is not sample-complete')
        row, paths = head_checks(folder, result, problems, frames, offsets, catalogues)
        particles.append(row)
        sources.extend(paths+[mask_path])
    np.savez_compressed(out/'proposal_ground_check.npz', **arrays)
    draw(out, problems, checks, arrays)
    passed = all(c['necessary_condition_passed'] for c in checks)
    result = dict(schema=SCHEMA, complete=True, object=report['object'], poses=report['poses'],
        status='ground_bound_passed_further_checks_required' if passed else 'fixed_registration_ground_impossible',
        fixed_registration_ground_necessary_condition_passed=passed, original_contact_success_count=len(particles),
        complete_fixture_verified=False, shape_constructed=False,
        scope='Saved same-patch correspondence and object-induced fixture transforms only; arbitrary fixture registrations not ruled out.',
        assumptions=dict(support_mass_ignored=True, ground_normal_nonnegative=True,
            original_object_ground_contact_retained=True, original_sample_count_per_pose=SAMPLE_COUNT,
            additional_loads_added=False, continuous_load_validation_performed=False,
            friction_model_changed=False, object_task_poses_changed=False),
        ground_checks=checks, particle_head_checks=particles,
        proposed_structure='Open rear spine or U frame, short ribs, and pose-specific ground pads on rib backs; placement feasibility first.',
        proposed_relaxation='A shared physical branch with separate contact faces per pose is a NEW representation, not the saved shared-patch solution.',
        next_required_check='Search legitimate contact registrations/fixture placements, then actual feet, whole-body sweeps and coupled object/support equilibrium.',
        provenance=dict(inputs=I.hashes(sources), code=I.hashes([Path(__file__), Path(G.__file__), Path(W.__file__)])),
        artifacts={name: I.sha256(out/name) for name in ('proposal_ground_check.npz', 'proposal_ground_check.png')})
    I.save(out/'proposal.json', result)
    print(report['poses'], f'{len(particles)} contact successes', result['status'],
          [c['violating_sample_count'] for c in checks], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('schedules', nargs='+', type=Path, help='Existing sequential schedule.json files')
    args = parser.parse_args()
    results = [run(path) for path in args.schedules]
    print('Pairs:', len(results), 'contact sets:', sum(r['original_contact_success_count'] for r in results),
          'pairs rejected by ground bound:', sum(not r['fixed_registration_ground_necessary_condition_passed'] for r in results))


if __name__ == '__main__':
    main()
