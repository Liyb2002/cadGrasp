"""Build a reversible body with integral feet from two saved three-contact sets.

The shared branch has TWO contact faces, one for each task. This is explicitly
a geometry proposal with independent fixture placements, not the old fixed
registration of one shared patch. Original task poses and loads stay unchanged.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import trimesh
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step1.needs import ROOT
from step2_local_support import geometry as G, circles as P, withdrawal as W
from step3_scheculer import contacts as I
from step3_scheculer.pair_tasks import read_task, pair_folder
from step4_floor_contact import equilibrium as Q
from step5_base.bearing import bearing_rays, grounded_matrix
from step5_connect_support import solids as SOL, branch_bodies as BODY
from step5_connect_support.surface_check import surface_distances

PALETTE = ['#ac7098', '#7196c0', '#dc9d47', '#50a59b', '#77a76a']


def cells_for(contact, domain, offsets):
    return [G.head_cell(domain.mesh, contact['triangles_m'][contact['source_faces'] == f, 1], int(f), offsets)
            for f in np.unique(contact['source_faces'])]


def frame(direction, inverted=False):
    x = np.asarray(direction, float)
    z = np.array([0., 0., -1. if inverted else 1.])
    return np.column_stack([x, np.cross(z, x), z])


def local_to_world(points, basis, offset):
    return (points-offset)@basis.T


def placement_search(problems, groups, bases, head_world):
    offsets = [np.r_[-(p.domain.mesh.vertices@b).mean(axis=0)[:2], 0.] for p, b in zip(problems, bases)]
    objects = [trimesh.Trimesh(p.domain.mesh.vertices@b+o, p.domain.mesh.faces, process=False)
               for p, b, o in zip(problems, bases, offsets)]
    heads = [[v@b+o for group in cells for v in group] for cells, b, o in zip(head_world, bases, offsets)]
    points = [np.unique(np.concatenate(cells), axis=0) for cells in heads]
    height_min = max(points[0][:, 2].max(), -points[1][:, 2].min())+.014
    height_max = objects[0].bounds[1, 2]-objects[1].bounds[0, 2]+.025
    trials = []
    # Opposite landing faces leave the entire slab available to both tasks.
    for height in np.arange(height_min, height_max+.021, .02):
        for sx, sy in [(0., 0.), (0., .04), (0., -.04), (.04, 0.), (-.04, 0.),
                       (.04, .04), (.04, -.04), (-.04, .04), (-.04, -.04)]:
            shift = np.array([sx, sy, height])
            blocked = objects[1].ray.intersects_any(points[0]-shift+[1e-7, 0, 0], np.tile([1., 0, 0], (len(points[0]), 1))).any()
            if not blocked:
                blocked = objects[0].ray.intersects_any(points[1]+shift+[1e-7, 0, 0], np.tile([1., 0, 0], (len(points[1]), 1))).any()
            row = dict(height_m=float(height), shift_xy_m=[sx, sy], vertex_ray_screen_passed=not bool(blocked))
            trials.append(row)
            if blocked:
                continue
            proposed = [offsets[0], offsets[1]+shift]
            all_local = heads[0]+[v+shift for v in heads[1]]
            checks = []
            for k, p in enumerate(problems):
                pieces = [SimpleNamespace(vertices=local_to_world(v, bases[k], proposed[k])) for v in all_local]
                analyzer = W.Analyzer(p.domain.mesh, P.DEPTH_FRACTION*p.domain.mesh.extents.max(),
                                      dict(vectors=[bases[k][:, 0].tolist()]))
                checks.append(analyzer.test(pieces, bases[k][:, 0]))
            row['all_head_sweeps'] = checks
            if all(c['clear'] for c in checks):
                print('Placement found:', row, flush=True)
                return proposed, height, trials
    raise RuntimeError('No compatible reversible placement in the finite search')


def verify(problems, groups, bases, offsets, mesh, parts):
    checks, certificates = [], {}
    for k, (p, contacts, basis, offset) in enumerate(zip(problems, groups, bases, offsets)):
        world = local_to_world(mesh.vertices, basis, offset)
        pieces = [SimpleNamespace(vertices=local_to_world(part.vertices, basis, offset)) for part in parts]
        analyzer = W.Analyzer(p.domain.mesh, P.DEPTH_FRACTION*p.domain.mesh.extents.max(), dict(vectors=[basis[:, 0].tolist()]))
        sweep = analyzer.test(pieces, basis[:, 0])
        vertices = world[np.abs(world[:, 2]) < 1e-9]
        if len(vertices) < 3:
            raise RuntimeError('No actual ground contact surface')
        xy = np.unique(vertices[:, :2], axis=0)
        footprint = dict(pads_xy_m=[xy[ConvexHull(xy).vertices]])
        points, normals, owners = bearing_rays(p.domain, contacts, p.floor, 64.)
        matrix, ground = grounded_matrix(points, normals, owners, p.domain.com, p.scale, [footprint], 64.)
        solver = Q.BatchSolver(matrix)
        result = solver.solve(Q.padded_targets(p.targets/p.scale, p.scale, 12))
        fitted = trimesh.Trimesh(world, mesh.faces, process=False)
        contact_points = np.unique(np.concatenate([c['triangles_m'].reshape(-1, 3) for c in contacts]), axis=0)
        gaps = surface_distances(fitted, contact_points)
        targets = Q.padded_targets(p.targets/p.scale, p.scale, 12)
        residual = 0.
        for j, columns in enumerate(solver.bases):
            rows = np.flatnonzero(result['assignment'] == j)
            if len(rows):
                residual = max(residual, float(np.max(np.abs(result['weights'][rows]@matrix[:, columns].T-targets[rows]))))
        check = dict(pose=p.pose, withdrawal=sweep, min_fixture_z_m=float(world[:, 2].min()),
            max_active_contact_gap_m=float(gaps.max()), actual_ground_hull_xy_m=footprint['pads_xy_m'][0].tolist(),
            original_sample_count=len(p.targets), coupled_equilibrium_passed=result['passed'],
            verified_sample_count=int((result['assignment'] >= 0).sum()), diagnostics=result['diagnostics'],
            friction_coefficient=64., lp_count=result['lp_count'], maximum_equilibrium_residual=residual,
            minimum_reaction_coefficient=float(result['weights'].min()),
            contact_distance_method='independent long-double face-plane and edge projections')
        checks.append(check)
        for key, value in dict(matrix=matrix, assignment=result['assignment'], weights=result['weights'],
                               bases=np.asarray(solver.bases), ground_points_m=ground['points_m']).items():
            certificates[f'{p.pose}_{key}'] = value
        print('Geometry / bearing:', {key: value for key, value in check.items() if key != 'actual_ground_hull_xy_m'}, flush=True)
    return checks, certificates


def pack(mesh):
    return dict(vertices=mesh.vertices.ravel().tolist(), faces=mesh.faces.ravel().tolist())


def export_viewer(out, mesh, visual, report, problems, bases, offsets, colors):
    from step5_connect_support.refresh_shared_geometry_view import head_regions, write_viewer
    data = dict(fixture=pack(mesh), parts=[dict(id=label, color=colors.get(label, '#98b2c0'), **pack(m)) for label, m in visual],
        poses=[dict(name=p.pose, object=pack(p.domain.mesh), work_faces=p.domain.work_ids.tolist(),
                    rotation=b.tolist(), translation=(-b@o).tolist()) for p, b, o in zip(problems, bases, offsets)],
        dimensions_mm=(mesh.extents*1000).tolist(), report=report)
    data['head_regions'] = head_regions(report, bases, offsets)
    write_viewer(out, data)


def run(name, poses, particle, reuse_foot_fit=False, fitted_feet=None):
    source = pair_folder(name, poses, 'step3_scheculer')/'sequential_3plus2'/f'from_{poses[0]}'/'terminal_expansion'
    saved = json.loads((source/f'particle_{particle:03d}/schedule.json').read_text())
    if not saved['passed']:
        raise ValueError('Start from a saved successful contact set')
    out = source.parents[3]/'step5'
    out.mkdir(parents=True, exist_ok=True)
    problems = [read_task(name, p, poses) for p in poses]
    groups = [I.read_contacts(source/f'particle_{particle:03d}/contacts_{p}.npz') for p in poses]
    cats = [json.loads((pair_folder(name, poses, 'step2_local_support')/'sequential_3plus2'/f'from_{poses[0]}'/'terminal_expansion'/f'candidates_{p}.json').read_text())['direction_catalogue'] for p in poses]
    ids = [saved['geometry']['per_pose'][k]['common_direction_ids'][-1] for k in range(2)]
    # Use the previously full-head checked directions when this example is selected.
    if poses == ['pose_1', 'pose_3'] and particle == 9:
        ids = [61, 65]
    bases = [frame(np.asarray(c['vectors'][j]), k == 1) for k, (c, j) in enumerate(zip(cats, ids))]
    offsets = [G.vertex_offsets(p.domain.mesh, P.DEPTH_FRACTION*p.domain.mesh.extents.max())[0] for p in problems]
    owned = {}
    frames = [np.asarray(p.domain.data['frame']['T_world_mesh']) for p in problems]
    head_world = []
    for k, contacts in enumerate(groups):
        cells = []
        for contact in contacts:
            ident = contact['candidate_id']
            if ident not in owned:
                owned[ident] = (k, cells_for(contact, problems[k].domain, offsets[k]))
            owner, parts = owned[ident]
            transform = frames[k]@np.linalg.inv(frames[owner])
            cells.append([v@transform[:3, :3].T+transform[:3, 3] for v in parts])
        head_world.append(cells)
    placements, height, trials = placement_search(problems, groups, bases, head_world)
    foot_parameters = fitted_feet
    if reuse_foot_fit:
        previous = json.loads((out/'report.json').read_text())
        if previous['source_schedule'] != str((source/f'particle_{particle:03d}/schedule.json').relative_to(ROOT)):
            raise ValueError('Saved sole fit belongs to another contact set')
        I.check_hashes(previous['provenance']['inputs'])
        for old, b, o in zip(previous['task_fixture_transforms'], bases, placements):
            np.testing.assert_allclose(old['rotation'], b, atol=1e-12, rtol=0)
            np.testing.assert_allclose(old['translation_m'], -b@o, atol=1e-12, rtol=0)
        foot_parameters = previous['body_design']['foot_parameters_m']
    mesh, parts, labels, visual, solid_check, body_design = BODY.build(
        problems, groups, head_world, bases, placements, height, foot_parameters)
    checks, certificate = verify(problems, groups, bases, placements, mesh, parts)
    mesh.export(out/'fixture.obj', file_type='obj', digits=17, include_normals=False)
    millimeters = mesh.copy(); millimeters.apply_scale(1000)
    # ASCII retains the tiny Boolean edges that binary STL float32 collapses.
    (out/'fixture_mm.stl').write_text(trimesh.exchange.stl.export_stl_ascii(millimeters))
    reread = trimesh.load(out/'fixture_mm.stl', force='mesh')
    if not reread.is_watertight or not reread.is_winding_consistent:
        raise RuntimeError('Exported STL did not retain the closed solid')
    np.testing.assert_allclose(reread.extents, millimeters.extents, atol=1e-10, rtol=0)
    np.savez_compressed(out/'geometry.npz', vertices_m=mesh.vertices, faces=mesh.faces,
                        rotations=np.asarray(bases), local_offsets_m=np.asarray(placements))
    np.savez_compressed(out/'equilibrium.npz', **certificate)
    passed = all(c['coupled_equilibrium_passed'] and c['withdrawal']['clear'] and c['min_fixture_z_m'] >= -1e-9
                 and c['max_active_contact_gap_m'] < 1e-8 for c in checks)
    report = dict(object=name, poses=poses, particle=particle, complete=True,
        status='geometry_and_fixed_sample_equilibrium_passed' if passed else 'geometry_proposal_with_failed_checks',
        same_rigid_solid_in_both_poses=True, physical_branch_groups=5, contact_patch_count=6,
        shared_branch_definition='one connected branch with two distinct contact surfaces; not the old single shared patch',
        original_task_poses_changed=False, original_active_contacts_preserved=True,
        support_mass_ignored=True, extra_loads_added=False, structural_strength_verified=False,
        dimensions_mm=(mesh.extents*1000).tolist(), volume_cm3=float(mesh.volume*1e6),
        solid=solid_check, checks=checks, placement_trials=trials, body_design=body_design,
        previous_frame_comparison=(dict(dimensions_mm=[240.21212579016876, 218.7530060096461, 144.8362226776343],
            volume_cm3=296.3889156094627, method='previous geometry report, same contacts and fixture placements')
            if name == 'B' and poses == ['pose_1', 'pose_3'] and particle == 9 else None),
        task_fixture_transforms=[dict(pose=p.pose, rotation=b.tolist(), translation_m=(-b@o).tolist()) for p, b, o in zip(problems, bases, placements)],
        source_schedule=str((source/f'particle_{particle:03d}/schedule.json').relative_to(ROOT)),
        exports=dict(obj_units='m', obj_preserve_original_vertex_indices=True,
                     stl_units='mm', stl_encoding='ASCII', reloaded_stl_watertight=True),
        artifacts={file: I.sha256(out/file) for file in ('fixture.obj', 'fixture_mm.stl', 'geometry.npz', 'equilibrium.npz')},
        provenance=dict(inputs=I.hashes([source/f'particle_{particle:03d}/contacts_{p}.npz' for p in poses]
            +[source/f'particle_{particle:03d}/schedule.json']+[path for p in problems for path in p.inputs]),
            code=I.hashes([Path(__file__), Path(__file__).with_name('shared_geometry_viewer.html'),
                          Path(__file__).with_name('surface_check.py'), Path(BODY.__file__), Path(Q.__file__), Path(G.__file__), Path(W.__file__), Path(SOL.__file__)])))
    I.save(out/'report.json', report)
    shared_id = saved['shared_head']['selected_id']
    colors = {shared_id: '#dc9d47'}
    colors.update(zip([i for i in saved['selected_ids'] if i != shared_id], ['#ac7098', '#7196c0', '#50a59b', '#77a76a']))
    export_viewer(out, mesh, visual, report, problems, bases, placements, colors)
    print('EXPORTED', out, report['status'], report['dimensions_mm'], flush=True)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object', nargs='?', default='B')
    parser.add_argument('--poses', nargs=2, default=['pose_1', 'pose_3'])
    parser.add_argument('--particle', type=int, default=9)
    parser.add_argument('--reuse-foot-fit', action='store_true', help='Replay saved sole dimensions after input/placement checks; rerun whole-body validation')
    args = parser.parse_args()
    run(args.object, args.poses, args.particle, args.reuse_foot_fit)
