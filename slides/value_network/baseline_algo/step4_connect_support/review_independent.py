"""Replay saved independent-head shapes and reaction certificates without solving."""
import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step3_scheculer.pair_tasks import read_task


def review(output):
    report_path = output/'data/report.json'
    report = I.check_report(report_path)
    assert report['schema'] == 'independent_pose_connected_shape_v1'
    assert report['complete'] and report['constructed']
    body = trimesh.load(output/'shape.obj', force='mesh', process=False)
    assert body.is_watertight and body.is_winding_consistent and body.volume > 0
    # Check the exported indexed topology directly. Reconstructing a Manifold
    # can simplify near-collinear triangles and split numerical zero-volume
    # slivers; it does not preserve the exported graph being reviewed here.
    shells = body.split(only_watertight=False)
    assert len(shells) == 1
    work = output/'data/independent_body'
    certificate_path = work/('equilibrium.npz' if report['passed'] else 'equilibrium_diagnostic.npz')
    checks, identifiers = [], []
    with np.load(work/'geometry.npz') as geometry, np.load(certificate_path) as cert:
        np.testing.assert_allclose(body.vertices, geometry['vertices_m'], atol=1e-14, rtol=0)
        np.testing.assert_array_equal(body.faces, geometry['faces'])
        for k, pose in enumerate(report['poses']):
            folder = output.parent.parent/'independent_poses'/pose
            I.check_report(folder/'step3_scheculer/schedule.json')
            task = read_task(report['object'], pose, folder=folder/'step_1_needs')
            source = folder/'step3_scheculer'/f'final_contacts_{pose}.npz'
            copied = output/'data/independent_input'/f'contacts_{pose}.npz'
            assert I.sha256(source) == I.sha256(copied)
            contacts = I.read_contacts(source)
            identifiers.extend(c['candidate_id'] for c in contacts)
            basis, offset = geometry['rotations'][k], geometry['local_offsets_m'][k]
            np.testing.assert_allclose(basis@basis.T, np.eye(3), atol=1e-12)
            assert abs(np.linalg.det(basis)-1) < 1e-12
            world = (body.vertices-offset)@basis.T
            assert world[:, 2].min() >= -1e-9
            floor_rays = np.array([[64., 0, 1], [-64., 0, 1], [0, 64., 1], [0, -64., 1]])
            points, forces = [np.tile(task.floor, (4, 1))], [floor_rays]
            for contact in contacts:
                p = contact['triangles_m'].reshape(-1, 3)
                f = np.repeat(-task.domain.mesh.face_normals[contact['source_faces']], 3, axis=0)
                ids = np.sort(np.unique(np.c_[p, f], axis=0, return_index=True)[1])
                points.append(p[ids]); forces.append(f[ids])
            points, forces = np.concatenate(points), np.concatenate(forces)
            ground = cert[f'{pose}_ground_points_m']
            # Independently assemble object and support balance, including the
            # equal/opposite interface reaction and the original object pivot.
            matrix = np.zeros((12, len(points)+len(ground)))
            for j, (p, f) in enumerate(zip(points, forces)):
                wrench = np.r_[f, np.cross(p-task.domain.com, f)]
                matrix[:6, j] = wrench
                if j >= 4:
                    matrix[6:, j] = -wrench
            for j, p in enumerate(ground):
                f = floor_rays[j % 4]
                matrix[6:, len(points)+j] = np.r_[f, np.cross(p-task.domain.com, f)]
            matrix *= np.r_[task.scale, task.scale][:, None]
            error = float(np.max(np.abs(matrix-cert[f'{pose}_matrix'])))
            assert error < 1e-12
            floor_vertices = world[np.abs(world[:, 2]) < 1e-9]
            ground_error = float(cKDTree(floor_vertices).query(ground)[0].max())
            assert ground_error < 1e-9
            assignment, weights, bases = [cert[f'{pose}_{key}'] for key in ('assignment', 'weights', 'bases')]
            assert len(task.targets) == len(assignment) == 32768
            assert weights.shape == (32768, 12) and np.isfinite(weights).all() and weights.min() >= 0
            assert (assignment < len(bases)).all()
            residual = 0.
            for index, columns in enumerate(bases):
                rows = np.flatnonzero(assignment == index)
                if len(rows):
                    target = np.c_[task.targets[rows], np.zeros((len(rows), 6))]
                    residual = max(residual, float(np.max(np.abs(weights[rows]@matrix[:, columns].T-target))))
            assert residual < 2e-9
            complete = bool((assignment >= 0).all())
            saved_check = report['construction']['checks'][k]
            assert complete == saved_check['coupled_equilibrium_passed']
            if report['passed']:
                assert complete
            checks.append(dict(pose=pose, original_sample_count=len(task.targets),
                all_loads_certified=complete, maximum_matrix_error=error,
                maximum_certificate_residual=residual, maximum_ground_vertex_error_m=ground_error))
    assert len(identifiers) == len(set(identifiers)) == report['physical_head_count']
    result = dict(complete=True, replay_passed=True, fixture_acceptance_passed=report['passed'],
        one_closed_connected_solid=True, indexed_boundary_component_count=len(shells),
        unchanged_independent_heads=True, checks=checks,
        partial_assignments_are_not_coverage=True,
        scope='Saved indexed geometry and existing reaction certificates; no strength or global infeasibility claim',
        reviewed_report_sha256=I.sha256(report_path),
        provenance=dict(inputs=I.hashes([report_path, output/'shape.obj', work/'geometry.npz', certificate_path]),
                        code=I.hashes([Path(__file__)])))
    I.save(output/'data/independent_replay.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('outputs', type=Path, nargs='+')
    for folder in p.parse_args().outputs:
        result = review(folder)
        print(folder.parent.name, 'replay passed', 'acceptance', result['fixture_acceptance_passed'])
