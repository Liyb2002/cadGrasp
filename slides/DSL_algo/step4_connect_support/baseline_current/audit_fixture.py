"""Read-only independent equilibrium replay of the saved rigid-fixture result.

Does not invoke the generator, grounded_matrix, bearing_rays, or BatchSolver.
Rebuilds both physical free-body equations from original contact triangles,
original floor pivot, mu=64 floor rays, and actual saved ground vertices; then
substitutes every saved sample-specific nonnegative reaction certificate.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'slides/baseline_algo'))
from step3_scheculer.pair_tasks import read_task
from step3_scheculer.contacts import read_contacts


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(out):
    report_path = out/'report.json'
    report = json.loads(report_path.read_text())
    if not report.get('passed') or report.get('diagnostic_only'):
        raise ValueError('No accepted complete fixture to replay; inspect the failure report instead')
    model = report.get('physical_head_definition')
    if model not in ('five_unique_heads_one_shared_patch', 'six_retained_contact_patches'):
        raise ValueError('Declare the physical contact model before reaction replay')
    body_ids = [b['candidate_id'] for b in report['body_design']['local_bodies']]
    expected = 5 if model == 'five_unique_heads_one_shared_patch' else 6
    if len(body_ids) != expected or len(set(body_ids)) != 5:
        raise ValueError('Physical contact patches disagree with the declared model')
    for category in ('inputs', 'code'):
        for path, digest in report['provenance'][category].items():
            assert sha256(ROOT/path) == digest, (category, path)
    for path, digest in report['artifacts'].items():
        assert sha256(out/path) == digest, path

    geometry = np.load(out/'geometry.npz')
    vertices = geometry['vertices_m']
    mesh = trimesh.Trimesh(vertices, geometry['faces'], process=False)
    components = len(mesh.split())
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert components == 1 and mesh.volume > 0
    checks = []
    certificates = np.load(out/'equilibrium.npz')
    poses = report['poses']
    source = ROOT/report['source_schedule']
    # Independent shared-surface check before replaying reactions. Per-task
    # equilibrium alone cannot prove that the same physical head was retained.
    shared_points = {}
    shared_errors = []
    for k, pose in enumerate(poses):
        for contact in read_contacts(source.parent/f'contacts_{pose}.npz'):
            points = contact['triangles_m'].reshape(-1, 3)@geometry['rotations'][k]+geometry['local_offsets_m'][k]
            ident = contact['candidate_id']
            if ident in shared_points:
                original = shared_points[ident]
                error = max(cKDTree(original).query(points)[0].max(), cKDTree(points).query(original)[0].max())
                if model == 'five_unique_heads_one_shared_patch' and error > 1e-8:
                    raise ValueError(f'Shared physical contact split by {error} m')
                shared_errors.append(error)
            else:
                shared_points[ident] = points
    assert len(shared_points) == 5 and len(shared_errors) == 1
    assert set(shared_points) == set(body_ids)
    for k, pose in enumerate(poses):
        task = read_task('B', pose, poses)
        contacts = read_contacts(source.parent/f'contacts_{pose}.npz')
        basis = geometry['rotations'][k]
        offset = geometry['local_offsets_m'][k]
        world = (vertices-offset)@basis.T
        assert np.max(np.abs(basis@basis.T-np.eye(3))) < 1e-12
        assert abs(np.linalg.det(basis)-1) < 1e-12
        assert world[:, 2].min() >= -1e-9

        saved = certificates[f'{pose}_matrix']
        ground = certificates[f'{pose}_ground_points_m']
        assignment = certificates[f'{pose}_assignment']
        weights = certificates[f'{pose}_weights']
        bases = certificates[f'{pose}_bases']
        floor_rays = np.array([[64, 0, 1], [-64, 0, 1],
                               [0, 64, 1], [0, -64, 1.]], float)
        points = [np.tile(task.floor, (4, 1))]
        forces = [floor_rays]
        owners = [np.full(4, -1)]
        for contact in contacts:
            p = contact['triangles_m'].reshape(-1, 3)
            n = np.repeat(-task.domain.mesh.face_normals[contact['source_faces']],
                          3, axis=0)
            ids = np.sort(np.unique(np.c_[p, n], axis=0, return_index=True)[1])
            points.append(p[ids])
            forces.append(n[ids])
            owners.append(np.zeros(len(ids)))
        points, forces, owners = map(np.concatenate, (points, forces, owners))

        # Workpiece rows 0:6; massless single support rows 6:12. The shared
        # object/support interface uses equal and opposite columns, not two
        # separately feasible reaction choices.
        matrix = np.zeros_like(saved)
        for j, (point, force, owner) in enumerate(zip(points, forces, owners)):
            wrench = np.r_[force, np.cross(point-task.domain.com, force)]
            matrix[:6, j] = wrench
            if owner >= 0:
                matrix[6:, j] = -wrench
        for j, point in enumerate(ground):
            force = floor_rays[j % 4]
            matrix[6:, len(points)+j] = np.r_[
                force, np.cross(point-task.domain.com, force)]
        conditioning = np.r_[task.scale, task.scale]
        conditioned = matrix*conditioning[:, None]
        matrix_error = float(np.max(np.abs(saved-conditioned)))
        assert matrix_error < 1e-12

        assert len(task.targets) == len(assignment) == 32768
        assert weights.shape == (32768, 12)
        assert np.isfinite(weights).all() and weights.min() >= 0
        assert (assignment >= 0).all() and (assignment < len(bases)).all()
        targets = np.zeros((32768, 12))
        targets[:, :6] = task.targets/task.scale
        residuals = np.empty_like(targets)
        for j, columns in enumerate(bases):
            ids = np.flatnonzero(assignment == j)
            if len(ids):
                residuals[ids] = weights[ids]@matrix[:, columns].T-targets[ids]
        force_residual = float(np.max(np.abs(residuals[:, [0, 1, 2, 6, 7, 8]])))
        torque_residual = float(np.max(np.abs(residuals[:, [3, 4, 5, 9, 10, 11]])))
        scaled_residual = float(np.max(np.abs(residuals*conditioning)))
        assert scaled_residual < 2e-9

        # Do not accept merely projected or invented ground points. Every
        # certificate floor point must coincide with a final mesh vertex on
        # the actual plane, under the saved fixture placement.
        floor_vertices = world[np.abs(world[:, 2]) < 1e-9]
        distances = cKDTree(floor_vertices).query(ground)[0]
        assert distances.max() < 1e-9
        assert np.abs(ground[:, 2]).max() < 1e-12
        checks.append(dict(
            pose=pose, original_load_count=len(task.targets),
            verified_reaction_certificates=len(assignment),
            active_contact_ids=[c['candidate_id'] for c in contacts],
            physical_matrix_shape=list(matrix.shape),
            conditioned_matrix_rebuild_max_error=matrix_error,
            minimum_reaction_coefficient=float(weights.min()),
            maximum_force_residual_mg=force_residual,
            maximum_torque_residual_mg_m=torque_residual,
            maximum_conditioned_residual=scaled_residual,
            maximum_ground_to_actual_mesh_vertex_distance_m=float(distances.max()),
            minimum_fixture_height_m=float(world[:, 2].min()),
            friction_coefficient=64., passed=True,
        ))

    return dict(
        schema='independent_rigid_fixture_equilibrium_replay_v1', passed=True,
        reviewed_report_sha256=sha256(report_path),
        reviewed_artifacts=report['artifacts'],
        original_input_hashes_checked=len(report['provenance']['inputs']),
        code_hashes_checked=len(report['provenance']['code']),
        artifact_hashes_checked=len(report['artifacts']),
        same_fixed_original_loads=True, new_loads_added=False,
        new_object_contacts_used=False, support_mass_ignored=True,
        independently_rebuilt_both_free_body_equations=True,
        independently_recomputed_complete_withdrawal_sweep=False,
        geometry=dict(watertight=True, consistently_wound=True,
                      connected_components=components, volume_cm3=float(mesh.volume*1e6),
                      physical_head_definition=model,
                      same_label_surface_separation_m=list(map(float, shared_errors)),
                      contact_groups=report['contact_groups'],
                      contact_patch_count=report['contact_patch_count']),
        checks=checks,
        interpretation=(
            'Independent physical matrix and all original per-sample reaction replay. '
            'Full sweep construction was reviewed and toy-tested separately; this '
            'replay does not reconstruct that Boolean sweep or certify strength, '
            'stiffness, minimal area, or material optimality.'),
        audit_script_sha256=sha256(__file__),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'slides/baseline_algo/output/B/pose1+3/step4')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.input)
    (args.output or args.input/'independent_audit.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))
