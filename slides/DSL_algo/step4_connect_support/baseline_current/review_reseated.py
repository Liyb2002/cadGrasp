"""Replay K-pose placements, whole-body motion and original reaction certificates."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step3_scheculer.pair_tasks import read_task
from step4_connect_support.baseline_current import reseating as C, build_coupled_saddle as S
from step4_connect_support.baseline_current.surface_check import surface_distances,winding_number
from step2_local_support import geometry as G


def review(output):
    output=output.resolve()
    report_path = output/'data/report.json'
    report = I.check_report(report_path)
    assert report['schema'] == 'independent_seating_step4_v2'
    assert report['complete']
    if not report['constructed']:return review_preview(output,report)
    body = trimesh.load(output/'shape.obj', force='mesh', process=False)
    fixture_solid=S.solid(body)
    assert body.is_watertight and body.is_winding_consistent and body.volume > 0
    # Check the exported indexed topology directly. Reconstructing a Manifold
    # can simplify near-collinear triangles and split numerical zero-volume
    # slivers; it does not preserve the exported graph being reviewed here.
    shells = body.split(only_watertight=False)
    assert len(shells) == 1
    work = output/report['body_directory']
    construction=I.check_report(work/'report.json')
    assert construction==report['construction']
    case=C.load_case(output)
    C.precompute(case)
    all_bases=np.asarray(report['placement']['bases'])
    all_offsets=np.asarray(report['placement']['offsets'])
    actual_span=max(np.ptp((body.vertices-o)@b.T,axis=0)[:2].max() for b,o in zip(all_bases,all_offsets))
    spatial=max(np.ptp((body.vertices-o)@b.T,axis=0).max() for b,o in zip(all_bases,all_offsets))
    assert bool(spatial<=report['maximum_spatial_span_m']+1e-9)==report['compactness_passed']
    assert abs(actual_span-report['metrics']['maximum_horizontal_span_m'])<1e-12
    certificate_path = work/('equilibrium.npz' if construction['passed'] else 'equilibrium_diagnostic.npz')
    source_schedule = I.ROOT/report['construction']['source_schedule']
    schedule = json.loads(source_schedule.read_text())
    source_folders = {}
    for relative in schedule['source_reports']:
        path = I.ROOT/relative
        source_report = I.check_report(path)
        assert len(source_report['poses']) == 1
        source_folders[source_report['poses'][0]] = path.parent.parent
    checks, identifiers = [], []
    with np.load(work/'geometry.npz') as geometry, np.load(certificate_path) as cert:
        np.testing.assert_allclose(body.vertices, geometry['vertices_m'], atol=1e-14, rtol=0)
        np.testing.assert_array_equal(body.faces, geometry['faces'])
        for k, pose in enumerate(report['poses']):
            folder = source_folders[pose]
            I.check_report(folder/'step3_scheculer/schedule.json')
            task = read_task(report['object'], pose, folder=folder/'step_1_needs')
            source = folder/'step3_scheculer'/f'final_contacts_{pose}.npz'
            copied = (I.ROOT/report['construction']['source_schedule']).parent/f'contacts_{pose}.npz'
            assert I.sha256(source) == I.sha256(copied)
            contacts = I.read_contacts(source)
            identifiers.extend(c['candidate_id'] for c in contacts)
            basis, offset = geometry['rotations'][k], geometry['local_offsets_m'][k]
            np.testing.assert_allclose(basis@basis.T, np.eye(3), atol=1e-12)
            assert abs(np.linalg.det(basis)-1) < 1e-12
            world = (body.vertices-offset)@basis.T
            assert world[:, 2].min() >= -1e-9
            np.testing.assert_allclose(basis,all_bases[k],atol=1e-14,rtol=0)
            np.testing.assert_allclose(offset,all_offsets[k],atol=1e-14,rtol=0)
            # Independent homogeneous inverse of the task-to-fixture map.
            transform=np.eye(4);transform[:3,:3]=basis.T;transform[:3,3]=offset
            inverse=np.linalg.inv(transform)
            raw=(np.c_[body.vertices,np.ones(len(body.vertices))]@inverse.T)[:,:3]
            np.testing.assert_allclose(raw,world,atol=1e-12,rtol=0)
            world_mesh=trimesh.Trimesh(raw,body.faces,process=False)
            solid=S.solid(world_mesh)
            selected_id=report['placement']['direction_ids'][k]
            sweep=case.sweeps[k][selected_id]
            overlap=abs(float((solid^sweep).volume()))*S.SCALE**3
            direction=-case.catalogues[k][selected_id]
            separation=float((task.domain.mesh.vertices@direction).min()+S.SWEEP_LENGTH-(raw@direction).max())
            assert bool(overlap<=C.TOL and separation>0)==report['construction']['checks'][k]['withdrawal']['clear']
            assert direction[2]>=-1e-12
            # Owner patches and their complete finite roots must be retained;
            # inactive roots must clear the padded continuous object corridor.
            union_missing=abs(float((case.root_solids[k]-solid).volume()))*S.SCALE**3
            # Evaluate the defining convex cells in the exported fixture's
            # construction frame. Alternate-frame differences are retained as
            # diagnostics: 1e-16 m coordinate roundoff can make near-coplanar
            # Boolean boundaries report false missing slivers. Also inspect
            # their witnesses using a separate winding-number implementation.
            own_missing=0.;task_missing=0.;root_diagnostics=[]
            for head_index,cells in enumerate(case.support_seeds[k]):
                for cell_index,cell in enumerate(cells):
                    canonical=abs(float((S.solid(G.hull_mesh(cell@basis+offset))-fixture_solid).volume()))*S.SCALE**3
                    own_missing=max(own_missing,canonical)
                    difference=S.solid(G.hull_mesh(cell))-solid
                    alternate=abs(float(difference.volume()))*S.SCALE**3
                    task_missing=max(task_missing,alternate)
                    if alternate>C.TOL:
                        point=S.unpack(difference).center_mass
                        root_diagnostics.append(dict(head_index=head_index,cell_index=cell_index,
                            alternate_frame_difference_m3=alternate,fixture_frame_difference_m3=canonical,
                            difference_centroid_m=point.tolist(),centroid_winding_number=winding_number(world_mesh,point),
                            centroid_surface_distance_m=float(surface_distances(world_mesh,[point])[0]),
                            coordinate_roundoff_m=float(np.abs(raw-world).max())))
            assert own_missing<=C.TOL
            foreign_overlap=0.
            for j in range(len(case.poses)):
                if j==k:continue
                foreign_fixture=C.transform_solid(case.root_solids[j],all_bases[j],all_offsets[j])
                foreign_world=C.transform_solid(foreign_fixture,basis.T,-offset@basis.T)
                foreign_overlap=max(foreign_overlap,abs(float((foreign_world^case.padded_sweeps[k][selected_id]).volume()))*S.SCALE**3)
            assert foreign_overlap<=C.TOL
            patches=np.concatenate([c['triangles_m'] for c in contacts])
            probes=np.vstack([patches.reshape(-1,3),patches.mean(axis=1)])
            gap=float(surface_distances(world_mesh,probes).max())
            assert gap<1e-8
            for owner,group in enumerate(case.groups):
                points=np.concatenate([c['triangles_m'].reshape(-1,3) for c in group])
                fixture_points=points@all_bases[owner]+all_offsets[owner]
                assert ((fixture_points-offset)@basis.T)[:,2].min()>=.002-1e-9
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
                maximum_certificate_residual=residual, maximum_ground_vertex_error_m=ground_error,
                entire_500mm_sweep_overlap_m3=overlap,terminal_separation_m=separation,
                maximum_missing_owner_root_cell_volume_m3=own_missing,
                root_containment_frame='exported_fixture',
                alternate_task_frame_maximum_cell_difference_m3=task_missing,
                root_boolean_diagnostics=root_diagnostics,
                preunion_root_difference_m3_diagnostic=union_missing,
                foreign_root_padded_sweep_overlap_m3=foreign_overlap,
                maximum_patch_probe_gap_m=gap))
    assert len(identifiers) == len(set(identifiers)) == report['physical_head_count']
    result = dict(complete=True, replay_passed=True, fixture_acceptance_passed=report['passed'],
        one_closed_connected_solid=True, indexed_boundary_component_count=len(shells),
        unchanged_independent_contact_surfaces=True, checks=checks,
        partial_assignments_are_not_coverage=True,
        scope='Original patches/roots, actual pose floors, continuous 500 mm translation and original-load certificates; no structural strength or global optimality/infeasibility claim',
        actual_maximum_horizontal_span_m=float(actual_span),
        actual_maximum_spatial_span_m=float(spatial),
        reviewed_report_sha256=I.sha256(report_path),
        provenance=dict(inputs=I.hashes([report_path, source_schedule, output/'shape.obj', work/'geometry.npz', certificate_path]),
                        code=I.hashes([Path(__file__)])))
    I.save(output/'data/independent_review.json', result)
    return result


def review_preview(output,report):
    """A missing body is a recorded search result, never a fixture pass."""
    assert not report['passed'] and not (output/'shape.obj').exists()
    case=C.load_case(output);C.precompute(case)
    placement=report['placement']
    b=np.asarray(placement['bases']);o=np.asarray(placement['offsets'])
    C.root_records(case,b,o)
    menu,tests=C.screen(case,placement)
    if report.get('preview_layout_passed'):assert menu is not None
    identifiers=[c['candidate_id'] for group in case.groups for c in group]
    assert len(identifiers)==len(set(identifiers))==report['physical_head_count']
    result=dict(complete=True,replay_passed=True,fixture_acceptance_passed=False,
        constructed=False,preview_head_layout_passed=menu is not None,
        all_ordered_head_task_checks=tests,unchanged_independent_contact_surfaces=True,
        scope='No body was constructed; verify original contact groups and the stated head-layout preview only',
        provenance=dict(inputs=I.hashes([output/'data/report.json']+case.paths),code=I.hashes([Path(__file__)])))
    I.save(output/'data/independent_review.json',result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('outputs', type=Path, nargs='+')
    for folder in p.parse_args().outputs:
        result = review(folder)
        print(folder.parent.parent.name, 'replay passed', 'acceptance', result['fixture_acceptance_passed'])
