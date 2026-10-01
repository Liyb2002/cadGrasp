"""Diagnose shared-head identity independently of the Step5 constructor.

Read saved geometry and original Step3 patches; do not rebuild or modify them.
Distance queries use millimetres to avoid absolute small-triangle tolerances in
trimesh. Surface samples are diagnostic evidence, not an area-contact proof.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'slides/baseline_algo'))
from step3_scheculer.contacts import read_contacts
from step3_scheculer.pair_tasks import read_task
from step4_floor_contact.whole_assembly import pressure_centers


def digest(data):
    return hashlib.sha256(data).hexdigest()


def distances_mm(mesh, points_m):
    scaled = mesh.copy()
    scaled.apply_scale(1000)
    chunks = np.array_split(points_m, max(1, int(np.ceil(len(points_m)/128))))
    return np.concatenate([trimesh.proximity.closest_point(scaled, p*1000)[1]
                           for p in chunks])


def cloud_disagreement(a, b):
    return float(max(cKDTree(a).query(b)[0].max(), cKDTree(b).query(a)[0].max()))


def inspect(report, geometry):
    poses = report['poses']
    source = ROOT/report['source_schedule']
    groups = [read_contacts(source.parent/f'contacts_{p}.npz') for p in poses]
    tasks = [read_task(report['object'], p, poses) for p in poses]
    frames = [np.asarray(t.domain.data['frame']['T_world_mesh']) for t in tasks]
    fixture = trimesh.Trimesh(geometry['vertices_m'], geometry['faces'], process=False)
    bases, offsets = geometry['rotations'], geometry['local_offsets_m']
    heads, records = [], []
    for k, contacts in enumerate(groups):
        for contact in contacts:
            triangles = contact['triangles_m']
            samples = np.unique(np.concatenate([
                triangles.reshape(-1, 3), triangles.mean(1),
                ((triangles+np.roll(triangles, 1, axis=1))/2).reshape(-1, 3),
            ]), axis=0)
            local = samples@bases[k]+offsets[k]
            centroid = np.average(triangles.mean(1), axis=0,
                                  weights=contact['triangle_areas_m2'])
            heads.append(dict(owner=k, contact=contact, points=local,
                              centroid=centroid@bases[k]+offsets[k]))
            per_pose = []
            for j, task in enumerate(tasks):
                world = (local-offsets[j])@bases[j].T
                gaps = distances_mm(task.domain.mesh, world)
                per_pose.append(dict(pose=poses[j], active=j == k,
                    minimum_surface_gap_mm=float(gaps.min()),
                    maximum_surface_gap_mm=float(gaps.max())))
            records.append(dict(candidate_id=contact['candidate_id'],
                owner_pose=poses[k], sampled_points=len(samples),
                maximum_fixture_surface_gap_mm=float(distances_mm(fixture, local).max()),
                per_pose=per_pose))

    shared = []
    for a in heads:
        for b in heads:
            if a['owner'] >= b['owner'] or a['contact']['candidate_id'] != b['contact']['candidate_id']:
                continue
            canonical = []
            for h in (a, b):
                transform = frames[h['owner']]
                vertices = h['contact']['triangles_m'].reshape(-1, 3)
                canonical.append((vertices-transform[:3, 3])@transform[:3, :3])
            separation = float(np.linalg.norm(a['centroid']-b['centroid'])*1000)
            shared.append(dict(candidate_id=a['contact']['candidate_id'],
                source_object_frame_vertex_disagreement_mm=cloud_disagreement(*canonical)*1000,
                fixture_area_centroid_separation_mm=separation,
                fixture_sampled_cloud_disagreement_mm=cloud_disagreement(a['points'], b['points'])*1000,
                same_physical_patch_preserved=separation < 1e-5 and
                    cloud_disagreement(a['points'], b['points']) < 1e-8))

    # Necessary condition under EXACT original patch/material correspondence:
    # the two floor planes have their object-frame relative transform. Every
    # legal floor point (and the original object pivot) lies in the other floor's
    # upper halfspace. Their nonnegative normal-force CoP must do so as well.
    # This does not apply to re-seating on a different face or sliding contact.
    strict_floor = []
    for k, task in enumerate(tasks):
        xy = pressure_centers(task.targets/task.scale, task.domain.com)[0]
        assert len(xy) == 32768
        transform = frames[1-k]@np.linalg.inv(frames[k])
        mapped = np.c_[xy, np.zeros(len(xy))]@transform[:3, :3].T+transform[:3, 3]
        pivot_height = float((task.floor@transform[:3, :3].T+transform[:3, 3])[2])
        violations = np.flatnonzero(mapped[:, 2] < -1e-9)
        strict_floor.append(dict(pose=poses[k], sample_count=len(xy),
            violating_original_sample_indices=violations.tolist(),
            violating_sample_count=len(violations),
            minimum_other_floor_height_mm=float(mapped[:, 2].min()*1000),
            original_object_pivot_other_floor_height_mm=pivot_height*1000,
            necessary_condition_passed=not len(violations) and pivot_height >= -1e-9))
    return dict(schema='shared_head_identity_diagnostic_v1',
        source_schedule=report['source_schedule'], poses=poses,
        original_contact_label_count=report['contact_groups'],
        exported_physical_patch_count=report['contact_patch_count'],
        shared_patches=shared,
        original_five_head_definition_passed=(report['contact_patch_count'] == 5
            and len(shared) == 1 and all(s['same_physical_patch_preserved'] for s in shared)),
        head_surface_samples=records,
        exact_original_patch_registration_floor_necessary_conditions=strict_floor,
        floor_test_scope='Original fixed contact correspondence and task poses; no contact sliding, re-seating or added loads',
        physical_artifacts_modified=False,
        volume_cm3=report.get('volume_cm3'), dimensions_mm=report['dimensions_mm'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object', default='B')
    parser.add_argument('--pair', action='append')
    parser.add_argument('--include-archive', action='store_true')
    args = parser.parse_args()
    folders = sorted((ROOT/'slides/baseline_algo/output'/args.object).glob('pose*+*/step5'))
    for out in folders:
        if args.pair and out.parent.name not in args.pair:
            continue
        report_bytes = (out/'report.json').read_bytes()
        report = json.loads(report_bytes)
        geometry_name = 'head_layout.npz' if report.get('diagnostic_only') else 'geometry.npz'
        geometry_bytes = (out/geometry_name).read_bytes()
        with np.load(io.BytesIO(geometry_bytes)) as geometry:
            result = inspect(report, geometry)
            if geometry_name == 'head_layout.npz':
                ids = geometry['physical_head_ids'].tolist()
                assert len(ids) == len(set(ids)) == 5
                assert set(ids) == set(report['registration']['active_head_ids'][0]) | set(report['registration']['active_head_ids'][1])
                assert len(geometry['head_vertex_offsets']) == len(geometry['head_face_offsets']) == 6
                result['exported_unique_head_ids'] = ids
                result['head_layout_only'] = True
        result['provenance'] = dict(report_sha256=digest(report_bytes),
            geometry_file=geometry_name, geometry_sha256=digest(geometry_bytes), audit_code_sha256=digest(Path(__file__).read_bytes()),
            original_inputs=report['provenance']['inputs'])
        if args.include_archive:
            archive = Path(__file__).parent/'archive/2026-09-27_fixed_local_bodies/snapshot.tar.gz'
            prefix = str(out.relative_to(ROOT))+'/'
            with tarfile.open(archive) as tar:
                if prefix+'report.json' in tar.getnames():
                    old_report = json.load(tar.extractfile(prefix+'report.json'))
                    with np.load(io.BytesIO(tar.extractfile(prefix+'geometry.npz').read())) as geometry:
                        result['archived_comparison'] = inspect(old_report, geometry)
                    result['provenance']['archive_sha256'] = digest(archive.read_bytes())
        (out/'shared_head_audit.json').write_text(json.dumps(result, indent=2)+'\n')
        print(out.parent.name, 'original_five_head_definition_passed=',
              result['original_five_head_definition_passed'],
              'shared_centroid_separation_mm=',
              [s['fixture_area_centroid_separation_mm'] for s in result['shared_patches']], flush=True)


if __name__ == '__main__':
    main()
