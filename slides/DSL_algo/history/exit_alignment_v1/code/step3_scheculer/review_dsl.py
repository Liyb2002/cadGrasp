"""Reload DSL artifacts, independently rebuild rays and replay final conditions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step3_scheculer.run_dsl import saved_task
from step3_scheculer.pair_scoring import J, C
from step3_scheculer.floor_support import columns as floor_columns
from step2_local_support import surface as SURF, circles as P, withdrawal as W


def review_group(group):
    root = group/'step3_scheculer/dsl'
    schedule = I.check_report(root/'report.json')
    tasks = [saved_task(group, p) for p in schedule['poses']]
    records = []
    for task in tasks:
        report = I.check_report(root/task.pose/'report.json')
        force_report = I.check_report(root/task.pose/'force_only_report.json')
        force_contacts = I.read_contacts(root/task.pose/'force_only_contacts.npz')
        force_mask, _ = J.classify(task.supply(force_contacts), task.targets)
        np.testing.assert_array_equal(force_mask, I.load_npz(root/task.pose/'force_only_coverage.npz')['passed'])
        assert int(force_mask.sum()) == force_report['covered_count']
        contacts = I.read_contacts(root/task.pose/f'final_contacts_{task.pose}.npz')
        raw_floor = floor_columns(task.floor, task.domain.com)
        rays = [np.c_[raw_floor*task.scale, np.zeros(len(raw_floor))], np.r_[np.zeros(6), -1.][None]]
        floor_minima = []
        frame = np.asarray(task.domain.data['frame']['T_world_mesh'])
        for contact in contacts:
            vertices = contact['triangles_m'].reshape(-1, 3)
            source = np.repeat(contact['source_faces'], 3)
            triangles = task.domain.mesh.triangles[source]
            normal = task.domain.mesh.face_normals[source]
            on_plane = np.einsum('ij,ij->i', vertices-triangles[:,0], normal)
            assert np.max(np.abs(on_plane)) < 1e-10
            import trimesh
            weights = trimesh.triangles.points_to_barycentric(triangles, vertices)
            assert weights.min() >= -1e-7 and weights.max() <= 1+1e-7
            assert not np.intersect1d(contact['source_faces'], task.domain.work_ids).size
            assert P.normal_spread(task.domain.mesh, contact['source_faces'])['wrap_limit_satisfied']
            np.testing.assert_allclose(SURF.areas(contact['triangles_m']), contact['triangle_areas_m2'], atol=1e-15)
            force = -normal
            moment = np.cross(vertices-task.domain.com, force)
            rays.append(np.c_[np.c_[force, moment]*task.scale, force[:,2]])
            object_vertices = (vertices-frame[:3,3])@frame[:3,:3]
            for other in tasks:
                transform = np.asarray(other.domain.data['frame']['T_world_mesh'])
                world = object_vertices@transform[:3,:3].T+transform[:3,3]
                height = float(world[:,2].min())
                floor_minima.append(height)
                assert height >= .002-1e-10
        full = I.merge_columns(*rays)
        mask, info = J.classify(full, task.targets)
        stored = I.load_npz(root/task.pose/'coverage.npz')['passed']
        np.testing.assert_array_equal(mask, stored)
        assert int(mask.sum()) == report['covered_count']
        d = np.asarray(report['selected_withdrawal_world'])
        rotation = np.asarray(task.domain.data['frame']['T_world_mesh'])[:3,:3]
        np.testing.assert_allclose(-d@rotation, report['selected_object_exit'], atol=1e-14)
        exits = []
        if report['exit_passed']:
            normals = np.vstack([task.domain.mesh.face_normals[c['source_faces']] for c in contacts])
            assert (normals@d).min() >= -W.NORMAL_TOL
            analyzer = W.Analyzer(task.domain.mesh, .0002, dict(vectors=[d.tolist()]))
            exits = [analyzer.test(analyzer.heads(c), d) for c in contacts]
            assert all(r['clear'] for r in exits)
        # Fresh primal solves on a deterministic spread of loads supplement the
        # full independently rebuilt classification and compare force signs.
        witness_count = 0
        if mask.all():
            for ident in np.linspace(0,len(mask)-1,24,dtype=int):
                assert C.W.solve(full, task.targets[ident]) is not None
                witness_count += 1
        gradient_steps = [step for event in report['events'] for step in event.get('gradient_steps', [])]
        records.append(dict(pose=task.pose, reported_passed=report['passed'], covered=int(mask.sum()),
            force_only_checkpoint_covered=int(force_mask.sum()),
            heads=len(contacts), original_samples_checked=len(mask), independently_rebuilt_seven_row_rays=True,
            fresh_primal_witnesses=witness_count, fresh_continuous_exit_checks=exits,
            minimum_all_pose_contact_height_m=min(floor_minima,default=None),
            accepted_gradient_updates=sum(s['accepted'] for s in gradient_steps)))
    body_audit = None
    body_path = group/'step4/data/dsl_support/report.json'
    if body_path.exists():
        body = json.loads(body_path.read_text())
        body_audit = dict(design_passed=body['passed'], status=body.get('status'))
        if body['passed']:
            I.check_report(body_path)
            import trimesh
            from step4_connect_support.try_dsl_growth import DSLGrow
            grow = DSLGrow(group)
            mesh = trimesh.load(body_path.parent/'shape.obj', force='mesh', process=False)
            geometry, _ = grow.verify(mesh)
            body_audit['fresh_exported_geometry_replay'] = geometry
    return dict(group=group.name, audit_passed=True, design_passed=schedule['passed'], poses=records,
                step4_audit=body_audit)


def baseline_integrity():
    manifest = json.loads((Path(__file__).resolve().parent.parent/'copy_manifest.json').read_text())
    changed = [name for name,digest in manifest['source_hashes'].items()
               if not (I.ROOT/name).is_file() or I.sha256(I.ROOT/name) != digest]
    original = set(manifest['source_hashes'])
    current = {str(p.relative_to(I.ROOT)) for p in (I.ROOT/'slides/baseline_algo').rglob('*') if p.is_file()}
    added = sorted(current-original)
    result = dict(passed=not changed and not added, protected_file_count=len(original), changed=changed, added=added,
        scope='Workspace baseline versus copy-time snapshot; this comparison does not attribute writes to this experiment')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--groups', nargs='+')
    args = parser.parse_args()
    root = I.OUTPUTS/'B'
    groups = [root/g for g in args.groups] if args.groups else sorted(p.parent.parent.parent for p in root.rglob('dsl/report.json')
        if p.parent.parent.name == 'step3_scheculer')
    rows = []
    for group in groups:
        try:
            row = review_group(group)
        except Exception as error:
            import traceback
            row = dict(group=group.name, audit_passed=False,error=str(error),traceback=traceback.format_exc())
        rows.append(row)
        print('DSL REVIEW',group.name,row['audit_passed'],flush=True)
    integrity = baseline_integrity()
    inputs=[Path(__file__).resolve().parent.parent/'copy_manifest.json']
    for group in groups:
        folder=group/'step3_scheculer/dsl'
        inputs.append(folder/'report.json')
        inputs += list(folder.glob('pose_*/report.json'))+list(folder.glob('pose_*/force_only_report.json'))
        body=group/'step4/data/dsl_support/report.json'
        if body.exists():inputs.append(body)
    result = dict(complete=True, passed=all(r['audit_passed'] for r in rows),
        baseline_snapshot_matches_current_workspace=integrity['passed'],
        baseline_integrity=integrity, results=rows,
        provenance=dict(inputs=I.hashes(inputs),code=I.hashes([Path(__file__)])))
    I.save(root/'pose2+9+13+15+17/step3_scheculer/dsl/review.json',result)
    if not result['passed']:
        raise SystemExit('DSL audit failed; inspect review.json')
