"""Independently replay original loads on final continuous-batch mesh contacts."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
import argparse
from PIL import Image
from co_common import *
from run_all import output_root
from whole_search.classify import classify


def audit_case(root,row):
    out=root/row['id']/'step4/step4.2/continuous'
    report=I.check_report(out/'data/report.json');render=I.check_report(out/'data/render.json')
    assert report['poses']==row['poses'] and report['force_exit_work_passed']
    source=root/row['id']/'step4/step4.1'
    baseline=I.check_report(source/'data/report.json')
    assert report['baseline_volume_cm3'] is None or abs(report['baseline_volume_cm3']-baseline['volume_cm3'])<1e-4
    states=[state('B',p) for p in row['poses']];mesh=states[0][2]
    support=trimesh.load(out/'support.obj',force='mesh',process=False)
    assert abs(abs(support.volume)*1e6-report['volume_cm3'])<1e-4
    with np.load(out/'layout.npz') as arrays:
        placements=arrays['placements'];directions=arrays['directions'];hosts=arrays['hosts']
        np.testing.assert_array_equal(arrays['active'],np.arange(len(states)))
        np.testing.assert_allclose(arrays['native_world'],np.array([s[1] for s in states]),atol=1e-12,rtol=0)
    replay=[]
    for k,(task,T,body) in enumerate(states):
        q=placements[k];world=states[int(hosts[k])][1]
        np.testing.assert_allclose((world@q)[:3,:3],T[:3,:3],atol=1e-9,rtol=0)
        assert abs((world@q)[2,3]-T[2,3])<1e-9
        assert (world[:3,:3]@directions[k])[2]>=-1e-12
        np.testing.assert_allclose(np.linalg.norm(directions[k]),1.,atol=1e-12,rtol=0)
        allowed=np.setdiff1d(np.arange(len(mesh.faces)),task.domain.work_ids)
        allowed=allowed[mesh.face_normals[allowed]@(q[:3,:3].T@directions[k])<=1e-9]
        posed=mesh.copy();posed.apply_transform(q)
        triangles,faces=contact_boundary(posed,support,allowed)
        columns=supply(task,T@np.linalg.inv(q),triangles,faces)
        with np.load(out/f'{row["poses"][k]}_force.npz') as arrays:
            assert arrays['mask'].shape==(32768,) and arrays['mask'].all()
            np.testing.assert_array_equal(faces,arrays['source_faces'])
            np.testing.assert_allclose(triangles,arrays['triangles_fixture_m'],atol=1e-12,rtol=0)
            np.testing.assert_allclose(columns,arrays['supply_7d'],atol=1e-12,rtol=1e-12)
        mask,details=classify(columns,task.targets)
        assert len(mask)==32768 and mask.all()
        replay.append(dict(pose=row['poses'][k],original_loads=len(mask),passed=True,classifier=details))
    names=['nominal_sweep_overlap_m3','body_overlap_m3','padded_overlap_outside_contact_cores_m3',
           'work_band_overlap_m3','exported_boundary_max_exclusion_overlap_m3','endpoint_overlap_m3']
    assert max(report['diagnostics'][name] for name in names)<=1e-10
    assert report['diagnostics']['minimum_endpoint_projection_gap_m']>0
    assert all(check['passed'] for check in report['actual_work_surface_checks'])
    assert render['final_force_exit_work_verified'] and render['last_tile_uses_verified_final_geometry']
    for filename in ['process.png','final_result.png']:
        with Image.open(out/filename) as image:image.verify()
    quadrature=json.loads((out/'quadrature_check.json').read_text())
    assert quadrature['evaluated_layout_is_published_actual'] and not quadrature['continuous_domain_certified']
    if not report['step41_baseline_retained']:
        assert report['continuous_quadrature_gate_passed']
        assert report['baseline_volume_cm3'] is None or report['volume_cm3']<report['baseline_volume_cm3']-1e-4
    result=dict(complete=True,passed=True,id=row['id'],volume_cm3=report['volume_cm3'],
        original_load_replays=replay,original_loads_replayed=sum(r['original_loads'] for r in replay),
        exact_geometry_proofs_fingerprinted_not_recomputed=True,continuous_domain_certified=False,
        report_sha256=I.sha256(out/'data/report.json'),provenance=provenance([], [Path(__file__)]))
    save(out/'data/audit.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('object',nargs='?',default='B')
    args=parser.parse_args();root=output_root(args.object)
    batch=json.loads((root/'data/continuous_step4_batch.json').read_text());assert batch['complete']
    manifest=json.loads((root/'_history'/batch['experiment']/'manifest.json').read_text())
    assert len(batch['results'])==len(manifest['groups'])==batch['requested_sets']
    assert len({r['id'] for r in batch['results']})==len(manifest['groups'])
    assert len(manifest['deleted_stage41_folders'])==len(manifest['groups'])
    for relative,digest in manifest['protected_step3_hashes'].items():assert I.sha256(ROOT/relative)==digest
    for relative,digest in manifest['original_step4_hashes'].items():
        assert I.sha256(ROOT/manifest['archived_paths'][relative])==digest
    rows=[]
    for row in batch['results']:
        if row['passed']:
            result=audit_case(root,row);rows.append(result)
            print('CONTINUOUS AUDIT',row['id'],result['original_loads_replayed'],flush=True)
        else:
            out=root/row['id']/'step4/step4.2/continuous'
            if row.get('step42'):
                failure=json.loads((out/'data/failure.json').read_text());assert not failure['passed']
            rows.append(dict(id=row['id'],actual_final_passed=False,status=row['status']))
    result=dict(complete=True,passed=True,requested_sets=batch['requested_sets'],results=rows,
        actual_passed_sets=sum(r.get('passed',False) for r in rows),
        original_loads_replayed=sum(r.get('original_loads_replayed',0) for r in rows),
        protected_step3_files=len(manifest['protected_step3_hashes']),
        original_step4_archive_files=len(manifest['original_step4_hashes']),
        continuous_domain_certified=False,full_fixture_accepted=False)
    save(root/'data/continuous_step4_audit.json',result)
    print('AUDIT COMPLETE',result['actual_passed_sets'],result['original_loads_replayed'],flush=True)


if __name__=='__main__':main()
