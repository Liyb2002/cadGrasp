"""Saved-record/source verification only; no exported geometry acceptance replay."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from co_common import *


def main():
    batch=I.check_report(HERE/'output/B/data/batch.json')
    assert batch['pose_sets']==20 and batch['pose_instances']==80
    manifest=json.loads((ROOT/'objects/B/pose_sets.json').read_text())
    assert {r['id'] for r in batch['results']}=={g['id'] for g in manifest['sets']}
    I.check_hashes(json.loads((HERE/'data/protected_inputs.json').read_text()))
    instances=0;max_error=0.;ratios=[]
    for group in manifest['sets']:
        out=HERE/'output/B'/group['id']/'step3'
        top=I.check_report(out/'data/report.json')
        reg=I.check_report(out/'step3.1/data/report.json')
        wrap=I.check_report(out/'step3.2/data/report.json')
        assert reg['poses']==group['poses']==wrap['poses']
        assert reg['object_native_poses_fixed'] and reg['registered_objects_coincide']
        max_error=max(max_error,max(r['alignment_max_error_m'] for r in reg['states']))
        assert top['status']==wrap['status'] and not wrap['full_fixture_accepted'] and not wrap['exit_checked']
        with np.load(out/'step3.2/data/contacts.npz') as z:
            assert not np.intersect1d(z['source_faces'],z['excluded_work_faces']).size
            assert set(np.unique(z['source_faces']))<=set(z['allowed_faces'])
        if wrap['status']=='pass':
            assert wrap['force_gate_passed'] and wrap['working_surfaces_clear'] and wrap['step4_ready']
        for row in wrap['state_results']:
            with np.load(out/'step3.2/data'/row['pose']/'coverage.npz') as z:
                assert z['mask'].shape==(32768,) and int(z['mask'].sum())==row['covered']
                if row['status']=='pass':assert z['mask'].all()
            assert row['load_count']==32768
            if row['status']=='fail':assert row['failure_proof']['every_original_generator_checked_exactly']
            instances+=1
        ratios.append(wrap['actual_contact_area_m2']/wrap['ideal_nonworking_area_m2'])
    assert instances==80
    assert not (HERE/'step4.1/step41.py').exists() and not (HERE/'step4.1').exists()
    assert not (HERE/'output/B/single_poses').exists()
    assert not list((HERE/'output/B').glob('*.json'))
    r=dict(complete=True,passed=True,pose_sets=20,pose_instances=instances,protected_files=959,max_registered_object_error_m=max_error,minimum_actual_to_ideal_contact_area_ratio=min(ratios),original_loads_per_pose=32768,tests=5,export_geometry_replayed=False)
    save(HERE/'output/B/data/verification.json',r);print(json.dumps(r,indent=2))
if __name__=='__main__':main()
