"""Explain finite-search exit failures without asserting physical infeasibility."""
from pathlib import Path
from collections import Counter
import json
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step3_scheculer.run_dsl import saved_task
from step2_local_support import withdrawal as W

def main():
    root=I.OUTPUTS/'B';out=root/'pose2+9+13+15+17/step3_scheculer/dsl'
    batch=json.loads((out/'batch_report.json').read_text());rows=[];inputs=[]
    for case in batch['results']:
        if case['passed'] or 'error' in case:continue
        if case.get('local_contact_passed'):
            witness=I.ROOT/case['construction_witness']
            body=I.check_report(witness)
            rows.append(dict(case=case['group'],stage='complete_construction_witness',
                force_passed=case['force_passed'],local_contact_passed=True,
                complete_fixture_verified=False,reason=body.get('error',body.get('status')),
                interpretation='Rejected by Step3 because a complete constructive witness was not found',
                global_motion_infeasibility_claim=False))
            inputs.append(witness)
        for name in case['per_pose_reports']:
            path=I.ROOT/name;report=I.check_report(path)
            if report['exit_passed']:continue
            group=path.parents[3];task=saved_task(group,report['pose'])
            contacts=I.read_contacts(path.parent/f"final_contacts_{task.pose}.npz")
            vectors=np.asarray(report['catalogue']['vectors'])
            feasible=[];all_normals=[]
            for contact in contacts:
                normals=task.domain.mesh.face_normals[np.unique(contact['source_faces'])]
                all_normals.append(normals);feasible.append((normals@vectors.T).min(0)>=-W.NORMAL_TOL)
            common=np.logical_and.reduce(feasible)
            scores=(np.vstack(all_normals)@vectors.T).min(0)
            rows.append(dict(case=case['group'],pose=task.pose,force_passed=report['force_passed'],
                finite_option_count=len(vectors),per_head_initial_opening_counts=[int(v.sum()) for v in feasible],
                common_initial_opening_count=int(common.sum()),best_minimum_normal_dot=float(scores.max()),
                checked_path_failure_reasons=dict(Counter(r['check']['reason'] for r in report['exit_checks'] if not r['passed'])),
                cross_pose_angle_penalty=False,
                root_connectivity=report['root_connectivity'],
                interpretation='Baseline root-path connectivity prefilter failed' if not report['root_connectivity']['passed'] else 'No opening in this finite translation catalogue for this contact set' if not common.any() else 'Initial normal opening exists, but tested continuous paths collide or fail separation',
                global_motion_infeasibility_claim=False))
            inputs.append(path)
    I.save(out/'exit_diagnostics.json',dict(complete=True,failed_tasks=rows,provenance=dict(inputs=I.hashes(inputs),code=I.hashes([Path(__file__)]))))
    for row in rows:print(row)
if __name__=='__main__':main()
