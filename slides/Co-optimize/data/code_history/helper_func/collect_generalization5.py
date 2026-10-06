"""Audit fixed-five-pose batch records and original-input provenance; no mesh replay."""
import argparse,json
from pathlib import Path
from collections import Counter
import _bootstrap
from co_common import I,np,save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();manifest=json.loads(a.manifest.read_text());ledger=json.loads((root/'batch.json').read_text())
    if ledger['completed']!=ledger['total']:raise RuntimeError('Wait for the complete fixed-object batch')
    selection={r['object']:r['group'] for r in manifest['cases']}
    if set(r['object'] for r in ledger['results'])!=set(selection):raise RuntimeError('Object selection changed or cases are missing')
    if I.sha256(a.manifest)!=ledger['manifest_sha256']:raise RuntimeError('Manifest changed')
    for row in ledger['results']:
        group=selection[row['object']]
        if row['pose_set']!=group['id'] or row['poses']!=group['poses']:raise RuntimeError('Saved pose set changed')
        folder=root/row['object']/group['id'];report_path=folder/'report.json'
        if report_path.exists():
            I.check_report(report_path);report=json.loads(report_path.read_text())
            initialization=json.loads((folder/'initialization_input.json').read_text())
            if [r['pose'] for r in initialization['state_results']]!=group['poses']:raise RuntimeError('Original pose identities differ')
            inputs={str(Path(path).resolve()) for path in report['provenance']['inputs']}
            for pose in group['poses']:
                for filename in ['setup.npz','needs.json','samples.json']:
                    expected=I.ROOT/'objects'/row['object']/'poses'/pose/filename
                    if str(expected.resolve()) not in inputs:raise RuntimeError('An original pose/load input is missing')
            if len(report['load_count_per_pose'])!=5 or any(n!=32768 for n in report['load_count_per_pose']):raise RuntimeError('Original full load set not checked')
            geometry=report['clearance_diagnostics']['geometry_resolved'] and report['nominal_sweep_overlap_m3']<1e-10 and report['partition_error_m3']<1e-10 and max(report['endpoint_overlap_m3'])<1e-10
            passed=bool(report['force_passed'] and geometry)
            if passed!=bool(report['passed']):raise RuntimeError('Reported acceptance inconsistent')
            row.update(passed=passed,force_passed=bool(report['force_passed']),status='complete',
                final_counts=report['final_counts'],constructor_seconds=report.get('constructor_seconds'),
                optimization_seconds=report['seconds'],provenance_verified=True)
        elif (folder/'wall_budget.json').exists():row.update(passed=False,status='wall_budget_exhausted')
        else:row.update(passed=False)
        timing=folder/'evaluation_timing.json'
        if timing.exists():
            clock=json.loads(timing.read_text());row['gate_wait_seconds']=clock['gate_wait_seconds']
            row['evaluation_wall_seconds']=max(0.,row['seconds']-clock['gate_wait_seconds'])
    ledger.update(passed=sum(r['passed'] for r in ledger['results']),complete=True,
                  by_status=dict(Counter(r['status'] for r in ledger['results'])),
                  audited=True,validation_policy='original construction records, all original loads, input/code/artifact provenance; no exported-model replay',
                  full_fixture_accepted=False)
    save(root/'batch.json',ledger)
    for row in ledger['results']:save(root/row['object']/row['pose_set']/'case.json',row)
    lines=['# Fixed five-pose generalization results','',f"Accepted: **{ledger['passed']}/{ledger['total']} objects**, five original poses per object.",
           '', 'Acceptance: every original load and full 1% clearance exits. Connectivity, support-ground coverage, strength and robot motion are deferred.',
           '', 'Search budget/time exhaustion is unresolved; a certified prerequisite failure applies only to this fixed pose registration and original force model.',
           '', '| Object | Saved pose set | Status | Force/exit | Evaluation seconds |', '|---|---|---|---|---:|']
    for row in ledger['results']:
        seconds=row.get('evaluation_wall_seconds',row['seconds'])
        lines.append(f"| {row['object']} | {row['pose_set']} | {row['status']} | {'pass' if row['passed'] else 'unresolved/fail'} | {seconds:.1f} |")
    (root/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in ledger.items() if k!='results'},indent=2))

if __name__=='__main__':main()
