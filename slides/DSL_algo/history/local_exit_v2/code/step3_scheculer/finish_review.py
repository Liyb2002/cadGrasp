"""Finish an overlapping 29-case replay with the last completed construction.

The regular review_dsl.py command remains the full standalone replay entry.
"""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I, review_dsl as R

def main():
    root=I.OUTPUTS/'B';out=root/'pose2+9+13+15+17/step3_scheculer/dsl'
    partial=out/'review_first29.json'
    if not partial.exists():partial.write_bytes((out/'review.json').read_bytes())
    result=I.check_report(partial)
    assert len(result['results'])==29 and result['passed']
    group=root/'pose6+8+10+19'
    assert not any(r['group']==group.name for r in result['results'])
    row=R.review_group(group)
    result['results'].append(row)
    result['passed']=all(r['audit_passed'] for r in result['results'])
    integrity=R.baseline_integrity()
    result.update(baseline_integrity=integrity,baseline_snapshot_matches_current_workspace=integrity['passed'])
    inputs=[partial,group/'step3_scheculer/dsl/report.json',group/'step4/data/dsl_support/report.json']
    for pose in row['poses']:
        folder=group/'step3_scheculer/dsl'/pose['pose']
        inputs.extend([folder/'report.json',folder/'force_only_report.json'])
    result['provenance']=dict(inputs=I.hashes(inputs),code=I.hashes([Path(__file__),Path(R.__file__)]))
    I.save(out/'review.json',result)
    # Reconcile all ten bodies after overlapping group runs. No design is rerun
    # or relabeled: these rows are read from the final accepted/rejected reports.
    groups=sorted(p for p in root.iterdir() if p.is_dir() and '+' in p.name)
    rows=[];paths=[]
    for group in groups:
        path=group/'step4/data/dsl_support/report.json';report=json.loads(path.read_text())
        assert report.get('complete')
        if report['passed']:I.check_report(path)
        rows.append(dict(group=group.name,passed=report['passed'],constructed=report['constructed'],
            status=report.get('status'),error=report.get('error'),volume_cm3=report.get('volume_cm3'),
            space_budget=report.get('space_budget'),trial_count=report.get('exit_path_trials',report.get('trial_count'))))
        paths.append(path)
    I.save(root/'pose2+9+13+15+17/step4/data/dsl_support/batch_report.json',dict(complete=True,results=rows,
        provenance=dict(inputs=I.hashes(paths),code=I.hashes([Path(__file__)]))))
    print('Full replay:',len(result['results']),'cases; audit passed:',result['passed'])
    print('Constructed:',sum(r['passed'] for r in rows),'/',len(rows))
    if not result['passed']:raise SystemExit('Audit failed')
if __name__=='__main__':main()
