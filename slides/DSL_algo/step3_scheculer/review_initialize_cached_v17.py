"""Audit cached single-pose searches and publish the complete object table."""
import json
from pathlib import Path
import numpy as np
from codes.precompute_objects.registry import active_objects,task_poses
from step3_scheculer import contacts as I
from step3_scheculer.initialize_cached_v17 import SCHEMA,STAGE


def main():
    rows=[];failures=[];times=[];all_cases=[]
    for name in active_objects():
        batch=json.loads((I.OUTPUTS/name/STAGE/'batch.json').read_text())
        assert batch['complete'] and len(batch['results'])==len(task_poses(name))
        cases=[]
        for pose in task_poses(name):
            folder=I.OUTPUTS/name/STAGE/pose/'step3_scheculer'
            d=I.check_report(folder/'schedule.json')
            assert d['schema']==SCHEMA and d['complete']
            assert d['load_count']==32768 and not d['load_subsampling']
            assert d['actual_triangle_force_generators_checked']
            assert d['cached_geometry_only'] and not d['step2_recomputed']
            r=d['result'];mask=np.load(folder/'coverage.npz')['mask']
            assert len(mask)==32768 and int(mask.sum())==r['covered']
            assert 1<=r['heads']<=6 and d['chains_run']<=30
            if d['passed']:
                assert mask.all() and r['common_direction_ids'] and r['common_path_components']
            else:
                assert d['chains_run']==30
                failures.append(dict(object=name,pose=pose,covered=r['covered'],missing=32768-r['covered'],heads=r['heads'],unresolved_proposals=d['numerically_unresolved_proposals']))
            item=dict(object=name,pose=pose,passed=d['passed'],heads=r['heads'],covered=r['covered'],chains=d['chains_run'],area_fraction=r['area_fraction'],seconds=d['timings']['total_seconds'])
            cases.append(item);all_cases.append(item);times.append(item['seconds'])
        rows.append(dict(object=name,passed=sum(c['passed'] for c in cases),total=len(cases),mean_seconds=float(np.mean([c['seconds'] for c in cases])),failed_poses=[c['pose'] for c in cases if not c['passed']]))
    out=I.OUTPUTS/'B'/STAGE
    summary=dict(complete=True,input_and_generator_hashes_checked=True,objects=rows,cases=all_cases,failures=failures,
                 passed=sum(r['passed'] for r in rows),total=len(all_cases),
                 timing_seconds=dict(mean=float(np.mean(times)),median=float(np.median(times)),p90=float(np.percentile(times,90)),max=max(times)),
                 scope='Single-pose contacts, all original loads, cached individual continuous head exits and common ports. No full support constructed.',
                 failure_meaning='Thirty finite random trajectories found no accepted solution; this does not prove infeasibility.')
    I.save(out/'review.json',summary)
    lines=['# Cached single-pose initialization','',summary['scope'],'',
           '| Object | Passed / total | Mean seconds | Failed poses |','|---|---:|---:|---|']
    lines += [f"| {r['object']} | {r['passed']} / {r['total']} | {r['mean_seconds']:.2f} | {', '.join(r['failed_poses']) or '—'} |" for r in rows]
    lines += ['',f"Total: {summary['passed']} / {summary['total']}.",summary['failure_meaning'],'']
    (out/'review.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in summary.items() if k!='cases'},indent=2))


if __name__=='__main__':main()
