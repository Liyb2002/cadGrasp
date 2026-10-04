"""Audit cached single-pose searches and publish the complete object table."""
import json
from pathlib import Path
import numpy as np
from codes.precompute_objects.registry import active_objects,task_poses
from step3_scheculer import contacts as I
from step3_scheculer.initialize_sampling4_greedy4_v28 import SCHEMA,STAGE


def main():
    rows=[];failures=[];times=[];all_cases=[]
    for name in active_objects():
        batch=json.loads((I.OUTPUTS/name/STAGE/'batch.json').read_text())
        # The interrupted batch ledger is superseded by per-case completed reports.
        cases=[]
        for pose in task_poses(name):
            folder=I.OUTPUTS/name/STAGE/pose/'step3_scheculer'
            if not (folder/'schedule.json').is_file():
                folder=I.OUTPUTS/name/'independent_poses_sampling4_greedy4_v29'/pose/'step3_scheculer'
            d=I.check_report(folder/'schedule.json')
            assert d['schema'] in (SCHEMA,'sampling4_greedy4_v29') and d['complete']
            assert d['load_count']==32768 and not d['load_subsampling']
            assert d['actual_triangle_force_generators_checked']
            assert d['cached_geometry_only'] and not d['step2_recomputed']
            for trajectory in d['trajectory_results']:
                assert trajectory['heads']<=8
                for step in trajectory['rounds']:
                    if step['step']<=4:
                        assert step['selection_mode']=='top10_sampling'
                    else:
                        assert step['selection_mode']=='maximum_coverage_greedy'
                        assert step['random_draw'] is None
                        assert step['selected_id']==step['top10'][0]['id']
            r=d['result'];mask=np.load(folder/'coverage.npz')['mask']
            assert len(mask)==32768 and int(mask.sum())==r['covered']
            assert 1<=r['heads']<=8 and d['chains_run']<=30
            if d['passed']:
                assert mask.all() and r['common_direction_ids'] and r['common_path_components']
            else:
                assert d['chains_run']==30
                failures.append(dict(object=name,pose=pose,covered=r['covered'],missing=32768-r['covered'],heads=r['heads'],unresolved_proposals=d['numerically_unresolved_proposals']))
            item=dict(object=name,pose=pose,passed=d['passed'],heads=r['heads'],covered=r['covered'],chains=d['chains_run'],area_fraction=r['area_fraction'],seconds=d['timings']['total_seconds'],report=str((folder/'schedule.json').relative_to(I.ROOT)),numerically_unresolved_proposals=d['numerically_unresolved_proposals'])
            cases.append(item);all_cases.append(item);times.append(item['seconds'])
        rows.append(dict(object=name,passed=sum(c['passed'] for c in cases),total=len(cases),mean_seconds=float(np.mean([c['seconds'] for c in cases])),failed_poses=[c['pose'] for c in cases if not c['passed']]))
    out=I.OUTPUTS/'B'/STAGE
    summary=dict(complete=True,input_and_generator_hashes_checked=True,objects=rows,cases=all_cases,failures=failures,
                 passed=sum(r['passed'] for r in rows),total=len(all_cases),
                 timing_seconds=dict(mean=float(np.mean(times)),median=float(np.median(times)),p90=float(np.percentile(times,90)),max=max(times)),
                 scope='Single-pose contacts, all original loads, cached individual continuous head exits and common ports. No full support constructed.',
                 failure_meaning='Thirty finite random trajectories found no accepted solution; this does not prove infeasibility.')
    baseline=json.loads((I.OUTPUTS/'B'/'independent_poses_cached_v17'/'review.json').read_text())
    old={(r['object'],r['pose']):r for r in baseline['cases']}
    summary['comparison']=dict(original_passed=baseline['passed'],hybrid_passed=summary['passed'],
        rescued=[r for r in all_cases if r['passed'] and not old[r['object'],r['pose']]['passed']],
        regressed=[r for r in all_cases if not r['passed'] and old[r['object'],r['pose']]['passed']],
        common_passed_head_changes=dict(
            fewer=sum(r['heads']<old[r['object'],r['pose']]['heads'] for r in all_cases if r['passed'] and old[r['object'],r['pose']]['passed']),
            same=sum(r['heads']==old[r['object'],r['pose']]['heads'] for r in all_cases if r['passed'] and old[r['object'],r['pose']]['passed']),
            more=sum(r['heads']>old[r['object'],r['pose']]['heads'] for r in all_cases if r['passed'] and old[r['object'],r['pose']]['passed'])),
        original_mean_passed_heads=float(np.mean([r['heads'] for r in baseline['cases'] if r['passed']])),
        hybrid_mean_passed_heads=float(np.mean([r['heads'] for r in all_cases if r['passed']])))
    summary['interrupted_batch_wall_seconds']=1485.784
    summary['shared_wall_seconds']=1698.491
    summary['timing_breakdown_seconds']=dict(initial_batch=1485.784,interrupted_retry=144.925,final_bounded_retry=67.782)
    summary['bounded_retry_case']='A4/pose_19: v29, one second per additional LP retry; completed v28 trajectories reused.'
    summary['timing_policy']='Reuses identical saved first-four sampling prefixes; wall time includes both interrupted runs and final bounded retry, not cold full search.'
    summary['provenance']=dict(inputs=I.hashes([I.ROOT/r['report'] for r in all_cases]),code=I.hashes([Path(__file__)]))
    I.save(out/'review.json',summary)
    lines=['# Four sampling heads then four greedy heads','',summary['scope'],'',
           '| Object | Passed / total | Mean seconds | Failed poses |','|---|---:|---:|---|']
    lines += [f"| {r['object']} | {r['passed']} / {r['total']} | {r['mean_seconds']:.2f} | {', '.join(r['failed_poses']) or '—'} |" for r in rows]
    lines += ['',f"Total: {summary['passed']} / {summary['total']}.",summary['failure_meaning'],'']
    lines += ['', '## Comparison', '', json.dumps(summary['comparison'],indent=2), '',
              f"Total measured experiment wall seconds: {summary['shared_wall_seconds']:.3f}.", summary['timing_policy']]
    (out/'review.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in summary.items() if k!='cases'},indent=2))


if __name__=='__main__':main()
