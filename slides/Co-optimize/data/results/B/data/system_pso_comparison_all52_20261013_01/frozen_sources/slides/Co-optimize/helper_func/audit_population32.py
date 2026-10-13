"""Audit saved population traces and provenance, without physical replay."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json
import numpy as np
from PIL import Image

ROOT=Path('/home/yli581/Desktop/cadGrasp')
BASE=ROOT/'slides/Co-optimize/data/results/B'
EXPERIMENT='population32_all52_20261012_combined'
SOURCE='population32_all52_20261012_01'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',default=EXPERIMENT);parser.add_argument('--source',default=SOURCE)
    args=parser.parse_args()
    system=BASE/'data'/('system_'+args.experiment)
    source=BASE/'data'/('system_'+args.source)
    requested=json.loads((source/'requested_sets.json').read_text())
    batch=json.loads((BASE/'data'/args.experiment/'batch.json').read_text())
    soft=batch.get('options',{}).get('search_policy')=='population_soft'
    summary=json.loads((system/'step4_results.json').read_text())
    assert batch['complete'] and summary['complete']
    labels=[r['id'] for r in batch['results']]
    assert len(labels)==len(set(labels))==52 and set(labels)==set(requested['sets'])
    assert sum(r['object_pose_count'] for r in summary['cases'])==263
    inspected=[]; masks=0; initializations=0; rounds=0; events=0
    for result in batch['results']:
        label=result['id']; experiment=result.get('experiment',SOURCE)
        directory=BASE/label/'step4/step4.2'/experiment
        if result.get('status')=='runtime_failure':
            inspected.append(dict(id=label,runtime_failure=True));continue
        trace=json.loads((directory/'population32_trace.json').read_text())
        stats=result['frontier_search'];configured=stats.get('configured_rounds',batch.get('rounds',8))
        size=stats['population_size']
        assert len(trace)==stats['rounds'] and 0<=len(trace)<=configured
        if not stats.get('stop_when_feasible'):assert len(trace)==configured
        branch_counts=Counter()
        for row in trace:
            assert row['population_size']==size and row['generated_trials']==size
            assert len(row['events'])==len(row['retained'])==size
            assert sorted(x['slot']for x in row['events'])==list(range(size))
            assert sorted(x['slot']for x in row['retained'])==list(range(size))
            for event in row['events']:
                if 'error'in event:
                    assert not event['accepted'] and event['adjustment_passes'] in (0,1)
                else:assert event['adjustment_passes']==(0 if event['branch']=='hold'else 1)
                if soft:
                    assert not row['count_penalty_used']
                    probabilities=event['pose_probabilities'];owner=event['selected_pose']
                    if event['branch']=='hold':assert owner is None and sum(probabilities.values())==0
                    else:
                        assert owner not in event['protected_pose_indices'] and probabilities[str(owner)]>0
                        assert abs(sum(probabilities.values())-1)<1e-12
                        if 'trial'in event:
                            assert event.get('protected_pose_parameters_unchanged',event['successful_pose_parameters_unchanged'])
                            if not event['successful_pose_parameters_unchanged']:assert event['coupled_adjustment_enabled']
                    if event['weight_source']=='original_force_failure_fraction':
                        counts=event['before']['per_pose_failed_demands'];total=sum(counts.values())
                        assert all(abs(probabilities[k]-v/total)<1e-12 for k,v in counts.items())
                    if event.get('move',{}).get('action')=='role-exchange':
                        assert event['move']['restored'] not in event['protected_pose_indices']
                        assert event['move']['pose'] not in event['protected_pose_indices']
                branch_counts[event['branch']]+=1
                if event['branch']=='differential' and 'move' in event:
                    donors=event['move']['donor_slots']
                    assert len(set(donors))==2 and event['slot'] not in donors
            rounds+=1;events+=size
        original=json.loads((BASE/label/'step4/step4.1/data/report.json').read_text())
        assert set(result['counts'])==set(original['poses'])
        assert result['pose_count']==len(original['poses'])
        names=['discrete','differential','hold']if soft else['discrete','differential']
        assert stats['branch_counts']=={name:branch_counts[name]for name in names}
        if soft:
            assert not stats['count_penalty_used']
            assert stats['oracle_fallback_when_all_force_passed']==batch['options'].get('allow_oracle_fallback',False)
        assert stats['generated_trials']==len(trace)*size and stats['maximum_retained']==size
        assert stats['initialization_role_seeds']==0 and not stats['warm_start']
        initial=BASE/label/'step4/step4.1'/experiment
        files=sorted(initial.glob('slot_*.npz'));assert len(files)==size
        with np.load(files[0])as z: native=z['native_world_original'].copy()
        for file in files:
            with np.load(file)as z:
                active=z['active'];assert np.array_equal(z['hosts'][active],active)
                assert np.allclose(z['placements'][active],np.eye(4),atol=1e-12,rtol=0)
                assert np.array_equal(z['native_world_original'],native)
            initializations+=1
        with np.load(directory/'layout.npz')as z:
            assert np.array_equal(z['native_world_original'],native)
            for k in z['active']:
                U=z['native_world'][z['hosts'][k]]@z['placements'][k]
                assert np.allclose(U[:3,:3],native[k,:3,:3],atol=1e-10,rtol=0)
        for pose,count in result['counts'].items():
            with np.load(directory/(pose+'_force.npz'))as z:
                assert z['mask'].shape==(32768,) and int(np.count_nonzero(z['mask']))==count
                masks+=32768
        if result['passed']:
            assert result['force_passed'] and result['base_oracle_passed']
            assert result['solid_constructed_during_search']
            assert all(n==32768 for n in result['counts'].values())
            assert (directory/'support_with_base.obj').is_file()
        inspected.append(dict(id=label,rounds=len(trace),branches=dict(branch_counts),
                              passed=result['passed'],errors=len(stats.get('errors',[]))))
    for picture in summary['pictures']:
        path=ROOT/picture['path'];assert digest(path)==picture['sha256']
        with Image.open(path)as im:im.verify()
    if not any(r.get('status')=='runtime_failure'for r in batch['results']):
        assert masks==263*32768 and initializations==sum(r['frontier_search']['population_size']for r in batch['results'])
        assert events==sum(r['frontier_search']['rounds']*r['frontier_search']['population_size']for r in batch['results'])
    protected=json.loads((source/'protected_originals.json').read_text())
    changed=[p for p,h in protected.items()if not (ROOT/p).is_file() or digest(ROOT/p)!=h]
    frozen=json.loads((source/'frozen_sources.json').read_text())['files']
    changed_frozen=[r['frozen']for r in frozen if digest(ROOT/r['frozen'])!=r['frozen_sha256']]
    html=list((ROOT/'slides/Co-optimize/output/B').rglob('*.html'))
    audit=dict(complete=not changed and not changed_frozen and not html,
        groups=len(inspected),object_poses=263,initial_slot_files=initializations,
        rounds=rounds,slot_events=events,original_mask_entries=masks,
        verified_pngs=len(summary['pictures']),protected_files=len(protected),
        changed_protected_files=changed,frozen_files=len(frozen),changed_frozen_files=changed_frozen,
        html_files=[str(p.relative_to(ROOT))for p in html],cases=inspected,
        force_solver_run=False,boolean_run=False)
    (system/'saved_artifact_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    assert audit['complete'],audit
    print('SAVED ARTIFACT AUDIT',len(inspected),'groups',events,'slot events',masks,'mask entries',len(summary['pictures']),'PNGs')


if __name__=='__main__':main()
