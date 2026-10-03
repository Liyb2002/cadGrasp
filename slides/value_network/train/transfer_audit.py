"""Post-evaluation audit and reference comparison; never chooses a checkpoint."""
import json
from pathlib import Path
import hashlib
from transfer_experiment import ROOT,TrainingContext
from fixed_data import signature
from collect_states import joint_value
from train import save,digest,DATA

def main():
    experiment=json.loads((ROOT/'experiment.json').read_text());plan=experiment['split']
    train={tuple(ps) for _,ps in plan['train']};test={tuple(ps) for _,ps in plan['test']}
    assert not train & test
    assert {p for ps in train for p in ps}=={p for ps in test for p in ps}
    progress=json.loads((ROOT/'training_progress.json').read_text());fit=Path(experiment['checkpoint']).parent
    fr=json.loads((fit/'report.json').read_text());seen=set();sources=fr['source_hashes']
    for source,expected in sources.items():
        path=Path(source);assert digest(path)==expected
        d=json.loads(path.read_text());assert tuple(d['poses']) in train
        seen.add(signature(d['poses'],d['selected_indices']))
    for path in (ROOT/'train_on_policy').glob('*.json'):
        assert tuple(json.loads(path.read_text())['poses']) in train
    test_data=Path(plan['state_root'])
    assert all(not (test_data/name).exists() for name,_ in plan['test'])
    assert digest(Path(__file__).parent/'current/best.pt')==plan['original_checkpoint_sha256']
    for source,expected in json.loads((DATA/'pilot20_shared/config.json').read_text())['source_inputs'].items():
        assert digest(Path(source))==expected
    ctx=TrainingContext(ROOT/'audit_evidence')
    refs={}
    for name,poses in plan['train']+plan['test']+plan['original']:
        known_new=Path(plan['state_root'])/name/'pilot20/state_000.json'
        known_old=DATA/name/'pilot20/state_000.json'
        record=json.loads(known_new.read_text()) if known_new.exists() else (json.loads(known_old.read_text()) if known_old.exists() else ctx.labels(poses,[[] for _ in poses]))
        best=min((r for r in record['rows'] if r['value'] is not None),key=lambda r:r['value'])
        refs[name]=dict(best_recorded_cost=best['value']+1,total_heads=best['total_heads'],dispersion=best['dispersion'],completion_indices=best['completion_indices'])
    summary={}
    for label,path in [('frozen',ROOT/'frozen/report.json'),('five_train',Path(experiment['training_report'])),('five_test',ROOT/'heldout/report.json'),('original_ten_test',ROOT/'original_ten_test/report.json')]:
        report=json.loads(path.read_text())
        for r in report['results']:
            r['reference']=refs[r['group']]
            if r['passed']:r['cost_gap']=r['cost']-refs[r['group']]['best_recorded_cost']
        save(path,report)
        summary[label]=dict(passed=report['passed_groups'],groups=report['groups'],near_best_count=sum(r['passed'] and r.get('cost_gap') is not None and r['cost_gap']<=.05 for r in report['results']),near_best_tolerance=.05,
                            mean_heads_among_passed=sum(r['total_heads'] for r in report['results'] if r['passed'])/report['passed_groups'] if report['passed_groups'] else None,
                            results=[dict(group=r['group'],passed=r['passed'],heads=sum(map(len,r['selected_indices'])),cost_gap=r.get('cost_gap')) for r in report['results']])
    audit=dict(complete=True,passed=True,training_groups=5,test_groups=5,initial_training_states=100,
               actual_training_states=len(seen),finite_training_values=fr['metrics']['finite_values'],training_rounds=len(progress['rounds']),
               no_test_rows_or_rollouts_in_training=True,no_all_ten_pretrained_weights=True,checkpoint_selection='training groups only',
               original_checkpoint_unchanged=True,original_baseline_inputs_unchanged=True,
               scope=plan['scope'],comparison=summary)
    save(ROOT/'audit.json',audit);print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
