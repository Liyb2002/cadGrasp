"""Select only certified, same-layout postprocessing results from saved attempts."""
from pathlib import Path
import argparse, copy, json
from run_dynamic_postprocess import BASE, ROOT, save, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('experiment')
    args=parser.parse_args();directory=BASE/'data'/args.experiment
    original_path=directory/'batch.json';original=json.loads(original_path.read_text())
    assert original['complete']
    batches=[]
    for path in sorted((BASE/'data').glob(args.experiment+'_*/batch.json')):
        batch=json.loads(path.read_text());assert batch['complete'],path
        batches.append((path,batch))
    prunings={};constructions={}
    for row in original['results']:
        if row.get('pruning',{}).get('passed'):prunings[row['pruning']['output']]=row['pruning']
        if row.get('construction',{}).get('passed'):constructions.setdefault(row['id'],[]).append(row['construction'])
    for path,batch in batches:
        for row in batch['results']:
            if not row['passed']:continue
            row=dict(row,retry_batch=str(path.relative_to(ROOT)))
            if row['stage']=='step5.2':prunings[row['output']]=row
            elif row['stage']=='step5.3':constructions.setdefault(row['id'],[]).append(row)
    results=[]
    for previous in original['results']:
        row=copy.deepcopy(previous)
        if row['status']=='STEP4_FAIL':results.append(row);continue
        choices=[]
        for construction in constructions.get(row['id'],[]):
            folder=ROOT/construction['output'];report=json.loads((folder/'data/report.json').read_text())
            assert report['passed']and report['selected_step4_experiment']==row['experiment']
            pruning=prunings[report['step52_source']]
            prune_report=json.loads((ROOT/pruning['output']/'data/report.json').read_text())
            assert prune_report['passed']and prune_report['selected_step4_experiment']==row['experiment']
            assert len(report['force_counts'])==row['pose_count']and all(v==32768 for v in report['force_counts'].values())
            check=report['construction']['serialized_export_verification']
            assert check['passed']and check['history'][-1]['force_passed']and check['history'][-1]['geometry_passed']
            choices.append((report['construction']['material_cm3'],construction,pruning))
        if choices:
            _,construction,pruning=min(choices,key=lambda choice:choice[0])
            row.update(passed=True,status='PASS',construction=construction,pruning=pruning)
        results.append(row)
    resolved=dict(original,results=results,passed=sum(r['passed']for r in results),
        original_single_batch_passed=original['passed'],bounded_postprocessing_retries=True,
        source_batches=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p))for p in [original_path]+[p for p,b in batches]])
    save(directory/'resolved_batch.json',resolved)
    print('RESOLVED POSTPROCESS',len(results),'PASS',resolved['passed'],'original',original['passed'])


if __name__=='__main__':main()
