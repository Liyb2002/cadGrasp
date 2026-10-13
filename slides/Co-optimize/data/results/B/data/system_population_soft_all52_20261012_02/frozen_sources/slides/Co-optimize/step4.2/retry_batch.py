"""Retry recorded numerical failures from the same regenerated Step4.1."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
import argparse
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor,as_completed
from co_common import *
from run_all import output_root
import optimizer
import run_batch


def retry_case(arguments):
    name,group,options,original,token=arguments
    os.environ['COOPT_WHOLE_RUN_TOKEN']=token
    began=time.monotonic()
    initial=I.check_report(output_root(name)/group['id']/'step4/step4.1/data/report.json')
    final=optimizer.run_case(name,group,dict(options,resume=False))
    row=dict(original,step42=final,passed=final['passed'],
        status='pass' if final['passed'] else 'optimization_unresolved',
        primary_attempt=dict(original),additional_numerical_retry=True,
        retry_from_regenerated_initialization=True,
        retry_seconds=time.monotonic()-began,seconds=original['seconds']+time.monotonic()-began)
    if final['passed']:
        report=I.check_report(output_root(name)/group['id']/'step4/step4.2/continuous/data/report.json')
        row.update(volume_cm3=report['volume_cm3'],baseline_retained=report['step41_baseline_retained'],
            continuous_quadrature_gate_passed=report['continuous_quadrature_gate_passed'],
            counts=report['counts'],hosts=report['hosts'])
        row.pop('error',None)
    else:row['error']=final.get('error')
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('object',nargs='?',default='B')
    parser.add_argument('--jobs',type=int,default=4);args=parser.parse_args()
    root=output_root(args.object);path=root/'data/continuous_step4_batch.json'
    batch=json.loads(path.read_text());assert batch['complete'],'Wait for the primary pool before modifying its sources or retrying'
    directory=root/'data/continuous_B_rerun_v1';directory.mkdir(exist_ok=True)
    primary=directory/'primary_batch.json'
    if not primary.exists():save(primary,batch)
    manifest=json.loads((root/'_history'/batch['experiment']/'manifest.json').read_text())
    groups=manifest['groups'];by_group={g['id']:g for g in groups};by_row={r['id']:r for r in batch['results']}
    requested=[r for r in batch['results'] if not r['passed'] and 'Magnitude interval' in r.get('error','')]
    token=json.loads((root/'data/whole_step4_active_run.json').read_text())['run_token']
    source_hashes=I.hashes(optimizer.source_files());frozen=directory/'numerical_retry_sources'
    for relative,digest in source_hashes.items():
        target=frozen/relative;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/relative).read_bytes())
    retry=dict(complete=False,requested=[r['id'] for r in requested],results=[],
        extra_search_budget_per_set=batch['options'],jobs=args.jobs,
        previous_solutions_used=False,numerical_model_unchanged=True,
        code=source_hashes)
    batch.update(complete=False,additional_numerical_retries=len(requested),primary_batch='data/continuous_B_rerun_v1/primary_batch.json')
    save(directory/'numerical_retry.json',retry);save(path,batch)
    began=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.jobs,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(retry_case,(args.object,by_group[r['id']],batch['options'],r,token)):r for r in requested}
        for future in as_completed(futures):
            original=futures[future]
            try:row=future.result()
            except Exception as error:
                row=dict(original,error=f'{type(error).__name__}: {error}',additional_numerical_retry=True)
            by_row[row['id']]=row;retry['results'].append(row)
            batch['results']=[by_row[g['id']] for g in groups]
            batch.update(passed=sum(r['passed'] for r in batch['results']),unresolved=sum(not r['passed'] for r in batch['results']),
                numerical_retries_completed=len(retry['results']),numerical_retry_seconds=time.monotonic()-began)
            save(path,batch);save(directory/'numerical_retry.json',retry)
            run_batch.publish(root,groups,batch['results'],batch)
            print('NUMERICAL RETRY',len(retry['results']),'/',len(requested),row['id'],row['status'],
                row.get('volume_cm3'),row.get('error',''),flush=True)
    for relative,digest in manifest['protected_step3_hashes'].items():assert I.sha256(ROOT/relative)==digest
    for relative,digest in manifest['original_step4_hashes'].items():assert I.sha256(ROOT/manifest['archived_paths'][relative])==digest
    retry.update(complete=True,seconds=time.monotonic()-began);save(directory/'numerical_retry.json',retry)
    batch.update(complete=True,seconds=batch['seconds']+time.monotonic()-began)
    save(path,batch);save(root/'data/continuous_step42_batch.json',dict(complete=True,
        results=batch['results'],options=batch['options'],additional_numerical_retries=retry,experiment=batch['experiment']))
    run_batch.publish(root,groups,batch['results'],batch)
    print('RETRY COMPLETE',batch['passed'],'/',len(groups),'actual passes;',batch['unresolved'],'unresolved',flush=True)


if __name__=='__main__':main()
