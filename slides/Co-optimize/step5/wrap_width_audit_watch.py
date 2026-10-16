"""Supplement frozen numerical audits with the current verified width kernel."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import subprocess
import sys
import time
from wrap_pipeline import ROOT,COOPT,read,save,sha,freeze,DEFAULT_EXPERIMENT

FILES=['wrap_width.py','wrap_width_profiles.py','wrap_neck_width.py','wrap_dimensions.py','wrap_width_audit.py']


def archive_for(directory):
    for serial in range(1,100):
        archive=directory/('width_review_sources_'+str(serial).zfill(2))
        if not (archive/'index.json').exists():freeze(archive);return archive
        index=read(archive/'index.json')
        if all(index.get('slides/Co-optimize/step5/'+p)==sha(COOPT/'step5'/p) for p in FILES):return archive
    raise ValueError('Too many review-source revisions')


def binding(out):return {p:sha(out/p) for p in ['support.obj','initial_support.obj','states.json','data/width_features.json']}


def valid(out,archive):
    path=out/'data/latest_width_audit.json'
    if not path.exists():return False
    r=read(path);index=read(archive/'index.json')
    return (r['passed'] and r['artifacts']==binding(out)
        and all(r['semantic_sources'].get(p)==index['slides/Co-optimize/step5/'+p] for p in FILES))


def invoke(directory,row,archive,cpus):
    out=ROOT/row['output']
    if valid(out,archive):return read(out/'data/latest_width_audit.json')
    payload=read(directory/'payloads'/(row['id'].replace('/','__')+'.json'))
    payload.update(action='width_audit',source_snapshot=str(archive),cpus=cpus)
    path=directory/'payloads'/(row['id'].replace('/','__')+'.width_audit.json');save(path,payload)
    log=directory/'logs'/(row['id'].replace('/','__')+'.width_audit.log')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    with log.open('w') as handle:
        r=subprocess.run([sys.executable,str(COOPT/'step5/wrap_frozen_worker.py'),str(path)],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
    if r.returncode:raise ValueError('Latest width audit failed: '+row['id'])
    return read(out/'data/latest_width_audit.json')


def main():
    experiment=sys.argv[1] if len(sys.argv)>1 else DEFAULT_EXPERIMENT
    directory=COOPT/'data/results/B/data'/('step5_'+experiment);archive=archive_for(directory)
    seen=set();futures={};errors=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        while True:
            for path in (directory/'progress').glob('*.json'):
                row=read(path)
                if row['passed'] and row['id'] not in seen:
                    seen.add(row['id']);futures[pool.submit(invoke,directory,row,archive,[6+len(seen)%2])]=row['id']
            for future,label in list(futures.items()):
                if future.done():
                    try:print('LATEST_WIDTH_PASS',future.result()['id'],flush=True)
                    except Exception as e:errors.append(dict(id=label,error=str(e)));print('LATEST_WIDTH_FAILED',label,flush=True)
                    del futures[future]
            if (directory/'summary.json').exists() and read(directory/'execution.json').get('complete') and not futures:
                if len(seen)>=read(directory/'summary.json')['passed']:break
            time.sleep(3)
    save(directory/'latest_width_scheduler.json',dict(complete=True,attempted=len(seen),errors=errors,source_snapshot=str(archive.relative_to(ROOT))))


if __name__=='__main__':main()
