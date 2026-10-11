"""Recover the batch ledger and finish unchanged own-state repair / Step5.

All 52 cold numerical workers completed. The first entry point then failed
while reading a counts dictionary; neither optimization nor inputs are rerun.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import shutil
import sys
import time

PROOF = Path(__file__).resolve().parent
sys.path.insert(0,str(PROOF))
import run_experiment as W
C, BASE, ROOT = W.C, W.BASE, W.ROOT


def main():
    if (PROOF/'summary.json').exists():
        raise RuntimeError('The completed batch is protected')
    spec = json.loads((PROOF/'experiment.json').read_text())
    original_progress = json.loads((PROOF/'progress.json').read_text())
    previous = {r['case']['id']:r for r in original_progress['step5_results']}
    history = PROOF/'history'
    shutil.copy2(PROOF/'progress.json',history/'interrupted_progress.json')
    directory = BASE/'data/batch_recovery_v1'
    directory.mkdir(exist_ok=False)
    code = C.freeze(directory)
    manifest = directory/'execution.json'
    C.save(manifest,dict(code=code,cold_workers_rerun=False,
        problem='first ledger reader iterated pose keys instead of counts values',
        original_ledger_source='run_experiment.py',primary_workers_completed=52))
    token = json.loads((BASE/'data/whole_step4_active_run.json').read_text())['run_token']
    reader = W.S51.NativeInputs('B',W.trimesh.load(ROOT/'objects/B/mesh.stl',force='mesh',process=False))
    renderer = W.S51.V.Renderer()
    record = dict(complete=False,step4_attempts=[],step5_results=[],repair_groups=[],
                  ledger_recovered=True,cold_workers_rerun=False)
    began = time.monotonic()

    def progress():
        from datetime import datetime,timezone
        started = datetime.fromisoformat(spec['started_utc'])
        record['seconds'] = (datetime.now(timezone.utc)-started).total_seconds()
        C.save(PROOF/'progress.json',record)

    for group in spec['groups']:
        label = group['id']
        out = BASE/label/'step4/step4.2/proof_cold_v1'
        result = json.loads((out/'run_result.json').read_text())
        assert result['complete']
        result.update(experiment='proof_cold_v1',phase='cold')
        record['step4_attempts'].append(result)
        if result['passed'] and result.get('mesh_exported'):
            if label in previous:
                record['step5_results'].append(previous[label])
            else:
                record['step5_results'].append(W.stage5(result,reader,renderer,code,manifest))
        else:
            report = json.loads((out/'search_report.json').read_text())
            counts = report.get('counts',{})
            values = counts.values() if isinstance(counts,dict) else counts
            if any(int(n)<32768 for n in values) and (out/'sampled_layout.npz').exists():
                record['repair_groups'].append(label)
        progress()
    C.save(BASE/'data/proof_cold_v1/batch.json',dict(complete=True,groups=[g['id'] for g in spec['groups']],
        results=list(record['step4_attempts']),ledger_recovered=True,cold_workers_rerun=False))
    options = dict(output_name='proof_own_repair_v1',iterations=8,demand_nodes=32,
                   worker_cpus=4,resume_experiment='proof_cold_v1')
    with ProcessPoolExecutor(max_workers=min(6,len(record['repair_groups']) or 1),
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(W.worker,('repair',label,options,token)) for label in record['repair_groups']]
        for future in as_completed(futures):
            result = future.result()
            record['step4_attempts'].append(result)
            if result['passed'] and result.get('mesh_exported'):
                record['step5_results'].append(W.stage5(result,reader,renderer,code,manifest))
            progress()
    C.save(BASE/'data/proof_own_repair_v1/batch.json',dict(complete=True,groups=record['repair_groups'],
        options=options,results=[r for r in record['step4_attempts'] if r['phase']=='repair']))
    protected = json.loads((PROOF/'protected_existing.json').read_text())
    for path,expected in protected.items():
        if not (ROOT/path).is_file() or C.digest(ROOT/path)!=expected:
            raise RuntimeError(f'Protected original changed: {path}')
    for path,expected in code.items():
        assert C.digest(ROOT/path)==expected and C.digest(directory/'executed_sources'/path)==expected,path
    selections = [r['case'] for r in record['step5_results']]
    C.save(PROOF/'selection.json',dict(cases=selections,fresh_attempts_only=True))
    record.update(complete=True,step4_passed_count=len(selections),protected_existing_count=len(protected),
                  protected_existing_unchanged=True,algorithm_changed=False)
    progress()
    C.save(PROOF/'summary.json',record)
    C.save(directory/'verification.json',dict(complete=True,code=code,
        source_snapshots_match=True,original_inputs_and_results_unchanged=True))
    print('ALL COMPLETE',len(selections),'/ 52 fresh force passes; saved cold workers preserved',flush=True)


if __name__=='__main__':
    main()
