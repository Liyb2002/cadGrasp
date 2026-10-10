"""Check one restored cold-start jump without restarting the stopped batch."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
import json
import os
import shutil
import time
import numpy as np
from co_common import ROOT, I, save, HERE
from whole_search.fast_search import FastModel
from whole_search.model import Model
from whole_search.search import failed_count
from whole_pipeline import load_layout
from continuous_support.boundary_objective import ProductionObjective
from juxtapose.reuse_search import juxtapose
from whole_step4_render import render_search
import optimizer_reuse


def main():
    root=HERE/'output/B';label='pose1+3+7+10+14+18+23+27';base=root/label
    out=base/'step4/step4.2/reuse_jump_probe_v1';out.mkdir(parents=True,exist_ok=False)
    source=base/'step4/step4.1';started=time.monotonic()
    generation=root/'data/whole_step4_active_run.json';previous=json.loads(generation.read_text())
    save(out/'previous_generation.json',previous);token=os.urandom(16).hex()
    os.environ['COOPT_WHOLE_RUN_TOKEN']=token
    save(generation,dict(run_token=token,status='checking_one_restored_jump',experiment='reuse_jump_probe_v1',
                         old_52_case_batch_remains_stopped=True,v5_batch_remains_stopped=True))
    files=optimizer_reuse.source_files()+[Path(__file__)];hashes=I.hashes(files);aliases={}
    for relative in hashes:
        target=out/'executed_sources'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,target);aliases[relative]=str(target.relative_to(ROOT))
    save(out/'source_manifest.json',dict(code=hashes,snapshots=aliases))
    report=dict(complete=False,passed=False,initialization='saved Step4.1',old_answers_used_as_start=False,
                scope='one cold-start Juxtapose operation, not ten-set optimization')
    try:
        poses=I.check_report(source/'data/report.json')['poses']
        model=FastModel(poses,'B',initialization_report=base/'step3/step3.1/data/report.json')
        model.worker_program=HERE/'helper_func/whole_search/exact_worker.py'
        model.checkpoint_dir=out/'data/exact_states'
        layout=load_layout(source/'layout.npz');objective=ProductionObjective(model,layout,1,1)
        initial=objective.evaluate(layout)
        result,decision=juxtapose(objective,initial,budget=96,full_budget=6,max_refined=3)
        report.update(decision=decision,coarse_covered=objective.covered(result),
                      search_seconds=time.monotonic()-started)
        np.savez_compressed(out/'proposed_layout.npz',placements=result.layout.placements,
            directions=result.layout.directions,hosts=result.layout.hosts,active=np.asarray(result.layout.active))
        print('SELECTED',decision.get('selected'),'coarse feasible',objective.covered(result),flush=True)
        save(out/'probe.json',report)
        if not objective.covered(result):return
        checked=model.exact(result.layout)
        passed=failed_count(checked)==0 and all(row['passed'] for row in checked.get('actual_work_surface_checks',[]))
        report.update(passed=passed,actual_counts=checked['counts'],actual_volume_cm3=checked['volume_cm3'])
        if passed:
            optimizer_reuse.save_process(model,[(initial,'step4.1',{}),(result,'juxtapose',decision)],1,out)
            published=Model.save(model,checked,out,dict(strategy='restored_juxtapose_probe',passed=True,
                old_answers_used_as_start=False,original_loads_reused=True,options=dict(screen_budget=96,full_budget=6),
                numerical_search_scope='one operation, not a whole-ten-set benchmark'))
            published['provenance']['code'].update({aliases[p]:h for p,h in hashes.items()})
            save(out/'report.json',published)
            data=published.copy();data['artifacts']={'../'+k:v for k,v in published['artifacts'].items()}
            save(out/'data/report.json',data);render_search(model,out,published)
            I.check_report(out/'data/report.json')
        print('ACTUAL CHECK',passed,checked['counts'],checked['volume_cm3'],flush=True)
    except Exception as error:
        report['error']=str(error);print('UNRESOLVED',str(error),flush=True)
    finally:
        report.update(complete=True,seconds=time.monotonic()-started)
        save(out/'probe.json',report)
        save(generation,dict(run_token=token,status='code_repair_probe_complete',experiment='reuse_jump_probe_v1',
                             old_52_case_batch_remains_stopped=True,v5_batch_remains_stopped=True))


if __name__=='__main__':main()
