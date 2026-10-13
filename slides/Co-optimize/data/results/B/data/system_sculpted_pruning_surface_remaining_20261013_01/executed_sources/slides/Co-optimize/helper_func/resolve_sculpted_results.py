"""Select only actually certified current-campaign stages for all52 layouts."""
from pathlib import Path
import argparse,json
from run_dynamic_postprocess import ROOT,BASE,sha,save


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',required=True)
    p.add_argument('--sources',nargs='+',required=True);args=p.parse_args()
    system=BASE/'data'/('system_'+args.experiment);selection=json.loads((system/'selection.json').read_text())['results']
    chosen={};phases=[]
    for name in args.sources:
        path=BASE/'data'/name/'batch.json';batch=json.loads(path.read_text())
        phases.append(dict(experiment=name,path=str(path.relative_to(ROOT)),sha256=sha(path),
            complete=batch['complete'],finished=len(batch['results']),passed=batch['passed']))
        for row in batch['results']:
            if not row['passed']:continue
            for key in ['pruning','construction']:
                directory=ROOT/row[key]['output'];report=json.loads((directory/'data/report.json').read_text())
                assert report['passed']and report['selected_step4_experiment']==row['experiment']
                for artifact,h in report.get('artifacts',{}).items():assert sha(directory/artifact)==h
            chosen[row['id']]=dict(row,postprocess_experiment=name)
    required={c['id']for c in selection};missing=sorted(required-set(chosen));assert not missing,missing
    assert len(chosen)==len(required)==52
    result=dict(complete=True,group_count=52,passed=52,results=[chosen[c['id']]for c in selection],
        source_phases=phases,initial_phase_partial_or_failed=True,step4_rerun=False,step51_rerun=False)
    save(BASE/'data'/args.experiment/'resolved_batch.json',result)
    save(system/'final_selection.json',result)
    print('RESOLVED ALL52',52,'PASS;',len(phases),'source phases',flush=True)


if __name__=='__main__':main()
