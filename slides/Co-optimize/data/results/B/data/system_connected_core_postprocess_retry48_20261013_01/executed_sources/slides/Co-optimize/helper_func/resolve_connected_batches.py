"""Select certified same-layout results from explicitly recorded experiments.

Complete final supports are ranked by actual static BBX, then material.
Incomplete campaigns remain incomplete; failed rows retain their evidence.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
import trimesh
from run_dynamic_postprocess import ROOT,COOPT,BASE,save,sha
sys.path.insert(0,str(COOPT/'evaluation'))
from metrics import static_bounds


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--step4-output',required=True);p.add_argument('--postprocess-output',required=True)
    p.add_argument('--step4-batches',nargs='+',required=True)
    p.add_argument('--postprocess-batches',nargs='+',required=True)
    p.add_argument('--source-system',required=True)
    args=p.parse_args()
    request=json.loads((BASE/'data'/('system_'+args.source_system)/'requested_sets.json').read_text())
    candidates={};pipelines={};prunings={};constructions={};sources=[]
    for name in args.step4_batches:
        path=BASE/'data'/name/'batch.json';b=json.loads(path.read_text());sources.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),complete=b['complete']))
        for row in b['results']:candidates.setdefault(row['id'],{})[row.get('experiment',name)]=dict(row,experiment=row.get('experiment',name))
    for name in args.postprocess_batches:
        path=BASE/'data'/name/'batch.json'
        if not path.exists():continue
        b=json.loads(path.read_text());sources.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),complete=b['complete']))
        for row in b['results']:
            if 'stage'in row:
                if not row['passed']:continue
                if row['stage']=='step5.2':prunings[row['output']]=row
                elif row['stage']=='step5.3':constructions.setdefault(row['id'],[]).append(row)
            else:
                pipelines[row['id'],row['experiment']]=row
                if row.get('pruning',{}).get('passed'):prunings[row['pruning']['output']]=row['pruning']
                if row.get('construction',{}).get('passed'):constructions.setdefault(row['id'],[]).append(row['construction'])
    body=trimesh.load(ROOT/'objects/B/mesh.stl',force='mesh',process=False)
    selected=[];outcomes=[]
    for label in request['sets']:
        options=list(candidates.get(label,{}).values())
        if not options:continue
        choices=[]
        for construction in constructions.get(label,[]):
            folder=ROOT/construction['output'];report=json.loads((folder/'data/report.json').read_text())
            experiment=report['selected_step4_experiment']
            case=next((r for r in options if r['experiment']==experiment and r['passed']),None)
            if case is None:continue
            pruning=prunings.get(report['step52_source'])
            if pruning is None:continue
            prune=json.loads((ROOT/pruning['output']/'data/report.json').read_text())
            assert report['passed']and prune['passed']and prune['selected_step4_experiment']==experiment
            assert report['selected_step4_source']==prune['selected_step4_source']
            assert len(report['force_counts'])==case['pose_count']and all(v==32768 for v in report['force_counts'].values())
            check=report['construction']['serialized_export_verification']
            assert check['passed']and check['history'][-1]['force_passed']and check['history'][-1]['geometry_passed']
            source=ROOT/report['selected_step4_source']
            for filename,digest in prune['artifacts'].items():assert sha(ROOT/pruning['output']/filename)==digest
            fixture=trimesh.load(folder/report['support'],force='mesh',process=False)
            with np.load(source/'layout.npz')as z:layout={k:z[k].copy()for k in z.files}
            metric=static_bounds(body.vertices,layout,None,fixture.vertices,report['poses'])
            choices.append((metric['with_full_support']['volume_litres'],abs(float(fixture.volume))*1e6,case,pruning,construction))
        if choices:
            bbx,material,case,pruning,construction=min(choices,key=lambda r:r[:2])
            outcome=dict(pipelines.get((label,case['experiment']),{}),id=label,pose_count=case['pose_count'],
                experiment=case['experiment'],passed=True,status='PASS',pruning=pruning,construction=construction,
                selected_actual_bbx_l=bbx,selection_priority='actual final BBX, then material')
        else:
            passed=[r for r in options if r['passed']]
            def rank(r):
                return (not r['passed'],not r.get('force_passed'),
                    r.get('pose_count',0)*32768-sum(r.get('counts',{}).values()),r.get('seconds',float('inf')))
            case=min(passed or options,key=rank)
            outcome=dict(pipelines.get((label,case['experiment']),{}))
            outcome.update(id=label,pose_count=case['pose_count'],experiment=case['experiment'],passed=False,
                status=outcome.get('status','STEP4_FAIL'if not case['passed']else'POSTPROCESS_PENDING'))
            if not case['passed']:outcome['skipped_reason']='No certified connected Step4 contact core'
        selected.append(case);outcomes.append(outcome)
    complete=(len(selected)==len(request['sets'])and all(r['complete']for r in sources))
    source_batch=dict(complete=complete,sets=request['sets'],group_count=len(request['sets']),results=selected,
        passed=sum(r['passed']for r in selected),search_policy='connected_core',warm_start=False,
        source_system=args.source_system,source_batches=sources,selection_priority='certified final BBX')
    save(BASE/'data'/args.step4_output/'batch.json',source_batch)
    batch=dict(complete=complete,group_count=len(request['sets']),results=outcomes,
        passed=sum(r['passed']for r in outcomes),eligible_groups=sum(r['passed']for r in selected),
        source_batches=sources,selection_priority='actual final BBX, then material')
    save(BASE/'data'/args.postprocess_output/'batch.json',batch)
    save(BASE/'data'/args.postprocess_output/'resolved_batch.json',batch)
    print('RESOLVED CONNECTED',len(selected),'STEP4',source_batch['passed'],'FINAL',batch['passed'],'complete',complete)


if __name__=='__main__':main()
