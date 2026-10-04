"""Summarize actual Step5 results, original-frame comparability and value targets."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import experiment as E
from value import terminal_cost,absolute_spread
from step5_evaluate.evaluate import read_needs

def review():
    baseline=E.I.ROOT/'slides/baseline_algo/output/B';rows=[]
    for name,poses in E.specs():
        group=E.OUT/name;comparison=group/'comparison.json'
        current=json.loads(comparison.read_text()) if comparison.exists() else dict(passed=False,error='not run')
        old_path=baseline/name/'step5_evaluate/report.json'
        old=json.loads(old_path.read_text()) if old_path.exists() else None
        row=dict(group=name,passed=current['passed'],baseline_available=old is not None)
        if old:row['baseline_volume_cm3']=old['metrics']['object_and_support_poses']['box_volume_cm3']
        if current['passed']:
            report=E.I.check_report(group/'step4/data/growing_support/report.json');step5=E.I.check_report(group/'step5_evaluate/report.json')
            metric=step5['metrics'];V=metric['object_and_support_poses']['box_volume_cm3']
            row.update(volume_cm3=V,material_volume_cm3=report['volume_cm3'],extra_volume_ratio=terminal_cost(step5,True),
                absolute_object_exits=report['placement']['absolute_object_exit'],absolute_exit_spread=absolute_spread(report['placement']['absolute_object_exit']),
                force_normal_alignment=report['direction_objective']['force_alignment'],
                head_count_reported_only=sum(map(len,report['direction_objective']['selected_ids'])),head_count_penalty=0,
                full_sample_certificates=[dict(passed=c['passed'],sample_count=c['sample_count'],covered_count=c['covered_count']) for c in report['direction_objective']['force_certificates']])
            if old:
                old_source=baseline/name/'step4/data/growing_support/report.json'
                if not old_source.exists():old_source=baseline/name/'step4/data/report.json'
                old_report=json.loads(old_source.read_text());new_meshes=[];old_meshes=[]
                for pose in poses:
                    _,a=read_needs(group,pose,report);_,b=read_needs(baseline/name,pose,old_report)
                    if not np.array_equal(a.vertices,b.vertices) or not np.array_equal(a.faces,b.faces):raise AssertionError(f'Different baseline geometry: {name}/{pose}')
                new_tasks=E.read_problems(name,poses)
                old_tasks=E.HIST.read_reference().tasks if name in ['pose1+3','pose1+3copied'] else [E.read_task('B',p,folder=baseline/name/'step3_scheculer/independent_poses_floor2mm'/p/'step_1_needs') for p in poses]
                if not all(np.array_equal(a.targets,b.targets) and np.array_equal(a.floor,b.floor) and np.array_equal(a.scale,b.scale) for a,b in zip(new_tasks,old_tasks)):raise AssertionError('Changed original loads/floor: '+name)
                row['original_loads_and_floor_identical']=True
                old_V=old['metrics']['object_and_support_poses']['box_volume_cm3']
                row.update(baseline_volume_cm3=old_V,change_percent=100*(V/old_V-1),geometry_and_workstation_frame_identical=True)
        else:row['error']=current.get('error','unknown')
        rows.append(row)
    compared=[r for r in rows if r['passed'] and r['baseline_available']]
    result=dict(complete=True,total_sets=len(rows),passed_sets=sum(r['passed'] for r in rows),baseline_sets=10,
        completed_baseline_sets=len(compared),baseline_improvements=sum(r['change_percent']<0 for r in compared),
        aggregate_baseline_volume_cm3=sum(r['baseline_volume_cm3'] for r in compared),aggregate_new_volume_cm3=sum(r['volume_cm3'] for r in compared),
        missing_previous_comparator=[r['group'] for r in rows if not r['baseline_available']],
        all_relative_improvements_use_identical_saved_geometry=True,new_neural_network_trained=False,old_weights_unchanged=True,groups=rows)
    labels=[]
    for row in rows:
        if not row['passed']:continue
        group=E.OUT/row['group'];source=group/'step4/data/growing_support/report.json';report=E.json.loads(source.read_text())
        labels.append(dict(objective_revision='absolute_world_direction_step5_volume_v1',group=row['group'],poses=report['poses'],
            state=dict(selected_ids=[[] for _ in report['poses']]),successful_completion=dict(selected_ids=report['direction_objective']['selected_ids'],placement=report['placement']),
            best_observed_continuation_cost=row['extra_volume_ratio'],terminal_step5_volume_cm3=row['volume_cm3'],head_count_penalty=0,
            value_source='actual accepted Step4 and unchanged Step5; not a neural prediction',global_optimum_claim=False,
            provenance=E.I.hashes([source,group/'step5_evaluate/report.json'])))
    data=E.HERE/'data';data.mkdir(exist_ok=True)
    (data/'successful_completion_targets.jsonl').write_text(''.join(E.json.dumps(r,ensure_ascii=False)+'\n' for r in labels))
    if compared:result['aggregate_change_percent']=100*(result['aggregate_new_volume_cm3']/result['aggregate_baseline_volume_cm3']-1)
    snapshot=E.json.loads((E.HERE/'protected_snapshot.json').read_text())
    if any(E.I.sha256(E.I.ROOT/path)!=digest for path,digest in snapshot.items()):raise AssertionError('Protected original source/output/checkpoint changed')
    result['protected_source_metric_and_checkpoint_audit']=dict(passed=True,file_count=len(snapshot))
    E.save(E.HERE/'results.json',result)
    text=['# Absolute direction: actual Step4 / Step5 results','','The search uses common workstation +Z object withdrawal, jointly chooses XY seating/contact compatibility, and assigns no head-count cost. Reported volume is the unchanged Step5 aggregate XYZ bounding-box volume, not support material volume.','','| Set | Previous occupied cm3 | New occupied cm3 | Change | Status |','|---|---:|---:|---:|---|']
    for r in rows:
        old=f"{r.get('baseline_volume_cm3',0):.1f}" if r.get('baseline_volume_cm3') is not None else '—'
        new=f"{r['volume_cm3']:.1f}" if r['passed'] else '—'
        change=f"{r['change_percent']:+.1f}%" if 'change_percent' in r else '—'
        link=f"[passed](output/B/{r['group']}/step4/overview.png)" if r['passed'] else 'bounded search failed'
        text.append(f"| {r['group']} | {old} | {new} | {change} | {link} |")
    text+=['',f"Completed supports: {result['passed_sets']}/{len(rows)}. Improvements among comparable baseline sets: {result['baseline_improvements']}/{len(compared)}.",'','Comparisons use byte-equivalent saved mesh vertices/faces and unchanged workstation metric/frame. Contact choices, seating translations and construction are jointly changed, so the volume change is evidence for the combined method, not an isolated causal estimate for normal alignment. Additional transfer sets have no constructed previous comparator.','', 'Only +Z and a finite XY search are tested; results are neither a global optimum nor generalization evidence for a newly trained network. The old N_remaining/object-frame network has not been retrained for this target. Full mechanics comes from each original 32,768-load Step3 certificate; Step4 performs one full construction acceptance and Step5 only measures the accepted model. No independent export replay is claimed.','', '[Value definition](../method/method.md) · [Machine-readable results](results.json) · [Comparison figure](comparison.png)']
    (E.HERE/'REPORT.md').write_text('\n'.join(text)+'\n')
    if compared:
        names=[r['group'] for r in compared];x=np.arange(len(names));fig,ax=plt.subplots(figsize=(13,6));ax.bar(x-.18,[r['baseline_volume_cm3'] for r in compared],.36,label='Previous baseline');ax.bar(x+.18,[r['volume_cm3'] for r in compared],.36,label='World direction + joint contacts')
        ax.set_xticks(x,names,rotation=35,ha='right');ax.set_ylabel('Step5 occupied XYZ volume (cm3)');ax.set_title('Actual constructed supports; same saved objects and workstation axes');ax.legend();fig.tight_layout();fig.savefig(E.HERE/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k!='groups'},indent=2))
if __name__=='__main__':review()
