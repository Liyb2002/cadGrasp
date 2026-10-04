"""Audit saved acceptance/provenance and compare measured, constructed supports."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'absolute_direction'))
import experiment as E


def main():
    training=json.loads((HERE/'training.json').read_text())
    assert E.I.sha256(HERE/'best.pt')==training['checkpoint_sha256']
    E.I.check_hashes(training['provenance'])
    previous=json.loads((E.HERE/'results.json').read_text());lookup={r['group']:r for r in previous['groups']}
    rows=[]
    for name,poses in E.specs():
        group=HERE/'output/B'/name;path=group/'comparison.json'
        comparison=json.loads(path.read_text()) if path.exists() else {}
        row=dict(group=name,passed=False,search_volume_cm3=lookup[name]['volume_cm3'],baseline_volume_cm3=lookup[name].get('baseline_volume_cm3'))
        if comparison.get('passed'):
            report=E.I.check_report(group/'step4/data/growing_support/report.json')
            metric=E.I.check_report(group/'step5_evaluate/report.json')
            E.I.check_report(group/'step4/data/visualization.json')
            plan=json.loads((group/'step3/plan.json').read_text())
            assert report['passed'] and not plan['teacher_plan_loaded_at_inference']
            for cert in plan['force_certificates']:assert cert['passed'] and cert['covered_count']==cert['sample_count']==32768
            actions=[s['candidate'] for s in plan['neural_action_trace']]
            assert len(actions)==len(set(actions))
            assert set(actions)=={i for r in plan['selected_ids'] for i in r}
            assert len(list((group/'step4').glob('*.png')))==2
            volume=metric['metrics']['object_and_support_poses']['box_volume_cm3']
            row.update(passed=True,volume_cm3=volume,change_from_search_percent=100*(volume/row['search_volume_cm3']-1),
                change_from_baseline_percent=100*(volume/row['baseline_volume_cm3']-1) if row['baseline_volume_cm3'] else None,
                heads_reported_only=len(actions),predicted_cost_first_action=plan['neural_action_trace'][0]['predicted_cost'],
                actual_cost=volume/metric['metrics']['object_poses']['box_volume_cm3']-1,
                full_load_samples=sum(c['sample_count'] for c in plan['force_certificates']),
                source_hashes=E.I.hashes([group/'step3/plan.json',group/'step4/data/growing_support/report.json',group/'step5_evaluate/report.json']))
            if abs(row['change_from_search_percent'])<1e-10:row['change_from_search_percent']=0.
        else:row['error']=comparison.get('error','No completed result')
        rows.append(row)
    comparable=[r for r in rows if r['passed'] and r['baseline_volume_cm3']]
    summary=dict(complete=True,sets=len(rows),passed=sum(r['passed'] for r in rows),groups=rows,
        neural_network_trained=True,neural_network_used_in_step3=True,teacher_plans_loaded_at_inference=False,
        evaluation_scope='same 20 training groups; memorization and fresh construction, not unseen-group generalization',
        baseline_improvements=sum(r['change_from_baseline_percent']<0 for r in comparable),
        complete_original_load_evaluations=sum(r.get('full_load_samples',0) for r in rows),
        training_and_inference_source_hashes=E.I.hashes([HERE/'fit.py',HERE/'model.py',HERE/'run.py',HERE/'batch.py',HERE/'best.pt',HERE/'training.json']),
        aggregate_baseline_change_percent=100*(sum(r['volume_cm3'] for r in comparable)/sum(r['baseline_volume_cm3'] for r in comparable)-1) if comparable else None)
    E.save(HERE/'results.json',summary)
    lines=['# Neural absolute-volume Step3: actual Step4 / Step5','',
        'A newly trained network predicts witnessed-completion evidence, state/action final-volume cost and XY seating. Runtime does not retrieve contact/layout choices from the successful-completion training records. Historical reference files are opened only by the original task loader; their contact/layout choices are discarded. Every accepted model is constructed afresh and passes the original full-load and full-body acceptance. Step5 measures its actual occupied XYZ bounding-box volume, not material volume.','',
        'All 20 groups were included in training. This measures memorization and independent construction, not unseen-group generalization. There is one successful completion per group; actions that belong to that completion share its witnessed cost. Unsearched actions have no fake failed-volume labels. Evidence rejection means not demonstrated, not physically impossible. Within regression uncertainty, a separately learned ordering breaks value ties without a head-count penalty.','',
        '| Set | Search cm3 | Neural cm3 | vs search | vs original baseline | Support |','|---|---:|---:|---:|---:|---|']
    for r in rows:
        if r['passed']:
            base=f"{r['change_from_baseline_percent']:+.1f}%" if r['baseline_volume_cm3'] else '—'
            lines.append(f"| {r['group']} | {r['search_volume_cm3']:.1f} | {r['volume_cm3']:.1f} | {r['change_from_search_percent']:+.1f}% | {base} | [accepted](output/B/{r['group']}/step4/overview.png) |")
        else:lines.append(f"| {r['group']} | {r['search_volume_cm3']:.1f} | — | — | — | failed: {r['error']} |")
    lines.extend(['',f"Accepted fresh supports: {summary['passed']}/{summary['sets']}.",
                  f"Original-baseline comparisons: {summary['baseline_improvements']}/{len(comparable)} improve; aggregate occupied volume changes {summary['aggregate_baseline_change_percent']:+.1f}%. These gains reproduce the preceding geometric-search solutions, rather than a new improvement from training.",
                  '', '[Training diagnostics](training.json) · [Machine-readable results](results.json) · [Volume comparison](comparison.png)'])
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    if comparable:
        x=np.arange(len(comparable));fig,ax=plt.subplots(figsize=(16,6))
        ax.bar(x-.25,[r['baseline_volume_cm3'] for r in comparable],.25,label='Original baseline')
        ax.bar(x,[r['search_volume_cm3'] for r in comparable],.25,label='World-direction search')
        ax.bar(x+.25,[r['volume_cm3'] for r in comparable],.25,label='Trained network + fresh construction')
        ax.set_xticks(x,[r['group'] for r in comparable],rotation=38,ha='right')
        ax.set_ylabel('Step5 occupied XYZ volume (cm3)');ax.legend()
        ax.set_title('Actual constructed supports on training groups');fig.tight_layout();fig.savefig(HERE/'comparison.png',dpi=150);plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k!='groups'}),flush=True)


if __name__=='__main__':main()
