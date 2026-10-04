"""Compare actual constructed cavity search with previous direction finals."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
import trimesh
from step3_scheculer import cavity_dsl as C
from step3_scheculer.review_operation_dsl import load
F=C.F

def main():
    C.activate();rows=[];inputs=[]
    for group in sorted((F.I.OUTPUTS/'B').glob('pose*+*')):
        stage=group/'step3_scheculer/dsl_cavity';r=F.I.check_report(stage/'report.json')
        tasks=[C.saved_task(group,p) for p in r['poses']];checker=F.Checker(tasks)
        initial,_=load(stage/'initial',tasks);final,_=load(stage/'final',tasks)
        for state in (initial,final):
            check=checker.check(state);assert check['passed'] and all(v['covered']==32768 for v in check['per_pose']);F.roots(tasks,state)
        source=group/'step3_scheculer'/('dsl_absolute_floor' if (group/'step3_scheculer/dsl_absolute_floor/report.json').exists() else 'dsl_absolute_refined')/'final'
        baseline=C.A.M.evaluate(group,tasks,initial,source/'shape.obj',source/'report.json',stage/'baseline_step5',render=False)
        assert np.isclose(baseline['metrics']['object_and_support_poses']['box_volume_cm3'],r['initial_volume_cm3'])
        best=r['initial_volume_cm3'];count=0
        for e in r['events']:
            if e['operation']!='measured_trial':continue
            path=F.I.ROOT/e['witness'];body=F.I.check_report(path);state,_=load(path.parent,tasks)
            value=C.A.M.volume(tasks,state,trimesh.load(path.parent/'shape.obj',force='mesh',process=False))
            assert np.isclose(value,e['volume_cm3']) and np.isclose(best,e['best_before_cm3'])
            assert body['construction']['minimum_branch_thickness']['all_complete_cores_preserved']
            assert len(body['timing_seconds']['validation_passes'])==1
            assert body['validation_policy']['export_recheck'] is False
            if e['accepted']:assert value<best;best=value;count+=1;assert checker.check(state)['passed']
            inputs.append(path)
        assert np.isclose(best,r['final_volume_cm3'])
        public=F.I.check_report(group/'step4/report.json');metrics=F.I.check_report(group/'step5_evaluate/report.json')
        witness=F.I.ROOT/r['witness'];F.I.check_report(witness)
        for n in ('shape.obj','geometry_certificate.npz'):
            assert F.I.sha256(stage/'final'/n)==F.I.sha256(group/'step4'/n)==F.I.sha256(witness.parent/n)
        assert len(list((group/'step4').glob('*.png')))==2
        assert np.isclose(metrics['metrics']['object_and_support_poses']['box_volume_cm3'],best)
        F.I.check_report(stage/'final/report.json')
        reassigned=[e for e in r['events'] if e['operation']=='contact_reassignment']
        accepted_reassigned=0
        for e in r['events']:
            if e.get('operation')=='measured_trial' and e['accepted']:
                s,_=load((F.I.ROOT/e['witness']).parent,tasks)
                if any('_CAVITY_' in c['candidate_id'] for row in s.groups for c in row):accepted_reassigned+=1
        rows.append(dict(group=group.name,pose_count=len(tasks),before_cm3=r['initial_volume_cm3'],after_cm3=best,
            reduction_percent=100*(1-best/r['initial_volume_cm3']),accepted=count,
            contact_reassignment_proposals=len(reassigned),accepted_states_with_reassigned_contacts=accepted_reassigned,
            force_angle_before_deg=r['initial_directions']['force_mean_pair_angle_deg'],force_angle_after_deg=r['final_directions']['force_mean_pair_angle_deg'],
            exit_angle_before_deg=r['initial_directions']['exit_mean_pair_angle_deg'],exit_angle_after_deg=r['final_directions']['exit_mean_pair_angle_deg'],
            final_area_cm2=metrics['metrics']['object_and_support_poses']['xy_area_cm2'],
            object_only_volume_cm3=metrics['metrics']['object_poses']['box_volume_cm3'],
            occupied_box_lower_bound_reached=bool(np.isclose(best,metrics['metrics']['object_poses']['box_volume_cm3'],rtol=1e-8))))
        inputs.extend([stage/'baseline_step5/report.json',stage/'report.json',group/'step4/report.json',group/'step5_evaluate/report.json'])
        print('CAVITY AUDIT',group.name,'PASS',round(rows[-1]['reduction_percent'],2),flush=True)
    out=F.I.OUTPUTS/'B/pose1+3/step5_evaluate'
    lines=['# Joint cavity comparison','','Compared with the previous qualified absolute-direction finals. The primary metric remains actual Step5 occupied XYZ box volume; projection overlap and the contact/demand box are proposal guides only. Reaction angles are area-weighted surface-normal means, not load-weighted actual resultant forces. Direction coherence is not a hard Pareto constraint in this round; some volume gains increase the reaction-normal angle. Every task passes all original 32768 loads. One constructor acceptance; no exported-body geometry replay.','',
        '| Set | Previous cm³ | Cavity cm³ | Further reduction | Reaction-normal angle | Exit angle |','|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['group']} | {r['before_cm3']:.2f} | {r['after_cm3']:.2f} | {r['reduction_percent']:.2f}% | {r['force_angle_before_deg']:.1f} → {r['force_angle_after_deg']:.1f}° | {r['exit_angle_before_deg']:.1f} → {r['exit_angle_after_deg']:.1f}° |")
    (out/'cavity_comparison.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,6));y=np.arange(len(rows))
    ax.barh(y-.18,[r['before_cm3'] for r in rows],height=.35,label='Previous direction optimization')
    ax.barh(y+.18,[r['after_cm3'] for r in rows],height=.35,label='Joint common-cavity search')
    ax.set_yticks(y,[r['group'] for r in rows]);ax.invert_yaxis();ax.set_xlabel('Actual Step5 occupied XYZ volume (cm³)');ax.legend();fig.tight_layout();fig.savefig(out/'cavity_comparison.png',dpi=160);plt.close(fig)
    summary=dict(complete=True,passed=True,groups=len(rows),task_instances=sum(r['pose_count'] for r in rows),improved_groups=sum(r['reduction_percent']>1e-6 for r in rows),rows=rows,
        validation_policy='Fresh force/local path and saved construction/metric/source audit; no exported-model geometric replay',
        provenance=dict(inputs=F.I.hashes(inputs),code=F.I.hashes([Path(__file__)])),artifacts={n:F.I.sha256(out/n) for n in ('cavity_comparison.md','cavity_comparison.png')})
    F.I.save(out/'cavity_comparison.json',summary)
    print('CAVITY TOTAL',summary['groups'],summary['task_instances'],summary['improved_groups'],flush=True)
if __name__=='__main__':main()
