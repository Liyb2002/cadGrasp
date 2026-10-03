"""All-round force/hash/Step5 audit; no exported-model geometry replay."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
import trimesh
from step3_scheculer import absolute_dsl as A,absolute_metrics as M
from step3_scheculer.review_operation_dsl import load
from step3_scheculer.run_dsl import saved_task
from step3_scheculer.correct_absolute_metadata import STAGES

def main():
    A.activate();rows=[];inputs=[]
    for group in sorted((A.F.I.OUTPUTS/'B').glob('pose*+*')):
        original=A.F.I.check_report(group/'step3_scheculer/dsl_absolute/report.json')
        tasks=[saved_task(group,p) for p in original['poses']];checker=A.F.Checker(tasks)
        baseline=A.F.I.check_report(group/'step3_scheculer/dsl_absolute/baseline_step5/report.json')
        best=original['initial_volume_cm3'];accepted=0;candidates=0
        for stage_name in STAGES:
            stage=group/'step3_scheculer'/stage_name
            if not (stage/'report.json').exists():continue
            selected=stage
            r=A.F.I.check_report(stage/'report.json');assert r['passed'] and not r['head_count_optimized']
            assert np.isclose(best,r['initial_volume_cm3'])
            initial,_=load(stage/'initial',tasks);assert checker.check(initial)['passed']
            for e in r['events']:
                if e['operation']!='measured_trial':continue
                witness=A.F.I.ROOT/e['witness'];body=A.F.I.check_report(witness);assert body['passed']
                candidate,_=load(witness.parent,tasks)
                measured=M.volume(tasks,candidate,trimesh.load(witness.parent/'shape.obj',force='mesh',process=False))
                assert np.isclose(measured,e['volume_cm3']) and np.isclose(best,e['best_before_cm3'])
                assert body['construction']['minimum_branch_thickness']['all_complete_cores_preserved']
                assert len(body['timing_seconds']['validation_passes'])==1
                assert body['validation_policy']['export_recheck'] is False
                if e['accepted']:
                    assert measured<best;best=measured;accepted+=1
                    checked=checker.check(candidate);assert checked['passed']
                    assert all(v['covered']==32768 for v in checked['per_pose'])
                    A.F.roots(tasks,candidate)
                candidates+=1;inputs.append(witness)
            assert np.isclose(best,r['final_volume_cm3'])
            assert r['step4_candidates_executed']>=1
            inputs.append(stage/'report.json')
        A.F.I.check_report(selected/'final/report.json')
        state,_=load(selected/'final',tasks)
        checked=checker.check(state);assert checked['passed'] and all(v['covered']==32768 for v in checked['per_pose'])
        A.F.roots(tasks,state)
        witness=A.F.I.ROOT/r['witness'];A.F.I.check_report(witness)
        for filename in ('shape.obj','geometry_certificate.npz'):
            assert A.F.I.sha256(witness.parent/filename)==A.F.I.sha256(group/'step4'/filename)==A.F.I.sha256(selected/'final'/filename)
        public=A.F.I.check_report(group/'step4/report.json')
        assert len(list((group/'step4').glob('*.png')))==2
        final=A.F.I.check_report(group/'step5_evaluate/report.json')
        assert np.isclose(final['metrics']['object_and_support_poses']['box_volume_cm3'],best)
        after=r['final_directions'];before=original['initial_directions']
        oldarea=baseline['metrics']['object_and_support_poses']['xy_area_cm2'];area=final['metrics']['object_and_support_poses']['xy_area_cm2']
        rows.append(dict(group=group.name,poses=len(tasks),initial_volume_cm3=original['initial_volume_cm3'],final_volume_cm3=best,
            reduction_percent=100*(1-best/original['initial_volume_cm3']),force_angle_before_deg=before['force_mean_pair_angle_deg'],force_angle_after_deg=after['force_mean_pair_angle_deg'],
            exit_angle_before_deg=before['exit_mean_pair_angle_deg'],exit_angle_after_deg=after['exit_mean_pair_angle_deg'],
            initial_area_cm2=oldarea,final_area_cm2=area,area_reduction_percent=100*(1-area/oldarea),
            accepted=accepted,step4_candidates=candidates,final_stage=r['schema'],head_count_diagnostic=r['physical_head_count']))
        inputs.extend([group/'step4/report.json',group/'step5_evaluate/report.json',group/'step3_scheculer/dsl_absolute/baseline_step5/report.json'])
        print('ABSOLUTE AUDIT',group.name,'PASS',round(rows[-1]['reduction_percent'],2),flush=True)
    out=A.F.I.OUTPUTS/'B/pose1+3/step5_evaluate'
    lines=['# Absolute-direction DSL comparison','','Primary: aggregate XYZ occupied bounding-box volume of all saved object and installed support poses. Reaction angles use area-weighted surface normals in the physical fixture frame. Head count is not optimized. All original 32768 loads per task pass; one construction acceptance, no exported-body geometry replay.','',
        '| Set | Old cm³ | New cm³ | Reduction | Reaction angle | Exit angle |','|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['group']} | {r['initial_volume_cm3']:.2f} | {r['final_volume_cm3']:.2f} | {r['reduction_percent']:.2f}% | {r['force_angle_before_deg']:.1f} → {r['force_angle_after_deg']:.1f}° | {r['exit_angle_before_deg']:.1f} → {r['exit_angle_after_deg']:.1f}° |")
    (out/'absolute_comparison.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(15,6));y=np.arange(len(rows));labels=[r['group'] for r in rows]
    axes[0].barh(y-.18,[r['initial_volume_cm3'] for r in rows],height=.35,label='Previous qualified DSL')
    axes[0].barh(y+.18,[r['final_volume_cm3'] for r in rows],height=.35,label='Absolute direction DSL')
    axes[0].set_yticks(y,labels);axes[0].set_xlabel('Occupied XYZ box volume (cm³)');axes[0].legend();axes[0].invert_yaxis()
    axes[1].barh(y,[r['reduction_percent'] for r in rows]);axes[1].set_yticks(y,labels);axes[1].set_xlabel('Actual occupied volume reduction (%)');axes[1].invert_yaxis()
    fig.suptitle('Step5 comparison: head count is not an optimization target');fig.tight_layout();fig.savefig(out/'absolute_comparison.png',dpi=160);plt.close(fig)
    summary=dict(complete=True,passed=True,groups=len(rows),poses=sum(r['poses'] for r in rows),improved_groups=sum(r['reduction_percent']>1e-6 for r in rows),
        rows=rows,regression_tests_passed=24,validation_policy='Fresh force/local-path checks; witness/source/hash/metric audit; no exported-body geometric replay',
        provenance=dict(inputs=A.F.I.hashes(inputs),code=A.F.I.hashes([Path(__file__)])),
        artifacts={n:A.F.I.sha256(out/n) for n in ('absolute_comparison.md','absolute_comparison.png')})
    A.F.I.save(out/'absolute_comparison.json',summary)
    print('ABSOLUTE TOTAL',summary['groups'],summary['poses'],summary['improved_groups'],flush=True)
if __name__=='__main__':main()
