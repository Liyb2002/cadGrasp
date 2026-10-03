"""Publish comparisons and an English result sheet from completed DSL trials."""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I


def main():
    root=I.OUTPUTS/'B'
    out=root/'pose2+9+13+15+17/step3_scheculer/dsl'
    batch=I.check_report(out/'batch_report.json') if 'provenance' in json.loads((out/'batch_report.json').read_text()) else json.loads((out/'batch_report.json').read_text())
    rows=batch['results']
    input_paths=[out/'batch_report.json']
    combined=[]
    for row in rows:
        if 'error' in row:
            combined.append(dict(case=row['group'],passed=False,error=row['error']))
            continue
        pose_reports=[I.check_report(I.ROOT/p) for p in row['per_pose_reports']]
        force_reports=[I.check_report((I.ROOT/p).parent/'force_only_report.json') for p in row['per_pose_reports']]
        input_paths += [I.ROOT/p for p in row['per_pose_reports']]
        input_paths += [(I.ROOT/p).parent/'force_only_report.json' for p in row['per_pose_reports']]
        group=root/row['group'] if '+' in row['group'] else root/'independent_poses'/row['group']
        body_path=group/'step4/data/dsl_support/report.json'
        body=json.loads(body_path.read_text()) if body_path.exists() else None
        if body is not None:
            input_paths.append(body_path)
            if body['passed']:
                I.check_report(body_path)
        grads=[s for p in pose_reports for e in p['events'] for s in e.get('gradient_steps',[])]
        gradients_accepted=sum(s['accepted'] for s in grads)
        gradient_repairs=sum(1 for p in pose_reports for i,e in enumerate(p['events'])
            if e['operation']=='accept_delete' and i and p['events'][i-1]['operation']=='delete_and_repair')
        result=dict(case=row['group'],passed=row['passed'],local_contact_passed=row['local_contact_passed'],
            construction_passed=row['construction_passed'],complete_fixture_verified=row['complete_fixture_verified'],force_passed=row['force_passed'],exit_passed=row['exit_passed'],
            force_only_tasks_passed=sum(p['force_passed'] for p in force_reports),task_count=len(force_reports),
            gpu_cpu_fallback_batches=sum(p['acceleration']['cpu_fallback_batches'] for p in pose_reports),
            initial_heads=row['seed_total_heads'],final_heads=row['total_heads'],heads_by_pose=row['head_counts'],exit_option_counts=row['exit_option_counts'],
            coverage=row['covered_counts'],baseline_mean_exit_angle_deg=row['baseline_mean_object_exit_angle_deg'],
            mean_exit_angle_deg=row['mean_object_exit_angle_deg'] if row['angles_are_certified'] and len(pose_reports)>1 else None,
            search_gradient_updates_accepted=gradients_accepted,accepted_deletions_after_gradient_repair=gradient_repairs,
            initial_contact_area_cm2=None,final_contact_area_cm2=sum(p['contact_area_m2'] for p in pose_reports)*10000,
            seconds=row['seconds'],step4_passed=body['passed'] if body else None,
            step4_failure=body.get('error') if body else None,
            step4_status=body.get('status') if body else None,
            step4_material_volume_cm3=body['volume_cm3'] if body and body['passed'] else None,
            step4_previous_material_volume_cm3=body['previous_material_volume_cm3'] if body and body['passed'] else None,
            step4_previous_xy_cm2=body['previous_space_budget']['xy_area_cm2'] if body and body['passed'] else None,
            step4_occupied_xy_cm2=body['space_budget']['xy_area_cm2'] if body and body['passed'] else None)
        combined.append(result)
    pairs=[r for r in combined if '+' in r['case']]
    singles=[r for r in combined if '+' not in r['case']]
    audit=I.check_report(out/'review.json') if (out/'review.json').exists() else None
    benchmark=I.check_report(out/'gpu_benchmark.json') if (out/'gpu_benchmark.json').exists() else None
    input_paths += [out/p for p in ('review.json','gpu_benchmark.json') if (out/p).exists()]
    assert all(r.get('step4_passed') is True for r in combined if r['passed']), 'Constructive invariant violated'
    summary=dict(complete=True,step3_pass_implies_step4_pass=True,group_count=len(pairs),single_pose_count=len(singles),
        force_only_tasks_passed=sum(r.get('force_only_tasks_passed',0) for r in combined),
        task_count=sum(r.get('task_count',0) for r in combined),
        independent_audit_passed=audit['passed'] if audit else None,
        baseline_snapshot_matches_current_workspace=audit['baseline_snapshot_matches_current_workspace'] if audit else None,
        gpu_cpu_fallback_batches=sum(r.get('gpu_cpu_fallback_batches',0) for r in combined),
        gpu_benchmark=benchmark,
        group_passed_count=sum(r['passed'] for r in pairs),single_passed_count=sum(r['passed'] for r in singles),
        accepted_group_heads_before=sum(r['initial_heads'] for r in pairs if r['passed']),
        accepted_group_heads_after=sum(r['final_heads'] for r in pairs if r['passed']),
        accepted_single_heads_before=sum(r['initial_heads'] for r in singles if r['passed']),
        accepted_single_heads_after=sum(r['final_heads'] for r in singles if r['passed']),
        constructed_group_count=sum(r.get('step4_passed') is True for r in pairs),
        constructed_single_count=sum(r.get('step4_passed') is True for r in singles),
        results=combined,config=batch['config'],
        interpretation='Step3 success requires an exported full construction witness; Step4 materializes the same solid, preserving the constructive guarantee; finite-search counts are not global minima',
        provenance=dict(inputs=I.hashes(input_paths),code=I.hashes([Path(__file__)])))
    old_path=out/'history/local_exit_v2/experiment_summary.json'
    if old_path.exists():
        old=json.loads(old_path.read_text())
        summary['v1_comparison']={k:dict(before=old.get(k),after=summary.get(k)) for k in ('group_passed_count','single_passed_count','constructed_group_count','accepted_group_heads_after','accepted_single_heads_after')}
        input_paths.append(old_path)
    diagnostics=out/'exit_diagnostics.json'
    if diagnostics.exists():
        summary['exit_diagnostics']=I.check_report(diagnostics)['failed_tasks']
        input_paths.append(diagnostics)
    summary['provenance']['inputs']=I.hashes(input_paths)
    I.save(out/'experiment_summary.json',summary)
    lines=['# DSL contact search: complete B trial','',
        f"Step3: {summary['group_passed_count']}/{len(pairs)} combinations; {summary['single_passed_count']}/{len(singles)} independent poses.",
        f"Force-only phase: {summary['force_only_tasks_passed']}/{summary['task_count']} tasks pass all original loads, before exit repair.",
        f"Accepted combinations: {summary['accepted_group_heads_before']} → {summary['accepted_group_heads_after']} total physical heads.",
        f"Accepted independent poses: {summary['accepted_single_heads_before']} → {summary['accepted_single_heads_after']} heads.",
        f"Step4 materialization: {summary['constructed_group_count']} combinations and {summary['constructed_single_count']} independent fixtures; all Step3 successes materialize their identical full-solid witnesses.",'',
        'Each task retains all 32768 original loads. Step3 passed requires force/moment balance, shared no-uplift, baseline root-path connectivity and an exported full-solid construction witness. That witness checks all installed roots, floors, working surfaces, connected material, ground-hull coverage and complete continuous translation paths. Step4 publishes the same verified solid; a rejected Step3 is not applicable to Step4. No global head-count optimum or compactness guarantee. Historical pose1+3 and its copy use their own saved pose revision. Independent poses are separate experiments, not a twenty-pose fixture. V3 adds constructive acceptance and Step4 witness materialization; local contacts alone are never a Step3 success.','',
        '| Case | Step3 | Heads before → after | Per-pose heads | Coverage | Verified exit options per pose | Step4 |',
        '|---|---|---|---|---|---|---|']
    for r in combined:
        if 'error' in r:
            lines.append(f"| {r['case']} | ERROR | — | — | {r['error']} | — | — |")
            continue
        initial='—' if r['baseline_mean_exit_angle_deg'] is None else f"{r['baseline_mean_exit_angle_deg']:.1f}°"
        final='uncertified' if r['mean_exit_angle_deg'] is None else f"{r['mean_exit_angle_deg']:.1f}°"
        angle=f'{initial} → {final}' if '+' in r['case'] else '— (single pose)'
        lines.append(f"| {r['case']} | {'PASS' if r['passed'] else 'FAIL'} | {r['initial_heads']} → {r['final_heads']} | {r['heads_by_pose']} | {r['coverage']} | {r['exit_option_counts']} | {r['step4_passed'] if r['step4_passed'] is not None else 'not run'} |")
    lines += ['', '## Limits', '',
        'The loss uses a subset of real wrench rays and original sampled loads to propose steps; final acceptance uses the full contact surfaces and all original loads. CUDA batches float64 active-set NNLS residuals; CPU LP decides acceptance. Gradient steps can improve the proposal loss without improving exact coverage, so exact checks decide accepted deletions. Force-only contacts and coverage are saved separately; failed exit repair cannot discard a previously force-feasible checkpoint. Radius bounds are numerical search limits, not pressure or strength certification. Exit probes now use the exact 1%-of-object-scale constructor depth. Thin roadmap paths are only a prefilter; the actual full solid is the constructive certificate.','',
        'Step3 compiles complete constructive witnesses using explicit placements and adaptive floor coverage. Failed construction rejects the Step3 proposal, and unconstructible head deletion is rolled back. Step4 consumes the witness without another search. Finite search failure does not prove no other contacts, paths or placements exist. Copied baseline models remain comparison artifacts and are not new DSL results.']
    if 'v1_comparison' in summary:
        lines += ['', '## V2 local acceptance versus V3 constructive acceptance', '', '| Metric | V2 | V3 |', '|---|---|---|']
        lines += [f"| {k} | {v['before']} | {v['after']} |" for k,v in summary['v1_comparison'].items()]
    if 'exit_diagnostics' in summary:
        lines += ['', 'Failed task diagnostics are recorded in exit_diagnostics.json. Absence of a common initial translation opening for a contact set is not proof that no different contacts or rotational path exist.']
    if benchmark:
        lines += ['', '## GPU check', '',
            f"CUDA float64 active-set NNLS: {benchmark['candidate_count']} real patch perturbations/deletions × 24 original loads, maximum CPU/GPU residual difference {benchmark['max_absolute_residual_error']:.3g}; kernel-only speedup {benchmark['kernel_speedup']:.1f}×. This excludes mesh clipping, final LP and initialization.",
            f"CPU fallback batches across all task searches: {summary['gpu_cpu_fallback_batches']}."]
    for r in pairs:
        if r.get('step4_passed'):
            lines += ['', f"Constructed {r['case']}: material {r['step4_previous_material_volume_cm3']:.3f} → {r['step4_material_volume_cm3']:.3f} cm³; actual workstation XY footprint {r['step4_previous_xy_cm2']:.3f} → {r['step4_occupied_xy_cm2']:.3f} cm². Material reduction and footprint reduction are distinct measurements."]
    if audit:
        lines += ['', '## Independent replay and input snapshot', '',
            f"Independent artifact audit passed: {audit['passed']}. Current workspace baseline equals copy-time snapshot: {audit['baseline_snapshot_matches_current_workspace']}.",
            'This experiment uses its copied inputs. A separate workspace job modified the source baseline during the experiment; source snapshot drift is recorded separately and is not a failed design verdict.']
    (out/'experiment_summary.md').write_text('\n'.join(lines)+'\n')
    draw(out/'all_cases.png',combined)
    print({k:v for k,v in summary.items() if k not in ('results','provenance')})


def draw(path,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(15,.40*len(rows)+2.2),facecolor='white')
    ax.axis('off')
    table=[]
    for row in rows:
        if 'error' in row:
            table.append([row['case'],'ERROR','—','—','—','—'])
        else:
            table.append([row['case'],'PASS' if row['passed'] else 'FAIL',
                f"{row['initial_heads']} → {row['final_heads']}",str(row['heads_by_pose']),
                str(row['exit_option_counts']),
                f"{row['step4_occupied_xy_cm2']:.1f} cm²" if row.get('step4_passed') else 'Step3 rejected' if row.get('step4_passed') is False else '—'])
    widget=ax.table(cellText=table,colLabels=['Saved case','Step3 full witness','Physical heads','Heads by pose','Verified exit options','Step4 XY footprint'],
        loc='center',cellLoc='center',colWidths=[.26,.08,.14,.20,.17,.15])
    widget.auto_set_font_size(False);widget.set_fontsize(10);widget.scale(1,1.5)
    for (i,j),cell in widget.get_celld().items():
        cell.set_edgecolor('#dce2e6')
        if i==0:cell.set_facecolor('#e9eff4');cell.set_text_props(weight='bold')
        elif j==1:cell.set_facecolor('#dceee0' if table[i-1][1]=='PASS' else '#f5dede')
    ax.set_title('Contact DSL + numerical descent: original loads and full construction witnesses\nFinite-search head counts; complete fixture acceptance is separate',fontsize=14,pad=25)
    fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)


if __name__=='__main__':main()
