"""Audit real shared-head identities and publish the complete B experiment.

Force rays are rebuilt for all saved loads. Installed contact and owner-solid
views are compared before union. Complete body geometry uses the existing single
construction acceptance; this script performs no exported-body geometry replay.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I, shared_dsl as DSL
from step3_scheculer.run_dsl import saved_task
from step3_scheculer.floor_margin import transforms_from_owner
from step3_scheculer.pair_scoring import J
from step3_scheculer.construction_contract import accept_witness
from step4_connect_support.fixture_view import cells_for
from step2_local_support import geometry as G


def review(group):
    out=group/'step3_scheculer'/DSL.STAGE
    report=I.check_report(out/'report.json')
    if 'error' in report:
        return dict(case=group.name,passed=False,audit_passed=False,error=report['error'])
    sharing=I.check_report(out/'sharing.json')
    heads=I.read_contacts(out/'shared_heads.npz')
    assert len(heads)==report['physical_head_count']==sharing['physical_head_count']==report['total_heads']
    ids=[c['candidate_id'] for c in heads]
    assert len(set(ids))==len(ids)
    tasks=[saved_task(group,p) for p in report['poses']]
    transforms=transforms_from_owner(tasks[0],tasks)
    bases=np.asarray(report['placement']['bases']);offsets=np.asarray(report['placement']['offsets'])
    native_offsets=G.vertex_offsets(tasks[0].domain.mesh,report['shared_head_depth_m'])[0]
    force_results=[];maximum_error=0.;covered=[];failed_exit_poses=[];failed_force_poses=[];dependencies=[out/'report.json',out/'sharing.json']
    for k,(task,b,o,transform) in enumerate(zip(tasks,bases,offsets,transforms)):
        folder=out/task.pose;pose_report=I.check_report(folder/'report.json')
        contacts=I.read_contacts(folder/f'final_contacts_{task.pose}.npz')
        dependencies += [folder/'report.json',folder/'force_only_report.json']
        assert [c['candidate_id'] for c in contacts]==ids
        if not pose_report['exit_passed']:failed_exit_poses.append(task.pose)
        if not pose_report['force_passed']:failed_force_poses.append(task.pose)
        assert pose_report['shared_program'] and not pose_report['independent_pose_optimization']
        np.testing.assert_allclose(task.domain.mesh.vertices@b+o,tasks[0].domain.mesh.vertices,atol=2e-12,rtol=0)
        for native,view in zip(heads,contacts):
            error=float(np.max(np.abs(view['triangles_m']@b+o-native['triangles_m'])))
            maximum_error=max(maximum_error,error)
            assert error<2e-12
            assert not np.intersect1d(view['source_faces'],task.domain.work_ids).size
            assert view['triangles_m'][:,:,2].min()>=.002-1e-10
            # Same physical solid uses CANONICAL depth/offsets, then rigid views.
            for cell in cells_for(native,tasks[0].domain,native_offsets):
                world=cell@transform[:3,:3].T+transform[:3,3]
                np.testing.assert_allclose(world@b+o,cell,atol=2e-12,rtol=0)
        mask,_=J.classify(task.supply(contacts),task.targets)
        np.testing.assert_array_equal(mask,I.load_npz(folder/'coverage.npz')['passed'])
        assert mask.sum()==pose_report['covered_count']
        covered.append(int(mask.sum()))
        force_report=I.check_report(folder/'force_only_report.json')
        force_contacts=I.read_contacts(folder/'force_only_contacts.npz')
        force_mask,_=J.classify(task.supply(force_contacts),task.targets)
        np.testing.assert_array_equal(force_mask,I.load_npz(folder/'force_only_coverage.npz')['passed'])
        assert force_mask.sum()==force_report['covered_count']
        force_results.append(bool(force_mask.all()))
    assert covered==report['covered_counts']
    body_path=group/'step4/data'/DSL.BODY_STAGE/'report.json'
    body=I.check_report(body_path);dependencies.append(body_path)
    witness=I.check_report(I.ROOT/report['construction_witness']);dependencies.append(I.ROOT/report['construction_witness'])
    if report['passed']:
        assert all(x==32768 for x in covered) and accept_witness(witness)
        assert witness['physical_head_count']==len(heads)
        assert witness['installed_shared_solid_max_error_m']<2e-12
        assert body['passed'] and body['published_model_bit_identical_to_step3_witness']
        assert I.sha256(body_path.parent/'shape.obj')==I.sha256((I.ROOT/report['construction_witness']).parent/'shape.obj')
        assert witness['validation_policy'].startswith('single in-memory construction acceptance')
    else:
        assert not body['passed'] and body['status']=='not_applicable_step3_rejected'
    paths=witness.get('selected_exit_paths') or []
    vectors=[np.asarray(p['initial_object_exit_world'])@b for p,b in zip(paths,bases)]
    angles=[float(np.degrees(np.arccos(np.clip(a@b,-1,1)))) for i,a in enumerate(vectors) for b in vectors[i+1:]]
    reuse=None
    if report['passed']:
        selection=json.loads((I.ROOT/witness['trial_folder']/'exit_selection.json').read_text())
        individual=[]
        for pose,plan in zip(selection['diagnostics'],paths):
            option=next(v for v in pose['options'] if v['option_id']==plan['id'])
            individual.append(option['excluded_workspace_cm3'])
        union=witness['excluded_workspace_cm3']
        total=sum(individual)
        reuse=dict(individual_padded_sweep_sum_cm3=total,joint_padded_sweep_union_cm3=union,
                   union_saving_fraction=1-union/total if total else 0.,
                   includes_common_starting_object_volume=True,not_a_pure_channel_only_metric=True)
    progress=json.loads((out/'joint_progress.json').read_text())
    feasible_counts=[e['physical_heads'] for e in progress['events'] if e.get('covered')==32768*len(tasks)]
    steps=[s for e in progress['events'] for s in e.get('gradient_steps',[])]
    return dict(case=group.name,passed=report['passed'],audit_passed=True,
        force_passed=report['force_passed'],exit_passed=report['exit_passed'],
        local_contact_passed=report['local_contact_passed'],
        failed_exit_poses=failed_exit_poses,failed_force_poses=failed_force_poses,
        smallest_force_feasible_heads=min(feasible_counts) if feasible_counts else None,
        physical_heads=len(heads),shared_heads=report['shared_head_count'],
        original_independent_heads=report['seed_total_heads'],head_pose_uses=report['total_head_pose_uses'],
        task_count=len(tasks),force_checkpoint_tasks_passed=sum(force_results),coverage=covered,
        installed_contact_max_error_m=maximum_error,
        accepted_joint_gradient_updates=sum(s['accepted'] for s in steps),
        joint_gradient_steps=len(steps),seconds=report['seconds'],
        material_cm3=body.get('volume_cm3') if body['passed'] else None,
        xy_area_cm2=body.get('space_budget',{}).get('xy_area_cm2') if body['passed'] else None,
        selected_exit_paths=witness.get('selected_exit_paths'),
        mean_fixture_frame_exit_angle_deg=float(np.mean(angles)) if angles else None,
        exit_space_reuse=reuse,
        construction_failures=witness.get('all_trial_failures') if not report['passed'] else None,
        step3_implies_step4=True,dependencies=[str(p.relative_to(I.ROOT)) for p in dependencies])


def render(rows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups=[r for r in rows if '+' in r['case']]
    fig,ax=plt.subplots(figsize=(13,6))
    x=np.arange(len(groups));before=[r.get('original_independent_heads',0) for r in groups]
    after=[r.get('physical_heads',0) if r['passed'] else (r.get('smallest_force_feasible_heads') or r.get('physical_heads',0)) for r in groups]
    ax.bar(x-.18,before,.36,label='Saved independent head count',color='#b9bfc6')
    ax.bar(x+.18,after,.36,label='Constructed count / force-only candidate count',color=['#2b9366' if r['passed'] else '#d6a54f' for r in groups])
    for i,r in enumerate(groups):
        ax.text(i+.18,after[i]+.25,'PASS' if r['passed'] else 'FORCE ONLY',ha='center',fontsize=8)
    ax.set_xticks(x,[r['case'] for r in groups],rotation=30,ha='right')
    ax.set_ylabel('Distinct physical heads');ax.set_title('One shared head program per combination; rejected candidates are not fixtures')
    ax.legend();fig.tight_layout();fig.savefig(out/'all_cases.png',dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--groups',nargs='+');args=parser.parse_args()
    root=I.OUTPUTS/'B';out=root/'pose2+9+13+15+17/step3_scheculer'/DSL.STAGE
    batch=json.loads((out/'batch_report.json').read_text())
    groups=[root/g for g in args.groups] if args.groups else [root/r['group'] if '+' in r['group'] else root/'independent_poses'/r['group'] for r in batch['results']]
    rows=[]
    for group in groups:
        try:row=review(group)
        except Exception as error:
            import traceback
            row=dict(case=group.name,passed=False,audit_passed=False,error=str(error),traceback=traceback.format_exc())
        rows.append(row);print('SHARED AUDIT',group.name,row['audit_passed'],flush=True)
    combinations=[r for r in rows if '+' in r['case']];singles=[r for r in rows if '+' not in r['case']]
    dependencies=[out/'batch_report.json']+[I.ROOT/p for r in rows for p in r.get('dependencies',[])]
    summary=dict(complete=True,schema=DSL.SCHEMA,audit_passed=all(r['audit_passed'] for r in rows),
        group_count=len(combinations),group_passed_count=sum(r['passed'] for r in combinations),
        single_count=len(singles),single_passed_count=sum(r['passed'] for r in singles),
        force_checkpoint_tasks_passed=sum(r.get('force_checkpoint_tasks_passed',0) for r in rows),
        task_count=sum(r.get('task_count',0) for r in rows),
        accepted_group_independent_heads=sum(r['original_independent_heads'] for r in combinations if r['passed']),
        accepted_group_physical_heads=sum(r['physical_heads'] for r in combinations if r['passed']),
        accepted_group_shared_heads=sum(r['shared_heads'] for r in combinations if r['passed']),
        accepted_joint_gradient_updates=sum(r.get('accepted_joint_gradient_updates',0) for r in rows),
        step3_implies_step4=all(r.get('step3_implies_step4',False) for r in rows),
        validation_policy='fresh shared-contact/rigid-solid identity and force audit; complete fixture uses single construction acceptance; no independent exported-model geometry replay',
        limitations=['object-attached shared placement family, not arbitrary regrasp registration',
                     'finite numerical descent and rewrites, not global head minimum',
                     'full construction is still performed in Step3 for its acceptance contract',
                     'direction closeness uses normal-opening proxy; exact exit-space union uses discrete path beam search'],
        results=rows,config=batch.get('config'),
        provenance=dict(inputs=I.hashes(dependencies),code=I.hashes([Path(__file__)])))
    I.save(out/'experiment_summary.json',summary)
    lines=['# Joint shared-head DSL: B experiment','',
        f"Combinations: {summary['group_passed_count']}/{len(combinations)} constructed with genuinely shared physical heads.",
        f"Independent poses: {summary['single_passed_count']}/{len(singles)} constructed.",
        f"Force-only checkpoints: {summary['force_checkpoint_tasks_passed']}/{summary['task_count']} tasks, each 32768 original loads.",
        f"Accepted combinations: {summary['accepted_group_independent_heads']} saved independent heads -> {summary['accepted_group_physical_heads']} physical heads, of which {summary['accepted_group_shared_heads']} serve multiple poses.",
        f"Accepted joint gradient updates: {summary['accepted_joint_gradient_updates']}.",'',
        'The physical head count is measured once in the canonical fixture frame. Per-pose views of the same contact triangles and finite-depth solids must coincide when installed. All heads are available in every pose; equilibrium may assign zero force to any head. Work faces from all tasks are excluded. Step4 publishes the identical Step3 construction witness.','',
        '| Case | Constructed | Independent seed heads | Physical heads | Shared heads | Coverage |','|---|---|---|---|---|---|']
    for r in rows:lines.append(f"| {r['case']} | {r['passed']} | {r.get('original_independent_heads','—')} | {r.get('physical_heads','—')} | {r.get('shared_heads','—')} | {r.get('coverage',r.get('error'))} |")
    lines+=['','## Scope and limits','']+['- '+v for v in summary['limitations']]
    lines+=['','Old independent DSL experiments remain in `dsl/` and `dsl_support/`. Active shared results are in `dsl_shared/` and `dsl_shared_support/`.']
    (out/'experiment_summary.md').write_text('\n'.join(lines)+'\n')
    render(rows,out)
    if not summary['audit_passed']:raise SystemExit('Shared-head audit failed; inspect experiment_summary.json')

if __name__=='__main__':main()
