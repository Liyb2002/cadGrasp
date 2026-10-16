"""Execute one fixed-layout, connected-wrap-then-delete Step5 case."""
from pathlib import Path
import json
import sys
import time
import traceback
from types import SimpleNamespace
from wrap_geometry import *
from wrap_dimensions import dimensions


def worker(payload):
    began=time.monotonic();args=SimpleNamespace(**payload)
    import wrap_geometry as geometry
    geometry.CONNECTOR_RADIUS_M=float(payload.get('connector_radius_m',geometry.CONNECTOR_RADIUS_M))
    geometry.GROUND_MARGIN_SCALE=float(payload.get('ground_margin_scale',1.))
    geometry.TERMINAL_FOOT_WIDTH_M=float(payload.get('terminal_foot_width_m',MINIMUM_WIDTH_M))
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    inputs=WrapInputs(args.object,args.selection_summary)
    data=inputs.load(args.case)
    demands=original_demands(inputs,data,out)
    with np.load(data['selected']/'layout.npz') as z:layout=Layout.from_arrays(z)
    witness_path=data['selected']/'oracle_connected_witness.obj'
    inputs.protected[C.relative(witness_path)]=C.digest(witness_path)
    witness=C.trimesh.load(witness_path,force='mesh',process=False)
    validator=Validator(data,layout,witness,args.object)
    print('INPUTS',args.case['id'],'original demands and fixed layout loaded',flush=True)
    # Retain the complete Step4 CONTACT CORE before growing any feet. An old
    # complete reference is optional construction material, never a replacement
    # for the core or a hidden contact-pruning stage.
    previous=C.A.resolve_artifact(Path(args.baseline)) if args.baseline else None
    reference=None
    if previous is not None:
        inputs.protected[C.relative(previous)]=C.digest(previous)
        reference=C.trimesh.load(previous,force='mesh',process=False)
    baseline=data['fixture']
    _,_,baseline_checks=validator.check(baseline,ground=False,width=False,connected=False)
    baseline_checks['construction_input_only']=True
    complete_path,complete_record=C.constructed_reference(inputs,data,SimpleNamespace(reference_stage='step4'))
    completion=C.trimesh.load(complete_path,force='mesh',process=False)
    print('BASELINE',args.case['id'],baseline_checks['material_cm3'],flush=True)
    construction_reuse=None
    if payload.get('initial_source'):
        from wrap_initial_reuse import reuse
        mesh,value,proof,checks,construction,construction_reuse=reuse(
            payload['initial_source'],data,validator,out,inputs)
        for key,name in [('connector_radius_m','CONNECTOR_RADIUS_M'),
                         ('ground_margin_scale','GROUND_MARGIN_SCALE'),
                         ('terminal_foot_width_m','TERMINAL_FOOT_WIDTH_M')]:
            setattr(geometry,name,construction_reuse['construction_parameters'][key])
    else:
        mesh,value,proof,checks,construction=construct(data,baseline,validator,out,completion,reference)
    core_loss=volume(solid(baseline)-value)
    if core_loss>1e-10:raise ValueError('Initial wrapping deleted Step4 core material')
    C.save(out/'construction.json',dict(complete=True,history=construction,initial_core_loss_m3=core_loss))
    deleter=Deleter(mesh,value,proof,checks,validator,out)
    deleter.run(tuple(args.grids))
    accepted_deletions=len(deleter.states)-1
    pruning_final_material_cm3=deleter.checks['material_cm3']
    # Keep the completed pruning result, then add only a small local finish.
    from wrap_surface_repair import repair
    save_mesh(deleter.mesh,out/'pruned_support.obj')
    refined_mesh,refined_value,refined_proof,refined_checks,refinement=repair(
        deleter.mesh,validator,out/'surface_reinforcement')
    if refinement['changed']:
        deleter.mesh,deleter.value,deleter.proof,deleter.checks=refined_mesh,refined_value,refined_proof,refined_checks
        deleter.save('local_surface_expansion',refinement)
        save_mesh(deleter.mesh,out/'support.obj')
        state_record=json.loads((out/'states.json').read_text())
        state_record.update(complete=True,additive_finishing_stage=True,original_pruning_prefix_unchanged=True)
        C.save(out/'states.json',state_record)
    inputs.verify()
    for path,sha in inputs.protected.items():
        if C.digest(C.ROOT/path)!=sha:raise ValueError('Input changed: '+path)
    report=dict(complete=True,passed=True,schema='coopt_connected_wrap_delete_v1',
        object=args.object,id=args.case['id'],pose_count=args.case['pose_count'],
        experiment=args.experiment,pipeline='connected_hull_wrap_then_progressive_deletion_then_local_surface_expansion',
        selected_step4_source=C.relative(data['selected']),selected_step4_experiment=args.case['experiment'],
        support='support.obj',support_sha256=C.digest(out/'support.obj'),
        initial_support='initial_support.obj',initial_support_sha256=C.digest(out/'initial_support.obj'),
        initial_material_cm3=deleter.states[0]['material_cm3'],final_material_cm3=deleter.checks['material_cm3'],
        initial_bbx_l=deleter.states[0]['bbx_l'],final_bbx_l=deleter.checks['bbx_l'],
        checks=deleter.checks,demands=demands,construction=construction,
        baseline_source=C.relative(data['selected']/'support.obj'),baseline_checks=baseline_checks,
        optional_complete_reference=C.relative(previous) if previous else None,
        initial_step4_core_loss_m3=core_loss,
        accepted_deletions=accepted_deletions,attempts=len(deleter.history),
        ring_perimeter_protected=False,contact_material_frozen=False,layout_changed=False,
        reconstruction_after_pruning=False,material_added_after_pruning_started=refinement['changed'],
        pruning_final_material_cm3=pruning_final_material_cm3,local_surface_reinforcement=refinement,
        step4_rerun=False,original_demands_per_pose=32768,seconds=time.monotonic()-began,
        provenance=dict(inputs=inputs.protected),
        executed_frozen_sources=getattr(sys,'wrap_executed_sources',{}))
    report['construction_reuse']=construction_reuse
    report['execution_launcher_sha256']=payload.get('execution_launcher_sha256')
    report['construction_parameters']=dict(connector_radius_m=geometry.CONNECTOR_RADIUS_M,
        ground_margin_scale=geometry.GROUND_MARGIN_SCALE,terminal_foot_width_m=geometry.TERMINAL_FOOT_WIDTH_M,
        hull_band_width_m=INITIAL_FOOT_WIDTH_M,foot_height_m=INITIAL_FOOT_HEIGHT_M)
    report['dimensions']=dimensions()
    report['width_policy']=dict(hard_minimum_enforced=False,
        method='coarse_outside_first_pruning_with_retained_volume_budget',
        retained_volume_fraction=dimensions()['retained_volume_fraction'],
        visual_goal='avoid excessive thinning',width_or_strength_certificate=False)
    report['execution_source_snapshot']=args.source_snapshot
    np.savez_compressed(out/'force_certificate.npz',**{f'pose_{k}':v for k,v in deleter.proof['masks'].items()})
    report['artifacts']={name:C.digest(out/name) for name in
        ['support.obj','initial_support.obj','pruned_support.obj','states.json','deletions.json',
         'force_certificate.npz','surface_reinforcement/report.json']}
    if refinement['changed']:
        report['artifacts']['surface_reinforcement/added_material.obj']=C.digest(out/'surface_reinforcement/added_material.obj')
    C.save(out/'data/report.json',report)
    print('COMPLETE',args.case['id'],report['accepted_deletions'],report['seconds'],flush=True)
    return report


if __name__=='__main__':
    payload=json.loads(Path(sys.argv[1]).read_text())
    try:
        if payload.get('action')=='reinforce':
            from wrap_reinforce_worker import worker as reinforce
            reinforce(payload)
        elif payload.get('action')=='width_audit':
            from wrap_width_audit import audit
            audit(payload)
        elif payload.get('action')=='audit':
            from wrap_audit import audit
            audit(payload)
        else:worker(payload)
    except BaseException as error:
        out=Path(payload['output']);out.mkdir(parents=True,exist_ok=True)
        name='data/width_audit_failure.json' if payload.get('action')=='width_audit' else 'failure.json'
        C.save(out/name,dict(complete=True,passed=False,error=str(error),traceback=traceback.format_exc()))
        raise
