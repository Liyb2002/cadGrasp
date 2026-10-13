"""Certify genuinely rounded bodies and feet without increasing saved material."""
from pathlib import Path
import json,sys,time
import numpy as np
from run_dynamic_postprocess import ROOT,COOPT,BASE,load_stage,sha,save
sys.path[:0]=[str(COOPT/x)for x in ['step5.2','step5.3','helper_func']]
import stage5_shared as C
from sculpted_pruning import sharp_edges
from serialized_guard_pruner import SerializedGuardPruner as SculptedPruner
from prune_support import ForceCertificate,material_components
from deterministic_base import solid,unpack,volume,enclosure,union,SCALE
from whole_search.model import Layout
from export_stability import stabilize_export


def saved_leg_stock(core,final,construction,rows):
    """Select saved legs using their construction envelopes, with overlap.

    Avoid subtracting two nearly coincident complete bodies. Slightly padded
    original proposals only select stock from the saved final solid.
    """
    import manifold3d as md
    from flat_tapered_base import flat_leg
    frames=list({np.round(r['fixture_transform'],11).tobytes():r['fixture_transform']for r in rows}.values())
    proposals=[];ball=md.Manifold.sphere(.00025/SCALE,12)
    for h in construction['history']:
        if h['operation']!='flat_tapered_support_leg':continue
        root=np.asarray(h['root_m']);tip=np.asarray(h['tip_m'])
        for T in frames:
            if abs((T[:3,:3]@tip+T[:3,3])[2])>1e-8:continue
            for angle in [0.,np.pi/4,np.pi/2]:
                proposals.append(flat_leg(root,tip,T[2,:3],h['root_width_m'],h['root_thickness_m'],
                    h['tip_width_m'],h['tip_thickness_m'],angle).minkowski_sum(ball))
    region=union(proposals);original=solid(core);feet=[]
    for T in frames:
        feet.append(original.trim_by_plane(tuple(-T[2,:3]),float((T[2,3]+.0006)/SCALE)))
    roots=union([original^region]+feet)
    return unpack(solid(final)^region),unpack(roots)


def publish(data,mesh,stage,source=None):
    visual=COOPT/'output/B'/data['case']['id']/stage;visual.mkdir(parents=True,exist_ok=True)
    renderer=C.V.Renderer();R=load_stage('step5.3').R
    R.reference(renderer,data,visual/'overview.png',fixture=mesh)
    tiles=[]
    for row in data['rows']:
        tile,_=C.H.render_pose(renderer,C.posed(data['body'],row['object_transform']),
            C.posed(mesh,row['fixture_transform']),row['cloud'],row['color'],visual/(row['pose']+'.png'),hull=row['hull'],size=700)
        tiles.append(tile)
    C.V.sheet(tiles,visual/'all_poses.png')
    if source is not None:
        removed=unpack(solid(source)-solid(mesh));T=data['states'][0]
        camera=C.V.camera_for(C.posed(source,T).vertices)
        parts=[C.V.piece(C.posed(mesh,T),C.V.BLUE)]
        if len(removed.faces):parts.append(C.V.piece(C.posed(removed,T),'#CB5B64',.48))
        renderer.render([C.V.CPU.placed(p)for p in parts],camera,900).save(visual/'removed_material.png')
    return {p.name:sha(p)for p in visual.glob('*.png')}


def run_case(case,args,previous):
    began=time.monotonic();inputs=C.Inputs('B',args.selection_summary,args.step51_experiment);data=inputs.load(case)
    if not C.connected_core_policy(data):raise ValueError('Only accepted connected Step4 layouts are in scope')
    def old_mesh(key):
        folder=ROOT/previous[key];path=folder/'data/report.json';record=json.loads(path.read_text())
        if not record['passed']or record['selected_step4_source']!=C.relative(data['selected']):raise ValueError('Prior geometry belongs to a different or failed layout')
        mesh_path=folder/record['support']
        for p in [path,mesh_path]:inputs.protected[C.relative(p)]=sha(p)
        return C.trimesh.load(mesh_path,force='mesh',process=False),record,folder
    old_core,old_prune,prune_reference=old_mesh('step52_source')
    old_final,old_construction,final_reference=old_mesh('step53_source')
    if old_construction['step52_source']!=previous['step52_source']:raise ValueError('Prior body and feet dependency mismatch')
    shape_reference=ROOT/old_prune['shape_reference'];reference_path=shape_reference/'data/report.json'
    reference=json.loads(reference_path.read_text());reference_mesh=shape_reference/reference['support']
    if not reference['passed']or reference['selected_step4_source']!=C.relative(data['selected']):raise ValueError('Compact guide belongs to another or failed layout')
    for p in [reference_path,reference_mesh]:inputs.protected[C.relative(p)]=sha(p)
    minimal_core=C.trimesh.load(reference_mesh,force='mesh',process=False)
    leg_stock,root_stock=saved_leg_stock(old_core,old_final,old_construction['construction'],data['rows'])
    witness_path=data['selected']/'oracle_connected_witness.obj';inputs.protected[C.relative(witness_path)]=sha(witness_path)
    witness=C.trimesh.load(witness_path,force='mesh',process=False)
    with np.load(data['selected']/'layout.npz')as z:layout=Layout.from_arrays(z)
    checker=ForceCertificate(data['record']['poses'],layout,data['body'],data['rows'])
    modules=[Path(__file__),COOPT/'step5.2/serialized_guard_pruner.py',COOPT/'step5.2/sculpted_pruning.py',COOPT/'step5.2/surface_flow.py',COOPT/'step5.2/smooth_edge_fillet.py',COOPT/'step5.2/prune_support.py',
        COOPT/'step5.3/deterministic_base.py',COOPT/'step5.3/flat_tapered_base.py',COOPT/'helper_func/export_stability.py']
    source=dict(inputs=inputs.protected,code={C.relative(p):sha(p)for p in modules})
    result=dict(id=case['id'],pose_count=case['pose_count'],experiment=case['experiment'],
        floor=dict(passed=True,output=previous['step51_source'],reused_saved_stage=True))
    body_sculptor=SculptedPruner(old_core,witness,checker,rows=data['rows'],object_vertices=data['body'].vertices,
        require_connected=True,future_legs=leg_stock,compaction_guard=root_stock,guide_reference=minimal_core)
    core,pruning,proof=body_sculptor.run()
    core,proof,check=stabilize_export(core,witness,data['body'],layout.placements[list(layout.active)],
        data['rows'],checker,final_support=False)
    target=C.occupied_bounds(data['body'],old_core,data['rows'])
    def verify(mesh,original,mandatory=None,final=False):
        value=solid(mesh);expected=C.occupied_bounds(data['body'],original,data['rows']);actual=C.occupied_bounds(data['body'],mesh,data['rows'])
        checks=dict(added_m3=volume(value-solid(original)),components=len(material_components(value)),
            max_bbx_bounds_delta_m=float(np.max(abs(actual-expected))),bbx_volume_delta_l=abs(C.volume(actual)-C.volume(expected))*1000,
            mandatory_loss_m3=volume(solid(mandatory)-value)if mandatory is not None else 0.,
            floor_passed=not final or all(r['passed']for r in enclosure(mesh,data['rows'])))
        if (checks['added_m3']>1e-10 or checks['components']!=1 or checks['max_bbx_bounds_delta_m']>1e-8 or
                checks['bbx_volume_delta_l']>1e-8 or checks['mandatory_loss_m3']>1e-10 or not checks['floor_passed']):
            raise ValueError('Final serialized geometry changed stock, core, BBX or floor enclosure: '+str(checks))
    def select_stable(sculptor,record,original,mandatory=None,final=False,future=False):
        choices=[(sculptor.best_mesh,len(sculptor.history))]+list(reversed(sculptor.checkpoints))+[(original,0)]
        errors=[]
        for mesh,index in choices:
            try:
                actual,certificate,export=stabilize_export(mesh,witness,data['body'],layout.placements[list(layout.active)],
                    data['rows'],checker,final_support=final,max_cycles=0)
                verify(actual,original,mandatory=mandatory,final=final)
                next_stock=None
                if future:
                    next_stock=old_final if index==0 else unpack(solid(actual)+solid(leg_stock))
                    for repair in range(4):
                        try:
                            verify(next_stock,old_final,mandatory=actual,final=True);break
                        except ValueError:
                            if repair==3:raise
                            next_stock=unpack(solid(next_stock)+solid(actual))
                applied=[h for h in sculptor.history[:index]if h.get('accepted')]
                rounds=[float(h['operation'].removeprefix('local_edge_fillet_'))for h in applied if h['operation'].startswith('local_edge_fillet_')]
                record.update(retained_material_cm3=volume(solid(actual))*1e6,removed_material_cm3=volume(solid(original)-solid(actual))*1e6,
                    added_material_cm3=volume(solid(actual)-solid(original))*1e6,counts=certificate['counts'],contact_triangles=certificate['contact_triangles'],
                    rounding_radius_m=max(rounds)if rounds else None,accepted_edge_fillets=len(rounds),edge_fillet_radii_m=rounds,
                    selected_history_entries=index,serialization_rollbacks=errors,
                    compaction_radius_m=next((float(h['operation'].removeprefix('compact_rounded_block_'))for h in applied if h['operation'].startswith('compact_rounded_block_')),None),
                    serialized_export_verification=export,sharp_edges_after=sharp_edges(actual))
                if not any(h['operation'].startswith('surface_flow_')for h in applied):record['surface_flow']=None
                return actual,certificate,export,next_stock
            except ValueError as error:errors.append(dict(history_entries=index,error=str(error)))
        raise ValueError('No serialized stable rounded candidate: '+str(errors))
    core,proof,check,final_stock=select_stable(body_sculptor,pruning,old_core,future=True)
    pruning.update(retained_material_cm3=volume(solid(core))*1e6,serialized_export_verification=check,
        sharp_edges_after=sharp_edges(core),outside_saved_legal_component_cm3=check['history'][-1]['outside_saved_witness_cm3'],
        actual_object_overlap_cm3=check['history'][-1]['actual_object_overlap_cm3'])
    out=BASE/case['id']/'step5/step5.2'/args.experiment;out.mkdir(parents=True,exist_ok=False)
    core.export(out/'pruned_support.obj',digits=16)
    np.savez_compressed(out/'force_certificate.npz',**{**{'mask_'+str(k):v for k,v in proof['masks'].items()},
        **{'supply_'+str(k):v for k,v in proof['supplies'].items()}})
    pictures=publish(data,core,'step5.2',data['fixture']);inputs.verify()
    prune_report=dict(complete=True,passed=True,stage_type='support_pruning',connected_core_policy=True,
        object='B',id=case['id'],poses=data['record']['poses'],pose_count=case['pose_count'],
        selected_step4_experiment=case['experiment'],selected_step4_source=C.relative(data['selected']),
        step51_source=previous['step51_source'],source_contact_material=C.relative(prune_reference/old_prune['support']),
        support='pruned_support.obj',pruning=pruning,pruning_style='sculpted',shape_reference=C.relative(shape_reference),
        force_passed=True,force_solver_run=True,original_demand_count_per_pose=32768,step4_rerun=False,
        step51_rerun=False,layout_changed=False,final_base_constructed=False,published_images=pictures,provenance=source,
        artifacts={name:sha(out/name)for name in ['pruned_support.obj','force_certificate.npz']})
    save(out/'data/report.json',prune_report)
    result['pruning']=dict(id=case['id'],passed=True,pose_count=case['pose_count'],output=C.relative(out),protected=inputs.protected,
        retained_material_cm3=pruning['retained_material_cm3'],rounding_radius_m=pruning['rounding_radius_m'])
    for p in [out/'data/report.json',out/'pruned_support.obj',out/'force_certificate.npz']:
        inputs.protected[C.relative(p)]=sha(p)
    # Keep only this new core and the previously certified feet: never restore
    # the old core which Step5.2 just removed.
    core=C.trimesh.load(out/'pruned_support.obj',force='mesh',process=False)
    verify(final_stock,old_final,mandatory=core,final=True)
    foot_sculptor=SculptedPruner(final_stock,witness,checker,rows=data['rows'],object_vertices=data['body'].vertices,
        require_connected=True,mandatory=core,final_support=True,radii=(.0018,.0012,.0007))
    final,fillet,force=foot_sculptor.run()
    final,force,final_check,_=select_stable(foot_sculptor,fillet,final_stock,mandatory=core,final=True)
    verify(final,old_final,mandatory=core,final=True)
    if volume(solid(final)-solid(final_stock))>1e-10:raise ValueError('Final cleanup restored deleted body or added new feet')
    construction=dict(old_construction['construction']);construction.update(passed=True,connected=True,watertight=True,
        material_cm3=volume(solid(final))*1e6,added_material_cm3=volume(solid(final)-solid(core))*1e6,
        retained_legal_step4_material_loss_cm3=volume(solid(core)-solid(final))*1e6,
        outside_saved_legal_component_cm3=final_check['history'][-1]['outside_saved_witness_cm3'],
        actual_object_overlap_cm3=final_check['history'][-1]['actual_object_overlap_cm3'],per_pose=final_check['pressure_enclosure'],
        serialized_export_verification=final_check,connected_core_policy=True,saved_leg_geometry_reconstructed=True,
        leg_paths_and_sections_unchanged=False,leg_edges_rounded_by_deletion=True,locked_final_bbx_unchanged=True,
        original_core_not_restored=True,fillet=fillet)
    construction['history']=list(construction['history'])+[dict(operation='round_saved_legs_by_deletion',
        rounding_radius_m=fillet['rounding_radius_m'],body_loss_cm3=volume(solid(core)-solid(final))*1e6)]
    final_out=BASE/case['id']/'step5/step5.3'/args.experiment;final_out.mkdir(parents=True,exist_ok=False)
    final.export(final_out/'support_with_base.obj',digits=16)
    np.savez_compressed(final_out/'force_certificate.npz',**{'mask_'+str(k):v for k,v in force['masks'].items()})
    pictures=publish(data,final,'step5.3');inputs.verify()
    report=dict(complete=True,passed=True,stage_type='final_support_construction',object='B',id=case['id'],
        poses=data['record']['poses'],pose_count=case['pose_count'],selected_step4_experiment=case['experiment'],
        selected_step4_source=C.relative(data['selected']),step52_source=C.relative(out),support='support_with_base.obj',
        construction=construction,fixed_leg_reference=C.relative(final_reference),pruned_material_cm3=pruning['retained_material_cm3'],
        leg_profile='sculpted_flat',force_passed_reused_from_step52=False,force_solver_run=True,force_counts=force['counts'],
        step4_rerun=False,step51_rerun=False,full_ground_friction_yaw_solved=False,fixture_weight_ratio=0.,
        published_images=pictures,seconds=time.monotonic()-began,provenance=source,
        artifacts={name:sha(final_out/name)for name in ['support_with_base.obj','force_certificate.npz']})
    save(final_out/'data/report.json',report)
    result.update(construction=dict(id=case['id'],passed=True,pose_count=case['pose_count'],output=C.relative(final_out),protected=inputs.protected,
        material_cm3=construction['material_cm3'],rounding_radius_m=fillet['rounding_radius_m']),passed=True,status='PASS')
    return result
