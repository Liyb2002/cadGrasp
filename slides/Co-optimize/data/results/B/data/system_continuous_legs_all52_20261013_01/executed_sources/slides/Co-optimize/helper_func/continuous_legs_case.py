"""Rebuild Step5.3 only, keeping accepted Step4/5.1/5.2 bytes immutable."""
import json
import sys
import time
from pathlib import Path
from run_dynamic_postprocess import ROOT, COOPT, BASE, load_stage, save, sha
sys.path[:0] = [str(COOPT/x) for x in ['step5.2','step5.3','helper_func']]
import numpy as np
import stage5_shared as C
from whole_search.model import Layout
from prune_support import ForceCertificate, material_components
from deterministic_base import solid, volume, enclosure
from continuous_legs import Builder
from export_stability import stabilize_export


def pictures(data, mesh, previous, output):
    output.mkdir(parents=True, exist_ok=True)
    renderer=C.V.Renderer()
    load_stage('step5.3').R.reference(renderer,data,output/'overview.png',fixture=mesh)
    panels=[]
    for row in data['rows']:
        panel,_=C.H.render_pose(renderer,C.posed(data['body'],row['object_transform']),
            C.posed(mesh,row['fixture_transform']),row['cloud'],row['color'],
            output/(row['pose']+'.png'),hull=row['hull'],size=700)
        panels.append(panel)
    C.V.sheet(panels,output/'all_poses.png')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    state=data['states'][0]
    meshes=[C.posed(m,state) for m in [previous,mesh]]
    camera=C.V.camera_for(np.vstack([m.vertices for m in meshes]))
    panels=[renderer.render([C.V.CPU.placed(C.V.piece(m,C.V.BLUE))],camera,800) for m in meshes]
    fig,axes=plt.subplots(1,2,figsize=(12,6))
    for ax,panel,title in zip(axes,panels,['Previous Step5.3','Continuous solid legs']):
        ax.imshow(np.asarray(panel));ax.set_title(title);ax.axis('off')
    fig.suptitle(data['case']['id']+' | same camera, scale and BBX')
    fig.tight_layout();fig.savefig(output/'comparison.png',dpi=150);plt.close(fig)
    return {p.name:sha(p) for p in output.glob('*.png')}


def run_case(case, previous, selection, experiment, render=True):
    began=time.monotonic()
    inputs=C.Inputs('B',selection,Path(previous['step51_source']).name)
    data=inputs.load(case)
    old_folder=ROOT/previous['step53_source']
    old_report_path=old_folder/'data/report.json'
    old_report=json.loads(old_report_path.read_text())
    core_folder=ROOT/previous['step52_source']
    core_report_path=core_folder/'data/report.json'
    core_report=json.loads(core_report_path.read_text())
    if old_report['step52_source'] != previous['step52_source']:
        raise ValueError('Previous final belongs to a different core')
    if not old_report['passed'] or not core_report['passed'] or not C.connected_core_policy(data):
        raise ValueError('Only the accepted connected-core selection is in scope')
    paths=[old_report_path,core_report_path,old_folder/old_report['support'],
           core_folder/core_report['support'],data['selected']/'oracle_connected_witness.obj']
    for path in paths:inputs.protected[C.relative(path)]=sha(path)
    core=C.trimesh.load(paths[3],force='mesh',process=False)
    old=C.trimesh.load(paths[2],force='mesh',process=False)
    witness=C.trimesh.load(paths[4],force='mesh',process=False)
    bounds=C.occupied_bounds(data['body'],old,data['rows'])
    builder=Builder(core,old,witness,data['body'],data['rows'],bounds)
    final,construction=builder.run()
    with np.load(data['selected']/'layout.npz') as arrays:layout=Layout.from_arrays(arrays)
    checker=ForceCertificate(data['record']['poses'],layout,data['body'],data['rows'])
    final,proof,export=stabilize_export(final,witness,data['body'],
        layout.placements[list(layout.active)],data['rows'],checker,final_support=True,max_cycles=0)
    actual=solid(final);mandatory=solid(core)
    actual_bounds=C.occupied_bounds(data['body'],final,data['rows'])
    checks=dict(core_loss_m3=volume(mandatory-actual),components=len(material_components(actual)),
        max_bbx_bounds_delta_m=float(np.max(abs(actual_bounds-bounds))),
        bbx_volume_delta_l=abs(C.volume(actual_bounds)-C.volume(bounds))*1000,
        material_delta_cm3=volume(actual)*1e6-float(old.volume)*1e6,
        floor_passed=all(r['passed'] for r in enclosure(final,data['rows'])))
    if (checks['core_loss_m3']>1e-10 or checks['components']!=1 or
        checks['max_bbx_bounds_delta_m']>1e-8 or checks['bbx_volume_delta_l']>1e-8 or
        checks['material_delta_cm3']>1e-4 or not checks['floor_passed'] or not proof['passed']):
        raise ValueError('Final continuous-leg guard failed: '+str(checks))
    output=BASE/case['id']/'step5/step5.3'/experiment
    output.mkdir(parents=True,exist_ok=False)
    final.export(output/'support_with_base.obj',digits=16)
    np.savez_compressed(output/'force_certificate.npz',**{'mask_'+str(k):v for k,v in proof['masks'].items()})
    construction.update(material_cm3=float(final.volume)*1e6,
        outside_saved_legal_component_cm3=export['history'][-1]['outside_saved_witness_cm3'],
        actual_object_overlap_cm3=export['history'][-1]['actual_object_overlap_cm3'],
        serialized_export_verification=export,connected_core_policy=True,
        final_guards=checks,locked_final_bbx_unchanged=True,
        added_vs_previous_cm3=volume(actual-solid(old))*1e6,
        removed_vs_previous_cm3=volume(solid(old)-actual)*1e6)
    visual=output/'images'
    published=pictures(data,final,old,visual) if render else {}
    inputs.verify()
    code_paths=[Path(__file__),COOPT/'step5.3/continuous_legs.py',COOPT/'step5.2/prune_support.py',
        COOPT/'step5.3/deterministic_base.py',COOPT/'helper_func/export_stability.py']
    report=dict(complete=True,passed=True,stage_type='final_support_construction',
        object='B',id=case['id'],poses=data['record']['poses'],pose_count=case['pose_count'],
        selected_step4_experiment=case['experiment'],selected_step4_source=C.relative(data['selected']),
        step52_source=previous['step52_source'],support='support_with_base.obj',construction=construction,
        previous_step53_source=previous['step53_source'],pruned_material_cm3=float(core.volume)*1e6,
        leg_profile='continuous_rounded_solid',force_solver_run=True,force_counts=proof['counts'],
        step4_rerun=False,step51_rerun=False,step52_rerun=False,layout_changed=False,
        full_ground_friction_yaw_solved=False,fixture_weight_ratio=0.,published_images=published,
        images_source=C.relative(visual),seconds=time.monotonic()-began,
        provenance=dict(inputs=inputs.protected,code={C.relative(p):sha(p)for p in code_paths}),
        artifacts={name:sha(output/name)for name in ['support_with_base.obj','force_certificate.npz']})
    save(output/'data/report.json',report)
    return dict(id=case['id'],pose_count=case['pose_count'],experiment=case['experiment'],
        passed=True,status='PASS',floor=dict(passed=True,output=previous['step51_source'],reused_saved_stage=True),
        pruning=dict(passed=True,output=previous['step52_source'],reused_saved_stage=True),
        construction=dict(passed=True,output=C.relative(output)),postprocess_experiment=experiment,
        previous_material_cm3=float(old.volume)*1e6,material_cm3=float(final.volume)*1e6,
        seconds=time.monotonic()-began)
