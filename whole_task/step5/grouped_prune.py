"""Prune the accepted grouped fixture, reusing its existing compact ground feet.

The original Deleter proposal family, 55% retained-volume budget, complete
load targets and actual-solid Validator are unchanged. No ground regrowth.
"""
import argparse
import json
from pathlib import Path
import time
import shutil
import tempfile
import numpy as np
from whole_task.step1.common import DEFAULT_OUTPUT,write_json,sha256
from whole_task.step3.legacy import kernel as K,loaded_sources
from whole_task.step4.export_geometry import condition_core
from .source_model import restore
from .geometry import SparseFeet,Validator,save_mesh,solid
from .prune import Deleter


def solve(step4,out,grids=(3,5),video=True,foot_width_mm=5.):
    start=time.monotonic();step4=Path(step4).resolve();out=Path(out).resolve()
    if (out/'report.json').exists():raise ValueError('Output already exists: '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    old=json.loads((step4/'report.json').read_text())
    if not old['full_fixture_accepted']:raise ValueError('Step4 has not passed')
    for path,digest in old['input_sha256'].items():
        if sha256(path)!=digest:raise ValueError('Changed Step4 input: '+path)
    for path,digest in old['artifact_sha256'].items():
        if sha256(step4/path)!=digest:raise ValueError('Changed Step4 artifact: '+path)
    model,layout=restore(old,step4/'step4.2/layout.npz')
    core=K.trimesh.load(step4/'step4.2/support.obj',force='mesh',process=False)
    witness=K.trimesh.load(step4/'step4.2/legal_component.obj',force='mesh',process=False)
    reference=K.trimesh.load(step4/'step4.2/complete_support.obj',force='mesh',process=False)
    # Step4 already constructed compact pads and one accepted connected solid.
    # Re-growing its feet from the contact core needlessly enlarges the station.
    feet=[row for row in old['construction']['history'] if row.get('operation')=='ground_patches']
    if not feet or not all(row.get('compact_feet_used') for row in feet):
        raise ValueError('This grouped entry requires certified compact Step4 feet')
    print('REUSE accepted compact Step4 stock',flush=True)
    stock=reference.copy()
    shutil.copy2(step4/'step4.2/complete_support.obj',out/'stock.obj')
    conditioning=dict(input_stock_serialization_preserved=True,
                      existing_step4_feet_reused=True,new_ground_material_generated=False)
    validator=Validator(model,layout,witness,stock)
    value,proof,checks=validator.check(stock)
    construction=dict(passed=True,connected=True,existing_step4_feet_reused=True,
        source_complete_support_sha256=sha256(out/'stock.obj'),
        perimeter_ring_generated=False,new_ground_material_generated=False,
        material_cm3=checks['material_cm3'],actual_conditioned_stock_certified=True)
    write_json(out/'construction.json',construction)
    write_json(out/'run_status.json',dict(status='pruning',construction=construction,initial_checks=checks))
    cutter=Deleter(stock,value,proof,checks,validator,out)
    cutter.run(grids)
    np.savez_compressed(out/'force_certificate.npz',**{f'sample_mask_{k}':v for k,v in cutter.proof['masks'].items()},
        **{f'endpoint_mask_{k}':v for k,v in cutter.proof['endpoint_masks'].items()},
        **{f'supply_{k}':v for k,v in cutter.proof['supplies'].items()})
    shutil.copy2(step4/'step4.2/layout.npz',out/'layout.npz')
    shutil.copy2(step4/'execution.json',out/'execution.json')
    report=dict(schema='whole_task_step5_sparse_feet_v1',status='passed',step4=str(step4),object=model.name,
        poses=model.poses,output_root=old['output_root'],
        pose_model='continuous_axial' if hasattr(layout,'angles_deg') else 'fixed',
        input_root=old.get('input_root',old['output_root']),
        construction=construction,conditioning=conditioning,
        initial_material_cm3=abs(float(stock.volume))*1e6,step4_complete_material_cm3=abs(float(reference.volume))*1e6,
        final_checks=cutter.checks,accepted_deletions=len(cutter.states)-1,attempted_deletions=len(cutter.history),
        proposal_grids=list(grids),retained_fraction_heuristic=.55,perimeter_ring_generated=False,initial_foot_width_mm=foot_width_mm,
        continuous_force_certificate=True,minimum_wall_or_strength_certified=False,
        seconds=time.monotonic()-start,
        input_sha256={str(p):sha256(p) for p in set(model.inputs+loaded_sources()+[step4/'report.json',step4/'step4.2/layout.npz',step4/'step4.2/support.obj',step4/'step4.2/complete_support.obj',step4/'step4.2/legal_component.obj'])},
        artifacts={name:sha256(out/name) for name in ['support.obj','layout.npz','execution.json','force_certificate.npz']})
    write_json(out/'report.json',report)
    from whole_task.evaluation.run import evaluate
    evaluation=evaluate(out,model,layout)
    report['evaluation']=evaluation
    write_json(out/'report.json',report)
    if video:
        from .movie import make_movie
        report['video']=make_movie(out,model,layout)
    report['total_seconds']=time.monotonic()-start
    write_json(out/'report.json',report)
    from .validate import validate
    validate(out)
    write_json(out/'run_status.json',dict(status='passed',seconds=report['total_seconds']))
    print('RESULT',json.dumps(dict(material_cm3=cutter.checks['material_cm3'],bbx_l=cutter.checks['bbx_l'],seconds=report['total_seconds'])),flush=True)
    return report

