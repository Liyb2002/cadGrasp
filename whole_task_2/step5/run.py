"""Grow sparse ground feet, prune actual material, evaluate and make the movie."""
import argparse
import json
from pathlib import Path
import time
import shutil
import tempfile
import numpy as np
from whole_task_2.step1.common import ROOT,DEFAULT_OUTPUT,write_json,sha256
from whole_task_2.step3.legacy import kernel as K,loaded_sources
from whole_task_2.step4.export_geometry import condition_core
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
    from .geometry import construction_rows
    rows=construction_rows(model,layout)
    print('CONSTRUCT sparse feet',flush=True)
    builder=SparseFeet(core,witness,rows,width=float(foot_width_mm)/1000)
    stock,construction=builder.run()
    write_json(out/'construction.json',construction)
    save_mesh(stock,out/'construction/stock.obj')
    if old.get('continuous_angles_optimized_during_juxtaposition'):
        from whole_task_2.step4.family_gate import condition
        from prune_support import material_components
        from whole_search.common import unpack_solid
        stock,conditioning=condition(model,layout,stock,out/'construction/stock.obj',False,witness)
        parts=material_components(solid(stock))
        if len(parts)>1:
            retained=max(parts,key=lambda part:abs(float(part.volume())))
            before=abs(float(stock.volume))*1e6
            stock=unpack_solid(retained.simplify(0.))
            stock,repair=condition(model,layout,stock,out/'construction/stock.obj',True,witness)
            conditioning.update(detached_material_removed_cm3=before-abs(float(stock.volume))*1e6,
                                retained_component_conditioning=repair)
    else:
        stock,conditioning=condition_core(model,layout,stock,out/'construction/stock.obj')
    validator=Validator(model,layout,witness,stock)
    value,proof,checks=validator.check(stock)
    construction['raw_builder_passed']=construction['passed']
    construction.update(passed=True,actual_conditioned_stock_certified=True,
        material_cm3=checks['material_cm3'],outside_saved_legal_component_cm3=checks['outside_m3']*1e6,
        connected=checks['components']==1)
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
    from whole_task_2.evaluation.run import evaluate
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


def run(step4,out,grids=(3,5),video=True,foot_width_mm=5.):
    # Intermediate meshes, checkpoints and render inputs are temporary.
    with tempfile.TemporaryDirectory(prefix='cadgrasp_step5_') as temp:
        solve(step4,Path(temp),grids,video,foot_width_mm)
        from .publish import publish
        report,data=publish(temp,out)
    from .validate import validate
    validate(data)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--step4',type=Path)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT/'B/step5')
    p.add_argument('--grids',type=int,nargs='+',default=[3,5])
    p.add_argument('--no-video',action='store_true')
    p.add_argument('--foot-width-mm',type=float,default=5.,help='Initial pad width/height; default matches original Step4 floor construction')
    a=p.parse_args()
    if a.step4 is None:
        active=json.loads((ROOT/'whole_task_2/data/B/step4_active.json').read_text())
        a.step4=Path(active['workdir'])
    report=json.loads((a.step4/'report.json').read_text())
    if report.get('method')=='independent_membrane_family_merges':
        from .membrane_feet import run as run_membranes
        if a.foot_width_mm!=5.:raise ValueError('Membrane feet currently use the declared 5 mm pad family')
        return run_membranes(a.step4,grids=tuple(a.grids),video=not a.no_video,output=a.output)
    return run(a.step4,a.output,tuple(a.grids),not a.no_video,a.foot_width_mm)


if __name__=='__main__':main()
