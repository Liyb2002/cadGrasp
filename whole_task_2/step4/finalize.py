"""Recheck and condition a saved trial without rerunning its population search.

The original search report, runner sources, layout and raw core are retained
under diagnostics/before_export_conditioning before any export is replaced.
"""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
from whole_task_2.step1.common import DEFAULT_OUTPUT,write_json,sha256
from whole_task_2.step3.legacy import kernel as K,loaded_sources
from .adapter import TaskModel
from .search import lock_seats
from .export_geometry import condition_core
from .render import render_support
from whole_search.model import Layout
from whole_step4_render import NominalBuilder
from whole_search.common import unpack_solid
from connected_core import largest_material_component
from material_force import certify_material_contacts
from whole_task_2.common.wrenches import endpoint_demands


def finalize(folder):
    began=time.monotonic();folder=Path(folder)
    report=json.loads((folder/'report.json').read_text())
    if report['nominal_complete_solid_accepted']:
        raise ValueError('Complete solids require their separate full-solid audit; this command conditions incomplete trials')
    archive=folder/'diagnostics/before_export_conditioning'
    if not archive.exists():
        archive.mkdir(parents=True)
        shutil.copytree(Path(__file__).parent,archive/'sources',ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(folder/'report.json',archive/'report.json')
        for path in (folder/'step4.2').glob('*'):
            if path.is_file()and path.suffix in ('.obj','.npz'):shutil.copy2(path,archive/path.name)
    old=json.loads((archive/'report.json').read_text())
    source_root=Path(__file__).parent.resolve();search_sources={}
    for filename,digest in old['input_sha256'].items():
        path=Path(filename)
        archived=archive/'sources'/path.name if path.parent==source_root else path
        if sha256(archived)!=digest:raise ValueError(f'Changed original search input: {filename}')
        if path.parent==source_root:
            search_sources[filename]=dict(sha256=digest,snapshot=str(archived.relative_to(folder)))
    model=TaskModel(report['poses'],report['object'],report['output_root'])
    with np.load(archive/'layout.npz')as z:original=Layout.from_arrays(z)
    layout=lock_seats(model,original)
    difference=float(np.max(np.abs(original.placements-layout.placements)))
    if difference>1e-12:raise ValueError('Canonicalization would change the actual task layout')
    rebuilt=NominalBuilder(model).solid(layout)
    rebuilt,_=largest_material_component(rebuilt)
    out=folder/'step4.2'
    core,conditioning=condition_core(model,layout,unpack_solid(rebuilt),out/'support.obj')
    np.savez_compressed(out/'layout.npz',**layout.arrays(model.native))
    proof=certify_material_contacts(model,layout,core)
    endpoints={}
    for k in layout.active:
        demand=endpoint_demands(model.tasks[k].domain)['endpoint_need_wrench']*model.tasks[k].scale
        mask,_=K.J.classify(proof['supplies'][k],demand)
        endpoints[model.poses[k]]=dict(passed=int(mask.sum()),total=len(mask))
        np.savez_compressed(out/f'{model.poses[k]}_force.npz',mask=proof['masks'][k],
                            supply_7d=proof['supplies'][k],endpoint_mask=mask)
    render_support(model,layout,core,out/'overview.png')
    report.update(final_core_counts={model.poses[k]:int(v.sum())for k,v in proof['masks'].items()},
        continuous_endpoint_counts=endpoints,core_volume_cm3=abs(float(core.volume))*1e6,
        export_conditioning=dict(conditioning,maximum_transform_canonicalization=difference,
                                 nominal_core_rebuilt_from_original_inputs=True),
        search_execution_source_sha256=search_sources,
        search_execution_report=str((archive/'report.json').relative_to(folder)),
        export_postprocessing_seconds=time.monotonic()-began,
        current_sources_include_postprocessing_revision=True)
    report['input_sha256']={p:sha256(p)for p in report['input_sha256']}
    for path in [*loaded_sources(),*source_root.glob('*.py')]:report['input_sha256'][str(path)]=sha256(path)
    report['artifact_sha256']={str(p.relative_to(folder)):sha256(p)for p in folder.rglob('*')
        if p.is_file()and p not in (folder/'report.json',folder/'validation.json',folder/'run_status.json')}
    write_json(folder/'report.json',report)
    print(json.dumps(dict(counts=report['final_core_counts'],conditioning=report['export_conditioning']),indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--object',default='B');a=p.parse_args();finalize(a.output_root/a.object/'step4')


if __name__=='__main__':main()
