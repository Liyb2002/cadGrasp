"""Reuse the exact completed wrap; recheck native physics before new pruning."""
from pathlib import Path
import json
import shutil
import numpy as np
from wrap_dimensions import dimensions


def reuse(source,data,validator,out,inputs):
    from feasible_geometry import C
    source=Path(source).resolve()
    report_path=source/'data/report.json'
    report=json.loads(report_path.read_text())
    construction=json.loads((source/'construction.json').read_text())
    if not report['complete'] or not report['passed'] or not construction['complete']:
        raise ValueError('Construction source is not complete')
    if report['selected_step4_source']!=C.relative(data['selected']):
        raise ValueError('Construction source has a different fixed layout')
    current=dimensions()
    for key in ['initial_foot_width_m','initial_foot_height_m',
                'initial_foot_outward_offset_m','initial_foot_inward_offset_m']:
        if report['dimensions'][key]!=current[key]:raise ValueError('Construction dimensions changed: '+key)
    original=C.trimesh.load(source/'construction/000.obj',force='mesh',process=False)
    if not (np.array_equal(original.vertices,data['fixture'].vertices)
            and np.array_equal(original.faces,data['fixture'].faces)):
        raise ValueError('Reused construction comes from a different Step4 core')
    if C.digest(source/'initial_support.obj')!=report['initial_support_sha256']:
        raise ValueError('Construction source mesh changed')
    mesh=C.trimesh.load(source/'initial_support.obj',force='mesh',process=False)
    value,proof,checks=validator.check(mesh)
    files=[source/'initial_support.obj',source/'construction.json',*sorted((source/'construction').glob('*.obj'))]
    hashes={C.relative(p):C.digest(p) for p in [report_path,*files]}
    inputs.protected.update(hashes)
    inputs.protected.update(report['provenance']['inputs'])
    for p in files:
        target=out/p.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    history=construction['history']
    for row in history:
        if 'mesh' in row:
            relative=(C.ROOT/row['mesh']).relative_to(source)
            row['mesh']=C.relative(out/relative)
    result=dict(source=C.relative(source),source_artifacts=hashes,
                initial_support_sha256=report['initial_support_sha256'],
                construction_parameters=report['construction_parameters'],
                prior_pruning_result_reused=False,native_guards_recomputed=True)
    C.save(out/'data/construction_reuse.json',result)
    print('REUSED_EXACT_CONSTRUCTION',C.relative(source),flush=True)
    return mesh,value,proof,checks,history,result
