"""Review all saved states with the latest width kernel and exact source bytes."""
from pathlib import Path
import json
import time
import numpy as np
from wrap_geometry import C
from wrap_width import ConnectionWidthGuard
from deterministic_base import solid


SEMANTIC_FILES=['wrap_width.py','wrap_width_profiles.py','wrap_neck_width.py','wrap_dimensions.py','wrap_width_audit.py']


def binding(out):
    return {p:C.digest(out/p) for p in ['support.obj','initial_support.obj','states.json','data/width_features.json']}


def audit(payload):
    began=time.monotonic();out=Path(payload['output'])
    report=json.loads((out/'data/report.json').read_text())
    before=binding(out);source=C.ROOT/report['selected_step4_source']/'support.obj'
    core=C.trimesh.load(source,force='mesh',process=False)
    guard=ConnectionWidthGuard.load(out,core)
    states=json.loads((out/'states.json').read_text())['states']
    previous=None;pending=0
    for state in states:
        path=C.ROOT/state['mesh']
        if C.digest(path)!=state['mesh_sha256']:raise ValueError('Width-review checkpoint changed')
        mesh=C.trimesh.load(path,force='mesh',process=False)
        check=guard.assess(mesh) if previous is None else guard.require(mesh,previous)
        pending+=int(not check['absolute_passed']);previous=solid(mesh)
    final=C.trimesh.load(out/'support.obj',force='mesh',process=False)
    check=guard.require(final)
    if before!=binding(out):raise ValueError('Width-review source changed')
    archive=Path(payload['source_snapshot']);index=json.loads((archive/'index.json').read_text())
    prefix='slides/Co-optimize/step5/'
    result=dict(complete=True,passed=True,id=report['id'],artifacts=before,
        source_core=C.relative(source),source_core_sha256=C.digest(source),
        state_count=len(states),initial_clipped_width_pending_states=pending,
        no_new_narrow_connections=True,final_absolute_width_passed=check['absolute_passed'],
        minimum_width_m=check['minimum_width_m'],wall_thickness_constrained=False,
        measurement_revision=check['measurement_revision'],semantic_sources={p:index[prefix+p] for p in SEMANTIC_FILES},
        review_source_snapshot=C.relative(archive),seconds=time.monotonic()-began)
    C.save(out/'data/latest_width_audit.json',result)
    failure=out/'data/width_audit_failure.json'
    if failure.exists():failure.unlink()
    print('LATEST_WIDTH_AUDIT',report['id'],len(states),flush=True)
    return result
