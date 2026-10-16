"""Independent final-solid replay through the unchanged original validator."""
from pathlib import Path
import json
import time
from types import SimpleNamespace
import numpy as np
from feasible_geometry import C,validate,ForceCertificate,Layout
from deterministic_base import solid,volume,enclosure,union,moved,unpack
from prune_support import material_components
from wrap_geometry import WrapInputs,original_demands,Validator
from wrap_dimensions import physical_component_count


def audit(payload):
    began=time.monotonic();out=Path(payload['output'])
    record=json.loads((out/'data/report.json').read_text())
    inputs=WrapInputs(payload['object'],payload['selection_summary'])
    data=inputs.load(payload['case'])
    original_demands(inputs,data,None)
    for name,digest in record['artifacts'].items():
        if C.digest(out/name)!=digest:raise ValueError('Final artifact changed: '+name)
    with np.load(data['selected']/'layout.npz') as z:layout=Layout.from_arrays(z)
    witness=C.trimesh.load(data['selected']/'oracle_connected_witness.obj',force='mesh',process=False)
    final=C.trimesh.load(out/'support.obj',force='mesh',process=False)
    initial=C.trimesh.load(out/'initial_support.obj',force='mesh',process=False)
    checker=ForceCertificate(data['record']['poses'],layout,data['body'],data['rows'],payload['object'])
    refinement=record.get('local_surface_reinforcement')
    if refinement and refinement['changed']:
        pruned=C.trimesh.load(out/'pruned_support.obj',force='mesh',process=False)
        if volume(solid(pruned)-solid(final))>1e-10:
            raise ValueError('Surface finish removed original pruned material')
    proof,checks=validate(final,witness,data['body'],layout.placements[list(layout.active)],
        data['rows'],checker,stock=None if refinement and refinement['changed'] else initial,
        bounds=C.occupied_bounds(data['body'],initial,data['rows']))
    states=json.loads((out/'states.json').read_text())['states']
    previous=None;previous_bounds=None;total_masks=0
    for state in states:
        path=C.ROOT/state['mesh'];masks=C.ROOT/state['force_masks']
        if C.digest(path)!=state['mesh_sha256'] or C.digest(masks)!=state['force_masks_sha256']:
            raise ValueError('Checkpoint binding changed')
        mesh=C.trimesh.load(path,force='mesh',process=False);value=solid(mesh)
        if not mesh.is_watertight or physical_component_count(value)!=1:
            raise ValueError('Open/disconnected checkpoint')
        if volume(value)<abs(float(initial.volume))*record['width_policy']['retained_volume_fraction']-1e-10:
            raise ValueError('Checkpoint exceeded conservative material budget')
        if previous is not None:
            if state['operation']=='local_surface_expansion':
                if not refinement or volume(previous-value)>1e-10:
                    raise ValueError('Surface reinforcement removed prior material')
            elif volume(value-previous)>1e-10:
                raise ValueError('Deletion checkpoint added material')
        bounds=C.occupied_bounds(data['body'],mesh,data['rows'])
        if previous_bounds is not None and (np.max(previous_bounds[0]-bounds[0])>1e-8
                or np.max(bounds[1]-previous_bounds[1])>1e-8):
            raise ValueError('Checkpoint BBX expanded')
        if not all(r['passed'] for r in enclosure(mesh,data['rows'])):
            raise ValueError('Checkpoint lost demand hull coverage')
        with np.load(masks) as z:
            if len(z.files)!=payload['case']['pose_count'] or any(z[k].shape!=(32768,) or not z[k].all() for k in z.files):
                raise ValueError('Incomplete checkpoint demand masks')
            total_masks+=sum(z[k].size for k in z.files)
        previous=value;previous_bounds=bounds
    last=C.trimesh.load(C.ROOT/states[-1]['mesh'],force='mesh',process=False)
    if not np.array_equal(last.vertices,final.vertices) or not np.array_equal(last.faces,final.faces):
        raise ValueError('Final model differs from last actual checkpoint')
    inputs.verify()
    if record.get('pruning_source'):
        source=C.ROOT/record['pruning_source']
        if C.digest(source/'data/report.json')!=record['pruning_source_report_sha256']:
            raise ValueError('Original pruning report changed')
        prefix=json.loads((source/'states.json').read_text())['states']
        if prefix!=states[:len(prefix)]:raise ValueError('Original pruning prefix changed')
    result=dict(complete=True,passed=True,id=record['id'],independent_original_validator=True,
        final_full_demands=sum(len(v) for v in proof['masks'].values()),checks=checks,
        checkpoint_count=len(states),checkpoint_masks_checked=total_masks,
        checkpoint_subset_and_bbx_rechecked=True,final_equals_last_checkpoint=True,
        all_checkpoints_connected=True,conservative_material_budget_rechecked=True,
        hard_minimum_width_enforced=False,width_or_strength_certificate=False,
        additive_surface_finish_superset_independently_checked=bool(refinement and refinement['changed']),
        original_pruning_preserved=bool(record.get('pruning_source')),
        seconds=time.monotonic()-began,executed_frozen_sources=getattr(__import__('sys'),'wrap_executed_sources',{}))
    C.save(out/'data/independent_audit.json',result)
    print('AUDIT',record['id'],result['final_full_demands'],flush=True)
    return result
