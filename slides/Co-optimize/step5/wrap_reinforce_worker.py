"""Append surface reinforcement to an unchanged, completed pruning result."""
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from wrap_pipeline import read, save, sha, ROOT
from wrap_geometry import WrapInputs, original_demands, Validator, Layout, C
from wrap_surface_repair import repair
from exact_mesh import save_mesh


def worker(payload):
    began = time.monotonic()
    source, out = Path(payload['source']), Path(payload['output'])
    parent = read(source/'data/report.json')
    assert parent['complete'] and parent['passed']
    assert sha(source/'support.obj') == parent['support_sha256']
    inputs = WrapInputs(payload['object'], payload['selection_summary'])
    data = inputs.load(payload['case'])
    assert C.relative(data['selected']) == parent['selected_step4_source']
    original_demands(inputs, data, None)
    inputs.protected.update(parent['provenance']['inputs'])
    for path in [source/'data/report.json', *[source/name for name in parent['artifacts']]]:
        inputs.protected[C.relative(path)] = sha(path)
    out.mkdir(parents=True, exist_ok=True)
    for name in ['initial_support.obj','construction.json','deletions.json']:
        shutil.copy2(source/name, out/name)
    shutil.copytree(source/'construction', out/'construction', dirs_exist_ok=True)
    shutil.copy2(source/'support.obj', out/'pruned_support.obj')
    with np.load(data['selected']/'layout.npz') as z:
        layout = Layout.from_arrays(z)
    witness_path = data['selected']/'oracle_connected_witness.obj'
    inputs.protected[C.relative(witness_path)] = sha(witness_path)
    witness = C.trimesh.load(witness_path, force='mesh', process=False)
    validator = Validator(data, layout, witness, payload['object'])
    mesh = C.trimesh.load(source/'support.obj', force='mesh', process=False)
    mesh, value, proof, checks, refinement = repair(mesh, validator, out/'surface_reinforcement')
    shutil.copy2(out/'surface_reinforcement/support.obj', out/'support.obj')
    states_record = read(source/'states.json')
    states = states_record['states']
    if refinement['changed']:
        path = out/f'checkpoints/{len(states):04d}.obj'
        path.parent.mkdir(parents=True, exist_ok=True)
        save_mesh(mesh, path)
        masks = path.with_suffix('.npz')
        np.savez_compressed(masks, **{f'pose_{k}':v for k,v in proof['masks'].items()})
        states.append(dict(index=len(states),operation='local_surface_expansion',
            detail=refinement,mesh=C.relative(path),mesh_sha256=sha(path),
            force_masks=C.relative(masks),force_masks_sha256=sha(masks),**checks))
    save(out/'states.json', dict(states_record, states=states,
        original_pruning_prefix_unchanged=True, additive_finishing_stage=True))
    np.savez_compressed(out/'force_certificate.npz', **{f'pose_{k}':v for k,v in proof['masks'].items()})
    report = dict(parent, experiment=payload['experiment'],
        pipeline='connected_hull_wrap_then_progressive_deletion_then_local_surface_expansion',
        support_sha256=sha(out/'support.obj'),checks=checks,final_material_cm3=checks['material_cm3'],
        final_bbx_l=checks['bbx_l'],material_added_after_pruning_started=refinement['changed'],
        pruning_final_material_cm3=parent['final_material_cm3'],local_surface_reinforcement=refinement,
        pruning_source=C.relative(source),pruning_source_report_sha256=sha(source/'data/report.json'),
        pruning_checkpoint_count=len(states_record['states'])-int(refinement['changed']),
        seconds=parent['seconds']+time.monotonic()-began,
        reinforcement_seconds=time.monotonic()-began,provenance=dict(inputs=inputs.protected),
        executed_frozen_sources=getattr(sys,'wrap_executed_sources',{}),
        execution_launcher_sha256=payload.get('execution_launcher_sha256'),
        execution_source_snapshot=payload['source_snapshot'])
    names=['support.obj','initial_support.obj','pruned_support.obj','states.json','deletions.json',
           'force_certificate.npz','surface_reinforcement/report.json']
    if refinement['changed']:
        names.append('surface_reinforcement/added_material.obj')
    report['artifacts']={name:sha(out/name) for name in names}
    inputs.verify()
    save(out/'data/report.json', report)
    print('REINFORCED',parent['id'],refinement['added_material_cm3'],time.monotonic()-began,flush=True)
    return report
