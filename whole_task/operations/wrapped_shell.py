"""Keep only the fitted material layer surrounding an operation's object."""
import hashlib
import json

import numpy as np
import trimesh

from whole_task.step1.common import ROOT, sha256, write_json
from whole_task.step3 import independent_geometry as G
from whole_task.step3.independent_physics import certificate
from .geometry import moved_mesh


def wrapped_state(model, state, tag, index, owner=0):
    folder = ROOT / 'whole_task/data/B/operation_videos' / tag
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{index:03}.npz'
    record = path.with_suffix('.json')
    seed = state['support']
    digest = hashlib.sha256(seed.vertices.tobytes() + seed.faces.tobytes()).hexdigest()
    binding = dict(source_geometry_sha256=digest,
        outer_source_sha256=sha256(model.inputs[-1]),
        input_step32_report_sha256=sha256(model.folder / 'report.json'),
        method='connected_fitted_shell_with_full_contact_certificate_v2')
    if path.exists() and record.exists():
        meta = json.loads(record.read_text())
        if all(meta.get(key) == value for key, value in binding.items()) and sha256(path) == meta['mesh_sha256']:
            with np.load(path) as saved:
                shell = trimesh.Trimesh(saved['vertices'], saved['faces'], process=False)
            return dict(state, support=shell, volume_cm3=meta['volume_cm3'],
                support_representation='wrapped_shell', audit=meta['audit'],
                source_backed_volume_cm3=state['volume_cm3'])
    matrix = state['layout'].material[owner] @ state['layout'].poses[owner]
    outer = G.solid(moved_mesh(model.outer, matrix))
    source = G.solid(seed)
    layer = source ^ outer
    component = max(layer.decompose(), key=lambda part: float(part.volume()))
    shell = G.mesh(component ^ layer)
    actual = G.solid(shell)
    body = G.solid(moved_mesh(model.source, matrix))
    check = dict(outside_fitted_outer_m3=G.volume(actual - outer),
        outside_source_support_m3=G.volume(actual - source), body_overlap_m3=G.volume(actual ^ body))
    assert shell.is_watertight and shell.is_winding_consistent and shell.volume > 0.
    assert max(check.values()) <= model.tolerance
    data = model.world_data(model.records[owner], state['layout'].poses[owner])
    force, _ = certificate(data['body_world'],
        moved_mesh(shell, np.linalg.inv(state['layout'].material[owner])),
        data['domain'], data['endpoints'], data['samples'], search=True)
    assert force['passed']
    audit = dict(passed=True, geometry=check, certified_work_and_exit_clearance_preserved_by_subset=True,
        force=force, full_loads=True, single_connected_shell=True,
        working_face_ids_unchanged=True, support_representation='wrapped_shell',
        nominal_outward_thickness_m=model.policy['mold_outward_displacement_m'])
    np.savez_compressed(path, **state['layout'].arrays(), vertices=shell.vertices, faces=shell.faces)
    write_json(record, dict(**binding, mesh_sha256=sha256(path), volume_cm3=float(shell.volume * 1e6),
        audit=audit, support_representation='wrapped_shell', no_base_plate=True,
        no_vertical_backing=True, freestanding_floor_certificate_claimed=False))
    print(tag, index, 'SHELL PASS', round(shell.volume * 1e6, 3), flush=True)
    return dict(state, support=shell, volume_cm3=float(shell.volume * 1e6),
        support_representation='wrapped_shell', audit=audit, source_backed_volume_cm3=state['volume_cm3'])
