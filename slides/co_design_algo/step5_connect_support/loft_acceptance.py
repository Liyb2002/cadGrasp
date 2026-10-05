"""Accept the real union of selected lofts, including numerical mesh cleanup."""
import numpy as np

from step5_connect_support import build_coupled_saddle as S, material_graph as M, material_network as N
from step5_connect_support.loft_growth import selection_solid


def remove_numerical_debris(full, tolerance_m3=8e-14):
    parts = sorted(full.decompose(), key=lambda s:-s.volume())
    volumes = [float(s.volume()*S.SCALE**3) for s in parts]
    removed = float(sum(volumes[1:]))
    if not parts or removed > tolerance_m3:
        raise RuntimeError(f'Loft union has genuinely disconnected components: {volumes}')
    return parts[0], dict(discarded_component_volumes_m3=volumes[1:],
        discarded_volume_m3=removed, tolerance_m3=tolerance_m3)


def inspect(case, context, graph, selected):
    full = selection_solid(graph, selected)
    full, debris = remove_numerical_debris(full)
    full, regularization = M.regularize_export(full, context['heads'])
    full, after = remove_numerical_debris(full)
    regularization.update(before_export_cleanup=debris, after_export_cleanup=after)
    mesh = S.unpack(full)
    solid = dict(component_count=len(full.decompose()), watertight=bool(mesh.is_watertight),
        consistently_wound=bool(mesh.is_winding_consistent), volume_m3=float(mesh.volume))
    solid['one_solid'] = solid['component_count'] == 1 and solid['watertight'] and solid['consistently_wound'] and mesh.volume > 0
    if not solid['one_solid']: raise RuntimeError(f'Loft union is not one closed solid: {solid}')
    floor_checks = []
    for k in (0,1):
        world = (mesh.vertices-context['offsets'][k])@context['bases'][k].T
        floor_checks.append(N.containment(world[np.abs(world[:,2])<1e-9,:2],case.demands[k]))
    if not all(c['passed'] for c in floor_checks): raise RuntimeError('Actual loft union lost floor containment')
    passed = True
    try:
        checks, certificate = S.verify(case.tasks,case.groups,case.heads,context['directions'],
            context['bases'],context['offsets'],mesh,context['sweeps'])
    except RuntimeError as error:
        if not hasattr(error,'certificate'): raise
        passed = False; checks,certificate = error.checks,error.certificate
    return dict(full=full,mesh=mesh,solid=solid,checks=checks,certificate=certificate,
        floor_checks=floor_checks,passed=passed,selected=selected,regularization=regularization)
