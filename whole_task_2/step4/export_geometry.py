"""Condition an exported core only by removing actual illegal material."""
import numpy as np
from whole_task_2.step3.legacy import kernel as K
from whole_search.common import transform_solid, unpack_solid
from connected_core import largest_material_component
from fixture_state import installed_state


def exclusions(model,layout,mesh):
    work=[];bodies=[];sweeps=[]
    for k in layout.active:
        Q=layout.placements[k];d=Q[:3,:3].T@layout.directions[k]
        local=(mesh.vertices-Q[:3,3])@Q[:3,:3]
        length=max(model.length,float((local@d).max()-(model.mesh.vertices@d).min())+.01)
        work.append(transform_solid(model.work_rays[k].solid(),Q))
        bodies.append(transform_solid(model.body,Q))
        sweeps.append(transform_solid(model.clearance.sweep(length*d,padded=False),Q))
    return work,bodies,sweeps


def condition_core(model,layout,mesh,path):
    model.check_station(layout)
    original=K.S.solid(mesh)
    work,bodies,sweeps=exclusions(model,layout,mesh)
    forbidden=work+bodies+sweeps;initial=None
    for cycle in range(5):
        K.D.export_exact_obj(mesh,path)
        actual=K.trimesh.load(path,force='mesh',process=False)
        solid=K.S.solid(actual)
        overlaps=[K.material_volume(solid^x)for x in forbidden]
        if initial is None:initial=overlaps
        if max(overlaps)<=1e-10:
            from prune_support import material_components
            if len(material_components(solid))!=1:raise RuntimeError('Export is not one material component')
            if not actual.is_watertight or not actual.is_winding_consistent:raise RuntimeError('Invalid exported core boundary')
            added=K.material_volume(solid-original)
            if added>1e-10:raise RuntimeError('Export conditioning added material')
            return actual,dict(conditioning_cycles=cycle,initial_maximum_overlap_m3=max(initial),
                final_maximum_overlap_m3=max(overlaps),removed_material_cm3=K.material_volume(original-solid)*1e6,
                added_material_m3=added,original_volume_tolerance_m3=1e-10,
                largest_component_reselected_after_cut=True,actual_contacts_require_recheck=True)
        # Keep the full source solids; only illegal material is removed.
        # Avoid replacing exclusions by approximate rays or a rounded mesh.
        remaining=solid-K.union([x for x,v in zip(forbidden,overlaps)if v>1e-10])
        for host in set(int(layout.hosts[k])for k in layout.active):
            T=installed_state(model.native,layout,host)
            remaining=remaining.trim_by_plane(tuple(T[2,:3]),float(-T[2,3]/K.S.SCALE))
        remaining,_=largest_material_component(remaining)
        mesh=unpack_solid(remaining)
    raise RuntimeError('Serialized core exclusions remain unresolved after conservative recuts')
