"""Whole-set initialization: register, fit a wrap and cut all work cones."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
import time
from co_common import *
from work_access import RegisteredWorkVolumes,unpack_solid

SCHEMA='whole_registered_work_cones_v1'


def register(name,group):
    states={};reference=None;rows=[];inputs=[];reference_world=None
    for pose in group['poses']:
        task,T,mesh=state(name,pose)
        if reference is None:reference=mesh;reference_world=T.copy()
        np.testing.assert_array_equal(mesh.faces,reference.faces)
        np.testing.assert_allclose(mesh.vertices,reference.vertices,atol=1e-10,rtol=0)
        alignment=reference_world@np.linalg.inv(T)
        aligned=transform_points(task.domain.mesh.vertices,alignment)
        target=transform_points(reference.vertices,reference_world)
        np.testing.assert_allclose(aligned,target,atol=1e-10,rtol=0)
        states[pose]=(task,T,mesh);inputs+=task.inputs
        rows.append(dict(pose=pose,T_fixture_to_world=T.tolist(),
            T_world_to_fixture=np.linalg.inv(T).tolist(),
            T_native_world_to_reference_world=alignment.tolist(),
            T_reference_world_to_native_world=(T@np.linalg.inv(reference_world)).tolist(),
            object_to_fixture=np.eye(4).tolist(),
            alignment_max_error_m=float(np.max(np.abs(aligned-target))),work_faces=task.domain.work_ids.tolist()))
    return states,reference,dict(reference_pose=group['poses'][0],
        T_fixture_to_reference_world=reference_world.tolist(),states=rows),inputs


def build_seed(mesh,states,thickness=.005):
    access=RegisteredWorkVolumes(mesh,states)
    work_faces=access.ids;allowed=np.setdiff1d(np.arange(len(mesh.faces)),work_faces)
    offsets=wrap_offsets(mesh,thickness)
    cells=[S.solid(G.hull_mesh(G.head_cell(mesh,triangle,i,offsets)))
           for i,triangle in enumerate(mesh.triangles)]
    body=S.solid(mesh)
    seed=union([cells[i] for i in allowed])-body-union([cells[i] for i in work_faces])
    # Cap the Boolean cones beyond EVERY point of the complete fitted seed.
    bounds=np.array([mesh.vertices.min(0)-thickness,mesh.vertices.max(0)+thickness])
    corners=np.array([[x,y,z] for x in bounds[:,0] for y in bounds[:,1] for z in bounds[:,2]])
    forbidden,lengths=access.solid_for_points(corners)
    remaining=seed-forbidden
    work_overlap=material_volume(remaining^forbidden);object_overlap=material_volume(remaining^body)
    if max(work_overlap,object_overlap)>1e-10:
        raise RuntimeError('Initialization leaves material in a body or full work exclusion')
    wrapped=unpack_solid(remaining)
    if len(wrapped.faces):
        if not wrapped.is_watertight or not wrapped.is_winding_consistent:
            raise RuntimeError('Work-carved wrap boundary unresolved')
        triangles,sources=contact_boundary(mesh,wrapped,allowed)
    else:triangles=np.empty((0,3,3));sources=np.empty(0,int)
    return wrapped,triangles,sources,allowed,access,dict(
        uncut_wrap_material_volume_cm3=material_volume(seed)*1e6,
        material_volume_cm3=material_volume(remaining)*1e6,
        work_cut_material_volume_cm3=(material_volume(seed)-material_volume(remaining))*1e6,
        continuous_work_union_overlap_m3=work_overlap,object_overlap_m3=object_overlap,
        continuous_geometry_tolerance_m3=1e-10,finite_cone_lengths_m=lengths,
        finite_caps_beyond_entire_uncut_seed=True,complete_work_union_cut=True,
        component_count=len(remaining.decompose()))


def force_diagnostics(states,triangles,sources,out):
    rows=[]
    for pose,(task,T,mesh) in states.items():
        full=supply(task,T,triangles,sources);error=None
        try:
            mask,info=J.classify(full,task.targets);status='pass' if mask.all() else 'fail'
        except (RuntimeError,np.linalg.LinAlgError,AssertionError) as exception:
            mask=np.zeros(len(task.targets),bool);info={}
            error=f'{type(exception).__name__}: {exception}';status='unresolved'
        row=dict(pose=pose,status=status,force_passed=bool(mask.all()) if error is None else None,
            covered=int(mask.sum()),load_count=len(mask),error=error,classifier=info,
            working_surface_clear=True,full_work_cone_clear=True,
            scope='current registered work-carved contact model; not an infeasibility proof for other layouts')
        save(out/'data'/pose/'force.json',row)
        np.savez_compressed(out/'data'/pose/'coverage.npz',mask=mask,supply_7d=full)
        rows.append(row);print('INITIAL CONTACTS',pose,status,int(mask.sum()),flush=True)
    return rows


def run(name,group,out,thickness=.005):
    began=time.monotonic();out=Path(out);(out/'data').mkdir(parents=True,exist_ok=True)
    states,mesh,registration,inputs=register(name,group)
    wrapped,triangles,sources,allowed,access,geometry=build_seed(mesh,states,thickness)
    D.export_exact_obj(mesh,out/'registered_object.obj');D.export_exact_obj(wrapped,out/'wrapped_support.obj')
    np.savez_compressed(out/'data/contacts.npz',triangles_mesh_m=triangles,
        source_faces=sources,allowed_faces=allowed,excluded_work_faces=access.ids)
    rows=force_diagnostics(states,triangles,sources,out)
    force_status='pass' if all(row['status']=='pass' for row in rows) else (
        'unresolved' if any(row['status']=='unresolved' for row in rows) else 'fail')
    area=float(trimesh.triangles.area(triangles).sum()) if len(triangles) else 0.
    inputs+=[ROOT/'objects'/name/'pose_sets.json']
    report=dict(complete=True,passed=True,status='initialized',stage='step3.1',schema=SCHEMA,
        object=name,pose_set=group['id'],poses=group['poses'],**registration,
        coordinate_frame='reference object mesh in meters; world registration anchored at the first saved pose',
        registered_objects_coincide=True,object_native_poses_fixed=True,support_can_reorient_between_states=True,
        initialization_kind='whole-set registered wrap minus every complete work cone',
        work_access_definition=access.definition(),working_surfaces_clear=True,
        maximum_wrap_vertex_displacement_m=thickness,common_nonworking_face_count=len(allowed),
        excluded_work_face_count=len(access.ids),contact_triangle_count=len(triangles),actual_contact_area_m2=area,
        original_loads_reused=True,load_subsampling=False,state_results=rows,
        initial_force_status=force_status,force_gate_passed=force_status=='pass',
        force_diagnostics_are_acceptance_gates=False,step4_ready=True,full_fixture_accepted=False,
        exit_checked=False,ground_constructed=False,downstream_requires_rebuild=True,
        force_failure_scope='Fixed registration may fail; Direction/Juxtapose search is still allowed',
        **geometry,seconds=time.monotonic()-began,
        provenance=provenance(inputs,code_sources()+[HERE/'helper_func/work_access.py']),
        artifacts={'../registered_object.obj':I.sha256(out/'registered_object.obj'),
            '../wrapped_support.obj':I.sha256(out/'wrapped_support.obj'),'contacts.npz':I.sha256(out/'data/contacts.npz'),
            **{f'{pose}/{filename}':I.sha256(out/'data'/pose/filename)
               for pose in group['poses'] for filename in ['force.json','coverage.npz']}})
    save(out/'data/report.json',report)
    return states,mesh,report
