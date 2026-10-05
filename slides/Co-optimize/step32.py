"""Full non-working-surface wrapping and unchanged original-load force gate."""
import time
from co_common import *

def run(name,group,states,mesh,out,thickness=.005):
    began=time.monotonic();out.mkdir(parents=True,exist_ok=True);(out/'data').mkdir(exist_ok=True)
    work_faces=np.unique(np.concatenate([task.domain.work_ids for task,T,m in states.values()]))
    allowed=np.setdiff1d(np.arange(len(mesh.faces)),work_faces)
    offsets=wrap_offsets(mesh,thickness)
    cells=[S.solid(G.hull_mesh(G.head_cell(mesh,tri,i,offsets))) for i,tri in enumerate(mesh.triangles)]
    # Exclude all states' work surfaces; shared edges/vertices stay permitted.
    shell=union([cells[i] for i in allowed])-S.solid(mesh)-union([cells[i] for i in work_faces])
    material_volume(shell)
    wrapped=S.unpack(shell)
    if shell.is_empty():
        triangles=np.empty((0,3,3));sources=np.empty(0,int)
    else:
        if not wrapped.is_watertight or not wrapped.is_winding_consistent:raise RuntimeError('Wrap boundary unresolved')
        triangles,sources=contact_boundary(mesh,wrapped,allowed)
    D.export_exact_obj(wrapped,out/'wrapped_support.obj')
    np.savez_compressed(out/'data/contacts.npz',triangles_mesh_m=triangles,source_faces=sources,allowed_faces=allowed,excluded_work_faces=work_faces)
    results=[];inputs=[];force_passed=True;unresolved=False
    contact_area=float(np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1).sum()/2)
    for pose,(task,T,m) in states.items():
        inputs+=task.inputs;full=supply(task,T,triangles,sources);proof=None;error=None
        try:
            mask,info=J.classify(full,task.targets)
            if not mask.all():
                failed=int(np.flatnonzero(~mask)[0])
                # An upper-bound fail is only conclusive for ALL allowed face
                # pressure, not merely for a deficient Boolean shell.
                ideal=supply(task,T,mesh.triangles[allowed],allowed)
                ideal_target=U.target(task.targets[failed])
                proof=C.W.exact_separator(ideal,ideal_target)
                if proof is None:
                    error='No exact full-surface failure proof; shell or numerical result unresolved';unresolved=True
                else:
                    proof.update(load_index=failed,target_7d=ideal_target.tolist(),scope='All pressure on all common non-working source triangles, plus unchanged original floor/no-uplift model')
        except (RuntimeError,np.linalg.LinAlgError,AssertionError) as e:
            mask=np.zeros(len(task.targets),bool);info={};error=f'{type(e).__name__}: {e}';unresolved=True
        passed=bool(mask.all() and error is None);force_passed &= passed
        world=wrapped.copy();world.apply_transform(T)
        work=WORK.check(world,task) if len(world.faces) else dict(passed=True)
        minimum=float(world.vertices[:,2].min()) if len(world.vertices) else None
        row=dict(pose=pose,force_passed=passed,covered=int(mask.sum()),load_count=len(mask),status='pass' if passed else 'fail' if proof is not None else 'unresolved',failure_proof=proof,error=error,classifier=info,working_surface_clear=work['passed'],working_surface_check=work,min_world_z_m=minimum,floor_clear=minimum is not None and minimum>=-1e-9)
        save(out/'data'/pose/'force.json',row)
        np.savez_compressed(out/'data'/pose/'coverage.npz',mask=mask,supply_7d=full)
        results.append(row)
        print('FULL SURFACE',group['id'],pose,row['status'],row['covered'],flush=True)
    mechanical_fail=any(r['status']=='fail' for r in results)
    work_clear=all(r['working_surface_clear'] for r in results)
    status='fail' if mechanical_fail else 'pass' if force_passed and work_clear else 'unresolved'
    report=dict(complete=True,passed=status=='pass',status=status,object=name,pose_set=group['id'],poses=group['poses'],force_gate_passed=force_passed,working_surfaces_clear=work_clear,step4_ready=status=='pass',full_fixture_accepted=False,exit_checked=False,original_loads_reused=True,load_subsampling=False,geometry_kind='Dense outer skin on every common non-working source surface; no contact candidate menu or greedy selection',work_policy='A shared entity must leave every state work-face interior free; union of work IDs excluded',maximum_wrap_vertex_displacement_m=thickness,minimum_thickness_or_strength_claim=False,material_volume_cm3=material_volume(shell)*1e6,component_count=len(shell.decompose()),object_face_count=len(mesh.faces),common_nonworking_face_count=len(allowed),excluded_work_face_count=len(work_faces),contact_triangle_count=len(triangles),actual_contact_area_m2=contact_area,ideal_nonworking_area_m2=float(mesh.area_faces[allowed].sum()),passive_support_constraint=U.description(),floor_model=FLOOR.description(),state_results=results,ground_policy='Native ground reaction model preserved; wrap may cross state floors and has no full fixture acceptance. Step4 must carve forbidden floor material and exits while rechecking remaining contact loads.',failure_scope='Conclusive fail only for fixed pose registration, common non-working surfaces, original unilateral reaction/no-uplift model and stored loads; not a claim about other layouts or physical models',seconds=time.monotonic()-began,provenance=provenance(inputs+[out.parent/'step3.1/data/report.json'],code_sources()),artifacts={**{'../wrapped_support.obj':I.sha256(out/'wrapped_support.obj'),'contacts.npz':I.sha256(out/'data/contacts.npz')},**{f'{pose}/{filename}':I.sha256(out/'data'/pose/filename) for pose in group['poses'] for filename in ('force.json','coverage.npz')}})
    save(out/'data/report.json',report)
    return wrapped,triangles,sources,report
