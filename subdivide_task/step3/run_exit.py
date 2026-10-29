"""Step3.2: independent task exits, certified on the final carved support."""
import argparse,json,shutil,time
from pathlib import Path
import numpy as np
import trimesh
import shapely
from subdivide_task.step1.common import ROOT,DEFAULT_OUTPUT,load_shape,sha256,write_json
from . import independent_geometry as G
from .exit_geometry import whole_object_sweep,grounded_material,condition_exit,refine_face_prisms
from .exit_search import independent_search,AssemblyRejected
from .run_independent import make_domain
from .independent_physics import demands,certificate

POLICY=Path(__file__).with_name('exit_policy.json')


def directions(axis,policy):
    rows=[]
    def add(d):
        d=np.asarray(d,float);d/=np.linalg.norm(d)
        if d[2]>=policy['minimum_upward_component'] and not any(np.linalg.norm(d-r)<1e-10 for r in rows):rows.append(d)
    seed=-np.asarray(axis,float);seed/=np.linalg.norm(seed);add(seed)
    # Small adjustments about this task's seed precede broader fallbacks.
    helper=np.eye(3)[np.argmin(np.abs(seed))];u=np.cross(seed,helper);u/=np.linalg.norm(u);v=np.cross(seed,u)
    for theta in policy['local_adjustment_angles_deg']:
        for phi in np.arange(0,360,policy['local_adjustment_azimuth_step_deg']):
            t,p=np.deg2rad([theta,phi]);add(np.cos(t)*seed+np.sin(t)*(np.cos(p)*u+np.sin(p)*v))
    for theta in policy['preferred_yz_tilts_from_up_deg']:
        t=np.deg2rad(theta);add([0.,np.sin(t),np.cos(t)])
    for theta in policy['hemisphere_polar_deg']:
        for phi in np.arange(0,360,policy['hemisphere_azimuth_step_deg']):
            t,p=np.deg2rad([theta,phi]);add([np.sin(t)*np.cos(p),np.sin(t)*np.sin(p),np.cos(t)])
    return rows


def load_source(output_root,object_name):
    folder=Path(output_root)/object_name/'step3/step3.1';report=json.loads((folder/'report.json').read_text())
    validation=json.loads((folder/'validation.json').read_text())
    assert validation['status']=='passed' and validation['report_sha256']==sha256(folder/'report.json')
    for filename,h in report['input_sha256'].items():assert sha256(filename)==h,filename
    for filename,h in report['artifact_sha256'].items():assert sha256(folder/filename)==h,filename
    with np.load(folder/'initialization.npz')as z:arrays={k:z[k]for k in z.files}
    source=load_shape(ROOT/'objects'/object_name/'mesh.stl')
    support=trimesh.Trimesh(arrays['support_vertices_m'],arrays['support_faces'],process=False)
    return folder,report,arrays,source,support


def prepare_tasks(source,support,report):
    solid=G.solid(support);tasks=[];policy=report['policy']
    for record in report['tasks']:
        T=np.asarray(record['world_from_object']);body=source.copy();body.apply_transform(T)
        low,high=record['cell_x_interval_m']
        box=G.box([low,support.bounds[0,1]-.001,0.],[high,support.bounds[1,1]+.001,support.bounds[1,2]+.001])
        local=G.mesh(grounded_material(solid^box))
        domain=make_domain(source,T,np.asarray(record['working_face_ids']),record['id'],record['workspace'],report['physical_properties'],policy)
        endpoints,samples=demands(domain,policy);work,_=G.work_sector(record['workspace'],policy)
        tasks.append(dict(record=record,body=body,local=local,domain=domain,endpoints=endpoints,samples=samples,work=work))
    return tasks


def exit_length(task,support,direction,policy):
    margin=policy['final_vertical_separation_m']
    return max(G.exit_length(task['body'],task['local'],direction,margin),
        (support.bounds[1,2]+margin-task['body'].bounds[0,2])/direction[2])


def run(output_root=DEFAULT_OUTPUT,object_name='B'):
    start=time.monotonic();output_root=Path(output_root).resolve()
    source_folder,previous,saved,source,support=load_source(output_root,object_name)
    policy=json.loads(POLICY.read_text());physics=previous['policy'];tol=physics['boolean_volume_tolerance_m3']
    tasks=prepare_tasks(source,support,previous);by_id={t['record']['id']:t for t in tasks};task_ids=list(by_id)
    original=G.solid(support);bodies=G.union([G.solid(t['body'])for t in tasks]);works=G.union([t['work']for t in tasks])
    catalogs={id:directions(t['domain'].axis,policy)for id,t in by_id.items()}
    trace=[];workdir=ROOT/'subdivide_task/data'/object_name/'step3_exit';workdir.mkdir(parents=True,exist_ok=True)
    def event(row):
        row['elapsed_seconds']=time.monotonic()-start;trace.append(row);write_json(workdir/'search_trace.json',trace)
        print('EXIT SEARCH',row,flush=True)

    def try_local(id,index,direction):
        task=by_id[id];length=exit_length(task,support,direction,policy)
        print('EXIT LOCAL TRY',id,index,np.round(direction,4).tolist(),'length',round(length,5),flush=True)
        sweep,metadata=whole_object_sweep(task['body'],direction,length)
        before=G.solid(task['local']);cut=grounded_material(before-sweep)
        actual,_=condition_exit(cut,[G.solid(task['body']),task['work'],sweep],before,tol)
        proof,_=certificate(task['body'],actual,task['domain'],task['endpoints'],task['samples'],search=True)
        if not proof['passed']:raise RuntimeError('Paired loads fail after this task exit carve: '+str(proof))
        return dict(direction=direction,length=float(length),sweep=sweep,metadata=metadata,local_force=proof)

    def assemble_and_audit(selected):
        try:
            all_sweeps=G.union([selected[id]['sweep']for id in task_ids])
            native=grounded_material(original-all_sweeps)
            result,conditioning=condition_exit(native,[bodies,works,all_sweeps],original,tol)
            paths=[(id,by_id[id]['body'],selected[id]['direction'],selected[id]['length'])for id in task_ids]
            result,facewise=refine_face_prisms(G.solid(result),paths,original,[bodies,works],tol)
            conditioning['facewise']=facewise;current=G.solid(result)
            assert result.is_watertight and result.is_winding_consistent and result.volume>0
            assert abs(result.bounds[0,2])<1e-12
            assert sum(p.volume()*G.SCALE**3>1e-15 for p in current.decompose())==1
            ground=result.triangles[np.abs(result.triangles[:,:,2]).max(1)<1e-9,:,:2]
            footprint=shapely.union_all(shapely.polygons(ground))
        except (RuntimeError,ValueError,AssertionError,np.linalg.LinAlgError)as ex:
            raise AssemblyRejected(task_ids,'Final assembly geometry unresolved: '+str(ex))from ex
        proofs={};contact_data={};failed=[]
        # All tasks are rechecked, even those whose selected direction did not change.
        for id,task in by_id.items():
            try:
                proof,data=certificate(task['body'],result,task['domain'],task['endpoints'],task['samples'],search=True)
                if not proof['passed']:raise RuntimeError('paired force/moment demands fail')
                pressure=G.pressure_points(task['endpoints']['endpoint_need_wrench'],task['domain'].com)
                assert shapely.covers(footprint,shapely.points(pressure)).all(),'ground pressure outside actual footprint'
                assert G.volume(current^G.solid(task['body']))<=tol,'object overlap'
                assert G.volume(current^task['work'])<=tol,'working space overlap'
                end=task['body'].vertices+selected[id]['length']*selected[id]['direction']
                assert end[:,2].min()>result.bounds[1,2]+policy['final_vertical_separation_m']-1e-12,'exit endpoint not clear'
                proofs[id]=proof;contact_data[id]=data
                print('EXIT FINAL ASSEMBLY',id,'PASS',flush=True)
            except (RuntimeError,ValueError,AssertionError,np.linalg.LinAlgError)as ex:
                failed.append(dict(id=id,reason=str(ex)))
        if failed:raise AssemblyRejected([r['id']for r in failed],str(failed))
        return result,conditioning,proofs,contact_data

    selected,assembled,indices=independent_search(task_ids,catalogs,try_local,assemble_and_audit,event)
    result,conditioning,proofs,contact_data=assembled;final=G.solid(result)
    bed_before=trimesh.Trimesh(saved['bedding_vertices_m'],saved['bedding_faces'],process=False)
    bedding=G.mesh(final^G.solid(bed_before))
    folder=workdir/'publish'/object_name/'step3/step3.2'
    if folder.exists():shutil.rmtree(folder)
    folder.mkdir(parents=True)
    arrays={k:v for k,v in saved.items()if k in ('source_vertices_object','source_faces','plate_bounds_m')}
    arrays.update(support_vertices_m=result.vertices,support_faces=result.faces,bedding_vertices_m=bedding.vertices,bedding_faces=bedding.faces)
    records=[]
    for id,task in by_id.items():
        proposal=selected[id];direction=proposal['direction'];length=proposal['length'];metadata=proposal['metadata']
        seed=-task['domain'].axis
        adjustment=float(np.rad2deg(np.arccos(np.clip(seed@direction,-1,1))))
        record=dict(task['record'],force=proofs[id],exit_direction_world=direction.tolist(),exit_length_m=length,exit_sweep=metadata,
            exit_candidate_index=indices[id],exit_adjustment_from_seed_deg=adjustment,
            final_assembly_audit_passed=True)
        records.append(record)
        arrays[id+'_world_from_object']=np.asarray(record['world_from_object'])
        arrays[id+'_working_face_ids']=np.asarray(record['working_face_ids'])
        arrays[id+'_exit_direction_world']=direction;arrays[id+'_exit_length_m']=np.asarray(length)
        arrays[id+'_sweep_leading_face_count']=np.asarray(metadata['leading_face_count'])
        arrays.update({id+'_'+key:value for key,value in contact_data[id].items()})
    np.savez_compressed(folder/'initialization.npz',**arrays)
    (folder/'support.obj').write_text(trimesh.exchange.obj.export_obj(result,digits=17,include_normals=False))
    write_json(folder/'search_trace.json',trace)
    report={k:v for k,v in previous.items()if k not in ('artifact_sha256','visualization','conditioning','tasks')}
    values=list(proofs.values())
    report.update(schema_version=2,stage='step3.2',method='independent_upper_hemisphere_exits_with_final_assembly_load_audit',tasks=records,
        exit_policy=policy,exit_carving_performed=True,exit_direction_mode='independent_per_task',exit_length_mode='independent_per_task',
        feasible_initialization=True,optimization_performed=False,all_final_assembly_audits_passed=True,
        working_tool_retracted_before_extraction=True,conditioning=conditioning,material_volume_cm3=float(result.volume*1e6),
        selected_candidate_indices=indices,assembly_repair_count=sum(r['stage']=='assembly'and r['status']!='accepted'for r in trace),
        original_sample_count=sum(p['original_sample_count']for p in values),passed_original_sample_count=sum(p['passed_original_sample_count']for p in values),
        continuous_outer_target_count=sum(p['continuous_outer_target_count']for p in values),passed_continuous_outer_target_count=sum(p['passed_continuous_outer_target_count']for p in values),
        source_step31=str(source_folder.resolve()),seconds=time.monotonic()-start)
    report['input_sha256']=dict(previous['input_sha256'])
    for p in [source_folder/'report.json',source_folder/'validation.json',source_folder/'initialization.npz',source_folder/'support.obj',POLICY,
              *[Path(__file__).with_name(n)for n in ('run_exit.py','exit_geometry.py','exit_search.py','render_exit.py','validate_exit.py')]]:
        report['input_sha256'][str(p.resolve())]=sha256(p)
    from .render_exit import render
    render(source,result,bedding,records,saved['plate_bounds_m'],folder,report)
    report['artifact_sha256']={p.name:sha256(p)for p in folder.iterdir()if p.is_file()and p.name not in ('report.json','validation.json')}
    write_json(folder/'report.json',report)
    from .validate_exit import validate
    validate(workdir/'publish',object_name)
    target=output_root/object_name/'step3/step3.2'
    if target.exists():shutil.rmtree(target)
    shutil.copytree(folder,target)
    for p in folder.iterdir():
        if p.is_file():assert sha256(target/p.name)==sha256(p)
    root=target.parent;overall=json.loads((root/'report.json').read_text())
    for key in ('common_exit_direction_world','common_exit_length_m'):overall.pop(key,None)
    overall.update(step32_run=True,step32_report='step3.2/report.json',exit_direction_mode='independent_per_task',
        step32_feasible_initialization=True,step32_optimization_performed=False)
    write_json(root/'report.json',overall)
    write_json(root/'validation.json',dict(status='passed',step31_validation='step3.1/validation.json',step32_validation='step3.2/validation.json',
        report_sha256=sha256(root/'report.json'),step32_run=True))
    print('STEP3.2 PASS',len(tasks),'independent task exits; final support passes all loads',flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--object',default='B');a=p.parse_args();run(a.output_root,a.object)
