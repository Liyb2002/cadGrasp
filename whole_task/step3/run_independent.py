"""Step3.1: a row of task objects and fitted negative molds on one flat plate."""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
import numpy as np
import trimesh
from robots_config.constraints import load_config
from whole_task.step1.common import ROOT, DEFAULT_OUTPUT, load_shape, sha256, write_json
from whole_task.step1.needle_geometry import geometric_mask
from whole_task.common.poses import spin_pose
from whole_task.common.wrenches import build_domain, resolve_policy, endpoint_demands
from . import independent_geometry as G
from .independent_physics import demands, certificate

POLICY=Path(__file__).with_name('independent_policy.json')
PHYSICS_FILES=['whole_task/common/wrenches.py','whole_task/common/force_cone.py','whole_task/common/surface_hull.py',
 'whole_task/common/transforms.py','whole_task/common/poses.py','whole_task/step3/independent_geometry.py',
 'whole_task/step3/independent_physics.py','whole_task/step3/run_independent.py','whole_task/step3/independent_policy.json']


def load_inputs(output_root,object_name,count):
    base=Path(output_root)/object_name
    from whole_task.step1.validate_needle import validate
    validate(output_root,object_name)
    plan=json.loads((base/'step1/step1.2/coverage_plan.json').read_text())
    if not 1<=count<=plan['working_area_count']:raise ValueError('Invalid task prefix')
    with np.load(base/'step1/step1.2/coverage_plan.npz') as z:arrays={k:z[k] for k in z.files}
    config=load_config(base/'step1/step1.1/config.json')
    physical=json.loads((ROOT/'objects'/object_name/'meta.json').read_text())
    policy=resolve_policy(json.loads(POLICY.read_text()),config);policy['task_count']=count
    source=load_shape(ROOT/'objects'/object_name/'mesh.stl')
    inputs={str(ROOT/p):sha256(ROOT/p) for p in PHYSICS_FILES}
    for p in [base/'step1/step1.2/coverage_plan.json',base/'step1/step1.2/coverage_plan.npz',
              base/'step1/step1.1/config.json',ROOT/'objects'/object_name/'meta.json',ROOT/'objects'/object_name/'mesh.stl']:
        inputs[str(p.resolve())]=sha256(p)
    from .legacy import loaded_sources
    for p in loaded_sources():inputs[str(p)]=sha256(p)
    return source,plan,arrays,config,physical,policy,inputs


def make_domain(source,T,ids,id,workspace,physical,policy):
    pose=dict(id=id,world_from_object=T.tolist(),working_face_ids=ids,filled_face_ids=[],
        geometry_audit=dict(geometrically_passed_face_ids=ids),status='mutable_initialization_seed')
    station=dict(world_gravity_m_s2=[0.,0.,-9.81],tool_to_surface_axis_world=workspace['nominal_axis_world'])
    return build_domain(source,pose,station,physical,policy)


def moved_workspace(workspace,shift):
    result=dict(workspace)
    result['reference_head_front_world_m']=(np.asarray(workspace['reference_head_front_world_m'])+shift).tolist()
    return result


def resting_pose(source,reference,workspace,angle,plate_top):
    T=spin_pose(reference,workspace['nominal_axis_world'],workspace['reference_head_front_world_m'],angle)
    body=source.copy();body.apply_transform(T)
    shift=np.r_[-body.bounds.mean(0)[:2],plate_top-body.bounds[0,2]]
    T[:3,3]+=shift
    return T,moved_workspace(workspace,shift),shift


def initialize_task(vertices,faces,outer_vertices,outer_faces,reference,ids,id,workspace,physical,policy,cache,signature):
    cache=Path(cache);meta=cache.with_suffix('.json')
    if cache.exists() and meta.exists():
        saved=json.loads(meta.read_text())
        if saved.get('signature')==signature and saved.get('arrays_sha256')==sha256(cache):
            print(id,'REUSE',flush=True);return saved['record']
    source=trimesh.Trimesh(vertices,faces,process=False)
    outer_source=trimesh.Trimesh(outer_vertices,outer_faces,process=False)
    trace=[];top=policy['plate_thickness_m'];margin=policy['plate_edge_margin_m']
    for angle in policy['initial_axial_angles_deg']:
        try:
            T,ws,shift=resting_pose(source,reference,workspace,angle,top)
            body=source.copy();body.apply_transform(T)
            head=(np.asarray(ws['reference_head_front_world_m'])-T[:3,3])@T[:3,:3]
            assert geometric_mask(source,head,T[:3,:3].T@np.asarray(ws['nominal_axis_world']),ws)[ids].all()
            domain=make_domain(source,T,ids,id,ws,physical,policy);endpoints,samples=demands(domain,policy)
            pressure=G.pressure_points(endpoints['endpoint_need_wrench'],domain.com)
            outer=outer_source.copy();outer.apply_transform(T)
            lo=np.r_[np.minimum(outer.bounds[0,:2],pressure.min(0))-margin,0.]
            hi=np.r_[np.maximum(outer.bounds[1,:2],pressure.max(0))+margin,top]
            plate=G.box(lo,hi);work,work_meta=G.work_sector(ws,policy);body_solid=G.solid(body)
            stock=G.casting_stock(outer,top)
            value=(stock+plate)-body_solid-work
            positives=[p for p in value.decompose() if p.volume()*G.SCALE**3>1e-15]
            if len(positives)!=1:
                trace.append(dict(angle_deg=angle,status='disconnected_cast',component_count=len(positives)));continue
            support,conditioning=G.condition(value,[body_solid,work],policy)
            proof,_=certificate(body,support,domain,endpoints,samples,search=True)
            if not proof['passed']:
                trace.append(dict(angle_deg=angle,status='load_certificate_not_passed'));continue
            np.savez_compressed(cache,vertices=support.vertices,faces=support.faces,world_from_object=T,
                working_face_ids=ids,plate_bounds=np.vstack([lo,hi]))
            record=dict(id=id,initial_axial_angle_deg=float(angle),final_pose_fixed=False,
                local_world_from_object=T.tolist(),placement_shift_m=shift.tolist(),workspace_local=ws,
                working_face_ids=ids.tolist(),plate_bounds_local_m=[lo.tolist(),hi.tolist()],
                object_bottom_on_local_plate_top=True,work_exclusion=work_meta,local_force=proof,
                conditioning=conditioning,search_trace=trace,cache=str(cache.resolve()))
            write_json(meta,dict(signature=signature,arrays_sha256=sha256(cache),record=record))
            print(id,'CAST PASS',f'angle={angle:g}',flush=True);return record
        except (RuntimeError,ValueError,AssertionError,np.linalg.LinAlgError) as ex:
            trace.append(dict(angle_deg=angle,status='numerically_unresolved',reason=str(ex)))
    write_json(meta,dict(signature=signature,status='failed',search_trace=trace))
    raise RuntimeError(id+' casting initialization exhausted; see '+str(meta))


def assemble(source,records,policy,source_workspace,physical):
    packed=[];pieces=[];risers=[];bodies=[];works=[];left=None;all_xy=[]
    top=policy['plate_thickness_m']
    common_height=max(r['workspace_local']['reference_head_front_world_m'][2] for r in records)
    common_y=source_workspace['reference_head_front_world_m'][1]
    for record in records:
        with np.load(record['cache']) as z:
            T=z['world_from_object'].copy();support=trimesh.Trimesh(z['vertices'],z['faces'],process=False)
        work,_=G.work_sector(record['workspace_local'],policy)
        span=np.vstack([support.vertices,G.mesh(work).vertices])
        low,high=span[:,0].min(),span[:,0].max()
        x=-high if left is None else left-policy['row_gap_m']-high
        root=np.asarray(record['workspace_local']['reference_head_front_world_m'])
        lift=max(0.,common_height-root[2])
        offset=np.array([x,common_y-root[1],lift]);left=low+x
        T[:3,3]+=offset;support.apply_translation(offset)
        body=source.copy();body.apply_transform(T)
        ws=moved_workspace(record['workspace_local'],offset);work,_=G.work_sector(ws,policy)
        pieces.append(G.solid(support));bodies.append(G.solid(body));works.append(work)
        all_xy.append(support.bounds[:,:2])
        riser_bounds=None
        if lift>1e-12:
            base=np.asarray(record['plate_bounds_local_m'])
            # Overlap the base through its full thickness. Merely touching
            # z=top can leave a floating seat after floating-point Booleans.
            riser_bounds=np.array([np.r_[base[0,:2]+offset[:2],0.],
                                   np.r_[base[1,:2]+offset[:2],top+lift]])
            risers.append(G.box(*riser_bounds))
        domain=make_domain(source,T,np.asarray(record['working_face_ids']),record['id'],ws,
            physical,policy)
        endpoints=endpoint_demands(domain)
        pressure=G.pressure_points(endpoints['endpoint_need_wrench'],domain.com)
        all_xy.extend([pressure.min(0)-policy['plate_edge_margin_m'],pressure.max(0)+policy['plate_edge_margin_m']])
        packed.append(dict(record,world_from_object=T.tolist(),row_offset_m=offset.tolist(),workspace=ws,
            riser_height_m=float(lift),riser_bounds_m=None if riser_bounds is None else riser_bounds.tolist(),
            object_bottom_on_riser_top=True,tool_head_height_m=float(common_height),
            cell_x_interval_m=[float(low+x),float(high+x)]))
    xy=np.vstack(all_xy);lo=np.r_[xy.min(0),0.];hi=np.r_[xy.max(0),policy['plate_thickness_m']]
    plate=G.box(lo,hi)
    bedding_stock=G.union([plate,*risers])
    value=G.union([bedding_stock,*pieces])
    # A common plate must never fill an individual task's previously cut opening.
    body_union=G.union(bodies);work_union=G.union(works)
    value=value-body_union-work_union
    serialized=G.solid(G.mesh(value))
    value=G.grounded_component(serialized)
    removed=max(0.,G.volume(serialized)-G.volume(value))*1e6
    support,conditioning=G.condition(value,[body_union,work_union],policy)
    conditioning['detached_material_removed_cm3']=removed
    positives=[p for p in G.solid(support).decompose() if p.volume()*G.SCALE**3>1e-15]
    if len(positives)!=1:raise RuntimeError('The flat plate and its molds must be one connected solid')
    bedding,_=G.condition(G.solid(support)^bedding_stock,[body_union,work_union],policy,subset=G.solid(support))
    return support,bedding,packed,np.vstack([lo,hi]),conditioning,common_height


def run(output_root=DEFAULT_OUTPUT,object_name='B',count=13,workers=4):
    start=time.monotonic();output_root=Path(output_root).resolve()
    source,plan,arrays,config,physical,policy,inputs=load_inputs(output_root,object_name,count)
    workdir=ROOT/'whole_task/data'/object_name/'step3_mold_plate';workdir.mkdir(parents=True,exist_ok=True)
    signature=hashlib.sha256(json.dumps(dict(inputs=inputs,policy=policy),sort_keys=True).encode()).hexdigest()
    outer_cache=workdir/'fitted_outer.npz';outer_meta=workdir/'fitted_outer.json'
    if outer_cache.exists() and outer_meta.exists() and json.loads(outer_meta.read_text()).get('signature')==signature:
        with np.load(outer_cache) as z:outer=trimesh.Trimesh(z['vertices'],z['faces'],process=False)
    else:
        outer=G.mesh(G.fitted_outer(source,policy['mold_outward_displacement_m']))
        np.savez_compressed(outer_cache,vertices=outer.vertices,faces=outer.faces)
        write_json(outer_meta,dict(signature=signature,sha256=sha256(outer_cache)))
    arguments=[]
    for i in range(count):
        id=str(arrays['working_area_ids'][i]);ids=np.flatnonzero(arrays['working_area_face_membership'][i])
        arguments.append((source.vertices,source.faces,outer.vertices,outer.faces,arrays['screening_world_from_object'][i],
            ids,id,plan['workspace'],physical,policy,workdir/f'{id}.npz',signature))
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        records=list(pool.map(_initialize,arguments))
    support,bedding,records,plate_bounds,conditioning,common_height=assemble(source,records,policy,plan['workspace'],physical)
    folder=workdir/'publish'/object_name/'step3/step3.1';folder.mkdir(parents=True,exist_ok=True)
    saved=dict(support_vertices_m=support.vertices,support_faces=support.faces,plate_bounds_m=plate_bounds,
        bedding_vertices_m=bedding.vertices,bedding_faces=bedding.faces,
        source_vertices_object=source.vertices,source_faces=source.faces)
    for record in records:
        id=record['id'];T=np.asarray(record['world_from_object']);ids=np.asarray(record['working_face_ids'])
        body=source.copy();body.apply_transform(T)
        domain=make_domain(source,T,ids,id,record['workspace'],physical,policy);endpoints,samples=demands(domain,policy)
        proof,data=certificate(body,support,domain,endpoints,samples)
        if not proof['passed']:raise RuntimeError(id+' fails on the actual common plate assembly')
        record['force']=proof
        saved[id+'_world_from_object']=T;saved[id+'_working_face_ids']=ids
        saved.update({id+'_'+key:value for key,value in data.items()})
        print(id,'COMMON PLATE FORCE PASS',flush=True)
    np.savez_compressed(folder/'initialization.npz',**saved)
    (folder/'support.obj').write_text(trimesh.exchange.obj.export_obj(support,digits=17,include_normals=False))
    report=dict(schema_version=3,stage='step3.1',status='passed',object=object_name,task_count=count,
        method='fitted_molds_with_risers_and_one_common_tool_height',tasks=records,policy=policy,
        source_workspace=plan['workspace'],physical_properties=physical,plate_bounds_m=plate_bounds.tolist(),
        plate_count=1,mold_seat_count=count,connected_material_component_count=1,objects_rest_on_height_adjusted_seats=True,
        common_tool_head_height_m=float(common_height),all_tool_heads_same_height=True,
        common_tool_head_y_m=float(plan['workspace']['reference_head_front_world_m'][1]),
        maximum_riser_height_m=max(r['riser_height_m'] for r in records),
        task_objects_pairwise_disjoint=True,pose_is_mutable_initial_seed=True,exit_carving_performed=False,
        task_selection='first_13_in_step1_greedy_order',
        selected_surface_union_fraction=float(source.area_faces[arrays['working_area_face_membership'][:count].any(0)].sum()/source.area),
        original_sample_count=sum(r['force']['original_sample_count'] for r in records),
        passed_original_sample_count=sum(r['force']['passed_original_sample_count'] for r in records),
        continuous_outer_target_count=sum(r['force']['continuous_outer_target_count'] for r in records),
        passed_continuous_outer_target_count=sum(r['force']['passed_continuous_outer_target_count'] for r in records),
        material_volume_cm3=float(support.volume*1e6),conditioning=conditioning,
        support_mesh='support.obj',arrays='initialization.npz',input_sha256=inputs,
        native_robot_motion_certified=False,fixture_stress_or_strength_certified=False)
    from .render_independent import render
    render(source,support,bedding,records,plate_bounds,folder,report)
    for name in ['render_independent.py','validate_independent.py']:
        p=Path(__file__).with_name(name);report['input_sha256'][str(p)]=sha256(p)
    report['artifact_sha256']={p.name:sha256(p) for p in folder.iterdir() if p.is_file() and p.name not in ('report.json','validation.json')}
    write_json(folder/'report.json',report)
    from .validate_independent import validate
    validate(workdir/'publish',object_name)
    target=output_root/object_name/'step3';target.mkdir(exist_ok=True)
    if (target/'step3.1').exists():shutil.rmtree(target/'step3.1')
    shutil.copytree(folder,target/'step3.1')
    if (target/'step3.2').exists():shutil.rmtree(target/'step3.2')
    overall=dict(status='passed',algorithm=report['method'],task_count=count,plate_count=1,
        common_tool_head_height_m=float(common_height),
        step31_report='step3.1/report.json',step32_run=False,step4_run=False,step2_used=False,
        seconds=time.monotonic()-start)
    write_json(target/'report.json',overall)
    write_json(target/'validation.json',dict(status='passed',step31_validation='step3.1/validation.json',
        report_sha256=sha256(target/'report.json'),step32_run=False))
    for path in folder.iterdir():
        if path.is_file():assert sha256(target/'step3.1'/path.name)==sha256(path)
    print('STEP3.1 COMMON PLATE PASS',json.dumps(overall),flush=True)
    return overall


def _initialize(arguments):return initialize_task(*arguments)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--object',default='B');p.add_argument('--count',type=int,default=13)
    p.add_argument('--output-root',type=Path,default=DEFAULT_OUTPUT);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();run(a.output_root,a.object,a.count,a.workers)
