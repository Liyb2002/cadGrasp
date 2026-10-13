"""Deterministic close-direction seed initialization and nonblocking diagnostics."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from co_common import *
from exit_clearance import ExitClearance
from scipy.spatial import ConvexHull
from pathlib import Path
import argparse,time

def initialize_direction(T_pose):
    canonical=T_pose[:3,:3].T@np.array([0.,0.,1.])
    return canonical,T_pose[:3,:3]@canonical

def selected_groups(name):
    selected=ROOT/'objects'/name/'selected_pose_sets.json'
    if selected.exists():
        from codes.precompute_objects.dataset import read_selected_pose_groups
        return read_selected_pose_groups(name)
    manifest=list(read_sets(name)['sets'])
    illegal=ROOT/'objects'/name/'illegal_pose_sets.json'
    if illegal.exists():manifest += [dict(g,id='illegal/'+g['id']) for g in json.loads(illegal.read_text())['sets']]
    return manifest

def _prepare_group_sweeps(arguments):
    name,group,record,length,folder=arguments
    states={p:state(name,p) for p in group['poses']}
    original=np.array([record['paths'][p]['direction_fixture'] for p in group['poses']])
    normals=np.array([initialize_direction(states[p][1])[0] for p in group['poses']])
    common=record['metadata']['common_direction_status']!='no_nonzero_common_direction'
    def attempt(directions):
        artifacts={}
        for pose,direction in zip(group['poses'],directions):
            _,_,mesh=states[pose];path=folder/group['id']/pose;path.mkdir(parents=True,exist_ok=True)
            for kind,distance in [('full',length),('display',.10)]:
                sweep=S.swept_solid(mesh,distance*direction);D.export_exact_obj(sweep,path/f'{kind}_sweep.obj')
                artifacts[f"{group['id']}/{pose}/{kind}_sweep.obj"]=I.sha256(path/f'{kind}_sweep.obj')
        return artifacts
    try:
        artifacts=attempt(original)
        print('PREPARED CLOSE DIRECTIONS',group['id'],record['metadata']['common_direction_status'],flush=True)
        return group['id'],artifacts,None,record
    except Exception as error:
        message=f'{type(error).__name__}: {error}'
    # Change the actual direction, rather than omitting a positive-volume prism.
    # A tangential face may otherwise produce a 1e-19 m^3 unrepresentable sliver.
    if 'collapsed numerically' in message:
        axes=np.array([[1.,np.sqrt(2),np.sqrt(3)],[-np.sqrt(3),1.,np.sqrt(2)],[np.sqrt(2),-np.sqrt(3),1.]])
        failures=[]
        for epsilon in [1e-6,1e-5,1e-4]:
            for axis in axes:
                bias=np.repeat(axis[None,:],len(original),axis=0)
                if common:
                    active=normals[np.abs(normals@original[0])<1e-9]
                    if len(active):
                        _,singular,right=np.linalg.svd(active,full_matrices=True);rank=np.sum(singular>1e-9);null=right[rank:]
                        if not len(null):continue
                        bias[:]=null.T@(null@axis)
                else:
                    bias[(bias*normals).sum(axis=1)<0]*=-1
                candidate=original+epsilon*bias
                candidate/=np.linalg.norm(candidate,axis=1)[:,None]
                if np.min(np.sum(candidate*normals,axis=1))<0:continue
                try:artifacts=attempt(candidate)
                except Exception as error:
                    failures.append(f'{type(error).__name__}: {error}');continue
                angles=np.degrees(np.arctan2(np.linalg.norm(np.cross(original,candidate),axis=1),np.sum(original*candidate,axis=1)))
                record['metadata']['numerical_direction_recovery']=dict(reason=message,original_directions=original.tolist(),recovered_directions=candidate.tolist(),angles_deg=angles.tolist(),maximum_angle_deg=float(angles.max()),floor_constraints_preserved=True,common_direction_preserved=common,failed_retry_count=len(failures),geometry_policy='Real deterministic legal direction perturbation; all full/display swept solids reconstructed; no prism omission')
                record['metadata']['minimum_upward_dot']=float(np.min(np.sum(candidate*normals,axis=1)))
                if common:record['metadata']['reference_direction']=candidate[0].tolist()
                reference=np.asarray(record['metadata']['reference_direction'])
                record['metadata']['dispersion_objective']=float(np.sum(1-candidate@reference))
                for pose,direction in zip(group['poses'],candidate):
                    record['paths'][pose]=dict(direction_fixture=direction.tolist(),direction_world=(states[pose][1][:3,:3]@direction).tolist())
                print('PREPARED NUMERICAL DIRECTION RECOVERY',group['id'],float(angles.max()),flush=True)
                return group['id'],artifacts,None,record
        record['numerical_recovery_failures']=failures
    print('INITIALIZATION GEOMETRY UNRESOLVED',group['id'],message,flush=True)
    return group['id'],{},message,record

def prepare(name, groups=None, jobs=1):
    from optimization.initial_directions import initialize_close_directions
    folder=(HERE/'output'/name if name=='B' else HERE/'data/object_inputs'/name)/'data/step41';folder.mkdir(parents=True,exist_ok=True)
    manifest=selected_groups(name) if groups is None else groups
    poses=sorted({p for group in manifest for p in group['poses']});states={p:state(name,p) for p in poses}
    maximum=0.;inputs=[];records={}
    for group in manifest:
        path=(HERE/'output'/name if name=='B' else HERE/'data/object_inputs'/name)/group['id']/'step3/step3.3/support_with_rings.obj'
        support=trimesh.load(path,force='mesh',process=False);inputs.append(path)
        normals=np.array([initialize_direction(states[pose][1])[0] for pose in group['poses']])
        directions,metadata=initialize_close_directions(normals)
        records[group['id']]=dict(metadata=metadata,paths={})
        for pose,direction in zip(group['poses'],directions):
            task,T,mesh=states[pose]
            maximum=max(maximum,float((support.vertices@direction).max()-(mesh.vertices@direction).min()))
            records[group['id']]['paths'][pose]=dict(direction_fixture=direction.tolist(),direction_world=(T[:3,:3]@direction).tolist())
    length=max(.5,maximum+.02);artifacts={}
    arguments=[(name,group,records[group['id']],length,folder) for group in manifest]
    for group in manifest:
        for pose in group['poses']:inputs+=states[pose][0].inputs
    if jobs>1 and arguments:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        with ProcessPoolExecutor(max_workers=jobs,mp_context=multiprocessing.get_context('spawn')) as pool:
            prepared=list(pool.map(_prepare_group_sweeps,arguments))
    else:prepared=[_prepare_group_sweeps(a) for a in arguments]
    for group_id,group_artifacts,error,group_record in prepared:
        records[group_id]=group_record
        artifacts.update(group_artifacts)
        if error:records[group_id]['preparation_error']=error
    record=dict(complete=True,initialization_kind='Deterministic common or close legal floor directions per pose set',groups=records,full_length_m=length,display_length_m=.10,initialization_is_arbitrary=False,paths_independent_variables=True,provenance=provenance(inputs,[HERE/'step4.1/step41.py',HERE/'helper_func/optimization/initial_directions.py',HERE/'helper_func/exit_clearance.py',Path(S.__file__).with_name('translation_sweep.py')]),artifacts=artifacts)
    save(folder/'initialization.json',record)
    return record

def ground_coverage(mesh,T,points):
    world=transform_points(mesh.vertices,T);xy=world[np.abs(world[:,2])<=1e-9,:2]
    if len(xy)<3:return dict(passed=False,covered=0,total=len(points),ground_vertex_count=len(xy))
    try:
        hull=ConvexHull(xy);mask=np.all(points@hull.equations[:,:2].T+hull.equations[:,2]<=1e-9,axis=1)
        return dict(passed=bool(mask.all()),covered=int(mask.sum()),total=len(mask),ground_vertex_count=len(xy))
    except Exception as error:return dict(passed=False,covered=0,total=len(points),error=str(error))

from step41_legacy_render import render

def run(name,group,initialization,render_images=True):
    began=time.monotonic();base=(HERE/'output'/name if name=='B' else HERE/'data/object_inputs'/name)/group['id'];out=base/'step4/step4.1';(out/'data').mkdir(parents=True,exist_ok=True)
    seed_path=base/'step3/step3.3/support_with_rings.obj';seed_mesh=trimesh.load(seed_path,force='mesh',process=False);seed=S.solid(seed_mesh)
    folder=(HERE/'output'/name if name=='B' else HERE/'data/object_inputs'/name)/'data/step41';length=initialization['full_length_m']
    group_initialization=initialization.get('groups',{}).get(group['id'],initialization)
    cache_folder=folder/group['id'] if 'groups' in initialization else folder
    cached_full=[];display=[];directions=[];states=[]
    for pose in group['poses']:
        task,T,mesh=state(name,pose);states.append((task,T));directions.append(np.array(group_initialization['paths'][pose]['direction_fixture']))
        cached_full.append(S.solid(trimesh.load(cache_folder/pose/'full_sweep.obj',force='mesh',process=False)))
        display.append(trimesh.load(cache_folder/pose/'display_sweep.obj',force='mesh',process=False))
    policy=ExitClearance(mesh)
    with np.load(initialized_contacts(base)) as z:allowed=z['allowed_faces']
    allowed=allowed[np.max(mesh.face_normals[allowed]@np.asarray(directions).T,axis=1)<=1e-9]
    chosen_fan=8;best=None;fan_timings=[]
    for fan in [8]:  # Initialization diagnoses one construction; no acceptance search.
        trial_start=time.monotonic()
        individual=cached_full if fan==8 else [S.solid(S.swept_solid(mesh,length*d,fan_in=fan)) for d in directions]
        sweep_candidate=union(individual)
        padded_individual=[policy.sweep(length*d,fan) for d in directions]
        construct_start=time.monotonic()
        construction=policy.construct(seed,individual,padded_individual,allowed)
        construct_seconds=time.monotonic()-construct_start
        candidate_remaining=construction['remaining'];candidate_removed=construction['removed']
        partition_error=abs(material_volume(seed)-material_volume(candidate_removed)-material_volume(candidate_remaining))
        overlap=max(material_volume(candidate_remaining^part) for part in individual)
        score=max(partition_error,overlap,construction['diagnostics']['padded_sweep_overlap_outside_contact_cores_m3'])
        if not construction['diagnostics']['contact_area_preserved']:score=max(score,1.)
        fan_timings.append(dict(fan=fan,seconds=time.monotonic()-trial_start,construct_seconds=construct_seconds,score=float(score)))
        if best is None or score<best[0]:best=(score,fan,sweep_candidate,candidate_remaining,candidate_removed,partition_error,overlap,individual,construction,padded_individual)
        if score<1e-10:break
    _,chosen_fan,full,remaining,removed,_,residual_sweep_overlap,individual,construction,padded_individual=best
    removed_mesh=S.unpack(removed);remaining_mesh=S.unpack(remaining)
    raw_intersection_volume=material_volume(seed^full)
    D.export_exact_obj(S.unpack(full),out/'data/applied_sweep.obj')
    D.export_exact_obj(removed_mesh,out/'removed_support.obj');D.export_exact_obj(remaining_mesh,out/'remaining_support.obj')
    _,_,mesh=state(name,group['poses'][0]);triangles,sources=contact_boundary(mesh,remaining_mesh,allowed) if len(remaining_mesh.faces) else (np.empty((0,3,3)),np.empty(0,int))
    np.savez_compressed(out/'data/remaining_contacts.npz',triangles_mesh_m=triangles,source_faces=sources)
    geometry_seconds=time.monotonic()-began;classification_seconds=0.
    rows=[];inputs=[seed_path,base/'step3/step3.3/data/report.json',initialized_contacts(base),Path(initialization.get('source_path',folder/'initialization.json'))]
    for k,pose in enumerate(group['poses']):
        task,T=states[k];direction=directions[k];inputs+=task.inputs+[cache_folder/pose/'full_sweep.obj',cache_folder/pose/'display_sweep.obj']
        full_supply=supply(task,T,triangles,sources);error=None
        classification_start=time.monotonic()
        try:mask,info=J.classify(full_supply,task.targets)
        except (RuntimeError,AssertionError,np.linalg.LinAlgError) as e:mask=np.zeros(len(task.targets),bool);info={};error=str(e)
        classification_seconds+=time.monotonic()-classification_start
        floor_path=ROOT/'objects'/name/'poses'/pose/'floor_contact.npz';points=np.load(floor_path)['floor_demands_xy_m'];inputs.append(floor_path)
        coverage=ground_coverage(remaining_mesh,T,points)
        native=T[:3,:3]@direction;end=mesh.vertices+length*direction
        separated=bool((end@direction).min()>(seed_mesh.vertices@direction).max()+1e-9)
        row=dict(pose=pose,direction_fixture=direction.tolist(),direction_world=native.tolist(),path_kind='Straight translation along deterministic common or close legal direction',path_fixture_m=[[0.,0.,0.],(direction*length).tolist()],full_length_m=length,endpoint_completely_separated=separated,minimum_object_world_z_along_path_m=float(min(task.domain.mesh.vertices[:,2].min(),task.domain.mesh.vertices[:,2].min()+length*native[2])),own_removed_volume_cm3=material_volume(seed-(seed-individual[k]))*1e6,force_covered=int(mask.sum()),load_count=len(mask),force_passed=bool(mask.all()) if error is None else None,force_error=error,classifier=info,ground_coverage=coverage)
        rows.append(row);np.savez_compressed(out/'data'/f'{pose}.npz',force_mask=mask,supply_7d=full_supply,T_fixture_to_world=T,direction_world=native)
        print('INITIAL EXIT',group['id'],pose,row['force_covered'],flush=True)
    from step41_render import render as render_current
    render_start=time.monotonic()
    if render_images:render_current(name,group,directions=dict(zip(group['poses'],directions)),computed=dict(policy=policy,seed=seed,direction_metadata=group_initialization['metadata'],full_length_m=length,final_construction=construction,nominal_sweeps=dict(zip(group['poses'],individual)),padded_sweeps=dict(zip(group['poses'],padded_individual)),fan=chosen_fan))
    render_seconds=time.monotonic()-render_start
    conserved=abs(material_volume(seed)-material_volume(removed)-material_volume(remaining))
    partition_tolerance=max(1e-10,material_volume(seed)*1e-5)
    partition_resolved=best[0]<1e-10 and conserved<partition_tolerance and residual_sweep_overlap<1e-10
    force_path_gate=partition_resolved and all(row['force_passed'] and row['minimum_object_world_z_along_path_m']>=-1e-9 for row in rows)
    gate=force_path_gate and all(row['ground_coverage']['passed'] for row in rows)
    diagnostic_status='geometry_unresolved' if not partition_resolved else 'initialization_meets_force_and_path' if force_path_gate else 'needs_path_optimization'
    report=dict(passed=True,construction_completed=True,step4_ready=True,diagnostics_are_acceptance_gates=False,diagnostic_status=diagnostic_status,timings=dict(joint_geometry_seconds=geometry_seconds,all_load_classification_seconds=classification_seconds,render_seconds=render_seconds,joint_fan_trials=fan_timings),clearance_diagnostics=construction['diagnostics'],exit_clearance=policy.metadata,complete=True,stage='step4.1_initialize',pose_set=group['id'],initialization=initialization,state_results=rows,remaining_force_and_ground_gate=gate,force_and_path_initialization_gate=force_path_gate,ground_coverage_is_diagnostic_only=True,status='initialized',full_fixture_accepted=False,optimized=False,seed_volume_cm3=material_volume(seed)*1e6,removed_volume_cm3=material_volume(removed)*1e6,remaining_volume_cm3=material_volume(remaining)*1e6,volume_partition_error_m3=conserved,removed_geometry_policy='S0 minus computed remaining material; complement partition avoids unstable direct-intersection coplanar triangulation',direct_intersection_discrepancy_m3=abs(material_volume(removed)-raw_intersection_volume),remaining_sweep_overlap_m3=residual_sweep_overlap,volume_partition_tolerance_m3=partition_tolerance,geometry_partition_resolved=partition_resolved,sweep_boolean_fan_in=chosen_fan,remaining_component_count=len(remaining.decompose()),remaining_contact_triangle_count=len(triangles),remaining_contact_area_cm2=float(trimesh.triangles.area(triangles).sum())*1e4,cut_policy=policy.metadata['policy'],sweep_policy='Full continuous nonconvex object translation; no convex-hull approximation or sampled-frame cutting',initial_sweeps_identical=bool(np.allclose(directions,directions[0],atol=1e-9)),ground_model='Original object-floor reaction model retained for force diagnostic; actual remaining support ground coverage reported separately',validation_policy='One construction computation; no exported-model replay. Object exit stays above its own floor; installed support floor legality, connectivity and strength are not accepted here.',seconds=time.monotonic()-began,provenance=provenance(inputs,[HERE/'step4.1/step41.py',HERE/'helper_func/exit_clearance.py',Path(S.__file__).with_name('translation_sweep.py'),HERE/'helper_func/co_common.py',Path(J.__file__)]),artifacts={**{f'../{f}':I.sha256(out/f) for f in (['exit_directions.png','final_result.png','exit_sweeps.png','direction_space.png'] if render_images else [])+['removed_support.obj','remaining_support.obj']},'remaining_contacts.npz':I.sha256(out/'data/remaining_contacts.npz'),'applied_sweep.obj':I.sha256(out/'data/applied_sweep.obj')})
    save(out/'data/report.json',report)
    lines=[f"# {group['id']}：Step4.1 退出初始化",'', '各状态优先使用共同合法退出方向；不存在时，用固定多起点算法初始化相近合法方向。这是方向初始化，不保证全局最优或承载通过。', '', 'direction_space.png 只展示首个 pose 的六个演示方向：四个合法、两个明显向下非法，非法箭头打叉；只画保留支撑，切除区域留空，透明 sweep 展示 100 mm，没有文字，模型放大 50%且总图尺寸不变。exit_directions.png 与 exit_sweeps.png 按保存 pose 顺序横向排布；灰色物体、蓝色支撑、红色各 pose 自己的切除材料。final_result.png 展示共同切除后的同一支撑。实际切除使用完整退出距离。', '', f"实际切除 {report['removed_volume_cm3']:.2f} cm³，剩余 {report['remaining_volume_cm3']:.2f} cm³。",'', '| Pose | 剩余接触承载需求通过数 | 剩余实际接地覆盖 |','|---|---:|---:|']
    for row in rows:lines.append(f"| {row['pose']} | {row['force_covered']}/{row['load_count']} | {row['ground_coverage']['covered']}/{row['ground_coverage']['total']} |")
    lines+=['','各 pose 的路径由当前集合共同初始化；共同切除是所有路径扫掠的并集。exit_sweeps.png 每格透明区域仅显示当前 pose 的 sweep，红色仅显示该 pose 自己要求切掉的材料；final_result.png 展示全部 pose 共同切除后的结果。接地圈覆盖仅作诊断，不要求保住固定环；本阶段尚未重建接地材料。每次从未修改的 Step3.3 材料重新计算，允许恢复先前切掉的材料。', '', 'Step4.1 只生成方向和切除后的初始化支撑。载荷、接地、路径与几何检查只作诊断，不否决当前初始化；这些结果不是完整夹具接受。退出路径优化尚未运行。']
    (out/'README.md').write_text('\n'.join(lines)+'\n');return dict(id=group['id'],status=report['status'],passed=True,construction_completed=True,diagnostic_status=diagnostic_status,diagnostics_are_acceptance_gates=False,force_and_path_initialization_gate=force_path_gate,force_and_ground_gate=gate,removed_volume_cm3=report['removed_volume_cm3'],remaining_volume_cm3=report['remaining_volume_cm3'],seconds=report['seconds'])

def main():
    import sys
    if '--render-only' in sys.argv:
        sys.argv.remove('--render-only')
        from step41_render import main as render_main
        return render_main()
    parser=argparse.ArgumentParser();parser.add_argument('--object',default='B');parser.add_argument('--sets',nargs='+');parser.add_argument('--jobs',type=int,default=2);args=parser.parse_args();name=args.object;groups=selected_groups(name)
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    selected=[g for g in groups if not args.sets or g['id'] in args.sets or g['id'].removeprefix('illegal/') in args.sets]
    root=HERE/'output'/name if name=='B' else HERE/'data/object_inputs'/name
    ready=[];skipped=[]
    for group in selected:
        report=root/group['id']/'step3/step3.3/data/report.json'
        prior=json.loads(report.read_text()) if report.exists() else {}
        if prior.get('construction_completed',prior.get('passed',False)):ready.append(group)
        else:
            skipped.append(dict(id=group['id'],status='step3_gate_stopped',stopped_stage='step3.3',reason='Step3.3 seed construction is missing or numerically unresolved; coverage / work / floor diagnostics do not block initialization',failed_ground_poses=[r['pose'] for r in prior.get('state_results',[]) if not r.get('all_saved_demands_covered',False)],failed_work_clearance_poses=[r['pose'] for r in prior.get('state_geometry',[]) if not r.get('working_surface_clear',False)]))
    selected=ready
    initialization=prepare(name,selected,jobs=args.jobs)
    usable=[]
    for group in selected:
        error=initialization['groups'][group['id']].get('preparation_error')
        if error:skipped.append(dict(id=group['id'],status='geometry_unresolved',stopped_stage='step4.1',reason=error))
        else:usable.append(group)
    selected=usable
    with ProcessPoolExecutor(max_workers=max(1,args.jobs),mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(run,name,g,initialization) for g in selected];rows=[f.result() for f in futures]
    save((HERE/'output'/name if name=='B' else HERE/'data/object_inputs'/name)/'data/step41_batch.json',dict(complete=True,results=rows,skipped=skipped,sets=len(rows),optimized=False))
    print('INITIALIZATION COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()
