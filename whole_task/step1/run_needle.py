"""Current Step1: all source surface inside the 100mm, 25-degree sector."""
import argparse,json,shutil,time
from pathlib import Path
import numpy as np
from robots_config.constraints import load_config
from .common import ROOT,DEFAULT_OUTPUT,load_shape,sha256,write_json
from .needle_geometry import POLICY,workspace,build_candidates,evaluate,proposal_depths
from .step1_3 import _adjacency
from .step1_2 import greedy_cover
from .pose_coverage import face_components

SOURCES=['run.py','needle_sampling.py','run_needle.py','needle_geometry.py','needle_render.py','validate_needle.py',
         'needle_workspace.json','common.py','step1_3.py','step1_2.py','pose_coverage.py','work_disks.py','coverage_render.py']


def bindings():
    return {str((Path(__file__).parent/n).relative_to(ROOT)):sha256(Path(__file__).parent/n) for n in SOURCES}


def transforms(rows):
    result=np.tile(np.eye(4),(len(rows),1,1))
    for T,row in zip(result,rows):T[:3,:3]=row['rotation_world_object'];T[:3,3]=row['translation_world_object_m']
    return result


def describe(mesh,row,index,id):
    ids=np.asarray(row['face_ids']);head=row['head_object_m'];points=mesh.triangles_center[ids]
    direction=(points-head);direction/=np.linalg.norm(direction,axis=1)[:,None]
    nominal=-row['outward_axis_object'];angles=np.rad2deg(np.arccos(np.clip(direction@nominal,-1,1)))
    return dict(id=id,candidate_index=index,face_ids=ids.tolist(),face_count=len(ids),
        area_m2=float(mesh.area_faces[ids].sum()),area_fraction=float(mesh.area_faces[ids].sum()/mesh.area),
        reference_head_object_m=head.tolist(),outward_axis_object=row['outward_axis_object'].tolist(),
        camera_view_object=row['view_direction_object'].tolist(),
        screening_world_from_object=transforms([row])[0].tolist(),
        minimum_centroid_needle_tilt_deg=float(angles.min()),maximum_centroid_needle_tilt_deg=float(angles.max()),
        all_geometrically_inside_source_faces_included=True,
        connected_component_count=len(face_components(ids,_adjacency(mesh))),selection_score_m2=row['selection_score_m2'])


def choose_examples(mesh,config,ws,rows):
    neighbors=_adjacency(mesh);chosen=[]
    targets=[(1071,None,'ear'),(2427,[-1,0,0],'mouth')]
    depths=proposal_depths(ws)
    for f,outward,label in targets:
        axis=mesh.face_normals[f] if outward is None else np.asarray(outward,dtype=float)
        options=[item for depth in depths for item in evaluate(mesh,mesh.triangles_center[f]+depth*axis,axis,config,ws,neighbors,anchor=f)]
        if not options:raise ValueError('No geometrically legal '+label+' display candidate')
        item=max(options,key=lambda r:r['area_m2']);item['display_target']=label;chosen.append(item)
    for item in rows:
        if all(item['outward_axis_object']@old['outward_axis_object']<np.cos(np.deg2rad(35)) for old in chosen):
            chosen.append(item)
        if len(chosen)==4:break
    if len(chosen)!=4:raise ValueError('Four distinct display candidates unavailable')
    return chosen


def candidate_rows(archive):
    return [dict(face_ids=np.flatnonzero(mask),head_object_m=archive['heads'][i],
        rotation_world_object=archive['rotations'][i],translation_world_object_m=archive['translations'][i],
        outward_axis_object=archive['axes'][i],view_direction_object=archive['views'][i],
        selection_score_m2=float(archive['scores'][i])) for i,mask in enumerate(archive['masks'])]


def run(output_root=DEFAULT_OUTPUT,object_name='B',config_path=None,inset=.05):
    start=time.monotonic();output_root=Path(output_root).resolve()
    source=Path(config_path).resolve() if config_path else ROOT/'robots_config/04_kuka_lbr_med14_r820.json'
    config=load_config(source);ws=workspace(config,inset)
    shape=ROOT/'objects'/object_name/'mesh.stl';mesh=load_shape(shape)
    work=ROOT/'whole_task/data'/object_name/'step1_needle_work';work.mkdir(parents=True,exist_ok=True)
    cache=work/'candidate_cache.json'
    cached=json.loads(cache.read_text()) if cache.exists() else None
    if (cached and cached['workspace']==ws
            and cached.get('arrays_sha256')==sha256(work/'candidates.npz')
            and all(sha256(ROOT/p)==v for p,v in cached['input_sha256'].items())):
        with np.load(work/'candidates.npz') as archive:rows=candidate_rows({p:archive[p]for p in archive.files})
    else:
        rows=build_candidates(mesh,config,ws)
        from .needle_sampling import augment
        rows=augment(mesh,config,ws,rows)
        pool_masks=np.zeros((len(rows),len(mesh.faces)),bool)
        for i,row in enumerate(rows):pool_masks[i,row['face_ids']]=True
        np.savez_compressed(work/'candidates.npz',masks=pool_masks,heads=[r['head_object_m']for r in rows],
            rotations=[r['rotation_world_object']for r in rows],translations=[r['translation_world_object_m']for r in rows],
            axes=[r['outward_axis_object']for r in rows],views=[r['view_direction_object']for r in rows],
            scores=[r['selection_score_m2']for r in rows])
        write_json(cache,dict(workspace=ws,arrays_sha256=sha256(work/'candidates.npz'),candidate_count=len(rows),
            input_sha256={**bindings(),str(shape.relative_to(ROOT)):sha256(shape),str(source.relative_to(ROOT)):sha256(source)}))
    for row in rows:row['area_m2']=float(mesh.area_faces[row['face_ids']].sum())
    masks=np.zeros((len(rows),len(mesh.faces)),bool)
    for i,row in enumerate(rows):masks[i,row['face_ids']]=True
    selected,trace=greedy_cover(masks,mesh.area_faces)
    examples=choose_examples(mesh,config,ws,rows);all_rows=rows+examples
    all_masks=np.vstack([masks,np.array([np.isin(np.arange(len(mesh.faces)),r['face_ids'])for r in examples])])
    staged=work/'publish';stage=staged/object_name/'step1'
    for sub in ('step1.1','step1.2','step1.3'):(stage/sub).mkdir(parents=True,exist_ok=True)
    effective=dict(config,working_tool=json.loads(POLICY.read_text()))
    write_json(stage/'step1.1/config.json',effective)
    from .needle_render import draw_configuration,draw_examples
    visual=draw_configuration(effective,ws,stage/'step1.1')
    write_json(stage/'step1.1/selection.json',dict(schema_version=3,stage='step1.1',object=object_name,
        selected_config=config['id'],mode='user_robot_plus_fixed_needle_task_input',source=str(source),source_sha256=sha256(source),
        task_tool_source=str(POLICY),task_tool_sha256=sha256(POLICY),config_file='config.json',config_sha256=sha256(stage/'step1.1/config.json'),
        needle_length_m=.1,needle_extensible=False,workspace=ws,figure='result.png',figure_sha256=sha256(stage/'step1.1/result.png'),visualization=visual))
    out=stage/'step1.3';Ts=transforms(all_rows)
    np.savez_compressed(out/'pose_coverage.npz',vertices_object=mesh.vertices,faces=mesh.faces,
        source_face_ids=np.arange(len(mesh.faces)),face_area_m2=mesh.area_faces,
        candidate_face_membership=all_masks,representative_candidate_indices=np.arange(len(rows),len(all_rows)),
        candidate_rotation_world_object=Ts[:,:3,:3],candidate_translation_world_object_m=Ts[:,:3,3],
        candidate_outward_axis_object=[r['outward_axis_object']for r in all_rows],
        candidate_reference_head_object_m=[r['head_object_m']for r in all_rows],
        candidate_camera_view_object=[r['view_direction_object']for r in all_rows],
        candidate_selection_score_m2=[r['selection_score_m2']for r in all_rows],
        working_area_face_membership=all_masks[-4:])
    source_bind=bindings();source_bind[str(shape.relative_to(ROOT))]=sha256(shape);source_bind[str(source.relative_to(ROOT))]=sha256(source)
    report=dict(schema_version=3,geometry_model=ws['model'],object=object_name,robot_config_id=config['id'],
        shape_sha256=sha256(shape),config_sha256=sha256(stage/'step1.1/config.json'),workspace=ws,
        needle_length_m=.1,needle_extensible=False,working_area_count=4,candidate_count=len(all_rows),automatic_candidate_count=len(rows),
        selected_area_fraction=float(mesh.area_faces[all_masks[-4:].any(0)].sum()/mesh.area),
        working_areas=[describe(mesh,r,len(rows)+i,f'C{i+1:02}')for i,r in enumerate(examples)],
        arrays='pose_coverage.npz',input_sha256=source_bind,
        hole_filling=dict(enabled=False,reason='removed; the whole geometric sector intersection is the region'),
        selection_filters=dict(surface_normal=False,visibility=False,inner_shell=False,head_clearance=False,
            flange_reach=False,connected_component=False),
        angular_budget=dict(tool_cone_half_angle_deg=ws['half_angle_deg'],working_tool_cone_half_angle_deg=ws['usable_half_angle_deg'],
            tool_direction_mode='geometric_radial_sector',normal_hard_limit_applied=False),
        native_robot_motion_verified=False,full_tool_collision_verified=False,head_sphere_proxy_screened=False)
    report['visualization']=draw_examples(mesh,all_masks[-4:],Ts[-4:],[r['head_object_m']for r in examples],
        [r['view_direction_object']for r in examples],ws,out)
    report['figures']=['working_areas_overview.png'];report['artifact_sha256']={p:sha256(out/p)for p in [report['arrays'],*report['figures']]}
    write_json(out/'pose_coverage.json',report)
    chosen=[rows[int(i)]for i in selected];masks=masks[selected];counts=masks.sum(0)
    for i,(row,item,index) in enumerate(zip(trace,chosen,selected)):
        row.update(describe(mesh,item,int(index),f'W{i+1:02}'))
    out=stage/'step1.2'
    np.savez_compressed(out/'coverage_plan.npz',vertices_object=mesh.vertices,faces=mesh.faces,
        source_face_ids=np.arange(len(mesh.faces)),face_area_m2=mesh.area_faces,candidate_indices=selected,
        working_area_ids=np.asarray([r['id']for r in trace]),working_area_face_membership=masks,
        coverage_count=counts,first_covering_area_index=masks.argmax(0),screening_world_from_object=transforms(chosen),
        outward_axis_object=[r['outward_axis_object']for r in chosen],reference_head_object_m=[r['head_object_m']for r in chosen])
    plan=dict(schema_version=3,geometry_model=ws['model'],object=object_name,algorithm='area_weighted_greedy_set_cover',
        candidate_count=len(rows),working_area_count=len(selected),requested_working_area_count=None,
        coverage_target='entire_original_surface',source_face_count=len(mesh.faces),surface_area_m2=float(mesh.area),
        covered_face_count=int((counts>0).sum()),coverage_fraction=float(mesh.area_faces[counts>0].sum()/mesh.area),
        uncovered_source_face_ids=np.flatnonzero(counts==0).tolist(),workspace=ws,needle_length_m=.1,needle_extensible=False,
        complete_geometric_intersections_preserved=True,connected_components_kept_together=True,overlap_allowed=True,
        summed_working_area_m2=float(mesh.area_faces@counts),mean_area_weighted_multiplicity=float(mesh.area_faces@counts/mesh.area),
        maximum_face_multiplicity=int(counts.max()),region_count_globally_minimized=False,
        fixture_aware_optimization=False,native_robot_motion_verified=False,support_verified=False,
        working_areas=trace,arrays='coverage_plan.npz',input_sha256=source_bind,
        upstream_sha256={'step1.1/config.json':sha256(stage/'step1.1/config.json'),'step1.3/pose_coverage.json':sha256(stage/'step1.3/pose_coverage.json'),
            'step1.3/pose_coverage.npz':sha256(stage/'step1.3/pose_coverage.npz')},
        computation_elapsed_s=time.monotonic()-start)
    from .coverage_render import render
    display=dict(plan,working_areas=[dict(row,outward_axis_object=row['camera_view_object'])for row in trace])
    plan['visualization']=render(mesh,masks,transforms(chosen),display,out)
    plan['visualization']['region_views']='from_each_reference_head_toward_selected_surface'
    plan['figures']=['working_areas_overview.png','coverage_progress.png']
    plan['artifact_sha256']={p:sha256(out/p)for p in [plan['arrays'],*plan['figures']]}
    write_json(out/'coverage_plan.json',plan)
    from .validate_needle import validate
    validate(staged,object_name)
    target=output_root/object_name/'step1';target.mkdir(parents=True,exist_ok=True)
    for sub in ('step1.1','step1.3','step1.2'):
        if (target/sub).exists():shutil.rmtree(target/sub)
        shutil.copytree(stage/sub,target/sub)
    validate(output_root,object_name)
    print('STEP1 NEEDLE PASS',json.dumps(dict(working_area_count=len(selected),coverage_fraction=plan['coverage_fraction'],
        first13_coverage=trace[12]['cumulative_coverage_fraction'] if len(trace)>=13 else 1.,seconds=time.monotonic()-start)),flush=True)
    return plan


if __name__ == '__main__':
    # Public execution uses the current bounded-translation model.
    from .plan_reach import main as current_main
    current_main()
