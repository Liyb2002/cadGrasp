"""Replay the actual common plate, task poses, work openings and paired loads."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import trimesh
import shapely
from PIL import Image
from subdivide_task.step1.common import ROOT,DEFAULT_OUTPUT,load_shape,sha256,write_json
from subdivide_task.step1.needle_geometry import geometric_mask
from .run_independent import make_domain,resting_pose,moved_workspace
from . import independent_geometry as G
from .independent_physics import demands,certificate


def validate(output_root=DEFAULT_OUTPUT,object_name='B'):
    start=time.monotonic();folder=Path(output_root)/object_name/'step3/step3.1'
    report=json.loads((folder/'report.json').read_text());policy=report['policy']
    for filename,h in report['input_sha256'].items():assert sha256(ROOT/filename)==h,filename
    for filename,h in report['artifact_sha256'].items():assert sha256(folder/filename)==h,filename
    with Image.open(folder/'overview.png') as image:assert list(image.size)==report['visualization']['overview_size']
    views=report['visualization']['working_poses']
    with Image.open(folder/'working_poses.png') as image:assert list(image.size)==views['size']
    with np.load(folder/'initialization.npz') as z:arrays={k:z[k] for k in z.files}
    source=load_shape(ROOT/'objects'/object_name/'mesh.stl')
    np.testing.assert_array_equal(arrays['source_vertices_object'],source.vertices)
    np.testing.assert_array_equal(arrays['source_faces'],source.faces)
    support=trimesh.Trimesh(arrays['support_vertices_m'],arrays['support_faces'],process=False)
    obj=trimesh.load(folder/'support.obj',force='mesh',process=False)
    np.testing.assert_allclose(obj.vertices,support.vertices,atol=1e-15,rtol=0)
    np.testing.assert_array_equal(obj.faces,support.faces)
    assert support.is_watertight and support.is_winding_consistent and support.volume>0
    solid=G.solid(support);tolerance=policy['boolean_volume_tolerance_m3']
    positives=[p for p in solid.decompose() if p.volume()*G.SCALE**3>1e-15]
    assert len(positives)==1,'Common plate and all molds must be one connected solid'
    assert abs(support.bounds[0,2])<1e-12
    plate=arrays['plate_bounds_m'];np.testing.assert_allclose(plate,report['plate_bounds_m'],atol=0,rtol=0)
    assert plate[0,2]==0 and plate[1,2]==policy['plate_thickness_m']
    bedding=trimesh.Trimesh(arrays['bedding_vertices_m'],arrays['bedding_faces'],process=False)
    bedding_solid=G.solid(bedding)
    assert G.volume(bedding_solid-solid)<=tolerance
    common_height=max(r['workspace_local']['reference_head_front_world_m'][2] for r in report['tasks'])
    assert abs(common_height-report['common_tool_head_height_m'])<1e-12
    ground=support.triangles[np.abs(support.triangles[:,:,2]).max(1)<1e-9,:,:2]
    footprint=shapely.union_all(shapely.polygons(ground));assert not footprint.is_empty
    plan_path=next(Path(p) for p in report['input_sha256'] if p.endswith('step1/step1.2/coverage_plan.json'))
    plan=json.loads(plan_path.read_text())
    with np.load(plan_path.with_suffix('.npz')) as z:
        references=z['screening_world_from_object'][:report['task_count']].copy()
        masks=z['working_area_face_membership'][:report['task_count']].copy()
    checked=[];intervals=[]
    for index,record in enumerate(report['tasks']):
        id=record['id'];ids=np.asarray(record['working_face_ids']);T=arrays[id+'_world_from_object']
        assert id==plan['working_areas'][index]['id'] and not record['final_pose_fixed']
        np.testing.assert_array_equal(ids,np.flatnonzero(masks[index]));np.testing.assert_array_equal(ids,arrays[id+'_working_face_ids'])
        expected,ws,shift=resting_pose(source,references[index],plan['workspace'],record['initial_axial_angle_deg'],plate[1,2])
        offset=np.asarray(record['row_offset_m']);expected[:3,3]+=offset;ws=moved_workspace(ws,offset)
        lift=common_height-record['workspace_local']['reference_head_front_world_m'][2]
        assert lift>=-1e-12
        assert abs(offset[2]-lift)<1e-12 and abs(record['riser_height_m']-lift)<1e-12
        assert abs(ws['reference_head_front_world_m'][2]-common_height)<1e-12
        assert abs(ws['reference_head_front_world_m'][1]-plan['workspace']['reference_head_front_world_m'][1])<1e-12
        np.testing.assert_allclose(T,expected,atol=1e-12,rtol=0)
        np.testing.assert_allclose(T,record['world_from_object'],atol=0,rtol=0)
        assert ws==record['workspace']
        body=source.copy();body.apply_transform(T)
        assert abs(body.bounds[0,2]-(plate[1,2]+lift))<1e-12,id+' is not resting on its height-adjusted seat'
        if lift>1e-12:
            riser=np.asarray(record['riser_bounds_m']);local_base=np.asarray(record['plate_bounds_local_m'])
            np.testing.assert_allclose(riser[:,:2],local_base[:,:2]+offset[:2],atol=1e-12,rtol=0)
            np.testing.assert_allclose(riser[:,2],[0.,plate[1,2]+lift],atol=1e-12,rtol=0)
        else:assert record['riser_bounds_m'] is None
        head=(np.asarray(ws['reference_head_front_world_m'])-T[:3,3])@T[:3,:3]
        np.testing.assert_array_equal(geometric_mask(source,head,T[:3,:3].T@np.asarray(ws['nominal_axis_world']),ws),masks[index])
        delta=body.triangles[ids]-np.asarray(ws['reference_head_front_world_m'])
        distances=np.linalg.norm(delta,axis=2)
        angles=np.rad2deg(np.arccos(np.clip((delta@np.asarray(ws['nominal_axis_world']))/distances,-1,1)))
        assert distances.max()<=ws['needle_length_m']+1e-12
        assert angles.max()<=ws['half_angle_deg']+1e-10
        view=views['scenes'][index];assert view['id']==id
        shift=np.asarray(view['display_translation_m']);assert abs(shift[2]+lift)<1e-12
        pictured=T.copy();pictured[:3,3]+=shift
        np.testing.assert_allclose(view['world_from_object'],pictured,atol=1e-12,rtol=0)
        np.testing.assert_allclose(view['head_front_world_m'],np.asarray(ws['reference_head_front_world_m'])+shift,atol=1e-12,rtol=0)
        np.testing.assert_allclose(np.asarray(view['head_front_world_m'])[:2],np.asarray(plan['workspace']['reference_head_front_world_m'])[:2],atol=1e-12,rtol=0)
        np.testing.assert_allclose(view['nominal_axis_world'],plan['workspace']['nominal_axis_world'],atol=0,rtol=0)
        assert view['half_angle_deg']==ws['half_angle_deg'] and view['radius_m']==ws['needle_length_m']
        work,_=G.work_sector(ws,policy)
        overlap=dict(object=G.volume(solid^G.solid(body)),working_tool_space=G.volume(solid^work))
        assert max(overlap.values())<=tolerance,(id,overlap)
        assert max(G.volume(bedding_solid^G.solid(body)),G.volume(bedding_solid^work))<=tolerance
        domain=make_domain(source,T,ids,id,ws,report['physical_properties'],policy);endpoints,samples=demands(domain,policy)
        force,contacts=certificate(body,support,domain,endpoints,samples)
        assert force['passed'],(id,force)
        for name in ['contact_triangles_world_m','contact_source_face_ids','supply_7d','wrench_scale']:
            np.testing.assert_allclose(arrays[id+'_'+name],contacts[name],atol=1e-12,rtol=0)
        assert not np.isin(contacts['contact_source_face_ids'],ids).any()
        assert arrays[id+'_sample_passed'].all() and arrays[id+'_endpoint_passed'].all()
        pressure=G.pressure_points(endpoints['endpoint_need_wrench'],domain.com)
        assert shapely.covers(footprint,shapely.points(pressure)).all(),id+' pressure lies outside actual plate contact'
        low,high=record['cell_x_interval_m']
        span=np.vstack([body.vertices,G.mesh(work).vertices])
        assert span[:,0].min()>=low-1e-12 and span[:,0].max()<=high+1e-12
        intervals.append([low,high])
        checked.append(dict(id=id,force=force,forbidden_overlap_m3=overlap,
            complete_step1_mask_preserved_in_gold_sector=True,
            selected_vertex_maximum_radius_mm=float(distances.max()*1000),
            selected_vertex_maximum_angle_deg=float(angles.max()),
            object_bottom_on_height_adjusted_seat=True,riser_height_m=float(lift),
            tool_head_height_m=float(ws['reference_head_front_world_m'][2]),actual_base_pressure_contained=True))
        print('REPLAY',id,'COMMON PLATE PASS',flush=True)
    intervals.sort()
    assert all(intervals[j+1][0]-intervals[j][1]>=policy['row_gap_m']-1e-12 for j in range(len(intervals)-1))
    # Adjacent task cells contain objects, their molds and their work sectors;
    # only the single flat base intentionally connects the cells.
    assert len(checked)==report['task_count']==13
    assert not report['exit_carving_performed']
    result=dict(status='passed',stage='step3.1',task_count=len(checked),plate_count=1,
        connected_material_component_count=1,all_seats_share_one_ground_plate=True,
        all_tool_heads_same_height=True,common_tool_head_height_m=float(common_height),
        bedding_is_actual_support_subset=True,
        original_sample_count=sum(c['force']['original_sample_count'] for c in checked),
        continuous_outer_target_count=sum(c['force']['continuous_outer_target_count'] for c in checked),
        original_masks_unchanged=True,actual_common_support_contacts_rebuilt=True,
        all_pink_source_triangles_inside_saved_step1_gold_sectors=True,
        task_objects_pairwise_disjoint=True,own_working_spaces_clear=True,exit_paths_checked=False,
        native_arm_or_grasp_certified=False,tasks=checked,seconds=time.monotonic()-start,
        report_sha256=sha256(folder/'report.json'),validator_sha256=sha256(__file__))
    write_json(folder/'validation.json',result)
    print('STEP3.1 REPLAY PASS',len(checked),'tasks; one flat plate',flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--object',default='B')
    p.add_argument('--output-root',type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();validate(a.output_root,a.object)
