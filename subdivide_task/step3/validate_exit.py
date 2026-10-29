"""Replay continuous whole-object exits and loads on the final saved support."""
import argparse,json,time
from pathlib import Path
import numpy as np
import trimesh
import shapely
from PIL import Image
from subdivide_task.step1.common import ROOT,DEFAULT_OUTPUT,load_shape,sha256,write_json
from . import independent_geometry as G
from .exit_geometry import continuous_checks
from .run_independent import make_domain
from .independent_physics import demands,certificate


def validate(output_root=DEFAULT_OUTPUT,object_name='B'):
    start=time.monotonic();folder=Path(output_root)/object_name/'step3/step3.2'
    report=json.loads((folder/'report.json').read_text());policy=report['policy'];tol=policy['boolean_volume_tolerance_m3']
    for name,h in report['input_sha256'].items():assert sha256(name)==h,name
    for name,h in report['artifact_sha256'].items():assert sha256(folder/name)==h,name
    with Image.open(folder/'overview.png')as im:assert list(im.size)==report['visualization']['overview_size']
    with np.load(folder/'initialization.npz')as z:arrays={k:z[k]for k in z.files}
    source_folder=Path(report['source_step31'])
    before_report=json.loads((source_folder/'report.json').read_text())
    with np.load(source_folder/'initialization.npz')as z:before={k:z[k]for k in z.files}
    old=G.solid(trimesh.Trimesh(before['support_vertices_m'],before['support_faces'],process=False))
    support=trimesh.Trimesh(arrays['support_vertices_m'],arrays['support_faces'],process=False)
    assert support.is_watertight and support.is_winding_consistent and support.volume>0
    exported=trimesh.load(folder/'support.obj',force='mesh',process=False)
    np.testing.assert_allclose(exported.vertices,support.vertices,atol=1e-15,rtol=0);np.testing.assert_array_equal(exported.faces,support.faces)
    current=G.solid(support);assert G.volume(current-old)<=tol
    assert sum(p.volume()*G.SCALE**3>1e-15 for p in current.decompose())==1
    assert abs(support.bounds[0,2])<1e-12
    assert report['exit_direction_mode']==report['exit_length_mode']=='independent_per_task'
    assert report['exit_policy']['direction_mode']==report['exit_policy']['displacement_mode']=='independent_per_task'
    assert report['feasible_initialization'] and not report['optimization_performed']
    assert len(report['tasks'])==len(before_report['tasks'])==report['task_count']
    assert len({r['id']for r in report['tasks']})==report['task_count']
    source=load_shape(ROOT/'objects'/object_name/'mesh.stl')
    np.testing.assert_array_equal(source.vertices,arrays['source_vertices_object']);np.testing.assert_array_equal(source.faces,arrays['source_faces'])
    ground=support.triangles[np.abs(support.triangles[:,:,2]).max(1)<1e-9,:,:2]
    footprint=shapely.union_all(shapely.polygons(ground));checked=[]
    for record,previous in zip(report['tasks'],before_report['tasks']):
        id=record['id'];assert id==previous['id'];T=arrays[id+'_world_from_object'];ids=arrays[id+'_working_face_ids']
        np.testing.assert_array_equal(T,before[id+'_world_from_object']);np.testing.assert_array_equal(ids,before[id+'_working_face_ids'])
        assert record['workspace']==previous['workspace']
        np.testing.assert_array_equal(T,record['world_from_object']);np.testing.assert_array_equal(ids,record['working_face_ids'])
        direction=np.asarray(record['exit_direction_world']);length=float(record['exit_length_m'])
        assert np.isfinite(direction).all() and np.isfinite(length)
        np.testing.assert_allclose(np.linalg.norm(direction),1.,atol=1e-12,rtol=0)
        assert direction[2]>=report['exit_policy']['minimum_upward_component'] and length>0
        np.testing.assert_array_equal(direction,arrays[id+'_exit_direction_world']);assert float(arrays[id+'_exit_length_m'])==length
        assert abs(record['workspace']['reference_head_front_world_m'][2]-report['common_tool_head_height_m'])<1e-12
        body=source.copy();body.apply_transform(T);body_solid=G.solid(body);work,_=G.work_sector(record['workspace'],policy)
        continuous,_=continuous_checks(current,[(id,body,direction,length)],tol)
        assert continuous[0]['passed'],continuous[0]
        assert continuous[0]['leading_face_count']==int(arrays[id+'_sweep_leading_face_count'])
        overlap=dict(object=G.volume(current^body_solid),work=G.volume(current^work),
            continuous_whole_object_exit=continuous[0]['continuous_overlap_upper_bound_m3'])
        assert max(overlap.values())<=tol,(id,overlap)
        # Additional independent placements supplement the full face-prism
        # construction; they are not used as a continuous-path certificate.
        for t in (.05,.25,.5,.75,1.):
            moved=body.copy();moved.apply_translation(t*length*direction);m=G.solid(moved)
            assert G.volume(current^m)<=tol,(id,t,'placement intersects support')
        end=body.vertices+length*direction
        assert end[:,2].min()>support.bounds[1,2]+report['exit_policy']['final_vertical_separation_m']-1e-12
        domain=make_domain(source,T,ids,id,record['workspace'],report['physical_properties'],policy)
        endpoints,samples=demands(domain,policy);proof,data=certificate(body,support,domain,endpoints,samples)
        assert proof['passed'],(id,proof)
        for key in ['contact_triangles_world_m','contact_source_face_ids','supply_7d','wrench_scale','endpoint_passed','sample_passed']:
            np.testing.assert_allclose(arrays[id+'_'+key],data[key],atol=1e-12,rtol=0)
        for key in ('original_sample_count','passed_original_sample_count','continuous_outer_target_count','passed_continuous_outer_target_count'):
            assert record['force'][key]==proof[key],(id,key)
        assert not np.isin(data['contact_source_face_ids'],ids).any()
        pressure=G.pressure_points(endpoints['endpoint_need_wrench'],domain.com)
        assert shapely.covers(footprint,shapely.points(pressure)).all(),id+' base pressure'
        checked.append(dict(id=id,exit_direction_world=direction.tolist(),exit_length_m=length,
            final_assembly_audit_passed=True,force=proof,forbidden_overlap_m3=overlap,
            all_positive_leading_prisms_checked=True,leading_face_count=continuous[0]['leading_face_count'],
            final_vertical_gap_m=float(end[:,2].min()-support.bounds[1,2])))
        print('EXIT REPLAY',id,'PASS',flush=True)
    assert len(checked)==report['task_count']==before_report['task_count']
    assert report['all_final_assembly_audits_passed'] and all(r['final_assembly_audit_passed']for r in report['tasks'])
    assert report['original_sample_count']==report['passed_original_sample_count']==sum(t['force']['original_sample_count']for t in checked)
    assert report['continuous_outer_target_count']==report['passed_continuous_outer_target_count']==sum(t['force']['continuous_outer_target_count']for t in checked)
    result=dict(status='passed',stage='step3.2',task_count=len(checked),exit_direction_mode='independent_per_task',
        feasible_initialization=True,optimization_performed=False,all_final_assembly_audits_passed=True,
        whole_object_continuous_paths_verified=True,
        poses_and_working_areas_unchanged_from_step31=True,actual_contact_loads_rebuilt_after_carving=True,
        original_sample_count=sum(t['force']['original_sample_count']for t in checked),
        continuous_outer_target_count=sum(t['force']['continuous_outer_target_count']for t in checked),
        connected_closed_grounded_material=True,tasks=checked,working_tool_retracted_before_extraction=True,
        native_grasp_or_full_arm_certified=False,seconds=time.monotonic()-start,
        report_sha256=sha256(folder/'report.json'),validator_sha256=sha256(__file__))
    write_json(folder/'validation.json',result)
    print('STEP3.2 REPLAY PASS',len(checked),'independent exits and final-assembly loads',flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--object',default='B');a=p.parse_args();validate(a.output_root,a.object)
