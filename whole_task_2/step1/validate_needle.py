"""Replay pure ball/cone membership, without claiming tool/path feasibility."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from robots_config.constraints import load_config
from .common import ROOT,DEFAULT_OUTPUT,load_shape,sha256,write_json
from .step1_3 import _adjacency
from .pose_coverage import face_components
from .needle_geometry import POLICY,workspace
from .step1_2 import greedy_cover


def check_region(mesh,mask,T,head,ws):
    O=np.asarray(head);forward=T[:3,:3].T@np.asarray(ws['nominal_axis_world'])
    np.testing.assert_allclose(T[:3,:3]@O+T[:3,3],ws['reference_head_front_world_m'],atol=1e-12,rtol=0)
    delta=mesh.triangles-O;radius=np.linalg.norm(delta,axis=2)
    expected=np.all((radius<=ws['needle_length_m']+1e-12)
        &(delta@forward>=radius*np.cos(np.deg2rad(ws['half_angle_deg']))-1e-12),axis=1)
    # Equality in both directions is essential: no normal/visibility/clearance
    # predicate may remove a triangle which is geometrically inside the sector.
    np.testing.assert_array_equal(mask,expected)
    assert mask.any()
    if mesh.is_watertight:assert not mesh.contains(O[None])[0]
    ids=np.flatnonzero(mask)
    norm=radius[mask];cosine=(delta[mask]@forward)/np.maximum(norm,1e-30)
    return dict(face_count=len(ids),surface_area_cm2=float(mesh.area_faces[ids].sum()*1e4),
        all_inside_source_faces_selected=True,omitted_inside_face_count=int((expected&~mask).sum()),
        selected_outside_face_count=int((mask&~expected).sum()),
        connected_component_count=len(face_components(ids,_adjacency(mesh))),
        maximum_head_distance_m=float(norm.max()),
        maximum_radial_angle_deg=float(np.rad2deg(np.arccos(np.clip(cosine,-1,1))).max()),
        normal_filter_applied=False,visibility_filter_applied=False,inner_shell_filter_applied=False,
        head_clearance_filter_applied=False,hole_filling_applied=False,
        boundary_discretization=ws['boundary_discretization'],native_or_full_tool_collision_certified=False)


def validate(output_root=DEFAULT_OUTPUT,object_name='B'):
    stage=Path(output_root)/object_name/'step1';first=json.loads((stage/'step1.1/selection.json').read_text())
    assert sha256(first['source'])==first['source_sha256']
    assert sha256(first['task_tool_source'])==first['task_tool_sha256']
    assert sha256(stage/'step1.1/config.json')==first['config_sha256']
    assert sha256(stage/'step1.1/result.png')==first['figure_sha256']
    config=json.loads((stage/'step1.1/config.json').read_text())
    assert config==dict(load_config(first['source']),working_tool=json.loads(POLICY.read_text()))
    mesh=load_shape(ROOT/'objects'/object_name/'mesh.stl')
    three=json.loads((stage/'step1.3/pose_coverage.json').read_text());two=json.loads((stage/'step1.2/coverage_plan.json').read_text())
    for report,folder in [(three,stage/'step1.3'),(two,stage/'step1.2')]:
        for p,h in report['input_sha256'].items():assert sha256(ROOT/p)==h,p
        for p,h in report['artifact_sha256'].items():assert sha256(folder/p)==h,p
        for p,h in report['visualization']['source_sha256'].items():assert sha256(ROOT/p)==h,p
    for p,h in two['upstream_sha256'].items():assert sha256(stage/p)==h,p
    assert three['workspace']==two['workspace']==first['workspace']
    ws=three['workspace'];assert ws==workspace(config,ws['proposal_inset_fraction'])
    assert ws['maximum_reference_distance_m']==ws['needle_length_m']==.1
    assert ws['half_angle_deg']==ws['usable_half_angle_deg']==25.
    assert not three['hole_filling']['enabled']
    with np.load(stage/'step1.3/pose_coverage.npz') as z:pool={p:z[p]for p in z.files}
    with np.load(stage/'step1.2/coverage_plan.npz') as z:plan={p:z[p]for p in z.files}
    for arrays in [pool,plan]:
        np.testing.assert_array_equal(arrays['vertices_object'],mesh.vertices)
        np.testing.assert_array_equal(arrays['faces'],mesh.faces)
        np.testing.assert_array_equal(arrays['source_face_ids'],np.arange(len(mesh.faces)))
    N=three['automatic_candidate_count'];selected,trace=greedy_cover(pool['candidate_face_membership'][:N],mesh.area_faces)
    np.testing.assert_array_equal(selected,plan['candidate_indices'])
    np.testing.assert_array_equal(plan['working_area_face_membership'],pool['candidate_face_membership'][selected])
    counts=plan['working_area_face_membership'].sum(0)
    np.testing.assert_array_equal(counts,plan['coverage_count']);assert (counts>0).all()
    assert two['coverage_fraction']==1. and two['uncovered_source_face_ids']==[]
    summaries=[]
    for i,(row,index) in enumerate(zip(two['working_areas'],selected)):
        for p,v in trace[i].items():assert row[p]==v,(i,p)
        mask=plan['working_area_face_membership'][i]
        np.testing.assert_array_equal(np.flatnonzero(mask),row['face_ids'])
        T=plan['screening_world_from_object'][i]
        np.testing.assert_allclose(T,row['screening_world_from_object'],atol=0,rtol=0)
        np.testing.assert_allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-12,rtol=0)
        summary=check_region(mesh,mask,T,plan['reference_head_object_m'][i],ws)
        summary['id']=row['id'];summaries.append(summary)
    demos=[]
    for i,row in enumerate(three['working_areas']):
        index=row['candidate_index'];T=np.eye(4);T[:3,:3]=pool['candidate_rotation_world_object'][index];T[:3,3]=pool['candidate_translation_world_object_m'][index]
        summary=check_region(mesh,pool['working_area_face_membership'][i],T,pool['candidate_reference_head_object_m'][index],ws)
        summary['id']=row['id'];demos.append(summary)
        scene=three['visualization']['scenes'][i]
        tip=np.asarray(scene['illustrated_tip_world_m']);root=np.asarray(scene['head_front_world_m'])
        np.testing.assert_allclose(root,ws['reference_head_front_world_m'],atol=1e-12,rtol=0)
        assert abs(np.linalg.norm(tip-root)-ws['needle_length_m'])<1e-12
    for report,folder in [(three,stage/'step1.3'),(two,stage/'step1.2')]:
        with Image.open(folder/'working_areas_overview.png') as image:
            assert list(image.size)==report['visualization']['overview_size']
    common=dict(status='passed',geometry_model=ws['model'],needle_length_m=ws['needle_length_m'],
        half_angle_deg=ws['half_angle_deg'],all_inside_source_faces_selected=True,
        visibility_filter_applied=False,normal_filter_applied=False,hole_filling_applied=False,
        native_robot_or_fixture_certified=False,validator_sha256=sha256(__file__))
    write_json(stage/'step1.2/validation.json',dict(common,working_area_count=len(summaries),source_face_count=len(mesh.faces),
        covered_face_count=int((counts>0).sum()),surface_coverage_fraction=1.,greedy_selection_replayed=True,
        working_areas=summaries,report_sha256=sha256(stage/'step1.2/coverage_plan.json')))
    write_json(stage/'step1.3/pose_coverage_validation.json',dict(common,working_areas=demos,report_sha256=sha256(stage/'step1.3/pose_coverage.json')))
    write_json(stage/'step1.3/visualization_validation.json',dict(common,rendered_needle_length_verified=True,
        sphere_center_equals_head=True,spherical_sector_radius_m=ws['needle_length_m'],onscreen_text=False))
    print('INDEPENDENT SECTOR PASS',len(summaries),'regions;',len(mesh.faces),'faces; no inside face omitted',flush=True)
    return dict(common,working_area_count=len(summaries),candidate_count=three['candidate_count'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--object',default='B');p.add_argument('--output-root',type=Path,default=DEFAULT_OUTPUT)
    a=p.parse_args();validate(a.output_root,a.object)
