"""Check exported whole initialization, unbounded work rays and preserved inputs."""
import _bootstrap
import argparse,time
from PIL import Image
from trimesh.ray.ray_pyembree import RayMeshIntersector
from co_common import *
from run_all import saved_groups,output_root
from work_access import RegisteredWorkVolumes,tangent_frame


def audit_case(name,group):
    base=output_root(name)/group['id']/'step3'
    report=I.check_report(base/'step3.1/data/report.json')
    visual=I.check_report(base/'step3.2/data/report.json')
    I.check_report(base/'data/report.json')
    assert report['schema']=='whole_registered_work_cones_v1'
    assert report['poses']==group['poses'] and report['reference_pose']==group['poses'][0]
    assert report['registered_objects_coincide'] and not report['exit_checked']
    assert not report['ground_constructed'] and not report['full_fixture_accepted']
    assert visual['presentation_only'] and not visual['geometry_changed'] and not visual['mechanics_rerun']
    assert visual['view_count']==8 and not visual['onscreen_text']
    assert visual['support_sha256']==I.sha256(base/'step3.1/wrapped_support.obj')
    assert visual['source_initialization_sha256']==I.sha256(base/'step3.1/data/report.json')
    for stage in ['step3.1','step3.2']:
        with Image.open(base/stage/'overview.png') as picture:picture.verify()
    states={pose:state(name,pose) for pose in group['poses']}
    body=trimesh.load(base/'step3.1/registered_object.obj',force='mesh',process=False)
    support=trimesh.load(base/'step3.1/wrapped_support.obj',force='mesh',process=False)
    reference=np.asarray(report['T_fixture_to_reference_world'])
    for row in report['states']:
        task,T,mesh=states[row['pose']]
        np.testing.assert_allclose(body.vertices,mesh.vertices,atol=1e-10,rtol=0)
        np.testing.assert_array_equal(body.faces,mesh.faces)
        np.testing.assert_allclose(transform_points(task.domain.mesh.vertices,np.asarray(row['T_native_world_to_reference_world'])),
            transform_points(body.vertices,reference),atol=1e-10,rtol=0)
        np.testing.assert_allclose(transform_points(body.vertices,np.asarray(row['T_fixture_to_world'])),task.domain.mesh.vertices,atol=1e-10,rtol=0)
    access=RegisteredWorkVolumes(body,states)
    np.testing.assert_array_equal(access.ids,report['work_access_definition']['face_ids'])
    with np.load(base/'step3.1/data/contacts.npz') as contacts:
        assert np.isin(contacts['source_faces'],contacts['allowed_faces']).all()
        assert not np.isin(contacts['source_faces'],access.ids).any()
        contact_area=float(trimesh.triangles.area(contacts['triangles_mesh_m']).sum())
        np.testing.assert_allclose(contact_area,report['actual_contact_area_m2'],rtol=1e-12,atol=1e-15)
    thickness=report['maximum_wrap_vertex_displacement_m']
    bounds=np.array([body.vertices.min(0)-thickness,body.vertices.max(0)+thickness])
    corners=np.array([[x,y,z] for x in bounds[:,0] for y in bounds[:,1] for z in bounds[:,2]])
    forbidden,lengths=access.solid_for_points(corners)
    actual=S.solid(support)
    work_overlap=material_volume(actual^forbidden)
    body_overlap=material_volume(actual^S.solid(body))
    tolerance=report['continuous_geometry_tolerance_m3']
    assert max(work_overlap,body_overlap)<=tolerance
    assert report['finite_caps_beyond_entire_uncut_seed']
    np.testing.assert_allclose(lengths,report['finite_cone_lengths_m'],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(abs(support.volume)*1e6,report['material_volume_cm3'],rtol=1e-8,atol=1e-6)
    caster=RayMeshIntersector(support)
    bary=np.array([[i,j,9-i-j] for i in range(1,8) for j in range(1,9-i)],float)/9
    tested=blocked=0;examples=[]
    # All objects coincide. Test each unique original face at its largest saved
    # angle; smaller cones on the same face are nested inside this domain.
    for family in access.families:
        origins=np.einsum('av,tvc->tac',bary,family.triangles).reshape(-1,3)
        face_ids=np.repeat(family.face_ids,len(bary))
        normals=np.repeat(family.normals,len(bary),axis=0)
        variants=[]
        for normal in family.normals:
            frame=tangent_frame(normal);directions=[normal]
            for degrees in [family.half_angle_deg/2,family.half_angle_deg]:
                angle=np.radians(degrees)
                directions.extend(np.cos(angle)*normal+np.sin(angle)*(np.cos(a)*frame[:,0]+np.sin(a)*frame[:,1])
                    for a in np.arange(16)*2*np.pi/16)
            variants.append(directions)
        directions=np.repeat(np.asarray(variants),len(bary),axis=0).reshape(-1,3)
        origins=np.repeat(origins+1e-6*normals,33,axis=0)
        locations,ray_ids,hit_faces=caster.intersects_location(origins,directions,multiple_hits=False)
        tested+=len(origins)
        if len(ray_ids):
            weights=trimesh.triangles.points_to_barycentric(support.triangles[hit_faces],locations)
            displacement=locations-origins[ray_ids]
            distance=np.einsum('ij,ij->i',displacement,directions[ray_ids])
            perpendicular=np.linalg.norm(displacement-distance[:,None]*directions[ray_ids],axis=1)
            valid=(weights.min(1)>=-1e-6)&(distance>1e-8)&(perpendicular<1e-7)
            blocked+=int(valid.sum())
            for j in np.flatnonzero(valid)[:8]:
                examples.append(dict(work_face=int(face_ids[ray_ids[j]//33]),origin=origins[ray_ids[j]].tolist(),
                    direction=directions[ray_ids[j]].tolist(),hit=locations[j].tolist(),
                    support_face=int(hit_faces[j]),distance_m=float(distance[j])))
    return dict(id=group['id'],poses=group['poses'],reference_pose=report['reference_pose'],passed=blocked==0,
        rays_tested=tested,unique_work_triangles=len(access.ids),blocked_rays=blocked,counterexamples=examples,
        exported_mesh_work_union_overlap_m3=work_overlap,exported_mesh_body_overlap_m3=body_overlap,
        tolerance_m3=tolerance,cap_beyond_entire_seed=True,force_status=report['initial_force_status'],
        original_load_count=sum(row['load_count'] for row in report['state_results']),
        support_sha256=I.sha256(base/'step3.1/wrapped_support.obj'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('object',nargs='?',default='B')
    args=parser.parse_args();root=output_root(args.object);began=time.monotonic();rows=[]
    manifest=json.loads((root/'data/whole_initialization_manifest.json').read_text())
    preservation={}
    for label in ['original_input_hashes','pose_set_search_source_hashes','protected_downstream_artifacts']:
        for relative,digest in manifest[label].items():
            path=root/relative if label=='protected_downstream_artifacts' else ROOT/relative
            if not path.exists() or I.sha256(path)!=digest:raise RuntimeError('Changed preserved file: '+str(path))
        preservation[label]=dict(files=len(manifest[label]),unchanged=True)
    groups=saved_groups(args.object)
    assert {(g['id'],tuple(g['poses'])) for g in groups}=={(g['id'],tuple(g['poses'])) for g in manifest['groups']}
    for group in groups:
        row=audit_case(args.object,group);rows.append(row)
        print('INITIALIZATION AUDIT',row['id'],row['passed'],row['rays_tested'],row['blocked_rays'],flush=True)
        save(root/'data/whole_initialization_audit.json',dict(complete=False,results=rows))
    report=dict(complete=True,passed=all(row['passed'] for row in rows),sets=len(rows),
        pose_instances=sum(len(row['poses']) for row in rows),rays_tested=sum(row['rays_tested'] for row in rows),
        blocked_rays=sum(row['blocked_rays'] for row in rows),preserved=preservation,
        interior_barycentric_samples_per_triangle=28,angles='normal, half saved half-angle, full saved half-angle',
        oblique_ring_azimuths=16,ray_origin_outward_offset_m=1e-6,ray_length='unbounded',
        continuous_check='exported support intersected with complete outer triangle-plus-cone union',
        sampled_no_hit_alone_is_not_a_continuous_certificate=True,
        force_pressure_acceptance=False,step4_rerun=False,results=rows,seconds=time.monotonic()-began,
        provenance=provenance([root/'data/whole_initialization_manifest.json'],[Path(__file__),HERE/'helper_func/work_access.py']))
    save(root/'data/whole_initialization_audit.json',report)
    (root/'data/whole_initialization_audit.md').write_text('# Whole 初始化检查\n\n'
        f'{len(rows)} 个导出支撑，{report["pose_instances"]} 个原 pose 使用，{report["rays_tested"]:,} 条无界工作射线；阻挡 {report["blocked_rays"]} 条。\n\n'
        '射线从每个唯一原工作三角形的28个内点出发，检查法向与半角中点/边界各16个方位；'
        '注册后重复工作面采用原始最大角度。另对导出实际网格做完整外接工作锥并集的连续交叠检查，'
        'Boolean远端盖面始终在完整未切包裹之外。射线抽样本身不是连续证明。\n\n'
        '原始物体数据、已完成pose_set_search代码以及原Step4文件的保存哈希均未改变。'
        '没有运行Step4或完整压力证书验收。\n')
    assert report['passed'],'Exported initial wrap blocks an original work ray'
    print('TOTAL',report['sets'],'sets',report['rays_tested'],'rays',report['blocked_rays'],'blocked',flush=True)


if __name__=='__main__':main()
