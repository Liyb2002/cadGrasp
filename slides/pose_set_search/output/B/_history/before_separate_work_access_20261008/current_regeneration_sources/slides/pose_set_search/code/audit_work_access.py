"""Continuous saved work-volume checks plus oblique rays on exported meshes."""
import argparse
import hashlib
import json
import time
from common import *
from model import Model
from case_sets import CASES,case_directory
from mesh_media import load_layout


def audit(model,out):
    geometry=json.loads((out/'mesh_geometry.json').read_text())
    layout=load_layout(out/'sampled_layout.npz')
    mesh=C.trimesh.load(out/'support.obj',process=False,force='mesh')
    raycaster=RayMeshIntersector(mesh)
    bary=np.array([[i,j,9-i-j] for i in range(1,8) for j in range(1,9-i)],float)/9
    rows=[]
    for k in layout.active:
        work=model.work_rays[k];q=layout.placements[k]
        origins=np.einsum('av,tvc->tac',bary,work.triangles).reshape(-1,3)
        face_ids=np.repeat(work.face_ids,len(bary));normal=np.repeat(work.normals,len(bary),axis=0)
        variants=[]
        for n in work.normals:
            frame=tangent_frame(n);directions=[n]
            for degrees in [work.half_angle_deg/2,work.half_angle_deg]:
                angle=np.radians(degrees)
                directions.extend(np.cos(angle)*n+np.sin(angle)*(np.cos(a)*frame[:,0]+np.sin(a)*frame[:,1])
                                  for a in np.arange(16)*2*np.pi/16)
            variants.append(directions)
        directions=np.repeat(np.asarray(variants),len(bary),axis=0)
        origins=origins+1e-6*normal
        origins=np.repeat(C.transform_points(origins,q),directions.shape[1],axis=0)
        directions=(directions.reshape(-1,3) @ q[:3,:3].T)
        locations,ray_ids,hit_faces=raycaster.intersects_location(origins,directions,multiple_hits=False)
        verified=[]
        if len(ray_ids):
            weights=C.trimesh.triangles.points_to_barycentric(mesh.triangles[hit_faces],locations)
            displacement=locations-origins[ray_ids]
            distances=np.einsum('ij,ij->i',displacement,directions[ray_ids])
            perpendicular=np.linalg.norm(displacement-distances[:,None]*directions[ray_ids],axis=1)
            valid=(weights.min(axis=1)>=-1e-6)&(distances>1e-8)&(perpendicular<1e-7)
            for j in np.flatnonzero(valid)[:8]:
                verified.append(dict(work_face=int(face_ids[ray_ids[j]//33]),
                    origin=origins[ray_ids[j]].tolist(),direction=directions[ray_ids[j]].tolist(),
                    hit=locations[j].tolist(),support_face=int(hit_faces[j]),distance_m=float(distances[j])))
            hit_count=int(valid.sum())
        else:hit_count=0
        continuous=geometry['continuous_work_access_checks'][k if tuple(layout.active)==tuple(range(len(layout.active))) else list(layout.active).index(k)]
        assert continuous['pose']==model.poses[k] and continuous['cap_is_beyond_entire_fitted_seed']
        assert continuous['continuous_outer_cone_overlap_m3']<=TOL
        rows.append(dict(pose=model.poses[k],ray_count=len(origins),verified_blocked_rays=hit_count,
            passed=hit_count==0,continuous_volume_check=continuous,counterexamples=verified))
    return dict(method=out.name,passed=all(row['passed'] for row in rows),poses=rows,
        support_obj_sha256=hashlib.sha256((out/'support.obj').read_bytes()).hexdigest(),
        sampled_layout_sha256=hashlib.sha256((out/'sampled_layout.npz').read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=HERE/'output/B')
    args=parser.parse_args();began=time.monotonic();results=[]
    for case,numbers in CASES.items():
        model=Model([f'pose_{n}' for n in numbers]);folder=case_directory(args.root,case)
        for method in ['whole','incremental']:
            row=audit(model,folder/method);row['case']=folder.name;results.append(row)
            print('WORK CONE AUDIT',folder.name,method,row['passed'],flush=True)
            C.save(args.root/'work_access_audit.json',dict(complete=False,results=results))
    report=dict(complete=True,passed=all(row['passed'] for row in results),
        run_count=len(results),pose_occurrences=sum(len(row['poses']) for row in results),
        rays_tested=sum(p['ray_count'] for row in results for p in row['poses']),
        blocked_rays=sum(p['verified_blocked_rays'] for row in results for p in row['poses']),
        angles_deg=[0,15,30],azimuth_count_per_oblique_ring=16,
        interior_barycentric_samples_per_work_triangle=28,ray_length='unbounded',
        ray_origin_outward_offset_m=1e-6,
        continuous_check='intersection with complete circumscribed triangle-plus-cone volumes; caps beyond entire seed',
        tolerance_m3=TOL,sampled_no_hit_alone_is_not_a_continuous_certificate=True,
        force_pressure_acceptance=False,results=results,seconds=time.monotonic()-began)
    C.save(args.root/'work_access_audit.json',report)
    (args.root/'work_access_audit.md').write_text('# 完整工作禁区检查\n\n'
        f'{len(results)} 个导出支撑，{report["pose_occurrences"]} 个 pose 使用：'
        f'{report["rays_tested"]:,} 条正向及倾斜射线，阻挡 {report["blocked_rays"]} 条。\n\n'
        '每个原始工作三角形取28个内点，检查法向以及15°、30°各16个方位；射线没有长度上限。'
        '此外，所有实际支撑与完整工作禁区的连续 Boolean 交叠均在原始几何容差内，截断面位于全部潜在支撑之外。'
        '射线抽样本身不是连续区域证明。本检查不进行完整受力/压力证书验收。\n')
    assert report['passed'],'Exported support blocks original work access'


if __name__=='__main__':main()
