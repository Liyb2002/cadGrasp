"""Optional strict-boundary and tool-ray diagnostics, NOT baseline acceptance.

This does not change fixture geometry or its existing force/withdrawal reports.
The baseline excludes working faces and permits adjacent shared edges/vertices;
working_surface.py implements that actual rule. Blocked rays are counterexamples
only to the EXTRA ray-clearance condition, which this baseline does not require.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step3_scheculer.pair_tasks import read_task
from step2_local_support import support_policy as POLICY
from step4_connect_support.run_reseated import groups
from step4_connect_support.surface_check import winding_number

SURFACE_TOLERANCE=1e-9


def triangle_overlaps(a,others,tolerance=SURFACE_TOLERANCE):
    """Triangle SAT including in-plane edge axes for coplanar contacts."""
    a=np.asarray(a,np.longdouble);b=np.asarray(others,np.longdouble)
    ae=np.roll(a,-1,axis=0)-a;be=np.roll(b,-1,axis=1)-b
    an=np.cross(ae[0],ae[1]);bn=np.cross(be[:,0],be[:,1])
    axes=np.concatenate([np.broadcast_to(an,(len(b),1,3)),bn[:,None],
        np.cross(ae[None,:,None,:],be[:,None,:,:]).reshape(len(b),9,3),
        np.broadcast_to(np.cross(an,ae),(len(b),3,3)),np.cross(bn[:,None,:],be)],axis=1)
    lengths=np.sqrt((axes*axes).sum(2));valid=lengths>0
    axes=np.divide(axes,lengths[:,:,None],out=np.zeros_like(axes),where=valid[:,:,None])
    # Subtract a shared origin to avoid large absolute projection cancellation.
    pa=np.einsum('nkj,vj->nkv',axes,a-a[0])
    pb=np.einsum('nkj,nvj->nkv',axes,b-a[0])
    separated=((pa.max(2)<pb.min(2)-tolerance)|(pb.max(2)<pa.min(2)-tolerance))&valid
    return ~separated.any(1)


def check_surface(body,task):
    tree=body.triangles_tree;triangles=body.triangles;rows=[];pairs=0
    for face in task.domain.work_ids:
        tri=task.domain.mesh.triangles[face]
        ids=np.array(list(tree.intersection(np.r_[tri.min(0)-SURFACE_TOLERANCE,
                                                  tri.max(0)+SURFACE_TOLERANCE])),int)
        pairs+=len(ids)
        if not len(ids):continue
        collided=ids[triangle_overlaps(tri,triangles[ids])]
        if len(collided):rows.append(dict(work_face_id=int(face),support_triangle_ids=collided.tolist()))
    # A complete triangle could be enclosed without crossing the boundary.
    centers=task.domain.mesh.triangles_center[task.domain.work_ids]
    inside=RayMeshIntersector(body).contains_points(centers)
    return dict(working_triangle_count=len(task.domain.work_ids),triangle_pairs_checked=pairs,
        boundary_intersection_face_count=len(rows),boundary_intersections=rows,
        centroid_inside_face_ids=task.domain.work_ids[inside].tolist(),
        passed=not rows and not inside.any(),tolerance_m=SURFACE_TOLERANCE,
        method='Every work triangle vs actual support triangles using long-double SAT; coplanar axes included; solid containment checked separately')


def exact_ray_hits(mesh,origin,direction):
    """Independent long-double Moller-Trumbore hits on the exported triangles."""
    tri=np.asarray(mesh.triangles,np.longdouble);o=np.asarray(origin,np.longdouble);u=np.asarray(direction,np.longdouble)
    e1,e2=tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]
    h=np.cross(u,e2);det=np.einsum('ij,ij->i',e1,h)
    valid=np.abs(det)>1e-18*np.sqrt((e1*e1).sum(1)*(e2*e2).sum(1))
    inv=np.divide(1,det,out=np.zeros_like(det),where=valid)
    s=o-tri[:,0];b1=np.einsum('ij,ij->i',s,h)*inv
    q=np.cross(s,e1);b2=(q@u)*inv;t=np.einsum('ij,ij->i',e2,q)*inv
    good=valid&(b1>=-1e-12)&(b2>=-1e-12)&(b1+b2<=1+1e-12)&(t>1e-9)
    ids=np.flatnonzero(good);order=np.argsort(t[ids]);ids=ids[order]
    return [(int(i),float(t[i]),[float(1-b1[i]-b2[i]),float(b1[i]),float(b2[i])]) for i in ids]


def witness(body,task,samples,blocked,points,directions,normals):
    for index in np.flatnonzero(blocked)[:64]:
        pt=points[index];u=directions[index]
        # Use the original Step1 self-visibility predicate, not a convex proxy.
        source=pt+task.domain.ray_offset*normals[index]
        if task.domain.mesh.ray.intersects_any(source[None],u[None])[0]:continue
        hits=exact_ray_hits(body,pt,u)
        for hit_index,(face,distance,bary) in enumerate(hits[:-1]):
            following=next((r for r in hits[hit_index+1:] if r[1]>distance+2e-6),None)
            if following is None:continue
            middle=(distance+following[1])/2;inside=pt+middle*u
            winding=winding_number(body,inside)
            if abs(winding)<.99:continue
            theta=float(np.degrees(np.arccos(np.clip(u@normals[index],-1,1))))
            if theta>task.domain.data['load']['cone_half_deg']+1e-8:continue
            # Re-evaluate the saved barycentric/angle/magnitude tuple independently.
            params=np.asarray(samples['parameters'][index])
            replay=task.domain.evaluate(samples['work_face_index'][index],*params)
            np.testing.assert_allclose(replay['q_m'],pt,atol=1e-12,rtol=0)
            np.testing.assert_allclose(-replay['force_push_mg']/np.linalg.norm(replay['force_push_mg']),u,atol=1e-12)
            assert bool(replay['tool_reachable'])
            return dict(original_sample_index=int(index),work_face_id=int(task.domain.work_ids[samples['work_face_index'][index]]),
                work_point_world_m=pt.tolist(),outward_direction_world=u.tolist(),
                angle_from_work_normal_deg=theta,cone_half_angle_deg=task.domain.data['load']['cone_half_deg'],
                original_step1_reachable=True,ray_hit_distance_m=distance,
                hit_point_world_m=(pt+distance*u).tolist(),support_triangle_index=face,
                support_triangle_barycentric=bary,interior_probe_world_m=inside.tolist(),
                interior_probe_winding_number=float(winding),interior_interval_length_m=following[1]-distance,
                independent_long_double_ray_triangle_verified=True)
    return None


def audit(group):
    out=group/'step4';data=out/'data';report_path=data/'report.json'
    report=I.check_report(report_path);mesh=trimesh.load(out/'shape.obj',force='mesh',process=False)
    paths=[report_path,out/'shape.obj'];rows=[];arrays={}
    for k,pose in enumerate(report['poses']):
        folder=group/'step3_scheculer/independent_poses_floor2mm'/pose
        task=read_task(group.parent.name,pose,folder=folder/'step_1_needs')
        sample_path=folder/'step_1_needs/samples.json';samples=json.loads(sample_path.read_text());paths+=task.inputs
        b=np.asarray(report['placement']['bases'][k]);o=np.asarray(report['placement']['offsets'][k])
        body=trimesh.Trimesh((mesh.vertices-o)@b.T,mesh.faces,process=False)
        assert body.is_watertight and body.is_winding_consistent
        points=np.asarray(samples['pt_m']);push=np.asarray(samples['force_push_mg'])
        directions=-push/np.linalg.norm(push,axis=1)[:,None]
        work_indices=np.asarray(samples['work_face_index'],int)
        normals=task.domain.mesh.face_normals[task.domain.work_ids[work_indices]]
        # Every original direction already passed the source object's own visibility
        # test. Count intersections with the full installed support, idle parts included.
        ray=RayMeshIntersector(body)
        blocked=ray.intersects_any(points+directions*1e-9,directions)
        offset_blocked=ray.intersects_any(points+normals*task.domain.ray_offset,directions)
        check=check_surface(body,task)
        w=witness(body,task,samples,blocked,points,directions,normals) if blocked.any() else None
        if blocked.any() and w is None:raise RuntimeError(f'No independently verified interior ray witness: {group.name}/{pose}')
        contact_path=folder/'step3_scheculer'/f'final_contacts_{pose}.npz';paths.append(contact_path)
        selected=I.read_contacts(contact_path)
        active_overlap=[dict(head=c['candidate_id'],work_face_ids=np.intersect1d(c['source_faces'],task.domain.work_ids).tolist()) for c in selected]
        arrays[pose+'_blocked_original_rays']=blocked
        arrays[pose+'_blocked_offset_rays']=offset_blocked
        arrays[pose+'_surface_boundary_face_ids']=np.array([r['work_face_id'] for r in check['boundary_intersections']],int)
        row=dict(pose=pose,original_sample_count=len(points),blocked_original_ray_count=int(blocked.sum()),
            blocked_step1_offset_ray_count=int(offset_blocked.sum()),
            samples_with_offset_classification_difference=int(np.count_nonzero(blocked!=offset_blocked)),
            affected_work_face_count=int(len(np.unique(work_indices[blocked]))),
            surface_check=check,active_heads_work_face_overlap=active_overlap,
            access_status='blocked_original_processing_ray' if w else 'no_sampled_hit_continuous_access_unverified',
            original_access_counterexample=w)
        rows.append(row)
        print('WORK ACCESS',group.name,pose,'surface',check['passed'],'blocked',int(blocked.sum()),'/',len(points),flush=True)
    np.savez_compressed(data/'work_access_check.npz',**arrays)
    result=dict(complete=True,object=group.parent.name,poses=report['poses'],checks=rows,
        acceptance_applicable=False,
        diagnostic_scope='Extra strict-boundary and tool-ray conditions; not the Step3-compatible working-face rule',
        current_surface_rule_result='working_surface_check.json',
        previous_inference_that_blocked_rays_invalidate_this_baseline_withdrawn=True,
        all_working_surfaces_disjoint=all(r['surface_check']['passed'] for r in rows),
        proven_access_obstruction=any(r['original_access_counterexample'] for r in rows),
        all_pose_continuous_access_certified=False,scope='Actual saved support in each independent installed placement, including inactive heads and connections',
        original_sample_counts_are_not_work_area_fractions=True,force_or_geometry_inputs_changed=False,
        whole_support_access_was_enforced_in_construction=False,
        source_policy_enforces_process_access=POLICY.ENFORCE_PROCESS_ACCESS,
        existing_fixture_passed_field_scope='Original mechanics/withdrawal and current working-face check; tool-ray-volume clearance is not required',
        finite_tool_radius_checked=False,
        provenance=dict(inputs=I.hashes(paths),code=I.hashes([Path(__file__),Path(POLICY.__file__),
            Path(__file__).with_name('surface_check.py')])),
        artifacts={'work_access_check.npz':I.sha256(data/'work_access_check.npz')})
    I.save(data/'work_access_check.json',result)


def batch(name,jobs):
    def worker(group):
        with (group/'step4/data/work_access_check.log').open('w') as log:
            p=subprocess.run([sys.executable,str(Path(__file__).resolve()),group.name,'--object',name],stdout=log,stderr=subprocess.STDOUT)
        print('ACCESS AUDIT',group.name,p.returncode,flush=True)
        if p.returncode:raise RuntimeError(f'Audit failed for {group.name}; inspect per-group log')
    with ThreadPoolExecutor(max_workers=jobs) as pool:list(pool.map(worker,groups(name)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('group',nargs='?',default='pose5+7')
    p.add_argument('--object',default='B');p.add_argument('--all',action='store_true');p.add_argument('--jobs',type=int,default=2)
    args=p.parse_args()
    if args.all:batch(args.object,args.jobs)
    else:audit(I.OUTPUTS/args.object/args.group)
