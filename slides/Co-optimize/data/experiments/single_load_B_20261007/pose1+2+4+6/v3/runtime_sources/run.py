"""Once-loaded Step4 candidates -> Step5 shared-layout/direction pilot.

Historical per-pose exits and translations are deliberately not inputs.
Run with .venv/bin/python slides/Co-optimize/single_load/run.py.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
from co_common import *
from exit_clearance import ExitClearance
from robot import Panda
from retention import check as retention_check
from scipy.spatial.transform import Rotation
import argparse,time

def candidates(mesh,allowed,limit):
    rng=np.random.default_rng(20261007)
    ids=allowed if len(allowed)<=1800 else rng.choice(allowed,1800,replace=False,p=mesh.area_faces[allowed]/mesh.area_faces[allowed].sum())
    points=mesh.triangles_center[ids];normals=mesh.face_normals[ids]
    delta=points[None]-points[:,None];width=np.linalg.norm(delta,axis=2)
    direction=delta/np.maximum(width[...,None],1e-12)
    valid=(width>.015)&(width<.070)
    valid &= np.einsum('ijk,ik->ij',direction,normals)<-.9
    valid &= np.einsum('ijk,jk->ij',direction,normals)>.9
    valid &= np.triu(np.ones_like(valid),1)
    a,b=np.nonzero(valid);scores=np.linalg.norm((points[a]+points[b])/2-mesh.center_mass,axis=1)
    seen=set();count=0
    for index in np.argsort(scores):
        i,j=a[index],b[index];center=(points[i]+points[j])/2;closing=direction[i,j]
        key=tuple(np.round(center/.004).astype(int))+tuple(np.round(closing/.12).astype(int))
        if key in seen:continue
        seen.add(key);count+=1
        base=np.eye(3)[np.argmin(np.abs(closing))];base-=closing*(base@closing);base/=np.linalg.norm(base)
        for roll in range(0,360,30):
            approach=Rotation.from_rotvec(closing*np.radians(roll)).apply(base)
            R=np.column_stack([np.cross(closing,approach),closing,approach]);T=np.eye(4)
            T[:3,:3]=R;T[:3,3]=center-R@np.array([0.,0.,.103])
            yield dict(hand=T,width=float(width[i,j]),contacts=points[[i,j]],face_ids=[int(ids[i]),int(ids[j])],roll_deg=roll)
        if count>=limit:return

def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v) for v in value]
    return value

def grasp_candidates(robot,mesh,allowed,states,limit):
    obstacle=S.solid(mesh);accepted=[];counts=dict(sampled=0,work=0,collision=0,ik=0)
    for item in candidates(mesh,allowed,limit=limit):
        counts['sampled']+=1
        if not set(item['face_ids']).issubset(set(allowed)):
            counts['work']+=1;continue
        if not robot.approach_clear(item['hand'],item['width'],obstacle):
            counts['collision']+=1;continue
        solutions={}
        for pose,(task,T,_) in states.items():
            world=T@item['hand'];world[:3,3]+=np.array([.45,0,.15])
            q=robot.ik(world)
            if q is None:break
            solutions[pose]=q
        if len(solutions)!=len(states):counts['ik']+=1;continue
        item['pose_ik']=solutions;accepted.append(item)
        print('STEP4 grasp',len(accepted),'sample',counts['sampled'],flush=True)
        if len(accepted)>=8:break
    return accepted,counts

def grip_core(mesh,grasp,seed,allowed):
    # Finite 5mm support under two 6mm-radius fingertip neighborhoods.
    points=grasp['contacts'];centers=mesh.triangles_center
    neighborhood=np.linalg.norm(centers[allowed,None]-points[None],axis=2)<.006
    aligned=mesh.face_normals[allowed]@mesh.face_normals[grasp['face_ids']].T>.99999
    ids=allowed[np.any(neighborhood & aligned,axis=1)]
    offsets=wrap_offsets(mesh,.005)
    parts=[S.solid(G.hull_mesh(G.head_cell(mesh,mesh.triangles[i],i,offsets))) for i in ids]
    return union(parts)^seed,ids

def run(args):
    began=time.monotonic();poses=args.poses.split(',');group='pose'+'+'.join(p.split('_')[1] for p in poses)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    states={p:state(args.object,p) for p in poses};mesh=next(iter(states.values()))[2]
    work=np.unique(np.concatenate([s[0].domain.work_ids for s in states.values()]))
    allowed=np.setdiff1d(np.arange(len(mesh.faces)),work)
    source=HERE/'output'/args.object/group/'step3/step3.3/support_with_rings.obj'
    seed=S.solid(trimesh.load(source,force='mesh',process=False))
    # The entire common fixture must remain above every task's native floor.
    for task,T,_ in states.values():
        seed=seed.trim_by_plane((T[:3,:3].T@np.array([0.,0.,1.])).tolist(),-float(T[2,3])/S.SCALE)
    robot=Panda();grasps,counts=grasp_candidates(robot,mesh,allowed,states,args.grasp_pairs)
    common=[]
    for grasp in grasps:
        proposal={**grasp,'width':grasp['width']+.010}
        if proposal['width']<=.08:
            proposal['support_contacts']=grasp['contacts']+.005*mesh.face_normals[grasp['face_ids']]
            proposal['support_inward_normals']=-mesh.face_normals[grasp['face_ids']]
            common.append(proposal)
    save(out/'step4/candidates.json',serial(dict(loading_candidates=grasps,common_candidates=common,counts=counts,work_faces=work,station_offset_m=[.45,0,.15],model=str(Path(__file__).parent/'assets/franka_emika_panda/panda.xml'),common_geometry='5mm outward support at the two grasp patches; validated against actual carved support in Step5',arm_path_collision_checked=False,retention_checked=False)))
    if not grasps:
        report=dict(status='no_grasp_candidate',accepted=False,step4=counts,step5_evaluated=0,seconds=time.monotonic()-began)
        save(out/'report.json',report);print(json.dumps(report),flush=True);return
    clearance=ExitClearance(mesh);trace=[];best=None
    # Both tools act on ONE common transform and ONE loading direction.
    for gi,grasp in enumerate(common[:args.finalists]):
        core,coreids=grip_core(mesh,grasp,seed,allowed)
        loading_pose=poses[0];T=states[loading_pose][1]
        up=T[:3,:3].T@np.array([0.,0.,1.])
        directions=[up,-grasp['hand'][:3,2]]
        n=mesh.face_normals[grasp['face_ids']]
        tangent=np.cross(n[0],n[1])
        if np.linalg.norm(tangent)>1e-8:
            tangent/=np.linalg.norm(tangent)
            inward=-(n[0]+n[1]);inward/=np.linalg.norm(inward)
            directions += [sign*tangent+weight*inward for weight in (.25,.5,1.,2.) for sign in (-1,1)]
            directions.append(inward)
        for axis in np.eye(3):
            for sign in (-1,1):directions.append(up+sign*.15*axis)
        for di,direction in enumerate(directions):
            direction=direction/np.linalg.norm(direction)
            for shift in ([np.zeros(3)] if di else [np.zeros(3),*[sign*.002*axis for axis in np.eye(3) for sign in (-1,1)]]):
                row=dict(grasp=gi,direction=direction.tolist(),shared_translation_m=np.asarray(shift).tolist())
                moved=seed.translate((np.asarray(shift)/S.SCALE).tolist())
                # Translation cannot consume required grip material.
                if material_volume(core-moved)>1e-11:
                    row['status']='grip_material_lost';trace.append(row);continue
                nominal=clearance.sweep(.5*direction,padded=False)
                if material_volume(core^nominal)>1e-11:
                    row['status']='loading_cuts_grip';trace.append(row);continue
                try:
                    permitted=allowed[mesh.face_normals[allowed]@direction<=1e-10]
                    result=clearance.construct(moved,[nominal],[clearance.sweep(.5*direction)],permitted)
                    remaining=result['remaining'];row['geometry']=result['diagnostics']
                    if material_volume(core-remaining)>1e-11:
                        row['status']='clearance_cuts_grip';trace.append(row);continue
                    if not result['diagnostics']['geometry_resolved']:
                        row['status']='geometry_unresolved';trace.append(row);continue
                    # Distinct loading and common grasps, joined by a release/regrasp.
                    loading=None
                    for li,load in enumerate(grasps):
                        blocked=False
                        for fraction in np.linspace(0,1,13):
                            hand=load['hand'].copy();hand[:3,3]+=.5*fraction*direction
                            if not robot.clear(hand,load['width'],remaining,allow_pads=False):blocked=True;break
                        if not blocked:loading=li;break
                    if loading is None:row['status']='loading_hand_collision';trace.append(row);continue
                    row['loading_grasp']=loading
                    boundary=S.unpack(remaining)
                    row['loading_path']=robot.loading_path(grasps[loading]['hand'],grasps[loading]['width'],mesh,boundary,T,direction)
                    if not row['loading_path']['passed']:
                        row['status']=row['loading_path']['reason'];trace.append(row);continue
                    assembly=S.unpack(remaining+S.solid(mesh))
                    if not robot.approach_clear(grasp['hand'],grasp['width'],remaining+S.solid(mesh)):
                        row['status']='common_hand_collision';trace.append(row);continue
                    # Both jaws must actually touch retained fixture material.
                    pads=robot.hand_solids(grasp['hand'],max(.015,grasp['width']-.001))
                    touching={body for body,pad,solid in pads if pad and material_volume(solid^remaining)>1e-12}
                    if touching != {'left_finger','right_finger'}:
                        row['status']='common_contact_missing';trace.append(row);continue
                    points,distances,faces=trimesh.proximity.closest_point(boundary,grasp['support_contacts'])
                    inward=-boundary.face_normals[faces]
                    if max(distances)>.003 or np.any(np.einsum('ij,ij->i',inward,grasp['support_inward_normals'])<.9):
                        row['status']='common_contact_normal_mismatch';trace.append(row);continue
                    row['actual_common_contacts']=points.tolist()
                    triangles,sources=contact_boundary(mesh,boundary,allowed)
                    results=[]
                    for pose,(task,world,_) in states.items():
                        supply7=supply(task,world,triangles,sources);mask,info=J.classify(supply7,task.targets)
                        placed=boundary.copy();placed.apply_transform(world)
                        results.append(dict(pose=pose,covered=int(mask.sum()),loads=len(mask),force_passed=bool(mask.all()),floor_clear=bool(len(placed.vertices) and placed.vertices[:,2].min()>=-1e-9),work_clear=WORK.check(placed,task)['passed']))
                    row.update(status='evaluated',poses=results,material_cm3=material_volume(remaining)*1e6,components=len(remaining.decompose()))
                    rotations=[world[:3,:3] for task,world,_ in states.values()]
                    row['retention']=retention_check(mesh,boundary,triangles,sources,points,inward,rotations)
                    row['carry_path']=robot.carry_path(grasp['hand'],grasp['width'],assembly,[s[1] for s in states.values()])
                    score=sum(r['covered'] for r in results)
                    if best is None or score>best[0]:
                        best=(score,row,remaining)
                        D.export_exact_obj(boundary,out/'step5/best_support.obj')
                        np.savez_compressed(out/'step5/best_contacts.npz',triangles=triangles,sources=sources,grip_source_faces=coreids)
                except (RuntimeError,ValueError,np.linalg.LinAlgError) as error:
                    row.update(status='unresolved',error=str(error))
                trace.append(row);save(out/'step5/trace.json',serial(trace))
                print('STEP5',gi,di,row['status'],row.get('poses',[]),flush=True)
    passed=best is not None and all(r['force_passed'] and r['floor_clear'] and r['work_clear'] for r in best[1]['poses']) and best[1]['components']==1 and best[1]['loading_path']['passed'] and best[1]['retention']['passed'] and best[1]['carry_path']['passed']
    report=dict(status='pilot_pass' if passed else 'pilot_fail',accepted=False,pilot_checks_passed=passed,step4=counts,step4_candidates=len(grasps),step5_evaluated=sum(r['status']=='evaluated' for r in trace),rejections={status:sum(r['status']==status for r in trace) for status in sorted({r['status'] for r in trace})},best=None if best is None else best[1],seconds=time.monotonic()-began,remaining_requirements=['continuous collision certificate and robot trajectory timing','retention under acceleration and real mass/friction/force limits','fixture strength'],scope='Original 32768 loads per pose; shared object-fixture transform; one loading path; sampled hand clearance and arm transport; coupled quasistatic retention. No full fixture acceptance.')
    save(out/'step5/trace.json',serial(trace));save(out/'report.json',serial(report));print(json.dumps(serial(report)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--object',default='B');parser.add_argument('--poses',default='pose_1,pose_2,pose_4,pose_6');parser.add_argument('--grasp-pairs',type=int,default=100);parser.add_argument('--finalists',type=int,default=8);parser.add_argument('--out',default=str(HERE/'data/experiments/single_load_B_20261007/pose1+2+4+6'))
    run(parser.parse_args())
