"""Diagnose and expand stock-Franka Step4; retain every actual work surface."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
from co_common import *
from robot import Panda
from scipy.spatial.transform import Rotation
import argparse,time,hashlib,shutil
from collections import Counter

def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v) for v in value]
    return value

def sample_pairs(mesh,allowed,minimum,maximum,ranking,limit):
    ids=allowed;points=mesh.triangles_center[ids];normals=mesh.face_normals[ids]
    work=mesh.submesh([np.setdiff1d(np.arange(len(mesh.faces)),allowed)],append=True)
    margin=trimesh.proximity.closest_point(work,points)[1] if len(work.faces) else np.full(len(points),1.)
    delta=points[None]-points[:,None];width=np.linalg.norm(delta,axis=2);closing=delta/np.maximum(width[...,None],1e-12)
    valid=(width>minimum)&(width<maximum)&np.triu(np.ones_like(width,dtype=bool),1)
    valid&=np.einsum('ijk,ik->ij',closing,normals)<-.90
    valid&=np.einsum('ijk,jk->ij',closing,normals)>.90
    a,b=np.nonzero(valid);center=(points[a]+points[b])/2
    distance=np.linalg.norm(center-mesh.center_mass,axis=1)
    clearance=np.minimum(margin[a],margin[b])
    # Prioritize work clearance; include width so both thin and broad regions
    # are represented by the exact pair library, with spatial deduplication.
    score=distance if ranking=='com' else -clearance+.05*distance
    seen=set();pairs=[]
    for index in np.argsort(score,kind='stable'):
        i,j=a[index],b[index];direction=closing[i,j]
        key=tuple(np.round(center[index]/.003).astype(int))+tuple(np.round(direction/.12).astype(int))
        if key in seen:continue
        seen.add(key)
        pairs.append(dict(contacts=points[[i,j]],face_ids=[int(ids[i]),int(ids[j])],closing=direction,width=float(width[i,j]),work_margin_m=float(clearance[index]),com_distance_m=float(distance[index])))
        if len(pairs)>=limit:break
    return pairs,dict(valid_pairs=len(a),distinct_pairs=len(pairs),nonwork_faces=len(allowed),work_area_fraction=float(mesh.area_faces[np.setdiff1d(np.arange(len(mesh.faces)),allowed)].sum()/mesh.area),nonwork_margin_quantiles_m=np.quantile(margin,[0,.1,.5,.9,1]).tolist())

def hand_candidates(pair,roll_step,depths):
    closing=pair['closing'];base=np.eye(3)[np.argmin(np.abs(closing))]
    base-=closing*(base@closing);base/=np.linalg.norm(base)
    for roll in np.arange(0,360,roll_step):
        approach=Rotation.from_rotvec(closing*np.radians(roll)).apply(base)
        R=np.column_stack([np.cross(closing,approach),closing,approach])
        for depth in depths:
            T=np.eye(4);T[:3,:3]=R;T[:3,3]=pair['contacts'].mean(0)-depth*approach
            yield dict(**pair,hand=T,roll_deg=float(roll),depth_m=float(depth))

def run(args):
    began=time.monotonic();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'report.json').exists():raise RuntimeError('Choose a fresh experiment directory')
    snapshots=out/'runtime_sources';snapshots.mkdir(exist_ok=True)
    paths=[Path(__file__),Path(__file__).with_name('robot.py'),Path(__file__).with_name('run.py'),HERE/'helper_func/co_common.py',HERE/'helper_func/current_task.py']
    for path in paths:shutil.copy2(path,snapshots/path.name)
    save(snapshots/'manifest.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    save(out/'arguments.json',vars(args))
    states={f'pose_{i}':state('B',f'pose_{i}') for i in [1,2,4,6]};mesh=next(iter(states.values()))[2]
    ids=np.unique(np.concatenate([s[0].domain.work_ids for s in states.values()]));allowed=np.setdiff1d(np.arange(len(mesh.faces)),ids)
    inputs={p for task,_,_ in states.values() for p in task.inputs}
    inputs.add(Path(__file__).parent/'assets/franka_emika_panda/panda.xml')
    save(out/'input_manifest.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    pairs,domain=sample_pairs(mesh,allowed,args.minimum,args.maximum,args.ranking,args.pairs)
    work=mesh.submesh([ids],append=True);obstacle=S.solid(mesh);robot=Panda()
    counts=Counter();collisions=Counter();stage_counts=Counter();accepted=[];trace=[];near=[]
    for pi,pair in enumerate(pairs):
        for item in hand_candidates(pair,args.roll_step,args.depths):
            counts['sampled']+=1
            row=dict(index=counts['sampled']-1,pair=pi,width_m=item['width'],depth_m=item['depth_m'],roll_deg=item['roll_deg'],work_margin_m=item['work_margin_m'],hand=item['hand'],face_ids=item['face_ids'],contacts=item['contacts'])
            geometry=robot.approach_report(item['hand'],item['width'],obstacle)
            if not geometry['passed']:
                row.update(status='collision',diagnostic=geometry);counts['collision']+=1;stage_counts[geometry['stage']]+=1
                for hit in geometry['hits']:collisions[hit['body']+('_pad' if hit['pad'] else '_shell')]+=1
                near.append((sum(h['overlap_mm3'] for h in geometry['hits']),row))
            elif not robot.work_contact_clear(item['hand'],item['width'],work):
                row['status']='work';counts['work']+=1
            else:
                contact=robot.object_contacts(item['hand'],item['width'],obstacle)
                if not contact['passed']:row.update(status='no_two_jaw_contact',contact=contact);counts['no_two_jaw_contact']+=1
                else:
                    # Loading only needs one reachable pose; common transport
                    # retains a separate all-poses endpoint-IK label.
                    solutions={};collision_free={};arm_reports={}
                    for pose,(task,T,_) in states.items():
                        world=T.copy();world[:3,3]+=[.45,0,.15]
                        q=robot.ik(world@item['hand'])
                        if q is not None:
                            solutions[pose]=q
                            moved=mesh.copy();moved.apply_transform(world)
                            arm_reports[pose]=robot.arm_report(q,item['width'],S.solid(moved))
                            collision_free[pose]=arm_reports[pose]['passed']
                    usable=[p for p in solutions if collision_free[p]]
                    row.update(pose_ik=solutions,arm_collision_free=collision_free,arm_reports=arm_reports,usable_loading_poses=usable,contact=contact)
                    if not usable:row['status']='arm';counts['arm']+=1
                    else:
                        row['status']='loading_candidate';counts['loading_candidate']+=1
                        row['all_pose_arm_endpoints_passed']=len(usable)==len(states)
                        accepted.append(row)
                        print('CANDIDATE',len(accepted),'sample',counts['sampled'],'pair',pi,'gap_mm',round(item['width']*1000,2),'poses',usable,flush=True)
            trace.append(row)
            if len(accepted)>=args.keep:break
        if (pi+1)%5==0:
            save(out/'progress.json',dict(pairs_done=pi+1,counts=dict(counts),seconds=time.monotonic()-began));print('PROGRESS',pi+1,dict(counts),flush=True)
        if len(accepted)>=args.keep:break
    near=sorted(near,key=lambda r:r[0])[:12]
    report=dict(object='B',poses=list(states),counts=dict(counts),collision_bodies=dict(collisions),collision_stages=dict(stage_counts),domain=domain,loading_candidates=len(accepted),common_grasp_verified=False,step4_fully_passed=False,seconds=time.monotonic()-began,scope='Stock Panda; exact hand/object solids, full pad/work contact exclusion, both-jaw pad overlap, native-task arm IK/self/object/floor checks. Loading fixture path and assembly common grasp require Step5 geometry; no dynamics acceptance.')
    save(out/'report.json',report);save(out/'candidates.json',serial(accepted));save(out/'trace.json',serial(trace));save(out/'near_collisions.json',serial([row for score,row in near]))
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--pairs',type=int,default=400);p.add_argument('--minimum',type=float,default=.004);p.add_argument('--maximum',type=float,default=.080);p.add_argument('--roll-step',type=float,default=15);p.add_argument('--depths',type=float,nargs='+',default=[.098,.1029,.110]);p.add_argument('--ranking',choices=['work','com'],default='work');p.add_argument('--keep',type=int,default=8)
    run(p.parse_args())
