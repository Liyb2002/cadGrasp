"""Test Step4's thin-support common grasp separately from object loading."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
from co_common import *
from robot import Panda
from retention import check
from research_step4 import serial
from scipy.spatial.transform import Rotation,Slerp
import argparse,time,hashlib,shutil

def run(args):
    began=time.monotonic();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'report.json').exists():raise RuntimeError('Choose a fresh experiment directory')
    snap=out/'runtime_sources';snap.mkdir(exist_ok=True)
    code=[Path(__file__),Path(__file__).with_name('robot.py'),Path(__file__).with_name('retention.py'),Path(__file__).with_name('research_step4.py')]
    for path in code:shutil.copy2(path,snap/path.name)
    save(snap/'manifest.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in code})
    source=HERE/'output/B/pose1+2+4+6/step3/step3.2/wrapped_support.obj'
    states={f'pose_{i}':state('B',f'pose_{i}') for i in [1,2,4,6]};mesh=next(iter(states.values()))[2]
    workids=np.unique(np.concatenate([s[0].domain.work_ids for s in states.values()]));allowed=np.setdiff1d(np.arange(len(mesh.faces)),workids)
    work=mesh.submesh([workids],append=True);proxy=trimesh.load(source,force='mesh',process=False)
    proxy_solid=S.solid(proxy);assembly=proxy_solid+S.solid(mesh)
    triangles,sources=contact_boundary(mesh,proxy,allowed)
    rotations=[];transforms=[s[1] for s in states.values()]
    for A,B in zip(transforms,transforms[1:]):
        rotations.extend(Slerp([0,1],Rotation.from_matrix([A[:3,:3],B[:3,:3]]))(np.linspace(0,1,13)).as_matrix())
    loading_path=Path(args.loading)/'candidates.json';loading=json.loads(loading_path.read_text());robot=Panda();trace=[];accepted=[]
    save(out/'inputs.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,loading_path]})
    for index,load in enumerate(loading):
        hand=np.asarray(load['hand']);row=dict(loading_candidate_index=index,loading_pair=load['pair'],hand=hand)
        centered=robot.center_on_first_contacts(hand,assembly)
        if not centered['passed']:row.update(status='centering',diagnostic=centered);trace.append(row);continue
        hand=centered['hand'];width=centered['width'];row.update(hand=hand,width_m=width,centering=centered)
        geometry=robot.approach_report(hand,width,assembly)
        if not geometry['passed']:row.update(status='assembly_hand_collision',diagnostic=geometry);trace.append(row);continue
        if not robot.work_contact_clear(hand,width,work):row['status']='work';trace.append(row);continue
        touches=robot.object_contacts(hand,width,proxy_solid)
        if not touches['passed']:row.update(status='two_jaw_support_contact_missing',diagnostic=touches);trace.append(row);continue
        patches=robot.contact_patches(hand,width-.0004,proxy)
        retention=check(mesh,proxy,triangles,sources,patches['points'],patches['normals'],rotations)
        row.update(actual_support_contact_patches=patches,retention=retention)
        if not retention['passed']:row['status']='retention';trace.append(row);continue
        usable=[];solutions={};reports={}
        for pose,(_,T,_) in states.items():
            world=T.copy();world[:3,3]+=[.45,0,.15]
            q=robot.ik(world@hand)
            if q is None:reports[pose]=dict(passed=False,reason='ik');continue
            placed=S.unpack(assembly);placed.apply_transform(world)
            reports[pose]=robot.arm_report(q,width,S.solid(placed));solutions[pose]=q
            if reports[pose]['passed']:usable.append(pose)
        row.update(pose_ik=solutions,arm_reports=reports,usable_common_poses=usable)
        if len(usable)!=len(states):row['status']='arm';trace.append(row);continue
        row['status']='proxy_common_candidate';accepted.append(row);trace.append(row)
        print('COMMON',index,'gap_mm',round(width*1000,2),'retention',retention['passed'],'endpoints',usable,flush=True)
    report=dict(loading_candidates=len(loading),proxy_common_candidates=len(accepted),statuses={status:sum(r['status']==status for r in trace) for status in sorted({r['status'] for r in trace})},proxy_kind='saved 5mm non-working wrap; floors/contact connectivity and insertion not yet certified',intermediate_retention_orientations=len(rotations),step4_fully_passed=False,seconds=time.monotonic()-began)
    save(out/'trace.json',serial(trace));save(out/'candidates.json',serial(accepted));save(out/'report.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--loading',required=True);p.add_argument('--out',required=True);run(p.parse_args())
