"""Validate a user input; never search or change its relative grasp pose."""
from kinematic_constraints import FixedGrasp,Panda
from co_common import *
from research_step4 import gravity_check,serial
import argparse,hashlib,shutil,time

def run(args):
    start=time.monotonic();source=Path(args.grasp);config=json.loads(source.read_text());out=Path(args.out)
    if (out/'report.json').exists():raise RuntimeError('Choose a fresh output directory')
    out.mkdir(parents=True,exist_ok=True);snap=out/'runtime_sources';snap.mkdir(exist_ok=True)
    paths=[Path(__file__),Path(__file__).with_name('kinematic_constraints.py'),HERE/'single_load/robot.py',HERE/'single_load/research_step4.py']
    for path in paths:shutil.copy2(path,snap/path.name)
    save(snap/'manifest.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    grasp=FixedGrasp(config['T_object_hand'],config['gap_m']);states={p:state(config['object'],p) for p in config['poses']};mesh=next(iter(states.values()))[2]
    workids=np.unique(np.concatenate([s[0].domain.work_ids for s in states.values()]));work=mesh.submesh([workids],append=True)
    robot=Panda();solid=S.solid(mesh);geometry=robot.approach_report(grasp.T_object_hand,grasp.gap_m,solid)
    work_clear=robot.work_contact_clear(grasp.T_object_hand,grasp.gap_m,work)
    contact=robot.object_contacts(grasp.T_object_hand,grasp.gap_m,solid)
    patches=robot.contact_patches(grasp.T_object_hand,grasp.gap_m-.0004,mesh);gravity=gravity_check(mesh,patches,states)
    poses=[]
    for pose,(_,T,_) in states.items():
        world=T.copy();world[:3,3]+=config.get('station_offset_m',[.45,0,.15])
        q=robot.ik(grasp.hand_pose(world));arm=dict(passed=False,reason='ik')
        if q is not None:
            moved=mesh.copy();moved.apply_transform(world);arm=robot.arm_report(q,grasp.gap_m,S.solid(moved))
        poses.append(dict(pose=pose,ik_passed=q is not None,q=q,arm=arm,gravity=gravity[pose]))
    usable=[r['pose'] for r in poses if r['ik_passed'] and r['arm']['passed'] and r['gravity']['passed']]
    passed=bool(geometry['passed'] and work_clear and contact['passed'] and usable)
    report=dict(object=config['object'],user_grasp_input_accepted=passed,loading_grasp_geometry_passed=geometry['passed'],work_clear=work_clear,two_jaw_contact_passed=contact['passed'],usable_loading_poses=usable,all_task_object_grasp_endpoints_passed=len(usable)==len(states),grasp_pose_optimized=False,T_object_hand=grasp.T_object_hand,gap_m=grasp.gap_m,pose_results=poses,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),seconds=time.monotonic()-start,common_support_grasp_passed=None,final_insertion_path_passed=None,final_fixture_accepted=False,meaning='User grasp accepted as a kinematic input; support common holding and insertion require actual Step5 geometry. Fixed relative grasp does not fix object SE3 path.')
    constraints=dict(T_object_hand=grasp.T_object_hand,loading_gap_m=grasp.gap_m,poses=config['poses'],station_offset_m=config.get('station_offset_m',[.45,0,.15]),loading_poses=usable,object_contact_patches=patches,support_required_regions_source_faces=np.unique(patches['sources']),support_material_and_jaw_gap='Step5 must design actual support at this fixed grasp region, with joint-limit-compliant closure; no hand-frame relocation allowed without changing the user input',entry_path_family='variable SE3 object waypoints with T_world_hand(t)=T_world_object(t) @ T_object_hand; straight direction is one possible initialization')
    save(out/'constraints_for_step5.json',serial(constraints));save(out/'report.json',serial(report));print(json.dumps(serial(report)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--grasp',required=True);p.add_argument('--out',required=True);run(p.parse_args())
