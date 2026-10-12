"""Search D2 measured equilibria in a continuous parallel-jaw rollout."""
import sys
sys.dont_write_bytecode=True
import json,argparse
import numpy as np,mujoco,trimesh
from scipy.spatial.transform import Rotation
import grasp as G,sequence as S,kuka_transfer as K
from d1_sequence import Derived
from sequence_export import WorkRegions,export,write,digest

class D2Trial(Derived):
 def move(self,target,lift=.025):
  pose,row,good=super().move(target,lift)
  if row['object_floor_normal_force_N']==0 and row['raw_mesh_floor_gap_m']>0:
   h=G.transform(self.data,self.hand)
   for j in range(2500):
    if j%20==0:
     mujoco.mj_forward(self.model,self.data)
     if S.contacts(self.model,self.data)[1]>.01:break
    h[2,3]-=.0000015
    self.step(h)
   for j in range(1500):self.step(h)
  mujoco.mj_forward(self.model,self.data)
  actual=G.transform(self.data,self.obj);gap=float((self.mesh.vertices@actual[2,:3]).min()+actual[2,3]);pose=actual.copy();pose[2,3]-=gap
  fingers,force=S.contacts(self.model,self.data);verts=trimesh.transform_points(self.mesh.vertices,pose);ids=np.flatnonzero(verts[:,2]<1e-8)
  com=pose[:3,:3]@self.mesh.center_mass+pose[:3,3];dof=int(self.model.jnt_dofadr[self.model.body_jntadr[self.obj]]);vel=self.data.qvel[dof:dof+6]
  row=dict(raw_mesh_floor_gap_m=gap,object_floor_normal_force_N=force,bilateral_grip=fingers==2,target_position_error_m=abs(gap),target_rotation_error_deg=0.,original_command_rotation_error_deg=float(np.rad2deg(Rotation.from_matrix(actual[:3,:3]@target[:3,:3].T).magnitude())),linear_speed_m_s=float(np.linalg.norm(vel[:3])),angular_speed_rad_s=float(np.linalg.norm(vel[3:])),static_com_contact_offset_m=float(np.linalg.norm(com[:2]-verts[ids[0],:2])) if len(ids) else 0)
  good=(len(ids)==1 and fingers==2 and force>.001 and -.0002<gap<.0001 and row['static_com_contact_offset_m']>.003 and row['linear_speed_m_s']<.0005 and row['angular_speed_rad_s']<.01 and self.regions.feasible(pose))
  return pose,row,good

def provenance(folder):
 manifest=folder/'poses.json';record=json.loads(manifest.read_text());record['generator']='codes/setup/d2_derived.py';record['generator_sha256']=digest(G.HERE/'d2_derived.py')
 record['generator_dependencies']={'codes/setup/d1_sequence.py':digest(G.HERE/'d1_sequence.py')}
 record['pose_selection'].pop('angles_deg',None)
 record['pose_selection']['command_schedule_degrees']={'start':6,'step':2,'end':'stop after ten accepted actual equilibria'}
 record['pose_selection']['reproduction_command']='PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 /Users/yuanboli/miniforge3/envs/cadgrasp/bin/python codes/setup/d2_derived.py --candidates 78 --placement 2'
 record['verification_scope']='Motion-derived free-object contact simulation with actual grounded targets, sampled KUKA IK/limits/speeds and scene collisions. Robot is kinematic; arm torques and physical hardware are not certified.'
 record['transfer_mode']='Same grasp throughout; lift 25 mm, rotate, lower, and continue lowering at 1.5 mm/s if ground contact is missing; freeze hand for settling. Skipped exploration states remain in the continuous trajectory.'
 z=np.load(folder/'trajectory.npz');model=G.build('D2',hand_xml=G.HERE/'assets/parallel_jaw.xml',floor_hull=True);data=mujoco.MjData(model);worst=0.;count=0
 for q,pos,quat in zip(z['qpos'],z['mocap_pos'],z['mocap_quat']):
  data.qpos[:]=q;data.mocap_pos[:]=pos;data.mocap_quat[:]=quat;mujoco.mj_forward(model,data)
  for contact in data.contact:
   names=[model.geom(int(g)).name for g in (contact.geom1,contact.geom2)]
   if 'floor' in names and any(n.startswith('hand_geom_') for n in names):worst=min(worst,float(contact.dist));count+=1
 for i in range(model.ngeom):
  if model.geom(i).name.startswith('hand_geom_'):model.geom_contype[i]=0;model.geom_conaffinity[i]=0
 rows=[]
 for i,t in enumerate(z['target_times_s']):
  idx=int(np.argmin(abs(z['time_s']-t)));data=mujoco.MjData(model);data.qpos[:]=z['qpos'][idx];data.mocap_pos[:]=z['mocap_pos'][idx];data.mocap_quat[:]=z['mocap_quat'][idx];mujoco.mj_forward(model,data)
  R0=G.transform(data,model.body('object').id)[:3,:3]
  for _ in range(1000):mujoco.mj_step(model,data)
  R=G.transform(data,model.body('object').id)[:3,:3]
  rows.append(dict(pose_id=f'pose_{i+1}',unheld_rotation_after_1s_deg=float(np.rad2deg(Rotation.from_matrix(R@R0.T).magnitude()))))
 record['verification']=dict(gripper_floor_collision=dict(sample_frames=len(z['qpos']),contact_count=count,maximum_penetration_m=-worst),unheld_release_probe=dict(scope='Independent release probes from nearest saved target frame with zero initial velocity and disabled gripper/object collision; these branches do not change the continuous placement trajectory.',results=rows))
 write(manifest,record)
 for task in sorted((folder/'tasks').iterdir()):
  path=task/'setup.npz'
  with np.load(path) as a: fields=dict(a)
  fields['poses_sha256']=digest(manifest);np.savez_compressed(path,**fields);report=json.loads((task/'setup.json').read_text());report['source_snapshot_sha256']=digest(path);write(task/'setup.json',report)

def main():
 p=argparse.ArgumentParser();p.add_argument('--candidates',default='78');p.add_argument('--placement',type=int,default=2);a=p.parse_args()
 folder=G.ROOT/'objects/D2';mesh=trimesh.load(folder/'mesh.stl',force='mesh');saved=json.loads((folder/'poses.json').read_text());initial=S.seat(mesh,np.array(saved['rest']['T_world_mesh'] if 'rest' in saved else saved['poses'][a.placement]['T_world_mesh'])[:3,:3]);regions=WorkRegions(mesh);model=G.build('D2',hand_xml=G.HERE/'assets/parallel_jaw.xml',floor_hull=True)
 choices=list(map(int,a.candidates.split(',')));candidates=dict((i,c) for i,c in enumerate(G.candidates(mesh,initial,max(choices)//39+1,max_width=.15,depths=(.10,.08,.112))) if i in choices)
 for ci in choices:
  candidate=candidates[ci]
  try:
   if not G.geometry_check(model,initial,candidate):continue
   trial=D2Trial(model,mesh,initial,candidate,regions=regions);trial.warmup();nominal,rule=trial.proposals();axis=np.array(rule['axis_world']);poses=[];checks=[];begin=5.
   for angle in np.arange(6.,69.,2.):
    target=S.seat(mesh,Rotation.from_rotvec(axis*np.deg2rad(angle)).as_matrix()@trial.start[:3,:3]);pose,row,good=trial.move(target)
    sep=180 if not poses else min(float(np.rad2deg(Rotation.from_matrix(pose[:3,:3]@x[:3,:3].T).magnitude())) for x in poses)
    print(json.dumps(dict(candidate=ci,angle=angle,good=bool(good),separation=sep,**row)),flush=True)
    if good and sep>=1.5:
     row.update(pose_id=f'pose_{len(poses)+1}',start_time_s=begin,end_time_s=trial.steps*.001);poses.append(pose);checks.append(row);begin=trial.steps*.001
     if len(poses)==10:break
   if len(poses)!=10:raise ValueError(f'Only {len(poses)} usable poses')
   arm=K.check(trial.history);trial.arm_frames,trial.arm_q,arm['scene_checks']=K.check_scene(model,trial.history,'D2')
   rule.update(method='motion-derived',candidate_index=ci,generator='codes/setup/d2_derived.py',description='Actual equilibria selected from one continuous physical rollout; only measured contact penetration is removed when seating task inputs; original commanded poses are not claimed achieved.')
   export('D2',mesh,initial,candidate,trial,poses,rule,checks,arm)
   provenance(folder)
   print(json.dumps(dict(success=True,checks=checks,arm=arm)),flush=True);return
  except ValueError as error:print(json.dumps(dict(candidate=ci,failed=str(error))),flush=True)
 raise RuntimeError('D2 exhausted')
if __name__=='__main__':main()
