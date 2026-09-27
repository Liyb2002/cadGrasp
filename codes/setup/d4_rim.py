"""D4: external rim grasp with a generic 180 mm parallel-jaw gripper."""
import sys
sys.dont_write_bytecode=True
import argparse,json
import numpy as np,mujoco,trimesh
from scipy.spatial.transform import Rotation
import grasp as G,sequence as S,kuka_transfer as K
from d1_sequence import Derived
from sequence_export import WorkRegions,export,write,digest

class D4Trial(Derived):
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
  h=G.transform(self.data,self.hand)
  for _ in range(4000):self.step(h)
  # Keep a small floor preload while the rim slides into equilibrium.
  h=G.transform(self.data,self.hand)
  for j in range(3500):
   if j%10==0:
    mujoco.mj_forward(self.model,self.data)
    floor_force=S.contacts(self.model,self.data)[1]
   if floor_force<.15:h[2,3]-=.0000005
   self.step(h)
  mujoco.mj_forward(self.model,self.data)
  actual=G.transform(self.data,self.obj);gap=float((self.mesh.vertices@actual[2,:3]).min()+actual[2,3]);pose=actual.copy();pose[2,3]-=gap
  fingers,force=S.contacts(self.model,self.data);verts=trimesh.transform_points(self.mesh.vertices,pose);ids=np.flatnonzero(verts[:,2]<1e-8)
  com=pose[:3,:3]@self.mesh.center_mass+pose[:3,3];dof=int(self.model.jnt_dofadr[self.model.body_jntadr[self.obj]]);vel=self.data.qvel[dof:dof+6]
  row=dict(raw_mesh_floor_gap_m=gap,object_floor_normal_force_N=force,bilateral_grip=fingers==2,target_position_error_m=abs(gap),target_rotation_error_deg=0.,original_command_rotation_error_deg=float(np.rad2deg(Rotation.from_matrix(actual[:3,:3]@target[:3,:3].T).magnitude())),linear_speed_m_s=float(np.linalg.norm(vel[:3])),angular_speed_rad_s=float(np.linalg.norm(vel[3:])),static_com_contact_offset_m=float(np.linalg.norm(com[:2]-verts[ids[0],:2])) if len(ids) else 0)
  good=(len(ids)==1 and fingers==2 and force>.001 and -.0002<gap<.0001 and row['static_com_contact_offset_m']>.003 and row['linear_speed_m_s']<.0005 and row['angular_speed_rad_s']<.01 and self.regions.feasible(pose))
  return pose,row,good


class RimTrial(D4Trial):
 def step(self,h,force=-70.):
  S.Trial.step(self,h,force*self.clamp_force/70 if force<0 else force)

def candidates(mesh,initial):
 v=trimesh.transform_points(mesh.vertices,initial)
 for yaw in np.arange(0.,180.,30.):
  closing=np.array([np.cos(np.deg2rad(yaw)),np.sin(np.deg2rad(yaw)),0.]);approach=np.array([0.,0.,-1.]);proj=v@closing
  ends=v[[np.argmin(proj),np.argmax(proj)]];middle=ends.mean(axis=0);width=float(np.ptp(proj));R=np.column_stack([np.cross(closing,approach),closing,approach])
  for depth in (.112,.108,.105):
   T=np.eye(4);T[:3,:3]=R;T[:3,3]=middle-depth*approach
   yield dict(hand=T,width=width,opening=.09,yaw_deg=float(yaw),depth_m=depth,contacts=ends)

def finalize(folder):
 manifest=folder/'poses.json';record=json.loads(manifest.read_text());record['generator_sha256']=digest(G.HERE/'d4_rim.py')
 record['generator_dependencies']={'codes/setup/d1_sequence.py':digest(G.HERE/'d1_sequence.py')}
 record['transfer_mode']='Same external-rim grasp throughout: pickup and first placement allow 25 mm lift, later rotations stay grounded. After settling, lower the hand at up to 0.5 mm/s while floor normal force is below 0.15 N.'
 record['verification_scope']='Motion-derived free-object contact rollout; actual held target orientations are seated by removing measured floor penetration. Sampled KUKA IK, limits, speed, arm/scene collisions, and gripper/floor contacts are checked. Arm actuator torques and hardware are not certified.'
 record['pose_selection']['reproduction_command']='PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 /Users/yuanboli/miniforge3/envs/cadgrasp/bin/python codes/setup/d4_rim.py --candidate 0'
 record['pose_selection']['trajectory_note']='Unselected exploration states, including a large slip during the 11 degree command before the controller returned to the selected 12 degree neighborhood, remain in the saved continuous rollout. No object state was reset.'
 record['grasp']['maximum_opening_m']=.18;record['grasp']['finger_dimensions_m']=[.04,.012,.08];record['grasp']['closing_actuator_force_N']=10.5
 z=np.load(folder/'trajectory.npz');tool=G.ROOT/record['grasp']['model'];model=G.build('D4',hand_xml=tool,floor_hull=True);data=mujoco.MjData(model);worst=0.;count=0
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
 record['verification']=dict(gripper_floor_collision=dict(sample_frames=len(z['qpos']),contact_count=count,maximum_penetration_m=-worst),unheld_release_probe=dict(scope='Independent release probes from nearest saved target frame with zero initial velocity and disabled gripper/object collision; these branches do not change the continuous placement trajectory.',results=rows),maximum_tilt_from_rest_during_rollout_deg=float(np.degrees(np.arccos(np.clip(z['T_world_object'][:,2,:3]@z['rest_transform'][2,:3],-1,1))).max()))
 write(manifest,record)
 for task in sorted((folder/'tasks').iterdir()):
  path=task/'setup.npz'
  with np.load(path) as a: fields=dict(a)
  fields['poses_sha256']=digest(manifest);np.savez_compressed(path,**fields);report=json.loads((task/'setup.json').read_text());report['source_snapshot_sha256']=digest(path);write(task/'setup.json',report)

def main():
 p=argparse.ArgumentParser();p.add_argument('--force',type=float,default=10.5);p.add_argument('--candidate',type=int,default=-1);a=p.parse_args()
 folder=G.ROOT/'objects/D4';mesh=trimesh.load(folder/'mesh.stl',force='mesh');saved=json.loads((folder/'poses.json').read_text());initial=S.seat(mesh,np.array(saved['rest']['T_world_mesh'] if 'rest' in saved else saved['poses'][1]['T_world_mesh'])[:3,:3]);regions=WorkRegions(mesh);tool=G.HERE/'assets/parallel_jaw_180.xml';model=G.build('D4',hand_xml=tool,floor_hull=True)
 for ci,candidate in enumerate(candidates(mesh,initial)):
  if a.candidate>=0 and ci!=a.candidate:continue
  try:
   if not G.geometry_check(model,initial,candidate):
    print(json.dumps(dict(candidate=ci,failed='initial geometry')),flush=True);continue
   trial=RimTrial(model,mesh,initial,candidate,regions=regions);trial.clamp_force=a.force;trial.warmup();poses=[];checks=[];begin=5.
   for angle in np.arange(4.,26.,1.):
    yaw=np.deg2rad(candidate['yaw_deg']);axis=np.array([-np.sin(yaw),np.cos(yaw),0.]);target=S.seat(mesh,Rotation.from_rotvec(axis*np.deg2rad(angle)).as_matrix()@trial.start[:3,:3]);pose,row,good=trial.move(target,lift=.025 if angle==4 else 0.)
    sep=180 if not poses else min(float(np.rad2deg(Rotation.from_matrix(pose[:3,:3]@x[:3,:3].T).magnitude())) for x in poses)
    tilt=float(np.degrees(np.arccos(np.clip(pose[2,:3]@initial[2,:3],-1,1))))
    print(json.dumps(dict(candidate=ci,rotation_deg=angle,good=bool(good),separation=sep,tilt=tilt,**row)),flush=True)
    if good and sep>=.75 and tilt>3:
     row.update(pose_id=f'pose_{len(poses)+1}',start_time_s=begin,end_time_s=trial.steps*.001,command_rotation_deg=float(angle));poses.append(pose);checks.append(row);begin=trial.steps*.001
     if len(poses)==10:break
   if len(poses)!=10:raise ValueError(f'Only {len(poses)} usable poses')
   arm=K.check(trial.history);trial.arm_frames,trial.arm_q,arm['scene_checks']=K.check_scene(model,trial.history,'D4',hand_xml=tool)
   rule=dict(method='motion-derived',candidate_index=ci,grasp_yaw_deg=candidate['yaw_deg'],grasp_depth_m=candidate['depth_m'],command_rotation_start_deg=4.,command_rotation_step_deg=1.,minimum_rotation_separation_deg=.75,closing_actuator_force_N=a.force,description='Actual equilibria selected from one continuous free-object rollout; only measured contact penetration is removed for exactly seated task inputs. Original commanded orientations are not claimed achieved.')
   export('D4',mesh,initial,candidate,trial,poses,rule,checks,arm,hand_xml=tool,generator='codes/setup/d4_rim.py',metadata=dict(transfer_mode='Same rim grasp throughout; pickup and first placement allow 25 mm lift, subsequent reorientation is grounded; hand is frozen to settle each observed target.'))
   finalize(folder)
   print(json.dumps(dict(success=True,checks=checks,arm=arm)),flush=True);return
  except ValueError as error:print(json.dumps(dict(candidate=ci,failed=str(error))),flush=True)
 raise RuntimeError('D4 rim search exhausted')
if __name__=='__main__':main()
