"""Reproducible D1 motion-derived, physically integrated pose sequence search."""
import sys
sys.dont_write_bytecode=True
import json, argparse
import numpy as np
import mujoco, trimesh
from scipy.spatial.transform import Rotation,Slerp
import grasp as G
import sequence as S
import kuka_transfer as K
from sequence_export import WorkRegions, export, write, digest

class Derived(S.Trial):
 def move(self,target,lift=.025):
  source=G.transform(self.data,self.obj)
  self.relative=np.linalg.inv(G.transform(self.data,self.hand))@source
  self.relative_inverse=np.linalg.inv(self.relative);self.offset[:]=0
  rotations=Slerp([0,1],Rotation.from_matrix([source[:3,:3],target[:3,:3]]))
  start_height=max(0,float((self.floor_vertices@source[2,:3]).min()+source[2,3]))
  xy0=(source[:3,:3]@self.mesh.center_mass+source[:3,3])[:2]
  for step in range(4500):
   t=step*.001;fraction=G.smooth((t-.6)/1.4)
   R=rotations(fraction).as_matrix()
   desired=np.eye(4);desired[:3,:3]=R
   desired[:2,3]=(1-fraction)*xy0-(R@self.mesh.center_mass)[:2]
   desired[2,3]=-(self.floor_vertices@R[2]).min()
   if t<.6:height=start_height+(lift-start_height)*G.smooth(t/.6)
   elif t<2:height=lift
   else:height=lift*(1-G.smooth((t-2)/.8))-.00015*G.smooth((t-2)/.8)
   desired[2,3]+=height
   self.follow(desired)
  # Freeze the final actuator pose and allow actual contact equilibrium to settle.
  h=np.eye(4);h[:3,3]=self.data.mocap_pos[0];h[:3,:3]=Rotation.from_quat(self.data.mocap_quat[0][[1,2,3,0]]).as_matrix()
  for _ in range(1000):self.step(h)
  mujoco.mj_forward(self.model,self.data)
  actual=G.transform(self.data,self.obj);gap=float((self.mesh.vertices@actual[2,:3]).min()+actual[2,3])
  pose=actual.copy();pose[2,3]-=gap
  fingers,force=S.contacts(self.model,self.data)
  verts=trimesh.transform_points(self.mesh.vertices,pose);ids=np.flatnonzero(verts[:,2]<1e-8)
  com=pose[:3,:3]@self.mesh.center_mass+pose[:3,3]
  dof=int(self.model.jnt_dofadr[self.model.body_jntadr[self.obj]])
  vel=self.data.qvel[dof:dof+6]
  row=dict(raw_mesh_floor_gap_m=gap,object_floor_normal_force_N=force,bilateral_grip=fingers==2,
    target_position_error_m=abs(gap),target_rotation_error_deg=0.,
    original_command_rotation_error_deg=float(np.rad2deg(Rotation.from_matrix(actual[:3,:3]@target[:3,:3].T).magnitude())),
    linear_speed_m_s=float(np.linalg.norm(vel[:3])),angular_speed_rad_s=float(np.linalg.norm(vel[3:])),
    static_com_contact_offset_m=float(np.linalg.norm(com[:2]-verts[ids[0],:2])) if len(ids) else 0)
  good=(len(ids)==1 and fingers==2 and force>.001 and -.0001<gap<.0001 and row['static_com_contact_offset_m']>.003
    and row['linear_speed_m_s']<.0005 and row['angular_speed_rad_s']<.01 and self.regions.feasible(pose))
  return pose,row,good

def release_probe(folder):
 z=np.load(folder/'trajectory.npz');model=G.build('D1',hand_xml=G.HERE/'assets/parallel_jaw.xml',floor_hull=True)
 for i in range(model.ngeom):
  if model.geom(i).name.startswith('hand_geom_'):model.geom_contype[i]=0;model.geom_conaffinity[i]=0
 rows=[]
 for i,t in enumerate(z['target_times_s']):
  idx=int(np.argmin(abs(z['time_s']-t)));data=mujoco.MjData(model)
  data.qpos[:]=z['qpos'][idx];data.mocap_pos[:]=z['mocap_pos'][idx];data.mocap_quat[:]=z['mocap_quat'][idx];mujoco.mj_forward(model,data)
  R0=G.transform(data,model.body('object').id)[:3,:3]
  for _ in range(1000):mujoco.mj_step(model,data)
  R=G.transform(data,model.body('object').id)[:3,:3]
  rows.append(dict(pose_id=f'pose_{i+1}',unheld_rotation_after_1s_deg=float(np.rad2deg(Rotation.from_matrix(R@R0.T).magnitude()))))
 return dict(scope='Independent release probes from the closest saved target frame, zero initial velocities, hand/object collision disabled. These branches do not modify the continuous placement trajectory.',results=rows)

def provenance(folder):
 manifest=folder/'poses.json';record=json.loads(manifest.read_text())
 record.setdefault('verification',{})['unheld_release_probe']=release_probe(folder)
 record['generator']='codes/setup/d1_sequence.py'
 record['transfer_mode']='Same grasp throughout; 25 mm lift/reorient/lower for the first nine selected poses, then a grounded rotation for the final pose; skipped exploration states remain in the continuous trajectory.'
 record['pose_selection']['reproduction_command']='PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 /Users/yuanboli/miniforge3/envs/cadgrasp/bin/python codes/setup/d1_sequence.py --candidates 125 --increment 2'
 record['pose_selection']['final_command']='3 degrees around world x from pose_9, then select the actual grounded equilibrium'
 record['pose_selection']['selection_warning']='The final commanded orientation differed from the selected reached pose by about 20.55 degrees; all pose transforms represent reached orientations.'
 record['generator_sha256']=digest(G.HERE/'d1_sequence.py')
 record['verification_scope']='Motion-derived free-object contact simulation targets, seated by the measured submillimetre contact penetration correction; sampled robot IK/limits/speeds and scene collisions. Robot is kinematic; actuator torques and physical hardware are not certified.'
 write(manifest,record)
 for task in sorted((folder/'tasks').iterdir()):
  path=task/'setup.npz'
  with np.load(path) as z: fields=dict(z)
  fields['poses_sha256']=digest(manifest);np.savez_compressed(path,**fields)
  report=json.loads((task/'setup.json').read_text());report['source_snapshot_sha256']=digest(path)
  report['pose_selection']='motion-derived from a continuous numerical contact rollout'
  write(task/'setup.json',report)

def main():
 p=argparse.ArgumentParser();p.add_argument('--candidates',default='125');p.add_argument('--increment',type=float,default=2.);p.add_argument('--axis',type=int,default=0);p.add_argument('--placement',type=int,default=0);a=p.parse_args()
 folder=G.ROOT/'objects/D1';mesh=trimesh.load(folder/'mesh.stl',force='mesh')
 saved=json.loads((folder/'poses.json').read_text());initial=S.seat(mesh,np.array(saved['rest']['T_world_mesh'] if 'rest' in saved else saved['poses'][a.placement]['T_world_mesh'])[:3,:3])
 regions=WorkRegions(mesh);model=G.build('D1',hand_xml=G.HERE/'assets/parallel_jaw.xml',floor_hull=True)
 choices=list(map(int,a.candidates.split(',')));candidates=dict((i,c) for i,c in enumerate(G.candidates(mesh,initial,max(choices)//39+1,max_width=.15,depths=(.10,.08,.112))) if i in choices)
 axes=[[.23,1,.13],[-.23,-1,.13],[1,.23,.13],[-1,-.23,.13]]
 axis=np.array(axes[a.axis]);axis/=np.linalg.norm(axis)
 for ci in choices:
  candidate=candidates[ci]
  try:
   if not G.geometry_check(model,initial,candidate):continue
   trial=Derived(model,mesh,initial,candidate,regions=regions);trial.warmup()
   poses=[];checks=[];begin=5.
   for angle in np.arange(6.,79.,a.increment):
    target=S.seat(mesh,Rotation.from_rotvec(axis*np.deg2rad(angle)).as_matrix()@trial.start[:3,:3])
    if len(poses)==9:target=S.seat(mesh,Rotation.from_euler('x',3,degrees=True).as_matrix()@poses[-1][:3,:3])
    pose,row,good=trial.move(target,lift=0. if len(poses)==9 else .025)
    sep=180 if not poses else min(float(np.rad2deg(Rotation.from_matrix(pose[:3,:3]@x[:3,:3].T).magnitude())) for x in poses)
    tilt=float(np.rad2deg(np.arccos(np.clip(pose[2,:3]@initial[2,:3],-1,1))))
    print(json.dumps(dict(candidate=ci,angle=angle,good=bool(good),separation=sep,tilt=tilt,**row)),flush=True)
    if good and sep>=1.5 and tilt>3:
     row.update(pose_id=f'pose_{len(poses)+1}',start_time_s=begin,end_time_s=trial.steps*.001)
     poses.append(pose);checks.append(row);begin=trial.steps*.001
     if len(poses)==10:break
   if len(poses)!=10:raise ValueError(f'Only {len(poses)} usable states')
   arm=K.check(trial.history);trial.arm_frames,trial.arm_q,arm['scene_checks']=K.check_scene(model,trial.history,'D1')
   rule=dict(method='motion-derived',axis_world=axis.tolist(),candidate_index=ci,
    description='Measured stationary contact orientations selected from one continuous rollout; exact floor seating removes only measured contact penetration. Original commanded targets are not claimed achieved.',minimum_rotation_separation_deg=1.5)
   export('D1',mesh,initial,candidate,trial,poses,rule,checks,arm)
   provenance(folder)
   print(json.dumps(dict(success=True,checks=checks,arm=arm)),flush=True);return
  except ValueError as error:print(json.dumps(dict(candidate=ci,failed=str(error))),flush=True)
 raise RuntimeError('D1 motion-derived search exhausted')
if __name__=='__main__':main()
