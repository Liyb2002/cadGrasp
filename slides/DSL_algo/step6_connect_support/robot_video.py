"""KUKA presentation using the simulation model and belt_test's connected tool.

Fixed world floor/camera; sampled IK for illustration; actual saved task arrows.
No dynamics or robot/object collision certificate is inferred from this movie.
"""
from pathlib import Path
import sys
import json
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from step1.needs import ROOT
from step3_scheculer import contacts as I
from step4_connect_support.video import mp4_writer
from step6_connect_support.snap_retention import SnapLips
from PIL import ImageFont

sys.path.insert(0,str(ROOT/'slides/belt_test/code'))
import shared_workflow as REFERENCE
kuka=REFERENCE.kuka
mujoco=REFERENCE.mujoco
WIDTH,HEIGHT,FPS,DURATION=1600,900,24,15.


def source_files():
    return [Path(__file__),Path(__file__).with_name('snap_retention.py'),Path(REFERENCE.__file__),Path(kuka.__file__),
            ROOT/'codes/simulation/workflow.py',ROOT/'codes/simulation/video.py',
            *sorted(kuka.ASSETS.rglob('*.dae')),*sorted(kuka.ASSETS.rglob('*.stl')),
            kuka.ASSETS/'urdf/lbr_med14_r820_macro.xacro',kuka.ASSETS/'config/joint_limits.yaml']


class Sequence:
    def __init__(self,domain,blue,base,report,plan,sample_path):
        from step6_connect_support.modular_workflow import pose_at,smooth
        self.pose_at,self.smooth=pose_at,smooth
        self.obj,self.blue,self.base=domain.mesh,blue,base
        self.plan,self.report=plan,report
        self.cases=[dict(work_ids=domain.work_ids)]
        self.rest=np.asarray(plan['T_initial_from_task']);self.d0=np.asarray(report['d0_withdrawal_direction'])
        self.retention=SnapLips(self.obj,blue,domain.work_ids,self.d0,self.rest,plan['initial_installation_length_m'])
        self.grip_blue=np.asarray(report['interface']['port_m'])-np.array([0,0,.015])
        # A body-fixed approach direction above both initial and final floors.
        self.normal=np.array([0.,0.,1.])+self.rest[:3,:3].T@np.array([0.,0.,1.])
        self.normal/=np.linalg.norm(self.normal)
        center=np.asarray(domain.com)
        hits,_,_=self.obj.ray.intersects_location([center],[self.normal],multiple_hits=True)
        self.grip_pair=(min(hits,key=lambda p:np.linalg.norm(p-center))-.018*self.normal
                        if len(hits) else center)
        # Keep the illustrative contact cross-section within the existing jaw
        # stroke. This geometric choice does not evaluate grasp forces.
        side=np.cross(self.normal,[1.,0.,0.]);side/=np.linalg.norm(side)
        tangent=np.cross(self.normal,side)
        axes=np.array([side*np.cos(a)+tangent*np.sin(a) for a in np.linspace(0,2*np.pi,48,endpoint=False)])
        if len(hits):
            surface=min(hits,key=lambda p:np.linalg.norm(p-center))
            for depth in (.018,.015,.012,.010,.025,.030,.035):
                candidate=surface-depth*self.normal
                points,ray,_=self.obj.ray.intersects_location(np.tile(candidate,(len(axes),1)),axes,multiple_hits=True)
                if len(np.unique(ray))!=len(axes) or not self.obj.contains([candidate])[0]:continue
                distances=np.array([np.linalg.norm(points[ray==i]-candidate,axis=1).min() for i in range(len(axes))])
                if distances.min()>.006 and distances.max()<.066:
                    self.grip_pair=candidate;break
        self.arm=kuka.Arm('B');self.park=self.grip_pair+np.array([.16,0,.30])
        samples=json.loads(sample_path.read_text());self.sample_path=sample_path
        pts=np.asarray(samples['pt_m']);forces=np.asarray(samples['force_push_mg'])
        eligible=np.flatnonzero(np.linalg.norm(forces,axis=1)>.35)
        # Three distinct, visible, admissible saved loads, not invented arrows.
        ids=np.asarray(domain.work_ids)[np.asarray(samples['work_face_index'])]
        score=domain.mesh.face_normals[ids]@np.array([-.34,.94,.6])
        candidates=eligible[np.argsort(score[eligible])[-min(2000,len(eligible)):]]
        chosen=[int(candidates[-1])]
        while len(chosen)<3:
            distance=np.min(np.linalg.norm(pts[candidates,None]-pts[chosen],axis=2),axis=1)
            chosen.append(int(candidates[np.argmax(distance)]))
        self.loads=[dict(sample_index=i,point_m=pts[i].tolist(),force_push_mg=forces[i].tolist()) for i in chosen]

    def state(self,t):
        u=float(np.interp(t,[0,.4,3.2,5.5,7.,8.7,9.6,11.,15.],
                              [0,.5,4.,5.,7.,10.,12.,14.,14.]))
        obj=self.pose_at(u,self.plan);blue=obj.copy()
        installed=self.smooth((u-.5)/3.5)
        if u<4:blue[:3,3]+=self.rest[:3,:3]@self.d0*self.plan['initial_installation_length_m']*(1-installed)
        module=trimesh.transform_points([self.grip_blue],blue)[0]
        pair=trimesh.transform_points([self.grip_pair],obj)[0]
        normal=obj[:3,:3]@self.normal
        point=module.copy();gap=.010
        if 3.2<=t<3.6:
            gap=.010+.060*self.smooth((t-3.2)/.4)
        elif 3.6<=t<4.1:
            point=module+normal*.09*self.smooth((t-3.6)/.5);gap=.070
        elif 4.1<=t<4.7:
            f=self.smooth((t-4.1)/.6)
            point=module+(pair-module)*f+normal*.09;gap=.070
        elif 4.7<=t<5.1:
            point=pair+normal*.09*(1-self.smooth((t-4.7)/.4));gap=.070
        elif t>=5.1:
            point=pair;gap=.070-.025*self.smooth((t-5.1)/.4)
        if t>=11.:
            f=self.smooth((t-11.)/.35);gap=.045+.025*f
            if t>=11.35:
                f=self.smooth((t-11.35)/.65)
                point=pair+(self.park-pair)*f
                normal=normal*(1-f)+np.array([0.,0.,1.])*f
        normal/=np.linalg.norm(normal)
        return dict(object=obj,blue=blue,point=point,normal=normal,gap=gap,
            installation_fraction=installed,work=0,task=t>=12.,
            load=min(2,int(max(0,t-12.))),time=float(t))

    def prepare_arm(self):
        self.times=np.linspace(0,DURATION,round(FPS*DURATION))
        states=[self.state(t) for t in self.times]
        points=np.array([s['point'] for s in states]);middle=(points.min(0)+points.max(0))/2
        failures=[]
        _,axes=np.linalg.eigh(np.cov(points[:,:2].T));side=axes[:,0]
        offsets=[]
        for radius in (.48,.42,.54,.36):
            for angle in (0.,np.pi,np.pi/6,-np.pi/6,5*np.pi/6,7*np.pi/6):
                c,s=np.cos(angle),np.sin(angle)
                offsets.append(np.r_[radius*np.array([[c,-s],[s,c]])@side,0.])
        for offset in offsets:
            self.arm.base=middle+offset;self.arm.base[2]=0
            direction=middle-self.arm.base;yaw=np.arctan2(direction[1],direction[0])
            self.arm.rotation=REFERENCE.Rotation.from_euler('z',yaw).as_matrix()
            qs=[];previous=None
            try:
                for t,state in zip(self.times,states):
                    previous=self.arm.solve(state['point'],state['normal'],previous);qs.append(previous)
            except ValueError as error:
                failures.append(dict(base_m=self.arm.base.tolist(),time_s=float(t),reason=str(error)));continue
            self.qs=np.array(qs);self.robot_yaw=float(yaw)
            try:
                for frame_time,state in zip(self.times,states):
                    self.jaw_gaps(frame_time,state)
            except RuntimeError as error:
                failures.append(dict(base_m=self.arm.base.tolist(),time_s=float(frame_time),reason=str(error)))
                continue
            errors=[];angles=[]
            for q,state in zip(self.qs,states):
                point,axis,_,_=self.arm.forward(q)
                errors.append(float(np.linalg.norm(point-state['point'])))
                angles.append(float(np.linalg.norm(axis+state['normal'])))
            speed=float(np.max(np.abs(np.diff(self.qs,axis=0))/np.diff(self.times)[:,None]/self.arm.velocity))
            frame_errors=[]
            for frame_time in np.linspace(0,DURATION,round(FPS*DURATION)):
                point,_,_,_=self.arm.forward(self.joints(frame_time))
                frame_errors.append(float(np.linalg.norm(point-self.state(frame_time)['point'])))
            return dict(model='LBR Med 14 R820 from codes/simulation',sampled_ik_passed=True,samples=len(self.times),
                maximum_tcp_error_m=max(errors),maximum_axis_error=max(angles),robot_base_m=self.arm.base.tolist(),
                maximum_rendered_frame_tcp_error_m=max(frame_errors),
                presentation_joint_speed_ratio=speed,reference_time_scale=max(1.,speed*1.05),
                robot_collision_verified=False,grasp_verified=False,dynamics_integrated=False,attempts=failures)
        raise RuntimeError('No continuous sampled arm IK sequence: '+json.dumps(failures))

    def joints(self,t):
        return np.array([np.interp(t,self.times,self.qs[:,i]) for i in range(7)])

    def jaw_gaps(self,t,state):
        """Fit the illustrated fingers to the object at the existing grasp site.

        Ray contact is a drawing aid, not a force-closure or finger collision test.
        """
        if not 5.1<=t<11.35:
            return np.repeat(state['gap'],2)
        _,rotation,_=self.arm.forward(self.joints(t),geometry=True)
        axis=state['object'][:3,:3].T@rotation[:,0]
        points,ray,_=self.obj.ray.intersects_location(
            [self.grip_pair,self.grip_pair],[-axis,axis],multiple_hits=True)
        if any(not np.any(ray==i) for i in (0,1)):
            raise RuntimeError('Illustrated grasp does not meet both sides of the object')
        closed=np.array([np.linalg.norm(points[ray==i]-self.grip_pair,axis=1).min()+.004 for i in (0,1)])
        if closed.max()>.070 or closed.min()<.009:
            raise RuntimeError('Illustrated object grasp exceeds the gripper stroke')
        fraction=self.smooth((t-5.1)/.4) if t<11 else 1-self.smooth((t-11)/.35)
        return .070*(1-fraction)+closed*fraction


class Display:
    def __init__(self,sequence):
        self.sequence=sequence
        self.model=REFERENCE.model_for(sequence)
        bid=self.model.body('robot_B_link_0').id
        self.model.body_pos[bid]=sequence.arm.base
        yaw=sequence.robot_yaw;self.model.body_quat[bid]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
        self.data=mujoco.MjData(self.model)
        self.camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(self.camera)
        self.camera.azimuth=np.rad2deg(sequence.robot_yaw)+290;self.camera.elevation=-28
        # Fit one camera to the entire arm/assembly trajectory, never per frame.
        clouds=[sequence.base.vertices]
        for t,q in zip(sequence.times[::4],sequence.qs[::4]):
            state=sequence.state(t)
            clouds += [trimesh.transform_points(sequence.obj.vertices,state['object']),
                       trimesh.transform_points(sequence.blue.vertices,state['blue'])]
            _,_,transforms=sequence.arm.forward(q,geometry=True)
            clouds += [v@r.T+p for v,(r,p) in zip(sequence.arm.collisions,transforms)]
        points=np.vstack(clouds);center=(points.min(0)+points.max(0))/2
        self.scene_points=points
        az=np.deg2rad(self.camera.azimuth);el=np.deg2rad(-self.camera.elevation)
        view=np.array([np.cos(az)*np.cos(el),np.sin(az)*np.cos(el),np.sin(el)])
        right=np.array([np.sin(az),-np.cos(az),0.]);up=np.cross(right,-view)
        local=points-center;tan=np.tan(np.deg2rad(self.model.vis.global_.fovy)/2)
        distance=np.max(local@view+np.maximum(np.abs(local@up)/tan,np.abs(local@right)/(tan*WIDTH/HEIGHT)))
        self.camera.lookat[:]=center;self.camera.distance=float(distance*1.10)
        self.addresses=[self.model.jnt_qposadr[self.model.joint(f'robot_B_joint_{i}').id] for i in range(1,8)]
        self.jaws=[self.model.jnt_qposadr[self.model.joint(f'gripper_slide_{s}').id] for s in (-1,1)]
        self.fixed_floor=self.model.geom_pos[self.model.geom('floor').id].copy()
        self.fixed_base=self.model.geom_pos[self.model.geom('shared_base').id].copy()
        self.detail_camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(self.detail_camera)
        normal=sequence.rest[:3,:3]@np.mean(np.vstack([x['normals'] for x in sequence.retention.lips]),axis=0)
        self.detail_camera.azimuth=np.rad2deg(np.arctan2(normal[1],normal[0]))+180
        self.detail_camera.elevation=-30
        self.detail_camera.lookat[:]=trimesh.transform_points([sequence.retention.center],sequence.rest)[0]
        self.detail_camera.distance=.18
        self.detail_option=mujoco.MjvOption();mujoco.mjv_defaultOption(self.detail_option)
        self.detail_option.geomgroup[2]=0  # Hide the robot only in the mechanism close-up.
        self.font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',24)

    def fit_camera(self,renderer):
        """Check the actual MuJoCo projection of the full motion envelope."""
        self.frame(renderer,0.)
        for _ in range(20):
            renderer.update_scene(self.data,camera=self.camera)
            a,b=renderer.scene.camera;origin=(a.pos.astype(float)+b.pos.astype(float))/2
            forward=a.forward.astype(float);up=a.up.astype(float);right=np.cross(forward,up)
            v=self.scene_points-origin;depth=v@forward
            focal=HEIGHT*a.frustum_near/(a.frustum_top-a.frustum_bottom)
            x=WIDTH/2+focal*(v@right)/depth;y=HEIGHT/2-focal*(v@up)/depth
            if depth.min()>0 and x.min()>.06*WIDTH and x.max()<.94*WIDTH and y.min()>.06*HEIGHT and y.max()<.94*HEIGHT:
                self.fit_detail_camera(renderer)
                return
            self.camera.distance*=1.08
        raise RuntimeError('Fixed camera could not fit the complete robot trajectory')

    def fit_detail_camera(self,renderer):
        """Pick a fixed local view where the module does not hide the snap lip."""
        state=self.sequence.state(3.3)
        REFERENCE.set_pose(self.model,self.data,'object',state['object'])
        REFERENCE.set_pose(self.model,self.data,'contact',state['blue'])
        mujoco.mj_forward(self.model,self.data)
        obstacle=trimesh.util.concatenate([self.sequence.obj,self.sequence.blue]).copy()
        obstacle.apply_transform(state['object'])
        best=None
        for lip in self.sequence.retention.lips:
            local=lip['points'][2:]+lip['normals'][2:]*.0008
            points=trimesh.transform_points(local,state['blue'])
            normal=lip['normals'][2:]@state['blue'][:3,:3].T
            center=points.mean(axis=0)
            self.detail_camera.lookat[:]=center
            self.detail_camera.distance=.14
            for elevation in (-20.,-40.,-60.):
                self.detail_camera.elevation=elevation
                for azimuth in np.arange(0.,360.,30.):
                    self.detail_camera.azimuth=azimuth
                    renderer.update_scene(self.data,camera=self.detail_camera,scene_option=self.detail_option)
                    a,b=renderer.scene.camera;origin=(a.pos.astype(float)+b.pos.astype(float))/2
                    direction=origin-points;direction/=np.linalg.norm(direction,axis=1)[:,None]
                    blocked=obstacle.ray.intersects_any(points+direction*.0002,direction)
                    visible=~blocked
                    score=float(np.sum(visible*(1+np.maximum(0,np.sum(normal*direction,axis=1)))))
                    if best is None or score>best[0]:
                        best=(score,azimuth,elevation,center.copy())
        _,self.detail_camera.azimuth,self.detail_camera.elevation,center=best
        self.detail_camera.lookat[:]=center

    def frame(self,renderer,t):
        state=self.sequence.state(t)
        REFERENCE.set_pose(self.model,self.data,'object',state['object'])
        REFERENCE.set_pose(self.model,self.data,'contact',state['blue'])
        self.data.qpos[self.addresses]=self.sequence.joints(t)
        assert .009<=state['gap']<=.07000001
        self.data.qpos[self.jaws]=self.sequence.jaw_gaps(t,state)-.009
        self.data.qvel[:]=0;mujoco.mj_forward(self.model,self.data)
        self.model.geom_rgba[self.model.geom('work_0').id,3]=1.
        np.testing.assert_array_equal(self.model.geom_pos[self.model.geom('floor').id],self.fixed_floor)
        np.testing.assert_array_equal(self.model.geom_pos[self.model.geom('shared_base').id],self.fixed_base)
        renderer.update_scene(self.data,camera=self.camera)
        self.sequence.retention.add_to_scene(mujoco,renderer.scene,state['blue'],state['installation_fraction'])
        picture=Image.fromarray(renderer.render())
        if state['task']:
            load=self.sequence.loads[state['load']];point=np.asarray(load['point_m']);force=np.asarray(load['force_push_mg'])
            tail=point-force/np.linalg.norm(force)*.075
            REFERENCE.annotate(ImageDraw.Draw(picture),REFERENCE.project(renderer.scene,tail,WIDTH,HEIGHT),
                               REFERENCE.project(renderer.scene,point,WIDTH,HEIGHT))
        if 1.2<=t<5.5:
            renderer.update_scene(self.data,camera=self.detail_camera,scene_option=self.detail_option)
            self.sequence.retention.add_to_scene(mujoco,renderer.scene,state['blue'],state['installation_fraction'])
            detail=Image.fromarray(renderer.render()).resize((576,324),Image.Resampling.LANCZOS)
            picture.paste(detail,(24,54))
            ink=ImageDraw.Draw(picture)
            ink.rounded_rectangle((24,20,600,57),radius=6,fill='white')
            label='Push on / lips flex' if t<3.2 else 'Lips recover / module stays attached'
            ink.text((36,25),label,font=self.font,fill='#175e98')
            ink.rectangle((24,54,600,378),outline='#b9cbd8',width=2)
        return np.asarray(picture)


def render(domain,blue,fixed,base,plan,out,design_passed):
    sequence=Sequence(domain,blue,fixed,base,plan,out.parent/'step_1_needs/samples.json')
    robot=sequence.prepare_arm();print('KUKA sampled IK:',robot['maximum_tcp_error_m'],'m',flush=True)
    display=Display(sequence)
    with mujoco.Renderer(display.model,height=HEIGHT,width=WIDTH) as renderer:
        display.fit_camera(renderer)
        with mp4_writer(out/'insertion.mp4',fps=FPS) as writer:
            for t in np.linspace(0,DURATION,round(FPS*DURATION)):
                writer.append_data(display.frame(renderer,float(t)))
        Image.fromarray(display.frame(renderer,13.5)).save(out/'connection.png')
    np.savez_compressed(out/'robot_motion.npz',times_s=sequence.times,joints_rad=sequence.qs,base_m=sequence.arm.base)
    I.save(out/'video_metadata.json',dict(frame_count=round(FPS*DURATION),fps=FPS,duration_seconds=DURATION,
        base_stationary=True,floor_stationary=True,camera_stationary=True,robot=robot,
        stages=['push_on_blue_and_flex_lips','lips_recover_and_release_module','grasp_object_and_lift_retained_blue','dock_into_stationary_base','release_and_show_task_loads'],
        retention=sequence.retention.metadata(),
        gripper_display='two fingers close to object ray intersections; grasp stability remains assumed',
        force_display=dict(source='saved Step1 admissible load samples',examples=sequence.loads,dynamics_applied=False),
        bearing_verified=design_passed,grasp_feasibility_assumed=True,robot_collision_verified=False,
        camera=dict(azimuth=float(display.camera.azimuth),elevation=float(display.camera.elevation),
                    lookat=display.camera.lookat.tolist(),distance=float(display.camera.distance)),
        source_sha256=I.hashes(source_files()+[sequence.sample_path])))
