"""Two-robot use of the saved Step5 fixtures: Panda loads, KUKA works.

The Panda takes each fixture from its park on the bench, installs it at the
station in the certified installation, inserts the bunny along the reversed
exit path, withdraws it after work and turns the empty fixture for the next
task.  The floor never moves: every saved tangent ground plane is the bench
top, and the KUKA translates its needle to that installation's head height
with the needle direction fixed.  Only relative placement enters the Step5
certificates, so this is the same geometry as a fixed needle over a lift.
This is a kinematic presentation of saved geometry, not a swept-collision or
dynamics certificate.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from subdivide_task.step1.common import ROOT,sha256,write_json
from subdivide_task.step4.membrane_search import read_group
from .franka import Franka
from .grasps import pose_blend,translated
from .suction import SuctionPool,ROD
from .kuka import Arm
from .work_path import SurfacePath
from .workflow import smooth
from .prepare import TABLE_Z,WORKBENCH_ABOVE_WORK_BASE,matrices,export_robot_meshes

# Loader beside the station: every exit leaves towards the needle side.
PANDA=dict(base=[.60,.47,TABLE_Z],yaw=-np.pi/2)
OBJECT_PARK=(.95,.40)
FIXTURE_PARKS=((1.00,.12),(1.00,-.24))
BENCH=dict(center=[.775,.085],size=[1.05,1.07])


def video_frame():
    S=np.eye(4);S[:3,:3]=Rotation.from_euler('z',90,degrees=True).as_matrix();S[2,3]=TABLE_Z
    return S


def station(folder):
    """Rows in the video frame from saved Step5 fixtures and installations.

    Each installation is lowered rigidly by its saved floor height so its
    tangent ground plane is the bench top; its needle head moves with it.
    """
    from subdivide_task.step5.grounded_feet import result_model
    folder=Path(folder);report=json.loads((folder/'report.json').read_text())
    model=result_model(report);S=video_frame();rows=[];fixtures=[]
    for s,g in enumerate(report['groups']):
        _,mesh=read_group(g);q,_=read_group(g['core'])
        heights=np.asarray(g['grounding']['global_floor_heights_m'])
        fixtures.append(dict(mesh=mesh,members=g['members'],file=g['file'],sha256=g['sha256']))
        for local,k in enumerate(g['members']):
            D=np.eye(4);D[2,3]=-heights[local]
            T=D@S@np.linalg.inv(q.material[local]);U=D@S@q.poses[local]
            d=S[:3,:3]@q.exits[local]
            sv=trimesh.transform_points(mesh.vertices,T);ov=trimesh.transform_points(model.source.vertices,U)
            if abs(sv[:,2].min()-TABLE_Z)>1e-7:raise ValueError('Fixture does not rest on its saved tangent ground')
            travel=max(.10,float((sv@d).max()-(ov@d).min()+.022))
            rows.append(dict(index=len(rows),task=int(k),pose=model.ids[k],set=s,fixture=T,object=U,direction=d,
                travel=travel,work_ids=np.asarray(model.records[k]['working_face_ids']),drop=float(heights[local]),
                axis=S[:3,:3]@model.axis,head=D[:3,:3]@(S[:3,:3]@model.head+S[:3,3])+D[:3,3]))
    return model,report,rows,fixtures


def station_bench(rows,fixtures,margin=.03):
    """The bench extended, if needed, under every installed ground contact."""
    xy=[]
    for row in rows:
        v=trimesh.transform_points(fixtures[row['set']]['mesh'].vertices,row['fixture'])
        xy.append(v[v[:,2]<=v[:,2].min()+1e-6,:2])
    xy=np.vstack(xy);c=np.asarray(BENCH['center']);h=np.asarray(BENCH['size'])/2
    lo=np.minimum(c-h,xy.min(0)-margin);hi=np.maximum(c+h,xy.max(0)+margin)
    return dict(center=((lo+hi)/2).tolist(),size=(hi-lo).tolist())


class SetsWorkflow:
    """Panda handling with several fixtures on one fixed bench."""
    def __init__(self,model,rows,fixtures,robot_options,park_xy=OBJECT_PARK,fixture_parks=FIXTURE_PARKS):
        self.robot=Franka(**robot_options);self.rows=rows;self.fixtures=fixtures
        # Idle loader pose: home turned about J1 to the side farthest from the needle.
        head=np.mean([r['head'] for r in rows],axis=0);rests=[]
        for turn in (-1.2,1.2):
            q=self.robot.home.copy();q[0]=np.clip(q[0]+turn,self.robot.lower[0]+.05,self.robot.upper[0]-.05)
            rests.append((float(np.linalg.norm(self.robot.fk(q)[:3,3]-head)),q))
        self.rest=max(rests,key=lambda x:x[0])[1]
        self.segments=[];self.clock=0.;self.order=[]
        obj=model.source
        parks,prob=obj.compute_stable_poses(sigma=0,n_samples=1,threshold=0)
        park=parks[int(np.argmax(prob))].copy();v=trimesh.transform_points(obj.vertices,park)
        park[:2,3]+=np.asarray(park_xy)-(v[:,:2].min(0)+v[:,:2].max(0))/2
        park[2,3]+=self.robot.table_z-v[:,2].min();self.park=park
        self.keys=[f'fixture_{s}' for s in range(len(fixtures))]
        # Vacuum-cup loader: the jaws stay closed behind a 60 mm suction rod.
        current=dict(object=park.copy(),hand=self.robot.fk(self.rest),gap=0.,task=None)
        self.fixture_parks=[]
        for s,fixture in enumerate(fixtures):
            first=next(r for r in rows if r['set']==s)
            P=first['fixture'].copy()
            v=trimesh.transform_points(fixture['mesh'].vertices,P)
            P[:2,3]+=np.asarray(fixture_parks[s%len(fixture_parks)])-(v[:,:2].min(0)+v[:,:2].max(0))/2
            if s>=len(fixture_parks):P[0,3]+=.30*(s//len(fixture_parks))
            current[self.keys[s]]=P;self.fixture_parks.append(P.copy())
        self.current=current
        self.initial={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in current.items()}
        self.object_pool=SuctionPool(obj)
        self.fixture_pools=[SuctionPool(f['mesh'],seed=731+s) for s,f in enumerate(fixtures)]
        self.add(1.5,label='establish')
        installed=None
        for s in range(len(fixtures)):
            for row in [r for r in rows if r['set']==s]:
                if installed!=s:
                    if installed is not None:
                        self.move_fixture(installed,self.fixture_parks[installed],label='park')
                    self.move_fixture(s,row['fixture'],label='install');installed=s
                elif not np.allclose(row['fixture'],self.current[self.keys[s]],atol=1e-10,rtol=0):
                    self.move_fixture(s,row['fixture'],label='turn')
                placed=fixtures[s]['mesh'].copy();placed.apply_transform(row['fixture'])
                target=row['object'];outer=translated(target,row['direction']*row['travel'])
                print('PLAN TASK',row['pose'],'set',s+1,flush=True)
                grip=self.choose_object(target,row['work_ids'],placed,outer)
                self.carry_object(self.park,outer,target,grip,row,inbound=True)
                # Hand-over: the work arm moves in only after the loader has left,
                # and swings away before the loader withdraws the bunny.
                self.add(1.2,label='work_arm_approach_'+row['pose'])
                self.add(4.0,task=row['index'],label='task_'+row['pose']);self.order.append(row['index'])
                self.add(1.2,label='work_arm_retreat_'+row['pose'])
                self.carry_object(target,outer,self.park,grip,row,inbound=False)
                self.add(.4,label='object_parked')
        self.add(1.5,label='end');self.duration=self.clock

    def choose_object(self,target,work_ids,placed,outer):
        """Default grasp pool first; a denser pool when the fixture hides most of the bunny."""
        try:return self.object_pool.choose(self.robot,self.park,target,work_ids,placed,transport_target=outer)
        except ValueError as error:
            print('DENSER OBJECT GRASP POOL',str(error)[-120:],flush=True)
            if not hasattr(self,'dense_object_pool'):
                self.dense_object_pool=SuctionPool(self.object_pool.mesh,seed=919,samples=5000,rolls=12,candidate_limit=8000)
            return self.dense_object_pool.choose(self.robot,self.park,target,work_ids,placed,transport_target=outer)

    def add(self,seconds,moving=None,pose=None,hand=None,gap=None,grasp=None,task=None,label='move',preferred_q=None):
        end={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in self.current.items()}
        end['task']=task
        if moving:end[moving]=pose.copy();end['hand']=pose@grasp
        if hand is not None:end['hand']=hand.copy()
        if gap is not None:end['gap']=gap
        self.segments.append(dict(start=self.clock,end=self.clock+seconds,a=self.current,b=end,
            moving=moving,grasp=grasp,label=label,preferred_q=preferred_q))
        self.clock+=seconds;self.current=end

    def acquire(self,key,grip):
        H=self.current[key]@grip['hand'];pre=translated(H,-H[:3,2]*.08)
        at_source=np.allclose(self.current[key],grip['source_pose'])
        self.add(1.0,hand=pre,gap=0.,label='approach_'+key,
            preferred_q=grip['q_pre_source'] if at_source else grip['q_pre_target'])
        self.add(.7,hand=H,label='touch_'+key,preferred_q=grip['q_source'] if at_source else grip['q_target'])
        self.add(.35,gap=0.,label='vacuum_on_'+key)

    def release(self,grip):
        H=self.current['hand'].copy();self.add(.35,gap=0.,label='vacuum_off')
        at_source=np.allclose(H,grip['source_pose']@grip['hand'])
        self.add(.65,hand=translated(H,-H[:3,2]*.08),label='retract',
            preferred_q=grip['q_pre_source'] if at_source else grip['q_pre_target'])
        self.add(.9,hand=self.robot.fk(self.rest),label='clear_work_area',preferred_q=self.rest)

    def move_fixture(self,s,destination,label):
        """Move an empty fixture; split a large turn into regrasped steps.

        Intermediate poses blend the two orientations and rest the fixture's
        lowest point on the bench, where the Panda releases it and takes a
        new grasp.
        """
        key=self.keys[s];first=self.current[key]
        for steps in (1,2,3,4,6):
            poses=[]
            for m in range(1,steps+1):
                if m==steps:P=destination
                else:
                    P=pose_blend(first,destination,m/steps)
                    v=trimesh.transform_points(self.fixtures[s]['mesh'].vertices,P)
                    P=translated(P,[0,0,self.robot.table_z-v[:,2].min()])
                poses.append(P)
            try:
                grips=[];previous=first
                for P in poses:grips.append(self.fixture_pools[s].choose(self.robot,previous,P));previous=P
            except ValueError:
                print('FIXTURE TURN NEEDS MORE REGRASPS',key,steps,flush=True);continue
            break
        else:raise ValueError('Empty fixture turn unreachable with up to 6 regrasps')
        self.regrasps=getattr(self,'regrasps',0)+steps-1;previous=first
        for P,grip in zip(poses,grips):
            self.acquire(key,grip)
            # Exactly the path GraspPool.choose verified: 160 mm above each end.
            A=translated(previous,[0,0,.16]);B=translated(P,[0,0,.16])
            self.add(.85,moving=key,pose=A,grasp=grip['hand'],label='lift_empty_fixture')
            self.add(2.4,moving=key,pose=B,grasp=grip['hand'],label=label+'_empty_fixture')
            self.add(.85,moving=key,pose=P,grasp=grip['hand'],label='seat_fixture')
            self.release(grip);previous=P

    def carry_object(self,source,outer,destination,grip,row,inbound):
        # Joint solutions the grasp check found at park+lift, outer+lift, outer and the seat.
        wp=grip.get('q_waypoints',[None]*4)
        self.acquire('object',grip)
        if not inbound:self.add(.85,moving='object',pose=outer,grasp=grip['hand'],label='withdraw_'+row['pose'],preferred_q=wp[2])
        start=self.current['object'];end=outer if inbound else destination
        A=translated(start,[0,0,.16]);B=translated(end,[0,0,.16])
        self.add(.7,moving='object',pose=A,grasp=grip['hand'],label='lift_object',preferred_q=wp[0] if inbound else wp[1])
        self.add(1.8,moving='object',pose=B,grasp=grip['hand'],label='carry_object',
            preferred_q=grip['q_lift_transport'] if inbound else grip['q_lift_source'])
        self.add(.7,moving='object',pose=end,grasp=grip['hand'],label='lower_object',preferred_q=wp[2] if inbound else grip['q_source'])
        if inbound:self.add(.85,moving='object',pose=destination,grasp=grip['hand'],label='insert_'+row['pose'],preferred_q=wp[3])
        self.release(grip)

    def desired(self,t):
        index=min(np.searchsorted([s['end'] for s in self.segments],t,side='right'),len(self.segments)-1)
        s=self.segments[index];u=smooth((t-s['start'])/(s['end']-s['start']));state={}
        for k in ['object','hand',*self.keys]:state[k]=pose_blend(s['a'][k],s['b'][k],u)
        state.update(gap=(1-u)*s['a']['gap']+u*s['b']['gap'],
            task=s['b']['task'],moving=s['moving'],grasp=s['grasp'],label=s['label'])
        if s['moving']:state['hand']=state[s['moving']]@s['grasp']
        return state

    def prepare(self,rate=12):
        q=self.rest.copy();times=[0.];joints=[q.copy()];errors=[];recoveries=0
        for index,s in enumerate(self.segments):
            first=q.copy();samples=max(2,int(np.ceil((s['end']-s['start'])*rate))+1)
            if not s['moving']:
                last=s['preferred_q']
                if last is None:last=self.robot.ik(s['b']['hand'],q)
                if last is None:raise ValueError('Free-hand endpoint IK failed: '+s['label'])
            transfer=s['moving'] and s['label']=='carry_object'
            if transfer:
                last=np.asarray(s['preferred_q'])
                if np.linalg.norm(self.robot.fk(last)[:3,3]-s['b']['hand'][:3,3])>1e-4:
                    raise ValueError('Transfer waypoint differs from the planned endpoint')
            fallback=None
            for u in np.linspace(0,1,samples)[1:]:
                t=s['start']+u*(s['end']-s['start']);state=self.desired(min(t,s['end']-1e-8))
                if transfer:q=(1-smooth(u))*first+smooth(u)*last
                elif s['moving'] and fallback is None:
                    q_next=self.robot.ik(state['hand'],q,multistart=False)
                    if q_next is None:q_next=self.robot.ik(state['hand'],joints[-1],multistart=True);recoveries+=1
                    if q_next is None:
                        # Straight Cartesian motion unreachable here: finish this
                        # segment in joint space to its end pose (part stays attached).
                        end=self.robot.ik(s['b']['hand'],q,multistart=True)
                        known=s['preferred_q']
                        if end is None and known is not None and np.linalg.norm(self.robot.fk(known)[:3,3]-s['b']['hand'][:3,3])<1e-4:
                            end=np.asarray(known)
                        if end is None:raise ValueError(f'IK path failed at {t:.3f}: {s["label"]}')
                        fallback=(u,q.copy(),end);self.joint_fallbacks=getattr(self,'joint_fallbacks',0)+1
                    else:
                        q=q_next;actual=self.robot.fk(q,state['gap']);errors.append(np.linalg.norm(actual[:3,3]-state['hand'][:3,3]))
                if fallback is not None:
                    u0,q0,end=fallback;v=smooth((u-u0)/max(1-u0,1e-9));q=(1-v)*q0+v*end
                elif not s['moving']:q=(1-smooth(u))*first+smooth(u)*last
                times.append(t);joints.append(q.copy())
            if index%16==0:print('FRANKA IK',round(s['end'],1),'/',round(self.duration,1),flush=True)
        self.times=np.asarray(times);self.joints=np.asarray(joints)
        speed=np.abs(np.diff(self.joints,axis=0))/np.diff(self.times)[:,None]
        return dict(samples=len(times),max_position_error_m=float(max(errors)),multistart_path_recoveries=recoveries,
            joint_space_fallback_segments=getattr(self,'joint_fallbacks',0),
            free_transfer_interpolation='native_joint_space_FK',max_reference_joint_speed_rad_s=float(np.max(speed)),
            joint_limits_passed=True)

    def state(self,t):
        desired=self.desired(t)
        q=np.array([np.interp(t,self.times,self.joints[:,j]) for j in range(7)])
        actual=self.robot.fk(q,desired['gap'])
        if desired['moving']:desired[desired['moving']]=actual@np.linalg.inv(desired['grasp'])
        desired['joints']=q;return desired


class OutlinePath(SurfacePath):
    """Same outline/stroke construction for patches with occluded source faces.

    Step1 counts every face inside the needle sector, including occluded
    ones. Their display stroke cannot pass the first-hit check, recorded below.
    """
    def validate(self):
        for points,faces in [(self.outline,self.outline_faces),(self.interior,self.interior_faces)]:
            close=trimesh.triangles.closest_point(self.mesh.triangles[faces],points)
            if np.linalg.norm(close-points,axis=1).max()>2e-5:raise ValueError('Path leaves selected triangle')

    def info(self):
        return dict(super().info(),path_first_hit_verified=False,occluded_sector_faces_present=True)


def face_components(mesh,ids):
    """Edge-connected pieces of one sector selection; Step1 keeps them together."""
    ids=np.asarray(ids);edges={}
    for f in ids:
        for a,b in zip(mesh.faces[f],np.roll(mesh.faces[f],-1)):edges.setdefault(tuple(sorted((int(a),int(b)))),[]).append(int(f))
    parent={int(f):int(f) for f in ids}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for owners in edges.values():
        for f in owners[1:]:parent[find(f)]=find(owners[0])
    groups={}
    for f in ids:groups.setdefault(find(int(f)),[]).append(int(f))
    return sorted(groups.values(),key=lambda g:-mesh.area_faces[g].sum())


def work_path(mesh,ids,axis,name):
    """Trace the whole selection when possible, else its largest piece."""
    try:return SurfacePath(mesh,ids,axis),'first_hit_verified'
    except ValueError as error:print('SURFACE PATH FALLBACK',name,str(error)[:90],flush=True)
    pieces=face_components(mesh,ids)
    for kind,cls in (('largest_component_first_hit_verified',SurfacePath),('largest_component_selected_faces_only',OutlinePath)):
        try:return cls(mesh,pieces[0],axis),kind+f'_{len(pieces)}_components'
        except ValueError as error:print('SURFACE PATH FALLBACK',name,kind,str(error)[:90],flush=True)
    raise ValueError(name+': no display path on the selected faces')


def park_pose(arm,workflow,rows,axis):
    from scipy.spatial import cKDTree
    robot=workflow.robot;bodies=[robot.model.body(b).id for b in [f'link{i}' for i in range(1,8)]+['hand']]
    clouds=[];obj=workflow.object_pool.mesh.vertices[::25]
    for t in np.linspace(0,workflow.duration,240):
        state=workflow.state(float(t));robot.fk(state['joints'],state['gap'])
        clouds.append(robot.data.xpos[bodies]);O=state['object'];clouds.append(obj@O[:3,:3].T+O[:3,3])
    tree=cKDTree(np.vstack(clouds));home=arm.solve(rows[0]['head']-axis*.11,-axis);best=None
    for a1 in np.radians([-120,-90,-60,60,90,120]):
        for a2 in np.radians([-40,-20,0]):
            for a4 in np.radians([-110,-80,-50]):
                q=home.copy();q[0]+=a1;q[1]=a2;q[3]=a4;q=np.clip(q,arm.lower+.05,arm.upper-.05)
                _,_,transforms=arm.forward(q,geometry=True)
                points=np.vstack([v[::8]@R.T+P for v,(R,P) in zip(arm.collisions[2:],transforms[2:])])
                if points[:,2].min()<arm.base[2]+.05:continue
                clearance=float(tree.query(points)[0].min())
                if best is None or clearance>best[0]:best=(clearance,q)
    print('WORK ARM PARK clearance m',round(best[0],3),flush=True)
    return best[1]


def prepare(folder,out=None,duration=None,fps=24):
    folder=Path(folder).resolve();out=Path(out or folder/'robot_video').resolve();out.mkdir(parents=True,exist_ok=True)
    model,report,rows,fixtures=station(folder);S=video_frame()
    workflow=SetsWorkflow(model,rows,fixtures,PANDA)
    grasp_verification=workflow.prepare(rate=12)
    work_base=S[:3,3]+[0,0,-WORKBENCH_ABOVE_WORK_BASE]
    arm=Arm(work_base,S[:3,:3]@Rotation.from_euler('z',-90,degrees=True).as_matrix(),tool=.155)
    axis=rows[0]['axis'];paths={};retract_q={};arrays={};path_kinds={}
    # While the loader works, park the work arm where it stays farthest from
    # every sampled loader link along the planned motion (bunny included).
    park_q=park_pose(arm,workflow,rows,axis)
    for row in rows:
        k=row['index'];U=row['object'];local_axis=U[:3,:3].T@axis
        path,path_kinds[k]=work_path(model.source,row['work_ids'],local_axis,row['pose'])
        paths[k]=path;start=trimesh.transform_points(path.outline[:1],U)[0]
        q=arm.solve(start,-axis);retract_q[k]=arm.solve(start-axis*.11,-axis,previous=q)
        arrays[f'patch_{k}_ids']=row['work_ids'];arrays[f'patch_{k}_f']=model.source.faces[row['work_ids']]
        for name in ('outline','outline_s','interior','interior_s'):arrays[f'path_{k}_{name}']=getattr(path,name)
    for s,f in enumerate(fixtures):arrays[f'fixture{s}_v']=f['mesh'].vertices;arrays[f'fixture{s}_f']=f['mesh'].faces
    bench=station_bench(rows,fixtures)
    np.savez_compressed(out/'scene_meshes.npz',object_v=model.source.vertices,object_f=model.source.faces,**arrays)
    parts=export_robot_meshes(out,workflow)
    work_seconds=3.2;arm_seconds=1.;handling_scale=.18
    def fixed(seg):
        if seg['b']['task'] is not None:return work_seconds
        return arm_seconds if seg['label'].startswith('work_arm_') else None
    duration=duration or sum(fixed(s) if fixed(s) is not None else (s['end']-s['start'])*handling_scale
                             for s in workflow.segments)
    frames=int(round(duration*fps))
    held=[s for s in workflow.segments if fixed(s) is not None]
    free_scale=(duration-sum(fixed(s) for s in held))/(workflow.duration-sum(s['end']-s['start'] for s in held))
    video_edges=[0.];reference_edges=[0.]
    for seg in workflow.segments:
        step=fixed(seg);video_edges.append(video_edges[-1]+(step if step is not None else (seg['end']-seg['start'])*free_scale))
        reference_edges.append(seg['end'])
    times=np.interp(np.linspace(0,duration,frames),video_edges,reference_edges)
    panda_bodies=[f'link{i}' for i in range(8)]+['hand','left_finger','right_finger']
    keys=workflow.keys;out_arrays={k:[] for k in ('panda_links','kuka_links','object_T','tool_tcp','kuka_q','panda_q',
        'gaps','task','work_face','paint_progress','work_phase','work_task')}
    fixture_T=[];progress=np.zeros(len(rows));previous=park_q.copy();errors=[];deviations=[];finished_q={}
    for t in times:
        state=workflow.state(float(t));task=state['task']
        if task is not None:
            row=rows[int(task)];seg=next(s for s in workflow.segments if s['start']<=t<=s['end'])
            u=float(np.clip((t-seg['start'])/(seg['end']-seg['start']),0,1))
            local_pt,hit_face,phase=paths[row['index']].at(u)
            pt=trimesh.transform_points(local_pt[None],row['object'])[0]
            ease=lambda v:v*v*(3-2*v)
            gap=.11*(1-ease(np.clip(u/.10,0,1))+ease(np.clip((u-.90)/.10,0,1)))
            previous=arm.solve(pt-axis*gap,-axis,previous);finished_q[row['index']]=previous.copy()
            tcp,R,transforms=arm.forward(previous,geometry=True)
            active=row['index'] if phase in (1,2) else -1
            progress[row['index']]=float(np.clip((u-.10)/.80,0,1))
            if active>=0:errors.append(np.linalg.norm(tcp-pt));deviations.append(np.degrees(np.arccos(np.clip(R[:,2]@axis,-1,1))))
            face=hit_face if active>=0 else -1;work_task=row['index']
        else:
            seg=next(s for s in workflow.segments if s['start']<=t<=s['end'])
            u=smooth((t-seg['start'])/(seg['end']-seg['start']))
            if seg['label'].startswith('work_arm_approach_'):
                k=next(r['index'] for r in rows if r['pose']==seg['label'][len('work_arm_approach_'):])
                previous=(1-u)*park_q+u*retract_q[k]
            elif seg['label'].startswith('work_arm_retreat_'):
                k=next(r['index'] for r in rows if r['pose']==seg['label'][len('work_arm_retreat_'):])
                previous=(1-u)*finished_q.get(k,retract_q[k])+u*park_q
            else:previous=park_q.copy()
            tcp,R,transforms=arm.forward(previous,geometry=True);active=-1;phase=-1;face=-1;work_task=-1
        workflow.robot.fk(state['joints'],state['gap']);native=workflow.robot
        out_arrays['panda_links'].append(matrices([(native.data.xmat[native.model.body(b).id].reshape(3,3),
            native.data.xpos[native.model.body(b).id]) for b in panda_bodies]))
        out_arrays['kuka_links'].append(matrices(transforms));out_arrays['object_T'].append(state['object'])
        fixture_T.append([state[k] for k in keys])
        for name,value in (('tool_tcp',tcp),('kuka_q',previous.copy()),('panda_q',state['joints']),('gaps',state['gap']),
                ('task',active),('work_face',face),('paint_progress',progress.copy()),('work_phase',phase),('work_task',work_task)):
            out_arrays[name].append(value)
    np.savez_compressed(out/'trajectory.npz',times=times,fixture_T=np.asarray(fixture_T),**{k:np.asarray(v) for k,v in out_arrays.items()})
    plan=dict(schema='two_robot_fixture_sets_presentation_v1',fps=fps,frames=frames,duration_s=duration,
        reference_duration_s=workflow.duration,table_height_m=TABLE_Z,work_robot_base_world_m=work_base.tolist(),
        presentation_transform=S.tolist(),robot_parts=parts,panda_bodies=panda_bodies,fixture_count=len(fixtures),
        fixture_members=[f['members'] for f in fixtures],fixture_sha256=[f['sha256'] for f in fixtures],
        object_poses=[r['object'].tolist() for r in rows],fixture_poses=[r['fixture'].tolist() for r in rows],
        task_set=[r['set'] for r in rows],task_ids=[r['pose'] for r in rows],needle_drop_m=[r['drop'] for r in rows],
        execution_order=workflow.order,bench=bench,panda_base=PANDA,fixture_regrasps=getattr(workflow,'regrasps',0),
        head_world_m=[r['head'].tolist() for r in rows],work_axis_world=axis.tolist(),
        grasp_verification=grasp_verification,max_work_tcp_error_m=float(max(errors)),
        max_work_axis_deviation_deg=float(max(deviations)),work_tool_length_from_link7_m=.155,
        work_path_kinds={rows[k]['pose']:v for k,v in path_kinds.items()},
        loader='Franka Panda with a vacuum cup on a 60 mm rod: installs/turns empty fixtures (regrasping on the bench when a turn is too large) and inserts/withdraws the bunny along the saved exit direction',
        loader_cup_rod_m=ROD,
        worker='KUKA LBR Med 14 R820: fixed needle axis, translation only, saved working area traced',
        station='fixed floor: every saved tangent ground plane is the bench top; the KUKA translates the needle head to each installation height, needle direction fixed',
        playback=f'{work_seconds}s per work dwell; handling accelerated',
        full_arm_swept_collision_certified=False,playback_joint_speeds_certified=False,
        sources={str(p):sha256(p) for p in [folder/'report.json',*[Path(f['file']) for f in fixtures],
            ROOT/'robots_config/04_kuka_lbr_med14_r820.json',Path(__file__)]})
    write_json(out/'plan.json',plan)
    write_json(out/'segments.json',[dict(start=s['start'],end=s['end'],label=s['label'],moving=s['moving'],
        task=s['b']['task'],grasp=None if s['grasp'] is None else s['grasp'].tolist()) for s in workflow.segments])
    print('ROBOT VIDEO PLAN',frames,'frames; work TCP error',plan['max_work_tcp_error_m'],flush=True)
    return plan


def clearance(out):
    """Sampled diagnostic: KUKA links and needle against the installed fixture."""
    out=Path(out);plan=json.loads((out/'plan.json').read_text())
    with np.load(out/'trajectory.npz') as z:t={k:z[k] for k in z.files}
    with np.load(out/'scene_meshes.npz') as z:g={k:z[k] for k in z.files}
    arm=Arm(plan['work_robot_base_world_m'],np.eye(3),tool=.155)
    fixtures=[trimesh.Trimesh(g[f'fixture{s}_v'],g[f'fixture{s}_f'],process=False) for s in range(plan['fixture_count'])]
    hits={}
    for f in np.flatnonzero(t['work_task']>=0)[::3]:
        k=int(t['work_task'][f]);s=plan['task_set'][k];T=np.linalg.inv(t['fixture_T'][f,s])
        points=[]
        for v,(r,p) in zip(arm.collisions[2:],[(m[:3,:3],m[:3,3]) for m in t['kuka_links'][f][2:]]):points.append(v[::4]@r.T+p)
        R=t['kuka_links'][f][-1][:3,:3];tcp=t['tool_tcp'][f]
        points.append(tcp-np.outer(np.linspace(.0,.12,25),R[:,2]))
        inside=fixtures[s].contains(trimesh.transform_points(np.vstack(points),T))
        hits.setdefault(plan['task_ids'][k],[0,0]);hits[plan['task_ids'][k]][0]+=int(inside.any());hits[plan['task_ids'][k]][1]+=1
    # The needle is lowered towards the fixed bench, so also sample the arm against it.
    lo=np.asarray(plan['bench']['center'])-np.asarray(plan['bench']['size'])/2;hi=lo+np.asarray(plan['bench']['size'])
    bench=[0,0]
    for f in range(0,plan['frames'],3):
        points=[v[::4]@m[:3,:3].T+m[:3,3] for v,m in zip(arm.collisions[2:],t['kuka_links'][f][2:])]
        R=t['kuka_links'][f][-1][:3,:3];points.append(t['tool_tcp'][f]-np.outer(np.linspace(.0,.12,25),R[:,2]))
        p=np.vstack(points);below=(p[:,2]<plan['table_height_m'])&np.all((p[:,:2]>lo)&(p[:,:2]<hi),axis=1)
        bench[0]+=int(below.any());bench[1]+=1
    result=dict(schema='sampled_work_arm_fixture_clearance_v2',sampled_work_frames_with_penetration=hits,
        sampled_frames_with_work_arm_in_bench=bench,
        method='every third work frame: KUKA collision-mesh vertices and needle centreline points inside the fixture; '
               'every third frame: the same points inside the bench box',
        certificate=False)
    write_json(out/'work_arm_clearance.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path)
    p.add_argument('--out',type=Path);p.add_argument('--duration',type=float)
    a=p.parse_args();prepare(a.folder,a.out,a.duration);clearance(a.out or a.folder/'robot_video')


def make(folder,video_path,fps=24):
    """Plan, diagnose and rasterize the two-robot use of a saved Step5 result."""
    import subprocess
    from .sets_raster import render
    folder=Path(folder);out=folder/'robot_video';video_path=Path(video_path)
    plan=prepare(folder,out,fps=fps);diagnostic=clearance(out);raster=render(out,video_path)
    probe=subprocess.run(['ffprobe','-v','error','-count_frames','-show_entries','stream=nb_read_frames',
        '-of','json',str(video_path)],check=True,capture_output=True,text=True)
    frames=int(json.loads(probe.stdout)['streams'][0]['nb_read_frames'])
    if frames!=plan['frames']:raise ValueError('Incomplete two-robot movie')
    return dict(file=str(video_path.resolve()),sha256=sha256(video_path),frames=frames,fps=fps,
        duration_s=plan['duration_s'],fixture_count=plan['fixture_count'],execution_order=[plan['task_ids'][k] for k in plan['execution_order']],
        plan=str(out/'plan.json'),plan_sha256=sha256(out/'plan.json'),trajectory_sha256=sha256(out/'trajectory.npz'),
        max_work_tcp_error_m=plan['max_work_tcp_error_m'],work_path_kinds=plan['work_path_kinds'],
        sampled_work_arm_fixture_penetration=diagnostic['sampled_work_frames_with_penetration'],
        sampled_frames_with_work_arm_in_bench=diagnostic['sampled_frames_with_work_arm_in_bench'],
        needle_drop_m=plan['needle_drop_m'],
        loader=plan['loader'],worker=plan['worker'],station=plan['station'],
        full_arm_swept_collision_certified=False,raster=raster)
