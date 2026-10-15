"""One Panda loads every task, then turns only the empty shared fixture."""
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from franka import Franka
from grasps import GraspPool,pose_blend,translated


def smooth(u):
    u=float(np.clip(u,0,1));return u*u*u*(10-15*u+6*u*u)


class Workflow:
    def __init__(self,data):
        self.data=data;self.robot=Franka();self.segments=[];self.clock=0.;self.story=[]
        obj=data['object']
        parks,prob=obj.compute_stable_poses(sigma=0,n_samples=1,threshold=0)
        park=parks[int(np.argmax(prob))].copy()
        v=trimesh.transform_points(obj.vertices,park)
        park[:2,3]+=np.array([.46,.30])-(v[:,:2].min(0)+v[:,:2].max(0))/2
        park[2,3]+=.24-v[:,2].min()
        self.park=park
        first=data['rows'][data['order'][0]]
        self.current=dict(object=park.copy(),fixture=translated(first['fixture'],[0,-.28,0]),
            hand=self.robot.fk(self.robot.home),gap=.08,task=None)
        self.initial={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in self.current.items()}
        options=data.get('grasp_options',{})
        object_pool=GraspPool(obj,**options)
        fixture_pool=GraspPool(data['fixture'],seed=731,**options)
        self.add(1.5,label='establish')
        self.move_fixture(first['fixture'],fixture_pool)
        host=first['host']
        for i in data['order']:
            row=data['rows'][i]
            if row['host']!=host:
                self.move_fixture(row['fixture'],fixture_pool);host=row['host']
            target=row['object'];outer=translated(target,row['direction']*row['travel'])
            placed=data['fixture'].copy();placed.apply_transform(row['fixture'])
            print('PLAN TASK',data['label'],row['pose'],flush=True)
            grip=object_pool.choose(self.robot,self.park,target,row['work_ids'],placed,transport_target=outer)
            self.carry_object(self.park,outer,target,grip,row,inbound=True)
            self.add(2.2,task=i,label='task_'+row['pose'])
            self.story.append(self.clock-1.)
            self.carry_object(target,outer,self.park,grip,row,inbound=False)
            self.add(.4,label='object_parked')
            print('PLANNED',data['label'],row['pose'],row['role'],flush=True)
        self.add(1.5,label='end')
        self.duration=self.clock

    def add(self,seconds,moving=None,pose=None,hand=None,gap=None,grasp=None,task=None,label='move',preferred_q=None):
        end={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in self.current.items()}
        end['task']=task
        if moving:end[moving]=pose.copy();end['hand']=pose@grasp
        if hand is not None:end['hand']=hand.copy()
        if gap is not None:end['gap']=gap
        self.segments.append(dict(start=self.clock,end=self.clock+seconds,a=self.current,b=end,
            moving=moving,grasp=grasp,label=label,preferred_q=preferred_q))
        self.clock+=seconds;self.current=end

    def acquire(self,body,grip):
        H=self.current[body]@grip['hand']
        pre=translated(H,-H[:3,2]*.08)
        at_source=np.allclose(self.current[body],grip['source_pose'])
        self.add(1.0,hand=pre,gap=.08,label='approach_'+body,
            preferred_q=grip['q_pre_source'] if at_source else grip['q_pre_target'])
        preferred=grip['q_source'] if at_source else grip['q_target']
        self.add(.7,hand=H,label='touch_'+body,preferred_q=preferred)
        self.add(.35,gap=grip['gap'],label='close_'+body)

    def release(self,grip):
        H=self.current['hand'].copy()
        self.add(.35,gap=.08,label='open')
        at_source=np.allclose(H,grip['source_pose']@grip['hand'])
        self.add(.65,hand=translated(H,-H[:3,2]*.08),label='retract',
            preferred_q=grip['q_pre_source'] if at_source else grip['q_pre_target'])
        self.add(.9,hand=self.robot.fk(self.robot.home),label='clear_work_area')

    def move_fixture(self,destination,pool):
        first=self.current['fixture'];grip=pool.choose(self.robot,first,destination)
        self.acquire('fixture',grip)
        A=translated(first,[0,0,.22]);B=translated(destination,[0,0,.22])
        self.add(.85,moving='fixture',pose=A,grasp=grip['hand'],label='lift_empty_fixture')
        self.add(2.4,moving='fixture',pose=B,grasp=grip['hand'],label='turn_empty_fixture')
        self.add(.85,moving='fixture',pose=destination,grasp=grip['hand'],label='seat_fixture')
        self.release(grip)

    def carry_object(self,source,outer,destination,grip,row,inbound):
        self.acquire('object',grip)
        if not inbound:self.add(.85,moving='object',pose=outer,grasp=grip['hand'],label='withdraw_'+row['pose'])
        start=self.current['object'];end=outer if inbound else destination
        A=translated(start,[0,0,.16]);B=translated(end,[0,0,.16])
        self.add(.7,moving='object',pose=A,grasp=grip['hand'],label='lift_object')
        self.add(1.8,moving='object',pose=B,grasp=grip['hand'],label='carry_object')
        self.add(.7,moving='object',pose=end,grasp=grip['hand'],label='lower_object')
        if inbound:self.add(.85,moving='object',pose=destination,grasp=grip['hand'],label='insert_'+row['pose'])
        self.release(grip)

    def desired(self,t):
        index=min(np.searchsorted([s['end'] for s in self.segments],t,side='right'),len(self.segments)-1)
        s=self.segments[index];u=smooth((t-s['start'])/(s['end']-s['start']))
        state={}
        for k in ['object','fixture','hand']:
            state[k]=pose_blend(s['a'][k],s['b'][k],u)
        state.update(gap=(1-u)*s['a']['gap']+u*s['b']['gap'],task=s['b']['task'],moving=s['moving'],
            grasp=s['grasp'],label=s['label'])
        if s['moving']:state['hand']=state[s['moving']]@s['grasp']
        return state

    def prepare(self,rate=12):
        q=self.robot.home.copy();times=[0.];joints=[q.copy()];errors=[]
        for index,s in enumerate(self.segments):
            first=q.copy();samples=max(2,int(np.ceil((s['end']-s['start'])*rate))+1)
            if not s['moving']:
                last=s['preferred_q']
                if last is None:last=self.robot.ik(s['b']['hand'],q)
                if last is None:raise ValueError('Free-hand endpoint IK failed: '+s['label'])
            for u in np.linspace(0,1,samples)[1:]:
                t=s['start']+u*(s['end']-s['start']);state=self.desired(min(t,s['end']-1e-8))
                if s['moving']:
                    q=self.robot.ik(state['hand'],q,multistart=False)
                    if q is None:raise ValueError(f'IK path failed at {t:.3f}: {s["label"]}')
                    actual=self.robot.fk(q,state['gap']);errors.append(np.linalg.norm(actual[:3,3]-state['hand'][:3,3]))
                else:q=(1-smooth(u))*first+smooth(u)*last
                times.append(t);joints.append(q.copy())
            if index%16==0:print('FRANKA IK',round(s['end'],1),'/',round(self.duration,1),flush=True)
        self.times=np.asarray(times);self.joints=np.asarray(joints);times=self.times
        speed=np.abs(np.diff(self.joints,axis=0))/np.diff(times)[:,None]
        self.time_scale=max(1.,float(np.max(speed))/.95)
        self.ik_max_error=float(max(errors))
        return dict(samples=len(times),max_position_error_m=self.ik_max_error,
            max_reference_joint_speed_rad_s=float(np.max(speed)),retiming_scale=self.time_scale,
            conservative_presentation_speed_limit_rad_s=.95,joint_limits_passed=True)

    def state(self,t):
        desired=self.desired(t)
        q=np.array([np.interp(t,self.times,self.joints[:,j]) for j in range(7)])
        actual=self.robot.fk(q,desired['gap'])
        if desired['moving']:
            desired[desired['moving']]=actual@np.linalg.inv(desired['grasp'])
        desired['joints']=q
        return desired
