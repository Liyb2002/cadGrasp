"""Generate sequential one-arm concept motion and the existing KUKA visual meshes.
Kinematic illustration only: grasps and insertion paths are not certified.
"""
from pathlib import Path
import sys,json
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from scipy.optimize import least_squares
import trimesh
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))
from codes.simulation import kuka
s=(OUT/'data.js').read_text();D=json.loads(s[s.index('=')+1:].rstrip(';\n'))
def T(r=np.eye(3),p=np.zeros(3)):
    t=np.eye(4);t[:3,:3]=r;t[:3,3]=p;return t
def shift(t,d):
    v=t.copy();v[:3,3]+=d;return v
def blend(a,b,u):
    u=np.clip(u,0,1);u=float(np.clip(u*u*u*(10-15*u+6*u*u),0,1))
    return T(Slerp([0,1],Rotation.from_matrix([a[:3,:3],b[:3,:3]]))([u]).as_matrix()[0],(1-u)*a[:3,3]+u*b[:3,3])
def tool(point,normal,closing):
    z=-np.array(normal,dtype=float);z/=np.linalg.norm(z)
    x=np.array(closing,dtype=float);x-=z*(z@x);x/=np.linalg.norm(x)
    return T(np.column_stack([x,np.cross(z,x),z]),point)
OP=[];FP=[];stage_yaws=[]
# The opening stays along world +Y. Rotate each complete task assembly about
# vertical, preserving its object/fixture relationship and height above ground.
# All tasks use the same station; only the fixture's roll changes between them.
stage_offsets=np.tile([.03,0.,0.],(len(D['poses']),1))
opening_direction=np.array([0.,1.,0.])
for k,p in enumerate(D['poses']):
    v=np.array(p['object']['v']).reshape(-1,3)
    source_fixture=np.asarray(p['fixtureR']);pivot=np.asarray(p['fixtureT'])
    opening=-source_fixture[:,0]
    yaw=np.pi/2-np.arctan2(opening[1],opening[0]);stage_yaws.append(yaw)
    turn=Rotation.from_euler('z',yaw).as_matrix()
    OP.append(T(turn@np.asarray(p['objectR']),turn@(v.mean(0)-pivot)+pivot+stage_offsets[k]))
    FP.append(T(turn@source_fixture,pivot+stage_offsets[k]))
    assert np.allclose(-FP[-1][:3,0],opening_direction,atol=1e-10)
    original_relative=np.linalg.inv(T(source_fixture,pivot))@T(np.asarray(p['objectR']),v.mean(0))
    assert np.allclose(np.linalg.inv(FP[-1])@OP[-1],original_relative,atol=1e-10)
source_vertices=np.asarray(D['poses'][0]['object']['v']).reshape(-1,3)
v=(source_vertices-source_vertices.mean(0))@np.asarray(D['poses'][0]['objectR'])
fixture_vertices=np.asarray(D['fixture']['v']).reshape(-1,3)
withdrawal_distances=[]
for pose in D['poses']:
    local=(np.asarray(pose['object']['v']).reshape(-1,3)-pose['fixtureT'])@np.asarray(pose['fixtureR'])
    withdrawal_distances.append(float(local[:,0].max()-fixture_vertices[:,0].min()+.012))
m=trimesh.Trimesh(vertices=v,faces=np.array(D['poses'][0]['object']['f']).reshape(-1,3),process=False)
def object_grasp(direction):
    normal=np.asarray(direction,dtype=float);normal/=np.linalg.norm(normal)
    hits,_,_=m.ray.intersects_location([m.center_mass],[normal],multiple_hits=True)
    point=min(hits,key=lambda p:np.linalg.norm(p-m.center_mass))-normal*.015
    first=np.cross([0,0,1],normal);first/=np.linalg.norm(first);second=np.cross(normal,first)
    choices=[]
    for angle in np.linspace(0,np.pi,60,endpoint=False):
        axis=first*np.cos(angle)+second*np.sin(angle)
        hit,rays,_=m.ray.intersects_location([point,point],[axis,-axis],multiple_hits=True)
        if len(np.unique(rays))!=2:continue
        ds=[min(np.linalg.norm(h-point) for h in hit[rays==i]) for i in range(2)]
        gap=sum(ds)/2+.004
        if .01<gap<.065:choices.append((gap,point+axis*(ds[0]-ds[1])/2,axis))
    gap,point,axis=min(choices,key=lambda x:x[0])
    return gap,tool(point,normal,axis)
# Regrasp at the parking station. Each task uses a surface approach that remains
# above the part in both its parked and assembled orientation.
object_grasps=[object_grasp(n) for n in ([.9,-.05,.35],[-.05,-.35,.94],[.9,-.05,.35])]
normal=np.array([.9,-.05,.35]);normal/=np.linalg.norm(normal)
parks,probs=m.compute_stable_poses(sigma=0,n_samples=1,threshold=0)
options=[(prob,t) for prob,t in zip(probs,parks) if (t[:3,:3]@normal)[2]>.25]
PARK=max(options,key=lambda x:x[0])[1].copy()
outward=PARK[:3,:3]@normal
PARK=T(Rotation.from_euler('z',-np.arctan2(outward[1],outward[0])+.3).as_matrix())@PARK
pv=trimesh.transform_points(v,PARK)
PARK[:2,3]+=np.array([.07,.27])-(pv[:,:2].min(0)+pv[:,:2].max(0))/2
PARK[2,3]-=pv[:,2].min()
HOME=tool([.10,.10,.50],[0,0,1],[1,0,0])
current=dict(o=PARK.copy(),f=FP[0].copy(),h=HOME.copy(),gap=.072,k=0)
segments=[];clock=0.;carry_groups=[]
def add(seconds,moving=None,pose=None,hand=None,g=None,k=None,grasp=None,horizontal=False):
    global clock,current
    end={key:value.copy() if isinstance(value,np.ndarray) else value for key,value in current.items()}
    if moving:
        end[moving]=pose.copy();end['h']=pose@grasp
    if hand is not None:end['h']=hand.copy()
    if g is not None:end['gap']=g
    if k is not None:end['k']=k
    segments.append(dict(start=clock,end=clock+seconds,a=current,b=end,moving=moving,grasp=grasp,horizontal=horizontal))
    clock+=seconds;current=end

def carry(body,dest,k,initial=False):
    first_segment=len(segments)
    # Grab exposed rear rails from a tilted approach that remains accessible
    # at both ends of the flip, keeping the hand away from the object opening.
    rear_point=[.178,.04,.074] if k==0 else [.178,-.074,.04]
    outward=[1,-1,1] if k==0 else [1,-1,-1]
    grasp=object_grasps[k][1] if body=='o' else tool(rear_point,outward,[1,0,0])
    closed=object_grasps[k][0] if body=='o' else .018
    source=current[body];h=source@grasp
    pre=tool(h[:3,3]+[0,0,.13],[0,0,1],[1,0,0])
    add(1.1,hand=pre,g=.072);add(.8,hand=h);add(.35,g=closed)
    # Remove horizontally first; adjust height only after clearing the opening.
    seated_source=body=='o' and not initial and not np.allclose(source,PARK)
    seated_dest=body=='o' and not np.allclose(dest,PARK)
    source_axis=source[:3,0] if body=='f' else FP[k][:3,0]
    dest_axis=dest[:3,0] if body=='f' else FP[k][:3,0]
    travel=withdrawal_distances[k]
    source_outer=shift(source,-source_axis*travel) if seated_source else source
    dest_outer=shift(dest,-dest_axis*travel) if seated_dest else dest
    if seated_source:add(1.,moving=body,pose=source_outer,grasp=grasp,horizontal=True)
    lift=.12 if body=='f' else .18
    high=shift(source_outer,[0,0,lift]);desthigh=shift(dest_outer,[0,0,lift])
    add(.9,moving=body,pose=high,grasp=grasp)
    airborne=clock+.65
    add(3.0,moving=body,pose=desthigh,k=k,grasp=grasp)
    add(.9,moving=body,pose=dest_outer,grasp=grasp)
    if seated_dest:add(1.,moving=body,pose=dest,grasp=grasp,horizontal=True)
    seated=clock-.05
    add(.35,g=.072)
    last=dest@grasp
    # Once released, return the empty hand to an upright posture above the
    # part instead of extending the loaded wrist orientation toward its limit.
    add(1.,hand=tool(last[:3,3]+[0,0,.13],[0,0,1],[1,0,0]))
    carry_groups.append(dict(body=body,rows=segments[first_segment:],grasp=grasp))
    return dict(airborne=airborne,seated=seated)

add(2.)
stamps=[1.]
opening_times=dict(fixture_ready=0.,separated=1.)
moments=carry('o',OP[0],0);opening_times['object_inserted']=moments['seated']
add(2.);stamps.append(clock-1)
for k in range(2):
    moments=carry('o',PARK,k);stamps.append(moments['airborne'])
    moments=carry('f',FP[k+1],k);stamps.append(moments['airborne'])
    moments=carry('o',OP[k+1],k+1);stamps.append(moments['seated'])
    add(2.)
    if k==0:stamps[-1]=clock-1
    else:stamps.append(clock-1)

end_motion=clock
add(2.)

def state(t):
    row=next((r for r in segments if t<=r['end']+1e-8),segments[-1]);a,b=row['a'],row['b']
    u=np.clip((t-row['start'])/(row['end']-row['start']),0,1)
    result={key:blend(a[key],b[key],u) for key in ('o','f','h')}
    # Turn through the shortest rigid rotation while lifted. The floor-level
    # object insertion/withdrawal segments keep their fixed horizontal axes.
    if row['moving']=='o':
        hand=result['o']@row['grasp'];flange=hand[:3,3]-hand[:3,2]*kuka.TOOL
        radial=flange[:2]-arm.base[:2];distance=np.linalg.norm(radial)
        height=float((v@result['o'][:3,:3].T+result['o'][:3,3])[:,2].min())
        weight=float(np.clip((height-.015)/.08,0,1));weight=weight*weight*(3-2*weight)
        result['o'][:2,3]+=radial*(max(distance,.55)/distance-1)*weight
    if row['moving']:result['h']=result[row['moving']]@row['grasp']
    result['gap']=a['gap']+(b['gap']-a['gap'])*u*u*u*(10-15*u+6*u*u);result['k']=b['k']
    return result
arm=kuka.Arm('B');arm.base=np.array([.60,0.,0.]);arm.upper[3]=-.08
links,joints,_=kuka.description()
robot=[]
for i,link in enumerate(links):
    offset=np.fromstring(link.find('visual/origin').get('xyz'),sep=' ')
    parts=[]
    for vertices,faces,rgba in kuka.visual_parts(i):
        if i in (2,4) and np.allclose(rgba[:3],.2):rgba=np.array([.6,.6,.6,1.])
        parts.append(dict(v=np.round(vertices+offset,6).ravel().tolist(),f=faces.ravel().tolist(),color=rgba[:3].tolist()))
    robot.append(parts)

def serialize(t):return dict(p=np.round(t[:3,3],7).tolist(),q=np.round(Rotation.from_matrix(t[:3,:3]).as_quat(),8).tolist())
# Segment boundaries are included so the jaw opens only after the part stops.
times=np.unique(np.round(np.r_[np.arange(0,clock,.0625),clock,[r['end'] for r in segments]],6))
frames=[];prev=None;worst=0.;worst_angle=0.;jumps=[]
def heights(q):
    _,_,transforms=arm.forward(q,geometry=True)
    return np.array([(vertices@r.T+point)[:,2].min() for vertices,(r,point) in zip(arm.collisions[1:],transforms[1:])])

def solve(goal,seed,max_step=None,retry=True):
    seed=np.clip(seed,arm.lower+1e-7,arm.upper-1e-7)
    lower,upper=arm.lower.copy(),arm.upper.copy()
    if max_step is not None:lower=np.maximum(lower,seed-max_step);upper=np.minimum(upper,seed+max_step)
    def residual(q):
        p,r,_=arm.forward(q,geometry=True)
        mid=(arm.lower+arm.upper)/2;half=(arm.upper-arm.lower)/2
        margin=.004*np.maximum(np.abs((q-mid)/half)-.72,0)
        return np.r_[p-goal[:3,3],.15*Rotation.from_matrix(goal[:3,:3]@r.T).as_rotvec(),.00015*(q-seed),margin,3*np.minimum(heights(q)-.003,0)]
    fit=least_squares(residual,seed,bounds=(lower,upper),max_nfev=180,ftol=1e-9,xtol=1e-9,gtol=1e-9)
    if np.linalg.norm(fit.fun[:6])>.00035 and max_step is None and retry:
        for extra in [np.zeros(7),*np.random.default_rng(31).uniform(arm.lower*.8,arm.upper*.8,(12,7))]:
            extra=np.clip(extra,arm.lower+1e-6,arm.upper-1e-6)
            candidate=least_squares(residual,extra,bounds=(arm.lower,arm.upper),max_nfev=180)
            if np.linalg.norm(candidate.fun)<np.linalg.norm(fit.fun):fit=candidate
    if np.linalg.norm(fit.fun[:6])>.0005:raise RuntimeError(f'IK target is unreachable: {goal}; error {fit.fun[:6]}; q={np.rad2deg(fit.x)}; seed={np.rad2deg(seed)}; heights={heights(fit.x)}')
    return fit.x

def solve_upright(point,seed):
    """Empty jaws may turn freely around vertical; preserve the nearby posture."""
    seed=np.clip(seed,arm.lower+1e-7,arm.upper-1e-7)
    def residual(q):
        p,r,_=arm.forward(q,geometry=True)
        return np.r_[p-point,.15*(r[:,2]-[0,0,-1]),.001*(q-seed),3*np.minimum(heights(q)-.003,0)]
    fit=least_squares(residual,seed,bounds=(arm.lower,arm.upper),max_nfev=180,ftol=1e-9,xtol=1e-9,gtol=1e-9)
    if np.linalg.norm(fit.fun[:6])>.0005:raise RuntimeError('No nearby upright empty-hand posture')
    return fit.x

# Rank complete carrying motions, including the parallel jaw's two equivalent
# orientations. Prefer a relaxed wrist and little joint travel over merely
# finding the first reachable start/end pose.
posture_records=[];last_end=np.deg2rad([0,55,0,-95,0,60,0])
for gi,entry in enumerate(carry_groups):
    group=entry['rows'];moving=[row for row in group if row['moving']]
    probe_times=np.concatenate([np.linspace(row['start'],row['end'],7)[1:] for row in moving])
    original=[row['b']['h'].copy() for row in group]
    original_grasp=entry['grasp'].copy()
    seeds=[last_end,np.deg2rad([0,55,0,-85,0,45,0]),np.deg2rad([35,65,-40,-85,45,40,-45]),np.deg2rad([-35,65,40,-85,-45,40,45])]
    seeds.append(np.random.default_rng(15).uniform(arm.lower*.6,arm.upper*.6,(2,7))[1])
    seeds.extend(np.random.default_rng(260926+gi).uniform(arm.lower*.65,arm.upper*.65,(4,7)))
    candidates=[]
    for flip in (0,1):
        symmetry=T(Rotation.from_euler('z',flip*np.pi).as_matrix())
        for row,h in zip(group,original):
            row['b']['h']=h@symmetry
            if row['moving']:row['grasp']=original_grasp@symmetry
        for si,seed in enumerate(seeds):
            try:
                initial_q=solve(group[1]['b']['h'],seed,retry=False);q=initial_q.copy();path=[q]
                for tm in probe_times:
                    q=solve(state(float(tm))['h'],q,max_step=1.1,retry=False);path.append(q)
                pre_q=solve_upright(group[0]['b']['h'][:3,3],initial_q)
                post_q=solve_upright(group[-1]['b']['h'][:3,3],q)
                path=np.array(path);travel=np.abs(np.diff(path,axis=0)).sum(0)
                wrist=float(np.max(np.abs(path[:,[4,6]])))
                weights=np.array([1.,1.,1.2,1.,1.8,1.5,1.8])
                empty_travel=np.abs(pre_q-last_end)+np.abs(initial_q-pre_q)+np.abs(post_q-q)
                score=float((travel+.8*empty_travel)@weights+1.5*wrist+3.*np.maximum(np.abs(path[:,[4,6]])-1.6,0).mean())
                candidates.append((score,flip,initial_q,post_q,wrist,travel.sum(),path,pre_q))
            except RuntimeError:
                continue
    if not candidates:raise RuntimeError(f'No continuous carrying posture found for transfer {gi+1} ({entry["body"]})')
    score,flip,chosen,last_end,wrist,travel,guide,pre_q=min(candidates,key=lambda item:item[0])
    symmetry=T(Rotation.from_euler('z',flip*np.pi).as_matrix())
    for row,h in zip(group,original):
        row['b']['h']=h@symmetry
        if row['moving']:
            row['grasp']=original_grasp@symmetry
            row['guide_times']=np.r_[group[1]['end'],probe_times]
            row['guide_joints']=guide
    group[0]['pickup_seed']=chosen;group[1]['pickup_seed']=chosen
    for row,q in ((group[0],pre_q),(group[1],chosen),(group[-1],last_end)):
        row['endpoint_q']=q.copy()
        p,r,_=arm.forward(q,geometry=True);row['b']['h']=T(r,p)
    # Refresh shared boundary hand poses after selecting a jaw orientation.
    for previous,following in zip(segments,segments[1:]):following['a']['h']=previous['b']['h'].copy()
    posture_records.append(dict(body=entry['body'],jaw_half_turn=bool(flip),feasible_candidates=len(candidates),max_wrist_angle_deg=float(np.rad2deg(wrist)),joint_travel_deg=float(np.rad2deg(travel))))
    print(f'Transfer {gi+1} ({entry["body"]}): {len(candidates)} candidates, wrist {np.rad2deg(wrist):.0f} deg, travel {np.rad2deg(travel):.0f} deg.',flush=True)
for i,tm in enumerate(times):
    st=state(tm);goal=st['h']
    row=next((r for r in segments if tm<=r['end']+1e-8),segments[-1])
    if prev is None:prev=solve(goal,np.deg2rad([0,55,0,-95,0,90,0]))
    if row['moving'] is None:
        # Unloaded moves return through joint space, avoiding folded-arm dead zones.
        if 'qa' not in row:
            # Preserve the wrist branch used by the subsequent carrying path.
            seed=row.get('pickup_seed',prev)
            row['qa']=prev.copy()
            if 'endpoint_q' in row:row['qb']=row['endpoint_q'].copy()
            elif np.allclose(row['a']['h'],row['b']['h'],atol=1e-6):row['qb']=prev.copy()
            else:row['qb']=solve(row['b']['h'],seed)
        u=np.clip((tm-row['start'])/(row['end']-row['start']),0,1);u=float(np.clip(u*u*u*(10-15*u+6*u*u),0,1))
        q=(1-u)*row['qa']+u*row['qb']
        if heights(q).min()<.003:
            nominal=q.copy()
            correction=least_squares(lambda v:np.r_[.03*(v-nominal),10*np.minimum(heights(v)-.003,0)],q,bounds=(arm.lower,arm.upper),max_nfev=100)
            q=correction.x
    else:
        seed=np.array([np.interp(tm,row['guide_times'],row['guide_joints'][:,j]) for j in range(7)])
        q=solve(goal,seed,max_step=.45)
    jump=float(np.max(np.abs(q-prev)));jumps.append(jump)
    if jump>.6 and row['moving']:print(f'Carried branch change {tm:.3f}s: {jump:.3f}',flush=True)
    prev=q
    p,r,transforms=arm.forward(q,geometry=True)
    if heights(q).min()<-.0005:raise RuntimeError(f'Robot enters floor at {tm}: {heights(q).min()}')
    if row['moving']:
        pe=float(np.linalg.norm(p-goal[:3,3]));ae=float(np.linalg.norm(Rotation.from_matrix(goal[:3,:3]@r.T).as_rotvec()))
        worst=max(worst,pe);worst_angle=max(worst_angle,ae)
        if pe>.001 or ae>.01:raise RuntimeError(f'IK misses grasp at {tm}: {pe}m {ae}rad')
    frames.append(dict(t=round(float(tm),6),o=serialize(st['o']),f=serialize(st['f']),links=[serialize(T(r,p)) for r,p in transforms],joints=q.tolist(),gap=st['gap'],k=st['k']))
    if i%80==0:print(f'Robot motion {tm:.1f}/{clock:.1f}s',flush=True)
# Time each segment from its actual joint motion; keep the held part synchronized.
raw_times=np.array([f['t'] for f in frames]);qs=np.array([f['joints'] for f in frames])
PLAYBACK_SPEED=2.0
retimed=0.;speed_limit=.85;accel_limit=2.;max_speed=0.;max_accel=0.
for row in segments:
    grid=np.linspace(row['start'],row['end'],max(5,int((row['end']-row['start'])*16)+1))
    qg=np.column_stack([np.interp(grid,raw_times,qs[:,j]) for j in range(7)])
    vel=np.diff(qg,axis=0)/np.diff(grid)[:,None]
    acc=np.diff(vel,axis=0)/((np.diff(grid)[:-1]+np.diff(grid)[1:])[:,None]/2)
    sp=float(np.max(np.abs(vel)));ac=float(np.max(np.abs(acc)))
    # Apply presentation speed after smoothing the underlying robot motion.
    scale=max(1.,sp/speed_limit,np.sqrt(ac/accel_limit))*1.03/PLAYBACK_SPEED
    row['new_start']=retimed;row['scale']=scale
    retimed+=(row['end']-row['start'])*scale;row['new_end']=retimed
    max_speed=max(max_speed,sp/scale);max_accel=max(max_accel,ac/scale**2)
def retime(t):
    row=next((r for r in segments if t<=r['end']+1e-7),segments[-1])
    return row['new_start']+(t-row['start'])*row['scale']
for f in frames:f['t']=round(retime(f['t']),7)
stamps=[retime(t) for t in stamps];end_motion=retime(end_motion);clock=retimed
horizontal_rows=[]
for row in segments:
    if not row['horizontal']:continue
    body=row['moving'];a,b=row['a'][body],row['b'][body]
    assert abs(a[2,3]-b[2,3])<1e-10 and np.allclose(a[:3,:3],b[:3,:3])
    horizontal_rows.append(dict(start=row['new_start'],end=row['new_end'],body=body,travel_m=float(np.linalg.norm(a[:3,3]-b[:3,3]))))
# Check the same joint interpolation used by the 24 fps viewer.
frame_times=np.array([f['t'] for f in frames]);minimum_clearance=1.
for t in np.arange(0,clock,1/24):
    q=np.array([np.interp(t,frame_times,qs[:,j]) for j in range(7)])
    minimum_clearance=min(minimum_clearance,float(heights(q).min()))
assert minimum_clearance>=0,minimum_clearance
measured_speed=float(np.max(np.abs(np.diff(qs,axis=0))/np.diff(frame_times)[:,None]))
D.update(playbackSpeed=PLAYBACK_SPEED,openingTimes={key:retime(t) for key,t in opening_times.items()},horizontalSegments=horizontal_rows,robot=robot,kinematics=dict(base=arm.base.tolist(),rotation=Rotation.from_matrix(arm.rotation).as_quat().tolist(),offsets=[x.tolist() for x in arm.offsets],axes=[x.tolist() for x in arm.axes]),motion=frames,duration=round(clock,3),motionEnd=end_motion,storyTimes=stamps,robotCheck=dict(max_tcp_error_m=worst,max_orientation_error_rad=worst_angle,samples=len(frames),max_sample_joint_change_rad=max(jumps),max_joint_speed_rad_s=max_speed,max_joint_acceleration_rad_s2=max_accel,unloaded_home_returns=0,minimum_robot_link_floor_clearance_m_24fps=minimum_clearance,max_joint_speed_rad_s_measured=measured_speed,scope='Kinematic concept only; no collision, grip strength or insertion certification.'))
D['videoLayout']=dict(stage_offsets_m=stage_offsets.tolist(),stage_yaw_degrees=np.rad2deg(stage_yaws).tolist(),opening_direction_world=opening_direction.tolist(),task_placements=[dict(source=p['source'],object=serialize(o),fixture=serialize(f)) for p,o,f in zip(D['poses'],OP,FP)],park_center_xy_m=[.07,.27],fixture_grasp='rear rails with tilted approach',airborne_rotation='fixture rolls around its fixed opening axis',withdrawal_distances_m=withdrawal_distances)
D['robotCheck'].update(playback_speed=PLAYBACK_SPEED,carrying_postures=posture_records,joint_total_travel_deg=np.rad2deg(np.abs(np.diff(qs,axis=0)).sum(0)).tolist())
(OUT/'data.js').write_text('window.REUSE_DATA='+json.dumps(D,separators=(',',':'))+';\n')
(OUT/'robot_check.json').write_text(json.dumps(D['robotCheck'],indent=2)+'\n')
print('Done',D['duration'],D['robotCheck'],flush=True)
