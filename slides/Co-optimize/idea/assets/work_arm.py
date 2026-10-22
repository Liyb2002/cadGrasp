"""Entertainment cut: a saved Franka movie plus the whole_task KUKA work arm.

The Panda motion, support, layout and force arrows are the saved ones in
data/<set>/; nothing is replanned.  A KUKA LBR Med 14 on a pedestal works
beside the station as in whole_task Step5: its needle axis is fixed in the
world, 45 deg down and away from the base, and the arm only translates it.
A working area is workable when every working triangle is first-hit visible
along that axis (bunny and support occlude, so the whole needle line is
clear) and the traced path (outline, then three strokes) stays continuously
reachable within the joint limits, with the arm, head and needle clear of
the bunny, the support, the pedestal, the floor and the parked Panda.
Otherwise the work arm flails for two seconds and the next task begins.

The cell is placed, not hand-picked: every base azimuth and distance on a
grid around the station is tried; placements whose pedestal meets anything
the saved plan moves, or that hide the station from the fixed far camera,
are dropped; the rest rank by workable tasks, then by work clearance.
Kinematic presentation only: no swept-collision, speed or dynamics claim.
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,hashlib,json,shutil,subprocess,sys,time
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import mujoco
import trimesh
from PIL import Image
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from shapely.geometry import LineString,MultiPoint,Point,Polygon
from shapely.ops import unary_union

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
from franka import Franka
from workflow import Workflow,smooth
from presentation import numbers,set_pose,SCENE_TRANSLATION,FOVY_DEG
from force_arrows import draw,project_points
sys.path.insert(0,str(ROOT))
from codes.simulation.kuka import ASSETS,description,visual_parts
from whole_task.step5.video.kuka import Arm

LABEL='pose1+2+3+4+6'
# The saved poses index the original 1000-face bunny; objects/B/mesh.stl has
# since become whole_task's 5000-face subdivision of the same surface.
SOURCE_MESH_SHA='d1353833e5056c615b28ad338d54ba9f71a8dd1643ded5882cdb48f154b251e4'
STATION=np.array([.43,0.])        # inputs.load: installation centre on the floor
AZIMUTHS=tuple(float(a) for a in range(0,360,15))   # base azimuth about the station; the camera is at +x
DISTANCES=(.60,.65,.70,.75,.80,.85)                  # base to station centre, horizontal
PEDESTAL_H,PEDESTAL_XY=.12,(.30,.34)   # whole_task work-robot pedestal
TILT=45.                          # whole_task needle: 45 deg down, away from the base
TOOL=.155                         # link 7 to needle tip, whole_task tool
TOOL_PARTS=[(.030,.050,.018,.018,(.16,.19,.22)),(.050,.105,.010,.010,(.60,.66,.72)),
            (.105,.123,.008,.003,(.60,.66,.72)),(.123,.155,.0012,.0006,(.60,.66,.72))]
GAP=.11                           # needle stand-off above the first path point
HIT_TOL=1e-4                      # first-hit depth tolerance, well below the 0.6-1.2 mm needle radius
LIFT=.012                         # needle lift between strokes and over other surfaces
WORK_S,ARM_S,FLAIL_S,HOLD_S=3.2,1.,2.,.3
WORK_RATE,CHECK_RATE=60,60        # samples per second for planning checks
WORK_MARGIN,FLAIL_MARGIN=.004,.03
OBSTACLE_AREA,ROBOT_AREA=2e-6,2e-5    # one sample per 2 mm^2 on bunny/support, 20 mm^2 on robots
FLAIL_DEG=np.array([75,30,110,40,140,70,175])
# Close on the Panda handling as in the saved movie; wide while the work arm acts.
# One far camera for the whole cut.  Distance and elevation stay fixed; the view
# turns from the saved movie's azimuth only as far as the placed cell needs.
CAMERA=dict(lookat=(.43,-.30,.60),distance=2.40,azimuth=180.,elevation=-22.)
CAMERA_TURNS=(0,15,-15,30,-30,45,-45,60,-60,75,-75,90,-90)
INK=(.72,.19,.43,1.)


def tool_mesh(start,end,r0,r1):
    """whole_task raster.tool_mesh; importing raster forces Mesa GL, which breaks MuJoCo EGL."""
    angles=np.linspace(0,2*np.pi,17)[:-1];v=[]
    for z,r in [(start,r0),(end,r1)]:v.extend(np.c_[np.cos(angles)*r,np.sin(angles)*r,np.full(16,z)])
    f=[]
    for i in range(16):j=(i+1)%16;f.extend([(i,j,j+16),(i,j+16,i+16)])
    for i in range(1,15):f.extend([(0,i+1,i),(16,16+i,17+i)])
    return np.asarray(v),np.asarray(f)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def ease(v):v=float(np.clip(v,0,1));return v*v*(3-2*v)

def samples(vertices,faces,area,seed=0):
    mesh=trimesh.Trimesh(vertices,faces,process=False)
    count=max(32,int(np.ceil(mesh.area/area)))
    return trimesh.sample.sample_surface(mesh,count,seed=seed)[0]

def restore(story,folder):
    """run.restore_workflow, without its check of the since-replaced bunny mesh."""
    w=Workflow.__new__(Workflow);w.robot=Franka();w.data={}
    w.segments=story['segments'];w.duration=story['duration'];w.story=story['story']
    for s in w.segments:
        for key in 'ab':
            for name in ('object','fixture','hand'):s[key][name]=np.asarray(s[key][name])
        if s['grasp'] is not None:s['grasp']=np.asarray(s['grasp'])
    with np.load(folder/'trajectory.npz') as a:w.times=a['times'];w.joints=a['joints']
    return w


class Saved:
    """The delivered movie's plan, support, arrows and task poses in the display frame."""
    def __init__(self,label,out,source_mesh=None):
        self.label=label;self.folder=HERE/'data'/label
        self.plan=json.loads((self.folder/'plan.json').read_text())
        story=json.loads((self.folder/'segments.json').read_text())
        self.arrows=json.loads((self.folder/'force_arrows.json').read_text())['fields']
        support=ROOT/self.plan['fixture_source']/'support.obj'
        if sha(support)!=self.plan['source_hashes'][str(support)]:raise ValueError('Saved support changed')
        copy=out/'B_mesh_1000faces.stl'
        if not copy.exists():
            if source_mesh is None:raise ValueError('Pass --source-mesh with the original 1000-face bunny')
            shutil.copyfile(source_mesh,copy)
        if sha(copy)!=SOURCE_MESH_SHA:raise ValueError('Not the mesh the saved poses index')
        self.body=trimesh.load(copy,force='mesh',process=False)
        # Shared vertices for path topology; the face order is unchanged.
        self.topology=self.body.copy();self.topology.merge_vertices()
        if not np.allclose(self.topology.triangles,self.body.triangles):raise ValueError('Vertex merge moved faces')
        self.support=trimesh.load(support,force='mesh',process=False)
        self.inputs={str(p):sha(p) for p in [copy,support,*(self.folder/n for n in
            ('plan.json','segments.json','trajectory.npz','force_arrows.json'))]}
        self.workflow=restore(story,self.folder);self.rows={}
        for s in self.workflow.segments:
            k=s['b']['task']
            if k is None:continue
            pose=s['label'][len('task_'):];setup=ROOT/'objects/B/poses'/pose/'setup.npz'
            with np.load(setup) as z:
                if str(z['mesh_sha256'])!=SOURCE_MESH_SHA:raise ValueError('Task indexes another mesh: '+pose)
                ids=np.flatnonzero(z['work_faces'])
            self.inputs[str(setup)]=sha(setup)
            U=s['b']['object'].copy();T=s['b']['fixture'].copy();U[:3,3]+=SCENE_TRANSLATION;T[:3,3]+=SCENE_TRANSLATION
            self.rows[k]=dict(index=k,pose=pose,object=U,fixture=T,work_ids=ids,segment=(s['start'],s['end']))
        self.order=sorted(self.rows,key=lambda k:self.rows[k]['segment'][0])
        self.factor=self.plan['video_playback_factor']


class WorkArm(Arm):
    """whole_task's LBR Med and needle, bounded by the real floor and pedestal."""
    def __init__(self,azimuth,distance):
        az=np.radians(azimuth);self.azimuth=float(azimuth);self.distance=float(distance)
        base=np.r_[STATION+distance*np.array([np.cos(az),np.sin(az)]),PEDESTAL_H]
        super().__init__(base,Rotation.from_euler('z',az+np.pi).as_matrix(),tool=TOOL)
        t=np.radians(TILT);self.axis=self.rotation@np.array([np.cos(t),0.,-np.sin(t)])
        rng=np.random.default_rng(17)
        self.seeds=[np.deg2rad(q) for q in ([0,55,0,-95,0,90,0],[0,80,90,95,-90,40,0],[90,70,0,-95,0,90,0])]
        self.seeds+=list(rng.uniform(self.lower*.7,self.upper*.7,(20,7)))
        links,_,_=description();self.surface=[]
        for i,link in enumerate(links):
            mesh=trimesh.load(ASSETS/f'meshes/collision/link_{i}.stl',force='mesh')
            offset=np.fromstring(link.find('collision/origin').get('xyz'),sep=' ')
            self.surface.append(samples(mesh.vertices+offset,mesh.faces,ROBOT_AREA,seed=i))
        head=[samples(*tool_mesh(a,b,r0,r1),ROBOT_AREA,seed=20+j) for j,(a,b,r0,r1,_) in enumerate(TOOL_PARTS[:3])]
        self.head=np.vstack(head)
        z=np.linspace(TOOL_PARTS[3][0],TOOL-.003,12);self.needle=np.c_[np.zeros((12,2)),z]

    def clouds(self,q):
        """Links 2-7 with the head, and the needle shaft short of its tip, in the world."""
        _,_,transforms=self.forward(q,geometry=True);r,p=transforms[-1]
        links=np.vstack([s@R.T+P for s,(R,P) in zip(self.surface[2:],transforms[2:])]+[self.head@r.T+p])
        return links,self.needle@r.T+p

    def reach(self,point,seeds,accept=lambda q:True):
        """Joints putting the needle tip at point along the fixed axis, else None."""
        for seed in seeds:
            def residual(q):
                tcp,z,_,_=self.forward(q);return np.r_[tcp-point,.18*(z-self.axis),1e-5*(q-seed)]
            def jacobian(q):
                _,_,jp,ja=self.forward(q);return np.vstack([jp,.18*ja,1e-5*np.eye(7)])
            q=least_squares(residual,seed,jac=jacobian,bounds=(self.lower,self.upper),
                            max_nfev=300,ftol=1e-12,xtol=1e-12,gtol=1e-12).x
            tcp,z,_,_=self.forward(q)
            # whole_task Arm.solve acceptance: 0.1 mm and 1e-3 axis error.
            if np.linalg.norm(tcp-point)<1e-4 and np.linalg.norm(z-self.axis)<1e-3 and accept(q):return q
        return None


class Surroundings:
    """Dense obstacle samples for the work arm at one Panda configuration."""
    def __init__(self,arm,points):
        self.arm=arm;self.tree=cKDTree(points)
        self.lo=np.r_[arm.base[:2]-np.asarray(PEDESTAL_XY)/2,0.];self.hi=np.r_[arm.base[:2]+np.asarray(PEDESTAL_XY)/2,PEDESTAL_H]

    def clearance(self,q,needle=True,cap=.06):
        """Smallest distance to anything, capped: only 'clear by cap' matters."""
        links,shaft=self.arm.clouds(q);points=np.vstack([links,shaft]) if needle else links
        gap=np.maximum(np.maximum(self.lo-points,points-self.hi),0.)
        pedestal=np.where(np.all((points>self.lo)&(points<self.hi),axis=1),-1.,np.linalg.norm(gap,axis=1)).min()
        near=self.tree.query(points,distance_upper_bound=cap)[0].min()
        return min(float(min(near,cap)),float(points[:,2].min()),float(pedestal))


def station_scene(saved,row):
    """Bunny and support at one task, with the working triangles marked."""
    body=saved.body.copy();body.apply_transform(row['object'])
    support=saved.support.copy();support.apply_transform(row['fixture'])
    owner=np.zeros(len(body.faces)+len(support.faces),bool);owner[row['work_ids']]=True
    return trimesh.util.concatenate([body,support]),owner


def visible_triangles(saved,row,axis,depth=1.):
    """Working triangles whose centroid and inner samples are each first hit along the needle."""
    scene,owner=station_scene(saved,row);ids=row['work_ids']
    bary=np.array([[1/3,1/3,1/3],[.6,.2,.2],[.2,.6,.2],[.2,.2,.6]])
    points=np.einsum('bk,fkj->fbj',bary,scene.triangles[ids]).reshape(-1,3);origins=points-axis*depth
    hit,ray,_=scene.ray.intersects_location(origins,np.repeat(axis[None],len(points),0),multiple_hits=False)
    first=np.full(len(points),np.inf);first[ray]=np.linalg.norm(hit-origins[ray],axis=1)
    return (np.abs(first-depth)<=HIT_TOL).reshape(len(ids),-1).all(1)


def resample2d(points,spacing):
    out=[points[:1]]
    for a,b in zip(points[:-1],points[1:]):
        n=max(1,int(np.ceil(np.linalg.norm(b-a)/spacing)));out.append(a+np.outer(np.linspace(0,1,n+1)[1:],b-a))
    return np.vstack(out)


class ViewPath:
    """The area as the fixed needle sees it: inset outline, then three strokes.

    Working triangles are projected along the needle axis.  Samples of each
    projected piece's inset outline and of three strokes across it are lifted
    back to the first surface hit along the axis; the needle touches only
    where that hit is a working triangle.  Between strokes, between pieces and
    over any other surface it stays LIFT above the first surface it would meet.
    """
    def __init__(self,scene,owner,axis,inset=5e-4,spacing=1e-3):
        self.axis=axis;self.scene=scene;self.owner=owner
        a=np.array([0.,0.,1.]) if abs(axis[2])<.9 else np.array([1.,0.,0.])
        e1=np.cross(axis,a);e1/=np.linalg.norm(e1);self.basis=np.column_stack([e1,np.cross(axis,e1)])
        self.front=float((scene.vertices@axis).min())-.05
        union=unary_union([Polygon(t@self.basis) for t in scene.triangles[owner]]).buffer(-inset)
        pieces=sorted([g for g in getattr(union,'geoms',[union]) if g.geom_type=='Polygon' and not g.is_empty],key=lambda g:-g.area)
        if not pieces:raise ValueError('Area vanishes after the inset')
        runs=[]
        for piece in pieces:
            runs.append(resample2d(np.asarray(piece.exterior.coords),spacing))
            x0,y0,x1,y1=piece.bounds
            for i,f in enumerate((.28,.5,.72)):
                cut=piece.intersection(LineString([(x0-.01,y0+f*(y1-y0)),(x1+.01,y0+f*(y1-y0))]))
                for g in getattr(cut,'geoms',[cut]):
                    if g.geom_type=='LineString' and g.length>2*spacing:
                        c=np.asarray(g.coords);runs.append(resample2d(c[::-1] if i%2 else c,spacing))
        tips=[];contact=[];self.off_area=0
        for j,run in enumerate(runs):
            if j:
                hop=resample2d(np.array([runs[j-1][-1],run[0]]),4*spacing)[1:-1]
                if len(hop):tips.append(self.lift(hop)[0]-axis*LIFT);contact.append(np.zeros(len(hop),bool))
            p,on=self.lift(run);self.off_area+=int((~on).sum())
            tips.append(p-np.outer(~on,axis)*LIFT);contact.append(on)
        self.tips=np.vstack(tips);self.contact=np.concatenate(contact);self.pieces=len(pieces);self.runs=len(runs)
        if not self.contact[0]:raise ValueError('Path does not start on the area')
        s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(self.tips,axis=0),axis=1))];self.length=float(s[-1]);self.s=s/s[-1]

    def lift(self,uv):
        """First surface hits along the axis, and whether each is a working triangle."""
        origins=uv@self.basis.T+self.axis*self.front
        hit,ray,tri=self.scene.ray.intersects_location(origins,np.repeat(self.axis[None],len(uv),0),multiple_hits=False)
        depth=np.full(len(uv),np.nan);depth[ray]=(hit-origins[ray])@self.axis
        on=np.zeros(len(uv),bool);on[ray]=self.owner[tri]
        known=~np.isnan(depth)
        if not known.any():raise ValueError('Needle rays miss the station')
        depth[~known]=np.interp(np.flatnonzero(~known),np.flatnonzero(known),depth[known])
        return origins+np.outer(depth,self.axis),on

    def at(self,u):
        """Needle tip target and contact at task progress u: descend, trace, rise."""
        if u<.10:return self.tips[0]-self.axis*GAP*(1-ease(u/.10)),False
        if u>.90:return self.tips[-1]-self.axis*GAP*ease((u-.90)/.10),False
        v=(u-.10)/.80;j=min(int(np.searchsorted(self.s,v,side='right')),len(self.s)-1);i=max(j-1,0)
        a=(v-self.s[i])/max(self.s[j]-self.s[i],1e-15)
        return (1-a)*self.tips[i]+a*self.tips[j],bool(self.contact[i] and self.contact[j])

    def ink(self,u):
        """Polylines the needle has drawn by progress u."""
        n=int(np.searchsorted(self.s,np.clip((u-.10)/.80,0,1),side='right'));lines=[];line=[]
        for p,on in zip(self.tips[:n],self.contact[:n]):
            if on:line.append(p)
            elif line:lines.append(np.asarray(line));line=[]
        tip,on=self.at(u)
        if on and .10<=u<=.90:line.append(tip)
        if line:lines.append(np.asarray(line))
        return [l for l in lines if len(l)>1]

    def info(self):
        return dict(model='needle_view_projection_first_hit_lift',pieces=self.pieces,runs=self.runs,
            contact_samples=int(self.contact.sum()),off_area_samples=self.off_area,tip_path_length_m=self.length,lift_m=LIFT)


def work_motion(arm,saved,row,park,near):
    """Approach, trace and retreat joints, or the first reason the arm cannot do it."""
    scene,owner=station_scene(saved,row)
    try:path=ViewPath(scene,owner,arm.axis)
    except ValueError as error:return dict(workable=False,reason='no_traceable_path',detail=str(error))
    accept=lambda q:near.clearance(q,needle=False)>=WORK_MARGIN
    q=arm.reach(path.at(0.)[0],arm.seeds,accept)
    if q is None:return dict(workable=False,reason='stand_off_out_of_reach',path=path.info())
    joints=[]
    for u in np.linspace(0,1,int(WORK_S*WORK_RATE)+1):
        q=arm.reach(path.at(u)[0],[q],accept)
        if q is None:return dict(workable=False,reason='path_out_of_reach_or_blocked',path=path.info(),at_u=float(u))
        joints.append(q)
    for a,b,name in ((park,joints[0],'approach'),(joints[-1],park,'retreat')) if park is not None else ():
        for u in np.linspace(0,1,int(ARM_S*CHECK_RATE)+1):
            if near.clearance((1-smooth(u))*a+smooth(u)*b)<WORK_MARGIN:return dict(workable=False,reason=name+'_collides',path=path.info())
    worst=min(near.clearance(q,needle=False) for q in joints[::4])
    return dict(workable=True,path=path.info(),path_object=path,joints=np.asarray(joints),min_work_clearance_m=worst)


def flail_q(params,park,t):
    """Smooth random waving about the park pose; zero at both ends."""
    ramp=.22;envelope=ease(t/ramp)*ease((FLAIL_S-t)/ramp)
    amp,freq,phase=params['amplitude'],params['frequency'],params['phase']
    return park+envelope*(amp*np.sin(2*np.pi*freq*t+phase)).sum(1)


def flail(arm,park,near,seed,camera):
    """Two seconds of random joint waving, clear of every obstacle and inside the frame."""
    rng=np.random.default_rng(seed);room=np.maximum(np.minimum(park-arm.lower,arm.upper-park)-.05,0);scale=1.
    for attempt in range(600):
        if attempt and attempt%15==0:scale*=.93
        total=np.minimum(np.radians(FLAIL_DEG)*scale*rng.uniform(.75,1.,7),room)
        weights=rng.dirichlet([3,2,1],7)
        params=dict(amplitude=total[:,None]*weights,
            frequency=np.c_[rng.uniform(.9,1.8,7),rng.uniform(1.8,3.2,7),rng.uniform(3.5,5.,7)],
            phase=rng.uniform(0,2*np.pi,(7,3)))
        worst=np.inf
        for t in np.linspace(0,FLAIL_S,int(FLAIL_S*CHECK_RATE)+1):
            q=flail_q(params,park,t);worst=min(worst,near.clearance(q))
            if worst<FLAIL_MARGIN or not camera.contains(arm.clouds(q)[0][::7]):break
        else:
            return dict(seed=seed,attempts=attempt+1,amplitude_scale=scale,min_clearance_m=worst,
                peak_amplitude_deg=np.degrees(params['amplitude'].sum(1)).tolist(),
                **{k:v.tolist() for k,v in params.items()}),params
    raise ValueError('No clear flail found')


class View:
    """The fixed MuJoCo free camera, for planning-time frame checks."""
    def __init__(self,camera,width,height,margin=.02):
        az,el=np.radians(camera['azimuth']),np.radians(camera['elevation'])
        self.forward=np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
        self.eye=np.asarray(camera['lookat'])-camera['distance']*self.forward
        self.right=np.cross(self.forward,[0,0,1]);self.right/=np.linalg.norm(self.right)
        self.up=np.cross(self.right,self.forward)
        self.tan=np.tan(np.radians(FOVY_DEG)/2);self.aspect=width/height;self.margin=margin

    def project(self,points):
        v=points-self.eye;depth=v@self.forward
        return np.c_[(v@self.right)/(depth*self.tan*self.aspect),(v@self.up)/(depth*self.tan)],depth

    def contains(self,points):
        xy,depth=self.project(points)
        return bool(np.all(depth>0) and np.abs(xy).max()<=1-self.margin)


def scene_model(saved,arm,width,height):
    """presentation.model_for plus the work arm on its pedestal, with the needle."""
    folder=HERE/'franka_emika_panda';root=ET.parse(folder/'panda.xml').getroot()
    root.set('model','franka_kuka_work_arm_presentation');root.find('compiler').set('meshdir',str(folder/'assets'))
    visual=ET.SubElement(root,'visual')
    ET.SubElement(visual,'global',offwidth=str(width),offheight=str(height),fovy=str(FOVY_DEG))
    ET.SubElement(visual,'quality',shadowsize='4096')
    ET.SubElement(visual,'headlight',ambient='.35 .35 .35',diffuse='.5 .5 .5',specular='.1 .1 .1')
    # The large floor would set the model extent and blur the shadow map: fix it at the cell.
    ET.SubElement(root,'statistic',center='.40 -.25 .45',extent='1.4')
    asset=root.find('asset');world=root.find('worldbody')
    world.find("body[@name='link0']").set('pos','0 0 0')
    for light in list(world.findall('light')):world.remove(light)
    ET.SubElement(asset,'texture',name='white_background',type='skybox',builtin='gradient',
        rgb1='.96 .97 .98',rgb2='.96 .97 .98',width='256',height='256')
    ET.SubElement(world,'light',pos='.4 -.4 1.8',dir='0 .2 -1',directional='true',diffuse='.35 .35 .35',ambient='.08 .08 .08')
    ET.SubElement(world,'geom',name='floor',type='plane',size='40 40 .01',rgba='.91 .93 .95 1',contype='0',conaffinity='0')
    def geom(parent,name,vertices,faces,rgba,pos='0 0 0'):
        ET.SubElement(asset,'mesh',name=name+'_mesh',vertex=numbers(vertices),face=' '.join(map(str,np.asarray(faces).ravel())))
        ET.SubElement(parent,'geom',name=name,type='mesh',mesh=name+'_mesh',pos=pos,rgba=numbers(rgba),
                      contype='0',conaffinity='0',group='2')
    for name,mesh,color in [('object',saved.body,(.55,.59,.63,1)),('fixture',saved.support,(.12,.43,.70,1))]:
        body=ET.SubElement(world,'body',name=name);ET.SubElement(body,'freejoint',name=name+'_free')
        ET.SubElement(body,'inertial',pos='0 0 0',mass='.2',diaginertia='.001 .001 .001')
        geom(body,name+'_visual',mesh.vertices,mesh.faces,color)
        if name=='object':
            for k,row in saved.rows.items():
                work=trimesh.Trimesh(saved.body.vertices,saved.body.faces[row['work_ids']],process=False)
                tri=work.triangles;normal=work.face_normals
                v=np.concatenate([tri+normal[:,None]*.0003,tri+normal[:,None]*.0001],axis=1).reshape(-1,3)
                f=np.concatenate([np.array([[0,1,2],[3,5,4],[0,3,4],[0,4,1],[1,4,5],[1,5,2],[2,5,3],[2,3,0]])+6*i for i in range(len(tri))])
                geom(body,f'work_{k}',v,f,(.22,.72,.37,0))
    ET.SubElement(world,'geom',name='kuka_pedestal',type='box',pos=numbers([*arm.base[:2],PEDESTAL_H/2]),
        size=numbers([PEDESTAL_XY[0]/2,PEDESTAL_XY[1]/2,PEDESTAL_H/2]),rgba='.86 .87 .89 1',contype='0',conaffinity='0',group='2')
    links,joints,_=description()
    body=ET.SubElement(world,'body',name='kuka_link_0',pos=numbers(arm.base),
        quat=numbers(Rotation.from_matrix(arm.rotation).as_quat()[[3,0,1,2]]))
    for i,link in enumerate(links):
        if i:
            joint=joints[i-1];body=ET.SubElement(body,'body',name=f'kuka_link_{i}',pos=joint.find('origin').get('xyz'))
            ET.SubElement(body,'joint',name=f'kuka_joint_{i}',type='hinge',axis=joint.find('axis').get('xyz'),
                          range=numbers([arm.lower[i-1],arm.upper[i-1]]))
        ET.SubElement(body,'inertial',pos='0 0 0',mass='1',diaginertia='.01 .01 .01')
        for j,(v,f,rgba) in enumerate(visual_parts(i)):
            # Upstream logos are dark patches on grey rings, as in whole_task.
            if i in (2,4) and np.allclose(rgba[:3],.2):rgba=np.array([.6,.6,.6,1.])
            geom(body,f'kuka_{i}_{j}',v,f,rgba,pos=link.find('visual/origin').get('xyz'))
    for j,(a,b,r0,r1,color) in enumerate(TOOL_PARTS):geom(body,f'kuka_tool_{j}',*tool_mesh(a,b,r0,r1),(*color,1))
    return mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))


class Stage:
    """MuJoCo scene: the saved Panda workflow plus the work-arm joints."""
    def __init__(self,saved,arm,width,height,camera=CAMERA):
        self.saved=saved;self.width=width;self.height=height
        self.model=scene_model(saved,arm,width,height);self.data=mujoco.MjData(self.model)
        self.option=mujoco.MjvOption();self.option.geomgroup[3]=0
        self.camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(self.camera)
        self.camera.lookat[:]=camera['lookat'];self.camera.distance=camera['distance']
        self.camera.azimuth=camera['azimuth'];self.camera.elevation=camera['elevation']
        # Trace radius of about 1.2 pixels at the station, so the ink reads at the far view.
        self.ink=1.2*2*camera['distance']*np.tan(np.radians(FOVY_DEG)/2)/height
        address=lambda name:self.model.jnt_qposadr[self.model.joint(name).id]
        self.panda=[address(f'joint{i}') for i in range(1,8)];self.jaws=[address(f'finger_joint{i}') for i in (1,2)]
        self.kuka=[address(f'kuka_joint_{i}') for i in range(1,8)]
        bodies={'hand','left_finger','right_finger',*[f'link{i}' for i in range(8)]};self.panda_parts=[]
        for g in range(self.model.ngeom):
            if self.model.geom_group[g]!=2 or self.model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:continue
            if self.model.body(self.model.geom_bodyid[g]).name not in bodies:continue
            m=self.model.geom_dataid[g];a,n=self.model.mesh_vertadr[m],self.model.mesh_vertnum[m]
            b,c=self.model.mesh_faceadr[m],self.model.mesh_facenum[m]
            self.panda_parts.append((g,samples(self.model.mesh_vert[a:a+n],self.model.mesh_face[b:b+c],ROBOT_AREA,seed=g)))

    def pose(self,t,kuka_q):
        state=self.saved.workflow.state(t)
        for name in ('object','fixture'):
            T=state[name].copy();T[:3,3]+=SCENE_TRANSLATION;set_pose(self.model,self.data,name,T)
        self.data.qpos[self.panda]=state['joints'];self.data.qpos[self.jaws]=np.clip((state['gap']-.003)/2,0,.04)
        self.data.qpos[self.kuka]=kuka_q;self.data.qvel[:]=0;mujoco.mj_forward(self.model,self.data)
        for k in self.saved.rows:self.model.geom_rgba[self.model.geom(f'work_{k}').id,3]=float(state['task']==k)
        return state

    def panda_cloud(self):
        d=self.data
        return np.vstack([v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g] for g,v in self.panda_parts])

    def frame(self,renderer,t,kuka_q,trace=None):
        state=self.pose(t,kuka_q);renderer.update_scene(self.data,camera=self.camera,scene_option=self.option)
        scene=renderer.scene
        for line in trace or []:
            for a,b in zip(line[:-1],line[1:]):
                if scene.ngeom>=scene.maxgeom:break
                g=scene.geoms[scene.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array(INK,dtype=np.float32))
                mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,self.ink,a,b);scene.ngeom+=1
        image=Image.fromarray(renderer.render())
        if state['task'] is not None:self.arrow(image,renderer.scene,state)
        return np.asarray(image)

    def arrow(self,image,scene,state):
        from PIL import ImageDraw
        field=self.saved.arrows[str(state['task'])];points=np.asarray(field['points_local_m'])
        directions=np.asarray(field['inward_directions_local']);lengths=np.asarray(field['display_length_m'])
        R=state['object'][:3,:3];offset=state['object'][:3,3]+SCENE_TRANSLATION
        heads=points@R.T+offset;tails=(points-directions*lengths[:,None])@R.T+offset
        start=project_points(scene,tails,self.width,self.height);end=project_points(scene,heads,self.width,self.height)
        ink=ImageDraw.Draw(image)
        for a,b in zip(start,end):draw(ink,a,b,self.width)


def plan_cloud(saved,stage,count=120):
    """The Panda, the carried bunny and the moved support, sampled over the whole saved plan."""
    clouds=[];body=saved.body.vertices;support=saved.support.vertices[::3]
    for t in np.linspace(0,saved.workflow.duration,count):
        state=stage.pose(float(t),np.zeros(7));clouds.append(stage.panda_cloud()[::6])
        for name,v in (('object',body),('fixture',support)):
            T=state[name];clouds.append(v@T[:3,:3].T+T[:3,3]+SCENE_TRANSLATION)
    return np.vstack(clouds)


def task_obstacles(saved,stage):
    """Dense bunny and support at each task, with the Panda parked as it waits there."""
    out={}
    for k in saved.order:
        row=saved.rows[k];stage.pose(sum(row['segment'])/2,np.zeros(7))
        out[k]=np.vstack([samples(saved.body.vertices,saved.body.faces,OBSTACLE_AREA,1)@row['object'][:3,:3].T+row['object'][:3,3],
            samples(saved.support.vertices,saved.support.faces,OBSTACLE_AREA,2)@row['fixture'][:3,:3].T+row['fixture'][:3,3],
            stage.panda_cloud()])
    return out


def cell_points(arm,saved):
    """What the fixed view must hold: the standing arm, every installation, both parks, the Panda base."""
    first=saved.workflow.segments[0]['a']   # bunny and empty support in their parks
    lift=np.eye(4);lift[:3,3]=SCENE_TRANSLATION
    placed=[(saved.body,lift@first['object']),(saved.support,lift@first['fixture'])]
    placed+=[(m,r[key]) for r in saved.rows.values() for m,key in ((saved.body,'object'),(saved.support,'fixture'))]
    station=[trimesh.transform_points(m.vertices[::4],T) for m,T in placed]
    return np.vstack([standing(arm),*station,[[0,0,0],[0,0,.35]]])


def framed(arm,saved,turn,size,margin=.02):
    """The far camera turned by turn degrees with its lookat centred on the cell, or None if it does not fit."""
    points=cell_points(arm,saved);view=dict(CAMERA,azimuth=CAMERA['azimuth']+turn)
    for _ in range(4):
        v=View(view,*size);xy,_=v.project(points);c=(xy.min(0)+xy.max(0))/2
        shift=view['distance']*v.tan*(v.right*c[0]*v.aspect+v.up*c[1])
        view=dict(view,lookat=tuple(float(x) for x in np.asarray(view['lookat'])+shift))
    xy,_=View(view,*size).project(points)
    return view if np.abs(xy).max()<=1-margin else None


def standing(arm):
    """The pedestal and a standing arm column: radius 0.12 m up to the folded arm's height."""
    angle=np.linspace(0,2*np.pi,16,endpoint=False);ring=np.c_[np.cos(angle),np.sin(angle)]*.12
    column=np.vstack([np.c_[arm.base[:2]+ring,np.full(16,z)] for z in np.linspace(0,arm.base[2]+1.1,7)])
    corners=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (0,1)])*np.r_[np.asarray(PEDESTAL_XY)/2,PEDESTAL_H]
    return np.vstack([column,corners+np.r_[arm.base[:2],0.]])


def hides_station(arm,saved,view,margin=.03):
    """Whether the standing arm covers, or nearly covers, any station point in the camera."""
    station=np.vstack([trimesh.transform_points(m.vertices[::4],r[key]) for r in saved.rows.values()
                       for m,key in ((saved.body,'object'),(saved.support,'fixture'))])
    bxy,bd=view.project(standing(arm));sxy,sd=view.project(station)
    hull=MultiPoint(bxy).convex_hull.buffer(margin);behind=sd>bd.min()
    return bool(any(hull.contains(Point(p)) for p in sxy[behind]))


def footprint_clear(arm,cloud,margin=.03):
    """The pedestal and the arm's base column stay clear of everything the plan moves."""
    lo=arm.base[:2]-np.asarray(PEDESTAL_XY)/2-margin;hi=arm.base[:2]+np.asarray(PEDESTAL_XY)/2+margin
    return not np.any(np.all((cloud[:,:2]>lo)&(cloud[:,:2]<hi),axis=1)&(cloud[:,2]<arm.base[2]+.40))


_SEARCH={}

def placement_score(job):
    saved,plan,obstacles=_SEARCH['saved'],_SEARCH['plan'],_SEARCH['obstacles']
    arm=WorkArm(*job);result=dict(azimuth_deg=arm.azimuth,distance_m=arm.distance,workable=[])
    if not footprint_clear(arm,plan):return dict(result,rejected='pedestal_meets_the_saved_plan')
    for turn in CAMERA_TURNS:
        view=framed(arm,saved,turn,_SEARCH['size'])
        if view is not None and not hides_station(arm,saved,View(view,*_SEARCH['size'])):break
    else:return dict(result,rejected='no_far_view_holds_the_cell')
    result.update(camera_turn_deg=turn,camera=view)
    clear=[]
    for k in saved.order:
        row=saved.rows[k]
        if not visible_triangles(saved,row,arm.axis).all():continue
        m=work_motion(arm,saved,row,None,Surroundings(arm,obstacles[k]))
        if m['workable']:result['workable'].append(row['pose']);clear.append(m['min_work_clearance_m'])
    return dict(result,min_work_clearance_m=min(clear) if clear else None)


def search_placement(saved,plan,obstacles,width,height,workers):
    """Most workable tasks, then the least camera turn, the largest work clearance, the nearer base."""
    import multiprocessing
    _SEARCH.update(saved=saved,plan=plan,obstacles=obstacles,size=(width,height))
    jobs=[(a,d) for a in AZIMUTHS for d in DISTANCES]
    with multiprocessing.get_context('fork').Pool(workers) as pool:ranking=pool.map(placement_score,jobs)
    key=lambda r:(len(r['workable']),-abs(r.get('camera_turn_deg',999)),r.get('min_work_clearance_m') or 0.,-r['distance_m'])
    ranking.sort(key=key,reverse=True)
    return (ranking[0]['azimuth_deg'],ranking[0]['distance_m']),ranking


def park_pose(arm,cloud,camera):
    """The resting pose farthest from the Panda, bunny and support over the whole plan, in frame."""
    near=Surroundings(arm,cloud);best=None
    for a1 in (-150,-120,-90,-60,60,90,120,150):
        for a2 in (-30,-15,0,15,30):
            for a4 in (-100,-80,-60):
                for a6 in (-40,0,40):
                    q=np.radians([a1,a2,0,a4,0,a6,0]);score=near.clearance(q,cap=.25)
                    framed=camera.contains(arm.clouds(q)[0][::7])
                    key=(framed,score)
                    if best is None or key>best[0]:best=(key,q)
    (framed,score),q=best
    return q,dict(joints_deg=np.degrees(q).tolist(),min_clearance_m=score,in_frame=framed)


def timeline(saved,tasks):
    """Video blocks over the saved plan: handling at the saved speed, task dwells expanded."""
    blocks=[];clock=0.
    for s in saved.workflow.segments:
        k=s['b']['task']
        if k is None:parts=[('handling',(s['end']-s['start'])/saved.factor)]
        elif tasks[k]['workable']:parts=[('approach',ARM_S),('work',WORK_S),('retreat',ARM_S)]
        else:parts=[('hold',HOLD_S),('flail',FLAIL_S),('hold',HOLD_S)]
        span=sum(d for _,d in parts);t0=s['start']
        for kind,d in parts:
            t1=t0+(s['end']-s['start'])*d/span
            blocks.append(dict(kind=kind,task=k,video=(clock,clock+d),plan=(t0,t1)));clock+=d;t0=t1
    return blocks,clock


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--set',default=LABEL)
    p.add_argument('--azimuth',type=float,help='Base azimuth about the station (deg); default: search')
    p.add_argument('--distance',type=float,default=.75,help='Base to station centre (m), with --azimuth')
    p.add_argument('--workers',type=int,default=16)
    p.add_argument('--source-mesh',type=Path,help='Original 1000-face bunny, copied beside the cut on first use')
    p.add_argument('--fps',type=int,default=24);p.add_argument('--width',type=int,default=1600)
    p.add_argument('--preview',action='store_true',help='Plan and storyboard only')
    args=p.parse_args();began=time.perf_counter()
    name=args.set+'_work_arm'
    out=HERE/'data'/name;out.mkdir(parents=True,exist_ok=True)
    width=args.width;height=round(width*9/16)
    saved=Saved(args.set,out,args.source_mesh);probe=Stage(saved,WorkArm(0.,DISTANCES[0]),width,height)
    plan=plan_cloud(saved,probe);obstacles=task_obstacles(saved,probe)
    if args.azimuth is None:placement,ranking=search_placement(saved,plan,obstacles,width,height,args.workers)
    else:
        _SEARCH.update(saved=saved,plan=plan,obstacles=obstacles,size=(width,height))
        ranking=[placement_score((args.azimuth,args.distance))];placement=(args.azimuth,args.distance)
    print('PLACEMENT',placement,[{k:v for k,v in r.items() if k!='camera'} for r in ranking[:6]],flush=True)
    if 'camera' not in ranking[0]:raise ValueError('Placement fails: '+ranking[0].get('rejected',''))
    arm=WorkArm(*placement);view=ranking[0]['camera'];stage=Stage(saved,arm,width,height,view);camera=View(view,width,height)
    park,park_info=park_pose(arm,plan,camera);print('PARK',park_info,flush=True)
    tasks={};flails={}
    for k in saved.order:
        row=saved.rows[k];near=Surroundings(arm,obstacles[k])
        visible=visible_triangles(saved,row,arm.axis);area=saved.body.area_faces[row['work_ids']]
        report=dict(pose=row['pose'],working_triangles=len(area),visible_triangles=int(visible.sum()),
            visible_area_fraction=float(area[visible].sum()/area.sum()),park_clearance_m=near.clearance(park))
        if not visible.all():result=dict(workable=False,reason='needle_line_blocked_by_bunny_or_support')
        else:result=work_motion(arm,saved,row,park,near)
        tasks[k]=result;report.update({key:v for key,v in result.items() if key not in ('path_object','joints')})
        if not result['workable']:
            report['flail'],flails[k]=flail(arm,park,near,20261020+k,camera)
        tasks[k]['report']=report;print('TASK',json.dumps(report)[:400],flush=True)
    blocks,duration=timeline(saved,tasks);frames=round(duration*args.fps)
    video_t=np.linspace(0,duration,frames);edges=np.array([b['video'][1] for b in blocks])
    plan_t=[];kuka=[];block_id=[];progress=[]
    for tau in video_t:
        i=min(int(np.searchsorted(edges,tau,side='right')),len(blocks)-1);b=blocks[i]
        u=float(np.clip((tau-b['video'][0])/(b['video'][1]-b['video'][0]),0,1))
        plan_t.append(b['plan'][0]+u*(b['plan'][1]-b['plan'][0]));k=b['task'];done=-1.
        if b['kind'] in ('handling','hold'):q=park
        elif b['kind']=='flail':q=flail_q(flails[k],park,u*FLAIL_S)
        else:
            J=tasks[k]['joints']
            if b['kind']=='approach':q=(1-smooth(u))*park+smooth(u)*J[0]
            elif b['kind']=='retreat':q=(1-smooth(u))*J[-1]+smooth(u)*park;done=1.
            else:q=np.array([np.interp(u,np.linspace(0,1,len(J)),J[:,j]) for j in range(7)]);done=u
        kuka.append(q);block_id.append(i);progress.append(done)
    plan_t=np.asarray(plan_t);kuka=np.asarray(kuka);progress=np.asarray(progress)
    np.savez_compressed(out/'trajectory.npz',video_t=video_t,plan_t=plan_t,kuka_q=kuka,block=np.asarray(block_id),work_u=progress)
    def trace(f):
        b=blocks[block_id[f]];k=b['task'];u=progress[f]
        if k is None or u<0 or not tasks[k]['workable']:return None
        return [line-arm.axis*stage.ink for line in tasks[k]['path_object'].ink(u)]
    report=dict(set=saved.label,kind='kinematic_entertainment_cut',saved_movie=str(saved.folder.relative_to(ROOT)),
        work_robot='KUKA LBR Med 14 R820 (whole_task Step5 model and needle)',work_robot_base_m=arm.base.tolist(),
        placement=dict(azimuth_deg=arm.azimuth,distance_m=arm.distance,searched=bool(ranking),
            grid=dict(azimuths_deg=list(AZIMUTHS),distances_m=list(DISTANCES)),
            rule=('drop pedestals meeting the saved plan; turn the fixed far camera (distance and elevation unchanged) the least '
                  'that holds the standing arm, installations and parks without hiding the station; rank by workable tasks, '
                  'then least camera turn, then work clearance, then nearer base'),
            ranking=ranking),
        work_robot_yaw_deg=arm.azimuth+180.,pedestal_m=dict(height=PEDESTAL_H,footprint=PEDESTAL_XY),
        needle_axis_world=arm.axis.tolist(),needle_tilt_deg=TILT,tool_length_from_link7_m=TOOL,
        rule=('workable iff every working triangle (centroid and three inner samples) is the first hit along the fixed needle axis '
              'with bunny and support as occluders, the outline-and-strokes path is continuously reachable with the axis fixed, and '
              'dense samples of links 2-7 and the head stay clear of bunny, support, parked Panda, pedestal and floor; '
              'otherwise the work arm flails for two seconds'),
        park=park_info,tasks={saved.rows[k]['pose']:tasks[k]['report'] for k in saved.order},
        workable=[saved.rows[k]['pose'] for k in saved.order if tasks[k]['workable']],
        flailing=[saved.rows[k]['pose'] for k in saved.order if not tasks[k]['workable']],
        handling_playback_factor=saved.factor,durations_s=dict(work=WORK_S,approach_retreat=ARM_S,flail=FLAIL_S,hold=HOLD_S),
        clearance_samples=dict(bunny_support_m2_per_point=OBSTACLE_AREA,robots_m2_per_point=ROBOT_AREA,
            work_margin_m=WORK_MARGIN,flail_margin_m=FLAIL_MARGIN,check_rate_hz=CHECK_RATE),
        camera=dict(view,fixed=True,turn_from_saved_movie_deg=ranking[0]['camera_turn_deg'],flail_kept_in_frame=True),fps=args.fps,frames=frames,duration_s=duration,width=width,height=height,
        blocks=[dict(b,pose=None if b['task'] is None else saved.rows[b['task']]['pose']) for b in blocks],
        inputs=saved.inputs,sources={str(p):sha(p) for p in [Path(__file__),HERE/'presentation.py',HERE/'force_arrows.py',
            HERE/'workflow.py',HERE/'franka.py',ROOT/'whole_task/step5/video/kuka.py',ROOT/'whole_task/step5/video/raster.py',
            ROOT/'codes/simulation/kuka.py']},
        flail_joint_speed_certified=False,swept_collision_certified=False,dynamics_integrated=False,no_text=True)
    with mujoco.Renderer(stage.model,height=height,width=width,max_geom=20000) as renderer:
        stamps=[0]
        for k in saved.order:
            b=next(i for i,x in enumerate(blocks) if x['task']==k and x['kind'] in ('flail','work'))
            v0,v1=blocks[b]['video'];stamps+=[int(np.searchsorted(video_t,v0+.35*(v1-v0))),int(np.searchsorted(video_t,v0+.7*(v1-v0)))]
        stamps.append(frames-1);thumb=(640,360)
        board=Image.new('RGB',(thumb[0]*3,thumb[1]*int(np.ceil(len(stamps)/3))),'white')
        for i,f in enumerate(stamps):
            image=Image.fromarray(stage.frame(renderer,plan_t[f],kuka[f],trace(f)))
            image.resize(thumb).save(out/f'keyframe_{i:02d}.png');board.paste(image.resize(thumb),((i%3)*thumb[0],(i//3)*thumb[1]))
        board.save(HERE/f'{name}_storyboard.png')
        if not args.preview:
            output=HERE.parent/f'{name}.mp4';pending=output.with_suffix('.pending.mp4')
            command=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(args.fps),
                '-i','-','-an','-c:v','libx264','-preset','fast','-crf','19','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(pending)]
            proc=subprocess.Popen(command,stdin=subprocess.PIPE)
            try:
                for f in range(frames):
                    proc.stdin.write(stage.frame(renderer,plan_t[f],kuka[f],trace(f)).tobytes())
                    if f%(args.fps*5)==0:print('VIDEO',round(f/args.fps),'/',round(duration),flush=True)
            finally:proc.stdin.close()
            if proc.wait()!=0:raise RuntimeError('Video encoder failed')
            pending.replace(output)
            probe=subprocess.run(['ffprobe','-v','error','-count_frames','-show_entries','stream=nb_read_frames','-of','json',str(output)],
                check=True,capture_output=True,text=True)
            decoded=int(json.loads(probe.stdout)['streams'][0]['nb_read_frames'])
            if decoded!=frames:raise ValueError('Incomplete movie')
            report.update(output=str(output.relative_to(ROOT)),video_sha256=sha(output),decoded_frames=decoded)
    report.update(storyboard=str((HERE/f'{name}_storyboard.png').relative_to(ROOT)),seconds=time.perf_counter()-began)
    save(out/'plan.json',report);print('COMPLETE',name,report['workable'],report['flailing'],round(duration,2),flush=True)


if __name__=='__main__':main()
