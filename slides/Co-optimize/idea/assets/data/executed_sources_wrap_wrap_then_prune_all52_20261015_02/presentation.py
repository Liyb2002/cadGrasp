"""Render the task from the front, with the native Franka behind it."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import trimesh
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw
from inputs import HERE
from force_arrows import select,draw,project_points,ARROW_COUNT

SCENE_TRANSLATION=np.array([0.,0.,-.24])
CAMERA_ZOOM=1.5
CAMERA_DISTANCE_M=1.25
BASE_FOVY_DEG=45.
FOVY_DEG=float(np.degrees(2*np.arctan(np.tan(np.radians(BASE_FOVY_DEG)/2)/CAMERA_ZOOM)))


def numbers(a):return ' '.join(f'{x:.12g}' for x in np.asarray(a).ravel())

def model_for(data,width,height):
    asset_folder=HERE/'franka_emika_panda'
    root=ET.parse(asset_folder/'panda.xml').getroot()
    root.set('model','franka_shared_fixture_presentation')
    root.find('compiler').set('meshdir',str(asset_folder/'assets'))
    visual=ET.SubElement(root,'visual')
    ET.SubElement(visual,'global',offwidth=str(width),offheight=str(height),fovy=str(FOVY_DEG))
    ET.SubElement(visual,'quality',shadowsize='2048')
    ET.SubElement(visual,'headlight',ambient='.35 .35 .35',diffuse='.5 .5 .5',specular='.1 .1 .1')
    asset=root.find('asset');world=root.find('worldbody')
    world.find("body[@name='link0']").set('pos','0 0 0')
    for light in list(world.findall('light')):world.remove(light)
    ET.SubElement(asset,'texture',name='white_background',type='skybox',builtin='gradient',
        rgb1='.96 .97 .98',rgb2='.96 .97 .98',width='256',height='256')
    ET.SubElement(world,'light',pos='.4 -.4 1.8',dir='0 .2 -1',directional='true',diffuse='.35 .35 .35',ambient='.08 .08 .08')
    ET.SubElement(world,'geom',name='floor',type='plane',size='2 2 .01',rgba='.91 .93 .95 1',contype='0',conaffinity='0')
    def geom(parent,name,mesh,rgba):
        ET.SubElement(asset,'mesh',name=name+'_mesh',vertex=numbers(mesh.vertices),face=numbers(mesh.faces).replace('.',''))
        ET.SubElement(parent,'geom',name=name,type='mesh',mesh=name+'_mesh',rgba=rgba,contype='0',conaffinity='0',group='2')
    for name,color in [('object','.55 .59 .63 1'),('fixture','.12 .43 .70 1')]:
        body=ET.SubElement(world,'body',name=name)
        ET.SubElement(body,'freejoint',name=name+'_free')
        ET.SubElement(body,'inertial',pos='0 0 0',mass='.2',diaginertia='.001 .001 .001')
        geom(body,name+'_visual',data[name],color)
        if name=='object':
            for i,row in data['rows'].items():
                work=row['work_mesh'];tri=work.triangles;normal=work.face_normals
                vertices=np.concatenate([tri+normal[:,None,:]*.0003,tri+normal[:,None,:]*.0001],axis=1).reshape(-1,3)
                faces=[]
                for k in range(len(tri)):
                    a=6*k;faces.extend(np.array([[0,1,2],[3,5,4],[0,3,4],[0,4,1],[1,4,5],[1,5,2],[2,5,3],[2,3,0]])+a)
                patch=trimesh.Trimesh(vertices,np.asarray(faces),process=False)
                geom(body,'work_'+str(i),patch,'.22 .72 .37 0')
    return mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))

def set_pose(model,data,name,T):
    body=model.body(name).id;j=model.body_jntadr[body];a=model.jnt_qposadr[j]
    quat=Rotation.from_matrix(T[:3,:3]).as_quat()[[3,0,1,2]]
    data.qpos[a:a+7]=np.r_[T[:3,3],quat]


class Display:
    def __init__(self,workflow,width=1600,height=900):
        self.workflow=workflow;self.width=width;self.height=height
        self.model=model_for(workflow.data,width,height);self.data=mujoco.MjData(self.model)
        self.option=mujoco.MjvOption();self.option.geomgroup[3]=0
        self.camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(self.camera)
        self.camera.lookat[:]=[.43,0.,.30];self.camera.distance=CAMERA_DISTANCE_M;self.camera.azimuth=180;self.camera.elevation=-30
        self.arm=[self.model.jnt_qposadr[self.model.joint('joint'+str(i)).id] for i in range(1,8)]
        self.jaws=[self.model.jnt_qposadr[self.model.joint('finger_joint'+str(i)).id] for i in [1,2]]
        scene=mujoco.MjvScene(self.model,maxgeom=2000)
        mujoco.mj_forward(self.model,self.data)
        mujoco.mjv_updateScene(self.model,self.data,self.option,None,self.camera,mujoco.mjtCatBit.mjCAT_ALL,scene)
        self.arrow_fields={i:select(row,workflow.data['object'],workflow.data['fixture'],scene,
            SCENE_TRANSLATION,width,height) for i,row in workflow.data['rows'].items()}

    def update(self,t):
        state=self.workflow.state(t)
        for name in ['object','fixture']:
            pose=state[name].copy();pose[:3,3]+=SCENE_TRANSLATION
            set_pose(self.model,self.data,name,pose)
        self.data.qpos[self.arm]=state['joints'];self.data.qpos[self.jaws]=np.clip((state['gap']-.003)/2,0,.04)
        self.data.qvel[:]=0;mujoco.mj_forward(self.model,self.data)
        for i in self.workflow.data['rows']:
            self.model.geom_rgba[self.model.geom('work_'+str(i)).id,3]=float(state['task']==i)
        return state

    def frame(self,renderer,t):
        state=self.update(t)
        renderer.update_scene(self.data,camera=self.camera,scene_option=self.option)
        image=Image.fromarray(renderer.render());self.arrow(image,renderer.scene,state)
        return np.asarray(image)

    def arrow(self,image,scene,state):
        if state['task'] is None:return
        field=self.arrow_fields[state['task']];points=np.asarray(field['points_local_m'])
        directions=np.asarray(field['inward_directions_local']);lengths=np.asarray(field['display_length_m'])
        rotation=state['object'][:3,:3];offset=state['object'][:3,3]+SCENE_TRANSLATION
        heads=points@rotation.T+offset;tails=(points-directions*lengths[:,None])@rotation.T+offset
        start=project_points(scene,tails,self.width,self.height);end=project_points(scene,heads,self.width,self.height)
        ink=ImageDraw.Draw(image)
        for a,b in zip(start,end):draw(ink,a,b,self.width)
