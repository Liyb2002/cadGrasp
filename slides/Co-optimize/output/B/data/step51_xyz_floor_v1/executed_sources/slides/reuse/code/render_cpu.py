"""Export the new reuse overview and one-arm animation without a browser.

Uses the exact prepared meshes, pose transforms and robot joint trajectory.
The camera/table are fixed for the complete motion; exported media has no text.
"""
from pathlib import Path
import argparse
import ctypes
import hashlib
import json
import subprocess
import time

import numpy as np
from PIL import Image, ImageColor
from scipy.spatial.transform import Rotation, Slerp
import trimesh

HERE=Path(__file__).resolve().parent
OUT=HERE.parent


def rgb(color):
    return np.array(ImageColor.getrgb(color) if isinstance(color,str) else np.array(color)*255.,float)


def piece(data,color,smooth=False,bias=0):
    v=np.asarray(data['v']).reshape(-1,3);f=np.asarray(data['f']).reshape(-1,3)
    mesh=trimesh.Trimesh(v,f,process=False)
    normals=mesh.vertex_normals[f] if smooth else np.repeat(mesh.face_normals[:,None,:],3,axis=1)
    colors=np.tile(rgb(color),(len(f),3,1)) if not isinstance(color,np.ndarray) else color
    return dict(triangles=v[f],normals=normals,colors=colors,bias=bias)


def box(dimensions,center,color='#414c54'):
    mesh=trimesh.creation.box(dimensions);mesh.apply_translation(center)
    return piece(dict(v=mesh.vertices.ravel(),f=mesh.faces.ravel()),color)


def placed(mesh,rotation=np.eye(3),point=np.zeros(3)):
    return (mesh,rotation,point)


class Renderer:
    def __init__(self):
        source=HERE/'raster.cpp';library=HERE/'raster.dylib'
        if not library.exists() or library.stat().st_mtime < source.stat().st_mtime:
            subprocess.run(['clang++','-O3','-std=c++17','-shared','-fPIC',str(source),'-o',str(library)],check=True)
        self.raster=ctypes.CDLL(str(library)).raster
        self.raster.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
        self.raster.restype=None

    def render(self,parts,camera,width,height,scale=1.0):
        focus,basis,span=camera
        w,h=round(width*scale),round(height*scale)
        tris=[];colors=[]
        light=basis[2]*.72+basis[1]*.5-basis[0]*.22;light/=np.linalg.norm(light)
        fill=basis[2]*.25+basis[1]*.3+basis[0]*.65;fill/=np.linalg.norm(fill)
        for mesh,rotation,point in parts:
            world=mesh['triangles']@rotation.T+point
            p=(world-focus)@basis.T
            p[...,0]=w/2+p[...,0]*h/span
            p[...,1]=h/2-p[...,1]*h/span
            p[...,2]+=mesh['bias']
            n=mesh['normals']@rotation.T
            shade=.64+.27*np.maximum(0,n@light)+.12*np.maximum(0,n@fill)
            if mesh.get('unlit'):shade[:]=1
            tris.append(p.reshape(-1,9));colors.append((mesh['colors']*shade[...,None]).reshape(-1,9))
        p=np.ascontiguousarray(np.concatenate(tris),dtype=np.float32)
        c=np.ascontiguousarray(np.concatenate(colors),dtype=np.float32)
        output=np.empty((h,w,3),dtype=np.uint8)
        self.raster(p.ctypes.data,c.ctypes.data,len(p),w,h,output.ctypes.data)
        im=Image.fromarray(output)
        return im.resize((width,height),Image.Resampling.LANCZOS) if scale!=1 else im


class Scene:
    def __init__(self,data):
        self.data=data
        self.fixture=[piece(data['fixture'],data['fixtureColor'])]
        self.fixture.extend(piece(p,p['color'],bias=.000002) for p in data['headOverlays'])
        raw=data['poses'][0]
        vertices=np.asarray(raw['object']['v']).reshape(-1,3)
        self.object_local=(vertices-vertices.mean(0))@np.asarray(raw['objectR'])
        self.objects=[]
        faces=np.asarray(raw['object']['f']).reshape(-1,3)
        for pose in data['poses']:
            colors=np.tile(rgb('#bac0c5'),(len(faces),3,1));colors[pose['work']]=rgb('#b4c7b2')
            self.objects.append(piece(dict(v=self.object_local.ravel(),f=faces.ravel()),colors,smooth=True))
        self.robot=[[piece(p,p['color'],smooth=True) for p in parts] for parts in data.get('robot',[])]
        self.gripper=[box([.065,.046,.025],[0,0,.05]),box([.158,.04,.025],[0,0,.073])]
        self.fingers=[box([.014,.035,.018],[0,0,.086]),box([.008,.020,.052],[0,0,.113])]
        self.table=box([1.2,1.,.012],[.15,.16,-.008],'#f0f2f4')
        self.table['unlit']=True
        self.times=np.asarray([f['t'] for f in data.get('motion',[])])
        self.motion=data.get('motion',[])
        if self.motion:
            self.object_slerp=Slerp(self.times,Rotation.from_quat([f['o']['q'] for f in self.motion]))
            self.fixture_slerp=Slerp(self.times,Rotation.from_quat([f['f']['q'] for f in self.motion]))

    def task(self,k,display=False):
        pose=self.data['poses'][k]
        if display:
            task=self.data['videoLayout']['task_placements'][k]
            return dict(k=k,o=(Rotation.from_quat(task['object']['q']).as_matrix(),np.array(task['object']['p'])),
                        f=(Rotation.from_quat(task['fixture']['q']).as_matrix(),np.array(task['fixture']['p'])))
        return dict(k=k,o=(np.array(pose['objectR']),np.array(pose['object']['v']).reshape(-1,3).mean(0)),
                    f=(np.array(pose['fixtureR']),np.array(pose['fixtureT'])))

    def sample(self,t):
        t=np.clip(t,self.times[0],self.times[-1])
        j=min(max(0,np.searchsorted(self.times,t,side='right')-1),len(self.times)-2)
        a,b=self.motion[j:j+2];u=(t-self.times[j])/(self.times[j+1]-self.times[j])
        lerp=lambda x,y:(1-u)*np.array(x)+u*np.array(y)
        return dict(k=a['k'],o=(self.object_slerp(t).as_matrix(),lerp(a['o']['p'],b['o']['p'])),
                    f=(self.fixture_slerp(t).as_matrix(),lerp(a['f']['p'],b['f']['p'])),
                    joints=lerp(a['joints'],b['joints']),gap=(1-u)*a['gap']+u*b['gap'])

    def parts(self,state,robot=False,floor=True):
        parts=[placed(p,*state['f']) for p in self.fixture]+[placed(self.objects[state['k']],*state['o'])]
        if robot:
            kin=self.data['kinematics'];point=np.array(kin['base']);rotation=Rotation.from_quat(kin['rotation']).as_matrix()
            for k,meshes in enumerate(self.robot):
                if k:
                    point=point+rotation@np.array(kin['offsets'][k-1])
                    rotation=rotation@Rotation.from_rotvec(np.array(kin['axes'][k-1])*state['joints'][k-1]).as_matrix()
                parts.extend(placed(p,rotation,point) for p in meshes)
            parts.extend(placed(p,rotation,point) for p in self.gripper)
            for sign in (-1,1):
                parts.extend(placed(p,rotation,point+rotation@np.array([sign*state['gap'],0,0])) for p in self.fingers)
        if floor:
            if robot:parts.insert(0,placed(self.table))
            else:
                points=np.concatenate([p['triangles'].reshape(-1,3)@r.T+t for p,r,t in parts])
                low,high=points.min(0),points.max(0)
                plate=box(np.r_[high[:2]-low[:2]+.07,.002],np.r_[(low[:2]+high[:2])/2,-.0011],'#f0f2f4')
                plate['unlit']=True;parts.insert(0,placed(plate))
        return parts


def axes(direction):
    view=np.asarray(direction,float);view/=np.linalg.norm(view)
    right=np.cross([0,0,1],view);right/=np.linalg.norm(right)
    return np.array([right,np.cross(view,right),view])


def corners(parts):
    result=[]
    for mesh,rotation,point in parts:
        lo=mesh['triangles'].min(axis=(0,1));hi=mesh['triangles'].max(axis=(0,1))
        cube=np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
        result.append(cube@rotation.T+point)
    return np.concatenate(result)


def fit(points,direction,aspect,padding=1.10):
    basis=axes(direction);p=points@basis.T;lo,hi=p.min(0),p.max(0)
    span=max(hi[1]-lo[1],(hi[0]-lo[0])/aspect)*padding
    return ((lo+hi)/2)@basis,basis,span


def run(images_only=False):
    data_path=OUT/'data.js';data=json.loads(data_path.read_text().split('=',1)[1].rstrip(';\n'))
    assert data['schema']=='saved_pose1_3_fixture_v1' and len(data['poses'])==2
    scene=Scene(data);renderer=Renderer()
    picture=Image.new('RGB',(2600,1300),'white')
    panel_width,panel_height=1250,1220
    for k in range(2):
        state=scene.task(k)
        parts=scene.parts(state)
        camera=fit(corners(parts),[-1,-1,.82],panel_width/panel_height)
        picture.paste(renderer.render(parts,camera,panel_width,panel_height,1.5),(25+1300*k,40))
    picture.save(OUT/'reuse_overview.png')
    print('New overview exported.',flush=True)
    if not scene.motion:
        if images_only:return
        raise ValueError('Generate the new robot trajectory first.')
    width,height,fps=1440,900,24
    # Fit every saved configuration once. Never move/zoom the camera to hide
    # an unreachable pose, and keep the table fixed during the complete clip.
    bounds=[]
    for t in scene.times:
        bounds.append(corners(scene.parts(scene.sample(t),robot=True,floor=False)))
    camera=fit(np.concatenate(bounds),[.28,1,.65],width/height,1.08)
    report=dict(renderer='CPU triangle z-buffer',task_pose_sources=[p['source'] for p in data['poses']],
                source_shape_sha256=data['sourceShapeSha256'],
                exported_images=['reuse_overview.png'],
                video=dict(encoded=not images_only,duration_seconds=data['duration'],fps=fps,width=width,height=height,
                           fixed_camera=True,fixed_table=True,cast_shadows=False,composition='single_full_scene',
                           camera_focus_m=camera[0].tolist(),camera_axes=camera[1].tolist(),camera_height_m=camera[2]),
                scope='Presentation and kinematic replay; not a robot collision or grasp certificate.')
    previews=OUT/'render_data';previews.mkdir(exist_ok=True)
    for i,t in enumerate(data['storyTimes']):
        im=renderer.render(scene.parts(scene.sample(t),robot=True),camera,width,height,1.25)
        im.save(previews/f'keyframe_{i:02d}.png')
    if not images_only:
        temp=OUT/'reuse_workflow.rendering.mp4'
        command=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
                 '-s',f'{width}x{height}','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264',
                 '-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(temp)]
        process=subprocess.Popen(command,stdin=subprocess.PIPE)
        start=time.perf_counter();count=round(data['duration']*fps)
        try:
            for i in range(count):
                im=renderer.render(scene.parts(scene.sample(i/fps),robot=True),camera,width,height,1.25)
                process.stdin.write(im.tobytes())
                if i%96==0:print(f'Video {i}/{count} frames; {time.perf_counter()-start:.1f}s',flush=True)
            process.stdin.close()
            if process.wait()!=0:raise RuntimeError('ffmpeg encoding failed')
        except BaseException:
            process.kill();process.wait();raise
        temp.replace(OUT/'reuse_workflow.mp4')
        report['video']['frames']=count
        report['video']['render_seconds']=time.perf_counter()-start
        report['video']['sha256']=hashlib.sha256((OUT/'reuse_workflow.mp4').read_bytes()).hexdigest()
    (OUT/'render_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Export complete.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images-only',action='store_true')
    run(parser.parse_args().images_only)
