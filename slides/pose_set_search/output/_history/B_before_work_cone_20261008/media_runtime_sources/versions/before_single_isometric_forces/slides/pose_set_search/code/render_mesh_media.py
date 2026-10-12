"""Text-free construction and rigid-reuse movies, using saved triangle solids."""
import argparse
import copy
import hashlib
import json
import math
import subprocess
import time
from scipy.spatial.transform import Rotation,Slerp
from PIL import Image,ImageDraw
from common import *
from case_sets import CASES,case_directory,POSE_SET_NAMES
from mesh_media import load_layout,read_mesh
from render_fast_process import Panel,chosen_owner,font
from solid_render import arrow_mesh
from reuse_first import registered

BLUE='#2A91D2';GRAY='#A5ADB5';GREEN='#21c875';RED='#ef4938';YELLOW='#ffe000'
FPS=24;SIZE=(800,900)


def smooth(u):
    u=float(np.clip(u,0,1));return u*u*(3-2*u)


def interpolate(a,b,u,center):
    rotation=Slerp([0,1],Rotation.from_matrix(np.array([a[:3,:3],b[:3,:3]])))([u]).as_matrix()[0]
    ca=a[:3,:3] @ center+a[:3,3];cb=b[:3,:3] @ center+b[:3,3]
    result=np.eye(4);result[:3,:3]=rotation
    result[:3,3]=(1-u)*ca+u*cb-rotation @ center
    return result


def direction_between(a,b,u):
    direction=(1-u)*a+u*b
    return direction/np.linalg.norm(direction)


def tint(a,b,u):
    def rgb(color):return np.array([int(color[i:i+2],16) for i in [1,3,5]])
    return '#'+''.join(f'{int(value):02x}' for value in np.rint((1-u)*rgb(a)+u*rgb(b)))


class Scene:
    def __init__(self,out,body):
        self.out,self.body=out,body
        self.rows=json.loads((out/'process.json').read_text())
        self.geometry=json.loads((out/'mesh_geometry.json').read_text())
        self.report=json.loads((out/'search_report.json').read_text())
        self.poses=self.report['requested_poses']
        self.layouts=[load_layout(out/row['layout']) for row in self.rows]
        self.meshes=[read_mesh(out/record['mesh']) for record in self.geometry['steps']]
        self.deltas=[{kind:read_mesh(out/path) for kind,path in record['delta_meshes'].items()}
                     for record in self.geometry['steps']]
        with np.load(out/self.rows[0]['layout']) as saved:
            self.native=saved['native_world'].copy()
        self.final=load_layout(out/'sampled_layout.npz')
        self.support=self.meshes[-1]

    def process_points(self):
        points=[]
        for row,layout,mesh in zip(self.rows,self.layouts,self.meshes):
            points.append(mesh.vertices)
            k=chosen_owner(row,layout,self.poses)
            body=transform_mesh(self.body,layout.placements[k])
            points.append(body.vertices)
            origin=body.vertices.mean(0);origin[2]=body.bounds[1,2]+.005
            points.append(arrow_mesh(origin,layout.directions[k],.045).vertices)
        return np.vstack(points)

    def result_points(self):
        points=[]
        for k in self.final.active:
            frame=self.native[self.final.hosts[k]]
            points.append(C.transform_points(self.support.vertices,frame))
            points.append(C.transform_points(self.body.vertices,frame @ self.final.placements[k]))
        return np.vstack(points)


def render_pair(panels,layers):
    canvas=Image.new('RGB',(1600,900),'white')
    for i,(panel,meshes) in enumerate(zip(panels,layers)):
        canvas.paste(panel.render(meshes),(800*i,0))
    ImageDraw.Draw(canvas).line((800,18,800,882),fill='#e5e9ec',width=1)
    return np.asarray(canvas)


class Movie:
    def __init__(self,path):
        self.path=path;self.pending=path.with_name('.'+path.stem+'.rendering.mp4')
        self.frames=0
        self.encoder=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24',
            '-s','1600x900','-r',str(FPS),'-i','-','-an','-c:v','libx264','-threads','2',
            '-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(self.pending)],stdin=subprocess.PIPE)

    def emit(self,pixels):
        self.encoder.stdin.write(pixels.tobytes());self.frames+=1

    def finish(self,record):
        self.encoder.stdin.close()
        if self.encoder.wait():raise RuntimeError('Video encoder failed')
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames',
            '-select_streams','v:0','-show_entries','stream=width,height,nb_read_frames,duration',
            '-of','json',str(self.pending)]))['streams'][0]
        assert int(probe['nb_read_frames'])==self.frames
        self.pending.replace(self.path)
        record.update(frame_count=self.frames,fps=FPS,duration_seconds=self.frames/FPS,
            frame_size=[1600,900],onscreen_text=False,panel_order=['whole','incremental'],
            file=self.path.name,sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())
        return record


def empty_orbit(panel,meshes,u):
    rotated=copy.copy(panel)
    azimuth=np.radians(-55+360*u);elevation=np.radians(30)
    rotated.camera=np.array([np.cos(elevation)*np.cos(azimuth),np.cos(elevation)*np.sin(azimuth),np.sin(elevation)])
    rotated.right=np.array([-np.sin(azimuth),np.cos(azimuth),0.]);rotated.up=np.cross(rotated.camera,rotated.right)
    return rotated,meshes


def camera(points):
    panel=Panel(None,size=SIZE,elevation=30,azimuth=-55,points=points)
    # One common center/scale; every azimuth of the ending orbit fits.
    center=(points.min(0)+points.max(0))/2
    radius=float(np.linalg.norm(points-center,axis=1).max())
    panel.center=center
    panel.scale=.94*min(SIZE)/(2*radius)
    return panel


def process_layers(scene,progress):
    position=progress*(len(scene.rows)-1)
    index=min(len(scene.rows)-2,int(position))
    if len(scene.rows)==1:index=0
    u=position-index
    if index==len(scene.rows)-2 and progress>=1:u=1.
    next_index=min(index+1,len(scene.rows)-1)
    before,after=scene.layouts[index],scene.layouts[next_index]
    row=scene.rows[next_index]
    k=chosen_owner(row,after,scene.poses)
    old_q=before.placements[k] if k in before.active else after.placements[k]
    q=interpolate(old_q,after.placements[k],smooth(min(u/.53,1)),scene.body.center_mass)
    body=transform_mesh(scene.body,q)
    old_d=before.directions[k] if k in before.active else after.directions[k]
    d=direction_between(old_d,after.directions[k],smooth(min(u/.53,1)))
    origin=body.vertices.mean(0);origin[2]=body.bounds[1,2]+.005
    arrow=arrow_mesh(origin,d,.045)
    parts=scene.deltas[next_index]
    if u<.26:
        materials=[(scene.meshes[index],BLUE)]
    elif u<.52:
        materials=[(parts['kept'],BLUE),(parts['removed'],RED)]
    elif u<.83:
        materials=[(parts['kept'],BLUE),(parts['added'],GREEN)]
    else:
        materials=[(parts['kept'],BLUE),(parts['added'],tint(GREEN,BLUE,smooth((u-.83)/.17)))]
    layers=[(body,GRAY)]+materials+[(arrow,YELLOW)]
    # The sweep cue is a displaced silhouette along the changing saved path.
    # It adds no support material and is never carved as a design relocation.
    if u<.55 and np.linalg.norm(old_d-after.directions[k])>1e-6:
        ghost=body.copy();ghost.apply_translation(.055*d)
        layers.append((ghost,'#cfebf5'))
    return layers,dict(step=next_index,phase=row['phase'],operation_progress=u,
                       object_to_fixture=q.tolist(),direction_fixture=d.tolist())


def process_movie(directory,scenes):
    points=np.vstack([scene.process_points() for scene in scenes])
    base=camera(points);panels=[base,base]
    transitions=max(len(scene.rows)-1 for scene in scenes)
    duration=max(12,min(27,transitions*.85+3))
    main_frames=int(round((duration-2)*FPS));orbit_frames=2*FPS
    movie=Movie(directory/'process.mp4');timeline=[]
    for n in range(main_frames):
        progress=n/max(1,main_frames-1)
        payload=[process_layers(scene,progress) for scene in scenes]
        movie.emit(render_pair(panels,[pair[0] for pair in payload]))
        timeline.append(dict(frame=n,panels=[pair[1] for pair in payload]))
    for n in range(orbit_frames):
        u=n/max(1,orbit_frames-1)
        pair=[empty_orbit(base,[(scene.support,BLUE)],u) for scene in scenes]
        movie.emit(render_pair([p[0] for p in pair],[p[1] for p in pair]))
        timeline.append(dict(frame=main_frames+n,phase='empty_support_orbit',degrees=360*u))
    return movie.finish(dict(kind='saved search operations',timeline=timeline,
        added_color=GREEN,removed_color=RED,representation='direct triangle Boolean meshes',
        interpolated_object_motion_is_design_configuration_change=True,
        relocation_path_carved=False,search_rerun=False,force_acceptance_run=False))


def result_layers(scene,index,u):
    layout=scene.final;k=layout.active[index]
    frame=scene.native[layout.hosts[k]]
    q=layout.placements[k]
    support=transform_mesh(scene.support,frame)
    body=transform_mesh(scene.body,frame @ q)
    d=frame[:3,:3] @ layout.directions[k]
    travel=max(.22,float((support.vertices @ d).max()-(body.vertices @ d).min())+.03)
    if u<.20:
        previous=layout.active[max(index-1,0)]
        old=scene.native[layout.hosts[previous]]
        empty=interpolate(old,frame,smooth(u/.20),scene.support.center_mass)
        if index and not np.allclose(old,frame,atol=1e-12,rtol=0):
            empty[2,3]+=.018*np.sin(np.pi*u/.20)
        return [(transform_mesh(scene.support,empty),BLUE)],dict(pose=scene.poses[k],phase='move_empty_fixture',
            support_transform=empty.tolist(),object_visible=False)
    if u<.46:
        offset=travel*(1-smooth((u-.20)/.26));phase='insert'
    elif u<.66:
        offset=0.;phase='seated'
    else:
        offset=travel*smooth((u-.66)/.34);phase='exit'
    body.apply_translation(offset*d)
    return [(body,GRAY),(support,BLUE)],dict(pose=scene.poses[k],phase=phase,
        support_transform=frame.tolist(),object_transform=(frame @ q).tolist(),
        object_visible=True,exit_direction_world=d.tolist(),exit_offset_m=offset,travel_m=travel)


def result_movie(directory,scenes):
    points=np.vstack([scene.result_points() for scene in scenes])
    base=camera(points)
    count=len(scenes[0].final.active)
    assert count==len(scenes[1].final.active)
    frames_per_pose=60
    movie=Movie(directory/'result.mp4');timeline=[]
    for index in range(count):
        for n in range(frames_per_pose):
            u=n/(frames_per_pose-1)
            payload=[result_layers(scene,index,u) for scene in scenes]
            movie.emit(render_pair([base,base],[item[0] for item in payload]))
            timeline.append(dict(frame=index*frames_per_pose+n,panels=[item[1] for item in payload]))
    for n in range(72):
        u=n/71
        shapes=[transform_mesh(scene.support,scene.native[scene.final.hosts[scene.final.active[-1]]]) for scene in scenes]
        pair=[empty_orbit(base,[(mesh,BLUE)],u) for mesh in shapes]
        movie.emit(render_pair([p[0] for p in pair],[p[1] for p in pair]))
        timeline.append(dict(frame=count*frames_per_pose+n,phase='empty_support_orbit',degrees=360*u))
    return movie.finish(dict(kind='all saved poses use one rigid support',timeline=timeline,
        same_support_shape_in_every_frame=True,object_removed_before_fixture_moves=True,
        original_world_object_orientation_and_seated_height_preserved=True,
        representation='direct triangle Boolean meshes',search_rerun=False,force_acceptance_run=False))


def pictures(scene):
    # No heading, index, labels, captions or legend in process.png.
    points=scene.process_points();panel=Panel(None,size=(540,500),points=points,elevation=30,azimuth=-55)
    columns=3;canvas=Image.new('RGB',(columns*1080,math.ceil(len(scene.rows)/columns)*500),'white')
    for index,(row,layout,mesh) in enumerate(zip(scene.rows,scene.layouts,scene.meshes)):
        k=chosen_owner(row,layout,scene.poses)
        body=transform_mesh(scene.body,layout.placements[k])
        origin=body.vertices.mean(0);origin[2]=body.bounds[1,2]+.005
        arrow=arrow_mesh(origin,layout.directions[k],.045)
        x,y=(index%columns)*1080,(index//columns)*500
        canvas.paste(panel.render([(mesh,BLUE)]),(x,y))
        canvas.paste(panel.render([(body,GRAY),(mesh,BLUE),(arrow,YELLOW)]),(x+540,y))
    canvas.save(scene.out/'process.png')
    count=len(scene.final.active);size=(610,570)
    result=Image.new('RGB',(3*size[0],80+math.ceil(count/3)*size[1]),'white');draw=ImageDraw.Draw(result)
    draw.text((24,19),scene.out.parent.name+' / '+scene.out.name+' | triangle mesh',font=font(25,True),fill='#253441')
    for index,k in enumerate(scene.final.active):
        frame=scene.native[scene.final.hosts[k]]
        support=transform_mesh(scene.support,frame);body=transform_mesh(scene.body,frame @ scene.final.placements[k])
        d=frame[:3,:3] @ scene.final.directions[k]
        origin=body.vertices.mean(0);origin[2]=body.bounds[1,2]+.005
        arrow=arrow_mesh(origin,d,.048)
        panel=Panel(None,size=(600,510),points=np.vstack([support.vertices,body.vertices,arrow.vertices]),elevation=30,azimuth=-55)
        x,y=(index%3)*610,80+(index//3)*570
        draw.text((x+24,y+12),scene.poses[k]+' | '+('rotate fixture' if registered(scene.final,k) else 'Juxtapose -> '+scene.poses[scene.final.hosts[k]]),font=font(20,True),fill='#253441')
        result.paste(panel.render([(body,GRAY),(support,BLUE),(arrow,YELLOW)]),(x+5,y+45))
    result.save(scene.out/'final_result.png')
    C.save(scene.out/'render.json',dict(complete=True,triangle_mesh=True,voxel_surface=False,
        process_png_onscreen_text=False,process_png_layout='chronological; each pair is support only / selected object seated',
        source_mesh='support.obj',source_mesh_sha256=hashlib.sha256((scene.out/'support.obj').read_bytes()).hexdigest(),
        material_volume_cm3=scene.geometry['final_material_volume_cm3'],search_sampling_unchanged=True,
        final_acceptance_run=False,same_rigid_support_in_every_final_pose=True))
    path=scene.out/'README.md'
    text=path.read_text()
    text=text.split('\n\n当前 mesh：',1)[0]
    text=text.replace('图里蓝色是同一采样材料模型的 1.25 mm 网格边界','旧网格图已经存档。图里蓝色是从保存布局直接 Boolean 构造的三角网格边界')
    text=text.replace('过程图左侧只看支撑，右侧加入该步相关物体','过程图不含文字；按行从左到右，每一对的左侧只看支撑、右侧加入该步相关物体')
    text=text.replace('显示网格 + / − cm³','旧显示网格 + / − cm³')
    path.write_text(text+'\n\n当前 mesh：[support.obj](support.obj)。本集合的两个视频位于上一级：'
        '[搜索过程](../process.mp4)、[逐 pose 使用](../result.mp4)，左 whole、右 incremental。'
        f'该方法最终三角网格材料体积 {scene.geometry["final_material_volume_cm3"]:.3f} cm³；仅重建展示几何，没有重新搜索或运行最终力学验收。\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',choices=list(CASES),required=True)
    parser.add_argument('--root',type=Path,default=HERE/'output/B')
    parser.add_argument('--pictures-only',action='store_true')
    args=parser.parse_args();directory=case_directory(args.root,args.case)
    _,_,body=C.state('B',f'pose_{CASES[args.case][0]}')
    scenes=[Scene(directory/method,body) for method in ['whole','incremental']]
    for scene in scenes:pictures(scene)
    if args.pictures_only:return
    began=time.monotonic()
    construction=process_movie(directory,scenes)
    print('PROCESS VIDEO',directory.name,construction['duration_seconds'],'s',flush=True)
    result=result_movie(directory,scenes)
    print('RESULT VIDEO',directory.name,result['duration_seconds'],'s',flush=True)
    C.save(directory/'videos.json',dict(complete=True,process=construction,result=result,
        style_reference='slides/idea/vis/single_use_fixture.mp4 and juxtaposed_fixture.mp4',
        panel_order=['whole','incremental'],geometry_preserved=True,final_acceptance_run=False,
        seconds=time.monotonic()-began))


if __name__=='__main__':main()
