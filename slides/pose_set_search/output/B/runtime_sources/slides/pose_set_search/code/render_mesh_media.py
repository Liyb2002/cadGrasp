"""Text-free construction and rigid-reuse movies, using saved triangle solids."""
import argparse
import hashlib
import json
import math
import subprocess
import time
from scipy.spatial.transform import Rotation,Slerp
from PIL import Image,ImageDraw
from scipy.ndimage import binary_erosion
from common import *
from case_sets import CASES,case_directory,POSE_SET_NAMES
from mesh_media import load_layout,read_mesh
from render_fast_process import Panel,chosen_owner,font,RASTER
from work_access import WorkAccess
from solid_render import arrow_mesh
from reuse_first import registered

BLUE='#2A91D2';GRAY='#A5ADB5';GREEN='#21c875';RED='#ef4938';YELLOW='#ffe000'
WORK='#FFBD66';FORCE='#D94B16'
FORBIDDEN='#F0A33E';ACCESS_ALPHA=.16
ACCESS_CACHE={}
FPS=24;SIZE=(1600,900)
ELEVATION=35.26438968;AZIMUTH=-45
POSE_PHASE_FRAMES={'move_empty_fixture':12,'insert':18,'seated':24,'exit':18}


class WorkingSurface:
    """Original task faces and reproducible, inward surface-force illustrations."""
    def __init__(self,body,pose):
        task,_,native_body=C.state('B',pose)
        assert np.array_equal(native_body.faces,body.faces)
        assert np.allclose(native_body.vertices,body.vertices,atol=1e-12,rtol=0)
        self.ids=np.asarray(task.domain.work_ids,dtype=int)
        self.area=float(body.area_faces[self.ids].sum())
        self.mesh=C.trimesh.Trimesh(body.vertices.copy(),body.faces[self.ids].copy(),process=False)
        key=(tuple(self.ids),float(task.domain.data['load']['cone_half_deg']))
        if key not in ACCESS_CACHE:
            access=WorkAccess(body,self.ids,key[1]);display_length=float(body.extents.max())*.5
            ACCESS_CACHE[key]=(access.mesh(display_length),access.definition(),display_length)
        self.region,self.access_definition,self.display_length=ACCESS_CACHE[key]
        rng=np.random.default_rng(1701+int(pose.split('_')[-1]))
        faces=rng.choice(self.ids,1024,p=body.area_faces[self.ids]/self.area)
        uv=rng.random((len(faces),2));root=np.sqrt(uv[:,0])
        bary=np.column_stack([1-root,root*(1-uv[:,1]),root*uv[:,1]])
        candidates=np.einsum('av,avc->ac',bary,body.triangles[faces])
        # Area-weighted candidates, then spread arrows out instead of clustering.
        chosen=[0];distances=np.full(len(candidates),np.inf)
        for _ in range(27):
            distances=np.minimum(distances,np.sum((candidates-candidates[chosen[-1]])**2,axis=1))
            chosen.append(int(np.argmax(distances)))
        self.face_ids=faces[chosen];self.barycentric=bary[chosen]
        self.points=candidates[chosen];self.outward=body.face_normals[self.face_ids].copy()
        self.length=.009
        # A tiny display offset leaves the tip visible on the original surface.
        # The shaft starts OUTSIDE the object and points inward to that surface.
        self.arrows=C.trimesh.util.concatenate([
            arrow_mesh(point+(self.length+2e-5)*normal,-normal,self.length)
            for point,normal in zip(self.points,self.outward)])

    def layers(self,transform,arrows=False,region_transform=None):
        # Draw the exact coplanar work faces before the gray body (depth ties).
        layers=[(transform_mesh(self.mesh,transform),WORK)]
        if arrows:layers.append((transform_mesh(self.arrows,transform),FORCE))
        region_transform=transform if region_transform is None else region_transform
        layers.append((transform_mesh(self.region,region_transform),FORBIDDEN,ACCESS_ALPHA))
        return layers

    def record(self):
        return dict(source='original task.domain.work_ids',face_ids=self.ids.tolist(),area_cm2=self.area*1e4,
            forbidden_access_region=self.access_definition,display_region_length_m=self.display_length,
            display_region_is_finite_excerpt_of_unbounded_exclusion=True,
            sample_face_ids=self.face_ids.tolist(),sample_barycentric=self.barycentric.tolist(),
            sample_points_object_m=self.points.tolist(),force_directions_object=(-self.outward).tolist(),
            arrow_count=len(self.points),arrow_length_m=self.length,
            sampling='area-weighted surface candidates, spatially distributed',
            arrow_meaning='possible inward applied forces, not support reactions or exit directions',
            sample_arrows_are_force_witnesses=False)


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
        self.work=[WorkingSurface(body,pose) for pose in self.poses]
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
            points.append(C.transform_points(self.work[k].arrows.vertices,layout.placements[k]))
            points.append(C.transform_points(self.work[k].region.vertices,layout.placements[k]))
            origin=body.vertices.mean(0);origin[2]=body.bounds[1,2]+.005
            points.append(arrow_mesh(origin,layout.directions[k],.045).vertices)
        return np.vstack(points)

    def result_points(self):
        points=[]
        for k in self.final.active:
            frame=self.native[self.final.hosts[k]]
            points.append(C.transform_points(self.support.vertices,frame))
            points.append(C.transform_points(self.body.vertices,frame @ self.final.placements[k]))
            points.append(C.transform_points(self.work[k].arrows.vertices,frame @ self.final.placements[k]))
            points.append(C.transform_points(self.work[k].region.vertices,frame @ self.final.placements[k]))
        return np.vstack(points)


def render_scene(panel,layers):
    def raster(parts):
        pixels=np.full((panel.height,panel.width,3),255,np.uint8)
        depth=np.full((panel.height,panel.width),-np.inf)
        triangles=[];colors=[]
        for mesh,color in parts:
            if not len(mesh.faces):continue
            normals=mesh.face_normals;keep=normals @ panel.camera>0
            ts,ns=mesh.triangles[keep],normals[keep];q=ts-panel.center
            projected=np.stack([q @ panel.right*panel.scale+panel.width/2,
                -q @ panel.up*panel.scale+panel.height/2,ts @ panel.camera],axis=2)
            shade=.60+.40*np.maximum(ns @ panel.light,0)
            rgb=np.array(ImageDraw.ImageColor.getrgb(color))
            triangles.append(projected);colors.append((shade[:,None]*rgb).astype(np.uint8))
        if triangles:RASTER(np.concatenate(triangles),np.concatenate(colors),pixels,depth,0,0,panel.width,panel.height)
        return pixels,depth
    image,depth=raster([(layer[0],layer[1]) for layer in layers if len(layer)==2])
    for mesh,color,alpha in [layer for layer in layers if len(layer)==3]:
        overlay,z=raster([(mesh,color)])
        visible=np.isfinite(z)&(z>=depth-1e-9)
        image[visible]=np.rint((1-alpha)*image[visible]+alpha*overlay[visible]).astype(np.uint8)
        outline=visible&~binary_erosion(visible)
        image[outline]=np.rint(.60*image[outline]+.40*np.array(ImageDraw.ImageColor.getrgb(color))).astype(np.uint8)
    return Image.fromarray(image)


def render_view(panel,layers):
    return np.asarray(render_scene(panel,layers))


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
            frame_size=list(SIZE),onscreen_text=False,view_count=1,
            method_order=['whole','incremental'],method_presentation='sequential',
            camera=dict(projection='orthographic',elevation_degrees=ELEVATION,azimuth_degrees=AZIMUTH,fixed=True),
            file=self.path.name,sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())
        return record


def camera(points):
    return Panel(None,size=SIZE,elevation=ELEVATION,azimuth=AZIMUTH,points=points)


def process_layers(scene,progress,forces=False,transition_index=None):
    position=progress*(len(scene.rows)-1)
    index=min(len(scene.rows)-2,int(position))
    if transition_index is not None:index=transition_index
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
    layers=scene.work[k].layers(q,forces)+[(body,GRAY)]+materials+[(arrow,YELLOW)]
    # The sweep cue is a displaced silhouette along the changing saved path.
    # It adds no support material and is never carved as a design relocation.
    if u<.55 and np.linalg.norm(old_d-after.directions[k])>1e-6:
        ghost=body.copy();ghost.apply_translation(.055*d)
        layers.append((ghost,'#cfebf5'))
    return layers,dict(step=next_index,phase=row['phase'],operation_progress=u,
                       object_to_fixture=q.tolist(),direction_fixture=d.tolist(),pose=scene.poses[k],
                       work_area_visible=True,forbidden_access_region_visible=True,possible_force_arrows_visible=forces)


def process_movie(directory,scenes):
    points=np.vstack([scene.process_points() for scene in scenes])
    base=camera(points)
    movie=Movie(directory/'process.mp4');timeline=[]
    segments=[]
    for method_index,scene in enumerate(scenes):
        start=movie.frames;transitions=max(1,len(scene.rows)-1)
        motion_frames=18
        for step in range(transitions):
            for n in range(motion_frames):
                progress=(step+n/(motion_frames-1))/transitions
                layers,record=process_layers(scene,progress,transition_index=step)
                movie.emit(render_view(base,layers))
                timeline.append(dict(frame=movie.frames-1,method=scene.out.name,**record))
            layers,record=process_layers(scene,(step+1)/transitions,forces=True,transition_index=step)
            still=render_view(base,layers)
            for _ in range(FPS):
                movie.emit(still)
                timeline.append(dict(frame=movie.frames-1,method=scene.out.name,**record,display_phase='seated_pause'))
        # Withdraw the final selected object along its saved exit, with no orbit.
        last=scene.layouts[-1];k=chosen_owner(scene.rows[-1],last,scene.poses)
        q=last.placements[k];d=last.directions[k];body=transform_mesh(scene.body,q)
        travel=max(.22,float((scene.support.vertices @ d).max()-(body.vertices @ d).min())+.03)
        for n in range(18):
            moved=q.copy();moved[:3,3]+=travel*smooth(n/17)*d
            layers=scene.work[k].layers(moved)+[(transform_mesh(scene.body,moved),GRAY),(scene.support,BLUE)]
            movie.emit(render_view(base,layers))
            timeline.append(dict(frame=movie.frames-1,method=scene.out.name,phase='final_object_exit'))
        still=render_view(base,[(scene.support,BLUE)])
        for _ in range(FPS):
            movie.emit(still)
            timeline.append(dict(frame=movie.frames-1,method=scene.out.name,phase='empty_support_isometric'))
        segments.append(dict(method=scene.out.name,start_frame=start,end_frame=movie.frames-1))
        if method_index<len(scenes)-1:
            for _ in range(8):
                movie.emit(np.full((SIZE[1],SIZE[0],3),255,np.uint8))
                timeline.append(dict(frame=movie.frames-1,phase='method_separator'))
    return movie.finish(dict(kind='saved search operations',timeline=timeline,
        method_segments=segments,work_area_color=WORK,forbidden_region_color=FORBIDDEN,possible_force_color=FORCE,seated_pause_seconds=1,
        added_color=GREEN,removed_color=RED,representation='direct triangle Boolean meshes',
        interpolated_object_motion_is_design_configuration_change=True,
        relocation_path_carved=False,search_rerun=False,force_acceptance_run=False))


def result_layers(scene,index,phase,u):
    layout=scene.final;k=layout.active[index]
    frame=scene.native[layout.hosts[k]]
    q=layout.placements[k]
    support=transform_mesh(scene.support,frame)
    body=transform_mesh(scene.body,frame @ q)
    d=frame[:3,:3] @ layout.directions[k]
    travel=max(.22,float((support.vertices @ d).max()-(body.vertices @ d).min())+.03)
    if phase=='move_empty_fixture':
        previous=layout.active[max(index-1,0)]
        old=scene.native[layout.hosts[previous]]
        empty=interpolate(old,frame,smooth(u),scene.support.center_mass)
        if index and not np.allclose(old,frame,atol=1e-12,rtol=0):
            empty[2,3]+=.018*np.sin(np.pi*u)
        return [(transform_mesh(scene.support,empty),BLUE)],dict(pose=scene.poses[k],phase='move_empty_fixture',
            support_transform=empty.tolist(),object_visible=False)
    if phase=='insert':
        offset=travel*(1-smooth(u))
    elif phase=='seated':
        offset=0.
    else:
        assert phase=='exit'
        offset=travel*smooth(u)
    object_transform=frame @ q;object_transform[:3,3]+=offset*d
    body=transform_mesh(scene.body,object_transform)
    arrows=phase=='seated'
    layers=scene.work[k].layers(object_transform,arrows,region_transform=frame @ q)+[(body,GRAY),(support,BLUE)]
    return layers,dict(pose=scene.poses[k],phase=phase,
        support_transform=frame.tolist(),object_transform=(frame @ q).tolist(),
        object_visible=True,exit_direction_world=d.tolist(),exit_offset_m=offset,travel_m=travel,
        work_area_visible=True,forbidden_access_region_visible=True,possible_force_arrows_visible=arrows,force_arrow_count=len(scene.work[k].points) if arrows else 0)


def result_movie(directory,scenes):
    points=np.vstack([scene.result_points() for scene in scenes])
    base=camera(points)
    movie=Movie(directory/'result.mp4');timeline=[];segments=[]
    for method_index,scene in enumerate(scenes):
        start=movie.frames
        for index in range(len(scene.final.active)):
            for phase,frames in POSE_PHASE_FRAMES.items():
                still=None
                for n in range(frames):
                    if phase=='seated' and still is not None:
                        pixels=still
                    else:
                        layers,record=result_layers(scene,index,phase,n/max(1,frames-1))
                        pixels=render_view(base,layers)
                        if phase=='seated':still=pixels
                    movie.emit(pixels)
                    timeline.append(dict(frame=movie.frames-1,method=scene.out.name,**record))
        frame=scene.native[scene.final.hosts[scene.final.active[-1]]]
        still=render_view(base,[(transform_mesh(scene.support,frame),BLUE)])
        for _ in range(FPS):
            movie.emit(still)
            timeline.append(dict(frame=movie.frames-1,method=scene.out.name,phase='empty_support_isometric'))
        segments.append(dict(method=scene.out.name,start_frame=start,end_frame=movie.frames-1))
        if method_index<len(scenes)-1:
            for _ in range(8):
                movie.emit(np.full((SIZE[1],SIZE[0],3),255,np.uint8))
                timeline.append(dict(frame=movie.frames-1,phase='method_separator'))
    return movie.finish(dict(kind='all saved poses use one rigid support',timeline=timeline,
        method_segments=segments,work_area_color=WORK,forbidden_region_color=FORBIDDEN,possible_force_color=FORCE,seated_pause_seconds=1,
        per_pose_phase_frames=POSE_PHASE_FRAMES,
        working_surfaces={scene.poses[k]:scene.work[k].record() for k in range(len(scene.poses))},
        same_support_shape_in_every_frame=True,object_removed_before_fixture_moves=True,
        original_world_object_orientation_and_seated_height_preserved=True,
        representation='direct triangle Boolean meshes',search_rerun=False,force_acceptance_run=False))


def pictures(scene):
    # No heading, index, labels, captions or legend in process.png.
    points=scene.process_points();panel=Panel(None,size=(720,570),points=points,elevation=ELEVATION,azimuth=AZIMUTH)
    columns=3;canvas=Image.new('RGB',(columns*720,math.ceil(len(scene.rows)/columns)*570),'white')
    for index,(row,layout,mesh) in enumerate(zip(scene.rows,scene.layouts,scene.meshes)):
        k=chosen_owner(row,layout,scene.poses)
        body=transform_mesh(scene.body,layout.placements[k])
        origin=body.vertices.mean(0);origin[2]=body.bounds[1,2]+.005
        arrow=arrow_mesh(origin,layout.directions[k],.045)
        x,y=(index%columns)*720,(index//columns)*570
        layers=scene.work[k].layers(layout.placements[k])+[(body,GRAY),(mesh,BLUE),(arrow,YELLOW)]
        canvas.paste(render_scene(panel,layers),(x,y))
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
        region=transform_mesh(scene.work[k].region,frame @ scene.final.placements[k])
        panel=Panel(None,size=(600,510),points=np.vstack([support.vertices,body.vertices,arrow.vertices,region.vertices]),elevation=ELEVATION,azimuth=AZIMUTH)
        x,y=(index%3)*610,80+(index//3)*570
        draw.text((x+24,y+12),scene.poses[k]+' | '+('rotate fixture' if registered(scene.final,k) else 'Juxtapose -> '+scene.poses[scene.final.hosts[k]]),font=font(20,True),fill='#253441')
        layers=scene.work[k].layers(frame @ scene.final.placements[k])+[(body,GRAY),(support,BLUE),(arrow,YELLOW)]
        result.paste(render_scene(panel,layers),(x+5,y+45))
    result.save(scene.out/'final_result.png')
    C.save(scene.out/'render.json',dict(complete=True,triangle_mesh=True,voxel_surface=False,
        process_png_onscreen_text=False,process_png_layout='chronological; one isometric view per selected state',
        view_count_per_state=1,camera=dict(elevation_degrees=ELEVATION,azimuth_degrees=AZIMUTH),
        working_surfaces_from_original_tasks=True,work_area_color=WORK,possible_force_color=FORCE,
        forbidden_access_regions_drawn=True,forbidden_region_color=FORBIDDEN,
        work_access_definition=[work.access_definition for work in scene.work],
        source_mesh='support.obj',source_mesh_sha256=hashlib.sha256((scene.out/'support.obj').read_bytes()).hexdigest(),
        material_volume_cm3=scene.geometry['final_material_volume_cm3'],search_sampling_unchanged=True,
        final_acceptance_run=False,same_rigid_support_in_every_final_pose=True))
    path=scene.out/'README.md'
    if not path.exists():
        from render_fast_process import write_details
        rows=[]
        for i,record in enumerate(scene.geometry['steps']):
            def volume(kind):return abs(float(scene.deltas[i][kind].volume))*1e6
            rows.append(dict(grid_added_cm3=volume('added'),grid_removed_cm3=volume('removed')))
        write_details(scene.out,scene.rows,scene.report,rows)
    text=path.read_text()
    text=text.split('\n\n当前 mesh：',1)[0]
    text=text.replace('图里蓝色是同一采样材料模型的 1.25 mm 网格边界','旧网格图已经存档。图里蓝色是从保存布局直接 Boolean 构造的三角网格边界')
    text=text.replace('过程图不含文字；按行从左到右，每一对的左侧只看支撑、右侧加入该步相关物体',
        '过程图不含文字；按行从左到右，每步只有一个等轴测视角，显示支撑、相关物体和橙色工作面积')
    text=text.replace('旧旧显示网格','旧显示网格')
    text=text.replace('工作带','完整工作锥禁入区域')
    text=text.replace('图里蓝色是同一采样材料模型的 1.25 mm 网格边界','图里蓝色是当前布局直接 Boolean 构造的三角网格边界')
    text=text.replace('过程图左侧只看支撑，右侧加入该步相关物体','过程图每步只有一个固定等轴测视角')
    text=text.replace('显示网格 + / − cm³','mesh + / − cm³')
    path.write_text(text+'\n\n当前 mesh：[support.obj](support.obj)。本集合的两个视频位于上一级：'
        '[搜索过程](../process.mp4)、[逐 pose 使用](../result.mp4)。固定单个等轴测视角，先 whole、后 incremental，'
        '中间短暂白场分隔。橙色面片为各 pose 原始工作面积；放稳后停顿 1 秒，工作面上的橙红色小箭头朝内，表示可能施加的力。'
        '半透明琥珀色区域为该 pose 的 30° 工作锥禁入区。图中仅截取有限长度；算法排除整个半无限区域，并对全部活动 pose 同时执行。'
        f'该方法最终三角网格材料体积 {scene.geometry["final_material_volume_cm3"]:.3f} cm³；这是加入完整工作锥后重新搜索的结果，仍未运行最终压力证书验收。\n')


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
        view_count=1,method_order=['whole','incremental'],method_presentation='sequential',
        geometry_preserved=True,final_acceptance_run=False,
        seconds=time.monotonic()-began))


if __name__=='__main__':main()
