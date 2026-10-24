"""Two-panel geometry review: grounded task use and the same empty fixture turn.

The right panel is a centered orientation diagram, not a handling trajectory.
Work installations and original exits are independently certified in Step5;
this film makes no robot, grasp or collision-free handling-path claim.
"""
import json
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial.transform import Rotation,Slerp
import trimesh

from whole_task.step1.common import sha256
from whole_task.step1.needle_render import Renderer,sector_lines
from whole_task.step3.render_independent import head_context,BLUE
from whole_task.operations.state import transform
from whole_task.step4.membrane_search import read_group
from .video import raster as r


class Scene(Renderer):
    def shape(self,mesh,color,mask=None):
        shape=r.Shape(self.ctx,self.shader,mesh.vertices,mesh.faces,color,smooth=False,
            dynamic=mask is not None,roughness=.57,metallic=.04)
        if mask is not None:
            colors=np.tile(color,(len(mesh.faces),1));colors[mask]=r.PINK;shape.paint(colors)
        return shape

    def picture(self,items,points,direction,lines=None):
        direction=np.asarray(direction,float);direction/=np.linalg.norm(direction)
        right=np.cross([0,0,1],direction);right/=np.linalg.norm(right);up=np.cross(direction,right)
        basis=np.array([right,up,direction]);p=np.asarray(points)@basis.T
        low,high=p.min(0),p.max(0);target=((low+high)/2)@basis
        scale=max(high[0]-low[0],(high[1]-low[1])*self.width/self.height)*1.16
        vp=r.orthographic(scale,scale*self.height/self.width)@r.view_matrix(direction,target)
        self.shader['view_projection'].write(np.asarray(vp,dtype='f4').T.tobytes())
        self.shader['eye'].value=tuple(target+direction*4)
        key=np.array([-.65,-.9,1.45]);key/=np.linalg.norm(key)
        light=r.orthographic(scale*1.8,scale*1.8)@r.view_matrix(key,target)
        for shader in (self.shader,self.shadow):shader['light_projection'].write(np.asarray(light,dtype='f4').T.tobytes())
        ctx=self.ctx;ctx.enable(self.gl.DEPTH_TEST)
        self.studio.shadow_buffer.use();ctx.viewport=(0,0,1536,1536);self.studio.shadow_buffer.clear(depth=1)
        for shape,T in items:shape.shadow(ctx,self.shadow,T)
        self.studio.gbuffer.use();ctx.viewport=(0,0,*self.studio.size)
        self.studio.gbuffer.clear(1,1,1,1,depth=1);self.studio.shadow.use(3)
        for shape,T in items:shape.draw(self.shader,T)
        if lines is not None:
            self.lines['vp'].write(np.asarray(vp,dtype='f4').T.tobytes());self.lines['ink'].value=(.78,.55,.16)
            buffer=ctx.buffer(np.asarray(lines,dtype='f4').tobytes())
            vao=ctx.vertex_array(self.lines,[(buffer,'3f','position')]);ctx.line_width=1.6
            vao.render(self.gl.LINES);vao.release();buffer.release()
        return Image.frombytes('RGB',(self.width,self.height),self.studio.finish(vp,direction)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def render_movie(model,groups,path,motion_path,size=(1600,800),fps=24):
    width,height=size;panel=width//2;scene=Scene(panel,height-90)
    identity=np.eye(4);view=np.array([.55,1.4,1.05]);records=[];parts=[];fit=[]
    lines=sector_lines(model.workspace);contexts=head_context(model.workspace)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    floor=trimesh.creation.box([2.,2.,.004]);floor.apply_translation([0,-.4,-.002])
    context_shapes=[(scene.shape(m,c),identity) for m,c,_ in contexts]
    floor_shape=scene.shape(floor,(.95,.96,.97))
    for family,group in enumerate(groups):
        q,mesh=read_group(group);center=mesh.bounds.mean(0);shape=scene.shape(mesh,BLUE)
        parts.append(dict(mesh=mesh,shape=shape,center=center))
        for local,owner in enumerate(group['members']):
            mask=np.zeros(len(model.source.faces),bool);mask[model.records[owner]['working_face_ids']]=True
            support_world=np.linalg.inv(q.material[local])
            records.append(dict(owner=owner,family=family,T=support_world,P=q.poses[local],
                body=scene.shape(model.source,r.GRAY,mask)))
            fit.extend([transform(mesh.vertices,support_world),transform(model.source.vertices,q.poses[local])])
    records.sort(key=lambda row:row['owner'])
    fit=np.vstack([*fit,lines,*[m.vertices for m,_,_ in contexts]])
    radius=max(float(np.linalg.norm(p['mesh'].vertices-p['center'],axis=1).max()) for p in parts)
    # Every rotated vertex lies inside the radius sphere. Bound its screen
    # projection directly; a 3-D cube unnecessarily shrinks the empty view.
    diagram_view=np.array([.6,1.4,1.1]);diagram_view/=np.linalg.norm(diagram_view)
    diagram_right=np.cross([0,0,1],diagram_view);diagram_right/=np.linalg.norm(diagram_right)
    diagram_up=np.cross(diagram_view,diagram_right)
    studio_points=np.array([radius*(x*diagram_right+y*diagram_up) for x in (-1,1) for y in (-1,1)])
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    process=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24',
        '-s',f'{width}x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','medium',
        '-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    frames=[];angles=[];review=[]

    def centered(row,rotation=None):
        matrix=row['T'].copy();matrix[:3,:3]=matrix[:3,:3] if rotation is None else rotation
        matrix[:3,3]=-matrix[:3,:3]@parts[row['family']]['center'];return matrix

    def installed(row,body=True):
        objects=[(floor_shape,identity),(parts[row['family']]['shape'],row['T']),*context_shapes]
        if body:objects.append((row['body'],row['P']))
        return scene.picture(objects,fit,view,lines)

    def empty_view(row,rotation=None):
        T=centered(row,rotation)
        return scene.picture([(parts[row['family']]['shape'],T)],studio_points,[.6,1.4,1.1])

    def emit(left,right,row,phase,rotation=None):
        canvas=Image.new('RGB',(width,height),'white');canvas.paste(left,(0,70));canvas.paste(right,(panel,70))
        draw=ImageDraw.Draw(canvas)
        draw.text((24,15),f"{model.ids[row['owner']]}  ·  Set {row['family']+1}  ·  Working installation",font=font,fill='#28485e')
        draw.text((panel+24,15),'Same complete fixture · orientation view',font=font,fill='#28485e')
        draw.text((24,height-25),'Pink: original task area   /   Gold: fixed needle sector',font=small,fill='#688391')
        draw.text((panel+24,height-25),'Empty during the turn',font=small,fill='#688391')
        process.stdin.write(canvas.tobytes())
        frames.append([row['owner'],row['family'],phase]);angles.append(row['T'][:3,:3] if rotation is None else rotation)
        return canvas

    try:
        for k,row in enumerate(records):
            work=installed(row);empty=installed(row,False);right=empty_view(row)
            for _ in range(2*fps):frame=emit(work,right,row,0)
            review.append(frame)
            if k==len(records)-1:continue
            following=records[k+1]
            next_empty=installed(following,False)
            for j in range(fps//4):emit(Image.blend(work,empty,(j+1)/(fps//4)),right,row,1)
            same=row['family']==following['family']
            interpolation=Slerp([0,1],Rotation.from_matrix(np.array([row['T'][:3,:3],following['T'][:3,:3]])))
            for j in range(fps):
                u=(j+.5)/fps;u=u*u*(3-2*u)
                current=row if u<.5 or same else following
                rotation=interpolation(u).as_matrix() if same else current['T'][:3,:3]
                right=empty_view(current,rotation)
                emit(empty if u<.5 else next_empty,right,current,2,rotation)
            next_work=installed(following);right=empty_view(following)
            for j in range(fps//4):emit(Image.blend(next_empty,next_work,(j+1)/(fps//4)),right,following,3)
            print('MOVIE',k+1,'/',len(records),flush=True)
    finally:
        process.stdin.close();scene.close()
    if process.wait()!=0:raise RuntimeError('Final review movie encoding failed')
    np.savez_compressed(motion_path,task_family_phase=np.asarray(frames),right_panel_rotation=np.asarray(angles))
    probe=subprocess.run(['ffprobe','-v','error','-show_entries','stream=width,height,nb_frames,r_frame_rate',
        '-show_entries','format=duration','-of','json',str(path)],check=True,capture_output=True,text=True)
    metadata=json.loads(probe.stdout)
    if int(metadata['streams'][0]['nb_frames'])!=len(frames):raise ValueError('Incomplete encoded movie')
    subprocess.run(['ffmpeg','-v','error','-i',str(path),'-f','null','-'],check=True)
    sheet=Image.new('RGB',(width*3,height*2),'white')
    for j,k in enumerate(np.linspace(0,len(review)-1,6).astype(int)):sheet.paste(review[k],((j%3)*width,(j//3)*height))
    sheet.save(path.with_name('video_review.jpg'))
    return dict(file=str(path),sha256=sha256(path),width=width,height=height,fps=fps,frames=len(frames),
        duration_s=len(frames)/fps,task_count=len(records),actual_final_geometry=True,
        left_panel='certified grounded working installations; same fixed needle',
        right_panel='centered empty-fixture orientation diagram; actual relative installation rotations',
        editing_transitions_are_handling_trajectories=False,robot_motion_certified=False,
        complete_video_decode_passed=True,motion_file=str(motion_path),motion_sha256=sha256(motion_path))
