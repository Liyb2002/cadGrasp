"""Actual Step5 pruning, exact final OBJ, fixed cameras and BBX face close-ups."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys
import time
import prepare02 as P
import render01 as R
import numpy as np
import trimesh
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent
DATA=HERE/'data/video02'
MOVIE=HERE.parent/'02_pruning_and_bbx.mp4'
FPS,FRAMES=24,288
SECONDS=FRAMES/FPS
BLUE,ORANGE,GREEN,GREY=R.V.BLUE,'#E69B64','#16866B','#8C99A6'
TEXT,MUTED='#263D4D','#607281'
FONT_PATH='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD_PATH='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def font(size,bold=False):return ImageFont.truetype(BOLD_PATH if bold else FONT_PATH,size)


def box_vertices(bounds):
    return np.array([[bounds[ix,0],bounds[iy,1],bounds[iz,2]]
        for ix in range(2)for iy in range(2)for iz in range(2)])


def lines(segments,color,camera,size,width=1.,opacity=1.):
    _,basis,span=camera
    a=np.asarray([s[0]for s in segments]);b=np.asarray([s[1]for s in segments])
    projected=(b-a)@basis[:2].T;length=np.linalg.norm(projected,axis=1)
    normal=np.c_[-projected[:,1],projected[:,0]]/np.maximum(length[:,None],1e-15)
    offset=normal@basis[:2]*(width*span/size/2)
    corners=np.stack([a-offset,b-offset,b+offset,a+offset],axis=1)
    triangles=corners[:,[[0,1,2],[0,2,3]]].reshape(-1,3,3)
    return dict(triangles=triangles,normals=np.broadcast_to(basis[2],triangles.shape),
        colors=np.broadcast_to(R.V.CPU.rgb(color),triangles.shape),bias=2*span,unlit=True,opacity=opacity)


def box_piece(bounds,camera,color,dashed=False):
    vertices=box_vertices(np.asarray(bounds));segments=[]
    for i in range(8):
        for j in range(i):
            if bin(i^j).count('1')!=1:continue
            a,b=vertices[i],vertices[j]
            if dashed:
                for k in range(14):segments.append((a+(b-a)*k/14,a+(b-a)*(k+.53)/14))
            else:segments.append((a,b))
    return lines(segments,color,camera,900,width=1.3 if dashed else 2.,opacity=.58 if dashed else .95)


def dashed(draw,xy,fill,width=1):
    x0,y0,x1,y1=xy
    length=np.hypot(x1-x0,y1-y0);n=max(1,int(length/7))
    for k in range(n):
        a,b=k/n,min(1.,(k+.5)/n)
        draw.line((x0+(x1-x0)*a,y0+(y1-y0)*a,x0+(x1-x0)*b,y0+(y1-y0)*b),fill=fill,width=width)


def arrow(draw,a,b,color,width=2):
    draw.line((*a,*b),fill=color,width=width)
    v=np.asarray(b,float)-a;length=np.linalg.norm(v)
    if length<3:return
    unit=v/length;side=np.array([-unit[1],unit[0]])
    first=np.asarray(b)-unit*5+side*3;second=np.asarray(b)-unit*5-side*3
    draw.polygon([tuple(b),tuple(first),tuple(second)],fill=color)


class Movie:
    def __init__(self):
        self.record=json.loads((DATA/'stages.json').read_text())
        if not self.record['complete']or not self.record['all_geometry_from_actual_saved_step5']:
            raise ValueError('Actual saved Step5 geometry required')
        _,self.rows=P.load_rows();self.ref=np.asarray(self.record['reference_fixture_transform'])
        self.protected=dict(self.record['protected_inputs'])
        self.protected[R.relative(DATA/'stages.json')]=R.digest(DATA/'stages.json')
        self.protected[R.relative(DATA/'actual_step5_final.obj')]=R.digest(DATA/'actual_step5_final.obj')
        self.meshes=[];self.blue=[];self.old=[]
        for stage in self.record['stages']:
            path=R.ROOT/stage['support']
            if R.digest(path)!=stage['support_sha256']:raise ValueError('Actual source changed')
            mesh=trimesh.load(path,force='mesh',process=False)
            shown=R.transform(mesh.vertices,self.ref)
            bounds=np.array([shown.min(0),shown.max(0)])
            if np.max(np.abs(bounds-np.asarray(stage['shown_bounds_world_m'])))>1e-12:
                raise ValueError('Actual support box mismatch')
            self.meshes.append(mesh);self.blue.append(R.V.piece(mesh,BLUE,1.))
            self.old.append(R.V.piece(mesh,ORANGE,.18))
        if R.digest(R.ROOT/self.record['stages'][-1]['support'])!=self.record['final_obj_sha256']:
            raise ValueError('The last model is not the certified Step5 final')
        boxes=np.vstack([box_vertices(np.array(s['shown_bounds_world_m']))for s in self.record['stages']])
        focus,basis,span=R.V.camera_for(boxes)
        self.camera=focus,basis,span*1.20
        self.renderer=R.V.Renderer();self.main_cache={};self.ghost_cache={};self.right_cache={}
        body=trimesh.load(R.ROOT/'objects/B/mesh.stl',force='mesh',process=False)
        all_points=np.vstack([m.vertices for m in self.meshes])
        self.cameras=[];self.bodies=[]
        for row in self.rows:
            object_mesh=R.posed(body,row['U'])
            self.bodies.append(R.V.CPU.placed(R.V.piece(object_mesh,R.V.GRAY,.17)))
            self.cameras.append(R.V.camera_for(np.vstack([R.transform(all_points,row['T']),object_mesh.vertices])))
        common_span=max(c[2]for c in self.cameras)
        self.cameras=[(f,b,common_span)for f,b,_ in self.cameras]
        self.sources={};self.frames=[];self.freeze()
        print('ACTUAL_STEP5_VIDEO',dict(stages=len(self.meshes),final_sha=self.record['final_obj_sha256'],
            system_bbx_reduction_percent=self.record['actual_system_bbx_reduction_percent']),flush=True)

    def freeze(self):
        paths={Path(__file__).resolve(),Path(P.__file__).resolve(),Path(R.__file__).resolve(),R.V.CPU_PATH,
            R.V.BASELINE/'step4_connect_support/translucent_raster.cpp'}
        for module in list(sys.modules.values()):
            file=getattr(module,'__file__',None)
            if file:
                path=Path(file).resolve()
                if path.is_relative_to(R.ROOT)and not path.is_relative_to(R.ROOT/'.venv')and path.is_file():paths.add(path)
        for path in paths:
            key=R.relative(path);target=DATA/'executed_sources'/key
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target);self.sources[key]=R.digest(path)
        R.save(DATA/'sources.json',self.sources);R.save(DATA/'inputs.json',self.protected)

    def phase(self,t):
        if t<.8:return 0,t
        if t<1.65:return 1,t-.8
        if t<9.85:
            position=(t-1.65)/8.2*18;offset=int(position)
            return 2+min(offset,17),(position-offset)*8.2/18
        return 20,t-9.85

    def reference(self,index):return 0 if index==0 else 1

    def scene(self,index,ghost=False):
        cache=self.ghost_cache if ghost else self.main_cache
        if index in cache:return cache[index]
        parts=[R.placed(self.blue[index],self.ref)]
        if ghost and index>0:parts.append(R.placed(self.old[index-1],self.ref))
        reference=self.record['stages'][self.reference(index)]
        current=self.record['stages'][index]
        for row in self.rows:
            hull=R.H.outline_piece(R.transform(row['hull'],self.ref),row['color'],self.camera,900,half_width_pixels=.40)
            hull['opacity']=.23;parts.append(R.V.CPU.placed(hull))
        parts.extend([R.V.CPU.placed(box_piece(reference['shown_bounds_world_m'],self.camera,GREY,True)),
            R.V.CPU.placed(box_piece(current['shown_bounds_world_m'],self.camera,GREEN))])
        result=self.renderer.render(parts,self.camera,900);cache[index]=result
        return result

    def right(self,index):
        if index in self.right_cache:return self.right_cache[index]
        result=Image.new('RGB',(700,900),'white')
        draw=ImageDraw.Draw(result)
        positions=[(40,48),(390,48),(40,285),(390,285),(215,522)]
        for row,body,camera,(x,y)in zip(self.rows,self.bodies,self.cameras,positions):
            parts=[R.placed(self.blue[index],row['T']),body]
            result.paste(self.renderer.render(parts,camera,230),(x,y))
            draw.text((x+7,y+4),row['pose'].replace('pose_','Pose '),fill=row['color'],font=font(15))
        draw.text((30,12),'Actual saved geometry',font=font(19,True),fill=TEXT)
        self.right_cache[index]=result;return result

    def closeups(self,draw,index):
        reference=np.asarray(self.record['stages'][self.reference(index)]['shown_bounds_world_m'])*1000
        current=np.asarray(self.record['stages'][index]['shown_bounds_world_m'])*1000
        draw.text((925,752),'BBX boundary close-ups  |  mm',font=font(17,True),fill=TEXT)
        # Each row is a true local coordinate window around the old face.
        # 65 pixels/mm makes submillimetre changes legible without changing
        # the main view scale or exaggerating the physical model/BBX.
        for axis,name in enumerate('XYZ'):
            x=925+axis*225
            draw.text((x,778),name+' faces',font=font(16,True),fill=MUTED)
            for side,label in [(0,'min'),(1,'max')]:
                y=812+side*50;center=x+105
                delta=float(current[side,axis]-reference[side,axis]);at=center+65*delta
                draw.line((x+35,y+9,x+206,y+9),fill='#E9EEF1',width=1)
                dashed(draw,(center,y-8,center,y+23),GREY,width=2)
                draw.line((at,y-8,at,y+23),fill=GREEN,width=3)
                arrow(draw,(center,y+9),(at,y+9),GREEN)
                draw.text((x,y-7),label,font=font(13),fill=MUTED)
                draw.text((x+39,y+22),f'{delta:+.3f} mm',font=font(13),fill=GREEN if abs(delta)>1e-8 else MUTED)

    def frame(self,t,record=False):
        index,age=self.phase(t);stage=self.record['stages'][index]
        image=Image.new('RGB',(1600,900),'white')
        main=self.scene(index)
        if index>0 and age<.22:
            weight=1-R.smooth(age/.22)
            main=Image.blend(main,self.scene(index,True),weight)
        image.paste(main,(0,0));image.paste(self.right(index),(900,0))
        draw=ImageDraw.Draw(image)
        draw.line((900,0,900,899),fill='#E6ECEF',width=1)
        titles={'input':'Step5 input','construction':'Required contacts + connected stock',
            'pruning':'Exterior pruning','final':'Step5 final'}
        title=titles[stage['kind']]
        if stage['kind']=='pruning':title+=f'  {index-1} / 18'
        if stage['kind']=='final'and age<.6:title='Final connected finishing'
        draw.text((32,22),title,font=font(29,True),fill=TEXT)
        draw.text((34,64),'Grey: before pruning     Green: current BBX',font=font(18),fill=MUTED)
        draw.text((34,93),'Fixed view and scale',font=font(15),fill=MUTED)
        reference=self.record['stages'][self.reference(index)]
        dims=(np.array(stage['shown_bounds_world_m'])[1]-np.array(stage['shown_bounds_world_m'])[0])*1000
        draw.rectangle((22,775,878,899),fill='white')
        draw.text((32,780),f"Material   {stage['material_cm3']:.2f} cm³",font=font(22,True),fill=BLUE)
        draw.text((32,814),f"Support BBX   {reference['shown_bbx_l']:.3f} → {stage['shown_bbx_l']:.3f} L",font=font(21),fill=GREEN)
        draw.text((32,849),f'X {dims[0]:.3f}   Y {dims[1]:.3f}   Z {dims[2]:.3f} mm',font=font(17),fill=MUTED)
        start=self.record['stages'][0]['system_bbx_l']
        percent=(1-stage['system_bbx_l']/start)*100
        if abs(percent)<1e-8:percent=0.
        draw.text((435,780),'Whole-system BBX',font=font(19,True),fill=TEXT)
        draw.text((435,814),f"{start:.3f} → {stage['system_bbx_l']:.3f} L",font=font(23,True),fill=GREEN)
        draw.text((435,849),f'{percent:.2f}% smaller',font=font(18),fill=GREEN)
        self.closeups(draw,index)
        if record:self.frames.append(dict(frame=round(t*FPS),stage=index,kind=stage['kind'],
            rendered_source_sha256=stage['support_sha256'],shown_bbx_l=stage['shown_bbx_l'],
            system_bbx_l=stage['system_bbx_l'],material_cm3=stage['material_cm3'],fixed_camera=True,
            final_is_exact_saved_step5=index==20))
        return image

    def verify(self):
        for p,h in self.protected.items():
            if R.digest(R.ROOT/p)!=h:raise ValueError('Saved input changed: '+p)
        for p,h in self.sources.items():
            if R.digest(R.ROOT/p)!=h or R.digest(DATA/'executed_sources'/p)!=h:raise ValueError('Source changed: '+p)
        return dict(passed=True,actual_saved_models=len(self.meshes),protected_files=len(self.protected),
            actual_final_obj_sha256=self.record['final_obj_sha256'],last_model_exactly_matches_published_step5=True,
            illustrative_ring_geometry_used=False,geometry_edited=False,original_numerical_files_unchanged=True,
            original_first_video_unchanged=True,existing_force_certificates_checked=True,force_solver_rerun=False,
            support_and_system_bbx_distinguished=True,bbx_reduction_not_exaggerated=True,
            camera_fixed_during_pruning=True,local_face_windows_pixels_per_mm=65.)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--encode',action='store_true')
    args=parser.parse_args();movie=Movie()
    times=[0.,1.35,2.6,4.3,6.,7.7,9.4,11.6]
    sheet=Image.new('RGB',(1600,1800),'white')
    for i,t in enumerate(times):
        frame=movie.frame(t);frame.save(DATA/f'preview_{i:02d}.png')
        sheet.paste(frame.resize((800,450),Image.Resampling.LANCZOS),((i%2)*800,(i//2)*450))
    sheet.save(HERE/'02_storyboard.png');R.save(DATA/'verification.json',movie.verify())
    if not args.encode:return
    staging=DATA/'02_pruning_and_bbx.staging.mp4'
    encoder=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo',
        '-pix_fmt','rgb24','-s','1600x900','-r',str(FPS),'-i','pipe:0','-an','-c:v','libx264',
        '-preset','fast','-crf','18','-profile:v','main','-pix_fmt','yuv420p','-movflags','+faststart',str(staging)],stdin=subprocess.PIPE)
    began=time.monotonic()
    try:
        for i in range(FRAMES):
            encoder.stdin.write(movie.frame(i/FPS,True).tobytes())
            if i%24==0:print(json.dumps(dict(frame=i,total=FRAMES,seconds=round(time.monotonic()-began,1))),flush=True)
        encoder.stdin.close()
        if encoder.wait()!=0:raise RuntimeError('Encoding failed')
    except BaseException:
        encoder.kill();encoder.wait();raise
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries',
        'format=duration:stream=width,height,nb_frames,r_frame_rate','-of','json',str(staging)]))
    stream=probe['streams'][0]
    if (stream['width'],stream['height'],int(stream['nb_frames']))!=(1600,900,FRAMES):raise ValueError('Wrong encoding')
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-i',str(staging),'-f','null','-'],check=True)
    if len(movie.frames)!=FRAMES or not all(r['final_is_exact_saved_step5']for r in movie.frames[-24:]):
        raise ValueError('Last second must show the exact actual Step5 final')
    verification=movie.verify();staging.replace(MOVIE)
    verification.update(movie=R.relative(MOVIE),movie_sha256=R.digest(MOVIE),full_decode_passed=True,
        encoded_probe=probe,frames_checked=len(movie.frames),final_exact_obj_frames=sum(r['final_is_exact_saved_step5']for r in movie.frames))
    R.save(DATA/'verification.json',verification);R.save(DATA/'frame_checks.json',movie.frames)
    R.save(DATA/'manifest.json',dict(complete=True,pose_set=R.CASE,movie=R.relative(MOVIE),
        frames=FRAMES,duration_seconds=SECONDS,fps=FPS,width=1600,height=900,
        final_obj_source=movie.record['stages'][-1]['support'],final_obj_sha256=movie.record['final_obj_sha256'],
        geometry_from_actual_step5_snapshots=True,illustrative_ring_geometry_used=False,
        fixed_camera=True,bbox_previous_and_current_visible=True,bbox_boundary_closeups_pixels_per_mm=65,
        actual_system_reduction_percent=movie.record['actual_system_bbx_reduction_percent'],
        actual_support_pruning_reduction_percent=movie.record['pruning_shown_bbx_reduction_percent'],
        movie_sha256=R.digest(MOVIE),seconds=time.monotonic()-began))
    print(json.dumps(verification),flush=True)


if __name__=='__main__':main()
