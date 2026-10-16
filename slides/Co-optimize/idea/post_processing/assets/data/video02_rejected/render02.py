"""Show genuine nested exterior deletions and the measured BBX, without text."""
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
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
DATA = HERE/'data/video02'
MOVIE = HERE.parent/'02_pruning_and_bbx.mp4'
FPS, FRAMES = 24, 288
SECONDS = FRAMES/FPS
START, END = .6, 10.8
BLUE, ORANGE, BOX = R.V.BLUE, '#EA9860', '#657584'


def box_vertices(bounds):
    return np.array([[bounds[ix,0],bounds[iy,1],bounds[iz,2]]
        for ix in range(2) for iy in range(2) for iz in range(2)])


def line_piece(segments, color, camera, size, half_width=.75, opacity=.56):
    _, basis, span = camera
    a = np.array([s[0] for s in segments]);b = np.array([s[1] for s in segments])
    projected = (b-a)@basis[:2].T
    length = np.linalg.norm(projected,axis=1)
    perpendicular = np.c_[-projected[:,1],projected[:,0]]/np.maximum(length[:,None],1e-15)
    offsets = perpendicular@basis[:2]*(half_width*span/size)
    corners = np.stack([a-offsets,b-offsets,b+offsets,a+offsets],axis=1)
    triangles = corners[:,[[0,1,2],[0,2,3]]].reshape(-1,3,3)
    return dict(triangles=triangles,normals=np.broadcast_to(basis[2],triangles.shape),
        colors=np.broadcast_to(R.V.CPU.rgb(color),triangles.shape),
        bias=2.*span,unlit=True,opacity=opacity)


def bbox_piece(bounds,camera,size):
    vertices=box_vertices(np.array(bounds))
    segments=[(vertices[i],vertices[j])for i in range(8)for j in range(i)
        if bin(i^j).count('1')==1]
    return line_piece(segments,BOX,camera,size)


class Movie:
    def __init__(self):
        self.record=json.loads((DATA/'stages.json').read_text())
        if not self.record['complete']:
            raise ValueError('Unfinished presentation geometry')
        first,self.rows=P.load_rows()
        self.protected=dict(self.record['protected_inputs'])
        self.protected[R.relative(DATA/'stages.json')]=R.digest(DATA/'stages.json')
        self.protected[R.relative(R.REFERENCE)]=R.digest(R.REFERENCE)
        for path in (R.COOPT/'idea').glob('*.mp4'):
            self.protected[R.relative(path)]=R.digest(path)
        self.ref=np.asarray(self.record['reference_fixture_transform'])
        self.meshes=[];self.removed=[];self.blue=[];self.orange=[]
        for stage in self.record['stages']:
            path=R.ROOT/stage['support']
            if R.digest(path)!=stage['support_sha256']:
                raise ValueError('Presentation checkpoint changed')
            self.protected[R.relative(path)]=R.digest(path)
            mesh=trimesh.load(path,force='mesh',process=False)
            self.meshes.append(mesh)
            self.blue.append(R.V.piece(mesh,BLUE,.80))
            if stage['removed']:
                path=R.ROOT/stage['removed'];self.protected[R.relative(path)]=R.digest(path)
                removed=trimesh.load(path,force='mesh',process=False)
                self.removed.append(removed)
                part=R.V.piece(removed,ORANGE,.7);part['bias']=8e-6
                self.orange.append(part)
            else:self.removed.append(None);self.orange.append(None)
        if R.digest(R.ROOT/self.record['stages'][0]['support'])!=R.digest(HERE/'data/illustrative_support.obj'):
            raise ValueError('Second movie does not start with first movie endpoint')
        self.focus=np.asarray(first['main_camera']['focus_m'])
        # The box corners also fit at every orbit angle; fixed scale prevents
        # camera zoom from being mistaken for genuine BBX reduction.
        points=box_vertices(np.array(self.record['stages'][0]['bounds_world_m']))
        box_span=2*max(float(np.abs((points-self.focus)@R.basis(a)[:2].T).max())
            for a in np.linspace(0.,2*np.pi,361))*1.08
        self.span=max(first['main_camera']['span_m'],box_span)
        body=trimesh.load(R.ROOT/'objects/B/mesh.stl',force='mesh',process=False)
        all_local=np.vstack([self.meshes[0].vertices,*[r['hull']for r in self.rows]])
        self.cameras=[];self.bodies=[]
        for row in self.rows:
            workpiece=R.posed(body,row['U'])
            self.bodies.append(R.V.CPU.placed(R.V.piece(workpiece,R.V.GRAY,.18)))
            self.cameras.append(R.V.camera_for(np.vstack([R.transform(all_local,row['T']),workpiece.vertices])))
        common_span=max(c[2]for c in self.cameras)
        self.cameras=[(f,b,common_span)for f,b,_ in self.cameras]
        self.renderer=R.V.Renderer()
        self.static_panels={};self.frames=[];self.sources={}
        self.freeze()
        print('RENDER_PREPARED',dict(stages=len(self.meshes),frames=FRAMES,
            initial_bbx_l=self.record['stages'][0]['bbx_l'],final_bbx_l=self.record['stages'][-1]['bbx_l'],
            material_only_steps=self.record['material_only_deletion_steps']),flush=True)

    def freeze(self):
        paths={Path(__file__).resolve(),Path(P.__file__).resolve(),Path(R.__file__).resolve(),R.V.CPU_PATH,
            R.V.BASELINE/'step4_connect_support/translucent_raster.cpp'}
        for module in list(sys.modules.values()):
            file=getattr(module,'__file__',None)
            if file:
                path=Path(file).resolve()
                if path.is_relative_to(R.ROOT)and not path.is_relative_to(R.ROOT/'.venv')and path.is_file():
                    paths.add(path)
        for path in paths:
            key=R.relative(path);target=DATA/'executed_sources'/key
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
            self.sources[key]=R.digest(path)
        R.save(DATA/'sources.json',self.sources)
        R.save(DATA/'inputs.json',self.protected)

    def phase(self,t):
        n=len(self.meshes)-1
        if t<START:return 0,0,0.,0.
        if t>=END:return n,n,1.,0.
        position=(t-START)/(END-START)*n
        event=min(int(position)+1,n);u=position-(event-1)
        if u<.26:
            active=event-1
            alpha=.62*R.smooth(u/.26)
        else:
            active=event
            alpha=.62*(1-R.smooth((u-.26)/.74))
        return active,event,u,alpha

    def parts(self,matrix,camera,size,active,event,alpha):
        parts=[R.placed(self.blue[active],matrix)]
        if event and alpha>1e-5:
            removed=dict(self.orange[event]);removed['opacity']=alpha
            parts.append(R.placed(removed,matrix))
        for row in self.rows:
            hull=R.H.outline_piece(R.transform(row['hull'],matrix),row['color'],camera,size,
                half_width_pixels=.55 if size==R.MAIN else .38)
            hull['colors']=np.broadcast_to(R.V.CPU.rgb(row['color']),hull['triangles'].shape)
            hull['opacity']=.30
            points=R.V.points_piece(R.transform(row['dots'],matrix),row['color'],camera,size,
                half_size_pixels=.95 if size==R.MAIN else .55)
            points['opacity']=.65
            parts.extend([R.V.CPU.placed(hull),R.V.CPU.placed(points)])
        return parts

    def frame(self,t,record=False):
        active,event,u,alpha=self.phase(t)
        stage=self.record['stages'][active]
        camera=self.focus,R.basis(2*np.pi*min(t/(SECONDS-.2),1.)),self.span
        parts=self.parts(self.ref,camera,R.MAIN,active,event,alpha)
        parts.append(R.V.CPU.placed(bbox_piece(stage['bounds_world_m'],camera,R.MAIN)))
        main=self.renderer.render(parts,camera,R.MAIN)
        cache_key=active if alpha<=1e-5 else None
        if cache_key in self.static_panels:
            panel=self.static_panels[cache_key]
        else:
            panel=Image.new('RGB',(700,900),'white')
            for row,body,cam,(x,y)in zip(self.rows,self.bodies,self.cameras,
                    [(905,4),(1251,4),(905,304),(1251,304),(1078,604)]):
                items=self.parts(row['T'],cam,R.SMALL,active,event,alpha)+[body]
                panel.paste(self.renderer.render(items,cam,R.SMALL),(x-900,y))
            if cache_key is not None:self.static_panels[cache_key]=panel
        image=Image.new('RGB',(1600,900),'white');image.paste(main,(0,0));image.paste(panel,(900,0))
        draw=ImageDraw.Draw(image)
        draw.line((900,0,900,899),fill='#eef0f2',width=1)
        for y in [300,600]:draw.line((901,y,1599,y),fill='#f1f2f4',width=1)
        if record:
            actual=R.transform(self.meshes[active].vertices,self.ref)
            measured=np.array([actual.min(0),actual.max(0)])
            error=float(np.max(np.abs(measured-np.array(stage['bounds_world_m']))))
            if error>1e-12:raise ValueError('Displayed BBX differs from actual retained geometry')
            corners=box_vertices(measured)
            projection=(corners-self.focus)@camera[1][:2].T
            if np.max(np.abs(projection))>.5*self.span:
                raise ValueError('Displayed box exceeds the fixed view')
            self.frames.append(dict(frame=round(t*FPS),stage=active,event=event,
                material_cm3=stage['material_cm3'],bbx_l=stage['bbx_l'],
                actual_box_error_m=error,deleted_material_overlay_alpha=alpha))
        return image

    def verify(self):
        for p,h in self.protected.items():
            if R.digest(R.ROOT/p)!=h:raise ValueError('A saved input changed: '+p)
        for p,h in self.sources.items():
            if R.digest(R.ROOT/p)!=h or R.digest(DATA/'executed_sources'/p)!=h:
                raise ValueError('A rendering source changed: '+p)
        return dict(passed=True,protected_files=len(self.protected),source_files=len(self.sources),
            original_first_video_and_numerical_bytes_unchanged=True,
            first_model_matches_video01_endpoint=True,bbx_measured_from_actual_blue_geometry=True,
            every_deletion_is_nested_subset=True,all_displayed_models_connected_and_watertight=True,
            bbx_reduction_steps=self.record['bbx_reduction_steps'],
            material_only_deletion_steps=self.record['material_only_deletion_steps'],
            physical_solver_run=False,explanatory_geometry=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--encode',action='store_true')
    args=parser.parse_args()
    movie=Movie()
    times=[0.,1.3,2.8,4.5,6.3,8.2,10.4,11.8]
    sheet=Image.new('RGB',(1600,1800),'white')
    for i,t in enumerate(times):
        frame=movie.frame(t);frame.save(DATA/f'preview_{i:02d}.png')
        sheet.paste(frame.resize((800,450),Image.Resampling.LANCZOS),((i%2)*800,(i//2)*450))
    sheet.save(HERE/'02_storyboard.png')
    R.save(DATA/'verification.json',movie.verify())
    if not args.encode:return
    staging=DATA/'02_pruning_and_bbx.staging.mp4'
    encoder=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo',
        '-pix_fmt','rgb24','-s','1600x900','-r',str(FPS),'-i','pipe:0','-an',
        '-c:v','libx264','-preset','fast','-crf','18','-profile:v','main','-pix_fmt','yuv420p',
        '-movflags','+faststart',str(staging)],stdin=subprocess.PIPE)
    began=time.monotonic()
    try:
        for i in range(FRAMES):
            encoder.stdin.write(movie.frame(i/FPS,record=True).tobytes())
            if i%FPS==0:print(json.dumps(dict(frame=i,total=FRAMES,seconds=round(time.monotonic()-began,1))),flush=True)
        encoder.stdin.close()
        if encoder.wait()!=0:raise RuntimeError('Encoding failed')
    except BaseException:
        encoder.kill();encoder.wait();raise
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-i',str(staging),'-f','null','-'],check=True)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries',
        'format=duration:stream=width,height,nb_frames,r_frame_rate,codec_name','-of','json',str(staging)]))
    stream=probe['streams'][0]
    if (stream['width'],stream['height'],int(stream['nb_frames']))!=(1600,900,FRAMES):
        raise ValueError('Wrong encoded video format')
    if len(movie.frames)!=FRAMES or any(b['bbx_l']>a['bbx_l']+1e-8 or b['material_cm3']>a['material_cm3']+1e-6
            for a,b in zip(movie.frames,movie.frames[1:])):
        raise ValueError('Invalid material/BBX progression')
    verification=movie.verify();staging.replace(MOVIE)
    verification.update(full_decode_passed=True,encoded_probe=probe,movie=R.relative(MOVIE),
        movie_sha256=R.digest(MOVIE),actual_frame_box_checks=len(movie.frames),
        storyboard=R.relative(HERE/'02_storyboard.png'),storyboard_sha256=R.digest(HERE/'02_storyboard.png'))
    R.save(DATA/'frame_checks.json',movie.frames)
    R.save(DATA/'verification.json',verification)
    R.save(DATA/'manifest.json',dict(complete=True,pose_set=R.CASE,movie=R.relative(MOVIE),
        duration_seconds=SECONDS,fps=FPS,frames=FRAMES,width=1600,height=900,
        text=False,grid=False,main_camera=dict(focus_m=movie.focus.tolist(),fixed_span_m=movie.span),
        deletion_overlay_color=ORANGE,bbx_color=BOX,bbox_definition=movie.record['bbx_definition'],
        first_video_endpoint_geometry_reused=True,cuts_begin_seconds=START,cuts_end_seconds=END,
        initial_material_cm3=movie.record['stages'][0]['material_cm3'],final_material_cm3=movie.record['stages'][-1]['material_cm3'],
        initial_bbx_l=movie.record['stages'][0]['bbx_l'],final_bbx_l=movie.record['stages'][-1]['bbx_l'],
        bbx_reduction_steps=movie.record['bbx_reduction_steps'],material_only_deletion_steps=movie.record['material_only_deletion_steps'],
        original_physical_results_modified=False,physical_solver_run=False,explanatory_geometry=True,
        movie_sha256=R.digest(MOVIE),encoding_seconds=time.monotonic()-began))
    print(json.dumps(verification),flush=True)


if __name__=='__main__':main()
