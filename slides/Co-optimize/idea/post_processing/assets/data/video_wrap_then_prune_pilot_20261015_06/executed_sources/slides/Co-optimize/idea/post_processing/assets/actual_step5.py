"""Two text-free movies of the actual connected-wrap/delete Step5 geometry.

01 grows subsets of saved construction additions; its final solid is the exact
certified initial stock. 02 shows ONLY actual accepted, certified checkpoints.
"""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import render01 as R
import render02 as M
import numpy as np
import trimesh
from PIL import Image

HERE=Path(__file__).resolve().parent
FPS,W,H=24,1600,900

def read(path):return json.loads(Path(path).read_text())

class Movies:
    def __init__(self,experiment):
        self.experiment=experiment
        self.out=R.BASE/'step5'/experiment
        self.report=read(self.out/'data/report.json')
        self.states=read(self.out/'states.json')['states']
        self.data=HERE/'data'/('video_'+experiment);self.data.mkdir(parents=True,exist_ok=True)
        self.protected={};self.frames={1:[],2:[]};self.cache={}
        def load(path,expected=None):
            path=Path(path);actual=R.digest(path)
            if expected and actual!=expected:raise ValueError('Source mesh changed: '+str(path))
            self.protected[R.relative(path)]=actual
            return trimesh.load(path,force='mesh',process=False)
        self.initial=load(self.out/'initial_support.obj',self.report['initial_support_sha256'])
        self.final=load(self.out/'support.obj',self.report['support_sha256'])
        self.checkpoints=[]
        self.boxes=[]
        self.hulls=[]
        for i,entry in enumerate(self.report['demands']):
            path=R.ROOT/entry['arrays'];self.protected[R.relative(path)]=R.digest(path)
            with np.load(path) as z:
                if i==0:self.ref=z['T_world_fixture'].copy()
                local=R.transform(z['floor_demands_world_m'][z['hull_indices']],np.linalg.inv(z['T_world_fixture']))
                self.hulls.append((R.transform(local,self.ref),R.V.COLORS[i]))
        for state in self.states:
            mesh=load(R.ROOT/state['mesh'],state['mesh_sha256']);self.checkpoints.append(mesh)
            world=R.transform(mesh.vertices,self.ref)
            self.boxes.append(np.array([world.min(0),world.max(0)]))
            masks=R.ROOT/state['force_masks']
            if R.digest(masks)!=state['force_masks_sha256']:raise ValueError('Demand masks changed')
            with np.load(masks) as z:
                if len(z.files)!=5 or any(z[k].shape!=(32768,) or not z[k].all() for k in z.files):
                    raise ValueError('Incomplete original five-pose certificate')
            self.protected[R.relative(masks)]=R.digest(masks)
        for a,b in [(self.initial,self.checkpoints[0]),(self.final,self.checkpoints[-1])]:
            if not np.array_equal(a.vertices,b.vertices) or not np.array_equal(a.faces,b.faces):
                raise ValueError('Movie endpoint does not equal actual checkpoint')
        self.construction=[load(p) for p in sorted((self.out/'construction').glob('*.obj'))]
        if R.volume(R.solid(self.construction[-1])-R.solid(self.initial))+R.volume(R.solid(self.initial)-R.solid(self.construction[-1]))>1e-10:
            raise ValueError('Construction does not end at initial deletion stock')
        self.construction[-1]=self.initial
        self.additions=[R.solid(b)-R.solid(a) for a,b in zip(self.construction[:-1],self.construction[1:])]
        basis=R.basis(2*np.pi)
        points=np.vstack([R.transform(self.initial.vertices,self.ref),M.box_vertices(self.boxes[0]),
                          *[p for p,c in self.hulls]])
        projected=points@basis.T;low,high=projected.min(0),projected.max(0)
        focus=((low+high)/2)@basis
        span=max((high[1]-low[1])/.82,(high[0]-low[0])/(W/H)/.82)
        self.camera=(focus,basis,float(span));self.renderer=R.V.CPU.Renderer()
        self.directions=[]
        for addition in self.additions:
            if addition.is_empty():self.directions.append(None);continue
            mesh=R.unpack(addition);direction=mesh.center_mass-self.construction[0].center_mass
            if np.linalg.norm(direction)<1e-8:direction=np.array([1.,0.,0.])
            direction=direction/np.linalg.norm(direction)
            coordinates=mesh.vertices@direction
            self.directions.append((direction,float(coordinates.min()),float(coordinates.max())))
        self.sources={}
        paths={Path(__file__).resolve(),R.V.CPU_PATH,R.V.BASELINE/'step4_connect_support/translucent_raster.cpp'}
        for module in list(sys.modules.values()):
            name=getattr(module,'__file__',None)
            if name:
                path=Path(name).resolve()
                if path.is_relative_to(R.ROOT) and not path.is_relative_to(R.ROOT/'.venv') and path.is_file():paths.add(path)
        for path in paths:
            key=R.relative(path);target=self.data/'executed_sources'/key
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
            self.sources[key]=R.digest(path)
        R.save(self.data/'render_sources.json',self.sources)

    def growth(self,amount):
        n=len(self.additions)
        if amount<=0:return self.construction[0],dict(construction_step=0,amount=0.)
        if amount>=1:return self.initial,dict(construction_step=n,amount=1.,source_sha256=self.report['initial_support_sha256'])
        x=amount*n;index=min(int(x),n-1);part=x-index
        addition=self.additions[index];direction=self.directions[index]
        if direction is None:return self.construction[index+1],dict(construction_step=index+1,amount=part)
        vector,low,high=direction;limit=low+R.smooth(part)*(high-low)
        reveal=addition.trim_by_plane(tuple(-vector),float(-limit/.2))
        mesh=R.unpack(R.solid(self.construction[index])+reveal)
        return mesh,dict(construction_step=index+1,amount=part,
            illustrative_reveal_of_actual_construction_addition=True)

    def first(self,frame):
        t=frame/FPS;amount=np.clip((t-1.)/(4.8-1.),0.,1.)
        mesh,record=self.growth(float(amount))
        parts=[R.placed(R.V.piece(mesh,R.V.BLUE),self.ref)]
        fade=1.-R.smooth((t-4.65)/.3)
        for points,color in self.hulls:
            if fade<=0:break
            polygon=trimesh.Trimesh(points,[[0,i,i+1] for i in range(1,len(points)-1)],process=False)
            piece=R.V.piece(polygon,color,.07*fade);piece['unlit']=True
            line=R.H.outline_piece(points,color,self.camera,H,half_width_pixels=.8)
            line['opacity']=.65*fade
            parts.extend([R.V.CPU.placed(piece),R.V.CPU.placed(line)])
        self.frames[1].append(dict(frame=frame,**record,hull_opacity=fade,text=0,points=0,side_panels=0))
        return self.renderer.render(parts,self.camera,W,H,scale=1.5)

    def at(self,frame):
        t=frame/FPS
        if t<.6:return 0
        if t>=13.2:return len(self.states)-1
        return min(len(self.states)-1,1+int((t-.6)/(13.2-.6)*(len(self.states)-1)))

    def second(self,frame):
        index=self.at(frame);row=self.states[index]
        self.frames[2].append(dict(frame=frame,checkpoint=index,mesh_sha256=row['mesh_sha256'],
            material_cm3=row['material_cm3'],system_bbx_l=row['bbx_l'],
            displayed_bbx_l=float(np.prod(np.diff(self.boxes[index],axis=0))*1000),
            text=0,points=0,side_panels=0,actual_certified_optimizer_checkpoint=True))
        if index not in self.cache:
            parts=[R.placed(R.V.piece(self.checkpoints[index],R.V.BLUE),self.ref),
                R.V.CPU.placed(M.box_piece(self.boxes[0],self.camera,'#D3DBDF',True)),
                R.V.CPU.placed(M.box_piece(self.boxes[index],self.camera,'#238E7B'))]
            self.cache[index]=self.renderer.render(parts,self.camera,W,H,scale=1.5)
        return self.cache[index]

    def encode(self,number,count):
        path=self.data/f'0{number}.staging.mp4'
        encoder=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo',
            '-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','pipe:0','-an',
            '-c:v','libx264','-threads','2','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
        preview={};selected=[int(x) for x in np.linspace(0,count-1,8)]
        try:
            for frame in range(count):
                image=self.first(frame) if number==1 else self.second(frame)
                encoder.stdin.write(image.tobytes())
                if frame in selected:preview[frame]=image.copy()
                if frame%24==0:print('FRAME',number,frame,count,flush=True)
            encoder.stdin.close()
            if encoder.wait():raise RuntimeError('Encoder failed')
        except BaseException:
            encoder.kill();encoder.wait();raise
        subprocess.run(['ffmpeg','-v','error','-i',str(path),'-f','null','-'],check=True)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries',
            'stream=width,height,nb_frames,r_frame_rate:format=duration','-of','json',str(path)]))
        stream=probe['streams'][0]
        if (stream['width'],stream['height'],int(stream['nb_frames']))!=(W,H,count):raise ValueError('Wrong movie dimensions/count')
        board=Image.new('RGB',(1600,1800),'white')
        for i,(frame,image) in enumerate(preview.items()):
            board.paste(image.resize((800,450),Image.Resampling.LANCZOS),((i%2)*800,(i//2)*450))
        board.save(self.data/f'0{number}_storyboard.png')
        R.save(self.data/f'0{number}_frames.json',self.frames[number])
        return path

    def verify(self):
        for path,sha in self.protected.items():
            if R.digest(R.ROOT/path)!=sha:raise ValueError('Numerical source changed: '+path)
        for path,sha in self.sources.items():
            if R.digest(R.ROOT/path)!=sha or R.digest(self.data/'executed_sources'/path)!=sha:
                raise ValueError('Render source changed: '+path)
        old=self.boxes[0];shrinks=0
        for bounds in self.boxes[1:]:
            if np.max(old[0]-bounds[0])>1e-8 or np.max(bounds[1]-old[1])>1e-8:
                raise ValueError('Displayed blue-solid BBX expanded')
            shrinks+=int(np.prod(bounds[1]-bounds[0])<np.prod(old[1]-old[0])-1e-12);old=bounds
        if shrinks==0:raise ValueError('No actual visible BBX shrink')
        return dict(passed=True,experiment=self.experiment,poses=R.POSES,
            initial_source_sha256=self.report['initial_support_sha256'],final_source_sha256=self.report['support_sha256'],
            video01_endpoint_equals_video02_start=True,video02_final_equals_actual_step5=True,
            checkpoints=len(self.states),actual_display_box_shrink_steps=shrinks,
            all_original_force_masks_checked=True,no_pruning_interpolation=True,
            hidden_final_retention_mask_used=False,closed_rings_protected=False,
            text=0,point_markers=0,side_panels=0,protected_inputs=self.protected,
            displayed_box_definition='actual displayed support at fixed first installation',
            full_system_bbx_in_frame_metadata=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',default='wrap_then_prune_all52_20261015_02')
    args=parser.parse_args()
    allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,allowed[-2:])
    movie=Movies(args.experiment)
    first=movie.encode(1,128);second=movie.encode(2,360)
    verification=movie.verify();R.save(movie.data/'verification.json',verification)
    destinations=[HERE.parent/'01_hulls_and_support.mp4',HERE.parent/'02_pruning_and_bbx.mp4']
    previous={R.relative(p):R.digest(p) for p in destinations if p.exists()}
    R.save(movie.data/'previous_versions.json',dict(replaced=previous,
        reason='Use actual legal connected wrapping and accepted deletion checkpoints; rings may be deleted'))
    for index,(source,target) in enumerate(zip([first,second],destinations),1):
        source.replace(target);shutil.copy2(movie.data/f'0{index}_storyboard.png',HERE/f'0{index}_storyboard.png')
    R.save(movie.data/'completion.json',dict(complete=True,verification=verification,
        movies={R.relative(p):R.digest(p) for p in destinations},full_decode_passed=True))
    print('MOVIES COMPLETE',movie.data,flush=True)

if __name__=='__main__':main()
