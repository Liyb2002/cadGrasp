"""Fixed R6 demand sets above one editable absolute-world geometry view.

Demand coordinates stay fixed; colors and exact original sample counts are
computed in the full original reaction model, independently of the 2D view.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import time

import numpy as np
import trimesh
from PIL import Image, ImageDraw

import exit_wrench_demo as old

HERE = old.HERE
DATA = HERE/'assets/data/r6_satisfaction'
BLUE = old.BLUE
GREEN = '#54A77B'
RED = '#D96D63'
INK = old.INK
MUTED = old.MUTED
ACTIVE = 3


class Experiment:
    def __init__(self):
        self.original = old.Experiment('pose1+2+3+27')
        self.poses = self.original.poses
        self.mesh = self.original.mesh
        self.native = self.original.native
        self.tasks = self.original.tasks
        self.source_rows = json.loads((old.DATA/'states.json').read_text())
        DATA.mkdir(parents=True,exist_ok=True)
        self.rows = json.loads((DATA/'states.json').read_text()) if (DATA/'states.json').exists() else []
        self.reference_length = float(self.mesh.extents.max())
        self.scale = np.r_[np.ones(3),np.full(3,1/self.reference_length)]
        physical = np.array([task.targets/task.scale for task in self.tasks])
        scaled = physical*self.scale
        self.mean = scaled.reshape(-1,6).mean(0)
        _,singular,vt = np.linalg.svd(scaled.reshape(-1,6)-self.mean,full_matrices=False)
        self.projection = vt[:2].copy()
        for row in self.projection:
            if row[np.argmax(np.abs(row))]<0:row *= -1
        self.points = (scaled-self.mean) @ self.projection.T
        low,high = self.points.reshape(-1,2).min(0),self.points.reshape(-1,2).max(0)
        middle=(low+high)/2
        radius=(high-low).max()*.58
        self.plot_low,self.plot_high = middle-radius,middle+radius
        self.plotted_ids = np.sort(np.random.default_rng(42).choice(32768,2048,replace=False))
        projection_file=DATA/f'.projection_{os.getpid()}.npz'
        np.savez_compressed(projection_file,physical_demands_world_com=physical,
            normalized_demands=scaled,projection=self.projection,mean=self.mean,
            projected_demands=self.points,plotted_sample_ids=self.plotted_ids,
            reference_length_m=self.reference_length,plot_low=self.plot_low,plot_high=self.plot_high)
        projection_file.replace(DATA/'demand_projection.npz')
        self.projection_metadata = dict(kind='one fixed common PCA projection of normalized R6 demands',
            equation='z = P (S w - mean), S=diag(1,1,1,1/L,1/L,1/L)',
            moment_reference='original object COM; invariant under rigid translation',
            reference_length_m=self.reference_length,
            displayed_variance_fraction=float((singular[:2]**2).sum()/(singular**2).sum()),
            projection=self.projection.tolist(),mean=self.mean.tolist(),
            color_decision='original full seven-equation unilateral equilibrium; never 2D overlap',
            score='number of supported original demands / 32768, for every task',
            plotted_points_per_pose=len(self.plotted_ids),scored_points_per_pose=32768)

    def score_exit(self,source_index):
        cache=DATA/f'exit_{source_index:03d}.npz'
        source=old.DATA/self.source_rows[source_index]['geometry']
        if cache.exists():
            with np.load(cache) as z:
                return dict(kind='exit',source_index=source_index,cache=cache.name,
                    geometry=str(source.relative_to(HERE)),active_pose_index=ACTIVE,
                    progress=(source_index-72)/24,counts=z['masks'].sum(1).tolist(),
                    world_translation_m=[0.,0.,0.],native_world=self.native.tolist())
        with np.load(source) as z:
            mesh=trimesh.Trimesh(z['vertices'],z['faces'],process=False)
            directions=z['directions'].copy()
        began=time.monotonic()
        supplies,areas=self.original.supplies(mesh,directions)
        masks=[];classifiers=[]
        for k,task in enumerate(self.tasks):
            mask,info=old.classify(supplies[k],task.targets)
            masks.append(mask);classifiers.append(info)
        np.savez_compressed(cache,masks=np.array(masks),directions=directions,
                            native_world=self.native,placements=np.tile(np.eye(4),(4,1,1)))
        metadata=dict(kind='exit',source_index=source_index,cache=cache.name,
            geometry=str(source.relative_to(HERE)),active_pose_index=ACTIVE,
            progress=(source_index-72)/24,counts=np.array(masks).sum(1).tolist(),
            world_translation_m=[0.,0.,0.],native_world=self.native.tolist(),
            contact_areas_m2=areas,classifiers=classifiers,seconds=time.monotonic()-began)
        old.save_json(DATA/f'exit_{source_index:03d}.json',metadata)
        print('FULL R6',source_index,metadata['counts'],round(metadata['seconds'],2),'s',flush=True)
        return metadata

    def prepare_exit(self):
        self.rows=[]
        for source_index in range(72,97):
            self.rows.append(self.score_exit(source_index))
            old.save_json(DATA/'states.json',self.rows)
        old.save_json(DATA/'projection.json',self.projection_metadata)

    def prepare_translation(self,steps=12):
        from whole_search.model import Model,Layout
        from whole_step4_render import NominalBuilder
        from material_force import certify_material_contacts
        from prune_support import material_components
        from whole_search.common import unpack_solid
        model=Model(self.poses,
            initialization_report=self.original.base/'step3/step3.1/data/report.json',
            allow_z_translation=True)
        model.support_floor_clearance_m=0.
        builder=NominalBuilder(model)
        self.rows=[row for row in self.rows if row['kind']=='exit']
        initial=self.score_exit(96)
        initial=dict(initial,kind='translation',translation_step=0,
                     world_translation_m=[0.,0.,0.],progress=0.)
        self.rows.append(initial)
        for step in range(1,steps+1):
            delta=np.array([0.,-.03*step/steps,0.])
            path=DATA/f'translation_{step:02d}.npz'
            metadata=path.with_suffix('.json')
            if path.exists() and metadata.exists():
                record=json.loads(metadata.read_text())
            else:
                began=time.monotonic()
                layout=Layout(np.tile(np.eye(4),(4,1,1)),self.original.target.copy(),
                              np.arange(4),tuple(range(4)))
                layout.placements[ACTIVE,:3,3]=self.native[ACTIVE,:3,:3].T@delta
                model.validate_placement(layout,ACTIVE)
                parts=material_components(builder.solid(layout))
                if not parts:raise RuntimeError('No connected material for translation illustration')
                solid=parts[0]
                mesh=unpack_solid(solid)
                proof=certify_material_contacts(model,layout,mesh)
                masks=np.array([proof['masks'][k] for k in range(4)])
                np.savez_compressed(path,vertices=mesh.vertices,faces=mesh.faces,
                    placements=layout.placements,directions=layout.directions,
                    native_world=self.native,masks=masks)
                record=dict(kind='translation',translation_step=step,cache=path.name,
                    geometry=str(path.relative_to(HERE)),active_pose_index=ACTIVE,
                    progress=step/steps,counts=masks.sum(1).tolist(),
                    world_translation_m=delta.tolist(),native_world=self.native.tolist(),
                    classifiers={str(k):v for k,v in proof['classifiers'].items()},
                    contact_areas_m2=[proof['contacts'][k]['area_m2'] for k in range(4)],
                    single_material_component=True,watertight=bool(mesh.is_watertight),
                    volume_cm3=old.C.material_volume(solid)*1e6,
                    seconds=time.monotonic()-began)
                old.save_json(metadata,record)
                print('TRANSLATION',step,record['counts'],round(record['seconds'],2),'s',flush=True)
            self.rows.append(record)
            old.save_json(DATA/'states.json',self.rows)
        old.save_json(DATA/'projection.json',self.projection_metadata)

    def evidence(self):
        paths=set(self.original.sources)
        paths.add(old.DATA/'states.json')
        for row in self.rows:
            paths.add(HERE/row['geometry'])
        inputs={str(path.relative_to(old.ROOT)):old.sha(path) for path in sorted(paths)}
        sources=[Path(__file__),Path(old.__file__),old.CO/'helper_func/whole_search/model.py',
                 old.CO/'helper_func/whole_search/classify.py',old.CO/'step4.2/material_force.py',
                 old.CO/'vis_func/whole_step4_render.py',old.CO/'step5.2/prune_support.py',
                 old.CO/'helper_func/co_common.py']
        old.save_json(DATA/'evidence.json',dict(complete=True,object='B',poses=self.poses,
            projection=self.projection_metadata,active_pose_index=ACTIVE,
            fixed_original_world_rotations=True,fixed_COM_relative_demand_sets=True,
            all_original_demands_per_pose=32768,full_load_subsampling=False,
            visual_point_subsampling_only=True,displayed_points_per_pose=len(self.plotted_ids),
            lower_view='one pose in absolute native world W; fixed camera and floor z=0',
            exit_phase='Only pose 27 exit changes; original native object positions and all other exits remain fixed',
            translation_phase='Only pose 27 seat translates in world -Y; exit directions and all original rotations remain fixed',
            presentation=dict(duration_seconds=15,compact_labels=True,
                              fully_supported_configurations_shown=False,
                              success_or_recovery_holds_shown=False),
            shared_material='one real connected component after all task work, body, full nominal exit and final floor exclusions',
            satisfaction='Exact count of original 32768 task demands balanced by actual retained material contacts in the unchanged original 7D model',
            metric_scope='original finite task force and torque balance; complete ground-base and structural certification are separate',
            trajectory='controlled illustrations, not an optimizer replay',
            inputs_sha256=inputs,
            sources_sha256={str(p.relative_to(old.ROOT)):old.sha(p) for p in sources},
            cache_sha256={row['cache']:old.sha(DATA/row['cache']) for row in self.rows}))


def label(draw,xy,string,size=22,fill=INK,bold=False,anchor=None):
    old.text(draw,xy,string,size,fill,bold,anchor)


class Renderer:
    def __init__(self,e):
        self.e=e
        self.raster=old.R.Renderer()
        self.view_size=(1190,391)
        self.world_pose=ACTIVE
        world=e.mesh.copy();world.apply_transform(e.native[ACTIVE])
        self.extent=float(e.mesh.extents.max())
        self.length=self.extent*.82
        center=e.tasks[ACTIVE].domain.com
        points=np.vstack([world.vertices,center+self.length*np.eye(3),center-self.length*np.eye(3)])
        self.camera=old.R.fit(points,np.array([1.,-1.,.8]),self.view_size[0]/self.view_size[1],padding=1.10)
        self.floor_bounds=(world.vertices[:,:2].min(0)-self.extent*.22,
                           world.vertices[:,:2].max(0)+self.extent*.22)

    def cloud(self,k,mask,width=413,height=211):
        image=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(image)
        points=self.e.points[k,self.e.plotted_ids]
        uv=(points-self.e.plot_low)/(self.e.plot_high-self.e.plot_low)
        uv=np.c_[17+uv[:,0]*(width-34),height-12-uv[:,1]*(height-26)]
        flags=mask[self.e.plotted_ids]
        # Point positions are fixed throughout; only these full-space labels vary.
        for supported,color in [(True,GREEN),(False,RED)]:
            for x,y in uv[flags==supported]:
                draw.ellipse((x-1.2,y-1.2,x+1.2,y+1.2),fill=color)
        label(draw,(width-12,height-13),'z₁',13,MUTED,anchor='ra')
        label(draw,(15,6),'z₂',13,MUTED)
        return image

    def world(self,row,mesh,placements,directions):
        T=np.asarray(row['native_world'])[ACTIVE]
        Q=placements[ACTIVE]
        U=T@Q
        body=self.e.mesh.copy();body.apply_transform(U)
        colors=np.broadcast_to(old.R.rgb('#AAB3BA'),(len(body.faces),3,3)).copy()
        colors[self.e.tasks[ACTIVE].domain.work_ids]=old.R.rgb(old.GREEN)
        body_piece=old.R.piece(dict(v=body.vertices,f=body.faces),colors)
        low,high=self.floor_bounds
        floor=np.array([[low[0],low[1],0.],[high[0],low[1],0.],
                        [high[0],high[1],0.],[low[0],high[1],0.]])
        floor_piece=old.R.piece(dict(v=floor,f=np.array([[0,1,2],[0,2,3]])),'#EDF2F5')
        floor_piece['unlit']=True
        parts=[old.R.placed(floor_piece),old.R.placed(body_piece)]
        if len(mesh.faces):
            support=old.R.piece(dict(v=mesh.vertices,f=mesh.faces),BLUE)
            parts.append(old.R.placed(support,T[:3,:3],T[:3,3]))
        picture=self.raster.render(parts,self.camera,*self.view_size,scale=1.3)
        draw=ImageDraw.Draw(picture)
        P=lambda p:old.project(p,self.camera,*self.view_size)
        center=old.C.transform_points(self.e.mesh.center_mass[None],U)[0]
        direction=T[:3,:3]@directions[ACTIVE]
        tip=P(center+self.length*direction)
        old.arrow(draw,[P(center),tip],old.EXIT,9,22)
        label(draw,tip+np.array([13,-18]),'d₂₇',23,old.EXIT,True)
        if row['kind']=='translation':
            start=self.e.tasks[ACTIVE].domain.com
            if np.linalg.norm(center-start)>1e-6:
                old.arrow(draw,[P(start),P(center)],'#8B6BB4',5,16)
                label(draw,P(start)+np.array([-20,-30]),'Δp',22,'#8B6BB4',True)
                origin=P(start)
                draw.ellipse((origin[0]-5,origin[1]-5,origin[0]+5,origin[1]+5),
                             fill='white',outline='#8B6BB4',width=2)
        return picture,U,direction

    def frame(self,row,stage='adjust'):
        with np.load(DATA/row['cache']) as z:
            masks=z['masks'].copy();directions=z['directions'].copy();placements=z['placements'].copy()
        if masks.all():
            raise ValueError('Fully satisfied configurations are excluded from this presentation')
        path=HERE/row['geometry']
        with np.load(path) as z:mesh=trimesh.Trimesh(z['vertices'],z['faces'],process=False)
        image=Image.new('RGB',(1920,1080),'#FAFBFC');draw=ImageDraw.Draw(image)
        label(draw,(47,38),'R⁶ demand sets',36,INK,True)
        for k,pose in enumerate(self.e.poses):
            x=42+469*k;y=132
            draw.rounded_rectangle((x,y,x+438,y+344),radius=15,fill='white',
                                   outline=old.EXIT if k==ACTIVE else '#DFE6EC',width=3 if k==ACTIVE else 2)
            label(draw,(x+19,y+15),pose.replace('_',' ').capitalize(),23,old.PALETTE[k],True)
            label(draw,(x+417,y+20),'D'+['₁','₂','₃','₂₇'][k]+' ⊂ R⁶',22,INK,anchor='ra')
            image.paste(self.cloud(k,masks[k]),(x+12,y+57))
            count=int(masks[k].sum());percent=count/32768*100
            score='100%' if count==32768 else (f'{percent:.1f}%' if percent<99.95 else '>99.9%')
            color=GREEN if count==32768 else RED
            label(draw,(x+20,y+268),score,32,color,True)
            draw.rounded_rectangle((x+20,y+316,x+418,y+324),radius=4,fill='#EDF1F4')
            if count:
                draw.rounded_rectangle((x+20,y+316,x+20+398*count/32768,y+324),radius=4,fill=GREEN)
        label(draw,(49,491),'P: R⁶ → R²',18,MUTED)
        # A single absolute world pose occupies the whole lower row.
        x,y=42,538
        draw.rounded_rectangle((x,y,1878,1025),radius=17,fill='white',outline='#DFE6EC',width=2)
        label(draw,(65,555),'Pose 27 · World W',27,INK,True)
        scene,U,direction=self.world(row,mesh,placements,directions)
        image.paste(scene,(68,606))
        draw=ImageDraw.Draw(image)
        draw.line((1284,603,1284,998),fill='#E6EBF0',width=2)
        cx=1322
        chapter='Exit direction' if row['kind']=='exit' else 'Translation'
        label(draw,(cx,610),chapter,24,old.EXIT if row['kind']=='exit' else '#8B6BB4',True)
        label(draw,(cx,665),'d₂₇ᵂ',22,INK,True)
        label(draw,(cx,704),'('+', '.join(f'{v:+.3f}' for v in direction)+')',23,old.EXIT)
        label(draw,(cx,761),'p₂₇ᵂ  [mm]',21,INK,True)
        label(draw,(cx,800),'('+', '.join(f'{v:+.1f}' for v in U[:3,3]*1000)+')',23,'#8B6BB4')
        draw.ellipse((cx,954,cx+12,966),fill=GREEN)
        label(draw,(cx+23,949),'Supported',18,MUTED)
        draw.ellipse((cx+207,954,cx+219,966),fill=RED)
        label(draw,(cx+230,949),'Unmet',18,MUTED)
        return image

    def preview(self):
        row=self.e.score_exit(84)
        self.frame(row).save(HERE/'r6_satisfaction_preview.png')
        old.save_json(DATA/'projection.json',self.e.projection_metadata)
        print('PREVIEW',HERE/'r6_satisfaction_preview.png',row['counts'],flush=True)

    def render(self,fps=24):
        partial=lambda r:not all(int(count)==32768 for count in r['counts'])
        exit_rows=[r for r in self.e.rows if r['kind']=='exit' and partial(r)]
        translations=[r for r in self.e.rows if r['kind']=='translation' and partial(r)]
        if not exit_rows or not translations:
            raise RuntimeError('Prepare exit and translation configurations first')
        def distribute(rows,frames,stage):
            quotient,remainder=divmod(frames,len(rows))
            return [(row,quotient+int(index<remainder),stage) for index,row in enumerate(rows)]
        # Retiming and omission only: preserve every original numerical mask.
        # 0.5 s initial + 6 s exits + 7.5 s translation + 1 s unresolved end.
        intro=round(.5*fps)
        exit_frames=6*fps
        final=fps
        translation_frames=15*fps-intro-exit_frames-final
        timeline=[(exit_rows[0],intro,'adjust')]
        timeline += distribute(exit_rows,exit_frames,'adjust')
        timeline += distribute(translations,translation_frames,'out')
        timeline += [(translations[-1],final,'out')]
        assert sum(repeats for _,repeats,_ in timeline)==15*fps
        assert all(partial(row) for row,_,_ in timeline)
        video=HERE/'r6_satisfaction.mp4'
        command=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
            '-s','1920x1080','-r',str(fps),'-i','-','-an','-c:v','libx264',
            '-threads','2','-preset','fast','-crf','18','-pix_fmt','yuv420p',
            '-movflags','+faststart',str(video)]
        encoder=subprocess.Popen(command,stdin=subprocess.PIPE)
        frames=0;previous=None;records=[];cached={}
        try:
            for index,(row,repeats,stage) in enumerate(timeline):
                key=(row['kind'],row['cache'],stage)
                if key not in cached:cached[key]=self.frame(row,stage)
                picture=cached[key]
                first=frames
                blend_frames=min(4,repeats//2) if previous is not None else 0
                for offset in range(repeats):
                    output=(Image.blend(previous,picture,(offset+1)/(blend_frames+1))
                            if offset<blend_frames else picture)
                    encoder.stdin.write(output.tobytes());frames+=1
                records.append(dict(kind=row['kind'],cache=row['cache'],stage=stage,
                                    first_video_frame=first,repetitions=repeats,
                                    image_dissolve_frames=blend_frames,counts=row['counts']))
                previous=picture
                if index%10==0:print('R6 RENDER',index,'/',len(timeline)-1,flush=True)
            encoder.stdin.close()
            if encoder.wait()!=0:raise RuntimeError('Video encoder failed')
        except BaseException:
            encoder.kill();encoder.wait();raise
        self.frame(translations[-1],'out').save(HERE/'r6_satisfaction.png')
        self.frame(translations[-1],'out').save(DATA/'translation_endpoint.png')
        # Every public still also excludes globally satisfied configurations.
        sheet=Image.new('RGB',(1920,1080),'white')
        selected=[(exit_rows[0],'adjust'),(self.e.score_exit(84),'adjust'),
                  (translations[len(translations)//2],'out'),(translations[-1],'out')]
        for index,(row,stage) in enumerate(selected):
            tile=self.frame(row,stage).resize((960,540),Image.Resampling.LANCZOS)
            sheet.paste(tile,((index%2)*960,(index//2)*540))
        sheet.save(HERE/'r6_satisfaction_storyboard.png')
        old.save_json(DATA/'video.json',dict(complete=True,file=video.name,sha256=old.sha(video),
            size=[1920,1080],fps=fps,frame_count=frames,duration_seconds=frames/fps,
            evaluated_configurations=len(set(row['cache'] for row,_,_ in timeline)),
            fully_supported_configurations_shown=False,success_or_recovery_holds_shown=False,
            labels='compact: pose/set identifiers, percentages, exit/position parameters, two-color legend',
            image_transitions='brief image cross-dissolves; numerical labels belong to evaluated configurations',
            timeline=records,source_sha256=old.sha(__file__)))
        self.e.evidence()
        print('COMPLETE',video,frames/fps,'seconds',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview',action='store_true')
    parser.add_argument('--prepare-exit',action='store_true')
    parser.add_argument('--prepare-translation',action='store_true')
    parser.add_argument('--render',action='store_true')
    args=parser.parse_args()
    e=Experiment()
    if args.prepare_exit:e.prepare_exit()
    if args.prepare_translation:e.prepare_translation()
    if args.render:Renderer(e).render()
    if args.preview or not any([args.prepare_exit,args.prepare_translation,args.render]):Renderer(e).preview()


if __name__=='__main__':main()
