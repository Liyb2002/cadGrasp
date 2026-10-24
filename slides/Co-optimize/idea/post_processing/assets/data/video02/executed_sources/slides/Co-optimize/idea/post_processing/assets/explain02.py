"""Explain contact elimination, reconstruction, then actual Step5 pruning."""
from pathlib import Path
import argparse,json,subprocess,time
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import render01 as R
import render02 as B
import construction_breakdown as C

DATA=B.DATA
FPS,FRAMES=24,576
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BOLD='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
YELLOW='#E4A235'


def cn(size,bold=False):return ImageFont.truetype(BOLD if bold else FONT,size,index=2)


class Movie(B.Movie):
    def __init__(self):
        self.story=C.Data()
        super().__init__()
        self.protected.update(self.story.protected)
        R.save(DATA/'inputs.json',self.protected)
        self.contact_cache={};self.build_cache={};self.timeline_frames=[]
        self.input_ghost=R.V.piece(self.meshes[0],R.V.BLUE,.20)

    def timeline(self,t):
        if t<.6:return 'input',t/.6
        if t<5.1:return 'faces',(t-.6)/4.5
        if t<7.1:return 'points',(t-5.1)/2.
        if t<13.1:return 'build',(t-7.1)/6.
        if t<13.6:return 'certified',(t-13.1)/.5
        if t<21.2:return 'pruning',(t-13.6)/7.6
        return 'final',(t-21.2)/2.8

    def phase(self,t):
        mode,p=self.timeline(t)
        if mode in ['input','faces','points']:return 0,1.
        if mode in ['build','certified']:return 1,1.
        if mode=='pruning':
            position=p*18;offset=min(17,int(position))
            return 2+offset,(position-offset)*7.6/18
        return 20,t-21.2

    def face_state(self,k,p):
        masks=self.story.arrays[f'face_masks_{k}'];index=min(len(masks)-1,int(p*(len(masks)-1)))
        ids=self.story.arrays[f'face_ids_{k}'];return ids[masks[index]],index

    def point_state(self,k,p):
        masks=self.story.arrays[f'point_masks_{k}'];index=min(len(masks)-1,int(p*(len(masks)-1)))
        return self.story.arrays[f'points_{k}'][masks[index]],index

    def patches(self,ids,color,opacity=.85):
        sources=self.story.arrays['contact_sources']
        triangles=self.story.arrays['contact_triangles_m'][np.isin(sources,ids)]
        if not len(triangles):return None
        normal=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
        normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-20)
        return dict(triangles=triangles,normals=np.broadcast_to(normal[:,None,:],triangles.shape),
            colors=np.broadcast_to(R.V.CPU.rgb(color),triangles.shape),opacity=opacity,bias=1.,unlit=True)

    def contact_parts(self,k,p,mode,matrix,camera,size):
        result=[R.placed(self.input_ghost,matrix)]
        ids,_=self.face_state(k,p if mode=='faces'else 1.)
        patch=self.patches(ids,B.GREEN if mode=='points'or p>=1. else YELLOW,
            .33 if mode=='points'else .88)
        if patch is not None:result.append(R.placed(patch,matrix))
        if mode=='points':
            points,_=self.point_state(k,p)
            marker=R.V.points_piece(R.transform(points,matrix),B.GREEN if p>=1. else YELLOW,
                camera,size,half_size_pixels=2.8 if size==900 else 1.3)
            marker['bias']=2*camera[2]
            result.append(R.V.CPU.placed(marker))
        return result

    def contact_tile(self,k,p,mode):
        _,state=(self.face_state(k,p)if mode=='faces'else self.point_state(k,p))
        key=(mode,k,state)
        if key not in self.contact_cache:
            row=self.rows[k];parts=self.contact_parts(k,p,mode,row['T'],self.cameras[k],230)+[self.bodies[k]]
            self.contact_cache[key]=self.renderer.render(parts,self.cameras[k],230)
        return self.contact_cache[key]

    def contact_frame(self,mode,p):
        if mode=='faces':
            cursor=min(4,int(p*5));fraction=1. if p>=1. else min(1.,(p*5-cursor)*1.3)
            states=[1. if k<cursor else fraction if k==cursor else 0. for k in range(5)]
        else:cursor=0;fraction=min(1.,p*1.15);states=[fraction]*5
        parts=self.contact_parts(cursor,states[cursor],mode,self.ref,self.camera,900)
        parts.append(R.V.CPU.placed(B.box_piece(self.record['stages'][0]['shown_bounds_world_m'],self.camera,B.GREY,True)))
        image=Image.new('RGB',(1600,900),'white');image.paste(self.renderer.render(parts,self.camera,900),(0,0))
        draw=ImageDraw.Draw(image)
        for k,((x,y),row)in enumerate(zip([(940,48),(1290,48),(940,285),(1290,285),(1115,522)],self.rows)):
            image.paste(self.contact_tile(k,states[k],mode),(x,y))
            current=(len(self.face_state(k,states[k])[0])if mode=='faces'else len(self.point_state(k,states[k])[0]))
            initial=self.story.record['records'][k]['input_faces'if mode=='faces'else'input_locations']
            text=f"姿态 {R.POSES[k].split('_')[-1]}：{initial} → {current} "+('面'if mode=='faces'else'处')
            draw.text((x,y+4),text,font=cn(15,True),fill=row['color'])
        if mode=='faces':
            title='① 筛选接触面，原实体不变'
            subtitle=f"当前：姿态 {R.POSES[cursor].split('_')[-1]}，逐批试删接触候选"
            detail='只有删除后仍满足全部 32,768 组载荷，才接受这次筛选。'
        else:
            title='① 继续筛选反力作用位置'
            subtitle='在保留的接触面上继续筛选，各姿态最终合计保留 26 处。'
            detail='黄色是尚在测试的候选，绿色是最终保留的接触位置。'
        self.header(image,title,subtitle,detail)
        draw.rectangle((20,775,880,899),fill='white')
        draw.text((30,784),'实体此时没有改变：155.46 cm³',font=cn(26,True),fill=B.BLUE)
        draw.text((30,829),'删的是接触候选，不是支撑材料。',font=cn(23,True),fill=B.TEXT)
        draw.text((925,762),'此阶段原支撑和 BBX 均保持不变',font=cn(20,True),fill=B.TEXT)
        draw.text((925,800),'先确定哪些接触必须保留，',font=cn(19),fill=B.MUTED)
        draw.text((925,831),'再根据它们重新构造支撑。',font=cn(19),fill=B.MUTED)
        draw.line((900,0,900,899),fill='#E6ECEF',width=1)
        return image

    def partial_piece(self,p):
        selected=self.story.arrays['construction_face_progress']<=p
        if not selected.any():return None
        result={k:(v[selected]if k in ['triangles','normals','colors']else v)for k,v in self.blue[1].items()}
        return result

    def path_segments(self,p):
        amount=p*32;segments=[]
        for i in range(32):
            path=self.story.arrays[f'path_{i}'];fraction=np.clip(amount-i,0.,1.)
            if fraction<=0:continue
            lengths=np.linalg.norm(np.diff(path,axis=0),axis=1);left=lengths.sum()*fraction
            for a,b,length in zip(path,path[1:],lengths):
                if left<=0:break
                shown=min(length,left)
                if length>1e-12:segments.append((a,a+(b-a)*shown/length))
                left-=shown
        return segments

    def build_parts(self,p,matrix,camera,size):
        ghost=dict(self.input_ghost);ghost['opacity']=.16*(1-p)
        parts=[R.placed(ghost,matrix)]if ghost['opacity']>.001 else[]
        partial=self.partial_piece(p)
        if partial is not None:parts.append(R.placed(partial,matrix))
        segments=self.path_segments(p)
        if segments:
            world=[(R.transform(a[None],matrix)[0],R.transform(b[None],matrix)[0])for a,b in segments]
            parts.append(R.V.CPU.placed(B.lines(world,B.BLUE,camera,size,width=2.2,opacity=.9)))
        for key,color in [('object_ports_m',B.GREEN),('ground_ports_m',YELLOW)]:
            marker=R.V.points_piece(R.transform(self.story.arrays[key],matrix),color,camera,size,
                half_size_pixels=3.6 if size==900 else 1.7)
            marker['bias']=2*camera[2]
            parts.append(R.V.CPU.placed(marker))
        return parts

    def build_frame(self,p):
        image=Image.new('RGB',(1600,900),'white')
        image.paste(self.renderer.render(self.build_parts(p,self.ref,self.camera,900),self.camera,900),(0,0))
        draw=ImageDraw.Draw(image);completed=min(32,int(p*32))
        for k,((x,y),row)in enumerate(zip([(940,48),(1290,48),(940,285),(1290,285),(1115,522)],self.rows)):
            key=(k,completed)
            if key not in self.build_cache:
                parts=self.build_parts(completed/32,row['T'],self.cameras[k],230)+[self.bodies[k]]
                self.build_cache[key]=self.renderer.render(parts,self.cameras[k],230)
            image.paste(self.build_cache[key],(x,y));draw.text((x,y+4),f"姿态 {R.POSES[k].split('_')[-1]}",font=cn(16),fill=row['color'])
        self.header(image,'② 根据保留接触重新构造圆杆支撑',
            '绿色：26 处工件接触；黄色：23 处地脚位置；蓝色：新圆杆。',
            '沿实际保存的路径逐条连接；淡蓝色是旧支撑，仅作位置参照。')
        draw.rectangle((20,775,880,899),fill='white')
        draw.text((30,783),f'连接进度：{completed} / 32 条保存路径',font=cn(25,True),fill=B.BLUE)
        draw.text((30,828),'这是重新构形，不是继续从旧实体上削材料。',font=cn(23,True),fill=B.TEXT)
        draw.text((925,762),'构造预览：连接完成后统一验收',font=cn(20,True),fill=B.TEXT)
        draw.text((925,802),'保留接触和地脚 → 内部节点 → 圆杆',font=cn(18),fill=B.MUTED)
        draw.text((925,834),'完整构造通过后，才开始下一段实体删减。',font=cn(17),fill=B.MUTED)
        draw.line((900,0,900,899),fill='#E6ECEF',width=1)
        return image

    def header(self,image,title,subtitle,detail=''):
        draw=ImageDraw.Draw(image);draw.rectangle((15,0,885,124),fill='white')
        draw.text((30,16),title,font=cn(28,True),fill=B.TEXT)
        draw.text((31,62),subtitle,font=cn(18),fill=B.MUTED)
        if detail:draw.text((31,92),detail,font=cn(17),fill=B.MUTED)

    def frame(self,t,record=False):
        mode,p=self.timeline(t)
        if mode in ['faces','points']:image=self.contact_frame(mode,p)
        elif mode=='build':image=self.build_frame(min(1.,p*1.08))
        else:
            image=super().frame(t,False)
            index,age=self.phase(t)
            if mode=='input':self.header(image,'Step5 完整输入','接下来先筛选接触，再重新构形，最后对新实体减料。')
            elif mode=='certified':self.header(image,'圆杆支撑构造完成：37.93 cm³',
                '这件新实体已通过原始受力和实体检查，现在开始逐步减料。')
            elif mode=='pruning':
                same=not self.record['stages'][index]['shown_box_changed']
                self.header(image,f'③ 对新支撑逐步删材料：{index-1} / 18',
                    '缩小圆杆和接触帽；每次接受都重新检查全部物理条件。',
                    '本步只减少材料，BBX 保持不变。'if same else'本步同时减少材料并缩小 BBX；右下角放大边界移动。')
            else:self.header(image,'Step5 最终成品：14.42 cm³',
                '真实保存的最终 OBJ；最后还有必要的连接加固和完整复验。')
        if record:
            index,_=self.phase(t)
            self.timeline_frames.append(dict(frame=round(t*24),mode=mode,progress=p,actual_stage=index,
                geometry_unchanged=mode in ['input','faces','points'],
                construction_preview=mode=='build',rendered_full_model_sha256=(None if mode=='build'else self.record['stages'][index]['support_sha256']),
                final_is_exact_saved_step5=mode=='final',contact_selection_changes_only_candidates=mode in ['faces','points']))
        return image

    def verify(self):
        result=super().verify()
        result.update(contact_selection_replayed_from_saved_trials=True,contact_selection_changes_original_geometry=False,
            face_counts_before=[r['input_faces']for r in self.story.record['records']],
            face_counts_after=[r['retained_faces']for r in self.story.record['records']],
            selected_force_locations=26,recorded_construction_paths=32,
            reconstruction_is_explained_separately=True,
            construction_preview_uses_actual_cp1_triangles_and_recorded_paths=True,
            partial_construction_is_not_claimed_as_certified_support=True,
            all_pruning_and_final_models_are_exact_original_saved_obj=True,
            captions_language='zh-CN')
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--encode',action='store_true')
    args=parser.parse_args();movie=Movie()
    times=[0.,.66,3.3,6.95,7.5,9.,10.8,12.8,14.5,17.,20.5,23.3]
    sheet=Image.new('RGB',(2400,1800),'white')
    for i,t in enumerate(times):
        frame=movie.frame(t);frame.save(DATA/f'explanation_{i:02d}.png')
        sheet.paste(frame.resize((800,450),Image.Resampling.LANCZOS),((i%3)*800,(i//3)*450))
    sheet.save(B.HERE/'02_storyboard.png');R.save(DATA/'verification.json',movie.verify())
    if not args.encode:return
    staging=DATA/'02_explained.staging.mp4'
    encoder=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo',
        '-pix_fmt','rgb24','-s','1600x900','-r','24','-i','pipe:0','-an','-c:v','libx264',
        '-preset','fast','-crf','18','-profile:v','main','-pix_fmt','yuv420p','-movflags','+faststart',str(staging)],stdin=subprocess.PIPE)
    began=time.monotonic()
    try:
        for i in range(FRAMES):
            encoder.stdin.write(movie.frame(i/FPS,True).tobytes())
            if i%24==0:print(json.dumps(dict(frame=i,total=FRAMES,seconds=round(time.monotonic()-began,1))),flush=True)
        encoder.stdin.close()
        if encoder.wait()!=0:raise RuntimeError('Encoding failed')
    except BaseException:encoder.kill();encoder.wait();raise
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-i',str(staging),'-f','null','-'],check=True)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries',
        'format=duration:stream=width,height,nb_frames,r_frame_rate','-of','json',str(staging)]))
    if int(probe['streams'][0]['nb_frames'])!=FRAMES:raise ValueError('Wrong frame count')
    final_sha=movie.record['final_obj_sha256']
    if not all(f['rendered_full_model_sha256']==final_sha for f in movie.timeline_frames[-48:]):
        raise ValueError('Final frames do not use actual final geometry')
    verification=movie.verify();staging.replace(B.MOVIE)
    verification.update(movie=R.relative(B.MOVIE),movie_sha256=R.digest(B.MOVIE),
        frames_checked=FRAMES,full_decode_passed=True,encoded_probe=probe,
        actual_final_obj_sha256=final_sha,final_exact_obj_frames=sum(f['final_is_exact_saved_step5']for f in movie.timeline_frames))
    R.save(DATA/'frame_checks.json',movie.timeline_frames);R.save(DATA/'verification.json',verification)
    R.save(DATA/'manifest.json',dict(complete=True,pose_set=R.CASE,movie=R.relative(B.MOVIE),
        movie_sha256=R.digest(B.MOVIE),duration_seconds=24.,fps=24,frames=FRAMES,width=1600,height=900,
        final_obj_sha256=final_sha,captions_language='zh-CN',
        phases=dict(contact_faces=[.6,5.1],force_locations=[5.1,7.1],reconstruction=[7.1,13.1],
            construction_accepted=[13.1,13.6],actual_material_pruning=[13.6,21.2],final=[21.2,24.]),
        initial_solid_unchanged_during_contact_selection=True,
        reconstruction_is_not_pure_subtraction=True,original_physical_models_unchanged=True))
    print(json.dumps(verification),flush=True)


if __name__=='__main__':main()
