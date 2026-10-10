"""Reuse the earlier exit-direction demonstration with current wrap, no base."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
from co_common import *
from animate_exit_directions import Renderer
import argparse,subprocess,time
from PIL import Image


def direction_schedule(initial,transforms,steps):
    directions=np.asarray(initial,dtype=float).copy()
    yield -1,0.,directions.copy()
    for active,T in enumerate(transforms):
        start=T[:3,:3]@directions[active];start/=np.linalg.norm(start)
        target=np.array([1.,1.,0.])/np.sqrt(2)
        angle=np.arccos(np.clip(start@target,-1,1))
        if angle>np.pi-1e-8:
            target=np.array([-1.,1.,0.])/np.sqrt(2)
            angle=np.arccos(np.clip(start@target,-1,1))
        for u in np.linspace(0,1,steps+1)[1:]:
            progress=u*u*(3-2*u)
            world=start if angle<1e-10 else (np.sin((1-progress)*angle)*start+np.sin(progress*angle)*target)/np.sin(angle)
            directions[active]=T[:3,:3].T@(world/np.linalg.norm(world))
            yield active,float(u),directions.copy()


def run(group,steps=16,fps=12):
    began=time.monotonic();base=HERE/'output/B'/group['id'];out=base/'step4/step4.1';process=out/'process';process.mkdir(exist_ok=True)
    seedpath=base/'step3/step3.2/wrapped_support.obj';objpath=base/'step3/step3.1/registered_object.obj'
    seedmesh=trimesh.load(seedpath,force='mesh',process=False);obj=trimesh.load(objpath,force='mesh',process=False);seed=S.solid(seedmesh)
    states=[state('B',p) for p in group['poses']];transforms=[T for _,T,_ in states]
    length=max(.5,float(np.linalg.norm(seedmesh.vertices,axis=1).max()+np.linalg.norm(obj.vertices,axis=1).max())+.02)
    initialization_path=out/'data/report.json';initialization=json.loads(initialization_path.read_text())
    saved={r['pose']:r['direction_fixture'] for r in initialization['state_results']}
    initial=np.asarray([saved[p] for p in group['poses']],dtype=float)
    schedule=list(direction_schedule(initial,transforms,steps));assert np.array_equal(schedule[0][2],initial)
    sweeps=[S.solid(S.swept_solid(obj,length*d)) for d in schedule[0][2]]
    renderer=Renderer(obj,seedmesh,transforms,group['poses'],length,columns=2,labels=False)
    video=process/'exit_direction_changes.mp4';poster=process/'exit_direction_changes_poster.png'
    width,height=renderer.fig.canvas.get_width_height()
    command=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    encoder=subprocess.Popen(command,stdin=subprocess.PIPE);rows=[];frames=0
    try:
        for index,(active,u,directions) in enumerate(schedule):
            if index:
                changed=np.flatnonzero(np.linalg.norm(directions-schedule[index-1][2],axis=1)>1e-10)
                assert all(int(k)==active for k in changed)
            assert all((T[:3,:3]@d)[2]>=-1e-8 for T,d in zip(transforms,directions))
            if active>=0:sweeps[active]=S.solid(S.swept_solid(obj,length*directions[active]))
            current=seed-union(sweeps);support=S.unpack(current)
            frame=renderer.frame(support,directions,active,sweeps)
            repetitions=fps if index==0 or u==1 else 1
            if index==len(schedule)-1:repetitions+=fps
            for _ in range(repetitions):encoder.stdin.write(frame.tobytes())
            if index==0:Image.fromarray(frame).save(poster)
            if index==steps//2:Image.fromarray(frame).save(process/'exit_direction_changes_preview.png')
            rows.append(dict(state=index,active_pose=None if active<0 else group['poses'][active],progress=u,directions_fixture=directions.tolist(),remaining_volume_cm3=material_volume(current)*1e6,first_frame=frames,repetitions=repetitions))
            frames+=repetitions
            if index%8==0:print('WRAP ANIMATION',index,'/',len(schedule)-1,flush=True)
        encoder.stdin.close()
        if encoder.wait()!=0:raise RuntimeError('ffmpeg failed')
    except BaseException:
        encoder.kill();encoder.wait();raise
    finally:renderer.plt.close(renderer.fig)
    save(out/'data/exit_direction_animation.json',dict(complete=True,demo_only=True,optimizer_run=False,force_acceptance_run=False,base_deferred=True,pose_set=group['id'],poses=group['poses'],support_source=str(seedpath.relative_to(ROOT)),geometry='Current Step3.2 wrap; no Step3.3 ground ring',sweep_method='Continuous full-object straight translation, nominal geometry demonstration without contact-core clearance certification',initial_direction_source=str(initialization_path.relative_to(ROOT)),initial_directions_fixture=initial.tolist(),motion='Start exactly from saved Step4.1 directions; one pose at a time follows a spherical arc toward a native horizontal direction',all_panels_share_same_material=True,support_recomputed_from_original_each_frame=True,columns=2,rows=2,text_labels=False,frame_size_px=[width,height],fps=fps,frame_count=frames,duration_seconds=frames/fps,full_exit_length_m=length,states=rows,seconds=time.monotonic()-began,provenance=provenance([seedpath,objpath,initialization_path]+[p for t,_,_ in states for p in t.inputs],[Path(__file__),Path(__file__).with_name('animate_exit_directions.py')]),artifacts={'../process/'+video.name:I.sha256(video),'../process/'+poster.name:I.sha256(poster)}))
    (process/'README.md').write_text('# 退出方向扫掠演示\n\npose1、pose2、pose4、pose6 按 2×2 四格排列（上排 1、2，下排 4、6），每次仅改变一个 pose 的退出方向，其他方向保持；橙色边框标记当前 pose。灰色物体、蓝色共享剩余包裹、透明 sweep、黄色方向。所有格显示同一份共享支撑，方向变化时可恢复原先切掉的材料。\n\n起点严格读取本组 Step4.1 data/report.json 中四个 direction_fixture，随后一次调整一个 pose，以球面弧线转向水平退出方向；其他 pose 保持当前方向。使用当前 Step3.2 包裹 shape，不含 base。连续完整名义 sweep 用于演示切除，非优化过程、非接触核／净空或受力验收；保存算法结果不变。\n')
    print('VIDEO COMPLETE',video,frames/fps,'seconds',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--set',default='pose1+2+4+6');p.add_argument('--steps',type=int,default=16);p.add_argument('--fps',type=int,default=12);a=p.parse_args()
    g=next(g for g in read_selected_pose_groups('B') if g['id']==a.set);run(g,a.steps,a.fps)
