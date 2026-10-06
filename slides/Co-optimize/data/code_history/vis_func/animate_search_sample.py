"""Replay the actual first-candidate success on B/pose2+3+4+7."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'helper_func'))
sys.path.insert(0,str(HERE/'step4.2'))
from animate_exit_directions import Renderer
from hybrid_clearance_fast import ClearanceFastHybridSearch
from co_common import *
from PIL import Image,ImageDraw,ImageFont
import subprocess


def run():
    group=next(g for g in json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets'] if g['id']=='pose2+3+4+7')
    source=HERE/'data/experiments/physics_guided_clearance_fast_batch/B'/group['id']
    I.check_report(source/'report.json')
    original=json.loads((source/'report.json').read_text())
    proposals=json.loads((source/'global_proposals.json').read_text())
    assert len(proposals)==1 and not json.loads((source/'optimization_trace.json').read_text())
    out=HERE/'output/B'/group['id']/'step4/step4.2'
    data=out/'data/search_sample';data.mkdir(parents=True,exist_ok=True)
    search=ClearanceFastHybridSearch('B',group,out=data,directions=source/'initial_directions.npz',max_proposals=1)
    initial=np.load(source/'initial_directions.npz')['directions']
    final=np.asarray(original['directions'])
    results=[]
    for label,directions,counts in [('initial',initial,original['initial_counts']),('selected',final,original['final_counts'])]:
        result=search.exact(directions)
        if result['counts']!=counts:raise RuntimeError('Replay differs from recorded all-load counts')
        results.append(result)
        print('REPLAY',label,counts,flush=True)
    transforms=[T for task,T in search.states]
    renderer=Renderer(search.mesh,search.seed_mesh,transforms,group['poses'],search.length)
    for ax in renderer.axes:
        box=ax.get_position();ax.set_position([box.x0*.79,box.y0*.86+.06,box.width*.79,box.height*.86])
    sidebar=renderer.fig.add_axes([.805,.35,.19,.32],projection='3d')
    sidebar.set_proj_type('ortho');sidebar.view_init(elev=25,azim=-45)
    sidebar.set(xlim=(-1.1,1.1),ylim=(-1.1,1.1),zlim=(-1.1,1.1));sidebar.set_box_aspect((1,1,1));sidebar.set_axis_off()
    u=np.linspace(0,2*np.pi,25);v=np.linspace(0,np.pi,13)
    sidebar.plot_wireframe(np.outer(np.cos(u),np.sin(v)),np.outer(np.sin(u),np.sin(v)),np.outer(np.ones_like(u),np.cos(v)),color='#d6dde4',alpha=.4,linewidth=.5)
    for index,(a,b) in enumerate(zip(initial,final)):
        sidebar.plot(*np.vstack([a,b]).T,color='#e5ad00',linestyle='--',linewidth=2)
        sidebar.scatter(*a,color='#808890',s=30)
        sidebar.scatter(*b,color='#2596cf',s=40)
        sidebar.text(*a,str([2,3,4,7][index]),fontsize=10)
    sidebar.text2D(0,.99,'Exit directions: shared coordinates',transform=sidebar.transAxes,fontsize=12)
    fontfile='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(fontfile,25);small=ImageFont.truetype(fontfile,21);large=ImageFont.truetype(fontfile,33)
    previous=results[0]['construction']['remaining'];current=results[1]['construction']['remaining']
    removed=S.unpack(previous-current)
    grown=material_volume(current-previous)*1e6;lost=material_volume(previous-current)*1e6
    if lost<1e-5:removed=None
    sweeps=[[search.clearance.sweep(search.length*d,padded=False) for d in ds] for ds in [initial,final]]
    scenes=[]
    for index,red in [(0,None),(1,removed),(1,None)]:
        scenes.append(Image.fromarray(renderer.frame(S.unpack(results[index]['construction']['remaining']),[initial,final][index],-1,sweeps[index],red)))
    stages=[('1  INITIALIZE',0,2.0,'Native upward exits','Four different directions. Check the remaining support.',original['initial_counts']),
            ('2  SELECT CANDIDATE #1',0,3.0,'Choose a common legal direction','One direction inside every pose\'s legal floor hemisphere.',original['initial_counts']),
            ('3  EVALUATE THE PROPOSAL',0,2.0,'Project + normalize for each pose','Dashed lines show the proposed jump. Geometry has not switched yet.',original['initial_counts']),
            ('4  JUMP TO CANDIDATE #1',1,.5,'Replace all four exit directions','Shared support regrows in every panel; this jump removes no measurable material.' if lost<1e-5 else 'Blue material regrows; newly removed material flashes red for 0.5 s.',original['final_counts']),
            ('5  ACCEPT AND STOP',2,3.0,'All original loads pass','This set needs no further samples and no gradient steps.',original['final_counts'])]
    video=out/'algorithm_search_sample.mp4';fps=20;width,height=scenes[0].size
    cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    encoder=subprocess.Popen(cmd,stdin=subprocess.PIPE);timeline=[];framecount=0
    try:
        for number,(title,scene,duration,action,description,counts) in enumerate(stages):
            img=scenes[scene].copy();draw=ImageDraw.Draw(img)
            draw.rectangle((0,0,width,75),fill='white');draw.text((30,15),'B  /  poses 2 + 3 + 4 + 7   |   '+title,font=large,fill='#233544')
            draw.rectangle((0,height-80,width,height),fill='white');draw.text((30,height-68),description,font=small,fill='#233544')
            x=int(width*.805);draw.rectangle((x,90,width,height*.25),fill='white')
            draw.text((x+8,100),'ONE SEARCH CANDIDATE',font=small,fill='#233544')
            draw.multiline_text((x+8,150),action.replace(' + ',' +\n').replace(' direction','\ndirection').replace(' for each','\nfor each'),font=font,fill='#233544',spacing=12)
            draw.text((x+8,int(height*.70)),'Original loads passed',font=small,fill='#233544')
            for k,count in enumerate(counts):
                draw.text((x+8,int(height*.74)+k*40),f'Pose {[2,3,4,7][k]}: {count:,} / 32,768',font=small,fill='#16834b' if count==32768 else '#a84b32')
            draw.text((x+8,int(height*.89)),f'Regrown: {grown:.1f} cm3' if scene else 'Gray: object   Blue: support',font=small,fill='#233544')
            if scene:draw.text((x+8,int(height*.92)),f'Removed: {lost:.1f} cm3',font=small,fill='#233544')
            repeats=round(duration*fps);timeline.append(dict(stage=title,first_frame=framecount,frames=repeats,counts=counts))
            for _ in range(repeats):encoder.stdin.write(np.asarray(img).tobytes())
            framecount+=repeats
            if number==3:img.save(out/'algorithm_search_sample_poster.png')
            print('RENDERED',title,flush=True)
    finally:
        encoder.stdin.close()
    if encoder.wait()!=0:raise RuntimeError('Video encoder failed')
    save(data/'replay.json',dict(poses=group['poses'],source_report=str(source/'report.json'),source_report_sha256=I.sha256(source/'report.json'),
        actual_candidate_count=1,actual_gradient_steps=0,initial_directions=initial.tolist(),selected_directions=final.tolist(),
        fps=fps,frame_count=framecount,duration_seconds=framecount/fps,timeline=timeline,red_delay_seconds=.5,
        jump_is_discrete=True,dashed_lines_are_proposal_links_not_evaluated_intermediate_directions=True,
        replay_counts_match_source=True,full_fixture_accepted=False,regrown_cm3=grown,removed_cm3=lost,
        video_sha256=I.sha256(video),renderer_sha256=I.sha256(Path(__file__))))
    print('VIDEO',video,flush=True)


if __name__=='__main__':run()
