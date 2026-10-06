"""Replay recorded sampling jump, without inventing descent or reoptimizing."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'helper_func'))
sys.path.insert(0,str(HERE/'step4.2'))
from animate_exit_directions import Renderer
from solver import ClearanceFastHybridSearch
from co_common import *
from PIL import Image
import subprocess


def run():
    group=next(g for g in json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets'] if g['id']=='pose2+3+4+7')
    source=HERE/'output/B'/group['id']/'step4/step4.2/data'
    # Saved code hashes are historical after authorized code relocation.
    # Validate input hashes only; record the source report hash in replay metadata.
    original_for_inputs=json.loads((source/'report.json').read_text())
    I.check_hashes(original_for_inputs['provenance']['inputs'])
    original=json.loads((source/'report.json').read_text())
    proposals=json.loads((source/'global_proposals.json').read_text())
    assert len(proposals)==1 and not json.loads((source/'optimization_trace.json').read_text())
    out=HERE/'output/B'/group['id']/'step4/step4.2'
    data=out/'data/actual_sampling_replay';data.mkdir(parents=True,exist_ok=True)
    search=ClearanceFastHybridSearch('B',group,out=data,directions=source/'initial_directions.npz',max_proposals=1)
    initial=np.load(source/'initial_directions.npz')['directions']
    final=np.asarray(original['directions'])
    results=[]
    for label,directions,counts in [('initial',initial,original['initial_counts']),('selected',final,original['final_counts'])]:
        result=search.exact(directions)
        if result['counts']!=counts:raise RuntimeError('Replay differs from recorded all-load counts')
        results.append(result)
        print('REPLAY',label,counts,flush=True)
    saved_mesh=trimesh.load(source/'remaining_support.obj',force='mesh',process=False)
    saved_solid=S.solid(saved_mesh)
    rebuilt=results[1]['construction']['remaining']
    symmetric_difference=material_volume(rebuilt-saved_solid)+material_volume(saved_solid-rebuilt)
    print('FINAL COMPARISON',symmetric_difference,'rebuilt volume',material_volume(rebuilt),'saved volume',material_volume(saved_solid),flush=True)
    # Historical Boolean reconstruction can differ at degenerate boundaries.
    # Display the exact saved OBJ, never silently replace it with a replay solid.
    from scipy.spatial import cKDTree
    rebuilt_mesh=S.unpack(rebuilt)
    vertex_distance=max(cKDTree(saved_mesh.vertices).query(rebuilt_mesh.vertices)[0].max(),
                        cKDTree(rebuilt_mesh.vertices).query(saved_mesh.vertices)[0].max())
    print('FINAL VERTEX DISTANCE',vertex_distance,flush=True)
    from hybrid_directions import project_common
    if not np.allclose(project_common(np.asarray(proposals[0]['common']),search.normals),final,atol=1e-10):
        raise RuntimeError('Logged proposal differs from final directions')
    if not np.allclose(initial,search.normals,atol=1e-10):raise RuntimeError('Initial directions are not native up')
    transforms=[T for task,T in search.states]
    renderer=Renderer(search.mesh,search.seed_mesh,transforms,group['poses'],search.length)
    sweeps=[[search.clearance.sweep(search.length*d,padded=False) for d in ds] for ds in [initial,final]]
    # The only visible text is Renderer's four pose labels.
    scenes=[Image.fromarray(renderer.frame(support,directions,-1,sweep))
            for support,directions,sweep in zip(
                [S.unpack(results[0]['construction']['remaining']),saved_mesh],
                [initial,final],sweeps)]
    stages=[('native_up_initial_reconstruction',0,2.0,original['initial_counts']),
            ('recorded_sample_1_saved_final',1,3.0,original['final_counts'])]
    presentation=out/'process_actual';presentation.mkdir(exist_ok=True)
    video=presentation/'actual_sampling_process.mp4';fps=20;width,height=scenes[0].size
    cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    encoder=subprocess.Popen(cmd,stdin=subprocess.PIPE);timeline=[];framecount=0
    try:
        for number,(title,scene,duration,counts) in enumerate(stages):
            img=scenes[scene]
            repeats=round(duration*fps)
            timeline.append(dict(stage=title,first_frame=framecount,frames=repeats,counts=counts))
            for _ in range(repeats):encoder.stdin.write(np.asarray(img).tobytes())
            framecount+=repeats
            if number==1:img.save(presentation/'actual_sampling_process_poster.png')
            print('RENDERED',title,flush=True)
    finally:
        encoder.stdin.close()
    if encoder.wait()!=0:raise RuntimeError('Video encoder failed')
    save(out/'data/actual_sampling_process.json',dict(poses=group['poses'],source_report=str(source/'report.json'),source_report_sha256=I.sha256(source/'report.json'),
        actual_candidate_count=1,actual_gradient_steps=0,final_geometry_is_saved_obj=True,final_symmetric_difference_m3=symmetric_difference,final_rebuilt_saved_vertex_distance_m=vertex_distance,
        replay_geometry_is_not_substituted_for_saved_result=True,initial_directions=initial.tolist(),selected_directions=final.tolist(),
        fps=fps,frame_count=framecount,duration_seconds=framecount/fps,timeline=timeline,visible_text=['pose 2','pose 3','pose 4','pose 7'],fixed_camera=True,
        jump_is_discrete=True,direction_interpolation=False,video_path=str(video),
        replay_counts_match_source=True,full_fixture_accepted=False,
        video_sha256=I.sha256(video),renderer_sha256=I.sha256(Path(__file__))))
    gif=presentation/'actual_sampling_process.gif'
    subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(video),'-filter_complex',
        '[0:v]fps=10,scale=1000:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=full[p];[b][p]paletteuse=dither=bayer:bayer_scale=3',
        '-loop','0',str(gif)],check=True)
    renderer.plt.close(renderer.fig)
    print('VIDEO',video,'GIF',gif,'FINAL_DIFF',symmetric_difference,flush=True)


if __name__=='__main__':run()
