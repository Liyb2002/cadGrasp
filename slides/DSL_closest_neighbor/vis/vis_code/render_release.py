"""Render recorded accepted head edits; no geometry acceptance or replay."""
import argparse,json,sys,tempfile,subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import merge_release as M

DEST=Path(__file__).resolve().parents[1]/'vis_result'

def axes_bounds(ax,points,extra=.1):
    lo,hi=points.min(0),points.max(0);center=(lo+hi)/2;span=max(hi-lo)*(1+extra)/2
    ax.set_xlim(center[0]-span,center[0]+span);ax.set_ylim(center[1]-span,center[1]+span);ax.set_zlim(center[2]-span,center[2]+span)
    ax.set_box_aspect((1,1,1));ax.set_axis_off();ax.view_init(elev=25,azim=35)

def render(group,stage):
    base=M.HERE/'output/B'/group/'step3_scheculer'/stage
    report=base/'report.json'
    if not report.exists():return
    r=json.loads(report.read_text());folders=[base/'initial']+sorted(base.glob('release_[0-9]*'))+[base/'final']
    tasks=[M.current_task('B','pose_'+p) for p in group.removeprefix('pose').split('+')]
    files=[report];cmap=plt.get_cmap('tab20')
    initial=json.loads((base/'initial/state.json').read_text());color_ids={h['id']:i for i,h in enumerate(initial['physical_heads'])}
    raw=np.load(base/'initial/head_geometry.npz');world_bounds=np.vstack([raw[f'h{i}_vertices'] for i in range(len(initial['physical_heads']))])
    DEST.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='head_frames_') as temp:
        sequence=[]
        for k,folder in enumerate(folders):
            state=json.loads((folder/'state.json').read_text());z=np.load(folder/'head_geometry.npz');sets=state['exits'];directions=np.array(sets['directions_world_xyz'])
            fig=plt.figure(figsize=(18,10),dpi=100);grid=fig.add_gridspec(2,max(2,len(tasks)))
            ownership={m:color_ids[h['id']] for h in state['physical_heads'] for m in h['members']}
            for i,task in enumerate(tasks):
                ax=fig.add_subplot(grid[0,i],projection='3d');ax.add_collection3d(Poly3DCollection(task.domain.mesh.triangles,facecolor='#999999',alpha=.12,edgecolor='none'))
                cp=folder/f'contacts_{task.pose}.npz';files.append(cp)
                for c in M.I.read_contacts(cp):
                    color=cmap(ownership[M.name(task,c)]%20)
                    ax.add_collection3d(Poly3DCollection(c['triangles_m'],facecolor=color,edgecolor='none'))
                d=directions[sets['selected_indices'][i]];span=max(task.domain.mesh.extents)
                ax.quiver(*task.domain.com,*d,length=.7*span,color=cmap(i),linewidth=2)
                axes_bounds(ax,task.domain.mesh.vertices,.5);ax.set_title(f"{task.pose}: contacts + selected world exit")
            ax=fig.add_subplot(grid[1,:-1],projection='3d')
            for i,h in enumerate(state['physical_heads']):
                vertices=z[f'h{i}_vertices'];faces=z[f'h{i}_faces']
                ax.add_collection3d(Poly3DCollection(vertices[faces],facecolor=cmap(color_ids[h['id']]%20),edgecolor='none'))
            axes_bounds(ax,world_bounds);ax.set_title(f"Common fixture coordinates: {state['physical_head_count']} physical head modules")
            sphere=fig.add_subplot(grid[1,-1],projection='3d');mask=np.array(sets['per_pose_mask'])
            for i,row in enumerate(mask):
                d=directions[row];sphere.scatter(d[:,0],d[:,1],d[:,2],s=24,color=cmap(i),label=tasks[i].pose)
            common=directions[mask.all(0)]
            if len(common):sphere.scatter(common[:,0],common[:,1],common[:,2],color='black',marker='*',s=100)
            sphere.set_xlim(-1,1);sphere.set_ylim(-1,1);sphere.set_zlim(-.1,1);sphere.set_box_aspect((1,1,1));sphere.legend(fontsize=8)
            sphere.set_title(f"Checked world rays; common={len(common)}");sphere.set_xlabel('X');sphere.set_ylabel('Y');sphere.set_zlabel('Z')
            label='Initial' if k==0 else ('Final' if k==len(folders)-1 else f'Accepted head edit {k}')
            status='PASS' if r['passed'] else 'FAIL'
            fig.suptitle(f'{group} | {label} | all original head loads checked | final full fixture: {status}\nAdaptive sampled directions, not the complete analytic exit set',fontsize=15)
            fig.tight_layout(rect=(0,0,1,.92));png=Path(temp)/f'{k:03d}.png';fig.savefig(png);plt.close(fig)
            sequence += [png]*(3 if k==0 else 4 if k==len(folders)-1 else 1)
            files += [folder/'state.json',folder/'head_geometry.npz']
        manifest=Path(temp)/'frames.txt'
        manifest.write_text(''.join(f"file '{p}'\nduration 1\n" for p in sequence)+f"file '{sequence[-1]}'\n")
        target=DEST/f'B_{group}_{stage}.mp4'
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',str(manifest),'-c:v','libx264','-threads','2','-pix_fmt','yuv420p','-r','12',str(target)],check=True)
        M.I.save(target.with_suffix('.json'),dict(group=group,stage=stage,recorded_states=len(folders),full_fixture_passed=r['passed'],head_common_count=r['final_head_exits']['common_count'],finite_sampled_sets=True,visualization_only=True,inputs=M.I.hashes(files),code=M.I.hashes([Path(__file__)])))
        print(target,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sets',nargs='+',default=M.GROUPS);p.add_argument('--stage',default='global_release_v7');a=p.parse_args()
    for g in a.sets:render(g,a.stage)
