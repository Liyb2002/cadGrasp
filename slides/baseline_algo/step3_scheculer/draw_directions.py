"""Show how each selected head consumes the shared 3-D direction catalogue."""
import argparse
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step1.cases import selected_pose,pose_name
from step3_scheculer import contacts as I
from step2_local_support import withdrawal as W


def run(name):
    out=I.OUTPUTS/name/pose_name()/'step3_scheculer'
    source=out/'insertion_directions.json'
    partial=not source.exists()
    if partial:
        records=sorted(out.parent.glob('step3.3_optimize_contact/round_*/insertion_directions.json'))
        if not records: raise RuntimeError('No optimized direction record is available')
        source=records[-1]
    result=I.check_report(source);cat=result['direction_catalogue']
    vectors=np.asarray(cat['vectors']);azimuth=np.rad2deg(np.arctan2(vectors[:,2],vectors[:,0]));elevation=np.rad2deg(np.arcsin(np.clip(vectors[:,1],-1,1)))
    states=[('Work-face + floor locks',cat['global_allowed_directions'])]
    sources=[source]
    for path in sorted(out.parent.glob('step3.3_optimize_contact/round_*/insertion_directions.json')):
        r=I.check_report(path);sources.append(path)
        states.append((f"Round {r['round']}: + {r['selected_ids'][-1]}",r['common_directions']))
    all_states=states
    if len(states)>8:
        indices=sorted(set([0,1]+np.linspace(2,len(states)-1,6).astype(int).tolist()))
        states=[states[i] for i in indices]
    panel_count=len(states)+1
    cols=min(3,panel_count);rows=int(np.ceil(panel_count/cols))
    fig,axes=plt.subplots(rows,cols,figsize=(6*cols,3.8*rows),squeeze=False)
    previous=set(range(len(vectors)))
    for ax,(label,remaining) in zip(axes.flat,states):
        alive=set(remaining['ids']);new=previous-alive
        colors=['#25876b' if i in alive else '#d65c4b' if i in new else '#d4d7da' for i in range(len(vectors))]
        ax.scatter(azimuth,elevation,c=colors,s=12,linewidths=0)
        ax.set(xlim=(-185,185),ylim=(-95,95),xlabel='Withdrawal azimuth (degrees)',ylabel='Elevation (degrees)',title=f'{label} / {len(alive)} directions remain')
        ax.grid(alpha=.18);previous=alive
    ax=list(axes.flat)[len(states)]
    ax.plot(range(len(all_states)),[len(r['ids']) for _,r in all_states],marker='o',color='#25876b',markersize=3)
    ax.set(xlabel='Completed optimization round',ylabel='Common direction count',title='Every round: remaining directions')
    ax.grid(alpha=.2)
    for ax in list(axes.flat)[panel_count:]:ax.set_visible(False)
    fig.suptitle(f'{name} / {pose_name()} / '+('PARTIAL Step3 / ' if partial else '')+f'Common withdrawal through greedy selection\nGreen: alive; red: locked since previous shown panel; gray: earlier locks. Finite 3-D directions, complete head-ray checks.',fontsize=12)
    fig.tight_layout(rect=(0,0,1,1-.14/rows));fig.savefig(out/'withdrawal_directions.png',dpi=150);plt.close(fig)
    I.save(out/'withdrawal_direction_views.json',dict(complete=True,partial_schedule=partial,selected_ids=result['selected_ids'],
        common_directions=result['common_directions'],round_survivor_counts=[len(r['ids']) for _,r in all_states],provenance=dict(inputs=I.hashes(sources),code=I.hashes([Path(__file__)])),
        artifacts={'withdrawal_directions.png':I.sha256(out/'withdrawal_directions.png')}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('objects', nargs='*', default=['A1-f', 'B', 'C5']);parser.add_argument('--pose',default=None);args=parser.parse_args()
    with selected_pose(args.pose):
        for name in args.objects: run(name)
