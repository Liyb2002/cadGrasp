"""Existing rendering function separated from the numerical solver."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from co_common import *
def render(mesh,remaining,removed,sweeps,states,rows,out,length):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.patches import Patch
    n=len(rows);cols=min(3,n);nr=(n+cols-1)//cols
    fig=plt.figure(figsize=(6*cols,6*nr))
    radius=max(np.ptp(np.vstack([transform_points(m.vertices,T) for m in [mesh,remaining,removed,sweeps[i]] if len(m.vertices)]),axis=0).max() for i,(task,T) in enumerate(states))*1000*.55
    for i,((task,T),row) in enumerate(zip(states,rows)):
        ax=fig.add_subplot(nr,cols,i+1,projection='3d');objects=[];sweep=sweeps[i]
        for m,color,alpha in [(sweep,'#36bbd0',.10),(remaining,'#989898',.5),(mesh,'#79a9d7',.26),(removed,'#ef4938',.8)]:
            if not len(m.vertices):continue
            world=m.copy();world.apply_transform(T);objects.append(world.vertices*1000)
            ax.add_collection3d(Poly3DCollection(world.triangles*1000,facecolor=color,edgecolor='none',alpha=alpha))
        vertices=np.vstack(objects);center=(vertices.min(0)+vertices.max(0))/2
        floor=np.array([[center[0]-radius,center[1]-radius,0],[center[0]+radius,center[1]-radius,0],[center[0]+radius,center[1]+radius,0],[center[0]-radius,center[1]+radius,0]])
        ax.add_collection3d(Poly3DCollection([floor],facecolor='#eeeeee',edgecolor='#bbbbbb',alpha=.15))
        start=transform_points(mesh.vertices.mean(0)[None,:],T)[0]*1000;native=np.asarray(row['direction_world'])
        ax.quiver(*start,*native,length=100,color='#009cae',arrow_length_ratio=.13,linewidth=2)
        for axis,x in zip('xyz',center):getattr(ax,'set_'+axis+'lim')(x-radius,x+radius)
        ax.set_box_aspect((1,1,1));ax.view_init(22,-55);ax.set_axis_off()
        force=f"{row['force_covered']}/{row['load_count']}" if row['force_error'] is None else 'unresolved'
        ax.set_title(f"{row['pose'].replace('_',' ').title()} — initial exit\nRemaining force / torque demands: {force}",fontsize=11)
    fig.suptitle('Step4.1: each pose exits along its own +Z; red = all required cuts\nCyan shows first 100 mm only; cutting uses the full continuous exit',fontsize=14)
    fig.legend(handles=[Patch(color=c,label=l) for c,l in [('#79a9d7','Object'),('#36bbd0','Exit sweep'),('#ef4938','Removed support'),('#989898','Remaining support')]],loc='lower center',ncol=4)
    fig.subplots_adjust(left=0,right=1,bottom=.05,top=.80 if nr==1 else .89,wspace=0,hspace=.08)
    fig.savefig(out/'overview.png',dpi=150,bbox_inches='tight');plt.close(fig)

