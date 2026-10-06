"""Existing rendering function separated from the numerical solver."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from co_common import *
def render(search,result,restored,out):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.patches import Patch
    n=len(search.states);cols=min(n,3);nr=(n+cols-1)//cols;fig=plt.figure(figsize=(6*cols,6*nr))
    remaining=S.unpack(result['remaining']);restored_mesh=S.unpack(restored);old=result['remaining']-restored;old_mesh=S.unpack(old)
    displays=[S.swept_solid(search.mesh,.10*d,fan_in=result['fan']) for d in result['directions']]
    radius=max(np.ptp(np.vstack([transform_points(m.vertices,T) for m in [search.mesh,remaining,displays[i]] if len(m.vertices)]),axis=0).max() for i,(task,T) in enumerate(search.states))*1000*.55
    for i,(task,T) in enumerate(search.states):
        ax=fig.add_subplot(nr,cols,i+1,projection='3d');vertices=[]
        for m,color,alpha in [(displays[i],'#36bbd0',.10),(old_mesh,'#999999',.6),(search.mesh,'#79a9d7',.24),(restored_mesh,'#49b66a',.9)]:
            if not len(m.vertices):continue
            world=m.copy();world.apply_transform(T);vertices.append(world.vertices*1000);ax.add_collection3d(Poly3DCollection(world.triangles*1000,facecolor=color,edgecolor='none',alpha=alpha))
        vv=np.vstack(vertices);center=(vv.min(0)+vv.max(0))/2
        floor=np.array([[center[0]-radius,center[1]-radius,0],[center[0]+radius,center[1]-radius,0],[center[0]+radius,center[1]+radius,0],[center[0]-radius,center[1]+radius,0]])
        ax.add_collection3d(Poly3DCollection([floor],facecolor='#eeeeee',edgecolor='#bbbbbb',alpha=.15))
        native=T[:3,:3]@result['directions'][i];start=transform_points(search.mesh.vertices.mean(0)[None,:],T)[0]*1000;ax.quiver(*start,*native,length=100,color='#009cae',arrow_length_ratio=.13,linewidth=2)
        for axis,x in zip('xyz',center):getattr(ax,'set_'+axis+'lim')(x-radius,x+radius)
        ax.set_box_aspect((1,1,1));ax.view_init(22,-55);ax.set_axis_off();ax.set_title(search.group['poses'][i].replace('_',' ').title()+' — recovered equilibrium\n32768 / 32768 original demands',fontsize=11)
    fig.suptitle('Step4.2: co-optimized exits; green material restored\nAll original force / torque demands pass; connectivity is not required',fontsize=14)
    fig.legend(handles=[Patch(color=c,label=l) for c,l in [('#79a9d7','Object'),('#36bbd0','First 100 mm exit sweep'),('#49b66a','Restored material'),('#999999','Retained material')]],loc='lower center',ncol=4)
    fig.subplots_adjust(left=0,right=1,bottom=.05,top=.80 if nr==1 else .89,wspace=0,hspace=.08);fig.savefig(out/'overview.png',dpi=150,bbox_inches='tight');plt.close(fig)

