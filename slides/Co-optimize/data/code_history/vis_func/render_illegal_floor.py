"""Side views of saved illegal-set supports; visualization only."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from co_common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

DEST=HERE/'output/B/illegal/data/floor_visualization'

def clip(poly, below):
    result=[]
    for a,b in zip(poly,np.roll(poly,-1,axis=0)):
        ia=a[2]<=0 if below else a[2]>=0
        ib=b[2]<=0 if below else b[2]>=0
        if ia:result.append(a)
        if ia!=ib:result.append(a+(b-a)*(-a[2]/(b[2]-a[2])))
    return np.asarray(result)

def panel(ax, base, pose, title):
    task,T,_=state('B',pose)
    support=trimesh.load(base/'step4/step4.2/remaining_support.obj',force='mesh',process=False)
    v=transform_points(support.vertices,T)*1000
    obj=task.domain.mesh.vertices*1000
    # View along the narrower horizontal dimension to expose the broad side.
    horizontal=int(np.argmax(np.ptp(np.vstack([v,obj])[:,:2],axis=0)))
    projection=[horizontal,2]
    for below,color in [(False,'#8b939c'),(True,'#e34336')]:
        polygons=[p[:,projection] for tri in v[support.faces] if len(p:=clip(tri,below))>=3]
        ax.add_collection(PolyCollection(polygons,facecolors=color,edgecolors='none',alpha=1,zorder=2))
    # Object is drawn translucent, with silhouette edges supplied by triangles.
    ax.add_collection(PolyCollection(obj[task.domain.mesh.faces][:,:,projection],facecolors='#42a9d5',edgecolors='none',alpha=.035,zorder=3))
    bounds=np.vstack([v[:,projection],obj[:,projection]])
    lo,hi=bounds.min(0),bounds.max(0);margin=max(hi-lo)*.07
    ax.axhspan(lo[1]-margin,0,color='#fff0ec',zorder=0)
    ax.axhline(0,color='#60432d',lw=1.8,zorder=4)
    deepest=v[np.argmin(v[:,2])];depth=max(0,-deepest[2])
    ax.plot([deepest[horizontal],deepest[horizontal]],[0,deepest[2]],color='#a51e18',lw=1.4,zorder=5)
    ax.scatter([deepest[horizontal]],[deepest[2]],s=12,color='#a51e18',zorder=5)
    ax.set_xlim(lo[0]-margin,hi[0]+margin);ax.set_ylim(lo[1]-margin,hi[1]+margin)
    ax.set_aspect('equal');ax.set_title(f'{title}\n{pose}: below floor {depth:.3f} mm',fontsize=10)
    ax.set_xlabel(('World X' if horizontal==0 else 'World Y')+' (mm)',fontsize=8)
    ax.set_ylabel('World Z (mm)',fontsize=8);ax.tick_params(labelsize=7)
    ax.spines[['top','right']].set_visible(False)
    return dict(group=base.name,pose=pose,minimum_support_world_z_mm=float(v[:,2].min()),minimum_object_world_z_mm=float(obj[:,2].min()),horizontal_axis=horizontal)

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    rows=json.loads((DEST.parent/'summary.json').read_text())['results']
    passed=[r for r in rows if r['passed']]
    records=[]
    fig,axes=plt.subplots(3,3,figsize=(15,12))
    for ax,r in zip(axes.flat,passed):
        worst=min(r['saved_support_floor_heights'],key=lambda x:x['minimum_world_z_m'])
        records.append(panel(ax,DEST.parent.parent/r['id'],worst['pose'],r['id']))
    fig.suptitle('Illegal pose sets: worst installed pose per saved support\nGray = support above floor | Red = support below floor | Blue = object | Brown line = floor (Z = 0)',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(DEST/'all_sets_floor_penetration.png',dpi=160);plt.close(fig)
    group=next(r for r in passed if r['id']=='pose7+11+13+19')
    fig,axes=plt.subplots(2,2,figsize=(12,10))
    for ax,r in zip(axes.flat,group['saved_support_floor_heights']):
        panel(ax,DEST.parent.parent/group['id'],r['pose'],'Same shared support, different installation')
    fig.suptitle('pose7+11+13+19: floor penetration in each installed pose\nGray = support above floor | Red = support below floor | Blue = object | Brown = floor',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(DEST/'pose7+11+13+19_floor_penetration.png',dpi=160);plt.close(fig)
    (DEST/'render_records.json').write_text(json.dumps(dict(visualization_only=True,geometry_changed=False,records=records),indent=2)+'\n')
    print(DEST)
if __name__=='__main__':main()
