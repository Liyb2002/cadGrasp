"""Pose1 direction-centered hemisphere illustration, distinct from swept solids."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
from co_common import *
from solid_render import depth_render
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import time


def render(name,group):
    began=time.monotonic();pose='pose_1' if 'pose_1' in group['poses'] else group['poses'][0]
    task,T,_=state(name,pose);obj=task.domain.mesh.copy()
    root=HERE/'output'/name if name=='B' else HERE/'data/object_inputs'/name
    out=root/group['id']/'step4/step4.1';out.mkdir(parents=True,exist_ok=True)
    center=(obj.bounds[0]+obj.bounds[1])/2;radius=float(obj.extents.max())*.67
    examples=[('Upward',[0,0,1],'#228653'),('Upward right',[1,1,1],'#e19725'),('Upward left',[-1,-1,1],'#497fd5'),('Shallow upward right',[1,1,.35],'#a456ba')]
    fig=plt.figure(figsize=(10,9));ax=fig.add_subplot(111,projection='3d',computed_zorder=False)
    elev=np.degrees(np.arctan(1/np.sqrt(2)));camera=np.array([1.,-1.,1.])/np.sqrt(3);light=camera+[0,0,.4];light/=np.linalg.norm(light)
    triangles=[obj.triangles*1000];brightness=.5+.5*np.maximum(obj.face_normals@light,0);colors=[np.c_[brightness[:,None]*np.array([.55,.57,.6]),np.ones(len(obj.faces))]]
    envelopes=[];tints=[];points=[obj.vertices*1000];records=[]
    for label,vector,color in examples:
        d=np.array(vector,dtype=float);d/=np.linalg.norm(d)
        sphere=trimesh.creation.icosphere(subdivisions=4,radius=radius)
        shell=trimesh.intersections.slice_mesh_plane(sphere,d,np.zeros(3),cap=False);shell.apply_translation(center)
        envelopes.append(shell.triangles*1000);tints.append(np.tile(matplotlib.colors.to_rgba(color,.075),(len(shell.faces),1)));points.append(shell.vertices*1000)
        records.append(dict(label=label,direction_world=d.tolist(),hemisphere_definition='unit displacement u with dot(u,direction)>=0',color=color))
    points=np.vstack(points);mid=(points.min(0)+points.max(0))/2;r=np.ptp(points,axis=0).max()*.54
    for axis,value in zip('xyz',mid):getattr(ax,'set_'+axis+'lim')(value-r,value+r)
    ax.set_proj_type('ortho');ax.set_box_aspect((1,1,1),zoom=1.3);ax.view_init(elev=elev,azim=-45);ax.set_axis_off()
    ax.set_title('Pose1 rabbit: four direction-centered hemispheres',fontsize=15,pad=0)
    fig.legend(handles=[Patch(facecolor=matplotlib.colors.to_rgba(c,.20),edgecolor='none',label=l) for l,_,c in examples],loc='lower center',ncol=2,frameon=False)
    fig.text(.5,.055,'Hemisphere overlays are direction illustrations, not actual exit sweeps.',ha='center',fontsize=10)
    fig.subplots_adjust(left=0,right=1,bottom=.12,top=.93);fig.set_dpi(165);fig.canvas.draw()
    depth_render(ax,np.concatenate(triangles),np.concatenate(colors),np.concatenate(envelopes),np.concatenate(tints))
    target=out/'direction_coverage.png';fig.savefig(target,dpi=165);plt.close(fig)
    save(out/'data/direction_coverage.json',dict(presentation_only=True,arrows=False,hemisphere_opacity=.075,overlap_style='transparent alpha composition; multiple coverings deepen color',pose=pose,hemisphere_radius_m=radius,center_world=center.tolist(),directions=records,actual_sweep=False,all_hemisphere_points_floor_valid=False,floor_valid_direction_condition='direction_world.z >= 0',seconds=time.monotonic()-began,artifacts={'../direction_coverage.png':I.sha256(target)},provenance=provenance(task.inputs,[Path(__file__),Path(__file__).with_name('solid_render.py')])))

if __name__=='__main__':
    from codes.precompute_objects.dataset import read_selected_pose_groups
    render('B',next(g for g in read_selected_pose_groups('B') if g['id']=='pose1+2+4+6'))
