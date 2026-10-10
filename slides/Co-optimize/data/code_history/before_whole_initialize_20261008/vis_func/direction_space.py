"""Actual Step3.3 support: one legal upward sweep and one illegal downward sweep."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
from co_common import *
from solid_render import arrow_mesh, depth_render
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as effects
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import argparse,time


def draw_panel(ax,obj,kept,removed,envelope,T,directions,crossed=False,bounds=None):
    elev=float(np.degrees(np.arctan(1/np.sqrt(2))));az=-45.
    e,a=np.radians([elev,az]);camera=np.array([np.cos(e)*np.cos(a),np.cos(e)*np.sin(a),np.sin(e)])
    right=np.array([-np.sin(a),np.cos(a),0.]);up=np.cross(camera,right)
    light=camera+np.array([-.15,-.2,.65]);light/=np.linalg.norm(light)
    triangles=[];colors=[];points=[]
    for source,color in [(obj,'#a4a8ac'),(kept,'#319cd7')]:
        if not len(source.faces):continue
        mesh=source.copy();mesh.apply_transform(T);points.append(mesh.vertices*1000)
        brightness=.58+.42*np.maximum(mesh.face_normals@light,0)
        triangles.append(mesh.triangles*1000);colors.append(np.c_[brightness[:,None]*np.array(matplotlib.colors.to_rgb(color)),np.ones(len(mesh.faces))])
    opaque_triangles=np.concatenate(triangles);opaque_colors=np.concatenate(colors)
    world_envelope=envelope.copy();world_envelope.apply_transform(T)
    tint='#e47d83' if crossed else '#58b69c'
    
    points.append(world_envelope.vertices*1000)
    world_obj=transform_points(obj.vertices,T)*1000;anchor=world_obj.mean(axis=0)
    # Move annotation origins slightly toward the camera, keeping the arrows on
    # the object rather than in a disconnected side diagram.
    anchor+=camera*(.65*float(obj.extents.max())*1000)
    length=.075*1000
    for direction in directions:
        d=np.asarray(direction);color='#d84038' if crossed else '#167f5b'
        mesh=arrow_mesh(anchor,d,length);brightness=.55+.45*np.maximum(mesh.face_normals@light,0)
        opaque_triangles=np.concatenate([opaque_triangles,mesh.triangles]);opaque_colors=np.concatenate([opaque_colors,np.c_[brightness[:,None]*np.array(matplotlib.colors.to_rgb(color)),np.ones(len(mesh.faces))]])
        points.append(np.array([anchor,anchor+length*d]))
    if crossed:
        center=anchor+length*np.asarray(directions[0])*.68;size=.016*1000
        for axis in [right+up,right-up]:
            axis/=np.linalg.norm(axis);line=np.array([center-size*axis,center+size*axis]);stroke=ax.plot(*line.T,color='#cf302e',linewidth=4,zorder=11)[0];stroke.set_path_effects([effects.withStroke(linewidth=7,foreground='white')])
    all_points=np.vstack(points) if bounds is None else bounds
    center=(all_points.min(0)+all_points.max(0))/2;radius=float(np.ptp(all_points,axis=0).max())*.53
    for axis,value in zip('xyz',center):getattr(ax,'set_'+axis+'lim')(value-radius,value+radius)
    ax.set_box_aspect((1,1,1),zoom=1.8);ax.set_proj_type('ortho');ax.view_init(elev=elev,azim=az);ax.set_axis_off()
    floor=transform_points(obj.vertices,T)*1000;lo=floor[:,:2].min(0)-25;hi=floor[:,:2].max(0)+25
    plane=np.array([[lo[0],lo[1],0],[hi[0],lo[1],0],[hi[0],hi[1],0],[lo[0],hi[1],0],[lo[0],lo[1],0]])
    ax.plot(*plane.T,color='#c5ccd4',linewidth=.8,zorder=1)
    ax._depth_geometry=(opaque_triangles,opaque_colors,world_envelope.triangles*1000,matplotlib.colors.to_rgba(tint,.105))


def render(name,group,directions=None,metadata=None,support_source=None):
    """Draw six fixed illustrative directions, independent of the optimizer."""
    began=time.monotonic();root=HERE/'output'/name if name=='B' else HERE/'data/object_inputs'/name
    base=root/group['id'];out=base/'step4/step4.1';(out/'data').mkdir(parents=True,exist_ok=True)
    support_path=Path(support_source) if support_source is not None else base/'step3/step3.2/wrapped_support.obj';object_path=base/'step3/step3.1/registered_object.obj'
    support=trimesh.load(support_path,force='mesh',process=False)
    poses=['pose_1'] if 'pose_1' in group['poses'] else group['poses'][:1]
    states=[state(name,p) for p in poses];inputs=[support_path,object_path]
    length=.10;fig=plt.figure(figsize=(16,10.5),facecolor='white');records=[];artifacts={}
    headers=['Valid: upward','Valid: upward right','Valid: upward left','Valid: shallow upward right','Invalid: straight downward (X)','Invalid: steep downward left (X)']
    examples=[('upward',[0.,0.,1.]),('upward_right',[1.,1.,1.]),('upward_left',[-1.,-1.,1.]),('shallow_upward_right',[1.,1.,.35]),('downward',[0.,0.,-1.]),('steep_downward_left',[-.55,-.55,-1.])]
    for i,(pose,(task,T,_)) in enumerate(zip(poses,states)):
        inputs+=task.inputs
        world_obj=task.domain.mesh.copy();world_support=support.copy();world_support.apply_transform(T)
        seed=S.solid(world_support);values=[];branches=[]
        for kind,vector in examples:
            direction=np.asarray(vector,dtype=float);direction/=np.linalg.norm(direction)
            sweep_mesh=S.swept_solid(world_obj,length*direction)
            sweep=S.solid(sweep_mesh);removed=seed^sweep;kept=seed-sweep
            removed_mesh=S.unpack(removed);kept_mesh=S.unpack(kept)
            for suffix,exported in [('sweep',sweep_mesh),('removed',removed_mesh)]:
                path=out/'data'/f'{pose}_{kind}_{suffix}.obj';D.export_exact_obj(exported,path);artifacts[path.name]=I.sha256(path)
            branches.append((kind,direction,sweep_mesh,kept_mesh,removed_mesh))
            values.append(dict(kind=kind,direction_world=direction.tolist(),direction_fixture=(T[:3,:3].T@direction).tolist(),floor_legal=bool(direction[2]>=0),minimum_sweep_world_z_m=float(sweep_mesh.vertices[:,2].min()),removed_support_volume_cm3=material_volume(removed)*1e6,sweep_artifact=f'{pose}_{kind}_sweep.obj',removed_support_artifact=f'{pose}_{kind}_removed.obj',artifact_coordinate_frame='native world'))
        bounds=np.vstack([world_support.vertices,world_obj.vertices]+[b[2].vertices for b in branches])*1000
        for j,(kind,direction,sweep_mesh,kept_mesh,removed_mesh) in enumerate(branches):
            ax=fig.add_subplot(2,3,j+1,projection='3d',computed_zorder=False)
            draw_panel(ax,world_obj,kept_mesh,removed_mesh,sweep_mesh,np.eye(4),[direction],crossed=direction[2]<0,bounds=bounds)
        records.append(dict(pose=pose,T_fixture_to_world=T.tolist(),sweeps=values))
        print('UP / DOWN SWEEPS',group['id'],pose,flush=True)
    fig.subplots_adjust(left=.015,right=.985,top=.965,bottom=.045,wspace=0,hspace=.08)
    fig.set_dpi(165);fig.canvas.draw()
    for ax in fig.axes:
        if hasattr(ax,'_depth_geometry'):depth_render(ax,*ax._depth_geometry)
        for line in ax.lines:
            if line.get_zorder()==11:
                from mpl_toolkits.mplot3d import proj3d
                from matplotlib.lines import Line2D
                x,y,z=line.get_data_3d();px,py,_=proj3d.proj_transform(x,y,z,ax.get_proj());xy=fig.transFigure.inverted().transform(ax.transData.transform(np.c_[px,py]))
                overlay=Line2D(*xy.T,transform=fig.transFigure,color='#cf302e',linewidth=4,zorder=3);overlay.set_path_effects([effects.withStroke(linewidth=7,foreground='white')]);fig.add_artist(overlay)
    target=out/'direction_space.png';fig.savefig(target,dpi=165,facecolor='white');plt.close(fig)
    (out/'direction_space.html').unlink(missing_ok=True)
    artifacts['../direction_space.png']=I.sha256(target)
    record=dict(complete=True,presentation_only=True,object=name,pose_set=group['id'],poses=poses,support_source=str(support_path.relative_to(ROOT)),uses_actual_step33_shape=False,base_deferred=True,display='Pose1: four distinct upper-hemisphere valid directions and two strongly lower-hemisphere invalid directions; solid 3D arrows',arrows_are_optimizer_directions=False,text_or_labels=False,presentation_scale_relative_to_previous=1.5,removed_support_displayed=False,removed_support_display='Omitted; draw only retained support',states=records,display_length_m=length,red_material_method='Boolean intersection of original Step3.3 support with each continuous straight translation sweep',sweep_method='Continuous full-object translation; no hemisphere displacement family',html_deleted=True,mechanics_rerun=False,saved_support_changed=False,seconds=time.monotonic()-began,provenance=provenance(inputs,[Path(__file__),Path(S.__file__).with_name('translation_sweep.py')]),artifacts=artifacts)
    save(out/'data/direction_space.json',record)
    return record


def main():
    p=argparse.ArgumentParser();p.add_argument('--object',default='B');p.add_argument('--sets',nargs='+');args=p.parse_args()
    groups=read_selected_pose_groups(args.object)
    if args.sets:groups=[g for g in groups if g['id'] in args.sets]
    for g in groups:
        result=render(args.object,g);print('SIX DIRECTION SWEEPS',g['id'],result['seconds'],flush=True)

if __name__=='__main__':main()
