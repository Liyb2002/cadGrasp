"""Actual Step3.3 support, selected exit sweeps and hemisphere sweep envelopes.

A hemisphere kernel describes candidate displacement vectors. Its Minkowski sum
with the object is the swept envelope of ALL those candidates. One selected
exit uses a segment kernel instead. Red is the actual support intersection with
the shown sweep/envelope, without changing the saved support or running loads.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
from co_common import *
from optimization.initial_directions import initialize_close_directions
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as effects
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import argparse,base64,html,time


def draw_panel(ax,obj,kept,removed,envelope,T,directions,crossed=False):
    elev=float(np.degrees(np.arctan(1/np.sqrt(2))));az=-45.
    e,a=np.radians([elev,az]);camera=np.array([np.cos(e)*np.cos(a),np.cos(e)*np.sin(a),np.sin(e)])
    right=np.array([-np.sin(a),np.cos(a),0.]);up=np.cross(camera,right)
    light=camera+np.array([-.15,-.2,.65]);light/=np.linalg.norm(light)
    triangles=[];colors=[];points=[]
    for source,color in [(obj,'#a4a8ac'),(kept,'#319cd7'),(removed,'#ed493c')]:
        if not len(source.faces):continue
        mesh=source.copy();mesh.apply_transform(T);points.append(mesh.vertices*1000)
        brightness=.58+.42*np.maximum(mesh.face_normals@light,0)
        triangles.append(mesh.triangles*1000);colors.append(np.c_[brightness[:,None]*np.array(matplotlib.colors.to_rgb(color)),np.ones(len(mesh.faces))])
    ax.add_collection3d(Poly3DCollection(np.concatenate(triangles),facecolors=np.concatenate(colors),edgecolors='none',zsort='average',zorder=3))
    world_envelope=envelope.copy();world_envelope.apply_transform(T)
    tint='#e47d83' if crossed else '#58b69c'
    ax.add_collection3d(Poly3DCollection(world_envelope.triangles*1000,facecolors=tint,edgecolors='none',alpha=.105,zsort='average',zorder=4))
    points.append(world_envelope.vertices*1000)
    world_obj=transform_points(obj.vertices,T)*1000;anchor=world_obj.mean(axis=0)
    # Move annotation origins slightly toward the camera, keeping the arrows on
    # the object rather than in a disconnected side diagram.
    anchor+=camera*(.12*float(obj.extents.max())*1000)
    length=.075*1000
    for direction in directions:
        d=np.asarray(direction);color='#d84038' if crossed else '#167f5b'
        arrow=ax.quiver(*anchor,*d,length=length,color=color,linewidth=2.4,arrow_length_ratio=.16,zorder=10)
        arrow.set_path_effects([effects.withStroke(linewidth=4.8,foreground='white')])
        points.append(np.array([anchor,anchor+length*d]))
    if crossed:
        center=anchor+length*np.asarray(directions[0])*.68;size=.016*1000
        for axis in [right+up,right-up]:
            axis/=np.linalg.norm(axis);line=np.array([center-size*axis,center+size*axis]);stroke=ax.plot(*line.T,color='#cf302e',linewidth=4,zorder=11)[0];stroke.set_path_effects([effects.withStroke(linewidth=7,foreground='white')])
    if crossed or len(directions)>1:
        # These arcs show the hemisphere travelled by the object's CENTER;
        # the translucent nonconvex Minkowski envelope shows the whole object.
        origin=world_obj.mean(axis=0);radius_mm=100.;sign=-1. if crossed else 1.
        color='#b9545d' if crossed else '#319675';theta=np.linspace(0,np.pi/2,80)
        for phi in [-np.pi/4,np.pi/4,3*np.pi/4,5*np.pi/4]:
            arc=origin+radius_mm*np.c_[np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),sign*np.cos(theta)]
            ax.plot(*arc.T,color=color,linewidth=.85,alpha=.65,zorder=6)
        phi=np.linspace(0,2*np.pi,160);equator=origin+radius_mm*np.c_[np.cos(phi),np.sin(phi),np.zeros(len(phi))]
        ax.plot(*equator.T,color=color,linewidth=.85,alpha=.65,zorder=6)
        points.append(equator)
    all_points=np.vstack(points);center=(all_points.min(0)+all_points.max(0))/2;radius=float(np.ptp(all_points,axis=0).max())*.53
    for axis,value in zip('xyz',center):getattr(ax,'set_'+axis+'lim')(value-radius,value+radius)
    ax.set_box_aspect((1,1,1),zoom=1.2);ax.set_proj_type('ortho');ax.view_init(elev=elev,azim=az);ax.set_axis_off()
    floor=transform_points(obj.vertices,T)*1000;lo=floor[:,:2].min(0)-25;hi=floor[:,:2].max(0)+25
    plane=np.array([[lo[0],lo[1],0],[hi[0],lo[1],0],[hi[0],hi[1],0],[lo[0],hi[1],0],[lo[0],lo[1],0]])
    ax.plot(*plane.T,color='#c5ccd4',linewidth=.8,zorder=1)


def render(name,group,directions=None,metadata=None):
    began=time.monotonic();root=HERE/'output'/name if name=='B' else HERE/'data/object_inputs'/name
    base=root/group['id'];out=base/'step4/step4.1';(out/'data').mkdir(parents=True,exist_ok=True)
    support_path=base/'step3/step3.3/support_with_rings.obj';object_path=base/'step3/step3.1/registered_object.obj'
    support=trimesh.load(support_path,force='mesh',process=False);obj=trimesh.load(object_path,force='mesh',process=False)
    states=[state(name,p) for p in group['poses']];normals=np.array([T[:3,:3].T@np.array([0.,0.,1.]) for task,T,mesh in states])
    initialized,computed_metadata=initialize_close_directions(normals);inputs=[support_path,object_path,ROOT/'objects'/name/'selected_pose_sets.json']
    initializer_path=root/'data/step41/initialization.json';saved=json.loads(initializer_path.read_text()) if initializer_path.exists() else {};saved_group=saved.get('groups',{}).get(group['id'])
    if directions is None:
        if saved_group and not saved_group.get('preparation_error'):
            directions=np.array([saved_group['paths'][p]['direction_fixture'] for p in group['poses']]);metadata=saved_group['metadata'];inputs.append(initializer_path)
        else:directions=initialized
    elif isinstance(directions,dict):directions=np.array([directions[p] for p in group['poses']])
    else:directions=np.asarray(directions)
    metadata=computed_metadata if metadata is None else metadata
    length=.10;seed=S.solid(support);ball=S.solid(trimesh.creation.icosphere(subdivisions=1,radius=length))
    fig=plt.figure(figsize=(16,4.5*len(group['poses'])),facecolor='white');records=[];artifacts={};up_volumes=[]
    headers=['Selected close exit direction','Up-hemisphere sweep envelope','Down-hemisphere sweep envelope (X)']
    for i,(pose,(task,T,pose_mesh),direction) in enumerate(zip(group['poses'],states,directions)):
        inputs+=task.inputs
        print('SWEEP ENVELOPE BEGIN',group['id'],pose,flush=True)
        object_solid=S.solid(pose_mesh)
        cache=root/'data/step41'/group['id']/pose/'display_sweep.obj'
        cached_ok=saved_group and np.allclose(direction,saved_group['paths'][pose]['direction_fixture'],atol=1e-12,rtol=0) and saved.get('display_length_m')==length and cache.exists()
        if cached_ok:chosen=S.solid(trimesh.load(cache,force='mesh',process=False));inputs.append(cache)
        else:chosen=S.solid(S.swept_solid(pose_mesh,length*direction))
        upper_kernel=ball.trim_by_plane(normals[i].tolist(),0.)
        lower_kernel=ball.trim_by_plane((-normals[i]).tolist(),0.)
        upper=object_solid.minkowski_sum(upper_kernel);lower=object_solid.minkowski_sum(lower_kernel)
        values=[]
        for kind,envelope in [('selected',chosen),('upper',upper),('lower',lower)]:
            removed=seed^envelope;kept=seed-envelope
            mesh=S.unpack(envelope);removed_mesh=S.unpack(removed);kept_mesh=S.unpack(kept)
            for suffix,exported in [('envelope',mesh),('removed',removed_mesh)]:
                path=out/'data'/f'{pose}_{kind}_{suffix}.obj';D.export_exact_obj(exported,path);artifacts[path.name]=I.sha256(path)
            native=T[:3,:3]@direction
            if kind=='selected':arrows=[native];crossed=False
            elif kind=='upper':
                r=np.array([1.,1.,0.])/np.sqrt(2)
                arrows=[np.array([0.,0.,1.]),.65*r+np.array([0.,0.,.76]),-.65*r+np.array([0.,0.,.76])];arrows=[d/np.linalg.norm(d) for d in arrows];crossed=False
            else:arrows=[np.array([0.,0.,-1.])];crossed=True
            ax=fig.add_subplot(len(group['poses']),3,3*i+len(values)+1,projection='3d',computed_zorder=False)
            draw_panel(ax,pose_mesh,kept_mesh,removed_mesh,mesh,T,arrows,crossed)
            if i==0:ax.set_title(headers[len(values)],fontsize=14,pad=-4)
            if kind=='selected':ax.text2D(.02,.88,pose.replace('_',' '),transform=ax.transAxes,fontsize=13,color='#25334a')
            volume=material_volume(removed)*1e6;values.append(dict(kind=kind,removed_support_volume_cm3=volume,envelope_artifact=f'{pose}_{kind}_envelope.obj',removed_support_artifact=f'{pose}_{kind}_removed.obj'))
        records.append(dict(pose=pose,T_fixture_to_world=T.tolist(),selected_direction_fixture=direction.tolist(),selected_direction_world=(T[:3,:3]@direction).tolist(),sweeps=values))
        print('SWEEP ENVELOPE COMPLETE',group['id'],pose,flush=True)
    fig.subplots_adjust(left=.015,right=.985,top=.97,bottom=.055,wspace=0,hspace=0)
    fig.text(.5,.018,'Blue: retained support  |  Red: swept support  |  Translucent: whole-object sweep  |  Arcs: center hemisphere  |  Travel: 100 mm',ha='center',fontsize=12)
    target=out/'direction_space.png';fig.savefig(target,dpi=155,facecolor='white');plt.close(fig)
    image_data=base64.b64encode(target.read_bytes()).decode()
    document='<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(name+'/'+group['id'])+' · Step4.1</title><style>body{margin:24px;font:16px system-ui;color:#303840;line-height:1.7}img{width:100%;max-width:1500px;display:block;margin:auto}p{max-width:1500px;margin:16px auto}</style><p>'+html.escape(name+'/'+group['id'])+'：左列是 Step4.1 选中的相近退出方向及其扫掠；中列是向上半球所有候选方向的扫掠包络；右列是向下半球扫掠，红叉表示会进入原生地面。蓝色保留支撑，红色是当前扫掠与 Step3.3 支撑的实际交集，半透明部分是整个物体的扫掠包络，弧线显示物体中心的半球位移范围。显示 100 mm 退出范围。</p><p>目标：每个 pose 方向合法，彼此尽量接近，让联合扫掠少切共享支撑。当前初始化优化方向接近程度，还没有直接最小化切除量。半球包络采用多面体半球核的连续 Minkowski 和作示意，不用于完整夹具验收。</p><img alt="选定退出方向、向上半球和向下半球的实际扫掠" src="data:image/png;base64,'+image_data+'"></html>'
    (out/'direction_space.html').write_text(document)
    artifacts.update({'../direction_space.png':I.sha256(target),'../direction_space.html':I.sha256(out/'direction_space.html')})
    record=dict(complete=True,presentation_only=True,object=name,pose_set=group['id'],poses=group['poses'],support_source=str(support_path.relative_to(ROOT)),object_source=str(object_path.relative_to(ROOT)),uses_actual_step33_shape=True,display='Selected single-direction sweep versus upper/lower hemisphere candidate swept envelopes',states=records,normals_fixture=normals.tolist(),directions_fixture=directions.tolist(),initializer=metadata,common_direction_exists=computed_metadata['common_direction_status']!='no_nonzero_common_direction',common_direction_certificate=group.get('common_direction'),maximum_pair_angle_deg=float(np.degrees(np.arccos(np.clip(directions@directions.T,-1,1))).max()),display_length_m=length,hemisphere_guide='Center displacement hemisphere wire arcs, distinct from whole-object swept envelope',hemisphere_envelope_method='Nonconvex object Minkowski sum with clipped subdivision1 icosphere kernel; continuous polyhedral approximation to hemispherical displacement ball',red_material_method='Boolean intersection of original Step3.3 support with each shown sweep/envelope',selected_direction_method='continuous translation sweep',selected_is_not_entire_hemisphere=True,full_exit_certified=False,mechanics_rerun=False,saved_support_changed=False,seconds=time.monotonic()-began,provenance=provenance(inputs,[Path(__file__),HERE/'helper_func/optimization/initial_directions.py',Path(S.__file__).with_name('translation_sweep.py')]),artifacts=artifacts)
    save(out/'data/direction_space.json',record)
    return record


def main():
    p=argparse.ArgumentParser();p.add_argument('--object',default='B');p.add_argument('--sets',nargs='+');args=p.parse_args()
    groups=read_selected_pose_groups(args.object)
    if args.sets:groups=[g for g in groups if g['id'] in args.sets]
    for g in groups:
        result=render(args.object,g);print('DIRECTION SWEEPS',g['id'],result['seconds'],flush=True)

if __name__=='__main__':main()
