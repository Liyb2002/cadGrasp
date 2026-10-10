"""Step3.3 support in native poses, with legal upward and crossed downward arrows."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
from co_common import *
from optimization.initial_directions import initialize_close_directions
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import argparse,base64,html


def render(name,group,directions=None,metadata=None):
    root=HERE/'output'/name if name=='B' else HERE/'data/object_inputs'/name
    base=root/group['id'];out=base/'step4/step4.1';(out/'data').mkdir(parents=True,exist_ok=True)
    support_path=base/'step3/step3.3/support_with_rings.obj'
    object_path=base/'step3/step3.1/registered_object.obj'
    support=trimesh.load(support_path,force='mesh',process=False)
    obj=trimesh.load(object_path,force='mesh',process=False)
    states=[state(name,p) for p in group['poses']]
    normals=np.array([T[:3,:3].T@np.array([0.,0.,1.]) for task,T,mesh in states])
    initialized,computed_metadata=initialize_close_directions(normals)
    if directions is None:directions=initialized
    elif isinstance(directions,dict):directions=np.array([directions[p] for p in group['poses']])
    else:directions=np.asarray(directions)
    metadata=computed_metadata if metadata is None else metadata
    elevation=float(np.degrees(np.arctan(1/np.sqrt(2))));azimuth=-45.
    elev,az=np.radians([elevation,azimuth])
    camera=np.array([np.cos(elev)*np.cos(az),np.cos(elev)*np.sin(az),np.sin(elev)])
    right=np.array([-np.sin(az),np.cos(az),0.]);up=np.cross(camera,right)
    light=camera+np.array([-.15,-.2,.65]);light/=np.linalg.norm(light)
    allowed=np.array([[0.,0.,1.],.72*right+np.array([0.,0.,.69]),-.72*right+np.array([0.,0.,.69])])
    allowed/=np.linalg.norm(allowed,axis=1)[:,None]
    prohibited=np.array([0.,0.,-1.])
    n=len(group['poses']);cols=2 if n<=4 else 3;rows=int(np.ceil(n/cols))
    fig=plt.figure(figsize=(6*cols,5.6*rows),facecolor='white');records=[]
    for i,(pose,(task,T,_)) in enumerate(zip(group['poses'],states)):
        ax=fig.add_subplot(rows,cols,i+1,projection='3d',computed_zorder=False)
        world_obj=obj.copy();world_obj.apply_transform(T)
        world_support=support.copy();world_support.apply_transform(T)
        np.testing.assert_allclose(world_obj.vertices,task.domain.mesh.vertices,atol=1e-10,rtol=0)
        triangles=[];colors=[]
        for mesh,color in [(world_obj,'#a4a8ac'),(world_support,'#319cd7')]:
            brightness=.58+.42*np.maximum(mesh.face_normals@light,0)
            colors.append(np.c_[brightness[:,None]*np.array(matplotlib.colors.to_rgb(color)),np.ones(len(mesh.faces))])
            triangles.append(mesh.triangles*1000)
        ax.add_collection3d(Poly3DCollection(np.concatenate(triangles),facecolors=np.concatenate(colors),edgecolors='none',zsort='average',zorder=1))
        points=np.vstack([world_obj.vertices,world_support.vertices])*1000
        span=float(np.ptp(points,axis=0).max());center=world_obj.bounds.mean(axis=0)*1000
        anchor=center+right*(.64*span);anchor[2]=points[:,2].max()-.12*span
        length=.44*span
        for d in allowed:ax.quiver(*anchor,*d,length=length,color='#20986e',linewidth=2.25,arrow_length_ratio=.17,zorder=10)
        ax.quiver(*anchor,*prohibited,length=length,color='#dc443f',linewidth=2.5,arrow_length_ratio=.18,zorder=11)
        cross_center=anchor+prohibited*(.63*length);cross_size=.055*span
        for axis in [right+up,right-up]:
            axis/=np.linalg.norm(axis);ends=np.array([cross_center-cross_size*axis,cross_center+cross_size*axis]);ax.plot(*ends.T,color='#dc443f',linewidth=3.7,zorder=12)
        points=np.vstack([points,anchor,anchor+length*allowed,anchor+length*prohibited])
        center=(points.min(0)+points.max(0))/2;radius=np.ptp(points,axis=0).max()*.53
        for axis,value in zip('xyz',center):getattr(ax,'set_'+axis+'lim')(value-radius,value+radius)
        ax.set_box_aspect((1,1,1),zoom=1.16);ax.set_proj_type('ortho');ax.view_init(elev=elevation,azim=azimuth);ax.set_axis_off()
        floor=transform_points(support.vertices,T)*1000;lo=floor[:,:2].min(0)-6;hi=floor[:,:2].max(0)+6
        boundary=np.array([[lo[0],lo[1],0],[hi[0],lo[1],0],[hi[0],hi[1],0],[lo[0],hi[1],0],[lo[0],lo[1],0]])
        ax.plot(*boundary.T,color='#cbd0d6',linewidth=.7,zorder=0)
        ax.set_title(pose.replace('_',' '),fontsize=17,pad=-10,color='#303840')
        records.append(dict(pose=pose,T_fixture_to_world=T.tolist(),allowed_arrow_directions_world=allowed.tolist(),allowed_arrow_directions_fixture=(allowed@T[:3,:3]).tolist(),crossed_direction_world=prohibited.tolist(),crossed_direction_fixture=(T[:3,:3].T@prohibited).tolist()))
    fig.subplots_adjust(left=.015,right=.985,top=.965,bottom=.025,wspace=0,hspace=.01)
    target=out/'direction_space.png';fig.savefig(target,dpi=180,facecolor='white');plt.close(fig)
    image_data=base64.b64encode(target.read_bytes()).decode()
    document='<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(name+'/'+group['id'])+' · 退出方向</title><style>body{margin:24px;font:16px system-ui;color:#303840}img{width:100%;max-width:1200px;display:block;margin:auto}p{max-width:1200px;margin:16px auto}</style><p>'+html.escape(name+'/'+group['id'])+'：Step3.3 实际支撑与物体。绿色箭头表示向上半球内的退出方向；向下红色箭头打叉。</p><img alt="原生 pose 中的退出方向示意" src="data:image/png;base64,'+image_data+'"></html>'
    (out/'direction_space.html').write_text(document)
    pair_angles=np.degrees(np.arccos(np.clip(directions@directions.T,-1,1)))
    record=dict(complete=True,presentation_only=True,object=name,pose_set=group['id'],poses=group['poses'],support_source=str(support_path.relative_to(ROOT)),object_source=str(object_path.relative_to(ROOT)),uses_actual_step33_shape=True,display='Native pose views with upward hemisphere example arrows and crossed downward arrow; no sphere surfaces',arrows_are_hemisphere_examples=True,states=records,normals_fixture=normals.tolist(),directions_fixture=directions.tolist(),initializer=metadata,common_direction_exists=computed_metadata['common_direction_status']!='no_nonzero_common_direction',common_direction_certificate=group.get('common_direction'),maximum_pair_angle_deg=float(pair_angles.max()),mechanics_rerun=False,geometry_changed=False,provenance=provenance([support_path,object_path,ROOT/'objects'/name/'selected_pose_sets.json']+[ROOT/'objects'/name/'poses'/p/'setup.npz' for p in group['poses']],[Path(__file__),HERE/'helper_func/optimization/initial_directions.py']),artifacts={'../direction_space.png':I.sha256(target),'../direction_space.html':I.sha256(out/'direction_space.html')})
    save(out/'data/direction_space.json',record)
    return record


def main():
    p=argparse.ArgumentParser();p.add_argument('--object',default='B');p.add_argument('--sets',nargs='+');args=p.parse_args()
    groups=read_selected_pose_groups(args.object)
    if args.sets:groups=[g for g in groups if g['id'] in args.sets]
    for g in groups:
        result=render(args.object,g);print('DIRECTION ARROWS',g['id'],result['maximum_pair_angle_deg'],flush=True)

if __name__=='__main__':main()
