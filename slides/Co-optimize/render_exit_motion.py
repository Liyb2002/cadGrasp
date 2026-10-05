"""Text-free motion sheet: opaque saved support and translucent moving objects."""
from co_common import *
import argparse

def render(group):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    out=HERE/'output/B'/group['id']/'step4/step4.2';source=out/'data/report.json';report=I.check_report(source)
    support=trimesh.load(out/'remaining_support.obj',force='mesh',process=False)
    cols=min(3,len(group['poses']));nr=(len(group['poses'])+cols-1)//cols
    fig=plt.figure(figsize=(6*cols,5.5*nr),facecolor='white');records=[];inputs=[source,out/'remaining_support.obj'];panels=[]
    for row in report['state_results']:
        task,T,mesh=state('B',row['pose']);direction=np.array(row['direction_fixture']);native=np.array(row['direction_world']);world_support=support.copy();world_support.apply_transform(T)
        distance=max(.001,float((support.vertices@direction).max()-(mesh.vertices@direction).min())+.012)
        assert distance<=report['full_length_m']
        times=np.array([0.,.5,1.])*distance;objects=[]
        for t in times:
            obj=task.domain.mesh.copy();obj.apply_translation(native*t);objects.append(obj)
        all_vertices=np.vstack([world_support.vertices]+[obj.vertices for obj in objects])*1000
        panels.append((world_support,objects,all_vertices,native));inputs+=task.inputs
        records.append(dict(pose=row['pose'],T_fixture_to_world=T.tolist(),direction_world=native.tolist(),display_displacements_m=times.tolist(),full_path_length_m=report['full_length_m'],last_display_object_fully_separated=True))
    radius=max(np.ptp(p[2],axis=0).max() for p in panels)*.55
    for i,(world_support,objects,vertices,native) in enumerate(panels):
        ax=fig.add_subplot(nr,cols,i+1,projection='3d');center=(vertices.min(0)+vertices.max(0))/2
        # Floor reference is a thin outline, without labels or axes.
        xy=np.vstack([world_support.vertices[:,:2],objects[0].vertices[:,:2],objects[-1].vertices[:,:2]])*1000;lo=xy.min(0)-10;hi=xy.max(0)+10
        corners=np.array([[lo[0],lo[1],0],[hi[0],lo[1],0],[hi[0],hi[1],0],[lo[0],hi[1],0],[lo[0],lo[1],0]])
        ax.plot(corners[:,0],corners[:,1],corners[:,2],color='#c8c8c8',linewidth=.7)
        light=np.array([-.3,-.5,1.]);light/=np.linalg.norm(light);shade=.64+.28*np.maximum(0,world_support.face_normals@light)
        colors=np.c_[np.repeat(shade[:,None],3,axis=1),np.ones(len(shade))]
        ax.add_collection3d(Poly3DCollection(world_support.triangles*1000,facecolors=colors,edgecolors='#606060',linewidths=.08,alpha=1.))
        for obj in objects:
            ax.add_collection3d(Poly3DCollection(obj.triangles*1000,facecolor='#4d9bc9',edgecolor='none',alpha=.25))
        azimuth=np.rad2deg(np.arctan2(native[1],native[0]))+90 if np.linalg.norm(native[:2])>1e-6 else -55
        ax.view_init(elev=23,azim=azimuth)
        for axis,c in zip('xyz',center):getattr(ax,'set_'+axis+'lim')(c-radius,c+radius)
        ax.set_box_aspect((1,1,1));ax.set_axis_off()
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1,wspace=0,hspace=0)
    fig.savefig(out/'exit_motion.png',dpi=170,bbox_inches='tight',pad_inches=.02);plt.close(fig)
    save(out/'data/exit_motion_render.json',dict(complete=True,pose_set=group['id'],panel_order=group['poses'],states=records,object_alpha=.25,support_alpha=1.,text_or_axes=False,sweep_rendered=False,geometry_changed=False,provenance=provenance(inputs,[HERE/'render_exit_motion.py']),artifacts={'../exit_motion.png':I.sha256(out/'exit_motion.png')}))
    p=out/'README.md';text=p.read_text();line='\n`exit_motion.png`：无文字的退出示意，各 pose 按上述表格顺序排列。灰色支撑为不透明实体，蓝色半透明物体显示起点、中途和完全脱离三个位置；示意距离按当前支撑计算，实际路径仍使用报告中的完整长度。\n'
    if '`exit_motion.png`' not in text:p.write_text(text+line)
    print('EXIT MOTION',group['id'],flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--sets',nargs='+');args=p.parse_args();groups=json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets']
    for group in groups:
        if not args.sets or group['id'] in args.sets:render(group)
if __name__=='__main__':main()
