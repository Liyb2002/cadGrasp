"""Text-free exit sheet: one initial object and one saved-direction arrow per pose."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from co_common import *
import argparse

def render(group):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from PIL import Image
    out=HERE/'output/B'/group['id']/'step4/step4.2';source=out/'data/report.json';report=I.check_report(source)
    support=trimesh.load(out/'remaining_support.obj',force='mesh',process=False)
    cols=min(3,len(group['poses']));nr=(len(group['poses'])+cols-1)//cols
    image_path=out/'exit_motion.png'
    with Image.open(image_path) as old:image_size=old.size
    dpi=170
    fig=plt.figure(figsize=(image_size[0]/dpi,image_size[1]/dpi),dpi=dpi,facecolor='white');records=[];inputs=[source,out/'remaining_support.obj'];panels=[]
    for row in report['state_results']:
        task,T,mesh=state('B',row['pose']);native=np.asarray(row['direction_world'],dtype=float)
        assert abs(np.linalg.norm(native)-1)<1e-8
        world_support=support.copy();world_support.apply_transform(T);obj=task.domain.mesh.copy()
        vertices=np.vstack([world_support.vertices,obj.vertices])*1000
        extent=np.ptp(vertices,axis=0).max();origin=obj.vertices.mean(0)*1000
        # Short illustrative arrow uses the accepted direction, not an invented motion.
        projected=(obj.vertices*1000-origin)@native
        start=origin+native*(projected.max()+extent*.025);length=extent*.32
        end=start+native*length
        bounds=np.vstack([vertices,start,end]);panels.append((world_support,obj,bounds,native,start,length))
        inputs+=task.inputs
        records.append(dict(pose=row['pose'],T_fixture_to_world=T.tolist(),direction_world=native.tolist(),display_displacements_m=[0.],arrow_origin_world_m=(start/1000).tolist(),arrow_length_m=length/1000,full_path_length_m=report['full_length_m'],object_instances=1,arrow_count=1))
    radius=max(np.ptp(p[2],axis=0).max() for p in panels)*.52
    for i,(world_support,obj,vertices,native,start,length) in enumerate(panels):
        ax=fig.add_subplot(nr,cols,i+1,projection='3d');center=(vertices.min(0)+vertices.max(0))/2
        xy=np.vstack([world_support.vertices[:,:2],obj.vertices[:,:2]])*1000;lo=xy.min(0)-5;hi=xy.max(0)+5
        corners=np.array([[lo[0],lo[1],0],[hi[0],lo[1],0],[hi[0],hi[1],0],[lo[0],hi[1],0],[lo[0],lo[1],0]])
        ax.plot(corners[:,0],corners[:,1],corners[:,2],color='#c8c8c8',linewidth=.7)
        light=np.array([-.3,-.5,1.]);light/=np.linalg.norm(light);shade=.64+.28*np.maximum(0,world_support.face_normals@light)
        colors=np.c_[np.repeat(shade[:,None],3,axis=1),np.ones(len(shade))]
        ax.add_collection3d(Poly3DCollection(world_support.triangles*1000,facecolors=colors,edgecolors='#606060',linewidths=.08,alpha=1.))
        ax.add_collection3d(Poly3DCollection(obj.triangles*1000,facecolor='#4d9bc9',edgecolor='none',alpha=.25))
        ax.quiver(*start,*(native*length),color='#15567d',linewidth=3,arrow_length_ratio=.24,normalize=False)
        azimuth=np.rad2deg(np.arctan2(native[1],native[0]))+90 if np.linalg.norm(native[:2])>1e-6 else -55
        ax.view_init(elev=23,azim=azimuth)
        for axis,c in zip('xyz',center):getattr(ax,'set_'+axis+'lim')(c-radius,c+radius)
        ax.set_box_aspect((1,1,1),zoom=1.25);ax.set_axis_off()
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1,wspace=0,hspace=0)
    fig.savefig(image_path,dpi=dpi);plt.close(fig)
    with Image.open(image_path) as new:assert new.size==image_size
    save(out/'data/exit_motion_render.json',dict(complete=True,pose_set=group['id'],panel_order=group['poses'],states=records,image_size_px=list(image_size),object_alpha=.25,support_alpha=1.,text_or_axes=False,sweep_rendered=False,geometry_changed=False,framing='initial object and complete support, enlarged without changing canvas',provenance=provenance(inputs,[HERE/'vis_func/render_exit_motion.py']),artifacts={'../exit_motion.png':I.sha256(image_path)}))
    p=out/'README.md';text=p.read_text();line='`exit_motion.png`：无文字的退出示意，各 pose 按上述表格顺序排列。灰色支撑为不透明实体，蓝色半透明物体仅显示初始位置；一个箭头表示报告中保存的退出方向，箭头长度仅用于展示，实际路径仍使用报告中的完整长度。'
    lines=text.splitlines();lines=[line if '`exit_motion.png`' in x else x for x in lines]
    if not any('`exit_motion.png`' in x for x in lines):lines.append(line)
    p.write_text('\n'.join(lines)+'\n')
    print('EXIT MOTION',group['id'],flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--sets',nargs='+');args=p.parse_args();groups=json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets']
    for group in groups:
        if not args.sets or group['id'] in args.sets:render(group)
if __name__=='__main__':main()
