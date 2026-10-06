"""Per-pose initialization cuts of the immutable Step3.3 support."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from co_common import *
from exit_clearance import ExitClearance
from PIL import Image, ImageOps
from io import BytesIO
import argparse


def draw_pose(obj, kept, removed, T, arrow=False, sweep=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    elevation=float(np.degrees(np.arctan(1/np.sqrt(2))));azimuth=-45.
    light=np.array([1.,-1.,1.])+np.array([-.15,-.2,.65]);light/=np.linalg.norm(light)
    meshes=[];triangles=[];colors=[]
    for source,color in [(obj,'#a4a8ac'),(kept,'#319cd7'),(removed,'#ef4938')]:
        if not len(source.faces):continue
        m=source.copy();m.apply_transform(T);meshes.append(m)
        brightness=.58+.42*np.maximum(m.face_normals@light,0)
        colors.append(np.c_[brightness[:,None]*np.array(matplotlib.colors.to_rgb(color)),np.ones(len(m.faces))])
        triangles.append(m.triangles*1000)
    fig=plt.figure(figsize=(9,8),facecolor='white');ax=fig.add_subplot(111,projection='3d')
    ax.add_collection3d(Poly3DCollection(np.concatenate(triangles),facecolors=np.concatenate(colors),edgecolors='none',zsort='average'))
    points=np.vstack([m.vertices for m in meshes])*1000
    if sweep is not None:
        world=sweep.copy();world.apply_transform(T)
        ax.add_collection3d(Poly3DCollection(world.triangles*1000,facecolor='#6ebad8',edgecolors='none',alpha=.084,zsort='average'))
        points=np.vstack([points,world.vertices*1000])
    if arrow:
        world_obj=transform_points(obj.vertices,T)*1000
        start=world_obj.mean(0);start[2]=world_obj[:,2].max()+5
        ax.quiver(*start,0,0,1,length=65,color='#303840',arrow_length_ratio=.18,linewidth=2.5)
        points=np.vstack([points,start,start+[0,0,65]])
    center=(points.min(0)+points.max(0))/2;radius=float(np.ptp(points,axis=0).max())*.53
    for axis,value in zip('xyz',center):getattr(ax,'set_'+axis+'lim')(value-radius,value+radius)
    ax.set_box_aspect((1,1,1),zoom=.92 if sweep is not None else 1.08);ax.set_proj_type('ortho');ax.view_init(elev=elevation,azim=azimuth,roll=0);ax.set_axis_off()
    lo=points[:,:2].min(0)-8;hi=points[:,:2].max(0)+8
    boundary=np.array([[lo[0],lo[1],0],[hi[0],lo[1],0],[hi[0],hi[1],0],[lo[0],hi[1],0],[lo[0],lo[1],0]])
    ax.plot(*boundary.T,color='#c7c7c7',linewidth=.7)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1);buffer=BytesIO();fig.savefig(buffer,dpi=180,facecolor='white');plt.close(fig);buffer.seek(0)
    tile=Image.open(buffer).convert('RGB');ys,xs=np.where(np.any(np.asarray(tile)<245,axis=2))
    if len(xs):tile=tile.crop((max(0,xs.min()-20),max(0,ys.min()-20),min(tile.width,xs.max()+21),min(tile.height,ys.max()+21)))
    return tile


def render(name,group):
    base=(HERE/'output'/name if name=='B' else HERE/'data/object_inputs'/name)/group['id'];out=base/'step4/step4.1';(out/'data').mkdir(parents=True,exist_ok=True)
    seed_path=base/'step3/step3.3/support_with_rings.obj';seedmesh=trimesh.load(seed_path,force='mesh',process=False);seed=S.solid(seedmesh)
    object_path=base/'step3/step3.1/registered_object.obj';obj=trimesh.load(object_path,force='mesh',process=False)
    policy=ExitClearance(obj)
    with np.load(base/'step3/step3.2/data/contacts.npz') as z:allowed=z['allowed_faces']
    padded_sweeps=[]
    tiles=[];sweep_tiles=[];rows=[];inputs=[seed_path,object_path];sweeps=[];transforms=[]
    for pose in group['poses']:
        task,T,mesh=state(name,pose);d=T[:3,:3].T@np.array([0.,0.,1.]);length=max(.5,float((seedmesh.vertices@d).max()-(mesh.vertices@d).min())+.02)
        best=None
        for fan in [8,2,16,64]:
            sweep=S.solid(S.swept_solid(mesh,length*d,fan_in=fan))
            padded=policy.sweep(length*d,fan)
            construction=policy.construct(seed,[sweep],[padded],allowed[mesh.face_normals[allowed]@d<=1e-9])
            kept=construction['remaining'];removed=construction['removed'];diag=construction['diagnostics']
            error=diag['partition_error_m3'];overlap=diag['nominal_sweep_overlap_m3']
            score=max(error,overlap,diag['padded_sweep_overlap_outside_contact_cores_m3'])
            if not diag['contact_area_preserved']:score=max(score,1.)
            if best is None or score<best[0]:best=(score,kept,removed,error,overlap,fan,sweep)
            if score<1e-10:break
        _,kept,removed,error,overlap,fan,sweep=best
        sweeps.append(sweep);padded_sweeps.append(policy.sweep(length*d,fan));transforms.append(T)
        resolved=best[0]<1e-10
        keptmesh=S.unpack(kept);removedmesh=S.unpack(removed)
        removed_path=out/'data'/f'{pose}_own_removed.obj';D.export_exact_obj(removedmesh,removed_path)
        tiles.append(draw_pose(obj,keptmesh,removedmesh,T,arrow=True))
        display=S.unpack(policy.sweep(.10*d,fan))
        sweep_tiles.append(draw_pose(obj,keptmesh,removedmesh,T,arrow=True,sweep=display));inputs+=task.inputs
        rows.append(dict(pose=pose,direction_world=[0.,0.,1.],direction_fixture=d.tolist(),full_length_m=length,own_removed_volume_cm3=material_volume(removed)*1e6,partition_error_m3=error,remaining_sweep_overlap_m3=overlap,geometry_partition_resolved=resolved,sweep_boolean_fan_in=fan,removed_artifact=f'{pose}_own_removed.obj'))
    final_construction=policy.construct(seed,sweeps,padded_sweeps,allowed[np.max(obj.face_normals[allowed]@np.asarray([r['direction_fixture'] for r in rows]).T,axis=1)<=1e-9])
    final=final_construction['remaining'];finalmesh=S.unpack(final)
    final_overlap=max(material_volume(final^sweep) for sweep in sweeps)
    final_path=out/'data/initialization_final_support.obj';D.export_exact_obj(finalmesh,final_path)
    final_tiles=[draw_pose(obj,finalmesh,trimesh.Trimesh(),T) for T in transforms]
    def sheet(panels,filename):
        cols=2 if len(panels)==4 else min(3,len(panels));nr=(len(panels)+cols-1)//cols;size=760
        image=Image.new('RGB',(cols*size,nr*size),'white')
        for i,tile in enumerate(panels):
            tile=ImageOps.contain(tile,(size-30,size-30),Image.Resampling.LANCZOS);image.paste(tile,((i%cols)*size+(size-tile.width)//2,(i//cols)*size+(size-tile.height)//2))
        image.save(out/filename)
    sheet(tiles,'exit_directions.png');sheet(final_tiles,'final_result.png');sheet(sweep_tiles,'exit_sweeps.png')
    (out/'overview.png').unlink(missing_ok=True)
    artifacts={f'../{file}':I.sha256(out/file) for file in ['exit_directions.png','final_result.png','exit_sweeps.png']}
    artifacts['initialization_final_support.obj']=I.sha256(final_path)
    artifacts.update({r['removed_artifact']:I.sha256(out/'data'/r['removed_artifact']) for r in rows})
    save(out/'data/render.json',dict(exit_clearance=policy.metadata,clearance_diagnostics=final_construction['diagnostics'],complete=True,presentation_only=True,images=['exit_directions.png','final_result.png','exit_sweeps.png'],display_sweep_length_m=.10,sweep_opacity=.084,sweep_exit_direction_arrows=True,cut_uses_full_sweep=True,final_support_policy=policy.metadata['policy'],final_remaining_sweep_overlap_m3=final_overlap,final_geometry_resolved=final_construction['diagnostics']['geometry_resolved'],pose_set=group['id'],poses=group['poses'],states=rows,support_source='step3/step3.3/support_with_rings.obj',object_color='gray',support_color='blue',own_removed_color='red',red_policy='Only this pose native +z continuous exit cuts; each panel starts from full immutable Step3.3 support',projection='orthographic',view_kind='isometric',view_elevation_deg=float(np.degrees(np.arctan(1/np.sqrt(2)))),view_azimuth_deg=-45.,text_or_labels=False,mechanics_rerun=False,step42_rerun=False,provenance=provenance(inputs,[Path(__file__),HERE/'helper_func/exit_clearance.py',Path(S.__file__).with_name('translation_sweep.py')]),artifacts=artifacts))
    I.check_report(out/'data/render.json')
    unresolved=[r['pose'] for r in rows if not r['geometry_partition_resolved']]
    (out/'README.md').write_text('# Step4.1 initialization\n\n三张图均按保存的 pose 顺序分格，正交等轴测视角，灰色兔子、蓝色支撑。所有完整退出扫掠留每侧 1% 物体最大尺寸的余量（B：1.548 mm）；仅真实、运动相容的承载面下的无碰撞材料核保留接触。\n\n- exit_directions.png：原始 Step3.3 支撑；每格显示自己的 +Z 退出箭头，红色仅为自己退出时切掉的材料。\n- final_result.png：所有 pose 的完整退出 sweep 并集切除后的最终初始化支撑，每格展示同一份结果。\n- exit_sweeps.png：每格显示自己的透明连续 sweep（展示前 100 mm），与原始支撑相交的材料红色；实际切除使用至少 500 mm 的完整退出。\n\n当前图和切除数据在 data/render.json，当前最终初始化支撑在 data/initialization_final_support.obj。历史 data/report.json、remaining_support.obj、removed_support.obj 和 Step4.2 未重跑，不能用旧力学结果接受新凸包支撑。\n'+('\n数值不一致的逐 pose 切除：'+', '.join(unresolved)+'。\n' if unresolved else '')+('\n共同切除仍有 sweep 重叠数值不一致，未接受为精确最终切除。\n' if final_overlap>=1e-10 else ''))
    print('STEP4.1 OWN CUT',group['id'],len(rows),'poses',flush=True)
    return rows


def render_sweeps_only(name,group):
    out=(HERE/'output'/name if name=='B' else HERE/'data/object_inputs'/name)/group['id']/'step4/step4.1'
    report_path=out/'data/render.json';record=json.loads(report_path.read_text())
    if not record.get('exit_clearance'):return render(name,group)
    base=out.parents[1];seed_path=base/'step3/step3.3/support_with_rings.obj'
    obj=trimesh.load(base/'step3/step3.1/registered_object.obj',force='mesh',process=False)
    seed=S.solid(trimesh.load(seed_path,force='mesh',process=False));panels=[];inputs=[]
    for row in record['states']:
        pose=row['pose'];_,T,mesh=state(name,pose)
        removed_path=out/'data'/row['removed_artifact'];inputs.append(removed_path)
        removed=trimesh.load(removed_path,force='mesh',process=False)
        kept=S.unpack(seed-S.solid(removed))
        policy=ExitClearance(obj)
        display=S.unpack(policy.sweep(record['display_sweep_length_m']*np.array(row['direction_fixture']),row['sweep_boolean_fan_in']))
        panels.append(draw_pose(obj,kept,removed,T,arrow=True,sweep=display))
    cols=2 if len(panels)==4 else min(3,len(panels));size=760;image=Image.new('RGB',(cols*size,((len(panels)+cols-1)//cols)*size),'white')
    for i,tile in enumerate(panels):
        tile=ImageOps.contain(tile,(size-30,size-30),Image.Resampling.LANCZOS)
        image.paste(tile,((i%cols)*size+(size-tile.width)//2,(i//cols)*size+(size-tile.height)//2))
    image.save(out/'exit_sweeps.png')
    record.update(sweep_opacity=.084,sweep_exit_direction_arrows=True)
    record['artifacts']['../exit_sweeps.png']=I.sha256(out/'exit_sweeps.png')
    inputs += [ROOT/path for path in record['provenance']['inputs']]
    record['provenance']=provenance(inputs,[Path(__file__),HERE/'helper_func/exit_clearance.py',Path(S.__file__).with_name('translation_sweep.py')])
    save(report_path,record);I.check_report(report_path)
    readme=out/'README.md';text=readme.read_text().replace('每格显示自己的透明连续 sweep（展示前 100 mm）','每格显示自己的透明连续 sweep（展示前 100 mm，透明度 alpha=0.084，比此前淡 30%）和退出方向箭头');readme.write_text(text)
    print('STEP4.1 SWEEP',group['id'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--sweeps-only',action='store_true');parser.add_argument('--sets',nargs='+');parser.add_argument('--scope',choices=['all','legal','illegal'],default='all');args=parser.parse_args();groups=[]
    if args.scope in ('all','legal'):groups+=json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets']
    if args.scope in ('all','illegal'):groups+=[dict(g,id='illegal/'+g['id']) for g in json.loads((ROOT/'objects/B/illegal_pose_sets.json').read_text())['sets']]
    if args.sets:groups=[g for g in groups if g['id'] in args.sets or g['id'].removeprefix('illegal/') in args.sets]
    if args.sweeps_only:
        for group in groups:render_sweeps_only('B',group)
        print('SWEEP UPDATE COMPLETE',len(groups),flush=True)
        return
    results=[dict(id=g['id'],states=render('B',g)) for g in groups]
    save(HERE/'output/B/data/step41_render_batch.json',dict(complete=True,images_per_set=3,total_images=3*len(results),pose_panels=sum(len(r['states']) for r in results),results=results,mechanics_rerun=False,step42_rerun=False))
    print('STEP4.1 TOTAL',len(results),'overviews',flush=True)

if __name__=='__main__':main()
