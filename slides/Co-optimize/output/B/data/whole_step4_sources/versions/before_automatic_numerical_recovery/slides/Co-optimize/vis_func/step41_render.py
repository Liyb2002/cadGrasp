"""Per-pose initialization cuts of the immutable Step3.3 support."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from co_common import *
from solid_render import arrow_mesh, depth_render
from exit_clearance import ExitClearance
from PIL import Image, ImageOps
from io import BytesIO
import argparse,time


def draw_pose(obj, kept, removed, T, arrow=False, sweep=None, direction=None):
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
    fig=plt.figure(figsize=(9,8),facecolor='white');ax=fig.add_subplot(111,projection='3d',computed_zorder=False)
    opaque_triangles=np.concatenate(triangles);opaque_colors=np.concatenate(colors)
    points=np.vstack([m.vertices for m in meshes])*1000
    if sweep is not None:
        world=sweep.copy();world.apply_transform(T)

        points=np.vstack([points,world.vertices*1000])
    if arrow:
        world_obj=transform_points(obj.vertices,T)*1000
        start=world_obj.mean(0);start[2]=world_obj[:,2].max()+5
        world_direction=np.array([0.,0.,1.]) if direction is None else T[:3,:3]@np.asarray(direction)
        mesh=arrow_mesh(start,world_direction,65);brightness=.55+.45*np.maximum(mesh.face_normals@light,0)
        opaque_triangles=np.concatenate([opaque_triangles,mesh.triangles]);opaque_colors=np.concatenate([opaque_colors,np.c_[brightness[:,None]*np.array(matplotlib.colors.to_rgb('#303840')),np.ones(len(mesh.faces))]])
        points=np.vstack([points,start,start+65*world_direction])
    center=(points.min(0)+points.max(0))/2;radius=float(np.ptp(points,axis=0).max())*(.72 if arrow else .53)
    for axis,value in zip('xyz',center):getattr(ax,'set_'+axis+'lim')(value-radius,value+radius)
    ax.set_box_aspect((1,1,1),zoom=.84 if sweep is not None else .88 if arrow else 1.08);ax.set_proj_type('ortho');ax.view_init(elev=elevation,azim=azimuth,roll=0);ax.set_axis_off()
    lo=points[:,:2].min(0)-8;hi=points[:,:2].max(0)+8
    boundary=np.array([[lo[0],lo[1],0],[hi[0],lo[1],0],[hi[0],hi[1],0],[lo[0],hi[1],0],[lo[0],lo[1],0]])
    ax.plot(*boundary.T,color='#c7c7c7',linewidth=.7,zorder=0)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1);fig.set_dpi(140);fig.canvas.draw()
    depth_render(ax,opaque_triangles,opaque_colors,world.triangles*1000 if sweep is not None else None,matplotlib.colors.to_rgba('#6ebad8',.084))
    buffer=BytesIO();fig.savefig(buffer,dpi=140,facecolor='white');plt.close(fig);buffer.seek(0)
    tile=Image.open(buffer).convert('RGB');ys,xs=np.where(np.any(np.asarray(tile)<245,axis=2))
    if len(xs):tile=tile.crop((max(0,xs.min()-20),max(0,ys.min()-20),min(tile.width,xs.max()+21),min(tile.height,ys.max()+21)))
    return tile


def render(name,group,directions=None,computed=None):
    began=time.monotonic()
    base=(HERE/'output'/name if name=='B' else HERE/'data/object_inputs'/name)/group['id'];out=base/'step4/step4.1';(out/'data').mkdir(parents=True,exist_ok=True)
    if directions is None and (out/'data/report.json').exists():
        saved=json.loads((out/'data/report.json').read_text())
        directions={row['pose']:row['direction_fixture'] for row in saved['state_results']}
    seed_path=base/'step3/step3.3/support_with_rings.obj';seedmesh=trimesh.load(seed_path,force='mesh',process=False);seed=computed['seed'] if computed is not None else S.solid(seedmesh)
    object_path=base/'step3/step3.1/registered_object.obj';obj=trimesh.load(object_path,force='mesh',process=False)
    policy=computed['policy'] if computed is not None else ExitClearance(obj)
    with np.load(initialized_contacts(base)) as z:allowed=z['allowed_faces']
    setup_seconds=time.monotonic()-began;own_construct_seconds=0.;own_sweep_seconds=0.;figure_seconds=0.
    padded_sweeps=[]
    tiles=[];sweep_tiles=[];rows=[];inputs=[seed_path,object_path];sweeps=[];transforms=[]
    for pose in group['poses']:
        task,T,mesh=state(name,pose);d=np.asarray(directions[pose]) if directions is not None else T[:3,:3].T@np.array([0.,0.,1.]);length=computed['full_length_m'] if computed is not None else max(.5,float((seedmesh.vertices@d).max()-(mesh.vertices@d).min())+.02)
        best=None
        for fan in [8,2,16,64]:
            sweep_start=time.monotonic()
            reused=computed is not None and fan==computed['fan']
            sweep=computed['nominal_sweeps'][pose] if reused else S.solid(S.swept_solid(mesh,length*d,fan_in=fan))
            padded=computed['padded_sweeps'][pose] if reused else policy.sweep(length*d,fan)
            own_sweep_seconds+=time.monotonic()-sweep_start
            construct_start=time.monotonic()
            construction=policy.construct(seed,[sweep],[padded],allowed[mesh.face_normals[allowed]@d<=1e-9],check_contacts=False)
            own_construct_seconds+=time.monotonic()-construct_start
            kept=construction['remaining'];removed=construction['removed'];diag=construction['diagnostics']
            error=diag['partition_error_m3'];overlap=diag['nominal_sweep_overlap_m3']
            score=max(error,overlap,diag['padded_sweep_overlap_outside_contact_cores_m3'])
            if diag['contact_check_performed'] and not diag['contact_area_preserved']:score=max(score,1.)
            if best is None or score<best[0]:best=(score,kept,removed,error,overlap,fan,sweep,padded)
            if score<1e-10:break
        _,kept,removed,error,overlap,fan,sweep,padded=best
        sweeps.append(sweep);padded_sweeps.append(padded);transforms.append(T)
        resolved=best[0]<1e-10
        keptmesh=S.unpack(kept);removedmesh=S.unpack(removed)
        removed_path=out/'data'/f'{pose}_own_removed.obj';D.export_exact_obj(removedmesh,removed_path)
        figure_start=time.monotonic()
        tiles.append(draw_pose(obj,keptmesh,removedmesh,T,arrow=True,direction=d))
        display=S.unpack(policy.sweep(.10*d,fan))
        sweep_tiles.append(draw_pose(obj,keptmesh,removedmesh,T,arrow=True,sweep=display,direction=d));inputs+=task.inputs
        figure_seconds+=time.monotonic()-figure_start
        rows.append(dict(pose=pose,direction_world=(T[:3,:3]@d).tolist(),direction_fixture=d.tolist(),full_length_m=length,own_removed_volume_cm3=material_volume(removed)*1e6,partition_error_m3=error,remaining_sweep_overlap_m3=overlap,geometry_partition_resolved=resolved,contact_check_performed=False,presentation_only=True,sweep_boolean_fan_in=fan,removed_artifact=f'{pose}_own_removed.obj'))
    final_start=time.monotonic()
    final_construction=computed['final_construction'] if computed is not None else policy.construct(seed,sweeps,padded_sweeps,allowed[np.max(obj.face_normals[allowed]@np.asarray([r['direction_fixture'] for r in rows]).T,axis=1)<=1e-9])
    final_construct_seconds=time.monotonic()-final_start
    final=final_construction['remaining'];finalmesh=S.unpack(final)
    final_overlap=final_construction['diagnostics']['nominal_sweep_overlap_m3'] if computed is not None else max(material_volume(final^sweep) for sweep in sweeps)
    final_path=out/'data/initialization_final_support.obj';D.export_exact_obj(finalmesh,final_path)
    figure_start=time.monotonic()
    final_tiles=[draw_pose(obj,finalmesh,trimesh.Trimesh(),T) for T in transforms]
    figure_seconds+=time.monotonic()-figure_start
    def sheet(panels,filename):
        cols=len(panels) if filename in ['exit_directions.png','exit_sweeps.png'] else (2 if len(panels)==4 else min(3,len(panels)));nr=(len(panels)+cols-1)//cols;size=760
        image=Image.new('RGB',(cols*size,nr*size),'white')
        for i,tile in enumerate(panels):
            tile=ImageOps.contain(tile,(size-30,size-30),Image.Resampling.LANCZOS);image.paste(tile,((i%cols)*size+(size-tile.width)//2,(i//cols)*size+(size-tile.height)//2))
        image.save(out/filename)
    sheet(tiles,'exit_directions.png');sheet(final_tiles,'final_result.png');sheet(sweep_tiles,'exit_sweeps.png')
    (out/'overview.png').unlink(missing_ok=True)
    from direction_space import render as render_direction_space
    render_direction_space(name,group,directions={r['pose']:r['direction_fixture'] for r in rows},metadata=computed.get('direction_metadata') if computed is not None else None)
    artifacts={f'../{file}':I.sha256(out/file) for file in ['exit_directions.png','final_result.png','exit_sweeps.png','direction_space.png']}
    artifacts['initialization_final_support.obj']=I.sha256(final_path)
    artifacts.update({r['removed_artifact']:I.sha256(out/'data'/r['removed_artifact']) for r in rows})
    save(out/'data/render.json',dict(timings=dict(setup_seconds=setup_seconds,own_construct_seconds=own_construct_seconds,own_sweep_seconds=own_sweep_seconds,figure_seconds=figure_seconds,final_construct_seconds=final_construct_seconds,total_seconds=time.monotonic()-began),exit_clearance=policy.metadata,clearance_diagnostics=final_construction['diagnostics'],complete=True,presentation_only=True,reuses_computed_final_construction=computed is not None,own_cut_contact_checks=False,own_cut_policy='Actual clearance/core Boolean construction with volume/overlap checks; final joint material retains complete contact checks',images=['exit_directions.png','final_result.png','exit_sweeps.png'],display_sweep_length_m=.10,sweep_opacity=.084,sweep_exit_direction_arrows=True,cut_uses_full_sweep=True,final_support_policy=policy.metadata['policy'],final_remaining_sweep_overlap_m3=final_overlap,final_geometry_resolved=final_construction['diagnostics']['geometry_resolved'],pose_set=group['id'],poses=group['poses'],states=rows,support_source='step3/step3.3/support_with_rings.obj',object_color='gray',support_color='blue',own_removed_color='red',red_policy='Only this pose recorded initialization direction continuous exit cuts; each panel starts from full immutable Step3.3 support',projection='orthographic',view_kind='isometric',view_elevation_deg=float(np.degrees(np.arctan(1/np.sqrt(2)))),view_azimuth_deg=-45.,text_or_labels=False,mechanics_rerun=False,step42_rerun=False,provenance=provenance(inputs,[Path(__file__),HERE/'helper_func/exit_clearance.py',Path(S.__file__).with_name('translation_sweep.py')]),artifacts=artifacts))
    I.check_report(out/'data/render.json')
    unresolved=[r['pose'] for r in rows if not r['geometry_partition_resolved']]
    (out/'README.md').write_text('# Step4.1 initialization\n\nexit_directions.png 和 exit_sweeps.png 按保存的 pose 顺序横向分格；final_result.png 显示同一共同切除结果。正交等轴测视角，灰色兔子、蓝色支撑。所有完整退出扫掠留每侧 1% 物体最大尺寸的余量（B：1.548 mm）；仅真实、运动相容的承载面下的无碰撞材料核保留接触。\n\n- exit_directions.png：原始 Step3.3 支撑；每格显示自己的初始化退出箭头，红色仅为自己退出时切掉的材料。\n- final_result.png：所有 pose 的完整退出 sweep 并集切除后的最终初始化支撑，每格展示同一份结果。\n- exit_sweeps.png：每格显示自己的透明连续 sweep（展示前 100 mm），与原始支撑相交的材料红色；实际切除使用至少 500 mm 的完整退出。\n\n当前图和切除数据在 data/render.json，当前最终初始化支撑在 data/initialization_final_support.obj。data/report.json、remaining_support.obj、removed_support.obj 由实际阶段求解单独记录；绘图入口本身不重跑力学或 Step4.2，不能将图片刷新视为新验收。direction_space.png 展示首个 pose 的四个合法方向与两个明显向下非法方向：切除支撑留空，无文字，模型放大 50%且总尺寸不变。direction_coverage.png 若已保存，仅为淡色方向轴半球叠加，无箭头，不是实际 sweep 或完整合法方向空间。\n'+('\n数值不一致的逐 pose 切除：'+', '.join(unresolved)+'。\n' if unresolved else '')+('\n共同切除仍有 sweep 重叠数值不一致，未接受为精确最终切除。\n' if final_overlap>=1e-10 else ''))
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
        panels.append(draw_pose(obj,kept,removed,T,arrow=True,sweep=display,direction=np.asarray(row['direction_fixture'])))
    cols=len(panels);size=760;image=Image.new('RGB',(cols*size,((len(panels)+cols-1)//cols)*size),'white')
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


def refresh_saved_direction_images(name, group):
    """Refresh arrow panels from saved cut geometry; do not rerun mechanics/cuts."""
    base=(HERE/'output'/name if name=='B' else HERE/'data/object_inputs'/name)/group['id'];out=base/'step4/step4.1'
    render_path=out/'data/render.json';record=json.loads(render_path.read_text())
    seed=S.solid(trimesh.load(base/'step3/step3.3/support_with_rings.obj',force='mesh',process=False))
    obj=trimesh.load(base/'step3/step3.1/registered_object.obj',force='mesh',process=False);policy=ExitClearance(obj)
    tiles=[];sweep_tiles=[]
    for row in record['states']:
        _,T,_=state(name,row['pose']);d=np.asarray(row['direction_fixture'])
        removed=trimesh.load(out/'data'/row['removed_artifact'],force='mesh',process=False);kept=S.unpack(seed-S.solid(removed))
        display=S.unpack(policy.sweep(record['display_sweep_length_m']*d,row['sweep_boolean_fan_in']))
        tiles.append(draw_pose(obj,kept,removed,T,arrow=True,direction=d))
        sweep_tiles.append(draw_pose(obj,kept,removed,T,arrow=True,sweep=display,direction=d))
    def sheet(panels,filename):
        cols=len(panels);size=760;nr=(len(panels)+cols-1)//cols
        image=Image.new('RGB',(cols*size,nr*size),'white')
        for k,tile in enumerate(panels):
            tile=ImageOps.contain(tile,(size-30,size-30),Image.Resampling.LANCZOS)
            image.paste(tile,((k%cols)*size+(size-tile.width)//2,(k//cols)*size+(size-tile.height)//2))
        image.save(out/filename)
    sheet(tiles,'exit_directions.png');sheet(sweep_tiles,'exit_sweeps.png')
    for filename in ['exit_directions.png','exit_sweeps.png']:record['artifacts']['../'+filename]=I.sha256(out/filename)
    record['arrow_direction_source']='saved per-pose direction_fixture transformed to native world'
    record['arrow_camera_padding']=.72
    save(render_path,record)
    mechanics_path=out/'data/report.json'
    if mechanics_path.exists():
        mechanics=json.loads(mechanics_path.read_text())
        for filename in ['exit_directions.png','exit_sweeps.png']:mechanics['artifacts']['../'+filename]=I.sha256(out/filename)
        save(mechanics_path,mechanics)
    return group['id']


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--object',default='B');parser.add_argument('--sweeps-only',action='store_true');parser.add_argument('--sets',nargs='+');parser.add_argument('--scope',choices=['all','legal','illegal'],default='all');args=parser.parse_args();groups=[]
    from codes.precompute_objects.dataset import read_selected_pose_groups
    name=args.object;groups=read_selected_pose_groups(name,args.scope)
    if args.sets:groups=[g for g in groups if g['id'] in args.sets or g['id'].removeprefix('illegal/') in args.sets]
    if args.sweeps_only:
        for group in groups:render_sweeps_only(name,group)
        print('SWEEP UPDATE COMPLETE',len(groups),flush=True)
        return
    results=[dict(id=g['id'],states=render(name,g)) for g in groups]
    save((HERE/'output'/name if name=='B' else HERE/'data/object_inputs'/name)/'data/step41_render_batch.json',dict(complete=True,images_per_set=3,total_images=3*len(results),pose_panels=sum(len(r['states']) for r in results),results=results,mechanics_rerun=False,step42_rerun=False))
    print('STEP4.1 TOTAL',len(results),'overviews',flush=True)

if __name__=='__main__':main()
