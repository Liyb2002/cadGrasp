"""Present saved two-tool layouts in Step4.1 style without rerunning optimization."""
import sys,argparse,json,shutil,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
from co_common import HERE,ROOT,I,np,trimesh,S,save,state,transform_points
from step41_render import draw_pose
from PIL import Image,ImageOps
from concurrent.futures import ProcessPoolExecutor,as_completed


def sheet(tiles,path,columns):
    size=760;canvas=Image.new('RGB',(columns*size,((len(tiles)+columns-1)//columns)*size),'white')
    for i,tile in enumerate(tiles):
        thumb=ImageOps.contain(tile,(size-30,size-30),Image.Resampling.LANCZOS)
        canvas.paste(thumb,((i%columns)*size+(size-thumb.width)//2,(i//columns)*size+(size-thumb.height)//2))
    canvas.save(path)


def render(source_root,row):
    began=time.monotonic();group=row['pose_set'];source=Path(source_root).resolve()/group/'data'
    report=json.loads((source/'report.json').read_text());poses=report['poses']
    if not report['force_exit_passed']:raise ValueError('Saved result is not force/exit feasible')
    base=HERE/'output/B'/group;out=base/'step4/step4.2';out.mkdir(parents=True,exist_ok=True)
    archive=out/'data/history/before_two_tool_final_images';archive.mkdir(parents=True,exist_ok=True)
    for name in ['README.md','final_result.png','exit_directions.png','exit_sweeps.png']:
        if (out/name).exists() and not (archive/name).exists():shutil.copy2(out/name,archive/name)
    data=out/'data/two_tool_final';data.mkdir(parents=True,exist_ok=True)
    for name in ['report.json','layout.npz','remaining_support.obj','packing_certificate.json']:
        if (source/name).exists():shutil.copy2(source/name,data/name)
    objpath=base/'step3/step3.1/registered_object.obj'
    obj=trimesh.load(objpath,force='mesh',process=False);support=trimesh.load(source/'remaining_support.obj',force='mesh',process=False)
    with np.load(source/'layout.npz') as z:
        offsets=z['offsets_m'].copy();directions=z['directions'].copy();transforms=z['T_fixture_to_world'].copy()
    np.testing.assert_allclose(offsets,report['offsets_m'],atol=1e-12)
    np.testing.assert_allclose(directions,report['directions'],atol=1e-12)
    initial=json.loads((base/'step4/step4.1/data/report.json').read_text());length=float(initial['initialization']['full_length_m'])
    panels=[];sweep_panels=[];records=[]
    for i,pose in enumerate(poses):
        task,T,_=state('B',pose);placed=obj.copy();placed.apply_translation(offsets[i]);native=transforms[i]
        np.testing.assert_allclose(transform_points(placed.vertices,native),transform_points(obj.vertices,T),atol=1e-10)
        # Saved support remains intact: no crop, seed intersection or simplification.
        tile=draw_pose(placed,support,trimesh.Trimesh(),native,arrow=True,direction=directions[i])
        display=S.swept_solid(obj,min(.10,length)*directions[i]);display.apply_translation(offsets[i])
        sweep_tile=draw_pose(placed,support,trimesh.Trimesh(),native,arrow=True,direction=directions[i],sweep=display)
        panels.append(tile);sweep_panels.append(sweep_tile)
        tile.save(out/(pose+'_final.png'));sweep_tile.save(out/(pose+'_exit.png'))
        records.append(dict(pose=pose,offset_fixture_m=offsets[i].tolist(),direction_fixture=directions[i].tolist(),direction_world=(native[:3,:3]@directions[i]).tolist(),T_fixture_to_world=native.tolist()))
        print('POSE RENDERED',group,pose,flush=True)
    sheet(panels,out/'final_result.png',3);sheet(panels,out/'exit_directions.png',len(poses));sheet(sweep_panels,out/'exit_sweeps.png',len(poses))
    images=['final_result.png','exit_directions.png','exit_sweeps.png']+[p+s for p in poses for s in ['_final.png','_exit.png']]
    record=dict(complete=True,presentation_only=True,optimization_rerun=False,mechanics_rerun=False,pose_set=group,
        source_report=str((source/'report.json').relative_to(ROOT)),source_support_sha256=I.sha256(source/'remaining_support.obj'),
        source_layout_sha256=I.sha256(source/'layout.npz'),support_display='entire exact saved final shared support; translated per-pose native frame; no geometry clipping',
        pose_order=poses,states=records,full_exit_length_m=length,display_sweep_length_m=min(.10,length),
        object_color='#a4a8ac',support_color='#319cd7',sweep_opacity=.084,projection='orthographic',view_elevation_degrees=float(np.degrees(np.arctan(1/np.sqrt(2)))),view_azimuth_degrees=-45,
        style_source='vis_func/step41_render.py:draw_pose',text_or_labels=False,
        original_numeric_report_preserved='data/report.json belongs to the previous run; current rendered result is data/two_tool_final/report.json',
        artifacts={name:I.sha256(out/name) for name in images},seconds=time.monotonic()-began)
    save(data/'render.json',record)
    text='# Step4.2：最终两工具布局\n\n本页图片来自已完成的七组两工具实验，显示保存的完整最终蓝色支撑、灰色物体和各 pose 的最终退出方向。当前图片使用真实平移布局，并按该 pose 的原始安装坐标展示同一支撑。\n\n'
    text+='- [最终支撑与全部 pose](final_result.png)：3×2 排列，带退出方向箭头。\n- [退出方向](exit_directions.png)：按保存 pose 顺序横向排列。\n- [退出扫掠路径](exit_sweeps.png)：透明显示实际直线路径的前100 mm，完整验收路径为'+str(round(length*1000))+' mm，与 Step4.1 展示规则一致。\n\n'
    text+='面板顺序：'+', '.join(poses)+'。\n\n'
    text+='| pose | 最终布局 | 退出路径 |\n|---|---|---|\n'
    for p in poses:text+='| '+p+' | [图]('+p+'_final.png) | [图]('+p+'_exit.png) |\n'
    text+='\n当前数值来源：[report](data/two_tool_final/report.json)、[原始完整支撑](data/two_tool_final/remaining_support.obj)、[平移和方向](data/two_tool_final/layout.npz)。此前求解报告 data/report.json 与历史图保留，不代表本轮新展示结果。这里只绘图，未重新求解或改变实验支撑。连通、安装接地和强度的验收状态保持。\n'
    (out/'README.md').write_text(text)
    return dict(pose_set=group,complete=True,images=images,seconds=record['seconds'])


def render_full_paths(source_root,row):
    group=row['pose_set'];source=Path(source_root).resolve()/group/'data'
    base=HERE/'output/B'/group;out=base/'step4/step4.2';data=out/'data/two_tool_final'
    record=json.loads((data/'render.json').read_text());poses=record['pose_order'];length=record['full_exit_length_m']
    support=trimesh.load(source/'remaining_support.obj',force='mesh',process=False)
    obj=trimesh.load(base/'step3/step3.1/registered_object.obj',force='mesh',process=False)
    with np.load(source/'layout.npz') as z:
        offsets=z['offsets_m'];directions=z['directions'];transforms=z['T_fixture_to_world']
    tiles=[];images=[]
    for i,pose in enumerate(poses):
        placed=obj.copy();placed.apply_translation(offsets[i])
        display=S.swept_solid(obj,length*directions[i]);display.apply_translation(offsets[i])
        tile=draw_pose(placed,support,trimesh.Trimesh(),transforms[i],arrow=True,direction=directions[i],sweep=display)
        image=pose+'_full_exit.png';tile.save(out/image);images.append(image);tiles.append(tile)
        print('FULL EXIT RENDERED',group,pose,flush=True)
    sheet(tiles,out/'full_exit_paths.png',3);sheet(tiles,out/'full_exit_sweeps.png',len(poses));images+=['full_exit_paths.png','full_exit_sweeps.png']
    record['full_path_images']={name:I.sha256(out/name) for name in images};record['full_path_display_length_m']=length;save(data/'render.json',record)
    p=out/'README.md';text=p.read_text();text+='\n[完整退出路径（3×2）](full_exit_paths.png) · [完整退出路径（横向）](full_exit_sweeps.png)：显示完整 '+str(round(length*1000))+' mm 扫掠；每个 pose 的完整路径另存为 pose_*_full_exit.png。\n';p.write_text(text)
    return dict(pose_set=group,complete=True,images=images)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--sets',nargs='+');parser.add_argument('--jobs',type=int,default=2);parser.add_argument('--full-paths-only',action='store_true');args=parser.parse_args()
    rows=json.loads((args.source/'batch.json').read_text())['results']
    if args.sets:rows=[r for r in rows if r['pose_set'] in args.sets]
    results=[]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        pending={pool.submit(render_full_paths if args.full_paths_only else render,args.source,r):r for r in rows}
        for future in as_completed(pending):
            result=future.result();results.append(result);print('RENDER COMPLETE',result['pose_set'],flush=True)
    save(HERE/'output/B/data'/('two_tool_full_exit_render_batch.json' if args.full_paths_only else 'two_tool_final_render_batch.json'),dict(complete=True,presentation_only=True,source=str(args.source),results=results,total_pose_panels=(sum(len(r['images'])-2 for r in results) if args.full_paths_only else sum(len(r['images'])-3 for r in results)//2)))
    index=HERE/'step4.2/results_B.md';text='# Step4.2：七组最终布局\n\n蓝色为保存的完整最终支撑，灰色为物体，透明体为退出路径前100 mm。每组按保存顺序展示六个 pose；图片仅为展示，验收状态保持。\n\n'
    for row in rows:
        key=row['pose_set'];relative='../output/B/'+key+'/step4/step4.2/'
        text+='## '+key+'\n\n'+('[零平移方向微调]' if row['small_step_search_passed'] else '[计算分离保底；最大平移 '+str(round(row['maximum_translation_m']*1000,1))+' mm]')+'\n\n[分 pose 大图及数据]('+relative+'README.md) · [横向退出路径图]('+relative+'exit_sweeps.png)\n\n![]('+relative+'final_result.png)\n\n'
    if args.full_paths_only:
        for row in rows:
            key=row['pose_set'];text+='[完整退出路径：'+key+'](../output/B/'+key+'/step4/step4.2/full_exit_paths.png)\n\n'
    index.write_text(text)

if __name__=='__main__':main()
