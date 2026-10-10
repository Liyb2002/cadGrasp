"""Refresh saved Step4.1 figures without base; never rerun direction/force search."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
from co_common import *
from step41_render import draw_pose
from direction_space import render as direction_space
from direction_coverage import render as direction_coverage
from PIL import Image,ImageOps
import argparse,time,shutil


def render(name,group):
    began=time.monotonic();root=HERE/'output'/name if name=='B' else HERE/'data/object_inputs'/name
    base=root/group['id'];out=base/'step4/step4.1';data=out/'data'
    record=json.loads((data/'render.json').read_text());report=json.loads((data/'report.json').read_text())
    archive=data/'history/before_base_deferred';archive.mkdir(parents=True,exist_ok=True)
    images=['exit_directions.png','exit_sweeps.png','final_result.png','direction_space.png','direction_coverage.png']
    for file in images+['README.md']:
        if (out/file).exists() and not (archive/file).exists():shutil.copy2(out/file,archive/file)
    for file in ['render.json','report.json','direction_space.json','direction_coverage.json']:
        if (data/file).exists() and not (archive/file).exists():shutil.copy2(data/file,archive/file)
    seed_path=base/'step3/step3.2/wrapped_support.obj';seedmesh=trimesh.load(seed_path,force='mesh',process=False);seed=S.solid(seedmesh)
    objpath=base/'step3/step3.1/registered_object.obj';obj=trimesh.load(objpath,force='mesh',process=False)
    finalpath=data/'initialization_final_support.obj'
    if not finalpath.exists():finalpath=out/'remaining_support.obj'
    final=S.unpack(seed^S.solid(trimesh.load(finalpath,force='mesh',process=False)))
    panels=[];sweep_panels=[];final_panels=[];inputs=[seed_path,objpath,finalpath];rows=[]
    saved={r['pose']:r for r in report['state_results']}
    render_rows={r['pose']:r for r in record.get('states',[])}
    for pose in group['poses']:
        task,T,_=state(name,pose);row=render_rows.get(pose,saved[pose]);d=np.asarray(row['direction_fixture'])
        cutpath=data/row.get('removed_artifact',f'{pose}_own_removed.obj')
        if not cutpath.exists():cutpath=out/'removed_support.obj'
        cut=S.solid(trimesh.load(cutpath,force='mesh',process=False));removed=S.unpack(seed^cut);kept=S.unpack(seed-cut)
        display=S.swept_solid(obj,.10*d)
        panels.append(draw_pose(obj,kept,removed,T,arrow=True,direction=d))
        sweep_panels.append(draw_pose(obj,kept,removed,T,arrow=True,direction=d,sweep=display))
        final_panels.append(draw_pose(obj,final,trimesh.Trimesh(),T))
        inputs += [cutpath]+task.inputs;rows.append(dict(pose=pose,direction_fixture=d.tolist(),saved_cut_source=str(cutpath.relative_to(ROOT))))
    def sheet(tiles,file):
        cols=len(tiles) if file!='final_result.png' else (2 if len(tiles)==4 else min(3,len(tiles)));size=760
        canvas=Image.new('RGB',(cols*size,((len(tiles)+cols-1)//cols)*size),'white')
        for k,tile in enumerate(tiles):
            tile=ImageOps.contain(tile,(size-30,size-30),Image.Resampling.LANCZOS);canvas.paste(tile,((k%cols)*size+(size-tile.width)//2,(k//cols)*size+(size-tile.height)//2))
        canvas.save(out/file)
    sheet(panels,images[0]);sheet(sweep_panels,images[1]);sheet(final_panels,images[2])
    direction_space(name,group,support_source=seed_path);direction_coverage(name,group)
    artifacts={'../'+f:I.sha256(out/f) for f in images}
    save(data/'base_deferred_render.json',dict(complete=True,presentation_only=True,base_deferred=True,support_source=str(seed_path.relative_to(ROOT)),algorithm_changed=False,mechanics_rerun=False,directions_rerun=False,saved_support_changed=False,method='Clip saved per-pose cut and final material to Step3.2 wrap for presentation; six example sweeps recomputed on wrap only',states=rows,seconds=time.monotonic()-began,provenance=provenance(inputs,[Path(__file__),HERE/'vis_func/step41_render.py',HERE/'vis_func/direction_space.py',HERE/'vis_func/direction_coverage.py']),artifacts=artifacts))
    for file in ['render.json','report.json']:
        r=json.loads((data/file).read_text());r['artifacts'].update(artifacts);r['presentation_refresh']='base_deferred_render.json; saved numerical geometry and checks remain unchanged';save(data/file,r)
    (out/'README.md').write_text('# Step4.1：base 最后考虑\n\n当前所有图片均不显示 Step3.3 接地环，以 Step3.2 wrapped_support.obj 为支撑显示范围；方向与保存的算法结果不变。exit direction / sweep 按 pose 顺序横向排布。direction space 无文字、四个合法和两个向下非法例子、被切除支撑留空；coverage 是淡色半球叠加、无箭头。\n\n绘图记录为 data/base_deferred_render.json。本轮仅重画：未优化方向、未重跑受力、未改变保存支撑模型或 Step4.2。历史 report.json 的数值诊断仍属于原有含 base 模型，不能用于验收无 base 图片。原图保存在 data/history/before_base_deferred/。\n')
    print('BASE DEFERRED',name,group['id'],round(time.monotonic()-began,2),flush=True)
    return dict(object=name,id=group['id'],complete=True,seconds=time.monotonic()-began,images=images)


def main():
    p=argparse.ArgumentParser();p.add_argument('--object',default='B');p.add_argument('--sets',nargs='+');p.add_argument('--jobs',type=int,default=2);args=p.parse_args()
    groups=read_selected_pose_groups(args.object)
    if args.sets:groups=[g for g in groups if g['id'] in args.sets]
    from concurrent.futures import ProcessPoolExecutor,as_completed
    rows=[]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        pending={pool.submit(render,args.object,g):g for g in groups}
        for future in as_completed(pending):
            g=pending[future]
            try:rows.append(future.result())
            except Exception as e:rows.append(dict(object=args.object,id=g['id'],complete=False,error=str(e)));print('RENDER ERROR',g['id'],repr(e),flush=True)
            root=HERE/'output'/args.object if args.object=='B' else HERE/'data/object_inputs'/args.object
            save(root/'data/base_deferred_render_batch.json',dict(complete=len(rows)==len(groups),all_succeeded=len(rows)==len(groups) and all(r['complete'] for r in rows),requested=len(groups),finished=len(rows),results=rows))
    if not all(r['complete'] for r in rows):raise SystemExit(1)

if __name__=='__main__':main()
