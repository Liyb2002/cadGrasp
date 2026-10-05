"""Register and wrap every saved B pose set; force gate precedes carving."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json,time,traceback,contextlib,multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
from co_common import *
import step31,step32

def image(mesh,triangles,sources,out,kind,count,status=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig=plt.figure(figsize=(9,8));ax=fig.add_subplot(111,projection='3d')
    ax.add_collection3d(Poly3DCollection(mesh.triangles*1000,facecolor='#77b7dd',edgecolor='none',alpha=.25))
    if kind=='register':
        ax.add_collection3d(Poly3DCollection(mesh.triangles[sources]*1000,facecolor='#f3a74c',edgecolor='none',alpha=.8))
        title=f'Step3.1: {count} registered poses\nObjects coincide; orange = union of working faces'
        points=mesh.vertices*1000
    else:
        ax.add_collection3d(Poly3DCollection(triangles.triangles*1000,facecolor='#969696',edgecolor='none',alpha=.65))
        title=f'Step3.2: full non-working-surface wrap\nForce gate: {status.upper()}; exit channels are not carved'
        points=np.vstack([mesh.vertices,triangles.vertices])*1000 if len(triangles.vertices) else mesh.vertices*1000
    center=(points.min(0)+points.max(0))/2;r=float(np.ptp(points,axis=0).max())*.55
    for axis,c in zip(('x','y','z'),center):getattr(ax,'set_'+axis+'lim')(c-r,c+r)
    ax.set_box_aspect((1,1,1));ax.view_init(elev=25,azim=-55);ax.set_axis_off();ax.set_title(title)
    fig.savefig(out,dpi=150,bbox_inches='tight');plt.close(fig)

def run_case(args):
    name,group,thickness=args;out=HERE/'output'/name/group['id']/'step3';out.mkdir(parents=True,exist_ok=True)
    with (out/'data').joinpath('pipeline.log').open('w',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        began=time.monotonic()
        try:
            states,mesh,registration=step31.run(name,group,out/'step3.1')
            work=np.unique(np.concatenate([task.domain.work_ids for task,T,m in states.values()]))
            image(mesh,None,work,out/'step3.1/overview.png','register',len(states))
            wrapped,contacts,sources,r=step32.run(name,group,states,mesh,out/'step3.2',thickness)
            image(mesh,wrapped,None,out/'step3.2/overview.png','wrap',len(states),r['status'])
            row=dict(id=group['id'],poses=group['poses'],status=r['status'],passed=r['passed'],step4_ready=r['step4_ready'],common_nonworking_faces=r['common_nonworking_face_count'],contact_area_cm2=r['actual_contact_area_m2']*1e4,material_volume_cm3=r['material_volume_cm3'],seconds=time.monotonic()-began)
            lines=[f"# {group['id']}：全表面包裹初始化",'', 'Step3.1 将物体注册到原始 mesh 坐标：各 pose 的物体在共同支撑视图中重合，原生世界摆放、地面与载荷保持原样。', '', 'Step3.2 包裹所有状态共同允许的非工作表面。任何一个状态的工作面都不包裹；从实际包裹体边界提取真实接触区域，重算各状态的法向力与力矩生成元。', '', f"结果：**{r['status'].upper()}**；{r['common_nonworking_face_count']} 个共同非工作源面，实际接触面积 {r['actual_contact_area_m2']*1e4:.2f} cm²。",'', '| 状态 | 原始载荷通过数 | 力／力矩结果 | 工作面 |','|---|---:|---|---|']
            for s in r['state_results']:lines.append(f"| {s['pose']} | {s['covered']}/{s['load_count']} | {s['status']} | {'保持自由' if s['working_surface_clear'] else '几何待修复'} |")
            lines+=['', 'PASS 表示这些真实接触通过原始全部载荷及 shared no-uplift 模型，进入后续 Step4 通道雕刻；不是完整夹具成功。包裹体尚未挖出退出路径，也可能穿过某些状态的地面，后续切除必须重新检查剩余接触承载。', '', 'FAIL 必须附带对全部共同非工作表面反力生成元有效的分离证据：在固定注册和当前力学模型下，只删减接触不能补救。求解或包裹几何未决标为 UNRESOLVED，不冒充物理失败。', '', '`step3.1/overview.png`：重合物体和工作面；`step3.2/overview.png`：真实包裹实体；`step3.2/wrapped_support.obj`：原始 mesh 坐标中的实体。内部接触、载荷结果和来源记录在各阶段的 `data/`。']
            (out/'README.md').write_text('\n'.join(lines)+'\n')
            record=dict(complete=True,**row,object=name,full_fixture_accepted=False,provenance=provenance([out/'step3.1/data/report.json',out/'step3.2/data/report.json'],code_sources()),artifacts={'../step3.1/overview.png':I.sha256(out/'step3.1/overview.png'),'../step3.2/overview.png':I.sha256(out/'step3.2/overview.png')})
            save(out/'data/report.json',record);I.check_report(out/'data/report.json')
            return row
        except Exception as error:
            traceback.print_exc();row=dict(id=group['id'],poses=group['poses'],status='unresolved',passed=False,step4_ready=False,error=f'{type(error).__name__}: {error}',seconds=time.monotonic()-began)
            save(out/'data/failure.json',dict(complete=True,**row));(out/'README.md').write_text(f"# {group['id']}\n\nUNRESOLVED：{row['error']}\n\n此结果不是物理失败证明。\n")
            return row

def main():
    p=argparse.ArgumentParser();p.add_argument('object',nargs='?',default='B');p.add_argument('--sets',nargs='+');p.add_argument('--jobs',type=int,default=2);p.add_argument('--wrap-thickness-mm',type=float,default=5.);args=p.parse_args()
    if args.wrap_thickness_mm<=0:raise ValueError('Wrap thickness must be positive')
    source=ROOT/'objects'/args.object/'pose_sets.json';manifest=json.loads(source.read_text());groups=manifest['sets']
    if args.sets:
        names={g['id'] for g in groups}
        if not set(args.sets)<=names:raise ValueError('Unknown saved pose set')
        groups=[g for g in groups if g['id'] in args.sets]
    began=time.monotonic();rows=[]
    for g in groups:(HERE/'output'/args.object/g['id']/'step3/data').mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.jobs,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(run_case,(args.object,g,args.wrap_thickness_mm/1000)) for g in groups]
        for future in as_completed(futures):
            r=future.result();rows.append(r);print('SET COMPLETE',r,flush=True)
    order={g['id']:i for i,g in enumerate(groups)};rows.sort(key=lambda r:order[r['id']])
    batch=dict(complete=True,object=args.object,pose_sets=len(rows),pose_instances=sum(len(r['poses']) for r in rows),passed_sets=sum(r['status']=='pass' for r in rows),failed_sets=sum(r['status']=='fail' for r in rows),unresolved_sets=sum(r['status']=='unresolved' for r in rows),results=rows,seconds=time.monotonic()-began,provenance=provenance([source],code_sources()))
    save(HERE/'output'/args.object/'data/batch.json',batch)
    lines=['# B：重合物体与全表面包裹','', '当前输出已按新流程重新生成；旧单 pose greedy 支撑和旧 Step4 诊断已删除。', '', '| Pose set | Step3 结果 | 实际接触面积 cm² | Step4 |','|---|---|---:|---|']
    for r in rows:lines.append(f"| [{r['id']}]({r['id']}/step3/README.md) | {r['status'].upper()} | {r.get('contact_area_cm2',0):.2f} | {'可进入通道雕刻' if r['step4_ready'] else '停止'} |")
    lines+=['', f"共 {batch['pose_sets']} 组、{batch['pose_instances']} 个 pose 实例：{batch['passed_sets']} PASS，{batch['failed_sets']} FAIL，{batch['unresolved_sets']} UNRESOLVED。", '', 'PASS 只表示包裹接触的力／力矩阶段通过，尚未得到可装卸的完整共享夹具。内部数据在 `data/`，公开结果查看各组图片、模型和说明。']
    (HERE/'output'/args.object/'README.md').write_text('\n'.join(lines)+'\n')
    print('TOTAL',batch['passed_sets'],'PASS',batch['failed_sets'],'FAIL',batch['unresolved_sets'],'UNRESOLVED',flush=True)
    if batch['unresolved_sets']:raise SystemExit(2)
if __name__=='__main__':main()
