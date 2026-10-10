"""Production Step4: current whole initialization, recovery and material descent."""
import _bootstrap
import argparse,contextlib,multiprocessing,shutil,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from co_common import *
from run_all import saved_groups,output_root
from whole_search.fast_search import FastModel,FastReuseSearch
from whole_search.model import Model,Layout
from whole_search.search import failed_count
from whole_search.exact_worker import write_result,read_result
from whole_search.common import result_mesh,tangent_frame,legal_direction
from whole_step4_render import render_initialization,render_search

SCHEMA='whole_step4_v1'


def load_layout(path):
    with np.load(path) as z:
        return Layout(z['placements'].copy(),z['directions'].copy(),z['hosts'].copy(),tuple(map(int,z['active'])))


def prepare_case(name,group):
    base=output_root(name)/group['id'];stage=base/'step4'
    current=stage/'step4.1/data/report.json'
    if stage.exists() and (not current.exists() or json.loads(current.read_text()).get('schema')!=SCHEMA):
        backup=output_root(name)/'_history/before_whole_step4_20261008'/group['id']/'step4'
        if not backup.exists():backup.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(stage,backup)
        for path in stage.rglob('*'):
            if path.is_file():
                archived=backup/path.relative_to(stage)
                if not archived.exists() or I.sha256(path)!=I.sha256(archived):raise RuntimeError('Old Step4 differs from archive: '+str(path))
        shutil.rmtree(stage)
    (stage/'step4.1/data').mkdir(parents=True,exist_ok=True)
    return base


def stage_record(out,report,group,stage):
    report.update(schema=SCHEMA,pose_set=group['id'],stage=stage,strategy='whole',
        initialized_all_poses_together=True,fixture_placement_count_is_not_cost=True,
        objective='all original loads, full work volumes and legal complete exits; then solid material volume',
        installed_floor_acceptance_run=False,full_fixture_accepted=False)
    report['provenance']['code'].update(I.hashes([Path(__file__),HERE/'vis_func/whole_step4_render.py',HERE/'helper_func/work_access.py']))
    report['artifacts']={p.name:I.sha256(p) for p in out.iterdir()
        if p.is_file() and p.name not in ['report.json','README.md','run.log']}
    for filename in ['data/initial_geometry.npz','data/initial_geometry.json','data/render.json']:
        if (out/filename).exists():report['artifacts'][filename]=I.sha256(out/filename)
    save(out/'report.json',report)
    data=report.copy();data['artifacts']={'../'+k:v for k,v in report['artifacts'].items()}
    save(out/'data/report.json',data);I.check_report(out/'data/report.json')
    return report


def initialize(name,group,base):
    source=base/'step3/step3.1/data/report.json';I.check_report(source)
    model=FastModel(group['poses'],name,initialization_report=source)
    layout,metadata=model.initial();out=base/'step4/step4.1'
    trials=[layout]
    for axis in range(2):
        for degrees in [.25,-.25,1.,-1.]:
            trial=layout.copy()
            for k in layout.active:
                trial.directions[k]=legal_direction(layout.directions[k]+np.tan(np.radians(degrees))*tangent_frame(layout.directions[k])[:,axis],model.floor_normal(trial,k))
            trials.append(trial)
    failures=[]
    for index,trial in enumerate(trials):
        try:actual=model.exact(trial);layout=trial;break
        except (RuntimeError,ValueError,AssertionError) as error:
            failures.append(dict(trial=index,error=str(error)))
    else:raise RuntimeError('All Step4.1 numerical construction trials unresolved: '+str(failures))
    metadata.update(numerical_geometry_recovery=index>0,recovery_trial=index,construction_failures=failures)
    write_result(actual,out/'data/initial_geometry')
    report=Model.save(model,actual,out,dict(initialization=metadata,
        status='initialized',passed=True,construction_completed=True,
        force_and_path_initialization_gate=failed_count(actual)==0,
        initial_counts={model.poses[k]:int(actual['masks'][k].sum()) for k in layout.active},
        step4_ready=True,optimized=False,diagnostics_are_acceptance_gates=False))
    render_initialization(model,actual,out)
    report=stage_record(out,report,group,'step4.1')
    (out/'README.md').write_text('# Step4.1：整组退出初始化\n\n'
        '全部 pose 保留原注册布局；在物体坐标中找共同/相近合法退出方向，切除新Step3.1支撑中的完整退出路径。'
        '原物体朝向、工作区、32,768载荷/pose和不上抬模型不变。\n\n'
        '[退出方向](exit_directions.png) · [退出sweep](exit_sweeps.png) · [共享支撑](final_result.png)\n\n'
        '沿用原Step4.1的正交等轴测画法：灰色物体、蓝色真实支撑、红色为当前pose自己的切除、黑色退出箭头。'
        '透明sweep仅显示前100mm；实际切除使用完整、保证末端分离的路径，并保留接触核外1%净空。'
        '图内没有文字。`support.obj`是所有路径共同切除后的实际支撑。\n\n'
        f"材料 {report['volume_cm3']:.3f}cm³；当前原载荷诊断：{report['counts']}。"
        '初始化构造成功不等于承载通过，失败也保留给whole搜索。尚未做整件支撑接地、连通或强度验收。\n')
    return model,layout,metadata,actual,report


class WholeSearch(FastReuseSearch):
    def shortlist(self,proposals,targets):
        self.current_guidance_targets=targets
        return super().shortlist(proposals,targets)

    def score(self,result,anchor=None):
        base=super().score(result,anchor)
        if failed_count(result) and getattr(self,'current_guidance_targets',None):
            guidance=self.model.proxy(result['layout'],self.current_guidance_targets)
            return base[:2]+(guidance['loss'],guidance['sum_loss'])+base[2:]
        return base[:2]+(0.,0.)+base[2:]

    def run_from_initialization(self,layout,metadata):
        began=time.monotonic();current=self.initialize_exact(layout,'step4.1_all_registered')
        if getattr(self,'initial_actual_passed',False):self.initial_feasible_result=current
        initial_counts=current['counts'];self.baseline(current)
        current=self.solve_frontier(current,anchor=None)
        current=self.restore_reuse(current)
        current=self.polish_volume(current,getattr(self,'volume_rounds',2))
        self.last_sampled_result=current
        return self.save(current,'whole',began,metadata,initial_counts=initial_counts,
            fixture_placement_count_is_not_cost=True,all_pose_juxtapose_disabled=True,
            separated_fallback_used=False)


def optimize(model,layout,metadata,actual,group,base,options):
    out=base/'step4/step4.2';(out/'data').mkdir(parents=True,exist_ok=True)
    model.checkpoint_dir=out/'exact_states'
    model.worker_program=HERE/'helper_func/whole_search/exact_worker.py'
    # Initial geometry was already computed, not recomputed by search.
    model.exact_cache[layout.key()]=actual
    search=WholeSearch(model,out,options['iterations'],options['finalists'],options['branch_rounds'],options['seed'])
    search.volume_rounds=options['volume_rounds'];search.screen_budget=options['screen_budget']
    search.final_candidates=3;search.capture_process=True
    search.initial_actual_passed=failed_count(actual)==0
    search.initial_feasible_volume_cm3=actual['volume_cm3'] if search.initial_actual_passed else None
    try:
        report=search.run_from_initialization(layout,metadata)
    except RuntimeError:
        # Save the actual sampled construction and its drawings as a diagnosis;
        # never turn an optimistic or failed export into a passing result.
        traceback.print_exc()
        if hasattr(search,'last_sampled_result'):
            render_search(model,out,final_result=None)
        raise
    render_search(model,out,final_result=report)
    report.update(status='pass' if report['force_exit_work_passed'] else 'unresolved',
        passed=report['force_exit_work_passed'],step41_volume_cm3=actual['volume_cm3'],
        step41_counts={model.poses[k]:int(actual['masks'][k].sum()) for k in layout.active})
    report=stage_record(out,report,group,'step4.2')
    lines=['# Step4.2：whole优化','',
        '[过程图](process.png) · [最终各pose](final_result.png)','',
        f"状态：{report['status']}；材料 {actual['volume_cm3']:.3f} → {report['volume_cm3']:.3f} cm³。",
        f"转动支撑复用 {report['rotating_reuse_pose_count']} 个pose；Juxtapose {report['juxtaposed_pose_count']} 个pose。",'',
        '全部pose从初始化就参与约束。候选使用接触/材料增删更新；不同角度、平移步幅与host采样。'
        'Direction优先修复，停滞任务或明确阻挡者尝试Juxtapose，随后Translation/Direction局部调整。'
        '未整组可行时按全组未满足原载荷数改善；整组可行后只接受保持可行的材料下降。', '',
        'process.png每格是实际保存的候选布局的名义三角网格；候选受力是采样接触模型，过程几何不冒充最终净空验收。'
        '最后一格及final_result.png使用最终真实接触核/1%净空验收的support.obj。'
        '每个pose按其host摆放同一件支撑；物体保持规定世界朝向及高度。工作面为原始橙色区域。', '',
        '没有远距离分离保底；没有为设计平移打通滑动轨迹。完整工作禁区始终参与候选和真实网格计算。'
        '原始32,768载荷/pose最终按真实接触复核；整件接地、连通和强度验收仍未运行。', '',
        '| 步 | 操作 | 原载荷未满足数 | 材料估计cm³ | 具体变化 |','|---|---|---:|---:|---|']
    for row in json.loads((out/'process.json').read_text()):
        detail='；'.join(f"{r['pose']}: host {r.get('old_host','')}→{r.get('host','')}, 方向{r.get('direction_change_degrees',0):.3f}°, 平移{r.get('translation_change_mm',0):.3f}mm" for r in row['changes'])
        lines.append(f"| {row['index']} | {row['phase']} | {row['failed_load_count']} | {row['estimated_volume_cm3']:.3f} | {detail} |")
    lines+=['','完整候选选择、步幅和全部pose计数见 `trace.json`；布局见 `process_states/`。']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    return report


def run_case(arguments):
    name,group,options=arguments;base=prepare_case(name,group);began=time.monotonic()
    log_path=base/'step4/run.log'
    with log_path.open('w',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            if options.get('stage')=='step4.2':
                out=base/'step4/step4.1';record=I.check_report(out/'data/report.json')
                if record.get('schema')!=SCHEMA:raise RuntimeError('Run the current whole Step4.1 first')
                model=FastModel(group['poses'],name,initialization_report=base/'step3/step3.1/data/report.json')
                layout=load_layout(out/'layout.npz');metadata=record['initialization']
                actual=read_result(out/'data/initial_geometry',1);model.exact_cache[layout.key()]=actual
                initial=record
            else:model,layout,metadata,actual,initial=initialize(name,group,base)
            final=optimize(model,layout,metadata,actual,group,base,options) if options.get('stage')!='step4.1' else None
            row=dict(id=group['id'],poses=group['poses'],status=final['status'] if final else 'initialized',
                passed=final['passed'] if final else True,step41_force_passed=initial['force_and_path_initialization_gate'],
                step41_volume_cm3=initial['volume_cm3'],seconds=time.monotonic()-began)
            if final:row.update(final_volume_cm3=final['volume_cm3'],rotating_poses=final['rotating_reuse_pose_count'],
                juxtaposed_poses=final['juxtaposed_pose_count'],final_counts=final['counts'],sampled_evaluations=final['sample_evaluations'])
            save(base/'step4/data/report.json',dict(complete=True,schema=SCHEMA,**row,full_fixture_accepted=False))
            (base/'step4/data/failure.json').unlink(missing_ok=True)
            return row
        except Exception as error:
            traceback.print_exc();row=dict(id=group['id'],poses=group['poses'],status='unresolved',passed=False,
                error=f'{type(error).__name__}: {error}',seconds=time.monotonic()-began)
            if 'initial' in locals():row.update(step41_force_passed=initial['force_and_path_initialization_gate'],step41_volume_cm3=initial['volume_cm3'])
            save(base/'step4/data/failure.json',dict(complete=True,**row));return row


def publish(name,rows,stage):
    import html
    root=output_root(name);cards=[]
    lines=['# B：Step4 whole优化','',
        '沿用新Step3初始化和原B集合。所有pose始终参与约束；候选增删更新，保存真实三角支撑。'
        '先恢复原载荷承载，再保持全组可行减少实体材料。没有远距离分离保底。','',
        '[图片浏览](step4_index.html) · [新初始化](README.md)','',
        '| 集合 | 状态 | Step4.1承载 | 初始材料cm³ | 最终材料cm³ | 转动/Juxtapose | 图片 |',
        '|---|---|---|---:|---:|---|---|']
    for row in rows:
        folder=row['id']+'/step4';name_html=html.escape(row['id']);links=f'[初始化]({folder}/step4.1/final_result.png)'
        if (root/folder/'step4.2/final_result.png').exists():links+=f' · [过程]({folder}/step4.2/process.png) · [最终]({folder}/step4.2/final_result.png)'
        lines.append(f"| {row['id']} | {row['status']} | {row.get('step41_force_passed','—')} | {row.get('step41_volume_cm3',0):.3f} | {row.get('final_volume_cm3',0):.3f} | {row.get('rotating_poses','—')}/{row.get('juxtaposed_poses','—')} | {links} |")
        pictures=[]
        for file,label in [('step4.1/exit_directions.png','Step4.1退出方向'),('step4.1/exit_sweeps.png','Step4.1退出sweep'),('step4.1/final_result.png','Step4.1共享支撑'),('step4.2/process.png','Step4.2过程'),('step4.2/final_result.png','Step4.2最终复用')]:
            if (root/folder/file).exists():pictures.append(f'<article><h3>{label}</h3><a href="{folder}/{file}"><img loading="lazy" src="{folder}/{file}"></a></article>')
        cards.append(f'<section><h2>{name_html}</h2><p>{html.escape(str(row))}</p>'+''.join(pictures)+'</section>')
    (root/'step4_results.md').write_text('\n'.join(lines)+'\n')
    (root/'step4_index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>B Step4 whole</title><style>body{max-width:1500px;margin:25px auto;padding:20px;font-family:system-ui;color:#253441}img{max-width:100%}article{border:1px solid #dde4e9;padding:10px;margin:15px 0}p{overflow-wrap:anywhere}</style><h1>B Step4 whole优化</h1><p>实体支撑体积优先；原载荷、完整工作禁区、合法退出约束。整件接地、连通和强度尚未验收。</p>'+''.join(cards)+'</html>')


def main(stage='both'):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object',nargs='?',default='B');parser.add_argument('--sets',nargs='+')
    parser.add_argument('--jobs',type=int,default=2);parser.add_argument('--iterations',type=int,default=8)
    parser.add_argument('--finalists',type=int,default=3);parser.add_argument('--branch-rounds',type=int,default=3)
    parser.add_argument('--volume-rounds',type=int,default=3);parser.add_argument('--screen-budget',type=int,default=96)
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();groups=saved_groups(args.object)
    if args.sets:groups=[g for g in groups if g['id'] in args.sets or g['id'].removeprefix('illegal/') in args.sets]
    if not groups:parser.error('No existing initialized group matches')
    root=output_root(args.object);options=vars(args)|{'stage':stage};rows=[];pending=[];began=time.monotonic()
    for group in groups:
        report=root/group['id']/'step4/data/report.json'
        if args.resume and report.exists() and json.loads(report.read_text()).get('schema')==SCHEMA:
            row=json.loads(report.read_text())
            if row.get('passed') and (stage=='step4.1' or 'final_volume_cm3' in row):rows.append(row);continue
        pending.append(group)
    with ProcessPoolExecutor(max_workers=args.jobs,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1) as pool:
        for future in as_completed([pool.submit(run_case,(args.object,g,options)) for g in pending]):
            row=future.result();rows.append(row)
            save(root/'data/whole_step4_progress.json',rows)
            print('WHOLE STEP4',row['id'],row['status'],round(row['seconds'],2),row.get('error',''),flush=True)
            publish(args.object,rows,stage)
    order={g['id']:i for i,g in enumerate(groups)};rows.sort(key=lambda row:order[row['id']])
    batch=dict(complete=True,schema=SCHEMA,object=args.object,stage=stage,sets=len(rows),
        passed=sum(r['passed'] for r in rows),unresolved=sum(not r['passed'] for r in rows),results=rows,
        options=options,seconds=time.monotonic()-began,full_fixture_accepted=False)
    save(root/'data/whole_step4_batch.json',batch);publish(args.object,rows,stage)
    print('TOTAL',batch['passed'],'/',batch['sets'],'passed',batch['unresolved'],'unresolved',flush=True)
    if batch['unresolved']:raise SystemExit(2)


def main_step41():main('step4.1')


def main_step42():main('step4.2')


if __name__=='__main__':main()
