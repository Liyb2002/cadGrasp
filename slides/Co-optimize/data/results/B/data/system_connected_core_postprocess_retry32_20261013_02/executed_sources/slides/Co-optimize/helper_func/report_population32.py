"""Read saved Step4 results, publish per-set PNGs and a truthful result table."""
from pathlib import Path
import argparse, csv, hashlib, json, os, shutil, sys
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import ImageDraw,ImageFont,ImageOps
from collections import Counter

COOPT=Path(__file__).resolve().parents[1]
ROOT=COOPT.parents[1]
BASE=COOPT/'data/results/B'
VISUAL=COOPT/'output/B'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def population_plot(label,experiment):
    source=BASE/label/'step4/step4.2'/experiment/'population32_trace.json'
    if not source.exists():return None
    rows=json.loads(source.read_text());out=VISUAL/label/'step4.2';out.mkdir(parents=True,exist_ok=True)
    initial=BASE/label/'step4/step4.1'/experiment/'population.json'
    size=json.loads(initial.read_text())['population_size']
    values=np.zeros((size,len(rows)+1),dtype=int)
    for j,row in enumerate(rows,1):
        if row['population_size']!=size or len(row['events'])!=size:
            raise ValueError('Population size changed: '+label)
        for item in row['retained']:values[item['slot'],j]=item['juxtaposed']
    fig,(ax,bx)=plt.subplots(1,2,figsize=(10,8),gridspec_kw={'width_ratios':[3,1]})
    im=ax.imshow(values,origin='lower',aspect='auto',vmin=0,
                 vmax=len(json.loads((BASE/label/'step4/step4.1/data/report.json').read_text())['poses']),cmap='viridis')
    ax.set_xlabel('Round (0 = registered initialization)');ax.set_ylabel('Persistent slot')
    ax.set_xticks(range(len(rows)+1));ax.set_title(label)
    fig.colorbar(im,ax=ax,label='Juxtaposed object pose count')
    discrete=[sum(e['branch']=='discrete' for e in row['events'])for row in rows]
    differential=[sum(e['branch']=='differential' for e in row['events'])for row in rows]
    held=[sum(e['branch']=='hold' for e in row['events'])for row in rows];x=np.arange(1,len(rows)+1)
    bx.bar(x,discrete,label='Discrete',color='#319cd7')
    bx.bar(x,differential,bottom=discrete,label='Differential',color='#e8aa42')
    if any(held):bx.bar(x,held,bottom=np.array(discrete)+differential,label='Hold',color='#aaaaaa')
    bx.set_ylim(0,size+2);bx.set_xticks(x);bx.set_xlabel('Round');bx.set_ylabel('Slots taking each branch')
    bx.legend(loc='upper center');fig.tight_layout()
    destination=out/'population.png';fig.savefig(destination,dpi=150);plt.close(fig)
    return dict(source=str(source.relative_to(ROOT)),path=str(destination.relative_to(ROOT)),
                source_sha256=sha(source),sha256=sha(destination))


def saved_layout_views(label, experiment, body, fixture, layout, poses, status):
    """Draw failed as well as passed layouts from saved bytes, without Booleans."""
    from whole_step4_render import raster_tile, sheet
    out=VISUAL/label/'step4.2';out.mkdir(parents=True,exist_ok=True)
    tiles=[];records=[]
    states=layout['native_world'];hosts=layout['hosts'];placements=layout['placements']
    for k in layout['active']:
        T=states[int(hosts[k])];U=T@placements[k]
        posed_body=body.copy();posed_body.apply_transform(U)
        layers=[(posed_body,'#A5ADB5')]
        if fixture is not None:
            posed_fixture=fixture.copy();posed_fixture.apply_transform(T)
            layers.append((posed_fixture,'#2A91D2'))
        image=ImageOps.expand(raster_tile(layers),border=(0,34,0,0),fill='white')
        try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
        except OSError:font=ImageFont.load_default()
        suffix='certified support' if status=='PASS' else 'diagnostic layout'
        ImageDraw.Draw(image).text((10,7),f'{poses[k]} | {status} | {suffix}',font=font,fill='#222222')
        path=out/(poses[k]+'.png');image.save(path);tiles.append(image)
        records.append(dict(id=label,path=str(path.relative_to(ROOT)),sha256=sha(path),
                            source_kind='saved_step4_layout_and_mesh',force_solver_run=False,boolean_run=False))
    path=out/('final_result.png' if fixture is not None else 'layout.png');sheet(tiles,path)
    records.append(dict(id=label,path=str(path.relative_to(ROOT)),sha256=sha(path),
                        source_kind='saved_step4_layout_and_mesh',force_solver_run=False,boolean_run=False))
    return records


def evaluation_plot(label,row,metric):
    fig,(ax,bx)=plt.subplots(1,2,figsize=(10,4),gridspec_kw={'width_ratios':[1.2,1]})
    labels=['No support','Step4 nominal','Complete support']
    values=[row['no_support_bbx_l'],row.get('nominal_bbx_l'),row.get('full_bbx_l')]
    for i,value in enumerate(values):
        if value is None:ax.text(i,0,'No certified result',ha='center',va='bottom',rotation=90,color='#bb3333')
        else:
            ax.bar(i,value,color=['#a5adb5','#319cd7','#59ab72'][i]);ax.text(i,value,f'{value:.2f}',ha='center',va='bottom')
    ax.set_xticks(range(3),labels,rotation=12);ax.set_ylabel('Static bounding box (L)');ax.set_ylim(0,max(v or 0 for v in values)*1.25)
    bx.axis('off');members=['Support '+str(s['state'])+': '+', '.join(s['poses'])for s in metric['fixture_states']]
    bx.text(0,1,f'{row["status"]}\nObject poses: {row["object_pose_count"]}\nSupport installations: {row["support_pose_count"]}\n\n'+ '\n'.join(members),va='top',fontsize=10)
    fig.suptitle(label);fig.tight_layout();path=VISUAL/label/'step4.2/evaluation.png';fig.savefig(path,dpi=150);plt.close(fig)
    return dict(id=label,path=str(path.relative_to(ROOT)),sha256=sha(path),source_kind='saved_layout_static_evaluation',force_solver_run=False,boolean_run=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('experiment')
    parser.add_argument('--source-system',help='Frozen source experiment for a combined execution report')
    args=parser.parse_args()
    e=args.experiment;system=BASE/'data'/('system_'+e)
    source_experiment=args.source_system or e
    source_system=BASE/'data'/('system_'+source_experiment)
    request=json.loads((source_system/'requested_sets.json').read_text())
    system.mkdir(parents=True,exist_ok=True)
    if system!=source_system:
        save(system/'requested_sets.json',request)
        save(system/'source_system.json',dict(experiment=source_experiment,path=str(source_system.relative_to(ROOT))))
    batch=json.loads((BASE/'data'/e/'batch.json').read_text())
    soft=batch.get('options',{}).get('search_policy')=='population_soft'
    oracle_aware=soft and batch['options'].get('allow_oracle_fallback',False)
    early_stop=soft and batch['options'].get('stop_when_feasible',False)
    results={r['id']:r for r in batch['results']}
    frozen=source_system/'frozen_sources/slides/Co-optimize'
    os.environ.setdefault('COOPT_REPO_ROOT',str(ROOT))
    for folder in ['evaluation','helper_func','vis_func','step4.2']:
        sys.path.insert(0,str(frozen/folder))
    import _bootstrap
    from metrics import static_bounds
    from co_common import state
    body=state('B','pose_1')[2]
    records=[];pictures=[]
    for label in request['sets']:
        original=json.loads((BASE/label/'step4/step4.1/data/report.json').read_text())
        row=dict(id=label,object_pose_count=len(original['poses']),status='RUNNING',passed=None)
        result=results.get(label)
        if result is not None:
            case_experiment=result.get('experiment',source_experiment)
            row['source_experiment']=case_experiment
            row.update(passed=result['passed'],force_passed=result.get('force_passed',False),
                       oracle_passed=result.get('base_oracle_passed',False),
                       constructed=result.get('solid_constructed_during_search',False),
                       reuse=result.get('rotating_reuse_pose_count'),juxtaposed=result.get('juxtaposed_pose_count'),
                       target_satisfied=result.get('mixed_target_satisfied'),seconds=result.get('seconds'),
                       error=result.get('error'),oracle_status=result.get('oracle_status'))
            row['status']=('PASS' if row['passed'] else 'RUN_ERROR' if row['error'] else
                           'FORCE_FAIL' if not row['force_passed'] else
                           'ORACLE_FAIL' if not row['oracle_passed'] else 'BUILD_FAIL')
            folder=BASE/label/'step4/step4.2'/case_experiment
            layout=None;mesh=None
            if (folder/'layout.npz').exists():
                with np.load(folder/'layout.npz')as z:layout={k:z[k]for k in z.files}
                mesh=(trimesh.load(folder/'support.obj',force='mesh',process=False)
                      if (folder/'support.obj').exists()else None)
                full_path=folder/'support_with_base.obj'
                full=trimesh.load(full_path,force='mesh',process=False)if full_path.exists()else mesh
                metric=static_bounds(body.vertices,layout,mesh.vertices if mesh is not None else None,
                                     full.vertices if full is not None else None,original['poses'])
                row.update(support_pose_count=metric['support_pose_count'],
                           no_support_bbx_l=metric['object_static']['volume_litres'],
                           members=metric['object_pose_counts_per_support_pose'],
                           reuse=metric['rotating_reuse_pose_count'],juxtaposed=metric['juxtaposed_pose_count'],
                           nominal_bbx_l=(metric['with_step4_support']['volume_litres']
                                          if metric['with_step4_support'] is not None else None),
                           full_bbx_l=metric['with_full_support']['volume_litres']if row['passed']else None,
                           material_cm3=abs(float(full.volume))*1e6 if row['passed']else None,
                           failed_demands=(row['object_pose_count']*32768-sum(result['counts'].values())
                                           if result.get('counts') else None))
                report_path=folder/'report.json'
                report=json.loads(report_path.read_text()) if report_path.exists() else {}
                stats=report.get('frontier_search',{})
                if report and (stats.get('population_size')!=result.get('frontier_width',32) or stats.get('maximum_trials_per_slot')!=1):
                    raise ValueError('Fixed-slot contract failed: '+label)
                row['algorithm_statistics']=stats
                row['rounds']=stats.get('rounds')
                if soft:
                    save(system/'evaluations'/(label.replace('/','__')+'.json'),metric)
                    pictures.append(evaluation_plot(label,row,metric))
            destination=VISUAL/label/'step4.2';destination.mkdir(parents=True,exist_ok=True)
            for name in ['process.png']:
                source=VISUAL/label/'step4/step4.2'/case_experiment/name
                if source.exists():
                    target=destination/name;shutil.copy2(source,target)
                    pictures.append(dict(id=label,source=str(source.relative_to(ROOT)),
                                         path=str(target.relative_to(ROOT)),sha256=sha(target)))
                elif (destination/name).exists():
                    (destination/name).unlink()
            if layout is not None:
                selected_fixture=(mesh if (report or {}).get('step4_contact_core_connected')else
                    full if row['passed'] else mesh)
                row['layout_image']='final_result.png' if selected_fixture is not None else 'layout.png'
                stale=destination/('layout.png' if selected_fixture is not None else 'final_result.png')
                if stale.exists():stale.unlink()
                pictures.extend(saved_layout_views(label,case_experiment,body,selected_fixture,layout,original['poses'],row['status']))
            picture=population_plot(label,case_experiment)
            if picture:pictures.append(dict(id=label,**picture))
        records.append(row)
    completed=len(results);passed=sum(r['passed']for r in results.values())
    sizes=[]
    for count in sorted({row['object_pose_count']for row in records}):
        selected=[row for row in records if row['object_pose_count']==count]
        sizes.append(dict(object_pose_count=count,groups=len(selected),
            completed=sum(row['passed']is not None for row in selected),
            passed=sum(bool(row['passed'])for row in selected),
            failed=sum(row['passed']is False for row in selected)))
    successful=[row for row in records if row['passed']and row.get('full_bbx_l')is not None]
    largest=max(successful,key=lambda row:row['full_bbx_l'])if successful else None
    summary=dict(complete=batch['complete'],experiment=e,group_count=len(records),
        completed_groups=completed,passed_groups=passed,failed_groups=completed-passed,
        object_pose_count=sum(row['object_pose_count']for row in records),
        passed_object_pose_count=sum(row['object_pose_count']for row in records if row['passed']),
        passed_support_pose_count=sum(row.get('support_pose_count',0)for row in records if row['passed']),
        mix_target_groups=sum(4<=row['object_pose_count']<=8 for row in records),
        mix_target_completed_groups=sum(4<=row['object_pose_count']<=8 and row['passed'] is not None for row in records),
        mix_target_satisfied_groups=sum(4<=row['object_pose_count']<=8 and bool(row.get('target_satisfied'))for row in records),
        status_counts=dict(Counter(row['status']for row in records)),
        by_object_pose_count=sizes,
        largest_passed_bbx_case=({key:largest[key]for key in ['id','full_bbx_l','material_cm3']}if largest else None),
        population_size=batch.get('population_size',32),rounds=batch.get('rounds',8),warm_start=False,cases=records,pictures=pictures,
        force_solver_run=False,boolean_run=False)
    if soft:
        summary.update(count_penalty_used=False,oracle_fallback=batch['options'].get('allow_oracle_fallback',False))
        summary['passed_evaluation_totals']=dict(
            no_support_bbx_l=sum(r['no_support_bbx_l']for r in successful),
            nominal_bbx_l=sum(r.get('nominal_bbx_l')or 0 for r in successful),
            full_bbx_l=sum(r['full_bbx_l']for r in successful),
            material_cm3=sum(r['material_cm3']for r in successful))
        if early_stop:summary.update(rounds_are_maximum=True,actual_rounds=sum(row.get('rounds',0)or 0 for row in records),stop_when_feasible=True)
        for row in records:
            row['sampling_count_center']=row['object_pose_count']/3.
        previous_path=source_system/'previous_step4_results.json'
        if previous_path.exists():
            previous=json.loads(previous_path.read_text());old={r['id']:r for r in previous['cases']}
            summary['comparison']=dict(previous_experiment=previous['experiment'],previous_passed=previous['passed_groups'],
                passed=passed,newly_passed=[r['id']for r in records if r['passed']and not old[r['id']]['passed']],
                newly_failed=[r['id']for r in records if r['passed'] is False and old[r['id']]['passed']])
            paired=[r for r in records if r['passed']and old[r['id']]['passed']]
            summary['comparison']['matched_passed_groups']=len(paired)
            summary['comparison']['matched_old_full_bbx_l']=sum(old[r['id']]['full_bbx_l']for r in paired)
            summary['comparison']['matched_new_full_bbx_l']=sum(r['full_bbx_l']for r in paired)
            summary['comparison']['matched_old_material_cm3']=sum(old[r['id']]['material_cm3']for r in paired)
            summary['comparison']['matched_new_material_cm3']=sum(r['material_cm3']for r in paired)
    save(system/'step4_results.json',summary)
    report=BASE/('step4_'+e+'.md')
    lines=['# 固定32候选的新Step4结果','',
        f'批次 `{e}`：已结束 {completed}/{len(records)} 组，完整通过 {passed} 组，失败 {completed-passed} 组。'
        f'批次状态：{"完成" if batch["complete"] else "运行中"}。','',
        f'全部集合共 {summary["object_pose_count"]} 个object pose。当前通过组共 {summary["passed_object_pose_count"]} 个object pose、'
        f'{summary["passed_support_pose_count"]} 个支撑安装pose。失败组的安装数量仅描述保存的诊断布局。','',
        f'2–4原位复用／2–4Juxtapose目标适用于 {summary["mix_target_groups"]} 组；'
        f'当前已结束的 {summary["mix_target_completed_groups"]} 个适用组中，{summary["mix_target_satisfied_groups"]} 组所选布局满足组合数量目标。'
        '数量目标与完整物理通过分别统计。','',
        '32个固定槽位、8轮、注册原位初始化，无历史优化布局暖启动。每槽位每轮随机选择一次内部离散换座或跨槽位差分，再做一轮短调整。'
        '谁换座由优化过程选择；换座起点为世界XY质心对齐，保持原任务朝向。','',
        '[算法说明](../../../population32_algorithm.md)。当前每组图在 `output/B/[set]/step4.2/`：'
        '`final_result.png` 为有导出实体的所选布局；未过受力、没有实体导出的集合提供物体姿态诊断 `layout.png`。'
        '`process.png` 为已有的几何搜索过程，`population.png` 为32个槽位的换座数及每轮分支分配。','',
        'PASS要求原始需求、Oracle共同连接和搜索内实际成品构造通过。失败行的布局和名义BBX仅用于诊断，完整成品列为空。'
        '组合目标单独报告，原受力／几何验收不放宽。本轮只运行Step4。','',
        '| 每组object pose | 组数 | 已结束 | PASS | 失败 |',
        '|---:|---:|---:|---:|---:|']
    if soft:
        lines[0]='# 失败量加权＋软换座倾向：全52组Step4结果'
        start=next(i for i,line in enumerate(lines)if line.startswith('2–4原位'))
        lines[start]='Juxtapose采样数量中心为每组object pose数/3；只影响新增/取消概率，不是优化目标、通过条件或数量惩罚。最终数量由物理结果产生。'
        start=next(i for i,line in enumerate(lines)if line.startswith('32个固定槽位'))
        lines[start]=('32个固定槽位、8轮、注册原位初始化，无历史布局暖启动。按当前原需求失败比例与Oracle冲突抽pose；受力通过但Oracle仍失败的pose可继续参与优化。先离散或差分变异，再一次原范围调整，最终数量由优化决定。'if oracle_aware else
            '32个固定槽位、8轮、注册原位初始化，无历史布局暖启动。按当前32768需求失败比例抽pose，成功pose参数不主动移动；先离散或差分变异，再一次原范围调整。全受力通过时hold，继续原连接/构造认证。')
        if early_stop:lines[start]=lines[start].replace('8轮','最多8轮')+' 取得完整成品后提前返回，实际轮数记录在CSV与每组统计；不声称体积达到最优。'
        start=next(i for i,line in enumerate(lines)if line.startswith('[算法说明]'))
        lines[start]=lines[start].replace('population32_algorithm.md','population_soft_algorithm.md')+' Hold在population.png中显示为灰色。'
        start=next(i for i,line in enumerate(lines)if line.startswith('PASS要求'))
        lines[start]='PASS要求原始需求、Oracle共同连接和实际成品构造全部通过。失败图明确标注失败；失败布局的名义BBX用于诊断，成品BBX留空。本轮只运行Step4。'
        if summary.get('comparison'):
            c=summary['comparison'];lines[2]+=(f' 与上次同为32槽位、8轮的冷启动相比：{c["previous_passed"]}/52 → {passed}/52；新增通过{len(c["newly_passed"])}组，原通过转失败{len(c["newly_failed"])}组。')
    for size in sizes:
        lines.append(f'| {size["object_pose_count"]} | {size["groups"]} | {size["completed"]} | {size["passed"]} | {size["failed"]} |')
    if largest:
        lines.extend(['',f'通过组中完整BBX最大的集合为 `{largest["id"]}`：{largest["full_bbx_l"]:.2f}L。'
                      '完整通过也可能有较大的成品占用盒，须结合下方尺寸列判断搜索效果。'])
    lines.extend(['',
        '| Set | 状态 | object pose | 支撑pose | 原位复用 | Juxtapose | '+('采样中心N/3'if soft else'组合目标')+' | 未通过原需求数 | Oracle | 无支撑BBX L | 名义BBX L | 成品BBX L | 成品材料cm³ | 图 |',
        '|---|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|'])
    for row in records:
        def val(key):return str(row[key])if row.get(key)is not None else '—'
        def num(key):return f'{row[key]:.2f}'if row.get(key)is not None else '—'
        image=VISUAL/row['id']/'step4.2'/row.get('layout_image','layout.png')
        url=os.path.relpath(image,BASE)
        link=f'[布局]({url})'if row['status']!='RUNNING'and image.exists()else '—'
        population=VISUAL/row['id']/'step4.2/population.png'
        if row['status']!='RUNNING'and population.exists():
            link+=f' · [32候选]({os.path.relpath(population,BASE)})'
        evaluation=VISUAL/row['id']/'step4.2/evaluation.png'
        if soft and row['status']!='RUNNING'and evaluation.exists():
            link+=f' · [Evaluation]({os.path.relpath(evaluation,BASE)})'
        target=('通过' if row.get('target_satisfied') else '未达成') if 4<=row['object_pose_count']<=8 and row['passed'] is not None else '—'
        if soft:target=f'{row["sampling_count_center"]:.2f}'
        lines.append(f'| {row["id"]} | {row["status"]} | {row["object_pose_count"]} | '
            f'{val("support_pose_count")} | {val("reuse")} | {val("juxtaposed")} | '
            f'{target} | {val("failed_demands")} | {val("oracle_status")} | {num("no_support_bbx_l")} | {num("nominal_bbx_l")} | {num("full_bbx_l")} | {num("material_cm3")} | {link} |')
    lines.extend(['',f'[结构化结果与图片来源](data/system_{e}/step4_results.json)。',
        f'[组合执行批次](data/{e}/batch.json)。',
        f'[冻结源码与摘要](data/system_{source_experiment}/frozen_sources.json)。'])
    if batch['complete']:
        lines.append(f'[保存产物核对](data/system_{e}/saved_artifact_audit.json)。')
    if soft:
        lines.extend(['','每组 `evaluation.png` 比较无支撑／Step4名义模型／完整支撑的静态BBX，并列出每个支撑安装服务的object pose。失败组不提供完整成品体积。'])
        totals=summary['passed_evaluation_totals']
        lines.extend(['',f'通过组的逐组BBX之和：无支撑{totals["no_support_bbx_l"]:.3f}L／Step4名义模型{totals["nominal_bbx_l"]:.3f}L／完整支撑{totals["full_bbx_l"]:.3f}L；完整支撑材料之和{totals["material_cm3"]:.3f}cm³。这是不同集合的指标合计，不是一件共同支撑的尺寸。'])
        c=summary.get('comparison')
        if c:
            lines.extend(['',f'共同通过{c["matched_passed_groups"]}组，完整BBX之和：旧{c["matched_old_full_bbx_l"]:.3f}L → 新{c["matched_new_full_bbx_l"]:.3f}L；材料之和：旧{c["matched_old_material_cm3"]:.3f}cm³ → 新{c["matched_new_material_cm3"]:.3f}cm³。该比较只含两次均通过的相同集合。',
                '', '新增通过：'+('、'.join('`'+label+'`'for label in c['newly_passed'])or'无')+'。',
                '', '原通过转失败：'+('、'.join('`'+label+'`'for label in c['newly_failed'])or'无')+'。'])
    report.write_text('\n'.join(lines)+'\n')
    csv_path=report.with_suffix('.csv')
    fields=['id','status','source_experiment','object_pose_count','support_pose_count','reuse','juxtaposed','target_satisfied',
            'failed_demands','oracle_status','no_support_bbx_l','nominal_bbx_l','full_bbx_l','material_cm3','rounds','seconds']
    with csv_path.open('w',newline='')as file:
        writer=csv.DictWriter(file,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(records)
    print('STEP4 REPORT',completed,'/',len(records),'PASS',passed,'PNG',len(pictures),flush=True)


if __name__=='__main__':main()
