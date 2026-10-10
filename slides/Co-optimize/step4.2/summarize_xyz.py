"""Collect the XYZ pipeline, original volume comparison and actual heights."""
import argparse
import html
import json
from pathlib import Path
import sys
from statistics import median
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
from co_common import HERE,I,save
from whole_search.checked_solution import check_force_result


def summarize_force(experiment, pipeline):
    """Separate fresh search outcomes from material-incumbent selection."""
    root=HERE/'output/B';directory=root/'data'/experiment
    previous=json.loads((directory/'incumbents.json').read_text())['selected']
    stages=[directory]+sorted(p for p in directory.parent.glob(experiment+'_*')
                             if (p/'batch.json').exists())
    batches=[json.loads((p/'batch.json').read_text()) for p in stages]

    def row_for(label, result):
        out=root/label/'step4/step4.2'/result['experiment']
        row=dict(id=label,pose_count=len(label.split('+')),passed=bool(result['passed']),
                 experiment=result['experiment'],old_volume_cm3=previous[label]['volume_cm3'],
                 retained_incumbent=result.get('retained_incumbent',False),
                 process=None,final=None,airborne_poses=[],maximum_height_mm=0.)
        path=out/'data/report.json'
        if not path.exists():path=out/'report.json'
        data=check_force_result(path) if row['passed'] else (
             json.loads(path.read_text()) if path.exists() else {})
        for key in ('counts','acceptance','volume_measure','mesh_exported','search_seconds',
                    'mesh_export_seconds','validation_seconds','estimated_volume_cm3',
                    'exact_evaluations','geometry_recovery_run','full_demand_recheck_run'):
            if key in data:row[key]=data[key]
        row['volume_cm3']=data.get('volume_cm3',result.get('volume_cm3'))
        row['exported_material']=bool(row['passed'] and (out/'support.obj').exists()
            and row.get('volume_measure')!='nominal_material_occupancy_estimate')
        if row['passed']:
            assert all(n==32768 for n in data['counts'].values())
            with np.load(out/'layout.npz') as saved:
                heights={pose:float((saved['native_world'][saved['hosts'][k]]@saved['placements'][k])[2,3]
                    -saved['native_world'][k,2,3]) for k,pose in enumerate(data['poses'])}
            assert all(h>=-1e-9 for h in heights.values())
            row.update(heights_m=heights,airborne_poses=[p for p,h in heights.items() if h>1e-9],
                       maximum_height_mm=max(heights.values())*1000)
        if row['exported_material']:
            row['volume_change_percent']=(row['volume_cm3']/row['old_volume_cm3']-1)*100
        for key,name in [('process','process.png'),('final','final_result.png')]:
            if (out/name).exists():row[key]=str((out/name).relative_to(root))
        if not row['passed']:row['error']=result.get('error','unresolved original demands')
        return row

    raw=[row_for(label,pipeline['attempted'][label]) for label in pipeline['groups']]
    selected=[row_for(label,pipeline['selected'][label]) for label in pipeline['groups']]
    for row in raw:
        search_total=0.;export_total=0.;case_total=0.
        for stage,batch in zip(stages,batches):
            stage_out=root/row['id']/'step4/step4.2'/stage.name
            search_path=stage_out/'search_report.json'
            if search_path.exists():search_total+=json.loads(search_path.read_text()).get('search_seconds',0.)
            report_path=stage_out/'report.json'
            if report_path.exists():export_total+=json.loads(report_path.read_text()).get('mesh_export_seconds',0.)
            case_total+=sum(r.get('seconds',0.) for r in batch['results'] if r['id']==row['id'])
        row.update(all_stage_search_seconds=search_total,all_stage_mesh_export_seconds=export_total,
                   all_stage_seconds=case_total)
    raw_mesh=[r for r in raw if r['exported_material']]
    chosen=[r for r in selected if r['passed']]
    raw_volume=sum(r['volume_cm3'] for r in raw_mesh)
    raw_old=sum(r['old_volume_cm3'] for r in raw_mesh)
    volume=sum(r['volume_cm3'] for r in chosen)
    old=sum(r['old_volume_cm3'] for r in chosen)
    for row in chosen:assert row['volume_cm3']<=row['old_volume_cm3']
    report=dict(complete=True,experiment=experiment,groups=len(raw),cases=selected,
        raw_new_cases=raw,passed=len(chosen),new_search_passed=sum(r['passed'] for r in raw),
        cold_start_passed=pipeline['base_budget_passes'],
        additional_gradient_passed=pipeline['gradient_repair_passes'],
        additional_conditioning_passed=0,raw_new_exported_meshes=len(raw_mesh),
        raw_new_material_volume_cm3=raw_volume,
        previous_same_raw_new_passed_cases_volume_cm3=raw_old,
        raw_material_volume_change_percent=(raw_volume/raw_old-1)*100 if raw_old else None,
        total_material_volume_cm3=volume,previous_same_passed_cases_volume_cm3=old,
        material_volume_change_percent=(volume/old-1)*100 if old else None,
        raw_new_airborne_poses=sum(len(r['airborne_poses']) for r in raw),
        total_airborne_poses=sum(len(r['airborne_poses']) for r in chosen),
        retained_incumbents=pipeline['retained_incumbents'],improved_incumbents=pipeline['improved_incumbents'],
        median_new_search_seconds=median(r['all_stage_search_seconds'] for r in raw),
        median_new_mesh_export_seconds=median(r['all_stage_mesh_export_seconds'] for r in raw),
        median_new_case_seconds=median(r['all_stage_seconds'] for r in raw),
        total_new_batch_wall_seconds=sum(b['seconds'] for b in batches),
        cold_batch_wall_seconds=batches[0]['seconds'],cold_batch_jobs=batches[0]['options']['jobs'],
        acceptance=pipeline['acceptance'],translation_axis_priority='equal',
        geometry_recovery_run=False,final_full_demand_recheck_run=False,
        original_demand_samples_reused=True,old_optimized_answers_used_as_start=False,
        material_non_regression_checked=pipeline['material_non_regression_checked'],
        comparison='Smallest preserved selected v2/v1/original incumbents, read before fresh Step4.1 starts',
        retained_result_timings_are_historical=True,step5_installed_base_acceptance_run=False)
    save(directory/'summary.json',report)

    def pictures(row):
        return ' · '.join(f"[{title}]({row[key]})" for key,title in [('process','过程'),('final','最终')]
                          if row[key]) or '未导出图片'

    lines=['# 统一 XYZ：七组重跑','',
        f"本轮新搜索力／力矩通过 **{report['new_search_passed']}/{len(raw)}**，初次预算通过{report['cold_start_passed']}组，额外自身状态梯度接续通过{report['additional_gradient_passed']}组。",'',
        '连续 Translation 使用同一世界 XYZ 梯度、范数和步幅池；地面约束保留，抬高即去掉物体地面反力。每个 pose 沿用全部 32768 个原始需求。选定结果后复用已算出的通过掩码，不运行几何微扰补救或末尾重复求解。', '',
        f"本轮搜索中位数 **{report['median_new_search_seconds']:.1f}s/组**；固定布局 mesh 导出中位数{report['median_new_mesh_export_seconds']:.1f}s。{report['cold_batch_jobs']}个并行进程，所有阶段含出图总计 **{report['total_new_batch_wall_seconds']/60:.2f}min**。计入失败与接续预算，保留旧答案的历史耗时未算入。",'',
        '## 本轮新搜索','',
        '通过判据是搜索接触模型上的全部原始力／力矩需求。体积取固定布局导出的名义实体 mesh；未通过或未导出 mesh 的估计体积不计入实体总体积。对照为本轮开始前保存的逐组最小答案，合计713.007866cm³。','',
        '| pose set | 力／力矩 | 原体积 cm³ | 新实体体积 cm³ | 变化 | 离地 pose／最高 mm | 搜索 s | 图 |',
        '| --- | --- | ---: | ---: | ---: | --- | ---: | --- |']
    for row in raw:
        material=f"{row['volume_cm3']:.2f}" if row['exported_material'] else '—'
        delta=f"{row['volume_change_percent']:+.2f}%" if row['exported_material'] else '—'
        status='通过' if row['passed'] else '未通过'
        lines.append(f"| {row['id']} | {status} | {row['old_volume_cm3']:.2f} | {material} | {delta} | {len(row['airborne_poses'])}／{row['maximum_height_mm']:.3f} | {row['all_stage_search_seconds']:.1f} | {pictures(row)} |")
    if raw_mesh:
        lines+=['',f"已导出新实体的{len(raw_mesh)}组：合计{raw_volume:.2f}cm³，同组旧答案{raw_old:.2f}cm³，变化{report['raw_material_volume_change_percent']:+.2f}%。新结果中{report['raw_new_airborne_poses']}个 pose 离地。"]
    lines+=['','## 最终材料择优','',
        f"最终保留 **{len(chosen)}/{len(selected)}** 个通过方案；本轮改善{report['improved_incumbents']}组，沿用旧答案{report['retained_incumbents']}组。总体积{volume:.2f}cm³，旧答案{old:.2f}cm³，变化{report['material_volume_change_percent']:+.2f}%。这些保留答案未用作新搜索的初始化。",'',
        '| pose set | 最终体积 cm³ | 来源 | 图 |','| --- | ---: | --- | --- |']
    for row in selected:
        source='沿用旧答案' if row['retained_incumbent'] else '本轮新搜索'
        material=f"{row['volume_cm3']:.2f}" if row['passed'] else '—'
        lines.append(f"| {row['id']} | {material} | {source} | {pictures(row)} |")
    lines+=['',f"[图片浏览]({experiment}_index.html) · [完整记录](data/{experiment}/summary.json)"]
    (root/f'{experiment}_results.md').write_text('\n'.join(lines)+'\n')
    gallery="<!doctype html><meta charset='utf-8'><title>Unified XYZ results</title><style>body{font:16px sans-serif;margin:32px;background:#fafafa}article{margin:24px 0;padding:20px;background:white;border:1px solid #ddd}h2{font-size:19px}img{max-width:100%;display:block;margin-top:12px}</style>"
    gallery+=f"<h1>本轮新搜索：{report['new_search_passed']}/{len(raw)}</h1>"
    for title,group in [('本轮新搜索',raw),('最终材料择优',selected)]:
        gallery+=f'<h2>{title}</h2>'
        for row in group:
            status='通过' if row['passed'] else '未通过'
            material=f"{row['volume_cm3']:.2f}cm³" if row['exported_material'] else '无已通过的实体体积'
            source='沿用旧答案' if row['retained_incumbent'] else '本轮新搜索'
            images=''.join(f"<img loading='lazy' src='{html.escape(row[key],quote=True)}'>"
                           for key in ['process','final'] if row[key])
            gallery+=f"<article><h2>{html.escape(row['id'])}</h2><p>{status} · {material} · {source}</p>{images}</article>"
    (root/f'{experiment}_index.html').write_text(gallery)
    print(json.dumps({k:v for k,v in report.items() if k not in ('cases','raw_new_cases')},ensure_ascii=False))


def summarize(experiment):
    root=HERE/'output/B';directory=root/'data'/experiment
    pipeline=json.loads((directory/'pipeline.json').read_text())
    assert pipeline['complete']
    force_only=pipeline.get('acceptance')=='all_original_force_torque_demands_on_search_contacts'
    if force_only:return summarize_force(experiment,pipeline)
    previous=json.loads((root/'data/stable_gradient_results.json').read_text())['selected']
    rows=[]
    for label in pipeline['groups']:
        result=pipeline['selected'][label];out=root/label/'step4/step4.2'/result['experiment']
        row=dict(id=label,pose_count=len(label.split('+')),passed=result['passed'],
                 experiment=result['experiment'],old_volume_cm3=previous[label]['volume_cm3'])
        if result['passed']:
            report=check_force_result(out/'data/report.json')
            assert (report.get('force_passed') or report.get('force_exit_work_passed')) and all(n==32768 for n in report['counts'].values())
            with np.load(out/'layout.npz') as saved:
                heights={pose:float((saved['native_world'][saved['hosts'][k]]@saved['placements'][k])[2,3]
                    -saved['native_world'][k,2,3]) for k,pose in enumerate(report['poses'])}
            for pose,height in heights.items():
                assert height>=-1e-9
                assert report.get('workpiece_floor_contact_allowed',{}).get(pose,True)==(height<=1e-9)
                if 'workpiece_heights_m' in report:
                    np.testing.assert_allclose(height,report['workpiece_heights_m'][pose],atol=1e-12,rtol=0)
                if height>1e-9:
                    with np.load(out/f'{pose}_force.npz') as force:
                        supply=force['supply_7d'];is_head=np.max(abs(supply[:,:6]),axis=1)>1e-12
                        # No physical floor column: head equation7 equals head Fz.
                        np.testing.assert_allclose(supply[is_head,6],supply[is_head,2],atol=1e-12,rtol=0)
            delta=(report['volume_cm3']/row['old_volume_cm3']-1)*100
            row.update(volume_cm3=report['volume_cm3'],volume_change_percent=delta,
                retained_incumbent=result.get('retained_incumbent',False),
                incumbent_volume_cm3=result.get('incumbent_volume_cm3'),
                attempted_volume_cm3=result.get('attempted_volume_cm3'),
                heights_m=heights,airborne_poses=[pose for pose,h in heights.items() if h>1e-9],
                maximum_height_mm=max(heights.values())*1000,
                actual_force_and_geometry_passed=bool(report.get('force_exit_work_passed')),
                force_torque_passed=True,acceptance=report.get('acceptance','actual_geometry_and_demands'),
                volume_measure=report.get('volume_measure','actual_material_mesh'),
                workpiece_floor_state_and_saved_reactions_verified=True,
                process=str((out/'process.png').relative_to(root)) if (out/'process.png').exists() else None,
                final=str((out/'final_result.png').relative_to(root)) if (out/'final_result.png').exists() else None,
                search_seconds=report['search_seconds'],validation_seconds=report['validation_seconds'])
        else:
            row['error']=result.get('error','unresolved')
        rows.append(row)
    passed=[r for r in rows if r['passed']]
    new_volume=sum(r['volume_cm3'] for r in passed)
    old_volume=sum(r['old_volume_cm3'] for r in passed)
    batch=json.loads((directory/'batch.json').read_text())
    report=dict(complete=True,experiment=experiment,cases=rows,groups=len(rows),passed=len(passed),
        cold_start_passed=pipeline['base_budget_passes'],additional_gradient_passed=pipeline['gradient_repair_passes'],
        additional_conditioning_passed=pipeline['conditional_passes'],
        total_airborne_poses=sum(len(r['airborne_poses']) for r in passed),
        comparison='Previous selected completed stable-gradient answers; continuation budgets may differ',
        original_demand_samples_reused=True,step5_installed_base_acceptance_run=False)
    report.update(acceptance=pipeline.get('acceptance','actual_geometry_and_demands'),
                  geometry_recovery_run=pipeline.get('geometry_recovery_run',False))
    if 'material_non_regression_checked' in pipeline:
        report.update(material_non_regression_checked=pipeline['material_non_regression_checked'],
            retained_incumbents=pipeline['retained_incumbents'],improved_incumbents=pipeline['improved_incumbents'],
            new_search_passed=pipeline['new_search_passed'])
        for row in passed:
            if row['incumbent_volume_cm3'] is not None:assert row['volume_cm3']<=row['incumbent_volume_cm3']
    report.update(total_material_volume_cm3=new_volume,previous_same_passed_cases_volume_cm3=old_volume,
        material_volume_change_percent=(new_volume/old_volume-1)*100 if old_volume else None,
        median_search_seconds=median(r['search_seconds'] for r in passed) if passed else None,
        median_validation_seconds=median(r['validation_seconds'] for r in passed) if passed else None,
        cold_batch_wall_seconds=batch['seconds'],cold_batch_jobs=batch['options']['jobs'])
    if 'attempted' in pipeline:
        stages=[directory]+[p for p in directory.parent.glob(experiment+'_*') if (p/'batch.json').exists()]
        stage_batches=[json.loads((p/'batch.json').read_text()) for p in stages]
        search_times=[];validation_times=[]
        for label in pipeline['groups']:
            search_total=0.;validation_total=0.
            for stage in stages:
                stage_out=root/label/'step4/step4.2'/stage.name
                search_file=stage_out/'search_report.json'
                if search_file.exists():search_total+=json.loads(search_file.read_text()).get('search_seconds',0.)
                attempts=stage_out/'final_validation_attempts.json'
                if attempts.exists():validation_total+=sum(r['seconds'] for r in json.loads(attempts.read_text()))
            search_times.append(search_total);validation_times.append(validation_total)
        report.update(median_new_search_seconds=median(search_times),
            median_new_validation_seconds=median(validation_times),
            total_new_batch_wall_seconds=sum(b['seconds'] for b in stage_batches),
            retained_result_timings_are_historical=True)
    save(directory/'summary.json',report)
    lines=['# XYZ Translation：8–10 pose 结果','',
        f"{'力／力矩' if force_only else '实际'}通过 **{len(passed)}/{len(rows)}**；初次搜索通过{pipeline['base_budget_passes']}组，额外梯度修复通过{pipeline['gradient_repair_passes']}组。"+("不运行几何微扰或末尾重复需求检查。" if force_only else f"几何conditioning通过{pipeline['conditional_passes']}组。"),'',
        '在已Juxtapose位置允许世界XYZ移动，始终保持工件不穿地；抬高后移除物体地面反力，保留原重力、全部32768需求及第七条约束。原Step3／4.1与历史答案保留。', '',
        ('新方案的体积取固定布局导出的名义支撑mesh；mesh未能导出时只报告搜索体积估计，不能替换有实体体积的旧答案。' if force_only else '体积是实际支撑实体体积；')+'对照是此前stable-gradient最终所选答案，包含各自明确记录的接续，并非相同冷启动预算对照。', '',
        '| pose set | 实际结果 | 原体积 cm³ | 最终体积 cm³ | 变化 | 来源 | 悬空pose数／最高 mm | 图 |',
        '| --- | --- | ---: | ---: | ---: | --- | --- | --- |']
    cards=[]
    for row in rows:
        if row['passed']:
            source='保留已验证结果' if row['retained_incumbent'] else '本轮新搜索'
            pictures=' · '.join(f"[{title}]({row[key]})" for key,title in [('process','过程'),('final','最终')] if row[key]) or '未导出图片'
            lines.append(f"| {row['id']} | 通过 | {row['old_volume_cm3']:.2f} | {row['volume_cm3']:.2f} | {row['volume_change_percent']:+.1f}% | {source} | {len(row['airborne_poses'])}／{row['maximum_height_mm']:.3f} | {pictures} |")
            images=''.join(f"<img src='{html.escape(row[key],quote=True)}'>" for key in ['process','final'] if row[key])
            cards.append(f"<article><h2>{html.escape(row['id'])}</h2><p>{row['volume_cm3']:.2f} cm³ · {source} · 悬空{len(row['airborne_poses'])}个pose</p>{images}</article>")
        else:
            lines.append(f"| {row['id']} | 未解决 | {row['old_volume_cm3']:.2f} | — | — | — | — | — |")
    if 'material_non_regression_checked' in pipeline:
        lines[4:4]=[f"逐组真实材料体积不退步检查{pipeline['material_non_regression_checked']}组；本轮新搜索通过{pipeline['new_search_passed']}组，新改善{pipeline['improved_incumbents']}组，保留已验证答案{pipeline['retained_incumbents']}组。新搜索与保留结果明确分开，旧答案未用作冷启动。",'']
    lines+=['',f"最终通过方案中，共有{report['total_airborne_poses']}个pose实际离地。允许+z并不强制抬高；搜索只接受计入所有pose收益损失后的候选。",'',
            'Step5仍需按最终位置设计与检验系统—地面的基座；本实验未增加安装、连通或强度验收。', '',
            f"[图片浏览]({experiment}_index.html) · [完整记录](data/{experiment}/summary.json)"]
    if passed:
        if 'median_new_search_seconds' in report:
            timing=f"本轮各阶段搜索中位数{report['median_new_search_seconds']:.1f}s，实际候选检查中位数{report['median_new_validation_seconds']:.1f}s；批次运行合计{report['total_new_batch_wall_seconds']/60:.1f}min。计入失败、超时和接续预算，保留答案的历史耗时不算作新运行。"
        else:
            timing=f"搜索中位数{report['median_search_seconds']:.1f}s，实际检查中位数{report['median_validation_seconds']:.1f}s；本轮{report['cold_batch_jobs']}个进程，总运行{batch['seconds']/60:.1f}min。实际检查时间包含失败与超时，额外出图计入批次总时间。"
        lines[4:4]=[f"同组实体材料体积合计{new_volume:.2f}cm³，旧结果{old_volume:.2f}cm³，变化{report['material_volume_change_percent']:+.2f}%。"+timing,'']
    (root/f'{experiment}_results.md').write_text('\n'.join(lines)+'\n')
    gallery="<!doctype html><meta charset='utf-8'><title>XYZ Translation results</title><style>body{font:16px sans-serif;margin:32px;background:#fafafa}article{margin:24px 0;padding:20px;background:white;border:1px solid #ddd}h2{font-size:19px}img{max-width:100%;display:block;margin-top:12px}</style>"
    gallery+=f"<h1>XYZ Translation · {len(passed)}/{len(rows)}</h1>"+''.join(cards)
    (root/f'{experiment}_index.html').write_text(gallery)
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('experiment',nargs='?',default='stable_gradient_xyz_v1')
    summarize(parser.parse_args().experiment)
