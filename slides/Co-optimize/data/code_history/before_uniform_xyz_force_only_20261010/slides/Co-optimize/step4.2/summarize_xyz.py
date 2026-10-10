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
from whole_search.checked_solution import check_solution


def summarize(experiment):
    root=HERE/'output/B';directory=root/'data'/experiment
    pipeline=json.loads((directory/'pipeline.json').read_text())
    assert pipeline['complete']
    previous=json.loads((root/'data/stable_gradient_results.json').read_text())['selected']
    rows=[]
    for label in pipeline['groups']:
        result=pipeline['selected'][label];out=root/label/'step4/step4.2'/result['experiment']
        row=dict(id=label,pose_count=len(label.split('+')),passed=result['passed'],
                 experiment=result['experiment'],old_volume_cm3=previous[label]['volume_cm3'])
        if result['passed']:
            report=check_solution(out/'data/report.json')
            assert report['force_exit_work_passed'] and all(n==32768 for n in report['counts'].values())
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
                maximum_height_mm=max(heights.values())*1000,actual_force_and_geometry_passed=True,
                workpiece_floor_state_and_saved_reactions_verified=True,
                process=str((out/'process.png').relative_to(root)),
                final=str((out/'final_result.png').relative_to(root)),
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
        f"实际通过 **{len(passed)}/{len(rows)}**；初次搜索通过{pipeline['base_budget_passes']}组，额外梯度修复通过{pipeline['gradient_repair_passes']}组，几何conditioning通过{pipeline['conditional_passes']}组。",'',
        '在已Juxtapose位置允许世界XYZ移动，始终保持工件不穿地；抬高后移除物体地面反力，保留原重力、全部32768需求及第七条约束。原Step3／4.1与历史答案保留。', '',
        '体积是实际支撑实体体积；对照是此前stable-gradient最终所选答案，包含各自明确记录的接续，并非相同冷启动预算对照。', '',
        '| pose set | 实际结果 | 原体积 cm³ | 最终体积 cm³ | 变化 | 来源 | 悬空pose数／最高 mm | 图 |',
        '| --- | --- | ---: | ---: | ---: | --- | --- | --- |']
    cards=[]
    for row in rows:
        if row['passed']:
            source='保留已验证结果' if row['retained_incumbent'] else '本轮新搜索'
            lines.append(f"| {row['id']} | 通过 | {row['old_volume_cm3']:.2f} | {row['volume_cm3']:.2f} | {row['volume_change_percent']:+.1f}% | {source} | {len(row['airborne_poses'])}／{row['maximum_height_mm']:.3f} | [过程]({row['process']}) · [最终]({row['final']}) |")
            cards.append(f"<article><h2>{html.escape(row['id'])}</h2><p>{row['volume_cm3']:.2f} cm³ · {source} · 悬空{len(row['airborne_poses'])}个pose</p><img src='{html.escape(row['process'],quote=True)}'><img src='{html.escape(row['final'],quote=True)}'></article>")
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
