"""Publish all52 comparisons, with interrupted work and stage timing explicit."""
import argparse
import csv
import json
from pathlib import Path
from run_pso_campaign import ROOT, BASE, POLICIES, read, save
from analyze_pso_proposals import analyze


def number(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def percent(value, reference):
    return '—' if value is None or not reference else f'{100 * (value / reference - 1):+.2f}%'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--system', default='pso_comparison_all52_20261013_01')
    args = parser.parse_args()
    system = BASE / 'data' / ('system_' + args.system)
    audit = read(system / 'final_campaign_audit.json')
    assert audit['passed']
    diagnostics = analyze(args.system)
    summary = read(system / 'comparison_summary.json')
    pso = read(system / 'pso_evaluation.json')
    control = {r['id']: r for r in read(system / 'differential_evaluation.json')['cases']}
    previous = {r['id']: r for r in read(system / 'previous_evaluation.json')['cases']}
    rows = sorted(pso['cases'], key=lambda r: (r['object_pose_count'], r['id']))
    assert len(rows) == 52
    summary.update(pso_step51_completed=sum(r.get('step51_source') is not None for r in rows),
                   pso_step52_passed=sum(r.get('pruned_bbx_l') is not None for r in rows),
                   control_step51_completed=sum(r.get('step51_source') is not None for r in control.values()),
                   control_step52_passed=sum(r.get('pruned_bbx_l') is not None for r in control.values()))
    save(system / 'comparison_summary.json', summary)
    report = BASE / 'comparison_pso_all52_20261013_01.md'
    if report.exists():
        archive = system / 'prepared_report_before_final_audit.md'
        if not archive.exists():
            archive.write_bytes(report.read_bytes())
    skipped = [r for r in rows if not r['step4_passed']]
    failed_post = [r for r in rows if r['step4_passed'] and not r['passed']]
    lines = ['# PSO：全部52组到Step5.3的结果', '',
        f"全部52组、263个object pose均已结束。PSO Step4通过{summary['pso_step4_passed']}/52，"
        f"Step5.3通过{summary['pso_final_passed']}/52；同配置差分对照分别为"
        f"{summary['control_step4_passed']}/52、{summary['control_final_passed']}/52。"
        f"PSO有{len(skipped)}组因Step4未认证而未进入Step5，另{len(failed_post)}组后处理未完成。", '',
        f"PSO Step5.1完成{summary['pso_step51_completed']}组，最终圆整主体Step5.2认证"
        f"{summary['pso_step52_passed']}组；对照分别为{summary['control_step51_completed']}、"
        f"{summary['control_step52_passed']}组。中间参考实体不计作最终圆整主体。", '',
        '只将连续平移提案替换为PSO：w=c1=c2=0.25。原位初始化、动态角色、离散换座、方向扰动、'
        '一次调整、32槽位/最多8轮、连通主体、原始32768需求/pose及Oracle保持。Step5对每个新布局'
        '生成自己的参考并执行既有删减、长脚、固定脚与表面圆整；不混用旧布局几何。', '',
        '当前已接受的旧设计仍为`sculpted_pruning_all52_20261013_01`，52/52通过。旧活动包含32/48槽位'
        '独立重试和候选精修，不能和新32槽位预算称为相同计算量。新活动开始前已有两个版本各1组'
        '完成、各6组中断；完成组保留，中断组保留旧文件后从注册原位重跑。旧部分尝试另外留下52轮、'
        '1664个槽位事件，不计入选中结果的单组耗时，精确历史CPU总量无法恢复。', '',
        '## 共同通过集合的BBX', '',
        f"两个新版本共同取得成品的{summary['matched_final_groups']}组：差分成品"
        f"{summary['matched_control_final_bbx_l']:.6f}→PSO {summary['matched_pso_final_bbx_l']:.6f} L，"
        f"{percent(summary['matched_pso_final_bbx_l'], summary['matched_control_final_bbx_l'])}。", '',
        f"PSO通过组与旧已接受同组比较：旧成品{summary['previous_same_groups_final_bbx_l']:.6f}→"
        f"PSO {summary['pso_passed_final_bbx_l']:.6f} L。这些组共"
        f"{summary['pso_passed_object_poses']}个object pose、{summary['pso_passed_support_poses']}个实际支撑"
        f"安装pose，复用{summary['pso_passed_reuse']}/Juxtapose {summary['pso_passed_juxtaposed']}。", '',
        'BBX为固定布局下静态占用并集的世界XYZ盒，单位L；材料另列cm³，不计装卸行程。'
        '跨独立集合统计和不代表一个工作站的大小。', '',
        '## 全部52组', '',
        'O/S为object/实际支撑安装pose数；R/J为原位复用/Juxtapose。失败行的角色和S只是诊断布局，'
        '没有成品BBX。表中旧指已接受设计，对照指本次差分。每个成功组链接到自己的实际成品。', '',
        '| Set | O | S旧→PSO | R/J旧→PSO | 无支撑L | Step4L | Step5.2L | Step5.3L | 成品−Step4 L/% | 旧成品L | 成品−旧 L/% | 对照成品L | 状态 |',
        '| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | --- |']
    records = []
    for row in rows:
        old = previous[row['id']]
        peer = control[row['id']]
        core = row['step4_bbx_l'] if row['step4_passed'] else None
        final = row['final_bbx_l'] if row['passed'] else None
        delta_core = None if final is None or core is None else final - core
        delta_old = None if final is None else final - old['final_bbx_l']
        label = row['id']
        if row['passed']:
            label = f"[{label}](../../../output/B/{label}/step5.3/overview.png)"
        else:
            for name in ['final_result.png', 'layout.png']:
                if (ROOT / 'slides/Co-optimize/output/B' / label / 'step4.2' / name).exists():
                    label = f"[{label}](../../../output/B/{label}/step4.2/{name})"
                    break
        role = '—' if row['reuse'] is None else f"{row['reuse']}/{row['juxtaposed']}"
        lines.append(f"| {label} | {row['object_pose_count']} | {old['support_pose_count']}→{row['support_pose_count'] if row['support_pose_count'] is not None else '—'} | {old['reuse']}/{old['juxtaposed']}→{role} | "
            f"{number(row['no_support_bbx_l'])} | {number(core)} | {number(row['pruned_bbx_l'])} | {number(final)} | "
            f"{number(delta_core)}/{percent(final, core)} | {number(old['final_bbx_l'])} | {number(delta_old)}/{percent(final, old['final_bbx_l'])} | "
            f"{number(peer['final_bbx_l'] if peer['passed'] else None)} | {row['status']} |")
        records.append(dict(set=row['id'], object_poses=row['object_pose_count'], support_poses=row['support_pose_count'],
            reuse=row['reuse'], juxtaposed=row['juxtaposed'], baseline_support_poses=old['support_pose_count'],
            baseline_reuse=old['reuse'], baseline_juxtaposed=old['juxtaposed'], unsupported_bbx_l=row['no_support_bbx_l'],
            step4_bbx_l=core, step52_bbx_l=row['pruned_bbx_l'], step53_bbx_l=final,
            final_minus_core_l=delta_core, final_minus_old_l=delta_old, old_final_bbx_l=old['final_bbx_l'],
            control_final_bbx_l=peer['final_bbx_l'] if peer['passed'] else None, status=row['status']))
    lines += ['', '## 用时', '',
        '单位秒。Step4两版各6 worker同时运行；部分先通过的组在Step4队列仍运行时各用1 worker顺序完成'
        'Step5，其余组各用6 worker、按版本顺序后处理。单组壁钟包含校验、构造和绘图；并行负载、'
        '提前成功及缓存会影响时间。中断窗口与额外计算已经单独保留，不据此宣称PSO更快或体积最优。', '',
        '| 范围 | 差分对照 | PSO |', '| --- | ---: | ---: |',
        f"| Step4选中执行的单worker壁钟和（全部52） | {summary['control_all_case_step4_seconds']:.3f} | {summary['pso_all_case_step4_seconds']:.3f} |",
        f"| 共同成品组Step5单worker壁钟和 | {summary['matched_control_step5_seconds']:.3f} | {summary['matched_pso_step5_seconds']:.3f} |",
        f"| Evaluation与绘图 | {summary['control_evaluation_seconds']:.3f} | {summary['pso_evaluation_seconds']:.3f} |", '',
        '| Set | Step4对照→PSO | Step5对照→PSO |', '| --- | --- | --- |']
    for row in rows:
        peer = control[row['id']]
        lines.append(f"| {row['id']} | {number(peer['step4_seconds'], 1)}→{number(row['step4_seconds'], 1)} | "
                     f"{number(peer['step5_seconds'], 1)}→{number(row['step5_seconds'], 1)} |")
    lines += ['', '逐阶段时间与各实际支撑状态成员在[pso_evaluation.json](data/system_pso_comparison_all52_20261013_01/pso_evaluation.json)，'
        '对照在[differential_evaluation.json](data/system_pso_comparison_all52_20261013_01/differential_evaluation.json)。', '',
        '## 连续提案的效果', '',
        f"同组已结束的{diagnostics['summary']['paired_finished']}对实验，PSO有"
        f"{diagnostics['summary']['pso_fewer_rounds']}组轮数更少、"
        f"{diagnostics['summary']['pso_equal_rounds']}组相同、"
        f"{diagnostics['summary']['pso_more_rounds']}组更多；轮数耗尽也按实际状态保留。", '',
        '旧提案为随机供体差分加高斯位置噪声；本次PSO的整个平移提案为历史最优吸引与惯性，'
        '没有位置噪声。注册初始化与离散换座后速度为零，贪心接受又经常令个人最好等于当前布局；'
        '这些情形会使惯性项或个人吸引项消失。当前、个人和群体位置全重合时，PSO平移为零。'
        '整套布局更好也不能证明其中单个pose位置在另一套host/座位组合中更好。'
        '这是提案机制的限制，不能据此认定原受力验收变了，或将未完成布局记作成功。', '',
        '[提案轨迹统计](data/system_pso_comparison_all52_20261013_01/proposal_diagnostics.json)'
        '区分原始提案与一次调整后的接受，列出速度、零分量、原始代理损失和失败需求变化。'
        '代理损失单独下降不等同于原物理评分改善；速度数值在原地面投影和高度修复之前。'
        '[代码与配置范围核对](data/system_pso_comparison_all52_20261013_01/proposal_scope_audit.json)'
        '记录30个未改变的函数/方法，增加PSO提案与历史记忆，其他变化只在初始化记忆和分支调用。', '',
        '## 保存证据与审计', '',
        '[最终审计](data/system_pso_comparison_all52_20261013_01/final_campaign_audit.json)检查全部组初始化、'
        '固定槽位事件、任务朝向、真实主体标记、原始通过mask及输入/产物哈希；Evaluation另查实际实体'
        '连通、封闭、合法空间、主体保留和删减包含关系。523个冻结数值源码、11539个旧文件及旧图存档'
        '保持。失败与中断记录全部保留。', '',
        '[搜索说明](../../../population_soft_algorithm.md) · [批次续跑入口](../../../helper_func/run_pso_campaign.py)']
    report.write_text('\n'.join(lines) + '\n')
    csv_path = BASE / 'pso_all52_stage_bbx_20261013_01.csv'
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    for policy, experiment in POLICIES.items():
        path = BASE / 'data' / (experiment + '_sculpted') / 'batch.json'
        batch = read(path)
        batch['eligible_groups'] = read(BASE / 'data' / experiment / 'batch.json')['passed']
        save(path, batch)
    print('FINAL REPORT', report, flush=True)


if __name__ == '__main__':
    main()
