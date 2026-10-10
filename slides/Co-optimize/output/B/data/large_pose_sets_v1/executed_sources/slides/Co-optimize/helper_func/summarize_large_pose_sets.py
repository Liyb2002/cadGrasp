"""Compare old nominal meshes with new checked meshes and their nominal forms."""
import _bootstrap
import argparse
import html
from co_common import *
from run_all import output_root
from run_large_pose_sets import groups, publish, EXPERIMENT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args(); root=output_root('B'); requested=groups()
    rows=publish(root,requested)
    if not args.allow_partial and not all(row['complete'] for row in rows):
        raise RuntimeError('Not all ten sets have both completed methods')
    initial_path=root/'data'/EXPERIMENT/'initial_budget_results.json'
    initial={r['id']:r for r in json.loads(initial_path.read_text())}
    cards=[]
    lines=['# 同一批7–10 pose集合的体积对照', '',
        '原六个小集合的减少百分比是相对其新版Step4.2可行支撑，未与旧pose_set_search比较。'
        '本表使用旧实验完全相同的十个集合，全部从新的Step3整组初始化独立求解。', '',
        '旧whole和incremental列是已保存的名义三角实体。新名义列是新版选中布局的名义网格，'
        '用于比较相近几何构造口径；新真实列是同一布局最终通过全部原始载荷、完整工作禁区、退出及接触核/1%净空检查的实体。'
        '原本没有对旧名义网格做这些新版最终检查，不能据此认为它们已通过。单位均为cm³，均非包围盒。', '',
        '| 集合 | 旧whole名义 | 旧incremental名义 | 新greedy真实 | 新beam真实 | 新选中名义 | 新选中真实 | 选择 |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for row in rows:
        new=row['new_actual_cm3'];best=row['selected_method'];old=row['old_nominal_cm3']
        row['initial_budget_passed']=initial[row['id']]['status']=='pass'
        row['separately_recorded_continuation']=not row['initial_budget_passed']
        nominal=None
        if best:
            report_path=root/row['report_paths'][best]
            out=report_path.parent.parent
            render=I.check_report(out/'data/render.json')
            # The selected-path endpoint is the final chosen layout; render
            # records its nominal mesh separately from the appended REAL tile.
            steps=render['process_states']
            with np.load(out/'layout.npz') as final:
                for endpoint in reversed(steps):
                    with np.load(out/endpoint['layout']) as z:
                        matches=all(np.array_equal(z[field],final[field])
                                    for field in ['placements','directions','hosts','active'])
                    if matches:
                        nominal=endpoint['nominal_material_volume_cm3'];break
            if nominal is None:
                # Production feasibility may select an earlier checked
                # finalist that is absent from the displayed accepted path.
                # Reconstruct only its nominal volume, never another search.
                from whole_search.fast_search import FastModel
                from whole_step4_render import NominalBuilder, read_layout
                model=FastModel(row['poses'],'B',initialization_report=root/row['id']/'step3/step3.1/data/report.json')
                nominal=material_volume(NominalBuilder(model).solid(read_layout(out/'layout.npz')))*1e6
            row['selected_nominal_cm3']=nominal
            row['nominal_ratio_to_old_whole']=nominal/old['whole'] if nominal is not None else None
            row['nominal_ratio_to_best_old']=nominal/min(old.values()) if nominal is not None else None
            row['actual_ratio_to_old_whole']=row['selected_actual_cm3']/old['whole']
            audit=root/row['id']/'data'/EXPERIMENT/'audit.json'
            row['independent_audit']=json.loads(audit.read_text()) if audit.exists() else None
            save(root/row['id']/'data'/EXPERIMENT/'best.json',row)
            images=''.join('<a href="'+row[key]+'"><img loading="lazy" src="'+row[key]+'"></a>'
                           for key in ['process_image','final_image'])
            cards.append('<section><h2>'+html.escape(row['id'])+'</h2><p>'+
                html.escape(f"旧whole / incremental名义：{old['whole']:.2f} / {old['incremental']:.2f} cm³；新版名义 / 真实：{nominal:.2f} / {row['selected_actual_cm3']:.2f} cm³；选择 {best}。")+
                '</p><p>'+('主预算通过' if row['initial_budget_passed'] else '追加自身布局搜索后通过')+
                '；'+' · '.join('<a href="'+str((Path(row['report_paths'][m]).parent.parent/'process.png'))+'">'+m+'过程</a> / '
                                '<a href="'+str((Path(row['report_paths'][m]).parent.parent/'final_result.png'))+'">最终形状</a>'
                                for m in ['greedy','beam'] if m in row['report_paths'])+
                '</p>'+images+'</section>')
        def fmt(value):return f'{value:.2f}' if value is not None else '—'
        lines.append(f"| {row['id']} | {fmt(old['whole'])} | {fmt(old['incremental'])} | {fmt(new.get('greedy'))} | {fmt(new.get('beam'))} | {fmt(nominal)} | {fmt(row['selected_actual_cm3'])} | {best or '未决'} |")
    comparable=[r for r in rows if r.get('selected_nominal_cm3') is not None]
    stats=dict(requested_sets=len(rows),completed_both_methods=sum(r['complete'] for r in rows),
        checked_feasible_sets=sum(r['selected_method'] is not None for r in rows),
        independently_audited_sets=sum(bool(r.get('independent_audit',{}).get('passed')) for r in rows if r.get('independent_audit')),
        old_whole_nominal_total_cm3=sum(r['old_nominal_cm3']['whole'] for r in comparable),
        old_best_nominal_total_cm3=sum(min(r['old_nominal_cm3'].values()) for r in comparable),
        new_selected_nominal_total_cm3=sum(r['selected_nominal_cm3'] for r in comparable),
        new_selected_actual_total_cm3=sum(r['selected_actual_cm3'] for r in comparable),
        nominal_at_most_old_whole_sets=sum(r['selected_nominal_cm3']<=r['old_nominal_cm3']['whole'] for r in comparable),
        nominal_at_most_old_best_sets=sum(r['selected_nominal_cm3']<=min(r['old_nominal_cm3'].values()) for r in comparable),
        initial_budget_passed_sets=sum(r['initial_budget_passed'] for r in rows),
        continuation_required_sets=sum(r['separately_recorded_continuation'] for r in rows),
        greedy_selected_sets=sum(r['selected_method']=='greedy' for r in rows),
        beam_selected_sets=sum(r['selected_method']=='beam' for r in rows),
        baseline_actual_total_cm3=sum(r['new_actual_cm3']['baseline'] for r in comparable))
    save(root/'data'/EXPERIMENT/'comparison.json',dict(rows=rows,statistics=stats,
        old_geometry_rule='nominal',new_geometry_rule='actual contact core and exit clearance',
        nominal_new_is_same_accepted_layout=True,same_original_object_poses_work_angles_and_loads=True,
        initial_budget_report=str(initial_path.relative_to(root)),
        old_warm_start_used_in_published_solutions=False))
    lines += ['', f"已完成两条优化路线 {stats['completed_both_methods']}/{len(rows)}；至少一个真实可行结果 {stats['checked_feasible_sets']}/{len(rows)}。",
        '同一实际可行基线分别做greedy与beam，各6轮、每轮96个材料筛选、48个完整原载荷候选检查、3个真实网格候选。'
        '最终选最小真实体积，包含保留基线；两条路占两个搜索预算。初始化数值重试与最终数值续跑单独留档。', '',
        '过程图中绿增红减蓝保留，末格与最终图为真实接受网格；图内无文字，原工作面为橙色。'
        '旧实验、原42组初始化/支撑与14个小集合优化输出均保持原样。']
    lines += ['', '## 成功率与追加预算', '',
        f"主预算仅 {stats['initial_budget_passed_sets']}/{len(rows)} 通过；其余四组从各自保存的新版布局续跑。最终成功率不能写成同预算10/10。"
        '所有pose始终共同参与，未采用逐个加入、互不重叠远距摆放或降低原承载要求。', '',
        '- 七pose分散组：实际接触反例发现搜索点过于乐观；保守剔除25个查询点，并修正反馈后的接触/载荷缓存失效。仅改变搜索指导，最终真实接触和原载荷保持原定义。成功续跑20次全载荷候选检查、1次实际候选；此前失败尝试也留档。',
        '- 九pose连续组：追加12轮frontier、6个布局分支、4轮分支修复、每轮128个材料筛选，另有细方向和3轮减材料；241次全载荷候选、2次实际检查。',
        '- 十pose连续组：上述追加搜索得到采样可行解，但3次实际网格候选超时。再检查13个微小方向候选，其中0.03125°向上偏置的实际候选通过；发布已完成检查的同一网格，没有重复搜索。',
        '- 十pose分散组：追加8轮frontier/6布局/4修复/128筛选；阻挡pose按其锁住接触对最远需求的供力贡献排序。604次全载荷候选、1次实际检查。', '',
        '额外私有诊断均保留在各组目录：九pose的13个微小方向候选产生另一个真实可行解，但未替换已完成基线；'
        '十pose分散组也测试了旧whole布局热启动的3个实际候选，均未通过新版导出边界检查，全部未被用于本表。'
        '这些属于额外实验预算，不能隐藏成主预算的成功。', '',
        '## 体积结论', '',
        f"以相近构造口径的名义网格比较，新版 {stats['nominal_at_most_old_whole_sets']}/{len(comparable)} 组不大于旧whole，"
        f"{stats['nominal_at_most_old_best_sets']}/{len(comparable)} 组不大于旧whole/incremental的较小者。"
        f"新版名义体积合计 {stats['new_selected_nominal_total_cm3']:.3f} cm³，旧whole {stats['old_whole_nominal_total_cm3']:.3f} cm³，"
        f"旧两方法逐组较小者 {stats['old_best_nominal_total_cm3']:.3f} cm³。", '',
        f"相对新版自身真实可行基线，所选真实体积合计从 {stats['baseline_actual_total_cm3']:.3f} 降至 "
        f"{stats['new_selected_actual_total_cm3']:.3f} cm³。Greedy选中{stats['greedy_selected_sets']}组，beam选中{stats['beam_selected_sets']}组。"
        '连续8/9/10 pose的紧凑程度仍落后旧实验，当前不能断言新版统一达到旧算法的体积表现，也不能断言beam总是更好。']
    (root/'large_pose_sets_comparison.md').write_text('\n'.join(lines)+'\n')
    (root/'large_pose_sets_index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>7–10 pose新版whole</title>'
        '<style>body{max-width:1600px;margin:24px auto;padding:20px;font-family:system-ui}img{max-width:100%}section{margin:45px 0}h2{overflow-wrap:anywhere}</style>'
        '<h1>同一批7–10 pose集合：新版whole</h1><p><a href="large_pose_sets_comparison.md">详细体积和比较口径</a></p>'+''.join(cards)+'</html>')
    print('LARGE SUMMARY',stats,flush=True)


if __name__=='__main__':main()
