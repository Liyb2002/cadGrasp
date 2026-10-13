"""Select the smallest CHECKED member of each group's solution portfolio."""
import _bootstrap
import argparse,html
from co_common import *
from run_all import saved_groups,output_root
from run_compact_volume import collect


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('object',nargs='?',default='B')
    args=parser.parse_args();root=output_root(args.object);all_rows=collect(args.object)
    ids={r['id'] for r in all_rows};rows=[];cards=[]
    lines=['# Whole：真实可行结果的材料体积对照','',
        '每组全部pose共同参与。各方法从同一个实际可行Step4.2基线出发，体积取支撑实体。'
        'greedy 为继续 Direction／Translation；beam 另保留落座替代和修复分支。'
        'structural-greedy 为额外消融：允许换落座，但只延续一个可行布局。','',
        'greedy、beam 各用至多48次原全部载荷候选评价、6轮／每轮96个材料筛选、3次实际候选。'
        '两个方法一起运行使用两份预算；消融再加一份预算。实际未决／不合格候选保留原记录，不能按采样体积发布。'
        '四个初版方法已完成搜索与真实检查后遇到保存字段冲突，从相同检查点恢复发布；没有重跑搜索或额外几何候选。','',
        '| Pose set | 初始真实基线 cm³ | greedy cm³ | beam cm³ | structural-greedy cm³ | 最小 cm³ | 减少 | 选中 |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for group in saved_groups(args.object):
        if group['id'] not in ids:continue
        source=root/group['id']/'step4/step4.2';baseline=I.check_report(source/'data/report.json')
        candidates=[dict(method='baseline',volume_cm3=baseline['volume_cm3'],out=source)]
        by_method={}
        for mode in ['greedy','beam','structural-greedy']:
            out=source/'compact'/mode
            if not (out/'data/report.json').exists():continue
            report=I.check_report(out/'data/report.json')
            assert report['passed'] and report['force_exit_work_passed']
            assert report['original_baseline_report_sha256']==I.sha256(source/'data/report.json')
            by_method[mode]=report
            candidates.append(dict(method=mode,volume_cm3=report['volume_cm3'],out=out))
        best=min(candidates,key=lambda c:c['volume_cm3']);out=source/'compact';out.mkdir(exist_ok=True)
        primary=min([c for c in candidates if c['method']!='structural-greedy'],key=lambda c:c['volume_cm3'])
        gain=1-best['volume_cm3']/baseline['volume_cm3']
        portfolio_inputs=[c['out']/'data/report.json' for c in candidates]
        row=dict(id=group['id'],pose_count=len(group['poses']),baseline_cm3=baseline['volume_cm3'],
            final_cm3=best['volume_cm3'],saved_fraction=gain,selected_method=best['method'],
            primary_two_search_final_cm3=primary['volume_cm3'],
            primary_two_search_selected_method=primary['method'],
            primary_two_search_saved_fraction=1-primary['volume_cm3']/baseline['volume_cm3'],
            selected_report=str((best['out']/'data/report.json').relative_to(root)),
            selected_process=str((best['out']/'process.png').relative_to(root)),
            selected_result=str((best['out']/'final_result.png').relative_to(root)),
            methods={mode:dict(volume_cm3=r['volume_cm3'],budget=r['budget'],
                final_validation_attempts=r['final_validation_attempts']) for mode,r in by_method.items()})
        rows.append(row)
        save(out/'best.json',dict(complete=True,**row,
            baseline_retained_as_upper_bound=True,all_original_force_exit_work_constraints_preserved=True,
            full_fixture_accepted=False,provenance=provenance(portfolio_inputs,[Path(__file__)])))
        def value(mode):return f"{by_method[mode]['volume_cm3']:.3f}" if mode in by_method else '—'
        lines.append(f"| {group['id']} | {baseline['volume_cm3']:.3f} | {value('greedy')} | {value('beam')} | "
            f"{value('structural-greedy')} | {best['volume_cm3']:.3f} | {100*gain:.2f}% | {best['method']} |")
        linkdir=Path(os.path.relpath(best['out'],out))
        (out/'README.md').write_text('# 本组体积搜索\n\n'
            f"实际可行基线 {baseline['volume_cm3']:.3f} → 当前最小 {best['volume_cm3']:.3f}cm³，减少 {100*gain:.2f}%；"
            f"选中 `{best['method']}`。\n\n"
            f'[选中过程]({linkdir}/process.png) · [选中最终支撑]({linkdir}/final_result.png) · [全部结果](best.json)\n\n'
            '所有方法原始结果均保留，各候选与实际检查可追溯；不能将两个搜索的总预算说成一个搜索的预算。\n')
        title=html.escape(group['id']);result=html.escape(row['selected_result']);process=html.escape(row['selected_process'])
        cards.append(f'<article><h2>{title}</h2><p>{baseline["volume_cm3"]:.3f} → {best["volume_cm3"]:.3f} cm³；'
            f'减少 {100*gain:.2f}%；{best["method"]}。'
            f'<a href="{process}">过程</a> · <a href="{result}">最终图</a> · '
            f'<a href="{group["id"]}/step4/step4.2/compact/README.md">各方法及记录</a></p>'
            f'<a href="{result}"><img loading="lazy" src="{result}"></a></article>')
        readme=source/'README.md';text=readme.read_text();marker='\n<!-- compact-volume-results -->\n'
        text=text.split(marker)[0]
        readme.write_text(text+marker+'\n真实体积优化继续结果：[最小支撑与所有方法](compact/README.md)。'
            f"已检查最小材料 {best['volume_cm3']:.3f}cm³（原基线 {baseline['volume_cm3']:.3f}cm³）。"
            '本目录原基线文件完整保留，新网格和图见选中方法目录。\n')
    paired=[r for r in rows if 'greedy' in r['methods'] and 'beam' in r['methods']]
    stats=dict(groups=len(rows),complete_greedy_beam_pairs=len(paired),
        improved_groups=sum(r['saved_fraction']>1e-8 for r in rows),
        baseline_total_cm3=sum(r['baseline_cm3'] for r in rows),
        selected_total_cm3=sum(r['final_cm3'] for r in rows),
        unweighted_mean_saved_fraction=float(np.mean([r['saved_fraction'] for r in rows])) if rows else 0.,
        beam_beats_greedy=sum(r['methods']['beam']['volume_cm3']<r['methods']['greedy']['volume_cm3']-1e-4 for r in paired),
        greedy_beats_beam=sum(r['methods']['greedy']['volume_cm3']<r['methods']['beam']['volume_cm3']-1e-4 for r in paired))
    stats.update(primary_two_search_total_cm3=sum(r['primary_two_search_final_cm3'] for r in rows),
        primary_two_search_mean_saved_fraction=float(np.mean([r['primary_two_search_saved_fraction'] for r in rows])) if rows else 0.,
        extra_ablation_groups=sum('structural-greedy' in r['methods'] for r in rows))
    if rows:stats['volume_weighted_saved_fraction']=1-stats['selected_total_cm3']/stats['baseline_total_cm3']
    lines+=['',f"完成 {len(paired)} 组 greedy／beam 对照；beam 更小 {stats['beam_beats_greedy']} 组，"
        f"greedy 更小 {stats['greedy_beats_beam']} 组。最终选择各已检查答案的最小值，未证明全局最优。",'',
        '[图集](compact_index.html) · [逐方法原始表](compact_results.md) · [论文算法](../../step4.2/paper_algorithm.md)', '',
        '独立审计见 data/whole_compact_audit.json。仍未做整件接地、连通与强度验收。'
        '不把旧 pose_set_search 名义支撑体积与这里的接触核心／1%净空实际体积直接比较。']
    (root/'compact_comparison.md').write_text('\n'.join(lines)+'\n')
    save(root/'data/whole_compact_portfolio.json',dict(complete=True,rows=rows,stats=stats,
        full_fixture_accepted=False,provenance=provenance([root/'data/whole_compact_comparison.json'],[Path(__file__)])))
    (root/'compact_index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>Whole 支撑体积优化</title>'
        '<style>body{max-width:1400px;margin:24px auto;padding:20px;font-family:system-ui;color:#253441}'
        'article{border:1px solid #dde4e9;padding:16px;margin:20px 0}img{max-width:100%}</style>'
        '<h1>Whole 共享支撑：实际材料体积</h1><p>原基线、greedy、beam 中实际可行的最小支撑；原始承载／工作禁区／退出约束保持。'
        '额外 structural-greedy 消融单独记录。全部中间分支保留，图内没有文字；接地、连通与强度留到后续。'
        '<a href="compact_comparison.md">体积对照</a></p>'+''.join(cards)+'</html>')
    print('COMPACT PORTFOLIO',stats,flush=True)


if __name__=='__main__':main()
