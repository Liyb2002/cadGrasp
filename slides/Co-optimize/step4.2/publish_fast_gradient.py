"""Read-only verification and gallery for independently cold-started runs."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
import argparse
import json
import html
import statistics
import numpy as np
from co_common import HERE,I,save


def fixture_state_groups(poses,hosts,active,native_world):
    """Group actual fixture transforms; a state has no two-pose capacity cap."""
    groups=[]
    for k in active:
        transform=native_world[int(hosts[k])]
        group=next((g for g in groups if np.allclose(transform,g['fixture_to_world'],
                                                    atol=1e-10,rtol=0)),None)
        if group is None:
            group=dict(state=len(groups)+1,fixture_to_world=transform.tolist(),poses=[])
            groups.append(group)
        group['poses'].append(poses[k])
    return groups


def main():
    p=argparse.ArgumentParser();p.add_argument('--batches',nargs='+',default=['fast_gradient_pilot_v2','fast_gradient_remaining_v2'])
    p.add_argument('--prefix',default='fast_gradient')
    args=p.parse_args();root=HERE/'output/B';rows=[];groups=[]
    for name in args.batches:
        batch=json.loads((root/'data'/name/'batch.json').read_text())
        for label in batch['groups']:
            if label not in groups:groups.append(label)
        for result in batch['results']:
            row=dict(result,experiment=name,budget=batch['options'],
                     own_state_continuation=bool(batch['options'].get('resume_experiment')))
            out=root/row['id']/'step4/step4.2'/name
            if row['passed']:
                report=I.check_report(out/'data/report.json')
                assert report['force_exit_work_passed']
                assert all(n==32768 for n in report['counts'].values())
                process=json.loads((out/'process.json').read_text())
                first=out/process[0]['layout'];source=root/row['id']/'step4/step4.1/layout.npz'
                with np.load(first) as start,np.load(source) as original:
                    for key in ['placements','directions','hosts','active']:
                        np.testing.assert_array_equal(start[key],original[key])
                committed=[t for state in process for t in state.get('decision',{}).get('trials',[])
                           if t.get('accepted') and 'gradient' in t.get('operation','')]
                row['committed_gradient_steps_lower_bound']=len(committed)
                row['accepted_gradient_steps_including_provisional']=row.pop('accepted_gradient_steps',0)
                row['cold_start_array_identity_checked']=True
                with np.load(out/'layout.npz') as layout:
                    states=fixture_state_groups(report['poses'],layout['hosts'],layout['active'],layout['native_world'])
                row['fixture_state_groups']=states
                row['fixture_state_count']=len(states)
                row['maximum_poses_per_state']=max(len(s['poses']) for s in states)
                save(out/'state_groups.json',dict(presentation_only=True,geometry_changed=False,
                    state_capacity_limit=None,states=states,
                    layout_sha256=I.sha256(out/'layout.npz'),report_sha256=I.sha256(out/'report.json')))
            rows.append(row)
    def origin_chain(row):
        chain=[];name=row['experiment'];seen=set()
        while name:
            assert name not in seen, 'cyclic own-state continuation'
            seen.add(name)
            batch=json.loads((root/'data'/name/'batch.json').read_text())
            result=next(r for r in batch['results'] if r['id']==row['id'])
            out=root/row['id']/'step4/step4.2'/name
            report_path=out/('report.json' if result['passed'] else 'search_report.json')
            report=json.loads(report_path.read_text()) if report_path.exists() else {}
            process_path=out/'process.json'
            process=json.loads(process_path.read_text()) if process_path.exists() else []
            gradients=sum(1 for state in process for t in state.get('decision',{}).get('trials',[])
                          if t.get('accepted') and 'gradient' in t.get('operation',''))
            chain.append(dict(experiment=name,passed=result['passed'],
                              search_seconds=report.get('search_seconds'),
                              conditioning_screen_seconds=report.get('conditioning_screen_seconds',0.),
                              committed_gradient_steps_lower_bound=gradients,
                              complete_run_seconds=result['seconds'],
                              continuation_of=batch['options'].get('resume_experiment')))
            name=batch['options'].get('resume_experiment')
        return list(reversed(chain))
    selected={}
    for row in rows:
        label=row['id'];old=selected.get(label)
        if old is None or (row['passed'] and (not old['passed'] or row['volume_cm3']<old['volume_cm3'])):selected[label]=row
    passing=[r for r in selected.values() if r['passed']]
    for row in passing:
        row['origin_chain']=origin_chain(row)
        row['origin_chain_run_seconds']=sum(r['complete_run_seconds'] for r in row['origin_chain'])
        row['origin_chain_search_seconds']=sum((r['search_seconds'] or 0.)+r['conditioning_screen_seconds']
                                              for r in row['origin_chain'])
        row['origin_chain_committed_gradient_steps_lower_bound']=sum(
            r['committed_gradient_steps_lower_bound'] for r in row['origin_chain'])
    text=['# 快速全组梯度：7–10 pose 结果','',
          f"已完成 {len(selected)}/{len(groups)} 组，原完整载荷与真实几何通过 {len(passing)}/{len(groups)}。",'',
          '所有组的原始起点是保存的 Step4.1，完整起点数组已核对。标注“接续”的实验随后加载自己的前一轮状态，额外预算独立保存；不能称为一次统一预算冷启动。方向／平移以全组锥距离为目标；有限接触分辨率的数值梯度加少量候选采样，Juxtapose 为离散跳步。', '',
          '同一个支撑可以有多个摆放 state，每个 state 可支持多个 object pose，没有两个 pose 的容量限制。state 按实际支撑世界变换分组；Juxtapose 的 guest/host 是操作参数，不是容量。', '',
          '| Pose set | 状态 | 来源链总搜索 s | 本轮检查 s | 冷启动至本轮累计 s | 实体 cm³ | 转动复用／J | state数／最大fit数 | 来源链提交梯度步下界 | 图 |',
          '|---|---|---:|---:|---:|---:|---:|---:|---:|---|'];cards=[]
    for label in groups:
        row=selected.get(label)
        if row is None:
            text.append(f'| {label} | 未完成 | — | — | — | — | — | — | — | — |');continue
        relative=f"{label}/step4/step4.2/{row['experiment']}"
        if row['passed']:
            status='通过（自身状态接续）' if row['own_state_continuation'] else '通过（冷启动）'
            text.append(f"| {label} | {status} | {row['origin_chain_search_seconds']:.1f} | {row['validation_seconds']:.1f} | {row['origin_chain_run_seconds']:.1f} | {row['volume_cm3']:.2f} | {row['rotating_reuse_pose_count']}/{row['juxtaposed_pose_count']} | {row['fixture_state_count']}/{row['maximum_poses_per_state']} | {row['origin_chain_committed_gradient_steps_lower_bound']} | [过程]({relative}/process.png) · [最终]({relative}/final_result.png) · [states]({relative}/state_groups.json) |")
            cards.append('<section><h2>'+html.escape(label)+f"</h2><p>{row['volume_cm3']:.2f} cm³ · 来源链搜索 {row['origin_chain_search_seconds']:.1f} s · {row['fixture_state_count']} states · 每个state最多fit {row['maximum_poses_per_state']}个pose</p>"+
                         ''.join(f'<a href="{relative}/{f}"><img loading="lazy" src="{relative}/{f}"></a>' for f in ['process.png','final_result.png'])+'</section>')
        else:
            text.append(f"| {label} | 未解决 | — | — | — | — | — | — | — | [记录]({relative}/run_result.json) |")
    if passing:
        times=[r['origin_chain_search_seconds'] for r in passing]
        cumulative=[r['origin_chain_run_seconds'] for r in passing]
        text+=['',f'所选方案的完整来源链搜索（含额外conditioning筛选、不含真实检查／出图）：中位数 {statistics.median(times):.1f} s，范围 {min(times):.1f}–{max(times):.1f} s。',
               f'所选方案从冷启动至本轮的累计运行（含该来源链的失败检查及出图）：中位数 {statistics.median(cumulative):.1f} s，范围 {min(cumulative):.1f}–{max(cumulative):.1f} s。这不含其他弃用实验的研发成本。']
    text+=['','梯度步列只计过程图中明确已提交的梯度步，不把被丢弃分支的局部接受步算进成功路径。分支完整统计、逐步预算和原始耗时在 JSON 中保留。', '',
           '“来源链总搜索”和“冷启动至本轮累计”均包含接续之前的本算法搜索；“本轮检查”只列最终这一轮的检查成本。其他弃用试跑不混入该来源链。候选几何超时与失败都记入相应检查耗时，没有计为通过。', '',
           '实体体积来自实际接受网格，不是包围盒。每个 pose 原 32768 个需求以及工作／退出／核心／净空检查保持原定义；整件安装、连通与强度检查范围与原实验相同。', '',
           f'[算法与公式](../../step4.2/fast_gradient_algorithm.md) · [完整数据](data/{args.prefix}_results.json) · [图片浏览]({args.prefix}_index.html)。']
    (root/f'{args.prefix}_results.md').write_text('\n'.join(text)+'\n')
    (root/f'{args.prefix}_index.html').write_text('<!doctype html><meta charset="utf-8"><title>Fast whole gradient</title><style>body{font:16px sans-serif;max-width:1500px;margin:30px auto}img{width:100%}section{margin:40px 0}</style>'+''.join(cards))
    save(root/'data'/f'{args.prefix}_results.json',dict(groups=groups,completed=len(selected),passed=len(passing),
         complete=len(selected)==len(groups),selected=selected,all_attempts=rows,batches=args.batches,
         provisional_steps_not_counted_as_committed=True))
    print('VERIFIED',len(passing),'/',len(groups),'completed',len(selected))


if __name__=='__main__':main()
