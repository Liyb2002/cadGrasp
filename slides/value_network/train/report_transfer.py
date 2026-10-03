"""Write a concise Chinese report from completed, audited results."""
import json
from pathlib import Path
from transfer_experiment import ROOT

def main():
    a=json.loads((ROOT/'audit.json').read_text());e=json.loads((ROOT/'experiment.json').read_text());c=a['comparison'];p=e['split']
    lines=['# 五组训练与组合泛化实验','',
           '先冻结原来的十组记忆网络，测试十个新组合；随后仅用其中五组重新生成条件 value 数据，从随机权重训练同一完整 state-mask 网络。另外五组和 baseline 原有十组只用于训练结束后的测试。','',
           '| 实验 | 实际 Step3 成功 | 代价距最好记录解 ≤0.05 |','|---|---:|---:|',
           '| 原网络在原来训练组合上 | 10/10 | 10/10 |',
           f"| 原网络冻结后，在十个新组合上 | {c['frozen']['passed']}/10 | {c['frozen']['near_best_count']}/10 |",
           f"| 五组训练网络，在训练五组上 | {c['five_train']['passed']}/5 | {c['five_train']['near_best_count']}/5 |",
           f"| 五组训练网络，在新留出五组上 | {c['five_test']['passed']}/5 | {c['five_test']['near_best_count']}/5 |",
           f"| 五组训练网络，在 baseline 原有十组上 | {c['original_ten_test']['passed']}/10 | {c['original_ten_test']['near_best_count']}/10 |",'',
           '[结果图](results.png) · [完整审查](audit.json) · [模型和实验配置](experiment.json)','',
           '结论：五组训练后的网络有一定的硬约束通过能力迁移，但还没有稳定迁移“少用头＋退出方向接近”的优化能力。仅凭成功率不能说泛化已经解决。0.05 的阈值沿用事先规定的训练停止条件，不是看测试结果后设置。','',
           '## 数据与边界','',
           f"初始五组 × 20 state = 100 个 state；之后只在这五组补采实际搜索状态，以及已验证最好解的随机插入前缀。最终训练实际读取 {a['actual_training_states']} 个去重 state、{a['finite_training_values']} 个有限条件 value，训练／补采共 {a['training_rounds']} 轮。不是仅用 100 个 state 得出的结果。",
           '',
           '标签复用相同的单 pose 已验证解库，重新计算这五种组合的 Q(S,h)=N_remaining+D_final；预算内没有见证的动作保留 null。未把最终解、剩余头数或方向代价放进网络输入；已有单 pose 物理证据可共享。新组合的测试数据没有进入训练，测试成绩也没有用于挑选模型。',
           '',
           '所有新组合都使用原网络已有的 15 个 pose／200 个头身份，五组训练覆盖全部 15 个 pose。本实验测组合泛化，不测新物体、新 pose 或新候选头的几何泛化。baseline 现有的是原来的十个组合，并没有另外十个已保存组合；新十组在运行前固定划分，详见 plan.json。',
           '',
           '原有十组中 pose1+3 和 pose1+3copied 使用相同当前任务输入，等价于九个独立组合；原有十组结果按十个目录展示。',
           '',
           '## 各组结果','',
           '| 集合 | pose 组合 | Step3 | 已选头数 | 成功终局相对最好记录解的代价差 |',
           '|---|---|---|---:|---:|']
    for key,label in [('five_train','训练'),('five_test','留出'),('original_ten_test','原有组合留出')]:
        for r in c[key]['results']:
            gap=f"{r['cost_gap']:.5f}" if r['passed'] and r.get('cost_gap') is not None else '—'
            lines.append(f"| {label} | {r['group']} | {'PASS' if r['passed'] else 'FAIL'} | {r['heads']} | {gap} |")
    lines += ['',
              '失败行的头数是预算内尝试数量，不是可用解。成功意味着各 pose 的全部 32,768 条原始载荷通过联合受力及整体不上抬，并保持共同方向与连通分量；中间力学不满足不会直接判死。没有构造这些接触的 Step4 支撑实体。最好解仅指记录解库中的最好结果，不是全局最优保证。','',
              '## 代码与重跑','',
              '- `../transfer_experiment.py frozen`：冻结原模型测试十个新组合。',
              '- `../../data_producer/collect_transfer.py`：只生成五组初始数据。',
              '- `../seed_transfer_prefixes.py`：只补采训练组成功解的随机插入前缀。',
              '- `../improve_five.py`：从随机权重训练，按训练组 rollout 选择停止时机，最后测试留出组。',
              '- `../transfer_audit.py` 和 `../render_transfer.py`：结束后审查边界、比较记录解并生成图。','',
              f"最终 checkpoint：`{e['checkpoint']}`。原模型 current/best.pt 和原 baseline 的受保护输入均未修改。"]
    (ROOT/'README.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
