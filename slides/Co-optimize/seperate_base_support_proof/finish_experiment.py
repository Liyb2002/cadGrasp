"""Publish the completed all-set batch and audit saved masks and floor hulls."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path
import re
import shutil
import sys

PROOF = Path(__file__).resolve().parent
sys.path.insert(0,str(PROOF))
import diagnose as D
C, np = D.C, D.np


def main():
    spec = json.loads((PROOF/'experiment.json').read_text())
    batch = json.loads((PROOF/'summary.json').read_text())
    assert batch['complete'] and batch['protected_existing_unchanged']
    attempted = {r['id']:r for r in batch['step4_attempts']}
    passed = {r['case']['id']:r for r in batch['step5_results']}
    assert set(attempted)=={g['id'] for g in spec['groups']}
    certificates = {p.parent.relative_to(PROOF/'certificates').as_posix():json.loads(p.read_text())
                    for p in (PROOF/'certificates').rglob('report.json')}
    rows, protected, points, mask_count = [],{},0,0
    upstream_sources, actual_search_sources = {}, {}
    for group in spec['groups']:
        label = group['id']
        result = attempted[label]
        row = dict(id=label,pose_count=group['pose_count'],fresh_step4_passed=result['passed'],
                   phase=result['phase'],material_volume_cm3=result.get('volume_cm3'),
                   process_bbx_cm3=None,ground_impossibility_proved=None,
                   unbounded_work_impossibility_proved=None,diagnostic_scope='not tested')
        if label in passed:
            case = passed[label]['case']
            inputs = C.Inputs('B',D.BASE/label/'data/selection.json')
            inputs.base = D.BASE
            data = inputs.load(case)
            upstream_sources.update(inputs.protected)
            search_report = json.loads((data['selected']/'report.json').read_text())
            for path,expected in search_report['executed_search_sources_sha256'].items():
                if path in actual_search_sources:
                    assert actual_search_sources[path]==expected,path
                actual_search_sources[path] = expected
            for pose in group['poses']:
                with np.load(data['selected']/f'{pose}_force.npz',allow_pickle=False) as saved:
                    mask = saved['mask']
                    assert mask.shape==(32768,) and mask.all(),(label,pose)
                    mask_count += len(mask)
            for entry in data['record']['per_pose']:
                file = D.BASE/label/'step5/step5.1'/entry['arrays']
                with np.load(file,allow_pickle=False) as saved:
                    xy = saved['floor_demands_xy_m']
                    assert xy.shape==(32768,2)
                    expected = np.c_[-saved['wrench_world_origin'][:,4],saved['wrench_world_origin'][:,3]]
                    expected /= saved['wrench_world_origin'][:,2,None]
                    assert np.max(np.abs(expected-xy))<1e-12,(label,entry['pose'])
                    errors = xy@saved['hull_equations_xy'][:,:2].T+saved['hull_equations_xy'][:,2]
                    assert errors.max()<1e-10,(label,entry['pose'])
                    points += len(xy)
            saved_box = json.loads((D.BASE/label/'step5/step5.2/data/report.json').read_text())
            assert np.max(np.abs(data['bounds'][0]-saved_box['bounding_box_min_m']))<1e-12
            assert np.max(np.abs(data['bounds'][1]-saved_box['bounding_box_max_m']))<1e-12
            row['process_bbx_cm3'] = saved_box['bounding_box_volume_cm3']
        if label in certificates:
            cert = certificates[label]
            row.update(ground_impossibility_proved=cert['ground_rejected'],
                       unbounded_work_impossibility_proved=cert['unbounded_work_rejected'],
                       diagnostic_scope='unbounded necessary certificates; absent proof does not imply feasibility')
        rows.append(row)
    for path,expected in upstream_sources.items():
        assert C.digest(C.ROOT/path)==expected,path
    assert points==mask_count
    # All current numeric sources are checked against the actual cold/repair
    # execution archives. This never resolves the original force demands.
    numeric_sources = {}
    for name in ['proof_cold_v1','proof_own_repair_v1','step51']:
        directory = D.BASE/'data'/name
        manifest = json.loads((directory/'source_manifest.json').read_text())
        for path,expected in manifest['code'].items():
            assert C.digest(C.ROOT/path)==expected,path
            assert C.digest(directory/'executed_sources'/path)==expected,path
            numeric_sources[path] = expected
    # The original model records hashes when its R6 modules are imported
    # lazily. Preserve those exact, recorded bytes too; the pre-run objects
    # protection ledger independently proves they did not change.
    late = PROOF/'additional_executed_sources'
    late.mkdir(exist_ok=False)
    late_manifest = {}
    for path,expected in actual_search_sources.items():
        assert C.digest(C.ROOT/path)==expected,path
        if path not in numeric_sources:
            target = late/path
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(C.ROOT/path,target)
            assert C.digest(target)==expected,path
            late_manifest[path] = expected
        numeric_sources[path] = expected
    C.save(late/'manifest.json',dict(code=late_manifest,
        origin='actual executed_search_sources_sha256 recorded by the original model',
        source_and_snapshot_hashes_match=True))
    for run in (PROOF/'certificate_runs').iterdir():
        manifest = json.loads((run/'execution.json').read_text())
        assert manifest['complete'] and manifest['source_snapshots_match']
        for path,expected in manifest['code'].items():
            assert C.digest(C.ROOT/path)==expected,path
            assert C.digest(run/'executed_sources'/path)==expected,path
    recovery = D.BASE/'data/batch_recovery_v1'
    if recovery.exists():
        manifest = json.loads((recovery/'verification.json').read_text())
        assert manifest['complete'] and manifest['source_snapshots_match']
        for path,expected in manifest['code'].items():
            assert C.digest(C.ROOT/path)==expected,path
            assert C.digest(recovery/'executed_sources'/path)==expected,path
    with (PROOF/'all_sets.csv').open('w',newline='') as file:
        writer = csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    ground_count = sum(c['ground_rejected'] for c in certificates.values())
    work_count = sum(c['unbounded_work_rejected'] for c in certificates.values())
    both = [c['id'] for c in certificates.values() if c['ground_rejected'] and c['unbounded_work_rejected']]
    verification = dict(complete=True,group_count=len(rows),fresh_step4_pass_count=len(passed),
        complete_step51_count=len(passed),complete_step52_count=len(passed),
        saved_force_mask_demand_count=mask_count,saved_floor_point_count=points,
        every_selected_original_force_mask_passed=True,floor_hull_and_wrench_checks_passed=True,
        numerical_sources_and_execution_snapshots_match=True,numeric_source_count=len(numeric_sources),
        old_objects_and_results_unchanged=True,protected_existing_count=batch['protected_existing_count'],
        audited_upstream_count=len(upstream_sources),force_solver_replayed=False,
        diagnostic_4_to_6_pose_count=len(certificates),ground_rejection_count=ground_count,
        unbounded_work_rejection_count=work_count,both_rejection_sets=both,
        main_demo='pose1+2+3+4+5+6',experiment_wall_seconds=batch['seconds'])
    original_files = json.loads((PROOF/'protected_existing.json').read_text())
    old_object_files = {p for p in original_files if p.startswith('objects/')}
    now_object_files = {C.relative(p) for p in (C.ROOT/'objects').rglob('*') if p.is_file()}
    added_object_files = sorted(now_object_files-old_object_files)
    assert all('__pycache__' in Path(p).parts and p.endswith('.pyc') for p in added_object_files)
    verification['runtime_object_bytecode_cache_files_added'] = added_object_files
    verification['native_object_files_added_removed_or_changed'] = False
    verification['cold_workers_rerun'] = False
    verification['ledger_recovered_after_counts_format_error'] = batch.get('ledger_recovered',False)
    C.save(PROOF/'verification.json',verification)
    cards51,cards52 = [],[]
    for label,record in passed.items():
        escaped = html.escape(label)
        cards51.append(f'<section><h2>{escaped}</h2><a href="{label}/step5/step5.1/index.html">各 pose 数据</a>'
                       f'<img src="{label}/step5/step5.1/floor_demands.png"></section>')
        cards52.append(f'<section><h2>{escaped}</h2><a href="{label}/step5/step5.2/index.html">包围盒数据</a>'
                       f'<img src="{label}/step5/step5.2/occupied_volume.png"></section>')
    style = '<meta charset="utf-8"><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:24px auto;padding:0 18px}img{width:100%}</style>'
    (D.BASE/'step5.1_index.html').write_text('<!doctype html>'+style+'<h1>本轮 Step5.1：全部原始需求与凸包</h1>'+''.join(cards51))
    (D.BASE/'step5.2_index.html').write_text('<!doctype html>'+style+'<h1>本轮 Step5.2：全部过程包围盒</h1>'+''.join(cards52))

    def table(selected):
        content = ''
        for row in selected:
            label = html.escape(row['id'])
            stage = f'<a href="output/B/{label}/step5/step5.1/index.html">5.1</a> / <a href="output/B/{label}/step5/step5.2/index.html">5.2</a>' if row['fresh_step4_passed'] else '—'
            ground = '证实' if row['ground_impossibility_proved'] else '未证实' if row['ground_impossibility_proved'] is not None else '—'
            work = '证实' if row['unbounded_work_impossibility_proved'] else '未证实' if row['unbounded_work_impossibility_proved'] is not None else '—'
            cert = f'<a href="certificates/{label}/report.json">证书</a>' if row['ground_impossibility_proved'] is not None else '—'
            content += f'<tr><td>{label}</td><td>{row["pose_count"]}</td><td>{"PASS" if row["fresh_step4_passed"] else "FAIL"}</td><td>{stage}</td><td>{ground}</td><td>{work}</td><td>{cert}</td></tr>'
        return '<table><tr><th>Set</th><th>Pose 数</th><th>新 Step4</th><th>后续步骤</th><th>地面无解</th><th>工作锥无解</th><th>数据</th></tr>'+content+'</table>'
    four = [row for row in rows if 4<=row['pose_count']<=6]
    page = '<!doctype html><html lang="zh">'+style+'''<style>table{border-collapse:collapse;font-size:14px}th,td{border:1px solid #dce2e5;padding:7px 9px}.hero{background:#edf6fa;padding:16px}</style>
<h1>Separate base / support proof</h1>'''
    page += f'<p>本轮全部 {len(rows)} 组重新运行当前 Step4：{len(passed)} 组通过；通过的组均运行原 Step5.1、5.2。原 objects 输入和旧结果全部保留。</p>'
    page += '<div class="hero"><h2><a href="demo/index.html">主示例：同一组 6 个 pose 的两个失败</a></h2><p>pose1+2+3+4+5+6：Step4 全承载通过；地面证书 20.481 mm，工作锥证书 9.919 mm。两份证书都允许底座无限增大和稀疏接地。</p><a href="demo/index.html"><img src="demo/ground_after.png" style="max-width:450px"></a><a href="demo/index.html"><img src="demo/work_certificate.png" style="max-width:450px"></a></div>'
    page += '<p><a href="output/B/step5.1_index.html">全部 Step5.1 图</a> · <a href="output/B/step5.2_index.html">全部 Step5.2 图</a> · <a href="all_sets.csv">全组表格</a> · <a href="verification.json">核验记录</a> · <a href="explanation.txt">复现与定义</a></p>'
    page += f'<h2>全部 4–6 pose 组</h2><p>{len(certificates)} 组进行了无界几何证书诊断；{ground_count} 组有地面矛盾，{work_count} 组有工作锥矛盾，{len(both)} 组同时有两者。未证实表示本诊断没有给出证明，不代表底座可行。</p>'+table(four)
    page += '<details><summary>查看全部 52 组</summary>'+table(rows)+'</details>'
    page += '<p>本证明只针对保存的共享布局和安装状态，不否定重新联合优化相对位姿或高度后可能出现的解。正式 Step4 / Step5 算法没有修改。</p></html>'
    (PROOF/'index.html').write_text(page)
    explanation = f'''本次目的
在正式修改算法前，用已有 Step4 算法的全批次新结果记录后续共享底座的两个失败。

运行范围
对象 B，全部 52 组保存的 Step4.1 初始化；其中 30 组为 4–6 pose。
从原 Step4.1 冷启动，10 次原算法迭代；仅对仍有未满足需求的本轮布局按原流程做 8 次 own-state continuation。
6 个进程，每进程 4 个 CPU；demand_nodes=32。PASS 对每个 pose 的全部原始 32768 个需求判断。
直接调用现有 run_stable_gradient.worker / run_fast_state_seat_gradient.worker，数值源文件不修改。
唯一重定向是入口的输出根目录；读取复制的原 Step3/4.1，数值模块仍使用原 native 对象、工作锥和负载。
不使用旧优化结果作初值。本目录只展示本轮的新结果；原最小材料选择及其全部旧文件保留。
Step5.1 调用原 compute_case，Step5.2 调用原 main，仅提供独立实验的目录和选择记录。

完成情况
Step4 新通过 {len(passed)}/{len(rows)}，这些组均完成 Step5.1、5.2。
共保存并检查 {points:,} 个原始压力中心；所有原始 force mask 仍 PASS，不重新求解承载。
{batch['protected_existing_count']} 个已有 objects/Co-optimize 输出文件哈希保持不变。

主例
pose1+2+3+4+5+6，同一组六个 pose，两种错误均有不依赖 BBX 的必要条件反例。
1. Pose 4 的底座地面接点必须在 Pose 5 的地面上方；物体自身接点也都满足同一个线性不等式。
   一个原始需求违反这个不等式 20.481282780 mm。因此任何形状/大小/数量的接点都无法完成固定布局覆盖。
   示意底座在 Pose 4 地面上方；进入 Pose 5 的安装状态后最低点为 -22.223326854 mm。
2. 在 Pose 1 的地面，不考虑其他地面、连接、厚度、物体或退出路径，仅保留原工作锥。
   所有 250 mm 半径以外的接地点均被工作锥占据（解析方向圆覆盖证书）。圆内全部自由接地点与物体接地点的凸包仍缺少需求 9.918599245 mm。
   工作证明用整个原三角形加原 30 度圆锥的严格内接子集，向内缩 0.1 微米；允许的空间比真实问题更多。
   每个方向区间证明整条 r>=R 的射线在原工作锥内，而不是对远处采样后猜测。
   额外检查了 2048 个方向和 R、100R、1000000R 三个半径，作为解析证书的数值交叉核验。
两个失败均针对原固定布局。物体上移/左右移动会改变布局与系统压力中心，需要重新生成并评估；本记录没有声称所有可能布局都无解。
第二份证书针对最终系统的接地覆盖；没有把“任意空中几何凸包包裹”当作实际接地支撑。
不要求材料铺满凸包、完整外圈或连续接地。第一份图的外圈仅为直观示意。
支撑姿态转移前取出物体，并允许抬起空支撑；第一种冲突是落座后的碰撞，不能用抬高转移路径消除。

文件
index.html 本轮入口；demo/index.html 主例说明与图。
experiment.json 参数与全部 52 组输入清单；summary.json 实际结果；all_sets.csv 全组表。
output/B/<set>/step4/step4.2/proof_cold_v1 或 proof_own_repair_v1：原算法实际新结果和源代码快照。
output/B/<set>/step5/step5.1 和 step5.2：原计算及渲染结果。
certificates/<set>/report.json：无界必要条件证书；NPZ 保存完整工作锥切面方程。
diagnostics/ 为先前的有限窗口探索，不能拿其失败声称全局无解。
certificate_runs/ 和 demo/execution/ 为执行源快照。
history/ 为本轮图片生成的失败、早期视图及中断时汇总，未删除旧输入或旧优化结果。
首次入口读取失败组的 counts 字典时错误地迭代了 pose 键，导致汇总中断；52 个冷启动数值 worker 实际均完成。
resume_batch.py 从保存的 run_result.json 恢复记录，完成 Step5 和原有 own-state 修复，没有重跑冷启动数值 worker。
run_experiment.py 保留本次实际执行版本与哈希；run_all.py 是能自动调用恢复逻辑的完整复现入口。

复现
使用仓库 .venv/bin/python；依赖都是项目原依赖。结果目录受到保护，入口不会覆盖已有实验。
把本目录的七个 .py 文件复制到 Co-optimize 下一个新的同级文件夹，再运行该目录的 run_all.py。
完成后运行 certify.py、publish_demo.py、finish_experiment.py。diagnose.py 是证书模块的共享辅助实现，也必须复制。
正式数值代码、objects 输入和原 Step3/4.1 应保持当前版本；快照记录了本次实际源代码哈希。
'''
    (PROOF/'explanation.txt').write_text(explanation)
    # Validate every local link in the delivered root/demo and stage galleries.
    missing = []
    for file in [PROOF/'index.html',PROOF/'demo/index.html',D.BASE/'step5.1_index.html',D.BASE/'step5.2_index.html']:
        for target in re.findall(r'(?:href|src)="([^"]+)"',file.read_text()):
            if not target.startswith(('https:','http:','#')) and not (file.parent/html.unescape(target)).exists():
                missing.append((str(file),target))
    assert not missing,missing
    verification['published_local_links_valid'] = True
    C.save(PROOF/'verification.json',verification)
    print('PUBLISHED',len(rows),'sets,',len(passed),'PASS,',points,'original points;',len(both),'both-error sets',flush=True)


if __name__=='__main__':
    main()
