"""Curate completed experiments and audit saved evidence without rerunning search."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / 'slides/pose_set_search'
CASES = [
    ('1–7', 'joint', 'seven_v10/seven_chain/joint'),
    ('1–7', 'incremental', 'prefix_v15/seven_chain/incremental'),
    ('1,2,4,5,6,7,11', 'joint', 'hard_v16/seven_hard/joint'),
    ('1,2,4,5,6,7,11', 'incremental', 'hard_v16/seven_hard/incremental'),
    ('1,4,7,12,21,23,27', 'joint', 'spread_v16/seven_spread/joint'),
    ('1,4,7,12,21,23,27', 'incremental', 'spread_v16/seven_spread/incremental'),
    ('1–8', 'joint', 'eight_v16/eight_chain/joint'),
    ('1–8', 'incremental', 'prefix_v15/eight_chain/incremental'),
    ('1–10', 'joint', 'ten_v10/pose1-10/joint'),
    ('1–10', 'incremental', 'ten_incremental_v15/pose1-10/incremental'),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(directory, report):
    for name, expected in report['artifacts'].items():
        assert digest(directory / name) == expected, f'Changed artifact: {name}'
    for name, expected in report['provenance']['inputs'].items():
        assert digest(ROOT / name) == expected, f'Changed input: {name}'
    with np.load(directory / 'layout.npz') as layout:
        for k in layout['active']:
            world = layout['native_world'][layout['hosts'][k]] @ layout['placements'][k]
            native = layout['native_world'][k]
            np.testing.assert_allclose(world[:3, :3], native[:3, :3], atol=1e-10, rtol=0)
            assert abs(world[2, 3] - native[2, 3]) < 1e-9
            direction = layout['directions'][k]
            assert abs(np.linalg.norm(direction) - 1) < 1e-10
            assert (layout['native_world'][layout['hosts'][k], :3, :3] @ direction)[2] > 0
    result = dict(input_and_artifact_hashes_passed=True)
    if not report['force_exit_work_passed']:
        return result
    tolerance = report['numerical_tolerance_m3']
    assert all(value <= tolerance for name, value in report['diagnostics'].items()
               if name.endswith('_m3'))
    assert report['diagnostics']['minimum_endpoint_projection_gap_m'] > 0
    assert all(row['passed'] for row in report['actual_work_surface_checks'])
    for pose in report['poses']:
        with np.load(directory / f'{pose}_force.npz') as data:
            assert len(data['mask']) == report['load_count_per_pose'] == 32768
            assert data['mask'].all() and report['counts'][pose] == 32768
    certificate = directory / 'pressure_audit.json'
    if not certificate.exists():
        return dict(**result, pressure_certificate_complete=False)
    proof = json.loads(certificate.read_text())
    assert proof['complete'] and proof['passed']
    assert proof['actual_exported_mesh_contacts_regenerated']
    assert proof['support_sha256'] == digest(directory / 'support.obj')
    assert [row['pose'] for row in proof['checks']] == report['poses']
    for row in proof['checks']:
        assert row['passed'] and row['load_count'] == 32768
        assert row['minimum_coefficient'] >= 0
        assert row['maximum_residual_including_roundoff'] <= 2e-9
        assert digest(directory / f"{row['pose']}_pressure_certificate.npz") == row['certificate_sha256']
    result.update(pressure_certificate_complete=True,
                  certified_load_count=32768 * report['pose_count'],
                  maximum_pressure_replay_residual=max(
                      row['maximum_residual_including_roundoff'] for row in proof['checks']))
    return result


def main():
    rows = []
    runs = set()
    for label, mode, relative in CASES:
        directory = HERE / 'output' / relative
        row = dict(set=label, mode=mode, result=f'output/{relative}')
        path = directory / 'report.json'
        if not path.exists():
            row['status'] = '未完成' if not (directory / 'error.json').exists() else '数值异常，未完成验收'
            rows.append(row)
            continue
        report = json.loads(path.read_text())
        check = audit(directory, report)
        prefixes_passed = all(stage['passed'] for stage in report.get('insertion_stages', []))
        row.update(check, passed=report['force_exit_work_passed'] and prefixes_passed,
                   final_force_exit_work_passed=report['force_exit_work_passed'],
                   volume_cm3=report['volume_cm3'],
                   footprint_cm2=1e4 * report['maximum_projected_footprint_m2'],
                   fixture_placement_count=len(set(report['hosts'].values())),
                   counts=report['counts'], all_prefixes_passed=prefixes_passed)
        row['status'] = ('通过' if check.get('pressure_certificate_complete') else '等待反力复核') if row['passed'] else '有限搜索未通过'
        if report['force_exit_work_passed'] and not prefixes_passed:
            row['status'] = '仅最终布局通过，存在失败前缀'
        row['exact_evaluations'] = report.get('exact_search_evaluations_through_prefix', report['exact_evaluations'])
        if mode == 'incremental' and not report.get('exported_prefix_of_same_incremental_search'):
            joint = directory.parent / 'joint/report.json'
            if joint.exists():
                row['exact_evaluations'] -= json.loads(joint.read_text())['exact_evaluations']
        run = HERE / 'output' / relative.split('/')[0]
        if report.get('exported_prefix_of_same_incremental_search'):
            run = ROOT / report['source_run']
        runs.add(run)
        row['runtime_manifest'] = str((run / 'runtime_sources/manifest.json').relative_to(HERE))
        if not report.get('exported_prefix_of_same_incremental_search'):
            sources = json.loads((run / 'runtime_sources/manifest.json').read_text())['startup_sources']
            assert all(sources[name] == expected for name, expected in report['executed_search_sources_sha256'].items())
        rows.append(row)
    for run in runs:
        protected = json.loads((run / 'protected_coopt_sources.json').read_text())
        assert all(digest(ROOT / name) == expected for name, expected in protected.items())
        snapshot = run / 'runtime_sources'
        manifest = json.loads((snapshot / 'manifest.json').read_text())
        assert all(digest(snapshot / name) == expected for name, expected in manifest['startup_sources'].items())
    summary = dict(cases=rows, protected_coopt_sources_unchanged=True,
                   frozen_runtime_sources_verified=True,
                   certified_result_count=sum(row['status'] == '通过' for row in rows),
                   certified_load_count=sum(row.get('certified_load_count', 0) for row in rows),
                   full_fixture_acceptance_run=False)
    (HERE / 'results.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    lines = [
        '# 实验结果', '',
        '“通过”表示原始承载、完整装卸路径和工作面验收通过，并从导出的实体重新提取接触、保存及回代全部原始载荷的非负反力。每个 pose 有 32,768 条载荷，包含第七维不上抬约束。', '',
        '| pose set | 方法 | 验收 | 材料 cm³ | 最大投影占地 cm² | 支撑摆放数 | 精确候选数 |',
        '|---|---|---|---:|---:|---:|---:|',
    ]
    for row in rows:
        link = f"[{row['status']}]({row['result']}/report.json)" if 'passed' in row else row['status']
        values = [row['set'], row['mode'], link]
        values.extend(f"{row[name]:.1f}" if name in row else '—' for name in ['volume_cm3', 'footprint_cm2'])
        values.extend(str(row.get(name, '—')) for name in ['fixture_placement_count', 'exact_evaluations'])
        lines.append('| ' + ' | '.join(values) + ' |')
    lines.extend([
        '',
        '1–7 和 1–8 的 incremental 是 1–10 增量运行中真实通过的前缀，分别重新导出第 13、14 次精确评估的实体。它们的种子材料、物体及退出切除只包含当时已加入的任务，没有未来 pose 的材料；不是另外随机重跑的独立样本。', '',
        '1–10 的 joint 用一个支撑摆放容纳十个物体 pose；incremental 用八个摆放。增量材料少约 26.9%、最大投影占地小约 16.6%，但摆放数更多。当前没有把摆放数纳入代价，所以不能直接称哪一种更优。', '',
        '这里保留了实验过程中的有效冻结版本：1–7 / 1–10 joint 为 v10，1–10 incremental 为 v15，其他新集合为 v16。几何及物理验收条件相同；v15 增加了等几何工作面数值重参数化，v16 修正了初筛中用物体实际质心计算分散程度。预算、版本及并行运行时的负载不同，表格不构成受控速度或最优性比较。', '',
        '支撑摆放数是 host 对应的不同刚体摆放数量；占地是这些摆放下实体支撑的最大 XY 凸包投影面积，不是车间所有工位面积之和。精确候选数包含数值失败／超时候选；两个方法连跑时，incremental 表内已减掉 joint 的累计调用。', '',
        '搜索不以远距离分离保底。最终导出的每个已通过布局都另存实际物体两两重叠体积；材料重新构造，因此增加新 pose 时材料总量也可能减少。材料体积和占地是搜索结果，没有全局最小保证。', '',
        '两种 1–10 实体支撑的同尺度四视角对照：[四视角 PNG](vis/ten_pose_supports.png)。图中物体已移走，蓝色直接来自最终导出的实体，没有额外补画连接或底座。', '',
        '整体优化的 [十个物体 pose 图](vis/pose1-10_joint.png)：每格使用完全相同的蓝色支撑、固定世界摆放与镜头；灰色物体依次采用各个最终配置。最终为以 pose1 为 host 的成组 Juxtapose，pose2–10 保持原任务朝向并水平错开约 18.3–20.5 mm，退出都选择世界 +z。这里统计的是一个支撑摆放，实体连通尚未验收。', '',
        '当前验收沿用原模型：支撑接触只有非负法向反力、没有反力容量，地面仍是 μ=64 的四射线摩擦模型。整件支撑的连通、安装接地、强度和机器人路径尚未验收，报告的 `full_fixture_accepted` 均为 false。', '',
        '每个结果目录含 `support.obj`、`layout.npz`、`report.json`、逐 pose 的载荷覆盖及反力证书，`trace.json` 记录候选与原子提交。`runtime_sources/manifest.json` 给出源代码快照：新算法的精确 worker 直接运行冻结版本，借用的 Co-optimize / baseline 几何及物理文件由哈希检查保证与快照一致。复现历史结果时应先确认原输入及这些基线文件仍匹配。历史失败／中断保留作诊断，不混入上表的有效结果。', '',
        '原始输入、输出文件哈希、冻结源代码和 Co-optimize 源代码保护检查：[results.json](results.json)。重新汇总：`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python slides/pose_set_search/code/summarize.py`。', '',
    ])
    (HERE / 'results.md').write_text('\n'.join(lines))
    print('CERTIFIED RESULTS', summary['certified_result_count'], '/', len(rows))


if __name__ == '__main__':
    main()
