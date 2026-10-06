"""Audit saved evidence and describe floor heights; no new construction acceptance."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from co_common import *
from run_illegal import SOURCE
from codes.precompute_objects.floor_points import check_floor_points

groups=json.loads(SOURCE.read_text())['sets'];out=HERE/'output/B/illegal';rows=[];inputs=[SOURCE];instances=0
I.check_hashes(json.loads((HERE/'data/protected_inputs.json').read_text()))
for group in groups:
    base=out/group['id'];resultpath=base/'data/result.json';result=I.check_report(resultpath);inputs.append(resultpath)
    tasks=[state('B',p) for p in group['poses']]
    clouds=[np.load(ROOT/'objects/B/poses'/p/'floor_contact.npz')['floor_demands_xy_m'] for p in group['poses']]
    floors=check_floor_points(clouds,[T for task,T,mesh in tasks],group['poses'])
    actual_pairs={(r['pose'],pair['other_pose']):pair['violating_sample_count'] for r in floors.rows for pair in r['per_other_pose'] if pair['violating_sample_count']}
    expected_pairs={(p['source_pose'],p['target_pose']):p['violating_sample_count'] for p in group['violating_directed_pairs']}
    assert actual_pairs==expected_pairs
    row=dict(result,size=group['size'],sampled_floor_compatibility_passed=False,
        worst_mapped_demand_height_m=min(pair['minimum_height_m'] for r in floors.rows for pair in r['per_other_pose']))
    for stage in ['step3.1','step3.2','step3.3']:I.check_report(base/'step3'/stage/'data/report.json')
    I.check_report(base/'step4/step4.1/data/report.json')
    reportpath=base/'step4/step4.2/data/report.json'
    if result['passed']:
        report=I.check_report(reportpath);inputs.append(reportpath);I.check_report(reportpath.parent/'exit_motion_render.json')
        assert report['passed'] and report['connectivity_required'] and report['remaining_component_count']==1
        assert report['maximum_remaining_sweep_overlap_m3']<1e-10 and report['volume_partition_error_m3']<1e-10
        assert not report['full_fixture_accepted'] and not report['ground_coverage_required']
        z=np.load(reportpath.parent/'remaining_contacts.npz');tri,src=z['triangles_mesh_m'],z['source_faces']
        original=np.load(base/'step3/step3.2/data/contacts.npz');assert not np.intersect1d(src,original['excluded_work_faces']).size
        mesh=tasks[0][2]
        assert np.max(mesh.face_normals[src]@np.asarray([r['direction_fixture'] for r in report['state_results']]).T)<=1e-8
        for r,(task,T,mesh) in zip(report['state_results'],tasks):
            saved=np.load(reportpath.parent/(r['pose']+'.npz'))
            assert saved['mask'].shape==(32768,) and saved['mask'].all()
            assert r['force_covered']==r['load_count']==32768 and r['minimum_object_world_z_along_path_m']>=-1e-9 and r['endpoint_completely_separated']
            np.testing.assert_allclose(saved['supply_7d'],supply(task,T,tri,src),atol=1e-12,rtol=0)
            np.testing.assert_allclose(saved['T_fixture_to_world'],T,atol=1e-12,rtol=0)
            instances+=1
        # Descriptive heights of the already saved model, not an export replay,
        # force re-solve or added acceptance gate. Keep separate from PASS.
        support=trimesh.load(base/'step4/step4.2/remaining_support.obj',force='mesh',process=False)
        row['saved_support_floor_heights']=[dict(pose=p,minimum_world_z_m=float(transform_points(support.vertices,T)[:,2].min())) for p,(task,T,mesh) in zip(group['poses'],tasks)]
        row['saved_support_above_all_floors']=all(r['minimum_world_z_m']>=-1e-9 for r in row['saved_support_floor_heights'])
    rows.append(row)

summary=dict(complete=True,sets=len(rows),step42_passed_sets=sum(r['passed'] for r in rows),pose_instances_checked=instances,original_demands_checked=instances*32768,
    original_illegal_counts_reproduced=True,protected_files_unchanged=959,full_fixture_accepted=False,export_geometry_replayed=False,results=rows,
    provenance=provenance(inputs,[HERE/'helper_func/summarize_illegal.py']))
save(out/'data/summary.json',summary)
lines=['# B 非法 pose set：当前算法实验','',f"输入：[illegal_pose_sets.json]({SOURCE})。10 组，2–6 个 pose 各两组；复用原始姿态、工作面与全部 32,768 个需求，不重新采样。",'',
    f"**完整包裹承载检查 10/10 通过；Step4.2 承载、合法物体退出及连通 {summary['step42_passed_sets']}/10 通过。** 已核对 {instances} 个 pose 实例的 {instances*32768:,} 个保存需求通过记录。",'',
    '**这不表示原来的非法性已消除。** JSON 的失败条件是原生地面需求点映射到其他 pose 后落到地下；所有有向失败计数已复现。当前 Step4.2 尚未把安装后支撑的所有地面约束、真实接地材料覆盖纳入接受，因此两种结论可以同时成立。下表的地下深度是已有模型的描述性坐标诊断，不是新增接受或导出模型回放。','',
    '| Pose set | 原有有向非法样本数 | Step4.2 | 原始需求通过数 | 剩余块数 | 支撑最大地下深度 mm | 图 |',
    '|---|---:|---|---:|---:|---:|---|']
for r in rows:
    depth=max(0,-min(x['minimum_world_z_m'] for x in r.get('saved_support_floor_heights',[{'minimum_world_z_m':0}])))*1000
    demands=f"{r['size']*32768:,}/{r['size']*32768:,}" if r['passed'] else '未全部通过'
    image=f"[退出图]({r['id']}/step4/step4.2/exit_motion.png)" if r['passed'] else f"[未决结果]({r['id']}/README.md)"
    depth_text=f'{depth:.3f}' if r['passed'] else '未接受'
    lines.append(f"| {r['id']} | {r['original_floor_violation_count']:,} | {'PASS（当前阶段）' if r['passed'] else r['status']} | {demands} | {r.get('component_count','—')} | {depth_text} | {image} |")
    if not r['passed']:
        counts=r.get('force_best_counts');pairs=zip(next(g for g in groups if g['id']==r['id'])['poses'],counts or [])
        pending=['# '+r['id']+'：当前有界搜索未决','', '完整包裹承载通过；退出方向搜索尚未恢复全部原始需求。这不是无解证明。', '', '最佳实际候选的原始需求通过数：','']
        pending.extend(f'- {pose}：{count}/32768' for pose,count in pairs)
        pending+=['',f"实际搜索 {r.get('force_proposals',0)} 个方向组合。完整记录见内部 data/recovery.log；没有发布最终通过模型。"]
        (out/r['id']/'README.md').write_text('\n'.join(pending)+'\n')
lines+=['','搜索使用保存的自身 +z 初始化、共同方向趋势与连通块承载检查，初始物体位置固定。每组 force / connectivity 两个搜索阶段各设 180 秒预算；超时为 UNRESOLVED，不证明无解。','',
    '独立输出目录为 `output/B/illegal/`；原 20 组结果和原始 objects 输入保持不变。内部 `data/` 保存来源、失败计数复现、逐 pose 原始需求掩码和搜索记录。959 个受保护原始输入的哈希一致。','',
    '复现：先运行 `run_illegal.py --jobs 2`，再运行 `illegal_recovery.py --jobs 2 --seconds-per-phase 180`，最后运行 `summarize_illegal.py`。','',
    '下一步的初始位置平移，可以研究是否改善这些地面不兼容关系；必须同时重算真实接触、力矩、退出与地面条件。此批次没有启用位置平移。']
(out/'README.md').write_text('\n'.join(lines)+'\n')
print('ILLEGAL SUMMARY',summary['step42_passed_sets'],'/',len(rows),'saved support above floors:',sum(r.get('saved_support_above_all_floors',False) for r in rows),flush=True)
