"""Audit saved acceptance evidence and immutable sources; no exported replay."""
from co_common import *
manifest=json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets'];rows=[];instances=0
I.check_hashes(json.loads((HERE/'data/protected_inputs.json').read_text()))
for group in manifest:
    base=HERE/'output/B'/group['id'];path=base/'step4/step4.2/data/report.json'
    if not path.exists():continue
    report=I.check_report(path)
    assert report['passed'] and not report['full_fixture_accepted'] and not report['connectivity_required'] and not report['ground_coverage_required']
    assert report['maximum_remaining_sweep_overlap_m3']<1e-10 and report['volume_partition_error_m3']<1e-10
    assert {x.name for x in path.parent.parent.glob('*.png')}=={'overview.png','exit_motion.png'}
    for stage in ['step3.1','step3.2','step3.3']:I.check_report(base/'step3'/stage/'data/report.json')
    I.check_report(base/'step4/step4.1/data/report.json')
    with np.load(path.parent/'remaining_contacts.npz') as z:triangles=z['triangles_mesh_m'];sources=z['source_faces']
    original=np.load(base/'step3/step3.2/data/contacts.npz');assert not np.intersect1d(sources,original['excluded_work_faces']).size
    for row in report['state_results']:
        with np.load(path.parent/(row['pose']+'.npz')) as z:
            mask=z['mask'];full=z['supply_7d'];T=z['T_fixture_to_world'];world=z['direction_world']
        assert mask.shape==(32768,) and mask.all() and row['force_covered']==row['load_count']==32768
        task,expected_T,mesh=state('B',row['pose']);np.testing.assert_allclose(T,expected_T,atol=1e-12)
        # Saved force generators must be built from the saved real contacts
        # and the unchanged native floor/no-uplift model, without new solves.
        expected=supply(task,T,triangles,sources);np.testing.assert_allclose(full,expected,atol=1e-12,rtol=0)
        np.testing.assert_allclose(T[:3,:3]@row['direction_fixture'],world,atol=1e-12)
        assert row['minimum_object_world_z_along_path_m']>=-1e-9 and row['endpoint_completely_separated'];instances+=1
    rows.append(dict(id=group['id'],poses=len(group['poses']),restored_volume_cm3=report['restored_volume_cm3'],remaining_component_count=report['remaining_component_count'],proposals=report['proposal_count'],seconds=report['seconds']))
summary=dict(complete=len(rows)==20,passed_sets=len(rows),pose_instances=instances,original_demands_checked=instances*32768,protected_files=959,connectivity_required=False,export_geometry_replayed=False,results=rows)
save(HERE/'output/B/data/step42_verification.json',summary);print({k:v for k,v in summary.items() if k!='results'})
if len(rows)==20:
    lines=['','## Step4.2：退出路径共同优化、恢复承载','', '**20/20 组、80 个 pose 实例的全部原始力／力矩需求通过。** 路径连续退出且不进入自身地面；本阶段不要求材料连通，也未重建接地材料。绿色表示相对 Step4.1 恢复的材料。','', '| Pose set | 优化图 | 退出示意 | 恢复材料 cm³ | 搜索组合数 |','|---|---|---|---:|---:|']
    for row in rows:lines.append(f"| {row['id']} | [恢复后的支撑]({row['id']}/step4/step4.2/overview.png) | [无文字退出图]({row['id']}/step4/step4.2/exit_motion.png) | {row['restored_volume_cm3']:.2f} | {row['proposals']} |")
    p=HERE/'output/B/README.md';text=p.read_text();i=text.find('\n## Step4.2');text=text[:i] if i>=0 else text;p.write_text(text+'\n'.join(lines)+'\n')
