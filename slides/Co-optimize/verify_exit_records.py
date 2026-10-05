"""Inspect saved construction records and provenance; no exported replay."""
from co_common import *
manifest=json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets']
initial_path=HERE/'output/B/data/step41/initialization.json'
initial=json.loads(initial_path.read_text())
I.check_hashes(initial['provenance']['inputs']);I.check_hashes(initial['provenance']['code'])
for f,h in initial['artifacts'].items():assert I.sha256(initial_path.parent/f)==h
I.check_hashes(json.loads((HERE/'data/protected_inputs.json').read_text()))
rows=[];instances=0
for group in manifest:
    base=HERE/'output/B'/group['id'];path=base/'step4/step4.1/data/report.json';r=I.check_report(path)
    assert r['geometry_partition_resolved']
    assert r['remaining_sweep_overlap_m3']<1e-10
    assert not r['optimized'] and not r['full_fixture_accepted']
    assert len(list(path.parent.parent.glob('*.png')))==1
    for stage in ['step3.1','step3.2','step3.3']:I.check_report(base/'step3'/stage/'data/report.json')
    for state_result in r['state_results']:
        z=np.load(path.parent/(state_result['pose']+'.npz'));mask=z['force_mask'];assert mask.shape==(32768,);assert mask.sum()==state_result['force_covered']
        T=z['T_fixture_to_world'];np.testing.assert_allclose(z['direction_world'],[0,0,1],atol=1e-12)
        np.testing.assert_allclose(T[:3,:3].T@z['direction_world'],initial['paths'][state_result['pose']]['direction_fixture'],atol=1e-12)
        assert state_result['minimum_object_world_z_along_path_m']>=-1e-9
        assert state_result['endpoint_completely_separated'];instances+=1
    rows.append(dict(id=group['id'],force_passed=all(x['force_passed'] for x in r['state_results']),ground_passed=all(x['ground_coverage']['passed'] for x in r['state_results']),removed_volume_cm3=r['removed_volume_cm3']))
assert len(rows)==20 and instances==80
summary=dict(complete=True,sets=20,pose_instances=instances,force_passed_sets=sum(x['force_passed'] for x in rows),ground_passed_sets=sum(x['ground_passed'] for x in rows),protected_files=959,export_geometry_replayed=False,results=rows)
save(HERE/'output/B/data/step41_verification.json',summary)
print({k:v for k,v in summary.items() if k!='results'})
lines=['','## Step4.1：退出初始化','', '各 pose 的退出路径都初始化为自己的原生世界 +z 向上方向；未执行路径优化。青色为扫掠，红色为必须切除材料，灰色为剩余支撑。每组仅一张图。','', '| Pose set | 图 | 原始力／力矩需求 | 原环剩余覆盖（诊断） |','|---|---|---|---|']
for row in rows:lines.append(f"| {row['id']} | [初始化图]({row['id']}/step4/step4.1/overview.png) | {'通过' if row['force_passed'] else '未通过'} | {'通过' if row['ground_passed'] else '后续重建接地材料'} |")
p=HERE/'output/B/README.md';s=p.read_text();i=s.find('\n## Step4.1');s=s[:i] if i>=0 else s;p.write_text(s+'\n'.join(lines)+'\n')
