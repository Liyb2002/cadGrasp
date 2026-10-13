"""Read-only bounds of complete straight insertion/withdrawal sweeps.

The exact AABB of a translated mesh over [0,L] is the AABB of its two
endpoints. No swept solid, optimization, Boolean or force solve is needed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import trimesh

COOPT=Path(__file__).resolve().parents[1]
ROOT=COOPT.parents[1]
BASE=COOPT/'output/B'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def points(vertices, transform):
    return vertices@transform[:3,:3].T+transform[:3,3]


def box(clouds):
    lower=np.min([cloud.min(axis=0) for cloud in clouds],axis=0)
    upper=np.max([cloud.max(axis=0) for cloud in clouds],axis=0)
    dims=upper-lower
    return dict(min_m=lower.tolist(),max_m=upper.tolist(),dimensions_cm=(dims*100).tolist(),
        volume_cm3=float(np.prod(dims)*1e6),volume_litres=float(np.prod(dims)*1000))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',default='base_oracle_swept_bbox_v1')
    parser.add_argument('--baseline-direction',choices=['up','previous'],default='up')
    args=parser.parse_args()
    output=BASE/'data'/args.name;output.mkdir(parents=True,exist_ok=False)
    inputs={}
    def read_json(path):
        inputs[str(path.relative_to(ROOT))]=sha(path)
        return json.loads(path.read_text())
    mesh_path=ROOT/'objects/B/mesh.stl';inputs[str(mesh_path.relative_to(ROOT))]=sha(mesh_path)
    body=trimesh.load(mesh_path,force='mesh',process=False).vertices
    length=max(.5,4*float(np.ptp(body,axis=0).max()))
    old=read_json(BASE/'data/r6_unit_comparison_v1/comparison.json')
    previous={r['id']:r for r in old['rows'] if r['new_passed']}
    batch=read_json(BASE/'data/base_oracle_v3/batch.json');assert batch['complete']
    def load_layout(folder):
        path=folder/'layout.npz';inputs[str(path.relative_to(ROOT))]=sha(path)
        with np.load(path) as z:return {k:z[k].copy() for k in z.files}
    def measure(folder):
        layout=load_layout(folder);report=read_json(folder/'report.json')
        assert report.get('force_passed',report['passed'])
        path=folder/'support.obj';inputs[str(path.relative_to(ROOT))]=sha(path)
        fixture=trimesh.load(path,force='mesh',process=False).vertices
        clouds=[];object_clouds=[];directions={}
        for k in layout['active']:
            T=layout['native_world'][layout['hosts'][k]];U=T@layout['placements'][k]
            direction=T[:3,:3]@layout['directions'][k]
            assert np.isclose(np.linalg.norm(direction),1.,atol=1e-9)
            p=points(body,U);object_clouds.extend([p,p+length*direction])
            clouds.append(points(fixture,T));directions[report['poses'][k]]=direction
        return dict(box=box(clouds+object_clouds),object_sweeps_only=box(object_clouds),
            support_material_cm3=report['volume_cm3'],state_count=len(set(layout['hosts'].tolist())),
            source=str(folder.relative_to(ROOT))),directions,report['poses'],layout
    rows=[]
    for result in batch['results']:
        label=result['id'];current,_,poses,layout=measure(BASE/label/'step4/step4.2/base_oracle_v3')
        previous_row=None;previous_dirs={}
        if label in previous:
            previous_row,previous_dirs,old_poses,old_layout=measure(BASE/label/'step4/step4.2'/previous[label]['new_experiment'])
            assert poses==old_poses and np.allclose(layout['native_world'],old_layout['native_world'],atol=1e-10)
        original=load_layout(BASE/label/'step4/step4.1')
        assert np.allclose(original['native_world'],layout['native_world'],atol=1e-10)
        upright=[];native_previous=[];native_static=[]
        for k,pose in enumerate(poses):
            p=points(body,original['native_world'][k]);native_static.append(p)
            upright.extend([p,p+length*np.array([0.,0.,1.])])
            if previous_row is not None:native_previous.extend([p,p+length*previous_dirs[pose]])
        no_fixture_up=box(upright);no_fixture_previous=box(native_previous) if native_previous else None
        baseline=no_fixture_up if args.baseline_direction=='up' else no_fixture_previous
        row=dict(id=label,pose_count=len(poses),without_support=baseline,
            without_support_up=no_fixture_up,without_support_previous_direction=no_fixture_previous,
            native_static_object_only=box(native_static),before_oracle=previous_row,with_oracle=current)
        if baseline:row['current_vs_without_support_percent']=(current['box']['volume_cm3']/baseline['volume_cm3']-1)*100
        if previous_row:
            row['current_vs_before_oracle_percent']=(current['box']['volume_cm3']/previous_row['box']['volume_cm3']-1)*100
            row['extra_vs_before_oracle_litres']=current['box']['volume_litres']-previous_row['box']['volume_litres']
        rows.append(row)
    rows.sort(key=lambda r:(-r['pose_count'],r['id']))
    common=[r for r in rows if r['before_oracle']]
    sums={k:sum((r['without_support'] if k=='without_support' else r[k]['box'])['volume_litres'] for r in common)
          for k in ['without_support','before_oracle','with_oracle']}
    report=dict(complete=True,group_count=len(rows),common_pre_oracle_group_count=len(common),
        length_m=length,baseline_direction=args.baseline_direction,units='litres and cm3',
        metric='one task-world Cartesian AABB of all object insertion/exit sweeps and all installed support meshes',
        exact_endpoints_sufficient_for_fixed_orientation_straight_translation=True,
        native_objects_recentered=False,extra_margin_m=0,
        previous_selection='r6_unit_comparison_v1 final new_experiment for each of its 42 force-passing sets',
        pre_oracle_missing_reason='10 original illegal sets have no passed result in the preceding R6 unit batch',
        fixture_transfer_trajectory_included=False,robot_tool_motion_included=False,unbounded_work_cones_included=False,
        final_base_included=False,oracle_maximal_space_witness_counted_as_material=False,
        force_solver_run=False,new_boolean_run=False,
        common_group_bbox_sums_litres=sums,
        common_group_aggregate_increase_percent=(sums['with_oracle']/sums['before_oracle']-1)*100,
        common_group_median_increase_percent=float(np.median([r['current_vs_before_oracle_percent'] for r in common])),
        cases=rows)
    assert all(sha(ROOT/p)==expected for p,expected in inputs.items())
    (output/'inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
    (output/'source.json').write_text(json.dumps({str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))},indent=2)+'\n')
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    lines=['# 装入／退出扫掠BBX：无支撑、旧版、动态Oracle','',
        f'退出行程统一{length*100:.3f}cm；单位L（1L=1000cm³）。沿同一路径反向装入占同一空间。',
        '每组取所有运动和安装支撑的共同世界坐标包围盒；逐轴合并极值，不把各pose体积相加。',
        f'无支撑方向：{args.baseline_direction}。旧版沿用R6 unit最终保存方案，当前为base_oracle_v3。',
        '包含物体的整个直线扫掠和支撑的工作状态；不含尚未建造的实际base、机器人／工具运动或未指定的空支撑搬运轨迹。',
        '原illegal/的10组没有上一批通过方案，旧版一列留空；42组共同集合才做旧新汇总比较。','',
        '| Pose set | Pose数 | 无支撑扫掠BBX (L) | 旧版BBX (L) | Oracle版BBX (L) | 比旧版增加 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        a='—' if r['without_support'] is None else f"{r['without_support']['volume_litres']:.2f}"
        b='—' if r['before_oracle'] is None else f"{r['before_oracle']['box']['volume_litres']:.2f}"
        delta='—' if r['before_oracle'] is None else f"{r['current_vs_before_oracle_percent']:+.1f}%"
        lines.append(f"| {r['id']} | {r['pose_count']} | {a} | {b} | {r['with_oracle']['box']['volume_litres']:.2f} | {delta} |")
    lines.extend(['',f"42组BBX累加统计：无支撑{sums['without_support']:.2f}L，旧版{sums['before_oracle']:.2f}L，Oracle版{sums['with_oracle']:.2f}L（比旧版{report['common_group_aggregate_increase_percent']:+.1f}%）。这不是一个工作站同时容纳42组的体积。",
                  f'完整包围盒尺寸、两种无支撑方向基线及输入哈希见[data/{args.name}/report.json](data/{args.name}/report.json)。'])
    (BASE/f'{args.name}.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},ensure_ascii=False,indent=2))
    print('LARGE SETS',json.dumps([r for r in rows if r['pose_count']>=7],ensure_ascii=False))


if __name__=='__main__':main()
