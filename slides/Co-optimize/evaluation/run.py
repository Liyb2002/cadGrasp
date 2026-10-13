"""Evaluate overlapping unsupported poses and actual supported static layouts."""
from pathlib import Path
import argparse, html, json, os, shutil, sys, time
import numpy as np
import trimesh
from metrics import static_bounds, METRIC_VERSION, SUPPORT_POSE_METRIC_VERSION, BOX_KEYS, STATE_KEYS

COOPT=Path(__file__).resolve().parents[1];ROOT=COOPT.parents[1]
for folder in [ROOT,COOPT/'vis_func',COOPT/'helper_func',COOPT/'step5.1']:sys.path.insert(0,str(folder))
import step51_render as V
import hull_render as H
import artifact_store as A


def sha(p):return A.digest(p)
def relative(p):return str(Path(p).resolve().relative_to(ROOT))
def save(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def final_stage(root, experiment):
    for stage in ['step5.3','step5.2']:
        folder=root/'step5'/stage/experiment
        if (folder/'data/report.json').exists():
            record=json.loads((folder/'data/report.json').read_text())
            if record.get('stage_type')=='support_pruning':
                raise ValueError('Step5.2 is pruned material; construct Step5.3 before Evaluation')
            return folder
    raise FileNotFoundError('No completed final support for '+str(root))


def corners(record):
    lo,hi=record['min_m'],record['max_m']
    return np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])


def render(renderer,body,nominal,full,layout,result,out):
    cameras={key:V.camera_for(corners(result[key])) for key in BOX_KEYS}
    span=max(camera[2] for camera in cameras.values())
    baseline={p['layout_index']:np.array(p['unsupported_world_transform']) for p in result['per_pose']}
    tiles=[]
    for key,fixture in [('object_static',None),('with_step4_support',nominal),('with_full_support',full)]:
        focus,basis,_=cameras[key];camera=(focus,basis,span)
        pieces=[];seen=set()
        for k in layout['active']:
            T=layout['native_world'][layout['hosts'][k]]
            U=baseline[int(k)] if fixture is None else T@layout['placements'][k]
            start=body.copy();start.apply_transform(U)
            pieces.append(V.piece(start,V.GRAY,.5))
            state=np.round(T,11).tobytes()
            if fixture is not None and state not in seen:
                mesh=fixture.copy();mesh.apply_transform(T);pieces.append(V.piece(mesh,V.BLUE))
                seen.add(state)
        v=corners(result[key])
        for q in [[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]]:
            pieces.append(H.outline_piece(v[q],'#466474',camera,650,.65))
        tiles.append(renderer.render([V.CPU.placed(p) for p in pieces],camera,650))
    V.sheet(tiles,out/'static_occupancy.png')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object',nargs='?',default='B')
    parser.add_argument('--experiment',default='connected_base_v1')
    parser.add_argument('--selection-summary',type=Path)
    parser.add_argument('--batch-name',default='evaluation_connected_base_v1')
    parser.add_argument('--sets',nargs='+')
    parser.add_argument('--replace',action='store_true',help='Replace only this evaluation batch and its visualizations')
    args=parser.parse_args()
    base=A.result_root(args.object)
    selection=A.resolve_artifact(args.selection_summary or base/'data/base_oracle_results/summary.json')
    protected={}
    def protect(p):protected[relative(p)]=sha(p);return p
    selected=json.loads(protect(selection).read_text())
    cases=selected.get('selected',selected.get('results',selected.get('cases',[])))
    if args.sets:
        if set(args.sets)-{case['id'] for case in cases}:parser.error('Unknown set')
        cases=[case for case in cases if case['id'] in args.sets]
    mesh_path=protect(ROOT/'objects'/args.object/'mesh.stl')
    body=trimesh.load(mesh_path,force='mesh',process=False)
    if any(Path(name).name!=name for name in [args.experiment,args.batch_name]):
        raise ValueError('Experiment and batch names must be single directory names')
    directory=base/'data'/args.batch_name
    # Validate all physical inputs before replacing the obsolete evaluation.
    for case in cases:
        root=base/case['id'];s4=root/'step4/step4.2'/case['experiment'];s52=final_stage(root,args.experiment)
        final=json.loads(protect(s52/'data/report.json').read_text())
        if not final['complete'] or not final['passed'] or final['selected_step4_experiment']!=case['experiment']:
            raise ValueError(f'Incomplete or mismatched Step5.2: {case["id"]}')
        for p in [s4/'layout.npz',s4/'report.json',s4/'support.obj',s52/final['support']]:protect(p)
    destinations=[directory]+[base/c['id']/'evaluation'/args.experiment for c in cases]
    destinations += [A.VISUALS/args.object/c['id']/'evaluation'/args.experiment for c in cases]
    if args.replace:
        for p in destinations:shutil.rmtree(p,ignore_errors=True)
    elif any(p.exists() for p in destinations):
        raise FileExistsError('Evaluation exists; use --replace to regenerate the evaluation only')
    directory.mkdir(parents=True,exist_ok=False)
    code={}
    paths={Path(__file__).resolve(),COOPT/'evaluation/metrics.py',V.CPU_PATH,V.BASELINE/'step4_connect_support/translucent_raster.cpp'}
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file:
            p=Path(file).resolve()
            if p.is_relative_to(ROOT) and not p.is_relative_to(ROOT/'.venv') and p.is_file():paths.add(p)
    for p in sorted(paths):
        target=directory/'executed_sources'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target);code[relative(p)]=sha(p)
    save(directory/'execution.json',dict(executed_code=code,selection=relative(selection),
        metric_version=METRIC_VERSION,support_pose_metric_version=SUPPORT_POSE_METRIC_VERSION))
    renderer=V.Renderer();rows=[];began=time.monotonic()
    for case in sorted(cases,key=lambda c:(-c['pose_count'],c['id'])):
        root=base/case['id'];step4=root/'step4/step4.2'/case['experiment'];step52=final_stage(root,args.experiment)
        final=json.loads(protect(step52/'data/report.json').read_text())
        if not final['complete'] or not final['passed']:raise ValueError(f'Unfinished Step5.2: {case["id"]}')
        if final['selected_step4_experiment']!=case['experiment']:raise ValueError('Comparison layouts differ')
        with np.load(protect(step4/'layout.npz')) as z:layout={k:z[k].copy() for k in z.files}
        layout_report=json.loads(protect(step4/'report.json').read_text())
        nominal=trimesh.load(protect(step4/'support.obj'),force='mesh',process=False)
        full=trimesh.load(protect(step52/final['support']),force='mesh',process=False)
        result=static_bounds(body.vertices,layout,nominal.vertices,full.vertices,layout_report['poses'])
        for key in ['juxtaposed_pose_count','rotating_reuse_pose_count']:
            if result[key]!=layout_report[key]:raise ValueError(f'Layout membership differs from saved Step4: {case["id"]}')
        result.update(id=case['id'],step4_source=relative(step4),step53_source=relative(step52),
            material_step4_cm3=float(abs(nominal.volume)*1e6),material_full_cm3=float(abs(full.volume)*1e6),
            material_added_cm3=final['construction']['added_material_cm3'],
            material_removed_illegal_nominal_cm3=final['construction']['removed_illegal_nominal_material_cm3'])
        result['base_bbx_increase_litres']=result['with_full_support']['volume_litres']-result['with_step4_support']['volume_litres']
        result['base_bbx_increase_percent']=100*(result['with_full_support']['volume_cm3']/result['with_step4_support']['volume_cm3']-1)
        result['state_space_base_increase_percent']=100*(result['state_space_sums_litres']['with_full_support']/result['state_space_sums_litres']['with_step4_support']-1)
        out=root/'evaluation'/args.experiment;out.mkdir(parents=True,exist_ok=False)
        render(renderer,body,nominal,full,layout,result,out)
        result['output']=relative(out);save(out/'report.json',result)
        b=lambda k:f"{result[k]['volume_litres']:.2f} L ({' × '.join(f'{d:.2f}' for d in result[k]['dimensions_cm'])} cm)"
        state_list=''.join(f'<li>Support pose {s["state"]}: {s["pose_count"]} object poses — {html.escape(", ".join(s["poses"]))}</li>' for s in result['fixture_states'])
        sums=result['state_space_sums_litres']
        (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;max-width:1600px;margin:24px auto}img{width:100%}</style>'
            f'<h1>{html.escape(case["id"])} · Static occupied space</h1><p>Tasks run one at a time; object poses may overlap.</p>'
            '<p>The no-fixture baseline preserves task rotations, aligns horizontal box centers and rests each pose on z=0. '
            'It attains the smallest XYZ box under free translations. Supported columns retain the saved fixture layout.</p>'
            f'<p>Left: overlapping objects without support {b("object_static")}; middle: actual layout with Step4 {b("with_step4_support")}; '
            f'right: actual layout with full Step5.2 support {b("with_full_support")}.</p>'
            '<p>All three images use the same isometric orientation and length scale, with each box centered in its panel.</p>'
            '<img src="static_occupancy.png">'
            f'<p>Diagnostic objects at fixture seats: {b("objects_in_final_layout")}. '
            f'Sum of separately allocated state spaces: objects {sums["objects_in_final_layout"]:.2f} L; '
            f'with Step4 {sums["with_step4_support"]:.2f} L; with Step5.2 {sums["with_full_support"]:.2f} L. '
            'These sums are not the no-fixture baseline.</p>'
            f'<p>Support poses: {result["support_pose_count"]}. Object poses per support pose: '
            f'{", ".join(str(n) for n in result["object_pose_counts_per_support_pose"])}. '
            'Each support pose is a distinct saved world position and orientation of the same fixture; tasks run sequentially.</p><ul>'
            f'{state_list}</ul>'
            f'<p>Final Juxtapose seats: {result["juxtaposed_pose_count"]}; unchanged rotating-reuse poses: {result["rotating_reuse_pose_count"]}. '
            f'Multi-object-pose states: {result["multi_pose_fixture_state_count"]}; single-object-pose states: {result["singleton_fixture_state_count"]}. These are different classifications.</p>'
            '<p><a href="report.json">Per-pose bounds, state membership and material volumes</a></p>\n')
        visual=A.VISUALS/args.object/case['id']/'evaluation'
        visual.mkdir(parents=True,exist_ok=True)
        shutil.copy2(out/'static_occupancy.png',visual/'static_occupancy.png')
        (out/'index.html').unlink(missing_ok=True)
        (out/'static_occupancy.png').unlink()
        if args.object=='B':
            from publish_output_images import publish_case_images
            publish_case_images(case['id'],args.experiment,include_evaluation=True)
        rows.append(result)
        save(directory/'progress.json',dict(complete=False,completed=len(rows)))
        print('EVALUATION',case['id'],*[round(result[k]['volume_litres'],2) for k in BOX_KEYS],flush=True)
    for p,h in protected.items():
        if sha(ROOT/p)!=h:raise ValueError(f'Input changed: {p}')
    for p,h in code.items():
        if sha(ROOT/p)!=h or sha(directory/'executed_sources'/p)!=h:raise ValueError(f'Execution source changed: {p}')
    sums={k:sum(r[k]['volume_litres'] for r in rows) for k in BOX_KEYS}
    report=dict(complete=True,object=args.object,cases=rows,group_count=len(rows),pose_instance_count=sum(r['pose_count'] for r in rows),
        metric='static working occupancy: optimal overlapping unsupported poses versus saved supported world-axis union AABBs',
        metric_version=METRIC_VERSION,support_pose_metric_version=SUPPORT_POSE_METRIC_VERSION,
        same_final_object_placements_in_all_columns=False,
        unsupported_baseline_translation_optimal=True,
        object_withdrawal_motion_included=False,fixture_transfer_motion_included=False,
        robot_tool_motion_included=False,unbounded_work_exclusions_included=False,
        material_volume_is_not_bbox_volume=True,force_solver_run=False,boolean_run=False,
        independent_group_bbox_sums_litres=sums,
        independent_pose_object_sum_litres=sum(r['independent_pose_object_sum_litres'] for r in rows),
        state_space_sums_litres={k:sum(r['state_space_sums_litres'][k] for r in rows) for k in STATE_KEYS},
        layout_totals={k:sum(r[k] for r in rows) for k in ['juxtaposed_pose_count','rotating_reuse_pose_count',
            'state_count','support_pose_count','multi_pose_fixture_state_count','singleton_fixture_state_count','poses_in_multi_pose_fixture_states']},
        material_step4_sum_cm3=sum(r['material_step4_cm3'] for r in rows),material_full_sum_cm3=sum(r['material_full_cm3'] for r in rows),
        maximum_base_bbx_increase_percent=max(r['base_bbx_increase_percent'] for r in rows),
        median_base_bbx_increase_percent=float(np.median([r['base_bbx_increase_percent'] for r in rows])),
        seconds=time.monotonic()-began,upstream_unchanged=True,executed_source_hashes_match=True)
    save(directory/'summary.json',report);save(directory/'verification.json',dict(protected=protected,executed_code=code,passed=True))
    lines=['# 静态工作空间：无支撑叠放基线／真实支撑布局','',
        f'全部 {len(rows)} 组，共 {report["pose_instance_count"]} 个 pose。只计最终工作位置，单位 L；1 L=1000 cm³。',
        '无支撑基线保留各原任务朝向，将物体的水平包围盒中心对齐、最低点放在z=0，叠放全部pose后取一个BBX。其每轴长度为该轴各pose长度的最大值，达到自由平移时的最小XYZ盒。',
        '任务依次执行；一个 state 服务多个 object pose 不表示多个物体同时装入。Juxtapose 是最终 pose 是否重新落座，multi-object-pose state 是共享关系，两者不是互斥分类。',
        '支撑pose按保存的实际世界位置与朝向分组；同一件支撑换位置或转动后可以有多个支撑pose。每组列出服务的物体pose数量与成员。',
        '有支撑两列使用保存的真实优化位置及全部最终支撑state；不改变物理结果。三列图同等轴测方向、同长度比例，分别居中。','',
        '| Pose set | 物体pose数 | 无支撑叠放BBX (L) | +Step4 BBX (L) | +Step5.2 BBX (L) | 支撑pose数 | 每个支撑pose服务的物体pose数 | 物体pose分组 | Juxtapose pose数 | 原位转动复用pose数 |',
        '|---|---:|---:|---:|---:|---:|---|---|---:|---:|']
    for r in rows:
        url=os.path.relpath(A.VISUALS/args.object/r['id']/'evaluation'/args.experiment/'static_occupancy.png',base)
        membership='; '.join('+'.join(p.removeprefix('pose_') for p in s['poses']) for s in r['fixture_states'])
        counts=', '.join(str(n) for n in r['object_pose_counts_per_support_pose'])
        lines.append(f'| [{r["id"]}]({url}) | {r["pose_count"]} | {r["object_static"]["volume_litres"]:.2f} | {r["with_step4_support"]["volume_litres"]:.2f} | {r["with_full_support"]["volume_litres"]:.2f} | {r["support_pose_count"]} | {counts} | {membership} | {r["juxtaposed_pose_count"]} | {r["rotating_reuse_pose_count"]} |')
    allocated=report['state_space_sums_litres'];totals=report['layout_totals']
    lines.extend(['',f'各集合BBX统计和：无支撑叠放 {sums["object_static"]:.2f} L，Step4 {sums["with_step4_support"]:.2f} L，完整支撑 {sums["with_full_support"]:.2f} L；不是同一工作站的总体积。',
        f'支撑pose统计和：{totals["support_pose_count"]}，各组成员数量之和为{report["pose_instance_count"]}个物体pose实例。',
        f'最终 Juxtapose {totals["juxtaposed_pose_count"]} 个 pose，原位转动复用 {totals["rotating_reuse_pose_count"]} 个 pose；{totals["multi_pose_fixture_state_count"]} 个 state 服务多个 object pose，{totals["singleton_fixture_state_count"]} 个 state 只服务一个。',
        '', '以下为真实支撑落座物体的诊断空间，并非无支撑基线。', '',
        '| Pose set | 支撑落座物体BBX (L) | +Step4 (L) | +Step5.2 (L) | 地脚BBX增幅 | Step4实体 (cm³) | 全部实体 (cm³) |',
        '|---|---:|---:|---:|---:|---:|---:|'])
    for r in rows:
        lines.append(f'| {r["id"]} | {r["objects_in_final_layout"]["volume_litres"]:.2f} | {r["with_step4_support"]["volume_litres"]:.2f} | {r["with_full_support"]["volume_litres"]:.2f} | {r["base_bbx_increase_percent"]:+.2f}% | {r["material_step4_cm3"]:.2f} | {r["material_full_cm3"]:.2f} |')
    lines.extend(['',f'单独分配不同state时的空间诊断统计和：物体 {allocated["objects_in_final_layout"]:.2f} L，Step4 {allocated["with_step4_support"]:.2f} L，Step5.2 {allocated["with_full_support"]:.2f} L。',
        f'[逐 pose 尺寸、共享分组和材料数据](data/{args.batch_name}/summary.json)。'])
    (base/f'evaluation_{args.experiment}.md').write_text('\n'.join(lines)+'\n')
    headers='<tr><th>Set</th><th>Object poses</th><th>Overlapping objects, no support (L)</th><th>Step4 BBX (L)</th><th>Step5.2 BBX (L)</th><th>Support poses</th><th>Object poses per support pose</th><th>Object pose membership</th><th>Juxtapose poses</th><th>Rotating reuse poses</th></tr>'
    cells=''.join(f'<tr><td><a href="{r["id"]}/evaluation/{args.experiment}/index.html">{r["id"]}</a></td>'
        f'<td>{r["pose_count"]}</td><td>{r["object_static"]["volume_litres"]:.2f}</td>'
        f'<td>{r["with_step4_support"]["volume_litres"]:.2f}</td><td>{r["with_full_support"]["volume_litres"]:.2f}</td>'
        f'<td>{r["support_pose_count"]}</td><td>{", ".join(str(n) for n in r["object_pose_counts_per_support_pose"])}</td>'
        f'<td>{html.escape("; ".join("+".join(p.removeprefix("pose_") for p in s["poses"]) for s in r["fixture_states"]))}</td>'
        f'<td>{r["juxtaposed_pose_count"]}</td><td>{r["rotating_reuse_pose_count"]}</td></tr>' for r in rows)
    (base/f'evaluation_{args.experiment}_index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;max-width:1400px;margin:24px auto}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:right}td:first-child{text-align:left}</style>'
        '<h1>Static occupied spaces</h1><p>Overlapping object poses without support versus saved Step4 and full Step5.2 layouts. '
        'Fixed task rotations; the no-support translations attain the smallest XYZ union box. Working positions only; tasks run one at a time.</p>'
        f'<p>Support poses across these independent sets: {totals["support_pose_count"]}. '
        'Group tasks by the saved world position and orientation of the fixture.</p>'
        f'<p>Juxtapose: {totals["juxtaposed_pose_count"]} poses; unchanged rotating reuse: {totals["rotating_reuse_pose_count"]} poses. '
        f'{totals["multi_pose_fixture_state_count"]} multi-object-pose states and {totals["singleton_fixture_state_count"]} single-object-pose states. Seating and sharing are distinct classifications.</p><table>'+headers+cells+'</table>\n')
    A.publish_gallery(base/f'evaluation_{args.experiment}_index.html')
    A.finish_batch(directory)
    print('EVALUATION ALL',len(rows),'seconds',round(report['seconds'],2),flush=True)


if __name__=='__main__':main()
