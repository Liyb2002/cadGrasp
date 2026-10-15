"""Actual complete-solid figures and same-guard previous-run comparison."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys

ROOT=Path(__file__).resolve().parents[3]
COOPT=ROOT/'slides/Co-optimize'
BASE=COOPT/'data/results/B'


def read(path):return json.loads(path.read_text())


def arrays_equal(first,second):
    import numpy as np
    with np.load(first) as a,np.load(second) as b:
        return set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files)


def triangle_digest(mesh):
    """Exact binary64 triangle geometry, independent of OBJ index ordering."""
    import numpy as np
    triangles=mesh.vertices[mesh.faces]
    ordered=np.stack([triangle[np.lexsort(triangle.T[::-1])] for triangle in triangles])
    rows=ordered.reshape(len(ordered),9)
    rows=rows[np.lexsort(rows.T[::-1])]
    return hashlib.sha256(np.ascontiguousarray(rows).tobytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',default='pso_accelerated_all52_step4_20261014_01')
    parser.add_argument('--previous',default='pso_refine_all52_step4_20261014_01')
    args=parser.parse_args()
    system=BASE/'data'/('system_'+args.name)
    audit=read(system/'evaluation.json');assert audit['complete'] and audit['passed'] and audit['groups']==52
    batch=read(BASE/'data'/args.name/'batch.json')
    previous_batch=read(BASE/'data'/args.previous/'batch.json')
    option_exceptions={'output_name','runtime_cache','runtime_cache_memory_mb','gpu_replay'}
    option_differences={key:dict(previous=previous_batch['options'].get(key),current=batch['options'].get(key))
        for key in set(batch['options'])|set(previous_batch['options'])
        if previous_batch['options'].get(key)!=batch['options'].get(key)}
    assert set(option_differences)<=option_exceptions,option_differences
    results={r['id']:r for r in batch['results']}
    old_folder=BASE/'data'/args.previous/'workers'
    older={r['id']:r for p in old_folder.glob('*.json') for r in [read(p)]}
    tree=system/'frozen_sources';co=tree/'slides/Co-optimize'
    sys.path[:0]=[str(tree),str(co/'helper_func'),str(co/'vis_func'),str(co/'step4.2')]
    import _bootstrap
    from whole_step4_render import Panel,render_scene,layers_for_pose,read_layout
    from co_common import state
    from types import SimpleNamespace
    import numpy as np
    import trimesh
    from PIL import Image,ImageOps,ImageDraw
    pictures=system/'complete_solid_figures';pictures.mkdir(exist_ok=True)
    rows=[];tiles=[]
    for case in audit['cases']:
        label=case['id'];actual=results[label];old=older.get(label)
        row=dict(id=label,passed=case['passed'],seconds=actual.get('seconds'),
            status=case['status'],error=case.get('error'),
            oracle_status=case.get('oracle_status'),force_passed=case.get('force_passed'),
            duration_is_detection_upper_bound=bool(actual.get('duration_is_detection_upper_bound')),
            duration_is_incomplete=bool(actual.get('duration_is_incomplete')),
            old_duration_is_detection_upper_bound=bool(old and old.get('duration_is_detection_upper_bound')),
            old_passed=old.get('passed') if old else None,old_seconds=old.get('seconds') if old else None,
            old_status=old.get('status','PASS' if old.get('passed') else 'search_failed') if old else None,
            old_error=old.get('error') if old else None,
            gpu=actual.get('gpu_replay'),runtime=actual.get('runtime_cache'))
        row['telemetry_scope']=('Worker terminated before returning final telemetry; '
            'previous attempt counters are not attributed to this attempt'
            if actual.get('attempt_telemetry_available') is False else
            'Returned continuation fragment and restored checkpoint counters; '
            'uncommitted killed-fragment counters may be unavailable'
            if actual.get('checkpoint_recovery') or actual.get('serial_continuation')
            else 'Returned worker attempt')
        tile=Image.new('RGB',(640,530),'white');draw=ImageDraw.Draw(tile)
        if case['passed']:
            folder=ROOT/case['source'];report=read(folder/'report.json');pre=read(folder/'preconstructed_base.json')
            mesh=trimesh.load(folder/pre['support'],force='mesh',process=False)
            q=read_layout(folder/'layout.npz');states=[state('B',pose) for pose in report['poses']]
            model=SimpleNamespace(mesh=states[0][2],native=np.array([s[1] for s in states]),
                                  poses=report['poses'],tasks=[s[0] for s in states])
            layers=layers_for_pose(model,q,mesh,q.active[0])
            points=np.vstack([m.vertices for m,c in layers if len(m.vertices)])
            panel=Panel(None,size=(640,480),points=points,elevation=35.26438968,azimuth=-45)
            rendered=render_scene(panel,layers)
            tile.paste(ImageOps.contain(rendered,(640,470)),(0,40))
            row.update(full_bbx_l=case['full_bbx_l'],material_cm3=case['material_cm3'],
                pose_count=case['pose_count'],states=case['support_states'],juxtaposed=case['juxtaposed'],
                mesh_sha256=hashlib.sha256((folder/pre['support']).read_bytes()).hexdigest())
            if old and old['passed']:
                old_folder=BASE/label/'step4/step4.2'/args.previous
                old_pre=read(old_folder/'preconstructed_base.json')
                old_mesh_path=old_folder/old_pre['support']
                old_mesh_sha=hashlib.sha256(old_mesh_path.read_bytes()).hexdigest()
                same_mesh=(row['mesh_sha256']==old_mesh_sha or
                    triangle_digest(mesh)==triangle_digest(trimesh.load(old_mesh_path,force='mesh',process=False)))
                row.update(old_full_bbx_l=old_pre['evaluation']['with_full_support']['volume_litres'],
                           old_material_cm3=old_pre['evaluation']['material_full_cm3'],
                           old_mesh_sha256=old_mesh_sha,full_solid_geometry_equal=same_mesh,
                           layout_arrays_equal=arrays_equal(folder/'layout.npz',old_folder/'layout.npz'),
                           full_force_masks_equal=arrays_equal(folder/pre['full_force_certificate'],
                               old_folder/old_pre['full_force_certificate']))
            draw.text((12,12),label+'  PASS',fill='#263238')
            draw.text((12,510),f"{case['full_bbx_l']:.3f} L | {case['material_cm3']:.1f} cm3 | O={case['pose_count']} S={case['support_states']} J={case['juxtaposed']}",fill='#263238')
        else:
            title='  RUNTIME UNRESOLVED' if case.get('error') else '  SEARCH FAIL'
            draw.text((12,12),label+title,fill='#a82727')
            draw.text((12,80),case['status'],fill='#a82727')
            if case.get('oracle_status'):
                draw.text((12,110),'Oracle: '+case['oracle_status'],fill='#a82727')
            if case.get('error'):
                draw.text((12,140),'Runtime unresolved; no physical infeasibility conclusion',fill='#a82727')
        name=label.replace('/','__')+'.png';tile.save(pictures/name)
        folder=BASE/label/'step4/step4.2'/args.name
        row['temporary_cache_removed']=not any((folder/name).exists()
            for name in ('.search_runtime','pso_memory.npz','pso_refinement_memory.npz'))
        row['figure']=str((pictures/name).relative_to(ROOT));rows.append(row);tiles.append(tile)
    columns=4;sheet=Image.new('RGB',(640*columns,530*math.ceil(len(tiles)/columns)),'white')
    for i,tile in enumerate(tiles):sheet.paste(tile,((i%columns)*640,(i//columns)*530))
    sheet.save(system/'all_complete_supports.png')
    matched=[r for r in rows if r['passed'] and r['old_passed']]
    compared=[r for r in matched if r['seconds'] is not None and r['old_seconds'] is not None
              and not r['duration_is_incomplete']
              and not r['duration_is_detection_upper_bound'] and not r['old_duration_is_detection_upper_bound']]
    summary=dict(planned=52,completed=52,passed=sum(r['passed'] for r in rows),
        previous_actual_completed=len(older),matched_successes=len(matched),
        runtime_unresolved=[dict(id=r['id'],status=r['status'],error=r['error'])
            for r in rows if not r['passed'] and (r['error'] or r['status'] in ('runtime_failure','unresolved'))],
        previous_passed=sum(bool(r.get('passed')) for r in older.values()),
        previous_runtime_failures=[dict(id=r['id'],status=r.get('status'),error=r.get('error'))
            for r in older.values() if r.get('status')=='runtime_failure'],
        compared_worker_time_groups=len(compared),
        identical_bbx_groups=sum(abs(r['full_bbx_l']-r['old_full_bbx_l'])<=1e-8 for r in matched),
        identical_material_groups=sum(abs(r['material_cm3']-r['old_material_cm3'])<=1e-5 for r in matched),
        identical_layout_groups=sum(r['layout_arrays_equal'] for r in matched),
        identical_full_solid_groups=sum(r['full_solid_geometry_equal'] for r in matched),
        identical_full_force_masks_groups=sum(r['full_force_masks_equal'] for r in matched),
        temporary_cache_removed_groups=sum(r['temporary_cache_removed'] for r in rows),
        current_batch_wall_seconds=batch['seconds'],
        previous_batch_wall_seconds=previous_batch['seconds'] if previous_batch['complete'] else None,
        mathematical_search_options_equal=True,
        previous_seconds=sum(r['old_seconds'] for r in compared if r['old_seconds'] is not None),
        current_seconds=sum(r['seconds'] for r in compared if r['seconds'] is not None),
        gpu_used_groups=sum(bool(r['gpu'] and r['gpu']['used']) for r in rows),
        comparison='Same mathematical search, budget, physical inputs and complete-solid guards; different shared-machine load and scheduling')
    summary['measured_worker_time_ratio']=(summary['previous_seconds']/summary['current_seconds']
        if summary['current_seconds'] else None)
    evidence=system/'previous_runtime_failure_evidence.json'
    data=dict(complete=True,passed=True,summary=summary,cases=rows,option_differences=option_differences,
        previous_runtime_failure_evidence=read(evidence) if evidence.exists() else None,
        png_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in list(pictures.glob('*.png'))+[system/'all_complete_supports.png']})
    (system/'acceleration_comparison.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    lines=['# 缓存与GPU加速：全部52组Step4','',
        f"52组均有执行结果，{summary['passed']}/52通过完整实体验收，运行未解决{len(summary['runtime_unresolved'])}组。计划预算为32粒子／最多8轮冷搜索＋4轮PSO、seed42；资源退出不代表完成全部搜索轮次，也不代表物理不可行。原始需求与容差保持。",'',
        '[全部52组完整支撑／失败状态](data/system_'+args.name+'/all_complete_supports.png)', '',
        f"上一版同完整实体验收源已结束{len(older)}/52，通过{summary['previous_passed']}组、运行异常{len(summary['previous_runtime_failures'])}组。本轮与其共同通过{len(matched)}组，其中完整BBX一致{summary['identical_bbx_groups']}组、材料一致{summary['identical_material_groups']}组。",'',
        f"逐数组／实际三角面核对：布局一致{summary['identical_layout_groups']}组，完整支撑几何一致{summary['identical_full_solid_groups']}组，原完整需求mask一致{summary['identical_full_force_masks_groups']}组。三角面核对允许OBJ顶点／面编号顺序不同，不增加坐标容差。",'',
        f"共同通过且具有完整耗时的{len(compared)}组实际worker耗时统计和：旧{summary['previous_seconds']/3600:.3f}h→新{summary['current_seconds']/3600:.3f}h，比值{summary['measured_worker_time_ratio'] or 0.:.2f}。运行异常和只有检测时间上界的记录不纳入成功耗时对比。这不是批次墙钟时间，且并发负载／调度不同，不能当作受控全链路加速比。",'',
        f"本次52组批次墙钟时间{summary['current_batch_wall_seconds']/3600:.3f}h；上一版完整批次"+
            (f"{summary['previous_batch_wall_seconds']/3600:.3f}h。" if summary['previous_batch_wall_seconds'] is not None else
             f"未完成，仅有{len(older)}/52条实际结果，无完整批次墙钟时间。")+
            '本次墙钟包含排队、并发负载、用户要求的计时调查及暂停时段；两批时间不能作受控速度比较。','',
        f"最终worker记录中确认启用GPU的组数{summary['gpu_used_groups']}；GPU边界需求仍执行原CPU长双精度重放。缓存／GPU计数以返回记录及已提交检查点为准，被内核终止的未提交片段计数可能不完整；重试耗时另计入总执行时间。",'',
        f"临时搜索缓存已清理{summary['temporary_cache_removed_groups']}/52组，最终模型、完整需求证书及耗时记录保留。未解决异常的检查点保留用于恢复。",'',
        '| 组／实际成品 | 状态 | 完整BBX L | 材料 cm³ | 旧→新耗时s |',
        '| --- | --- | ---: | ---: | ---: |']
    for r in rows:
        figure=str(Path(r['figure']).relative_to(BASE.relative_to(ROOT)))
        old='—' if r['old_seconds'] is None else f"{r['old_seconds']:.1f}"
        if r['old_passed'] is False:old+=' ('+str(r['old_status'])+')'
        new='—' if r['seconds'] is None else f"{r['seconds']:.1f}"
        if r['duration_is_detection_upper_bound']:new+=' (检测上界)'
        if r['duration_is_incomplete']:new+=' (中断片段计时不完整)'
        status='PASS' if r['passed'] else ('运行未解决' if r['error'] else
            'Oracle: '+str(r['oracle_status']) if r['force_passed'] else '受力未通过')
        lines.append(f"| [{r['id']}]({figure}) | {status} | "+
            (f"{r['full_bbx_l']:.4f} | {r['material_cm3']:.2f} | {old}→{new} |" if r['passed'] else f"— | — | {old}→{new} |"))
    lines+=['','[完整实体、输入、源码及原需求审计](data/system_'+args.name+'/evaluation.json)',
            '[逐组运行缓存、GPU与旧／新数据](data/system_'+args.name+'/acceleration_comparison.json)',
            '[逐组完成后的临时文件清理记录](data/system_'+args.name+'/completed_cache_cleanup.json)','']
    if evidence.exists():
        lines+=['[旧版资源退出及原始内核记录](data/system_'+args.name+'/previous_runtime_failure_evidence.json)','']
    (BASE/(args.name+'_acceleration_results.txt')).write_text('\n'.join(lines))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
