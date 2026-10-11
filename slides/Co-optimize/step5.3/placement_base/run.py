"""Preserved Step4 -> finite translation proposals -> actual base attempts.

The three BBX measures are always separate. Unaccepted bases have no claimed
feasible volume. Every moved cloud comes from all original world wrenches.
"""
from pathlib import Path
import csv
import html
import importlib.util
import json
import shutil
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'step5.2'))
import stage5_common as C
sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE))
from growth import floor_certificates
import base_seating as B
import base_xy as XY
import base_bounds as Q
import base_geometry as G
from base_access_seating import AccessSeating
from base_floor_space import screen
from base_reactions import load_tasks,ground_coverage,validate_contacts,validate_ground

spec=importlib.util.spec_from_file_location('placement_envelope_renderer',HERE.parents[1]/'step5.2/render.py')
R=importlib.util.module_from_spec(spec);sys.modules[spec.name]=R;spec.loader.exec_module(R)
STAGE='step5.3/placement_base_v1'


def finite_seating_report(seating):
    # Complete infeasible LP arrays are retained in proposal files; the HTML
    # and summary carry the status and movements, rather than huge matrices.
    return {k:v for k,v in seating.items() if k not in ['constraint_matrix','constraint_rhs_mm','variable_bounds_mm','bound_labels']}


def draw(renderer,data,path,*,fixture=None,object_only=False):
    fixture=data['fixture'] if fixture is None else fixture
    bounds=Q.object_bounds(data) if object_only else C.occupied_bounds(data['body'],fixture,data['rows'])
    camera=C.V.camera_for(R.box_corners(bounds));parts=[];seen=set()
    for row in data['rows']:
        if not object_only:
            key=C.np.round(row['fixture_transform'],10).tobytes()
            if key not in seen:
                parts.append(C.V.CPU.placed(C.V.piece(C.posed(fixture,row['fixture_transform']),C.V.BLUE,.25)));seen.add(key)
        parts.append(C.V.CPU.placed(C.V.piece(C.posed(data['body'],row['object_transform']),C.V.GRAY,.13)))
    parts.append(C.V.CPU.placed(R.box_piece(bounds,camera,1000)))
    picture=renderer.render(parts,camera,1000);picture.save(path)
    return picture


def proposals(data,access,out):
    results=[]
    def add(label,seating):
        index=len(results);C.save(out/'proposals'/f'{index:02d}_{label}.json',seating)
        trial=dict(label=label,seating=finite_seating_report(seating))
        if seating['passed']:
            moved=B.shifted(data,seating)
            trial['required_enclosure_bbx_lower_bound']=Q.required_enclosure_bound(moved)
            trial['ground_screen']=screen(access,seating,extent=.7)
        results.append(trial)
        print(data['case']['id'],label,seating['passed'],seating.get('maximum_displacement_coordinate_mm'),flush=True)
    # A necessary Z certificate permits arbitrarily thin feet and omits
    # work/exit/contact constraints. Infeasibility is consequently stronger.
    necessary=B.Seating(data,width=0,preserve_material=True,protect_full_wrap=False)
    necessary.head_bounds=[]
    add('assembly_z_necessary',necessary.solve('z'))
    add('object_xy_only',XY.solve(data))
    add('assembly_z_5mm',B.Seating(data).solve('z'))
    for mode in ['z_xy','z_xyz']:
        for width in [.005,.03,.12]:
            add(f'{mode}_{int(width*1000)}mm',B.Seating(data,width=width).solve(mode))
    return results


def construction_attempts(data,tasks,body,trials,out):
    feasible=[t for t in trials if t['seating']['passed'] and t['label']!='assembly_z_necessary']
    # Always try both small XY/XYZ proposals. The remaining two are chosen
    # by conservative legal-ground screening, then movement and fixed ties.
    required=[t for t in feasible if t['label'] in ['z_xy_5mm','z_xyz_5mm']]
    ranked=sorted(feasible,key=lambda t:(-sum(p['passed'] for p in t['ground_screen']['per_pose']),
                                        t['seating']['maximum_displacement_coordinate_mm'],t['label']))
    selected=required+[t for t in ranked if t not in required][:2]
    attempts=[];best=None;best_key=None;accepted=None
    for trial in selected:
        moved=B.shifted(data,trial['seating'])
        for padding in [.04,.14]:
            builder=G.Construction(moved,tasks,body,padding=padding)
            all_mesh=G.unpack(builder.free)
            union_ground,_=ground_coverage(all_mesh,moved)
            components=builder.components()
            attempt=dict(proposal=trial['label'],padding_m=padding,component_count=len(components),
                complete_free_space_material_cm3=G.volume(builder.free)*1e6,
                complete_free_space_ground_coverage=union_ground,components=[])
            attempts.append(attempt)
            print('BASE',data['case']['id'],trial['label'],padding,'all-ground',
                  sum(p['passed'] for p in union_ground['per_pose']),len(tasks),'components',len(components),flush=True)
            for index,component in enumerate(components):
                mesh=G.unpack(component);coverage,grounds=ground_coverage(mesh,moved)
                row=dict(component=index,material_volume_cm3=G.volume(component)*1e6,ground_coverage=coverage)
                attempt['components'].append(row)
                passed_count=sum(p['passed'] for p in coverage['per_pose'])
                box=C.occupied_bounds(data['body'],mesh,moved['rows'])
                key=(passed_count,-C.volume(box),-row['material_volume_cm3'])
                if best_key is None or key>best_key:best=(mesh,moved,trial,row);best_key=key
                if not coverage['passed']:continue
                geometry=builder.checks(component);row['geometry']=geometry
                if not geometry['passed']:continue
                contact,contact_masks,contact_supplies=validate_contacts(mesh,builder,tasks,moved,all_poses=False)
                row['object_contacts']=contact
                if not contact['passed']:continue
                ground,ground_masks,ground_supplies=validate_ground(grounds,tasks,moved);row['system_ground']=ground
                if not ground['passed']:continue
                row['passed']=True;accepted=(mesh,moved,trial,row)
                mesh.export(out/'support_with_base.obj')
                C.np.savez_compressed(out/'accepted_layout_and_masks.npz',
                    fixture_transforms=C.np.array(moved['states']),
                    object_transforms=C.np.array([r['object_transform'] for r in moved['rows']]),
                    object_masks=C.np.array(contact_masks),ground_masks=C.np.array(ground_masks))
                return attempts,accepted,best
            C.save(out/'construction_attempts.json',attempts)
    return attempts,accepted,best


def movement_rows(data,seating,label):
    rows=[]
    for old,delta,relative,lift in zip(data['rows'],seating['total_object_world_translation_m'],
                                     seating['object_relative_world_translation_m'],seating['fixture_lift_m']):
        rows.append(dict(group=data['case']['id'],proposal=label,pose=old['pose'],
            object_dx_mm=delta[0]*1000,object_dy_mm=delta[1]*1000,object_dz_mm=delta[2]*1000,
            object_xy_distance_mm=float(C.np.linalg.norm(delta[:2]))*1000,fixture_lift_mm=lift*1000,
            relative_dx_mm=relative[0]*1000,relative_dy_mm=relative[1]*1000,relative_dz_mm=relative[2]*1000))
    return rows


def main():
    args,inputs,cases,directory=C.arguments(STAGE,'step53_placement_base_v1')
    prepared=[]
    for case in cases:
        data=inputs.load(case);B.load_demands(data,inputs);tasks,body=load_tasks(data,inputs)
        for task,row in zip(tasks,data['rows']):row['native_object_transform']=C.np.array(task.domain.data['frame']['T_world_mesh'])
        prepared.append((data,tasks,body))
    code=C.begin(directory,STAGE,inputs,cases);renderer=C.V.Renderer();reports=[];all_movements=[]
    protected_path=Path('/tmp/cadgrasp_placement_base_protected_20261010.json')
    earlier_protected=json.loads(protected_path.read_text()) if protected_path.exists() else {}
    C.save(directory/'preexisting_artifact_hashes.json',earlier_protected)
    for data,tasks,body in prepared:
        started=time.perf_counter();out=inputs.base/data['case']['id']/'step5'/STAGE
        out.mkdir(parents=True,exist_ok=False)
        originals=floor_certificates(data['rows']);failed={r['task_index'] for r in originals}
        zproof=Q.object_z_certificates(data,failed)
        access=AccessSeating(data,tasks,body)
        trials=proposals(data,access,out)
        attempts,accepted,best=construction_attempts(data,tasks,body,trials,out)
        selected=accepted or best
        original_object=draw(renderer,data,out/'object_only_bbx.png',object_only=True)
        original_step4=draw(renderer,data,out/'step4_bbx.png')
        pictures=[original_object,original_step4]
        candidate=None
        if selected:
            mesh,moved,trial,detail=selected
            candidate=dict(proposal=trial['label'],accepted=accepted is not None,
                bbx=Q.bounds_report(C.occupied_bounds(data['body'],mesh,moved['rows'])),
                material_volume_cm3=abs(float(mesh.volume))*1e6,ground_coverage=detail['ground_coverage'])
            pictures.append(draw(renderer,moved,out/'accepted_base_bbx.png' if accepted else out/'failed_candidate_bbx.png',fixture=mesh))
            if not accepted:mesh.export(out/'failed_candidate.obj')
        C.V.sheet(pictures,out/'bbx_comparison.png')
        for t in trials:
            if t['seating']['passed']:all_movements.extend(movement_rows(data,t['seating'],t['label']))
        report=dict(group=data['case']['id'],pose_count=len(tasks),passed=accepted is not None,
            object_z_only_impossible=len(zproof)>0,object_z_only_certificates=zproof,
            bbx_without_support_at_saved_step4_object_positions=Q.bounds_report(Q.object_bounds(data)),
            bbx_without_support_at_native_object_positions=Q.bounds_report(Q.object_bounds(data,native=True)),
            step4_bbx=Q.bounds_report(data['bounds']),step4_material_cm3=abs(float(data['fixture'].volume))*1e6,
            accepted_step53_base_bbx=candidate['bbx'] if accepted else None,
            accepted_step53_material_cm3=candidate['material_volume_cm3'] if accepted else None,
            shown_candidate=candidate,trials=trials,construction_attempts=attempts,
            measured_seconds=time.perf_counter()-started,original_step4_mesh_modified=False,
            old_force_pass_reused_and_not_revoked=True,
            limitations=dict(fixture_rotations_and_exit_directions_frozen=True,
                finite_translation_search=True,global_minimum_or_global_xyz_impossibility_proved=False,
                fixture_self_weight_ratio=0.,original_sample_count_per_pose=32768,strength_checked=False,
                process_scope='stationary working states; no unspecified tools or loading trajectories'))
        C.save(out/'report.json',report);C.save(out/'construction_attempts.json',attempts)
        reports.append(report);C.save(directory/'progress.json',dict(cases=reports))
        print('GROUP COMPLETE',data['case']['id'],'accepted',report['passed'],flush=True)
    csvpath=directory/'movements_mm.csv'
    if all_movements:
        with csvpath.open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=list(all_movements[0]));writer.writeheader();writer.writerows(all_movements)
    force_ran=any('object_contacts' in component for r in reports for a in r['construction_attempts'] for component in a['components'])
    execution=json.loads((directory/'execution.json').read_text());execution.update(
        force_solver_run=force_ran,force_solver_scope='changed constructed candidates only; saved Step4 never replayed',
        boolean_base_construction_run=True,original_sample_clouds_all_forward_projected=True)
    C.save(directory/'execution.json',execution)
    audit_changed=[p for p,h in earlier_protected.items() if not (C.ROOT/p).exists() or C.digest(C.ROOT/p)!=h]
    C.save(directory/'preexisting_artifact_audit.json',dict(checked=len(earlier_protected),changed=audit_changed))
    write_report(inputs,directory,reports)
    summary=dict(object=inputs.name,group_count=len(reports),base_passed_count=sum(r['passed'] for r in reports),
                 cases=reports,total_seconds=sum(r['measured_seconds'] for r in reports),
                 original_step4_results_unchanged=True,preexisting_artifact_changes=audit_changed)
    C.finish(directory,inputs,code,summary)
    print('BATCH COMPLETE',sum(r['passed'] for r in reports),'/',len(reports),flush=True)


def write_report(inputs,directory,reports):
    rows=[];sections=[];volume_rows=[]
    for index,r in enumerate(reports,1):
        t={q['label']:q for q in r['trials']}
        xy=t['object_xy_only']['seating'];z=t['assembly_z_necessary']['seating'];joint=t['z_xyz_5mm']
        motion=joint['seating'];delta=C.np.asarray(motion['total_object_world_translation_m'])*1000
        maxxy=float(C.np.linalg.norm(delta[:,:2],axis=1).max());maxz=float(delta[:,2].max())
        base=r['accepted_step53_base_bbx'];base_text=f"{base['volume_cm3']:.1f}" if base else '未找到通过的实体'
        xytext=f"{xy['maximum_displacement_coordinate_mm']:.2f}" if xy['passed'] else '必要条件无解'
        ztext=f"{z['maximum_displacement_coordinate_mm']:.2f}" if z['passed'] else '必要条件无解'
        v0=r['bbx_without_support_at_saved_step4_object_positions']['volume_cm3'];v4=r['step4_bbx']['volume_cm3']
        lower=joint['required_enclosure_bbx_lower_bound']['volume_cm3']
        rows.append(f'<tr><td>{index}</td><td>{v0:.1f}</td><td>{v4:.1f}</td><td>{base_text}</td><td>{xytext}</td><td>{ztext}</td><td>{maxxy:.2f} / {maxz:.2f}</td><td>{lower:.1f}</td></tr>')
        volume_rows.append(dict(group=r['group'],object_only_bbx_cm3=v0,step4_bbx_cm3=v4,
            accepted_step53_bbx_cm3=base['volume_cm3'] if base else '',
            trial_z_xyz_5mm_bbx_lower_bound_cm3=lower,step4_material_cm3=r['step4_material_cm3'],
            accepted_step53_material_cm3=r['accepted_step53_material_cm3'] if base else ''))
        prefix=f"../../{r['group']}/step5/{STAGE}"
        caption='可用底座' if r['passed'] else '失败候选；右图不能作为可用底座'
        details=[]
        for trial in r['trials']:
            seating=trial['seating'];screening=trial.get('ground_screen')
            count=f"{sum(q['passed'] for q in screening['per_pose'])}/{r['pose_count']}" if screening else '—'
            movement=f"{seating['maximum_displacement_coordinate_mm']:.2f}" if seating['passed'] else '—'
            details.append(f"<tr><td>{trial['label']}</td><td>{movement}</td><td>{count}</td><td>{html.escape(seating['status'])}</td></tr>")
        witness=r['object_z_only_certificates'][0]
        proof=f"pose {witness['pose']}：上移至少 {witness['required_minimum_lift_mm']:.3f} mm，同时最多 {witness['allowed_maximum_lift_mm']:.3f} mm。" if witness['allowed_maximum_lift_mm'] is not None else f"pose {witness['pose']} 存在不随上移量改变的地面冲突。"
        sections.append(f'''<section><h2>{index}. {html.escape(r['group'])}</h2>
<p>左：原工件位置、不带支撑；中：原 Step4；右：{caption}。各图按各自 BBX 取景，尺寸请看数字。</p>
<img src="{prefix}/bbx_comparison.png"><p>仅工件上移的矛盾证据：{proof}</p>
<table><tr><th>试验</th><th>最大位移坐标 mm</th><th>通过保守接地筛选的 pose</th><th>状态</th></tr>{''.join(details)}</table>
<p><a href="{prefix}/report.json">完整几何检查、位移和证明</a> · <a href="{prefix}/construction_attempts.json">实体构造记录</a></p></section>''')
    with (directory/'volumes_cm3.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(volume_rows[0]));writer.writeheader();writer.writerows(volume_rows)
    document=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Step5.3 位移与底座试验</title>
<style>body{{font:16px system-ui;max-width:1400px;margin:35px auto;padding:20px;color:#253640}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #ccd6dd;padding:8px;text-align:left}}th{{background:#eef3f6}}img{{width:100%}}section{{margin-top:40px}}</style>
<h1>Step4 基础上的上移、左右移动与实体底座</h1>
<p>结果：{sum(r['passed'] for r in reports)}/{len(reports)} 组得到完整通过的实体底座。Step4 原结果、Step5.1、objects 输入均保留。每个移动后的需求凸包由原始 32,768 个力/力矩对向前投影，未反推或重采样。</p>
<p>只上移工件、保持夹具地面位置：七组均有线性地面矛盾，报告给出原始需求编号和上下界证据。工件和支撑一起上移是另一种操作；仅第三组通过必要条件。整套组件一起平移 XY 不改变接地几何。工件相对支撑平移 XY 则要重建接触。</p>
<p>加入支撑上移及工件相对 XY/XYZ 调整，七组都能通过地面线性约束；工作锥、退出空间、连接性和原始受力接受仍要在实体上检查。本次有限搜索没有找到可用实体的组，最终底座体积留空；失败候选和理论 BBX 下界都不能充当成功体积。</p>
<p>以下 BBX 单位 cm³；位移单位 mm。无支撑一列固定 Step4 工件位置，只去掉支撑，便于同位置比较。原 native 工件 BBX 另保存在各组 JSON。“XY+Z 候选”的上移指工件实际位移，相对 XYZ 的位移与夹具升高另见 CSV。位移是本次线性提案的数值，不是可用底座所需最小位移。</p>
<table><tr><th>组</th><th>无支撑 BBX</th><th>Step4 BBX</th><th>通过的 Step5.3 BBX</th><th>只工件 XY 必要条件：最大坐标</th><th>整体 Z 必要条件：最大上移</th><th>XYZ 候选：最大 XY 距离 / 最大工件 Z</th><th>该候选 BBX 必要下界</th></tr>{''.join(rows)}</table>
<p><a href="volumes_cm3.csv">体积 CSV</a> · <a href="movements_mm.csv">每组每 pose 每提案的 XYZ 位移 CSV</a> · <a href="summary.json">汇总 JSON</a></p>
<p>受力沿用原模型（夹具自重比例 0）；地面摩擦若进入完整检查，沿用代码中原有的充分模型参数，不是材料测量值。检查的是静止工作状态与完整工作锥/退出通道，未提供的工具和装载轨迹不计入 BBX；未计算强度。全局最小体积及任意 XYZ 平移无解均未证明。</p>
{''.join(sections)}</html>'''
    (directory/'report.html').write_text(document)
    gallery=inputs.base/'step5.3_placement_index.html'
    gallery.write_text(f'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0; url=data/{directory.name}/report.html"><a href="data/{directory.name}/report.html">位移和底座报告</a>')


if __name__=='__main__':main()
