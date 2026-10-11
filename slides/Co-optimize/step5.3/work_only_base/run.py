"""Work-only enclosure -> +Z seating -> updated coverage -> optional feedback.

Original Step3/4/5.1/5.2 and all preceding Step5.3 trials are read-only.
Use new output names; write both stage results and a single combined report.
"""
from pathlib import Path
import csv
import html
import importlib.util
import json
import sys
import time

HERE = Path(__file__).resolve().parent
COOPT = HERE.parents[1]
for path in [COOPT/'step5.2', HERE.parent/'placement_base', COOPT/'step5.4', HERE]:
    sys.path.insert(0, str(path))
import stage5_common as C
import base_geometry as G
import base_seating as B
import base_bounds as Q
from base_reactions import load_tasks, validate_contacts, validate_ground
from geometry import WorkOnlyBase, enclosure
from policy import POLICY
import z_seating as Z

spec = importlib.util.spec_from_file_location('work_only_envelope_render', COOPT/'step5.2/render.py')
R = importlib.util.module_from_spec(spec); sys.modules[spec.name] = R; spec.loader.exec_module(R)
STAGE = 'step5.3/work_only_v2'
SEAT_STAGE = 'step5.4/work_only_v2'
PADDINGS = [0., .02, .06, .14, .5, 1.]


def thin_seating(result):
    return {k: v for k, v in result.items() if k not in
            ['constraint_matrix', 'constraint_rhs_mm', 'variable_bounds_mm', 'bound_labels']}


def candidates(builder, data):
    values = []
    for index, value in enumerate(builder.components()):
        mesh = G.unpack(value); cover = enclosure(mesh, data)
        checks = builder.checks(value)
        if checks['passed']:
            values.append((value, mesh, index, cover, checks))
    # Prefer the most old material retained; fixed deterministic tie-breaks.
    return sorted(values, key=lambda v: (-int(v[3]['passed']),
                                        -sum(r['passed'] for r in v[3]['per_pose']),
                                        -v[4]['retained_old_material_cm3'],
                                        v[4]['material_volume_cm3'], v[2]))


def evaluate(value, mesh, builder, original, moved, label, heights, coverage, checks):
    seating, grounds = Z.check(mesh, moved)
    final_cover = enclosure(mesh, moved)
    row = dict(label=label, heights_mm=(np.asarray(heights)*1000).tolist(),
        step53_enclosure=coverage, step53_geometry=checks, step54=seating,
        updated_convex_enclosure=final_cover,
        geometry_passed=checks['passed'] and seating['passed'] and final_cover['passed'],
        passed=False, bbx=Q.bounds_report(C.occupied_bounds(original['body'], mesh, moved['rows'])))
    contact_masks, ground_masks = [], []
    if row['geometry_passed']:
        contact, contact_masks, _ = validate_contacts(mesh, builder, builder.tasks, moved, all_poses=True)
        row['changed_candidate_contacts'] = contact
        if contact['passed']:
            ground, ground_masks, _ = validate_ground(grounds, builder.tasks, moved)
            row['changed_candidate_whole_ground'] = ground
            row['passed'] = ground['passed']
        if row['passed']:
            transfer, frames = Z.transfer(mesh, moved)
            row['empty_fixture_transfer'] = transfer
    return row, contact_masks, ground_masks


# Keep numpy bound explicitly: all scripts use the same loaded numerical stack.
np = C.np


def attempt_group(data, tasks, body, out53, out54):
    history = []; first = None; best = None; best_key = None; accepted = None
    def record(label, builder, provisional, fixed_heights=None):
        nonlocal first, best, best_key, accepted
        choices = candidates(builder, provisional)
        if not choices:
            row = dict(label=label, step53_passed=False,
                reason='no_single_connected_work_clear_component_encloses_all_hulls',
                component_count=len(builder.components()), passed=False)
            history.append(row); C.save(out54/'attempts.json', history)
            print('ENCLOSURE FAIL', data['case']['id'], label, flush=True)
            return None
        # At most two fully covering components, after old-material preference.
        updated_h = None
        for value, mesh, index, cover, checks in choices[:2]:
            if first is None and cover['passed']:
                first = (mesh, provisional, cover, checks)
                mesh.export(out53/'enclosure.obj')
                C.save(out53/'first_enclosure.json', dict(enclosure=cover, geometry=checks,
                    layout_heights_mm=[r.get('fixture_lift_m', 0.)*1000 for r in provisional['rows']],
                    floor_acceptance_deferred=True))
            heights = Z.minimum_lifts(mesh, data) if fixed_heights is None else np.asarray(fixed_heights)
            if fixed_heights is None:
                heights = np.maximum(heights, [r.get('fixture_lift_m', 0.) for r in provisional['rows']])
            moved = Z.move(data, heights)
            row, contact_masks, ground_masks = evaluate(value, mesh, builder, data, moved,
                label, heights, cover, checks)
            row['component_index'] = index; row['step53_passed'] = cover['passed']
            history.append(row); C.save(out54/'attempts.json', history)
            count = sum(p['passed'] for p in row['step54']['updated_ground_coverage']['per_pose'])
            key = (row['passed'], row['geometry_passed'], count, -row['bbx']['volume_cm3'],
                   checks['retained_old_material_cm3'])
            if best_key is None or key > best_key:
                best_key = key; best = (mesh, moved, row)
            print('SEAT', data['case']['id'], label, 'ground', count, '/', len(tasks),
                  'geometry', row['geometry_passed'], 'full', row['passed'], flush=True)
            if row['passed']:
                accepted = best
                mesh.export(out54/'support_with_base.obj')
                C.np.savez_compressed(out54/'accepted_layout.npz',
                    fixture_transforms=np.asarray(moved['states']),
                    object_transforms=np.asarray([r['object_transform'] for r in moved['rows']]),
                    heights_m=heights, object_masks=np.asarray(contact_masks), ground_masks=np.asarray(ground_masks))
                transfer, frames = Z.transfer(mesh, moved)
                C.np.savez_compressed(out54/'empty_transfer_frames.npz', fixture_transforms=frames)
                return heights
            updated_h = heights
        return updated_h

    # Literal sequential method. Step5.3 has no floor obstacle or exit sweep.
    for padding in PADDINGS:
        heights = np.zeros(len(tasks))
        for iteration in range(3):
            provisional = Z.move(data, heights)
            builder = WorkOnlyBase(provisional, tasks, body, padding=padding)
            next_h = record(f'sequential_pad{round(padding*1000)}_iteration{iteration}', builder, provisional)
            if accepted:
                return history, first, best, accepted, []
            if next_h is None or np.max(np.abs(next_h-heights)) <= 1e-9:
                break
            heights = next_h

    # Deterministic feedback. Height/floor envelopes are owned by Step5.4;
    # Step5.3 rebuilds the proposed base inside the returned design shape.
    # Each completed component is still accepted only by the final checks.
    proposals = Z.height_proposals(data)
    for proposal in proposals:
        C.save(out54/'proposals'/f"{proposal['label']}.json", proposal['result'])
        result = proposal['result']
        print('Z PROPOSAL', data['case']['id'], proposal['label'], result['passed'], flush=True)
        if not result['passed']:
            continue
        heights = np.asarray(result['fixture_lift_m']); moved = Z.move(data, heights)
        for padding in PADDINGS:
            builder = WorkOnlyBase(moved, tasks, body, padding=padding)
            returned = Z.design_envelope(builder, moved)
            builder = builder.in_envelope(returned)
            record(f"feedback_{proposal['label']}_pad{round(padding*1000)}", builder, moved, heights)
            if accepted:
                return history, first, best, accepted, proposals
    return history, first, best, accepted, proposals


def draw_group(renderer, data, mesh, path):
    return R.reference(renderer, data, path, fixture=mesh, size=1000)


def preexisting_manifest(inputs):
    source = Path('/tmp/cadgrasp_placement_base_protected_20261010.json')
    records = json.loads(source.read_text()) if source.exists() else {}
    # Snapshot the current bytes of previous results, not only an old session.
    for p in list(records):
        if (C.ROOT/p).is_file():
            records[p] = C.digest(C.ROOT/p)
    folders = [inputs.base/'data'] + [inputs.base/c['id']/'step5' for c in inputs.cases]
    for folder in folders:
        for file in folder.rglob('*'):
            if file.is_file() and '__pycache__' not in file.parts:
                records[C.relative(file)] = C.digest(file)
    return records


def main():
    args, inputs, cases, directory = C.arguments(STAGE, 'step53_work_only_then_z_v2')
    # Avoid including the newly created batch directory in the old-file audit.
    old = preexisting_manifest(inputs)
    prepared = []
    for case in cases:
        if (inputs.base/case['id']/'step5'/SEAT_STAGE).exists():
            raise ValueError('Existing Step5.4 output is protected')
        data = inputs.load(case); B.load_demands(data, inputs); tasks, body = load_tasks(data, inputs)
        prepared.append((data, tasks, body))
    code = C.begin(directory, STAGE, inputs, cases)
    C.save(directory/'preexisting_artifacts.json', old)
    C.save(directory/'policy.json', POLICY.record())
    renderer = C.V.Renderer(); reports = []; movements = []
    for data, tasks, body in prepared:
        start = time.perf_counter(); group = data['case']['id']
        out53 = inputs.base/group/'step5'/STAGE; out54 = inputs.base/group/'step5'/SEAT_STAGE
        out53.mkdir(parents=True, exist_ok=False); out54.mkdir(parents=True, exist_ok=False)
        (out54/'demands').mkdir()
        attempts, first, best, accepted, proposals = attempt_group(data, tasks, body, out53, out54)
        pictures = []
        if first:
            mesh, layout, cover, checks = first
            pictures.append(draw_group(renderer, layout, mesh, out53/'enclosure.png'))
        elif best:
            # Show the failed enclosure explicitly; never label it a result.
            mesh, moved, row = best
            mesh.export(out53/'failed_enclosure.obj')
            pictures.append(draw_group(renderer, moved, mesh, out53/'failed_enclosure.png'))
        if best:
            mesh, moved, row = best
            name = 'accepted_base' if accepted else 'failed_candidate'
            if not accepted:
                mesh.export(out54/'failed_candidate.obj')
            pictures.append(draw_group(renderer, moved, mesh, out54/f'{name}.png'))
            coverage = row['step54']['updated_ground_coverage']['per_pose']
            for index, (old_row, new_row) in enumerate(zip(data['rows'], moved['rows'])):
                movements.append(dict(group=group, pose=old_row['pose'],
                    fixture_lift_mm=row['heights_mm'][index], object_lift_mm=row['heights_mm'][index],
                    object_dx_mm=0., object_dy_mm=0., relative_object_shift_mm=0.,
                    actual_ground_coverage_passed=coverage[index]['passed'],
                    final_candidate_passed=row['passed'], selected_attempt=row['label']))
                # Save exact updated original samples, never a refit.
                np.savez_compressed(out54/'demands'/f"{new_row['pose']}.npz",
                    wrench_world_origin=new_row['wrench_world_origin'], floor_demands_world_m=new_row['cloud'],
                    hull_indices=new_row['hull_original_demand_indices'],
                    T_world_fixture=new_row['fixture_transform'], T_world_object=new_row['object_transform'])
        if pictures:
            C.V.sheet(pictures, out54/'process.png')
        report = dict(group=group, pose_count=len(tasks), policy=POLICY.record(),
            step53_enclosure_found=first is not None, step54_geometry_passed=bool(best and best[2]['geometry_passed']),
            passed=accepted is not None, selected_candidate=best[2] if best else None,
            attempts=attempts, height_proposals=[dict(label=p['label'], result=thin_seating(p['result'])) for p in proposals],
            object_only_bbx=Q.bounds_report(Q.object_bounds(data)), step4_bbx=Q.bounds_report(data['bounds']),
            accepted_step54_bbx=best[2]['bbx'] if accepted else None,
            original_step4_force_pass_reused_and_not_revoked=True,
            original_step4_mesh_modified=False, measured_seconds=time.perf_counter()-start,
            limitations=dict(finite_window_and_height_family=True, global_impossibility_proved=False,
                hull_enclosure_alone_is_not_actual_ground_coverage=True,
                strength_checked=False, full_exit_clearance_intentionally_not_required=True,
                transport_checked_only_for_defined_empty_fixture_path=True))
        C.save(out53/'report.json', dict(group=group, found=first is not None,
            policy=POLICY.record(), first_enclosure='first_enclosure.json' if first else None,
            seating_report=f'../../{SEAT_STAGE}/report.json'))
        C.save(out54/'report.json', report); reports.append(report)
        C.save(directory/'progress.json', dict(cases=reports))
        print('GROUP COMPLETE', group, '53', first is not None, '54', report['passed'], flush=True)
    if movements:
        with (directory/'movements_mm.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(movements[0])); writer.writeheader(); writer.writerows(movements)
    changed = [p for p, digest in old.items() if not (C.ROOT/p).is_file() or C.digest(C.ROOT/p) != digest]
    C.save(directory/'preexisting_artifact_audit.json', dict(checked=len(old), changed=changed))
    if changed:
        raise ValueError(f'Preexisting artifacts changed: {changed[:5]}')
    execution = json.loads((directory/'execution.json').read_text())
    execution.update(boolean_base_construction_run=True,
        force_solver_run=any('changed_candidate_contacts' in a for r in reports for a in r['attempts']),
        force_solver_scope='new constructed candidates only', policy=POLICY.record())
    C.save(directory/'execution.json', execution)
    write_html(directory, reports)
    C.finish(directory, inputs, code, dict(group_count=len(reports),
        step53_found_count=sum(r['step53_enclosure_found'] for r in reports),
        step54_geometry_passed_count=sum(r['step54_geometry_passed'] for r in reports),
        base_passed_count=sum(r['passed'] for r in reports), cases=reports,
        preexisting_artifact_count=len(old), preexisting_artifact_changes=changed))
    print('BATCH COMPLETE', sum(r['passed'] for r in reports), '/', len(reports), flush=True)


def write_html(directory, reports):
    rows = []; sections = []
    for i, report in enumerate(reports, 1):
        trial = report['selected_candidate']; count = 0; lift = '—'
        if trial:
            count = sum(r['passed'] for r in trial['step54']['updated_ground_coverage']['per_pose'])
            lift = f"{min(trial['heights_mm']):.2f}–{max(trial['heights_mm']):.2f}"
        accepted = report['accepted_step54_bbx']
        volume = f"{accepted['volume_cm3']:.1f}" if accepted else '未找到'
        rows.append(f"<tr><td>{i}</td><td>{'通过' if report['step53_enclosure_found'] else '未找到'}</td>"
                    f"<td>{count}/{report['pose_count']}</td><td>{lift}</td><td>{volume}</td>"
                    f"<td>{'通过' if report['passed'] else '未通过'}</td></tr>")
        prefix = f"../../{report['group']}/step5/{SEAT_STAGE}"
        picture = f"<img src='{prefix}/process.png'>" if trial else ''
        sections.append(f"<h2>{i}. {html.escape(report['group'])}</h2>"
            f"<p>左：Step5.3 工作锥避让包围候选；右：Step5.4 最佳候选。失败图不是可用底座。</p>"
            f"{picture}<p><a href='{prefix}/report.json'>完整记录</a> · "
            f"<a href='{prefix}/attempts.json'>全部构造尝试</a></p>")
    document = f"""<!doctype html><meta charset='utf-8'><title>Work-only base and +Z</title>
<style>body{{font:16px system-ui;max-width:1400px;margin:30px auto}}td,th{{padding:8px;border:1px solid #ccc}}
table{{border-collapse:collapse}}img{{max-width:100%}}</style><h1>Step5.3 工作锥避让 → Step5.4 +Z</h1>
<p>工作面全部方向锥和静态物体内部是硬禁区；允许底座与退出路径相交。前序结果均保留。
Step5.3 的三维材料凸包包围判据与 Step5.4 的实际接地覆盖分别报告。上移后重新正向计算每 pose 32768 个原始需求。</p>
<p>先试直接构造再上移；失败时由 Step5.4 提议高度与接地面设计包络，返回 Step5.3 重构。
这种反馈不会修改已导出的旧底座。正高度提案 LP 无解不代表所有着地分支无解。</p>
<p>过程路径：取出物体，空夹具先升高再旋转／移位，最后下降。若找到接受候选，保存连续避地面的解析半径证书。
没有把机器人、刀具或物体装卸轨迹包括在过程范围。</p>
<table><tr><th>组</th><th>5.3 包围与工作锥</th><th>5.4 实际接地覆盖</th><th>候选上移 mm</th>
<th>可用底座 BBX cm³</th><th>最终原需求检查</th></tr>{''.join(rows)}</table>
<p>有限确定性候选搜索，不宣称最小体积或全局无解。<a href='movements_mm.csv'>逐 pose 位移</a> ·
<a href='summary.json'>结果</a> · <a href='verification.json'>输入与执行源码验证</a></p>{''.join(sections)}"""
    (directory/'report.html').write_text(document)


if __name__ == '__main__':
    main()
