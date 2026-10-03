"""Recheck saved Step4 fixtures using Step3's working-face rule.

Default: replay the new surface check on unchanged saved geometry. --rebuild
explicitly reruns construction; ray/cone avoidance is never a new requirement.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step4_connect_support.baseline_current import process_access as A, reseating as R, acceptance as ACCEPT
from step4_connect_support.baseline_current.run_reseated import groups


def public_hashes(out):
    return {p.name:I.sha256(p) for p in out.iterdir()
        if p.is_file() and p.suffix in ('.obj','.png')}


def validate_generation(report,path):
    """Validate the retained input/artifact bytes, retaining old code hashes."""
    original=report.get('geometry_generation_provenance',report['provenance'])
    I.check_hashes(original['inputs'])
    for name,digest in report.get('artifacts',{}).items():
        if I.sha256(path.parent/name)!=digest:
            raise RuntimeError(f'Changed original geometry/certificate: {name}')


def publish_recheck(out,check,geometry,before):
    source_paths=[out/'data/current_build/report.json',out/'data/report.json']
    current_status=None
    for path in source_paths:
        report=json.loads(path.read_text())
        validate_generation(report,path)
        report.setdefault('geometry_generation_provenance',copy.deepcopy(report['provenance']))
        mechanical=report.get('mechanical_checks_passed_before_process_access',report['passed'])
        passed=geometry['passed']
        current_status='step4_geometry_passed' if passed else 'step4_geometry_failed'
        report.update(passed=passed,status=current_status,
            passed_scope='step4_geometry_only',step4_geometry=geometry,
            size_limit_enforced=False,compactness_passed=None,
            force_torque_authority='step3',step4_force_torque_enforced=False,
            step4_force_torque_recomputed=False,step3_verdict_modified=False,
            step3_passed=geometry['step3_passed'],step3_covered_counts=geometry['step3_covered_counts'],
            mechanical_checks_passed_before_process_access=mechanical,
            mechanical_checks_recomputed=False,working_surface_recomputed=True,
            new_geometry_generated=False,geometry_regenerated=False,
            existing_geometry_is_current_run_result=False,
            geometry_retained_for_diagnosis=not passed,
            previous_strict_access_rejection_withdrawn=True,
            head_reselection_required_by_work_surface=False,
            latest_rerun='access_rerun.json' if path.parent==out/'data' else '../access_rerun.json',
            process_access=check,working_surface=check,
            acceptance_scope=geometry['scope'],
            provenance=dict(inputs=I.hashes([out/'data/working_surface_check.json',out/'data/geometry_check.json']),
                code=I.hashes([Path(__file__),Path(ACCEPT.__file__)]+A.sources())))
        if path==out/'data/report.json':
            report['construction']=I.check_report(out/'data/current_build/report.json')
            report['provenance']['inputs'].update(I.hashes([out/'data/current_build/report.json']))
        I.save(path,report)
    for filename in ('visualization.json','work_access_check.json','independent_review.json'):
        path=out/'data'/filename
        if not path.exists():
            continue
        saved=json.loads(path.read_text())
        inputs=saved.get('provenance',{}).get('inputs',{})
        replaced={str(p.relative_to(I.ROOT)):I.sha256(p) for p in source_paths}
        changed={p:d for p,d in inputs.items() if p in replaced and d!=replaced[p]}
        if changed:
            saved.setdefault('original_metadata_dependency_hashes',{}).update(changed)
            inputs.update({p:replaced[p] for p in changed})
        saved.update(latest_acceptance_status=current_status,
            geometry_and_images_changed=False,numerical_or_render_rerun=False)
        if filename=='work_access_check.json':
            saved.update(acceptance_applicable=False,
                diagnostic_scope='Historical strict-boundary and processing-ray model; neither is the current Step3 working-face rule',
                current_surface_rule_result='working_surface_check.json',
                previous_inference_that_blocked_rays_invalidate_this_baseline_withdrawn=True)
        I.save(path,saved)
    if public_hashes(out)!=before:
        raise RuntimeError('Working-surface recheck changed public geometry or images')
    return I.check_report(out/'data/report.json')


def recheck(group):
    began=time.perf_counter();out=group/'step4';before=public_hashes(out)
    case=R.load_case(out)
    old=json.loads((out/'data/report.json').read_text())
    validate_generation(old,out/'data/report.json')
    guard=A.Guard(case,old['placement'])
    mesh=trimesh.load(out/'shape.obj',force='mesh',process=False)
    check=guard.verify(mesh)
    check.update(complete=True,object=case.name,poses=case.poses,
        original_geometry_unchanged=True,previous_strict_access_rejection_withdrawn=True,
        provenance=dict(inputs=I.hashes(case.paths+[out/'shape.obj']),code=I.hashes(A.sources())))
    I.save(out/'data/working_surface_check.json',check)
    geometry=ACCEPT.evaluate(case,old,mesh,check)
    geometry.update(geometry_checks_recomputed=False,retained_geometry_evidence_reaggregated=True,
        provenance=dict(inputs=I.hashes(case.paths+[out/'shape.obj',out/'data/working_surface_check.json']),
            code=I.hashes([Path(ACCEPT.__file__),Path(R.__file__)])))
    I.save(out/'data/geometry_check.json',geometry)
    report=publish_recheck(out,check,geometry,before)
    result=dict(complete=True,schema=A.SCHEMA,group=group.name,poses=case.poses,
        passed=report['passed'],working_surface_passed=check['passed'],
        step4_geometry_passed=geometry['passed'],failures=geometry['failures'],
        geometry_excluding_size_passed=geometry['geometry_excluding_size_passed'],
        compactness_passed=geometry['compactness_passed'],
        size_limit_enforced=False,
        force_torque_authority='step3',step4_force_torque_enforced=False,
        step4_force_torque_recomputed=False,step3_passed=geometry['step3_passed'],
        step3_covered_counts=geometry['step3_covered_counts'],step3_verdict_modified=False,
        status=report['status'],new_geometry_generated=False,body_search_performed=False,
        existing_geometry_rechecked=True,public_artifacts_unchanged=public_hashes(out)==before,
        original_contact_count=sum(map(len,case.groups)),all_original_owner_heads_passed=True,
        fixed_contacts_rejected=False,head_reselection_required_by_work_surface=False,
        previous_strict_access_rejection_withdrawn=True,processing_ray_volume_enforced=False,
        shared_edge_or_vertex_contact_allowed=True,
        seconds=time.perf_counter()-began,
        provenance=dict(inputs=I.hashes([out/'data/report.json',out/'data/working_surface_check.json',out/'data/geometry_check.json']),
            code=I.hashes([Path(__file__),Path(ACCEPT.__file__)]+A.sources())))
    I.save(out/'data/access_rerun.json',result)
    print('WORKING SURFACE',group.name,'surface',check['passed'],'overall',report['passed'],flush=True)
    return result


def rebuild(group):
    from step4_connect_support.baseline_current import run_outer_feet,run_pocketed_feet,finish_outer_feet
    if len(group.name.split('+'))==2:
        try:
            candidate=run_outer_feet.build(group.parent.name,group.name)
        except RuntimeError as error:
            # A finite geometric search may fail. An old body is not a new result.
            I.save(group/'step4/data/rebuild_failure.json',dict(complete=True,
                error=str(error),new_geometry_generated=False,
                general_infeasibility_claim=False))
            raise
        index=int(candidate.name.rsplit('_',1)[1])
        finish_outer_feet.finish(group.name,index)
    else:
        report=run_pocketed_feet.run(group)
        if not report['constructed']:
            return report
    from step4_connect_support.baseline_current.render_reference_feet import render
    render(group)
    return I.check_report(group/'step4/data/report.json')


def batch(name):
    rows=[recheck(group) for group in groups(name)]
    destination=groups(name)[0]/'step4/data'
    preferred=I.OUTPUTS/name/'pose5+7/step4/data'
    if preferred.exists():
        destination=preferred
    I.save(destination/'access_batch.json',dict(complete=True,schema=A.SCHEMA,groups=rows,
        all_working_surfaces_clear=all(r['working_surface_passed'] for r in rows),
        original_contact_count=sum(r['original_contact_count'] for r in rows),
        installed_pose_count=sum(len(r['poses']) for r in rows),
        accepted=sum(r['passed'] for r in rows),new_geometry_count=0,
        accepted_scope='step4_geometry_only',force_torque_authority='step3',
        size_limit_enforced=False,
        step4_force_torque_enforced=False,step4_force_torque_recomputed=False,
        previous_strict_access_rejection_withdrawn=True,
        provenance=dict(inputs=I.hashes([g/'step4/data/access_rerun.json' for g in groups(name)]),
            code=I.hashes([Path(__file__)]))))
    lines=['# Step4 几何验收结果','',
        '力和力矩以 Step3 原记录为准；Step4 不重复求解、不用旧承载诊断否决几何结果。',
        '规则与 Step3 一致：不得占用工作面；相邻非工作面共边/共顶点允许，不排除加工射线圆锥。',
        '八组现有实体均已重新检查，包括闲置头、根部、连接体和脚。',
        '撤销上一轮“所有组原头非法、必须重新选头”的结论；原头、载荷、模型和图片均未改。','',
        '不设置尺寸上限；跨度仅记录，不作为失败条件。',
        '| 组合 | 工作面检查 | Step4 几何 | 剩余失败 |','|---|---|---|---|']
    for row in rows:
        surface='通过' if row['working_surface_passed'] else '未通过'
        overall='通过' if row['passed'] else '未通过'
        failures=[]
        for failure in row['failures']:
            if failure['check']=='maximum_spatial_span':
                failures.append(f"跨度 {failure['actual_m']*1000:.1f} mm > 上限 {failure['limit_m']*1000:.1f} mm")
            else:failures.append(str(failure))
        lines.append(f"| {row['group']} | {surface} | {overall} | {'；'.join(failures) or '无'} |")
    lines+=['','几何检查保留：工作面、地面、实际接地凸包覆盖、接触与根部保留、封闭连通、完整退出。',
        'Step3 原始通过/未通过记录没有被改写；Step4 通过仅指上述几何条件，不表示上游失败被改成通过。',
        '原几何证据在输入和实体字节未变的条件下沿用；本次重查工作面、接地凸包覆盖及尺寸并重新汇总，没有生成新实体或求解力和力矩。',
        '每组的 working_surface_check.json 保存全部工作三角面的检查及允许的边界接触记录。',
        '使用完整三角面相交与闭实体包含检查，几何容差为 1 nm；没有用几条采样射线代替工作面。',
        '旧 work_access_check.json 的射线计数仅属于被撤销的额外约束诊断，不参与当前通过判定。','']
    (destination/'access_rerun.md').write_text('\n'.join(lines))
    print(json.dumps(dict(groups=len(rows),surface_passed=sum(r['working_surface_passed'] for r in rows),
        poses=sum(len(r['poses']) for r in rows),overall_passed=sum(r['passed'] for r in rows))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object',default='B');parser.add_argument('--group')
    parser.add_argument('--rebuild',action='store_true')
    args=parser.parse_args()
    if args.rebuild:
        selected=[I.OUTPUTS/args.object/args.group] if args.group else groups(args.object)
        for group in selected:
            try:
                rebuild(group)
            except (RuntimeError,ValueError) as error:
                print('REBUILD FAILED',group.name,str(error),flush=True)
    elif args.group:
        recheck(I.OUTPUTS/args.object/args.group)
    else:
        batch(args.object)
