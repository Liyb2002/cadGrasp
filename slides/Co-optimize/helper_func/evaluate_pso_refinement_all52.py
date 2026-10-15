"""Audit actual Step4 complete supports and compare same-run cold incumbents."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'slides/Co-optimize/data/results/B'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cold_membership(folder, baseline):
    trace = read(folder/'population32_trace.json')
    if trace and trace[-1].get('incumbent'):
        return {k:trace[-1]['incumbent'][k] for k in ('reuse', 'juxtaposed')}
    initial = read(folder/'initial_population_summary.json')
    candidates = [r for r in initial if r.get('completed')
                  and abs(r.get('material_cm3', -1)-baseline['material_cm3'])<1e-5]
    counts = {(r['reuse'],r['juxtaposed']) for r in candidates}
    if len(counts)==1:
        reuse, juxtaposed = counts.pop()
        return dict(reuse=reuse, juxtaposed=juxtaposed)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='pso_refine_all52_step4_20261014_01')
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    system = BASE/'data'/('system_'+args.name)
    batch = read(BASE/'data'/args.name/'batch.json')
    if args.partial:
        batch['results']=[read(p) for p in (BASE/'data'/args.name/'workers').glob('*.json')]
    else:
        assert batch['complete'] and len(batch['results'])==52
    tree = system/'frozen_sources'
    co = tree/'slides/Co-optimize'
    sys.path[:0] = [str(tree), str(co/'helper_func'), str(co/'evaluation'),
                   str(co/'step5.2'), str(co/'step5.3'), str(tree/'slides/tools')]
    import _bootstrap
    from metrics import static_bounds
    from deterministic_base import solid, volume, moved
    from prune_support import material_components
    files = read(system/'frozen_sources.json')['files']
    assert all(sha(ROOT/r['frozen'])==r['frozen_sha256'] for r in files)
    protected = read(system/'protected_originals.json')
    assert all((ROOT/p).is_file() and sha(ROOT/p)==h for p,h in protected.items())
    external = read(system/'external_numerical_sources.json')['files']
    for source in external:
        assert sha(ROOT/source['original'])==source['sha256']==sha(ROOT/source['archive'])
    allowed_external = {str(ROOT/r['original']) for r in external}
    body = trimesh.load(ROOT/'objects/B/mesh.stl', force='mesh', process=False)
    vertices, inverse = np.unique(body.vertices, axis=0, return_inverse=True)
    object_solid = solid(trimesh.Trimesh(vertices, inverse[body.faces], process=False))
    old_name = 'pso_all52_20261013_01'
    old_rows = {r['id']:r for r in read(BASE/'data'/old_name/'batch.json')['results']}
    rows = []
    masks_checked = 0
    for result in sorted(batch['results'], key=lambda r:batch['sets'].index(r['id'])):
        label = result['id']
        folder = BASE/label/'step4/step4.2'/args.name
        report = read(folder/'report.json') if (folder/'report.json').exists() else {}
        baseline = read(folder/'before_refinement.json') if (folder/'before_refinement.json').exists() else {}
        before = baseline.get('certified_incumbent')
        row = dict(id=label, passed=result['passed'], pose_count=result.get('pose_count'),
            seconds=result.get('actual_execution_wall_seconds',result.get('seconds',result.get('process_wall_seconds',0))), cold_before=before,
            status=result.get('status', 'PASS' if result['passed'] else 'search_failed'),
            error=result.get('error'), cold_rounds=baseline.get('cold_search',{}).get('rounds'),
            historical_pso_passed=old_rows[label]['passed'])
        additional=result.get('additional_executed_sources',{})
        for filename,digest in additional.items():
            path=Path(filename)
            assert any(path.is_relative_to(system/name) for name in
                ('reporting_adapter_sources','artifact_export_adapter_sources')),filename
            assert sha(path)==digest,filename
        row['additional_reporting_sources_verified']=len(additional)
        foreign = [r for r in result.get('execution_imports',[]) if not r['frozen']
                   and '/.venv/' not in r['path'] and r['path'] not in allowed_external
                   and r['path'] not in additional
                   and Path(r['path'])!=system/'execution_scripts/step4_worker.py']
        assert not foreign, foreign
        row['execution_imports_source_verified'] = bool(result.get('execution_imports')) and not foreign
        row['execution_imports_complete'] = result.get('execution_imports_complete',
            bool(result.get('execution_imports')))
        if result.get('attempt_provenance'):
            evidence = result['attempt_provenance']
            path = Path(evidence['path'])
            assert path.is_relative_to(system/'checkpoint_recovery'), path
            assert sha(path)==evidence['sha256'], path
            executed = read(path)
            for source in executed.get('known_executed_sources',[]):
                archive = Path(source['archive'])
                assert archive.is_relative_to(system), archive
                assert sha(archive)==source['sha256'], archive
            for artifact in executed.get('evidence_files',[]):
                archive = Path(artifact['path'])
                assert archive.is_relative_to(system), archive
                assert sha(archive)==artifact['sha256'], archive
            row['attempt_provenance_verified'] = dict(path=str(path.relative_to(ROOT)),
                sha256=evidence['sha256'], import_inventory_complete=False,
                known_executed_sources=len(executed.get('known_executed_sources',[])))
        for filename,h in report.get('executed_search_sources_sha256',{}).items():
            assert sha(ROOT/filename)==h, filename
        trace_path=folder/'pso_refinement_trace.json'
        trace=read(trace_path) if trace_path.exists() else []
        events=[e for r in trace for e in r['events']]
        moves=[e for e in events if e['branch']=='pso' and 'move' in e]
        row.update(extra_rounds=len(trace),continuous_proposals=len(moves),
            nonzero_components={k:int(sum(np.linalg.norm(e['move'][k+'_component_m'])>1e-12 for e in moves))
                                for k in ('inertia','cognitive','social')},
            proposal_errors=[e['error'] for e in events if 'error' in e])
        row['planned_search_completed'] = bool(report.get('search_complete')) and (
            len(trace)==batch['options'].get('pso_refinement_rounds',0))
        if not result['passed']:
            row.update(force_passed=result.get('force_passed'), base_oracle_passed=result.get('base_oracle_passed'),
                oracle_status=result.get('oracle_status',report.get('base_oracle',{}).get('status')),
                physical_infeasibility_established=False,
                diagnostic_construction=report.get('diagnostic_construction'),
                counts=result.get('counts'), source=str(folder.relative_to(ROOT)))
            rows.append(row)
            print('AUDITED FAILURE',label,row['status'],flush=True)
            continue
        assert report['passed'] and report['solid_constructed_during_search']
        for filename,h in report['artifacts'].items():
            assert sha(folder/filename)==h, filename
        for filename,h in report['provenance']['inputs'].items():
            from artifact_store import resolve_artifact
            assert sha(resolve_artifact(ROOT/filename))==h, filename
        pre = read(folder/'preconstructed_base.json')
        assert pre['complete'] and pre['force_solver_run']
        assert pre['construction']['passed'] and pre['construction']['actual_full_solid_certified']
        for filename,h in pre['source_sha256'].items():
            assert sha(folder/filename)==h
        with np.load(folder/'layout.npz') as archive:
            layout = {k:archive[k].copy() for k in archive.files}
        with np.load(folder/pre['full_force_certificate']) as proof:
            assert set(proof.files)=={'mask_'+str(k) for k in layout['active']}
            for k in layout['active']:
                mask=proof['mask_'+str(k)]
                assert mask.shape==(32768,) and mask.all()
                masks_checked+=len(mask)
        core = trimesh.load(folder/'support.obj', force='mesh', process=False)
        full = trimesh.load(folder/pre['support'], force='mesh', process=False)
        witness = trimesh.load(folder/'oracle_connected_witness.obj', force='mesh', process=False)
        assert full.is_watertight and full.is_winding_consistent
        actual = solid(full)
        checks = dict(material_components=len(material_components(actual)),
            outside_witness_m3=volume(actual-solid(witness)),
            lost_contact_core_m3=volume(solid(core)-actual),
            maximum_actual_object_overlap_m3=max(volume(actual^moved(object_solid,
                layout['placements'][k])) for k in layout['active']))
        assert checks['material_components']==1
        assert max(checks[k] for k in ('outside_witness_m3','lost_contact_core_m3',
                                       'maximum_actual_object_overlap_m3'))<=1e-10,(label,checks)
        assert pre['construction']['actual_full_solid_checks']['floor_passed']
        metric = static_bounds(body.vertices, layout, core.vertices, full.vertices, report['poses'])
        row.update(full_bbx_l=metric['with_full_support']['volume_litres'],
            material_cm3=volume(actual)*1e6, juxtaposed=metric['juxtaposed_pose_count'],
            rotating_reuse=metric['rotating_reuse_pose_count'], support_states=metric['support_pose_count'],
            actual_geometry_checks=checks, source=str(folder.relative_to(ROOT)))
        assert np.isclose(row['full_bbx_l'],pre['evaluation']['with_full_support']['volume_litres'],rtol=1e-9,atol=1e-8)
        if before:
            membership=cold_membership(folder,before)
            before.update(membership=membership)
            assert row['full_bbx_l']<=before['bbx_l']+1e-8
            row['bbx_change_percent']=100*(row['full_bbx_l']/before['bbx_l']-1)
            row['material_change_percent']=100*(row['material_cm3']/before['material_cm3']-1)
        if old_rows[label]['passed']:
            old_folder=BASE/label/'step4/step4.2'/old_name
            old=read(old_folder/'preconstructed_base.json')['evaluation']
            row['historical_pso_complete_support']=dict(bbx_l=old['with_full_support']['volume_litres'],
                material_cm3=old['material_full_cm3'], juxtaposed=old['juxtaposed_pose_count'],
                same_complete_geometry_guard=False)
        rows.append(row)
        print('AUDITED PASS',label,round(row['full_bbx_l'],5),flush=True)
    passing = [r for r in rows if r['passed']]
    matched = [r for r in passing if r['cold_before']]
    totals = {key:sum(r[key] for r in passing) for key in ('full_bbx_l','material_cm3','juxtaposed')}
    before_totals = {key:sum(r['cold_before'][source] for r in matched)
                     for key,source in [('full_bbx_l','bbx_l'),('material_cm3','material_cm3')]}
    after_totals = {key:sum(r[key] for r in matched) for key in before_totals}
    membership_rows = [r for r in matched if r['cold_before'].get('membership')]
    summary = dict(attempted=len(rows),planned=52,passed=len(passing),failed=len(rows)-len(passing),
        planned_search_completed=sum(r['planned_search_completed'] for r in rows),
        runtime_unresolved=sum(not r['passed'] and bool(r.get('error') or
            r['status'] in ('runtime_failure','unresolved')) for r in rows),
        complete_execution_import_inventories=sum(r['execution_imports_complete'] for r in rows),
        cold_passed=sum(bool(r['cold_before']) for r in rows),
        newly_completed=[r['id'] for r in passing if not r['cold_before']],
        cold_pass_regressions=[r['id'] for r in rows if r['cold_before'] and not r['passed']],
        cold_pass_runtime_interruptions=[r['id'] for r in rows if r['cold_before'] and not r['passed']
            and (r.get('error') or r['status'] in ('runtime_failure','unresolved'))],
        cold_pass_search_failures=[r['id'] for r in rows if r['cold_before'] and not r['passed']
            and not r.get('error') and r['status'] not in ('runtime_failure','unresolved')],
        totals=totals, matched_cold_groups=len(matched), matched_cold_before=before_totals,
        matched_cold_after=after_totals,
        matched_percent_change={k:100*(after_totals[k]/before_totals[k]-1) if before_totals[k] else None for k in before_totals},
        improved_groups=sum(r['full_bbx_l']<r['cold_before']['bbx_l']-1e-8 for r in matched),
        matched_membership_groups=len(membership_rows),
        juxtaposed_before=sum(r['cold_before']['membership']['juxtaposed'] for r in membership_rows),
        juxtaposed_after=sum(r['juxtaposed'] for r in membership_rows),
        continuous_proposals=sum(r['continuous_proposals'] for r in rows),
        nonzero_components={k:sum(r['nonzero_components'][k] for r in rows)
                            for k in ('inertia','cognitive','social')})
    pictures = [dict(path=str(p.relative_to(ROOT)),sha256=sha(p))
                for p in sorted((system/'private_output').rglob('*.png'))]
    audit=dict(complete=not args.partial,passed=True,stage='Step4 only',groups=len(rows),cases=rows,summary=summary,
        saved_full_demand_masks_checked=masks_checked,protected_original_files=len(protected),
        frozen_sources=len(files),private_pngs=pictures,memory_limit_gib=None,
        verified_external_numerical_sources=external,
        comparison='Same-run cold search then 4 extra PSO rounds, with identical complete-solid guards; extra budget is explicit',
        historical_comparison_guard_difference='Older PSO snapshot lacked current full exported-solid certification')
    target=system/('partial_evaluation.json' if args.partial else 'evaluation.json')
    target.write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n')
    if args.partial:
        print('PARTIAL SAVED-STEP4 AUDIT PASS',json.dumps(summary,ensure_ascii=False),flush=True)
        return
    lines=['# PSO续优化：全部52组 Step4', '',
        f"52组均有实际尝试结果，{len(passing)}组通过完整支撑认证；{52-len(passing)}组未通过。其中{summary['planned_search_completed']}组完成计划搜索并保存最终结果，{summary['runtime_unresolved']}组运行未解决、检查点保留；运行中断不代表完成全部搜索轮次或物理不可行。本次只运行Step4。", '',
        f"同次冷搜索完成{summary['cold_passed']}组，续优化新增完整成品{len(summary['newly_completed'])}组："+('、'.join(summary['newly_completed']) or '无')+'。', '',
        '每组从原Step4.1开始，保留原32粒子/最多8轮冷搜索，再增加4轮PSO。采用当前完整实体物理检查：原32768需求/pose、真实物体排除、合法域、核心保留、单一闭合材料体和底面压力包络。未设置内存限额。', '',
        f"同次冷搜索已有成品且本次完成认证的{len(matched)}组：完整BBX {before_totals['full_bbx_l']:.6f}→{after_totals['full_bbx_l']:.6f} L（{summary['matched_percent_change']['full_bbx_l']:+.2f}%）；材料 {before_totals['material_cm3']:.6f}→{after_totals['material_cm3']:.6f} cm³（{summary['matched_percent_change']['material_cm3']:+.2f}%）。{summary['improved_groups']}组BBX减小。另有冷搜索已有成品但运行中断{len(summary['cold_pass_runtime_interruptions'])}组、正常搜索未取得最终认证{len(summary['cold_pass_search_failures'])}组；资源中断不能视为物理不可行。", '',
        f"可恢复冷搜索角色记录的{len(membership_rows)}组Juxtapose合计 {summary['juxtaposed_before']}→{summary['juxtaposed_after']}。全部通过组合计完整BBX {totals['full_bbx_l']:.6f} L、材料 {totals['material_cm3']:.6f} cm³、J={totals['juxtaposed']}。", '',
        '优化仍为BBX优先、材料其次；额外搜索及原换座/局部调整共同参与收益，不能把改善全部归因于PSO平移公式，也不能把额外预算比较称为PSO优于同预算差分。旧PSO49组使用旧实体检查，与本次严格认证的成功数分开看。', '',
        '| 组 | Step4 | 冷搜索→续优化完整BBX L | 冷搜索→续优化材料 cm³ | 冷搜索→续优化J |',
        '| --- | --- | ---: | ---: | ---: |']
    for r in rows:
        if r['passed']:
            before=r['cold_before'] or {}
            bbx='—' if not before else f"{before['bbx_l']:.4f}"
            material='—' if not before else f"{before['material_cm3']:.2f}"
            j=(before.get('membership') or {}).get('juxtaposed','—')
            lines.append(f"| [{r['id']}]({Path(r['source']).relative_to(BASE.relative_to(ROOT))}/support_with_base.obj) | PASS | {bbx}→{r['full_bbx_l']:.4f} | {material}→{r['material_cm3']:.2f} | {j}→{r['juxtaposed']} |")
        else:
            lines.append(f"| {r['id']} | FAIL ({r['status']}) | — | — | — |")
    lines+=['',f"检查{masks_checked:,}个实际完整支撑原需求mask；{len(protected):,}个受保护旧文件字节不变，{len(files)}个执行源码与本批清单SHA一致。独立保存{len(pictures)}张PNG；逐组标准PNG显示接触主体，完整支撑以表中support_with_base.obj及其证书为准。", '',
            '[逐组数据、实际实体检查和提案记录](data/system_'+args.name+'/evaluation.json)', '']
    (BASE/(args.name+'_results.txt')).write_text('\n'.join(lines))
    print('ALL52 STEP4 AUDIT PASS',json.dumps(summary,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
