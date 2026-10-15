"""Resolve actual per-case sources and audit their saved solids, masks and PNGs."""
from pathlib import Path
import argparse
import sys

COOPT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(COOPT/'helper_func')]
import stage5_shared as C
from feasible_pipeline import read, save, relative, digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments',nargs='+',required=True)
    parser.add_argument('--name',required=True)
    args=parser.parse_args()
    base=C.A.result_root('B')
    rows={};evaluations={};protected={};sources=[];archives={}
    for experiment in args.experiments:
        directory=base/'data'/('step5_feasible_'+experiment)
        summary=read(directory/'summary.json')
        if not summary['complete']:
            raise ValueError('Unfinished source: '+experiment)
        verification=read(directory/'verification.json')
        if not verification['passed'] or not verification['source_and_execution_snapshot_hashes_match']:
            raise ValueError('Failed source verification: '+experiment)
        evaluation=read(directory/'evaluation.json')
        if not evaluation['passed']:
            raise ValueError('Failed saved-model evaluation: '+experiment)
        for path,h in verification['executed_code'].items():
            archive=directory/'executed_sources'/path
            if digest(archive)!=h:
                raise ValueError('Executed source archive changed: '+str(archive))
            archives[relative(archive)]=h
        protected.update(verification['protected_inputs'])
        protected.update(evaluation['protected_inputs'])
        sources.append(dict(experiment=experiment,source=relative(directory),passed=summary['passed']))
        for row in summary['results']:
            if row['passed'] or row['id'] not in rows:
                rows[row['id']]=row
        for row in evaluation['results']:
            evaluations[row['id']]=row
    for path,h in protected.items():
        if digest(C.ROOT/path)!=h:
            raise ValueError('Original or selected artifact changed: '+path)
    counts=0;pngs=0;crop_feet=0;rounded=0;cleaned=0;baseline_material=0.;step4_bbx=0.;final_material=0.
    verified=[]
    for case in sorted(rows.values(),key=lambda r:(r.get('pose_count',0),r['id'])):
        if not case['passed']:
            if case['status']!='STEP4_FAIL':
                raise ValueError('Unresolved Step5 case: '+case['id'])
            verified.append(case)
            continue
        folder=C.ROOT/case['output'];report=read(folder/'data/report.json')
        final_path=folder/report['support'];mesh=C.trimesh.load(final_path,force='mesh',process=False)
        if not mesh.is_watertight or not mesh.is_winding_consistent:
            raise ValueError('Open saved model: '+case['id'])
        for name,h in report['artifacts'].items():
            if digest(folder/name)!=h:
                raise ValueError('Changed artifact: '+case['id']+'/'+name)
        checkpoint=C.ROOT/report['checkpoint'];checkpoint_record=read(checkpoint/'report.json')
        if digest(final_path)!=checkpoint_record['support_sha256'] or not checkpoint_record['checks']['passed']:
            raise ValueError('Certificate does not bind the actual final model: '+case['id'])
        with C.np.load(folder/'force_certificate.npz')as cert:
            if len(cert.files)!=case['pose_count']or any(cert[k].shape!=(32768,)or not cert[k].all()for k in cert.files):
                raise ValueError('Incomplete original-demand mask: '+case['id'])
            counts+=sum(cert[k].size for k in cert.files)
        initial=C.ROOT/report['optional_operations'][0]['output'];baseline=read(initial/'report.json')
        baseline_material+=baseline['checks']['material_cm3']
        cleaned+=not baseline.get('byte_identical_step4_full',False)
        step4=C.ROOT/report['selected_step4_source']
        raw=read(step4/'preconstructed_base.json')
        step4_bbx+=raw['evaluation']['with_full_support']['volume_litres']
        result=evaluations[case['id']]
        crop_feet+=case['cropped_core_and_feet_committed']
        for h in checkpoint_record.get('history',[]):
            rounded+=h['operation']=='bounded_free_surface_rounding'and h['accepted']
        if result['with_full_support']['volume_litres']>case['baseline_bbx_l']+1e-8:
            raise ValueError('Full-support BBX increased: '+case['id'])
        final_material+=abs(float(mesh.volume))*1e6
        for pic in result['images'].values():
            if digest(C.ROOT/pic['path'])!=pic['sha256']:
                raise ValueError('Image changed: '+pic['path'])
            pngs+=1
        verified.append(dict(case,evaluation=result))
    passed=[r for r in verified if r['passed']]
    final_bbx=sum(r['evaluation']['with_full_support']['volume_litres']for r in passed)
    summary=dict(complete=True,passed=True,group_count=len(verified),eligible_groups=len(passed),
        final_supports=len(passed),step4_failures=len(verified)-len(passed),results=verified,
        actual_sources=sources,original_demands_passed=counts,independent_png_count=pngs,
        baseline_cleaned_or_explicit_reference_groups=cleaned,crop_and_new_feet_committed_groups=crop_feet,
        free_surface_rounding_accepted_groups=rounded,original_step4_full_bbx_sum_l=step4_bbx,
        certified_baseline_bbx_sum_l=sum(r['baseline_bbx_l']for r in passed),final_bbx_sum_l=final_bbx,
        certified_baseline_material_cm3=baseline_material,final_material_cm3=final_material,
        bbx_reduced_groups=sum(r['final_bbx_l']<r['baseline_bbx_l']-1e-8 for r in passed),
        originals_and_certified_final_artifacts_unchanged=True,protected_input_count=len(protected),
        frozen_execution_sources_verified=len(archives),force_solver_rerun=False,
        object_pose_count=sum(r['pose_count']for r in passed))
    directory=base/'data'/('step5_feasible_'+args.name)
    directory.mkdir(parents=True,exist_ok=False)
    save(directory/'resolved_batch.json',summary)
    save(directory/'verification.json',dict(passed=True,protected_inputs=protected,executed_archives=archives))
    report_path=base/(args.name+'.md')
    lines=['# Step5：裁BBX → 做脚 → 最后删减与圆整','',
        f'本轮原Step4通过的{len(passed)}组均取得认证成品，覆盖{summary["object_pose_count"]}个物体pose、{counts:,}个原始需求。原{len(verified)-len(passed)}组Step4失败保持失败。',
        '实际执行来源：'+'；'.join(f'{s["experiment"]}通过{s["passed"]}组'for s in sources)+'。完整基线来源逐组记录，布局保持，没有把失败圆整当成功。',
        f'实际原Step4完整支撑BBX统计和{step4_bbx:.6f} L → 最终{final_bbx:.6f} L；认证基线材料{baseline_material:.3f} → 最终{final_material:.3f} cm³。统计和属于独立集合，不是一个工作站。',
        f'{summary["bbx_reduced_groups"]}组缩小认证基线BBX；{crop_feet}组裁边和新短脚一起提交，其他保留认证旧脚／完整基线。{rounded}组接受小幅自由表面圆整；不声称所有成品都已圆滑。',
        f'{pngs}张独立PNG全部核对；{len(protected)}个原输入／选定产物、{len(archives)}份实际执行源码副本指纹核对通过。原受力、物体排除、合法空间、封闭、连通和地面包围门槛保持。',
        '被拒绝、主动终止或预算到期的美化操作均另存记录，并保留已有认证实体。默认美化预算180秒，不限制物理认证，不设人为地址空间上限。',
        '',f'[算法](../../../step5/feasible_algorithm.md) · [实际来源与验收](data/step5_feasible_{args.name}/resolved_batch.json)','',
        '| Set | O/S | 认证完整基线L | 最终L | 材料cm³ | 来源／成品 |',
        '| --- | --- | ---: | ---: | ---: | --- |']
    for row in verified:
        if not row['passed']:
            lines.append(f'| {row["id"]} | {row.get("pose_count",0)}/— | — | — | — | STEP4_FAIL |')
            continue
        e=row['evaluation'];url=str(Path(e['images']['overview.png']['path']).relative_to(base.relative_to(C.ROOT)))
        before_after=str(Path(e['images']['comparison.png']['path']).relative_to(base.relative_to(C.ROOT)))
        source=Path(row['output']).name
        lines.append(f'| {row["id"]} | {e["pose_count"]}/{e["support_pose_count"]} | {row["baseline_bbx_l"]:.4f} | {row["final_bbx_l"]:.4f} | {e["material_full_cm3"]:.3f} | [{source}]({url}) · [前后]({before_after}) |')
    report_path.write_text('\n'.join(lines)+'\n')
    print({k:v for k,v in summary.items()if k!='results'and k!='actual_sources'},flush=True)


if __name__=='__main__':main()
