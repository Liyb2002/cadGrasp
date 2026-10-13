"""Write the complete all52 BBX/material/geometry comparison and image links."""
from pathlib import Path
import argparse,json
from run_dynamic_postprocess import ROOT,COOPT,BASE,save,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',required=True);args=p.parse_args()
    s=BASE/'data'/('system_'+args.experiment)
    evaluation=json.loads((s/'evaluation_results.json').read_text());old=json.loads((s/'previous_evaluation.json').read_text())
    audit=json.loads((s/'sculpted_audit.json').read_text());assert audit['passed']and audit['groups']==52
    previous={r['id']:r for r in old['cases']};t=evaluation['passed_totals'];o=old['passed_totals']
    notes=[];body_flow=foot_flow=compact=body_edges=foot_edges=unchanged_body=unchanged_final=0
    for r in evaluation['cases']:
        p=json.loads((ROOT/r['step52_source']/'data/report.json').read_text())['pruning']
        f=json.loads((ROOT/r['step53_source']/'data/report.json').read_text())['construction']['fillet']
        body_flow+=p['surface_flow']is not None;foot_flow+=f['surface_flow']is not None;compact+=p['compaction_radius_m']is not None
        body_edges+=p['accepted_edge_fillets'];foot_edges+=f['accepted_edge_fillets']
        unchanged_body+=abs(r['pruned_material_cm3']-previous[r['id']]['pruned_material_cm3'])<1e-4
        unchanged_final+=abs(r['final_material_cm3']-previous[r['id']]['final_material_cm3'])<1e-4
        if p['surface_flow']is None and not p['accepted_edge_fillets']:notes.append(r['id'])
    rows=['# 更圆润的支撑：全部52组结果（2026-10-13）','',
        f'最终选择`{args.experiment}`。全部52组／263个object pose的Step5.2和Step5.3已完成并通过，Evaluation、实际OBJ审计及全需求／源文件／图片审计均通过。123个支撑安装pose，复用123／Juxtapose140，逐pose角色与上一版一致。Step4和Step5.1未重跑。','',
        '[算法与入口](sculpted_pruning_algorithm.md)。本版先整块删除冗余材料，再圆整真实非接触外表面和脚的自由部分。固定必要接触、整体BBX极值、地面及已认证主体；成品重新导出、重新检查全部原始需求。','',
        '## BBX与材料','',
        'BBX为每组整体静态占用并集盒的合计，单位L；材料为实际导出实体的体积，单位cm³。两种指标分别报告。','',
        '| BBX阶段 | 上一版（L） | 本次（L） |','| --- | ---: | ---: |']
    for label,key in [('无支撑','no_support_bbx_l'),('Step4主体','step4_bbx_l'),('Step5.2删减后','pruned_bbx_l'),('Step5.3成品','final_bbx_l')]:
        rows.append(f'| {label} | {o[key]:.9f} | {t[key]:.9f} |')
    rows+=['',f'全部52组的新旧BBX边界、尺寸和体积逐组核对。最大体积差{max(audit["max_step52_bbx_difference_l"],audit["max_step53_bbx_difference_l"]):.3e} L，为浮点误差；每组保持原整体BBX。各安装state的小盒另作诊断。','',
        '| 材料阶段 | 上一版（cm³） | 本次（cm³） | 变化 |','| --- | ---: | ---: | ---: |']
    for label,key in [('Step5.2','pruned_material_cm3'),('Step5.3','final_material_cm3')]:
        rows.append(f'| {label} | {o[key]:.3f} | {t[key]:.3f} | {(t[key]/o[key]-1)*100:+.2f}% |')
    rows+=['',f'每组两阶段的材料均未增加；{52-unchanged_body}组主体材料减少、{52-unchanged_final}组成品材料减少。圆整候选与已有材料相交，不向原Step4外补肉。脚沿用已有合法材料及位置，本版调整自由侧面，没有重新搜索安装布局。','',
        f'主体连续表面圆整通过{body_flow}/52组，脚自由表面圆整通过{foot_flow}/52组；{compact}组还通过了进一步整块删减。接受的局部圆角共主体{body_edges}条、脚{foot_edges}条。其余操作在原验收下拒绝或回退，不冒充统一圆角成功。','',
        '物体接触面、工作／退出禁区、地面和固定BBX仍限制形状，部分功能棱面必须保留。宽根部和连续圆整改善形状；当前验证为原静力模型与几何验证，不是抗弯或破坏强度证书。','',
        '## 全部52组及图片','',
        'O＝object pose数，S＝实际支撑安装pose数，R＝原位复用，J＝Juxtapose。BBX单位L；净增量／百分比比较成品与Step4。材料列比较上一版与本次成品，单位cm³。每组两阶段均PASS。','',
        '| Set | O | S | R | J | 无支撑 | Step4 | Step5.2 | Step5.3 | 净增量 | 增幅 | 成品材料旧→新 | 材料变化 | 图 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for r in sorted(evaluation['cases'],key=lambda r:(r['object_pose_count'],[int(x)for x in r['id'].split('/')[-1].removeprefix('pose').split('+')])):
        b=previous[r['id']];label=r['id'];delta=(r['final_material_cm3']/b['final_material_cm3']-1)*100
        rows.append(f'| {label} | {r["object_pose_count"]} | {r["support_pose_count"]} | {r["reuse"]} | {r["juxtaposed"]} | {r["no_support_bbx_l"]:.3f} | {r["step4_bbx_l"]:.3f} | {r["pruned_bbx_l"]:.3f} | {r["final_bbx_l"]:.3f} | {r["bbox_delta_l"]:+.3f} | {r["bbox_delta_percent"]:+.2f}% | {b["final_material_cm3"]:.3f}→{r["final_material_cm3"]:.3f} | {delta:+.2f}% | [5.2](output/B/{label}/step5.2/comparison.png) · [成品](output/B/{label}/step5.3/comparison.png) · [放大](output/B/{label}/step5.3/detail.png) |')
    rows+=['',
        '每组`step5.3/overview.png`为成品，`comparison.png`左旧右新，`detail.png`为同视角脚部放大；`all_poses.png`、全部`pose_N.png`及Evaluation图已更新。Step5.2同时更新删减主体、删除材料图和前后对比。所有图直接在各自set的阶段目录，无HTML。','',
        '![成品前后对比](output/B/pose1+4+7+9+24/step5.3/comparison.png)','',
        '![脚部前后放大](output/B/pose1+4+7+9+24/step5.3/detail.png)','',
        '## 验收与执行记录','',
        '- 每pose全部32,768个原始需求不变；两阶段均检查实际导出接触，分别核对8,617,984个通过mask。原反力模型及容差保持。',
        '- 独立实际OBJ审计核对连通、原材料包含、合法空间、物体内部排除及成品主体保留；原几何体积限值1e-10 m³。实际地面包围和整体BBX另行严格检查。',
        f'- {audit["protected_files"]}个旧数值／输入文件、{audit["archived_pngs"]}张旧PNG副本及{audit["verified_current_pngs"]}张当前PNG验证；原Step4图片保持原样。每组选用自己的执行版本与源码副本。',
        '- 初始批次有拒绝、资源问题和几何库崩溃，全部保留记录；已有通过结果保留，剩余组在新数值身份下执行分批查询、导出后守卫或连续表面保守分支。不能将部分／中断批次描述成一次52/52。最终选择只包含真实通过的52组。',
        '- 5个新几何回归测试和5个原删减测试通过；分批最近点查询与一次查询在700个点上的结果一致。','',
        '[最终52组来源](data/results/B/data/'+args.experiment+'/resolved_batch.json) · [Evaluation](data/results/B/evaluation_'+args.experiment+'_bbx.md) · [逐pose角色](data/results/B/evaluation_'+args.experiment+'_poses.csv) · [新旧BBX／材料CSV](data/results/B/comparison_'+args.experiment+'.csv) · [实际实体审计](data/results/B/data/system_'+args.experiment+'/actual_solid_geometry_audit.json) · [完整审计](data/results/B/data/system_'+args.experiment+'/sculpted_audit.json)。','']
    report=COOPT/'sculpted_pruning_results_20261013.md';report.write_text('\n'.join(rows))
    save(s/'report_generation.json',dict(complete=True,groups=52,report=str(report.relative_to(ROOT)),sha256=sha(report),
        body_surface_flow_groups=body_flow,foot_surface_flow_groups=foot_flow,compacted_groups=compact,
        body_edge_fillets=body_edges,foot_edge_fillets=foot_edges,unchanged_body_groups=unchanged_body,
        unchanged_final_groups=unchanged_final,body_without_new_rounding=notes,code={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}))
    print('ALL52 RESULT REPORT',report,'BODY/FREE FOOT FLOW',body_flow,foot_flow,flush=True)


if __name__=='__main__':main()
