"""Audit preserved per-group batches and publish their combined report."""
from pathlib import Path
import csv
import json
import shutil
import run as R
C=R.C


def main():
    inputs=C.Inputs('B');cases=inputs.cases;reports=[];sources=[]
    stages={};batches=[]
    for index,case in enumerate(cases,1):
        data=inputs.load(case);R.load_tasks(data,inputs)
        stage=('step5.3/placement_base_v1' if index<=3 else
               ('step5.3/placement_base_resume_v2' if index==4 else
                ('step5.3/placement_base_resume_v3' if index>=6 else 'step5.3/placement_base_resume_v1')))
        path=inputs.base/case['id']/'step5'/stage/'report.json'
        reports.append(json.loads(path.read_text()));sources.append(path);stages[case['id']]=stage
        batch=inputs.base/'data'/('step53_placement_base_v1' if index<=3 else
               (f'step53_placement_base_resume_{index}_v2' if index==4 else
                (f'step53_placement_base_resume_{index}_v3' if index>=6 else f'step53_placement_base_resume_{index}')))
        if batch not in batches:batches.append(batch)
    pure=inputs.base/'data/step53_pure_z_probe_v1';batches.append(pure)
    for batch in batches:
        executed=json.loads((batch/'execution.json').read_text())['executed_code']
        for path,h in executed.items():
            if C.digest(C.ROOT/path)!=h or C.digest(batch/'executed_sources'/path)!=h:
                raise RuntimeError(f'Constituent batch source mismatch: {path}')
        if batch.name!='step53_placement_base_v1' and not json.loads((batch/'summary.json').read_text())['complete']:
            raise RuntimeError(f'Unfinished batch: {batch}')
    directory=inputs.base/'data/step53_placement_base_combined_v1';directory.mkdir(exist_ok=False)
    code=C.begin(directory,'step5.3/placement_base_combined',inputs,cases)
    R.STAGE='step5.3/placement_base_v1';R.write_report(inputs,directory,reports)
    document=(directory/'report.html').read_text()
    for case in cases:
        original=f"../../{case['id']}/step5/{R.STAGE}"
        actual=f"../../{case['id']}/step5/{stages[case['id']]}"
        document=document.replace(original,actual)
    pure_report=json.loads((pure/'summary.json').read_text());pure_rows=[]
    for row in pure_report['trials']:
        seating=row['seating'];covered=row.get('complete_free_space_ground_coverage')
        count=sum(q['passed'] for q in covered['per_pose']) if covered else 0
        pure_rows.append(f"<tr><td>{row['extra_minimum_lift_m']*1000:.0f}</td><td>{seating.get('maximum_displacement_coordinate_mm',0):.2f}</td><td>{count}/8</td></tr>")
    extra=f'''<h2>第三组：进一步增加整体 Z 上移</h2>
<p>保留原相对位置，额外试了三种最小抬高限制并实际生成几何。均未找到完整通过的底座。提高最小升高限制不一定改变最终解，因为其他地面约束已要求更高的升高。</p>
<table><tr><th>额外最低升高限制 mm</th><th>得到的最大整体上移 mm</th><th>完整剩余材料覆盖的 pose</th></tr>{''.join(pure_rows)}</table>
<p><a href="../step53_pure_z_probe_v1/summary.json">额外整体上移：几何检查和源代码审计</a></p>
<p>主表的整体 Z 提案保留旧材料并含 1 mm 离地余量。补充去掉余量、允许可保持接地的工件继续接地的分支检查，仍只有第三组通过地面必要条件，最大整体上移下界同为 195.047 mm。</p>
<p><a href="exploratory/cadgrasp_exact_assembly_z_v1.json">无额外离地余量的 Z 分支结果</a> · <a href="exploratory/cadgrasp_loose_screen_v1.json">较自由的 XY／XYZ 提案（保守接地筛选）</a></p>'''
    document=document.replace('</html>',extra+'</html>');(directory/'report.html').write_text(document)
    moves=[]
    for case,report in zip(cases,reports):
        loaded=inputs.load(case)
        for trial in report['trials']:
            if trial['seating']['passed']:moves.extend(R.movement_rows(loaded,trial['seating'],trial['label']))
    with (directory/'movements_mm.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(moves[0]));writer.writeheader();writer.writerows(moves)
    exploratory=directory/'exploratory';exploratory.mkdir()
    for name in ['cadgrasp_exact_assembly_z_v1','cadgrasp_loose_screen_v1']:
        for extension in ['json','py','log']:
            source=Path('/tmp')/(name+'.'+extension)
            if source.exists():shutil.copy2(source,exploratory/source.name)
    C.save(exploratory/'provenance.json',dict(scope='supplemental exploratory proposals, not accepted bases',
        preflight_runner_sources_not_frozen_before_start=True,
        artifacts={p.name:C.digest(p) for p in exploratory.iterdir() if p.is_file()}))
    preserved=json.loads(Path('/tmp/cadgrasp_placement_base_protected_20261010.json').read_text())
    changed=[p for p,h in preserved.items() if not (C.ROOT/p).exists() or C.digest(C.ROOT/p)!=h]
    C.save(directory/'preexisting_artifact_audit.json',dict(checked=len(preserved),changed=changed,unchanged=not changed))
    summary=dict(group_count=len(reports),base_passed_count=sum(r['passed'] for r in reports),cases=reports,
        completed_proposal_count=sum(len(r['trials']) for r in reports),
        main_completed_solid_construction_count=sum(len(r['construction_attempts']) for r in reports),
        extra_pure_z_construction_count=len(pure_report['trials']),
        constituent_batches=[C.relative(p) for p in batches],
        case_report_sha256={C.relative(p):C.digest(p) for p in sources},
        original_step4_results_unchanged=not changed,preexisting_artifact_count=len(preserved))
    execution=json.loads((directory/'execution.json').read_text());execution.update(
        aggregation_only=True,boolean_base_construction_run=False,
        verified_completed_case_reports=len(reports),interrupted_partial_batch_preserved=True)
    C.save(directory/'execution.json',execution);C.finish(directory,inputs,code,summary)
    print('COMBINED',len(reports),'groups, accepted',summary['base_passed_count'],'original artifacts changed',changed)


if __name__=='__main__':main()
