"""Evaluate saved postprocessing results, including every upstream failure."""
from pathlib import Path
import argparse, csv, hashlib, importlib.util, json, os, shutil, sys, textwrap
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COOPT = Path(__file__).resolve().parents[1]
ROOT = COOPT.parents[1]
BASE = COOPT/'data/results/B'
VISUAL = COOPT/'output/B'
sys.path[:0] = [str(COOPT/name)for name in ['helper_func','vis_func','evaluation','step5.1']]
from metrics import static_bounds


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def percentage(value, reference):
    return 100*(value/reference-1) if reference else None


def picture(path, label, row, metric):
    fig, (ax,bx) = plt.subplots(1,2,figsize=(12,4.5),gridspec_kw={'width_ratios':[1.5,1]})
    labels = ['No support','Step4 nominal','Step5.2 pruned','Step5.3 final']
    if row.get('connected_core_policy'):labels[1]='Step4 connected core'
    values = [row.get(k)for k in ['no_support_bbx_l','step4_bbx_l','pruned_bbx_l','final_bbx_l']]
    for k,value in enumerate(values):
        if value is not None:
            ax.bar(k,value,color=['#a5adb5','#319cd7','#e8aa42','#59ab72'][k])
            ax.text(k,value,f'{value:.2f}',ha='center',va='bottom')
        else:
            ax.text(k,0,'No certified result',color='#bb3333',ha='center',va='bottom',rotation=90)
    ax.set_ylabel('Static bounding box (L)');ax.set_xticks(range(4),labels,rotation=12)
    ax.set_ylim(0,max([v or 0 for v in values])*1.25)
    bx.axis('off')
    lines = [row['status'],f'Object poses: {row["object_pose_count"]}',
        f'Reuse / Juxtapose: {row["reuse"]} / {row["juxtaposed"]}',
        f'Support installations: {row["support_pose_count"]}']
    if row.get('final_material_cm3')is not None:
        lines += [f'BBX: {row["step4_bbx_l"]:.2f} -> {row["pruned_bbx_l"]:.2f} -> {row["final_bbx_l"]:.2f} L',
            f'Deletion / legs: {row["pruning_bbx_delta_l"]:+.2f} / {row["legs_bbx_delta_l"]:+.2f} L',
            f'Net BBX change: {row["bbox_delta_l"]:+.2f} L ({row["bbox_delta_percent"]:+.1f}%)']
    lines += ['']+['S'+str(s['state'])+': '+', '.join(s['poses'])for s in metric['fixture_states']]
    bx.text(0,1,'\n'.join(lines),va='top',fontsize=10)
    fig.suptitle(label);fig.tight_layout();path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=150);plt.close(fig)


def all_bbx_picture(rows,experiment):
    ordered=sorted(rows,key=lambda r:(r['object_pose_count'],r['id']))
    fig,(ax,bx)=plt.subplots(1,2,figsize=(18,max(7,.38*len(rows))),gridspec_kw={'width_ratios':[3,1]})
    y=np.arange(len(ordered));height=.18
    for offset,key,label,color in [(-1.5*height,'no_support_bbx_l','No support','#a5adb5'),
            (-.5*height,'step4_bbx_l','Step4 core','#319cd7'),
            (.5*height,'pruned_bbx_l','Step5.2 deletion','#e8aa42'),
            (1.5*height,'final_bbx_l','Step5.3 final','#59ab72')]:
        ax.barh(y+offset,[r.get(key)or 0 for r in ordered],height,label=label,color=color)
    ax.set_yticks(y,[r['id']for r in ordered],fontsize=8);ax.invert_yaxis()
    ax.set_xlabel('Static BBX of objects + installed support (L)');ax.legend(loc='lower right')
    ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    reuse=np.array([r['reuse']for r in ordered]);jux=np.array([r['juxtaposed']for r in ordered])
    bx.barh(y,reuse,.6,label='Reuse',color='#319cd7')
    bx.barh(y,jux,.6,left=reuse,label='Juxtapose',color='#e8aa42')
    for k,row in enumerate(ordered):
        bx.text(row['object_pose_count']+.1,k,f'O{row["object_pose_count"]} / S{row["support_pose_count"]}',va='center',fontsize=8)
        if not row['step53_passed']:ax.text(0,k,'  '+row['status'],va='center',color='#bb3333',fontsize=8)
    bx.set_yticks(y,[]);bx.invert_yaxis();bx.set_xlim(0,max(r['object_pose_count']for r in ordered)+3.5)
    bx.set_xlabel('Object pose count; O = object, S = support');bx.legend(loc='lower right')
    fig.suptitle(f'{len(rows)} sets | no support -> connected core -> deletion -> legs');fig.tight_layout(rect=(0,0,1,.975))
    path=VISUAL/('all_bbx_'+experiment+'.png');fig.savefig(path,dpi=150);plt.close(fig)
    return dict(id='ALL',path=str(path.relative_to(ROOT)),sha256=sha(path))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',required=True)
    parser.add_argument('--step51-experiment',required=True)
    parser.add_argument('--source-summary',type=Path,required=True)
    args=parser.parse_args();experiment=args.experiment
    system=BASE/'data'/('system_'+experiment)
    resolved=BASE/'data'/experiment/'resolved_batch.json'
    batch=json.loads((resolved if resolved.exists()else BASE/'data'/experiment/'batch.json').read_text())
    outcomes={r['id']:r for r in batch['results']}
    cases=json.loads(args.source_summary.read_text())['results']
    body=trimesh.load(ROOT/'objects/B/mesh.stl',force='mesh',process=False)
    spec=importlib.util.spec_from_file_location('postprocess_evaluation_render',COOPT/'evaluation/run.py')
    evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)
    from stage5_shared import V
    renderer=V.Renderer();rows=[];pictures=[];protected={str(args.source_summary):sha(args.source_summary)}
    for case in sorted(cases,key=lambda r:(-r['pose_count'],r['id'])):
        label=case['id'];source=BASE/label/'step4/step4.2'/case['experiment']
        report=json.loads((source/'report.json').read_text())
        with np.load(source/'layout.npz')as z:layout={k:z[k].copy()for k in z.files}
        nominal=trimesh.load(source/'support.obj',force='mesh',process=False)if(source/'support.obj').exists()else None
        outcome=outcomes.get(label,{})
        row=dict(id=label,source_experiment=case['experiment'],status=outcome.get('status','RUNNING'),
                 object_pose_count=case['pose_count'],step4_passed=case['passed'])
        row['connected_core_policy']=bool(report.get('step4_contact_core_connected'))
        row['step4_core_components']=report.get('connected_contact_core',{}).get('component_count')
        stage51=(ROOT/outcome['floor']['output']if outcome.get('floor',{}).get('passed')else
            BASE/label/'step5/step5.1'/args.step51_experiment)
        row['step51_source']=str(stage51.relative_to(ROOT))
        row['step51_passed']=(stage51/'data/report.json').exists()and case['passed']
        pruned_root=(ROOT/outcome['pruning']['output']if outcome.get('pruning',{}).get('passed')else
                     BASE/label/'step5/step5.2'/experiment)
        final_root=(ROOT/outcome['construction']['output']if outcome.get('construction',{}).get('passed')else
                    BASE/label/'step5/step5.3'/experiment)
        prune_path=pruned_root/'data/report.json';final_path=final_root/'data/report.json'
        prune=json.loads(prune_path.read_text())if prune_path.exists()else {}
        final=json.loads(final_path.read_text())if final_path.exists()else {}
        row['step52_passed']=bool(prune.get('passed'));row['step53_passed']=bool(final.get('passed'))
        if prune:row['step52_source']=str(pruned_root.relative_to(ROOT))
        if final:row['step53_source']=str(final_root.relative_to(ROOT))
        pruned=trimesh.load(pruned_root/prune['support'],force='mesh',process=False)if row['step52_passed']else None
        full=trimesh.load(final_root/final['support'],force='mesh',process=False)if row['step53_passed']else None
        metric=static_bounds(body.vertices,layout,nominal.vertices if nominal is not None else None,
                             full.vertices if full is not None else None,report['poses'])
        row.update(reuse=metric['rotating_reuse_pose_count'],juxtaposed=metric['juxtaposed_pose_count'],
            support_pose_count=metric['support_pose_count'],members=metric['fixture_states'],
            reuse_members=[p['pose']for p in metric['per_pose']if p['rotating_reuse']],
            juxtaposed_members=[p['pose']for p in metric['per_pose']if p['juxtaposed']],
            no_support_bbx_l=metric['object_static']['volume_litres'])
        assert row['reuse']+row['juxtaposed']==row['object_pose_count']
        if nominal is not None:
            row.update(step4_material_cm3=abs(float(nominal.volume))*1e6,
                       step4_bbx_l=metric['with_step4_support']['volume_litres'])
        if pruned is not None:
            input_path=ROOT/prune['source_contact_material']
            row.update(pruned_material_cm3=abs(float(pruned.volume))*1e6,
                pruning_input_material_cm3=prune['pruning']['input_material_cm3'],
                pruning_input_source=prune['source_contact_material'],
                pruning_removed_cm3=prune['pruning']['removed_material_cm3'],
                pruned_bbx_l=static_bounds(body.vertices,layout,pruned.vertices,pruned.vertices,report['poses'])['with_full_support']['volume_litres'])
            protected[str(prune_path.relative_to(ROOT))]=sha(prune_path)
            protected[str((pruned_root/prune['support']).relative_to(ROOT))]=sha(pruned_root/prune['support'])
            with np.load(pruned_root/'force_certificate.npz')as proof:
                for k in layout['active']:
                    assert proof['mask_'+str(k)].shape==(32768,)and proof['mask_'+str(k)].all()
        reference_path=source/'support_with_base.obj'
        if reference_path.exists():
            reference=trimesh.load(reference_path,force='mesh',process=False)
            row.update(step4_preconstructed_material_cm3=abs(float(reference.volume))*1e6,
                step4_preconstructed_bbx_l=static_bounds(body.vertices,layout,nominal.vertices,reference.vertices,report['poses'])['with_full_support']['volume_litres'])
        if full is not None:
            (VISUAL/label/'step5.3/status.png').unlink(missing_ok=True)
            assert final['selected_step4_experiment']==case['experiment']
            assert final['step52_source']==str(pruned_root.relative_to(ROOT))
            assert full.is_watertight and full.is_winding_consistent
            check=final['construction']['serialized_export_verification']
            assert check['passed']and check['history'][-1]['geometry_passed']and check['history'][-1]['force_passed']
            assert len(final['force_counts'])==case['pose_count']and all(n==32768 for n in final['force_counts'].values())
            material=abs(float(full.volume))*1e6
            assert np.isfinite(material)and material>0
            row.update(construction_kernel_material_cm3=final['construction']['material_cm3'],
                       mesh_minus_kernel_material_cm3=material-final['construction']['material_cm3'])
            row.update(final_material_cm3=material,final_bbx_l=metric['with_full_support']['volume_litres'],
                material_delta_cm3=material-row['step4_material_cm3'],
                material_delta_percent=percentage(material,row['step4_material_cm3']),
                bbox_delta_l=metric['with_full_support']['volume_litres']-row['step4_bbx_l'],
                bbox_delta_percent=percentage(metric['with_full_support']['volume_litres'],row['step4_bbx_l']),
                construction_net_added_cm3=material-row['pruned_material_cm3'],
                delta_vs_preconstructed_material_cm3=material-row['step4_preconstructed_material_cm3'],
                delta_vs_preconstructed_bbx_l=metric['with_full_support']['volume_litres']-row['step4_preconstructed_bbx_l'])
            row.update(pruning_bbx_delta_l=row['pruned_bbx_l']-row['step4_bbx_l'],
                legs_bbx_delta_l=row['final_bbx_l']-row['pruned_bbx_l'])
            protected[str(final_path.relative_to(ROOT))]=sha(final_path)
            protected[str((final_root/final['support']).relative_to(ROOT))]=sha(final_root/final['support'])
            metric.update(id=label,step4_source=str(source.relative_to(ROOT)),step53_source=str(final_root.relative_to(ROOT)),
                material_step4_cm3=row['step4_material_cm3'],material_full_cm3=material,
                material_net_change_cm3=row['material_delta_cm3'],force_solver_run=False,boolean_run=False)
            numeric=BASE/label/'evaluation'/experiment
            save(numeric/'report.json',metric)
            visual=VISUAL/label/'evaluation';visual.mkdir(parents=True,exist_ok=True)
            evaluator.render(renderer,body,nominal,full,layout,metric,visual)
            shutil.copy2(visual/'static_occupancy.png',VISUAL/label/'step5.3/static_occupancy.png')
        else:
            error=outcome.get('pruning',{}).get('error')or outcome.get('construction',{}).get('error')or outcome.get('skipped_reason')or outcome.get('error')
            row['error']=error
            for stage in ['step5.3']+(['step5.1','step5.2']if not case['passed']else ['step5.2']if not prune else []):
                folder=VISUAL/label/stage;folder.mkdir(parents=True,exist_ok=True)
                for stale in folder.glob('*.png'):stale.unlink()
                fig,ax=plt.subplots(figsize=(9,3));ax.axis('off')
                ax.text(0,1,label+'\n'+row['status']+'\nNo certified '+stage+' result\n\n'+textwrap.fill(str(error),100)[:500],va='top',fontsize=11,color='#a82d35')
                fig.tight_layout();fig.savefig(folder/'status.png',dpi=120);plt.close(fig)
        if row['step51_passed']:
            visual51=VISUAL/label/'step5/step5.1'/stage51.name
            direct51=VISUAL/label/'step5.1';direct51.mkdir(exist_ok=True)
            for file in visual51.glob('*.png'):shutil.copy2(file,direct51/file.name)
        picture(VISUAL/label/'evaluation/volume_changes.png',label,row,metric)
        shutil.copy2(VISUAL/label/'evaluation/volume_changes.png',VISUAL/label/'step5.2/evaluation.png')
        shutil.copy2(VISUAL/label/'evaluation/volume_changes.png',VISUAL/label/'step5.3/evaluation.png')
        save(system/'evaluations'/(label.replace('/','__')+'.json'),dict(row,metrics=metric))
        rows.append(row)
        for stage in ['step5.1','step5.2','step5.3','evaluation']:
            for file in (VISUAL/label/stage).glob('*.png'):
                pictures.append(dict(id=label,path=str(file.relative_to(ROOT)),sha256=sha(file)))
    passed=[r for r in rows if r['step53_passed']]
    if any(r['connected_core_policy']for r in rows):pictures.append(all_bbx_picture(rows,experiment))
    fields=['id','status','object_pose_count','support_pose_count','reuse','juxtaposed','step51_passed','step52_passed','step53_passed',
        'no_support_bbx_l','step4_bbx_l','pruned_bbx_l','final_bbx_l','bbox_delta_l','bbox_delta_percent',
        'pruning_bbx_delta_l','legs_bbx_delta_l','connected_core_policy','step4_core_components',
        'step4_material_cm3','pruning_input_material_cm3','pruned_material_cm3','pruning_removed_cm3','final_material_cm3',
        'material_delta_cm3','material_delta_percent','construction_net_added_cm3',
        'construction_kernel_material_cm3','mesh_minus_kernel_material_cm3',
        'step4_preconstructed_material_cm3','delta_vs_preconstructed_material_cm3','step4_preconstructed_bbx_l','delta_vs_preconstructed_bbx_l',
        'source_experiment','step52_source','step53_source','pruning_input_source','error']
    totals={key:sum(r[key]for r in passed)for key in ['object_pose_count','support_pose_count','reuse','juxtaposed',
        'no_support_bbx_l','step4_bbx_l','pruned_bbx_l','final_bbx_l','bbox_delta_l','step4_material_cm3','pruned_material_cm3',
        'final_material_cm3','material_delta_cm3','step4_preconstructed_material_cm3','delta_vs_preconstructed_material_cm3',
        'step4_preconstructed_bbx_l','delta_vs_preconstructed_bbx_l']}
    summary=dict(complete=batch['complete'],experiment=experiment,groups=len(rows),object_pose_count=sum(r['object_pose_count']for r in rows),
        step51_passed=sum(r['step51_passed']for r in rows),step52_passed=sum(r['step52_passed']for r in rows),
        step53_passed=len(passed),passed_totals=totals,cases=rows,pictures=pictures,
        material_volume_definition='absolute signed volume of saved exported triangle mesh',force_solver_run=False,boolean_run=False)
    save(system/'evaluation_results.json',summary)
    md=BASE/('evaluation_'+experiment+'.md')
    with md.with_suffix('.csv').open('w',newline='')as file:
        writer=csv.DictWriter(file,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    with md.with_name(md.stem+'_poses.csv').open('w',newline='')as file:
        writer=csv.DictWriter(file,fieldnames=['set','pose','role','support_pose','host_pose','final_passed']);writer.writeheader()
        for row in rows:
            metric=json.loads((system/'evaluations'/(row['id'].replace('/','__')+'.json')).read_text())['metrics']
            for pose in metric['per_pose']:
                state=next(s['state']for s in metric['fixture_states']if pose['pose']in s['poses'])
                writer.writerow(dict(set=row['id'],pose=pose['pose'],role='reuse'if pose['rotating_reuse']else'juxtapose',support_pose=state,host_pose=pose['host_pose'],final_passed=row['step53_passed']))
    lines=['# 当前动态换座布局的Step5与Evaluation','',f'全部52组／263个object pose纳入报告。Step5.1 {summary["step51_passed"]}/50，Step5.2 {summary["step52_passed"]}/50，Step5.3 {len(passed)}/50；另2组Step4失败，未继续构造。','',
        f'最终通过组共{totals["object_pose_count"]}个object pose、{totals["support_pose_count"]}个实际支撑安装pose，原位复用{totals["reuse"]}、Juxtapose {totals["juxtaposed"]}。每组一件共享支撑，任务依次执行。','',
        '材料体积单位cm³；BBX单位L，是包含物体和支撑全部最终安装状态的静态包围盒。净变化统一比较Step5.3与Step4名义support.obj，负数表示减少。各组布局、角色、退出方向和原始需求保持不变。失败行没有成品体积。','',
        f'最终通过组：材料 {totals["step4_material_cm3"]:.3f} → {totals["final_material_cm3"]:.3f} cm³，净变化 {totals["material_delta_cm3"]:+.3f} cm³（{percentage(totals["final_material_cm3"],totals["step4_material_cm3"]):+.2f}%）；BBX {totals["step4_bbx_l"]:.3f} → {totals["final_bbx_l"]:.3f} L，净变化 {totals["bbox_delta_l"]:+.3f} L（{percentage(totals["final_bbx_l"],totals["step4_bbx_l"]):+.2f}%）。这是独立集合的指标合计。','',
        f'相对同一批Step4已经构造的完整参考实体：材料 {totals["step4_preconstructed_material_cm3"]:.3f} → {totals["final_material_cm3"]:.3f} cm³，净变化 {totals["delta_vs_preconstructed_material_cm3"]:+.3f} cm³；BBX {totals["step4_preconstructed_bbx_l"]:.3f} → {totals["final_bbx_l"]:.3f} L，净变化 {totals["delta_vs_preconstructed_bbx_l"]:+.3f} L。','',
        '| Set（图） | 状态 | object | 支撑安装 | 复用 | Juxtapose | Step4材料cm³ | Step5.2材料cm³ | Step5.3材料cm³ | 材料净变化 | 材料变化% | Step4 BBX L | Step5.3 BBX L | BBX净变化L | BBX变化% |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        def n(key,signed=False):return f'{row[key]:+.2f}'if signed and row.get(key)is not None else f'{row[key]:.2f}'if row.get(key)is not None else '—'
        image=VISUAL/row['id']/'step5.3'/('all_poses.png'if row['step53_passed']else'status.png')
        lines.append(f'| [{row["id"]}]({os.path.relpath(image,BASE)}) | {row["status"]} | {row["object_pose_count"]} | {row["support_pose_count"]} | {row["reuse"]} | {row["juxtaposed"]} | '+ ' | '.join(n(k,k in ['material_delta_cm3','material_delta_percent','bbox_delta_l','bbox_delta_percent'])for k in ['step4_material_cm3','pruned_material_cm3','final_material_cm3','material_delta_cm3','material_delta_percent','step4_bbx_l','final_bbx_l','bbox_delta_l','bbox_delta_percent'])+' |')
    lines+=['','## 每个object pose的角色与支撑成员','','| Set | 原位复用成员 | Juxtapose成员 | 支撑安装→object pose |','|---|---|---|---|']
    for row in rows:
        members='; '.join('S'+str(s['state'])+': '+', '.join(s['poses'])for s in row['members'])
        lines.append('| '+row['id']+' | '+(', '.join(row['reuse_members'])or'—')+' | '+(', '.join(row['juxtaposed_members'])or'—')+' | '+members+' |')
    lines+=['','## 验收与来源','','原需求仍是每pose32768个。Step5.2每次实际接触删减必须通过完整需求，Step5.3再次检查实际序列化实体、地面包围、合法空间及受力；不借用旧暖启动布局。','',
        'Step5.2优先读取Step4名义材料；若其真实接触不通过，则尝试同一布局已保存的构造实体。实际删减输入来源与体积、相对已构造参考的变化均列在CSV中。','',
        '[逐组完整CSV]('+md.with_suffix('.csv').name+') · [每个object pose角色CSV]('+md.stem+'_poses.csv) · [结构化结果](data/system_'+experiment+'/evaluation_results.json)。','',
        '图直接在各自step5.1、step5.2、step5.3、evaluation目录，无HTML。Step4及历史数值受保护。Evaluation从保存模型计算，没有再次求解受力或Boolean。']
    lines+=['',f'第一轮后处理完成{batch.get("original_single_batch_passed",len(passed))}/50组，保存的有界地脚参数／保留更多材料重试后完成{len(passed)}/50；每组实际Step5.2/5.3来源记录在CSV中，失败尝试保留。',
        '',f'通过组BBX合计：无支撑{totals["no_support_bbx_l"]:.3f}L／Step4名义{totals["step4_bbx_l"]:.3f}L／Step5.2删减材料{totals["pruned_bbx_l"]:.3f}L／Step5.3完整支撑{totals["final_bbx_l"]:.3f}L。',
        '', '材料体积统一取保存的导出三角网格的有向体积绝对值；CSV另列构造内核体积和差值，物理通过条件保持原值。']
    failed=[r for r in rows if r['status']not in ['PASS','RUNNING']]
    lines+=['','失败记录：']+['\n- `'+r['id']+'`：'+r['status']+'；'+str(r.get('error')) for r in failed]
    md.write_text('\n'.join(lines)+'\n')
    if any(row['connected_core_policy']for row in rows):
        eligible=sum(r['step4_passed']for r in rows)
        def number(row,key):
            value=row.get(key)
            return '—'if value is None else f'{value:.3f}'
        bbx=['# 连通 Step4 → BBX 优先删减 → 长脚：全部组结果','',
            f'全部 {len(rows)} 组／{summary["object_pose_count"]} 个 object pose。Step4 通过 {eligible} 组；Step5.1 {summary["step51_passed"]}/{eligible}，Step5.2 {summary["step52_passed"]}/{eligible}，Step5.3 {len(passed)}/{eligible}。失败组保留状态，未给失败实体填写成品 BBX。','',
            f'最终通过组：{totals["object_pose_count"]} 个 object pose／{totals["support_pose_count"]} 个支撑安装状态，复用 {totals["reuse"]}／Juxtapose {totals["juxtaposed"]}。','',
            f'同组 BBX 合计：无支撑 {totals["no_support_bbx_l"]:.3f} → Step4 {totals["step4_bbx_l"]:.3f} → 删减 {totals["pruned_bbx_l"]:.3f} → 长脚成品 {totals["final_bbx_l"]:.3f} L。删减变化 {totals["pruned_bbx_l"]-totals["step4_bbx_l"]:+.3f} L；长脚变化 {totals["final_bbx_l"]-totals["pruned_bbx_l"]:+.3f} L。各组仍是不同夹具，此处是指标合计。','',
            'Step4 使用原有限包覆范围内已连通、真实接触全需求通过的主体。Step5.2 只删，BBX 不增加；Step5.3 保留原合法长脚构造。BBX 包含物体与支撑所有最终安装状态，单位 L。','',
            f'[全部组 BBX 和 pose 数量图]({os.path.relpath(VISUAL/("all_bbx_"+experiment+".png"),BASE)})。','',
            '| Set（图） | 状态 | object / 支撑 pose | 复用 / Juxtapose | 无支撑 | Step4 | 删减后 | 长脚后 | 删减 Δ | 长脚 Δ | 总 Δ / % |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for row in rows:
            link=os.path.relpath(VISUAL/row['id']/'step5.2/overview.png'if row['step52_passed']else VISUAL/row['id']/'step5.2/status.png',BASE)
            delta='—'if row.get('bbox_delta_l')is None else f'{row["bbox_delta_l"]:+.3f} / {row["bbox_delta_percent"]:+.2f}%'
            bbx.append(f'| [{row["id"]}]({link}) | {row["status"]} | {row["object_pose_count"]} / {row["support_pose_count"]} | {row["reuse"]} / {row["juxtaposed"]} | '+
                ' | '.join(number(row,k)for k in ['no_support_bbx_l','step4_bbx_l','pruned_bbx_l','final_bbx_l','pruning_bbx_delta_l','legs_bbx_delta_l'])+f' | {delta} |')
        primary=md.with_name(md.stem+'_bbx.md');primary.write_text('\n'.join(bbx)+'\n')
        md.write_text('\n'.join(bbx)+f'\n\n[完整数据 CSV]({md.with_suffix(".csv").name}) · [逐 pose 角色]({md.stem}_poses.csv)。\n')
    changed=[p for p,h in protected.items()if sha(ROOT/p)!=h]
    save(system/'evaluation_verification.json',dict(passed=not changed,protected_sources=protected,changed=changed,
         original_demand_masks_checked=sum(r['object_pose_count']*32768 for r in rows if r['step52_passed']),
         force_solver_run=False,boolean_run=False))
    assert not changed
    print('POSTPROCESS EVALUATION',len(rows),'sets','STEP5.3 PASS',len(passed),'PNG',len(pictures),flush=True)


if __name__=='__main__':main()
