"""Rebuild every saved Step4.1 and run continuous Step4.2 in separate workers."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
import argparse
import html
import multiprocessing
import os
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from co_common import *
from run_all import saved_groups, output_root
import whole_pipeline
import optimizer


def archive_and_reset(root, groups, experiment):
    """Freeze old Step4 proof files before removing the explicitly reset stage."""
    archive=root/'_history'/experiment
    archive.mkdir(parents=True, exist_ok=False)
    mapping={};before={};protected={}
    for group in groups:
        base=root/group['id']
        for path in (base/'step3').rglob('*'):
            if path.is_file():protected[str(path.relative_to(ROOT))]=I.sha256(path)
        source=base/'step4';target=archive/group['id']/'step4'
        if not source.exists():continue
        shutil.copytree(source,target,ignore=shutil.ignore_patterns('_history','__pycache__'))
        for path in source.rglob('*'):
            relative=path.relative_to(source)
            if not path.is_file() or '_history' in relative.parts or '__pycache__' in relative.parts:continue
            digest=I.sha256(path);copy=target/relative
            if I.sha256(copy)!=digest:raise RuntimeError('Archive copy differs: '+str(path))
            original=str(path.relative_to(ROOT));before[original]=digest
            mapping[original]=str(copy.relative_to(ROOT))
    for filename in ['step4_results.md','step4_index.html','continuous_step42_results.md']:
        path=root/filename
        if path.exists():shutil.copy2(path,archive/filename)
    (archive/'data').mkdir(exist_ok=True)
    for filename in ['whole_step4_batch.json','whole_step4_progress.json','whole_step4_active_run.json',
                     'continuous_step42_batch.json','continuous_step42_verification.json']:
        path=root/'data'/filename
        if path.exists():shutil.copy2(path,archive/'data'/filename)
    manifest=dict(complete=True,archive=str(archive.relative_to(ROOT)),groups=groups,
        original_step4_hashes=before,archived_paths=mapping,protected_step3_hashes=protected,
        old_reports_unchanged=True,reset_stage='step4.1',deleted_stage41_folders=[])
    save(archive/'manifest.json',manifest)
    # Only these exact saved case directories are removed; history is retained.
    for group in groups:
        stage=root/group['id']/'step4/step4.1'
        if stage.exists():
            assert stage.resolve().is_relative_to(root.resolve()) and not stage.is_symlink()
            shutil.rmtree(stage)
            manifest['deleted_stage41_folders'].append(group['id'])
    save(archive/'manifest.json',manifest)
    print('RESET',len(manifest['deleted_stage41_folders']),'Step4.1 folders;',
          'archived',len(before),'old files;',archive,flush=True)
    return archive,protected


def run_case(arguments):
    name,group,options,token=arguments
    os.environ['COOPT_WHOLE_RUN_TOKEN']=token
    started=time.monotonic()
    row=dict(id=group['id'],poses=group['poses'],passed=False)
    try:
        initial=whole_pipeline.run_case((name,group,dict(stage='step4.1',run_token=token)))
        row['step41']=initial
        if not initial['passed']:
            row.update(status='initialization_unresolved',error=initial.get('error'))
        else:
            final=optimizer.run_case(name,group,options)
            row['step42']=final
            row.update(status='pass' if final['passed'] else 'optimization_unresolved',passed=final['passed'])
            if final['passed']:
                report=I.check_report(output_root(name)/group['id']/'step4/step4.2/continuous/data/report.json')
                row.update(volume_cm3=report['volume_cm3'],baseline_retained=report['step41_baseline_retained'],
                    continuous_quadrature_gate_passed=report['continuous_quadrature_gate_passed'],
                    counts=report['counts'],hosts=report['hosts'])
            else:row['error']=final.get('error')
    except Exception as error:
        traceback.print_exc();row.update(status='unresolved',error=f'{type(error).__name__}: {error}')
    row['seconds']=time.monotonic()-started
    return row


def publish(root, groups, rows, batch):
    by_id={r['id']:r for r in rows}
    lines=['# B：重新初始化与连续 Step4.2','',
        '全部旧 Step4.1 从活动目录删除后重建；所有集合重新运行 Direction、Translation 与 Juxtapose。'
        '原 Step3、工作面、物体、32768载荷/pose和物理容差保持不变。','',
        f"已完成 {len(rows)}/{len(groups)}；实际通过 {sum(r['passed'] for r in rows)}；"
        f"未解决 {sum(not r['passed'] for r in rows)}。",'',
        '[图片浏览](continuous_step4_index.html) · [批次记录](data/continuous_step4_batch.json) · '
        f"[旧结果归档](_history/{batch['experiment']}/)", '',
        '通过指最终真实网格满足全部原载荷与原工作／退出检查；初始化承载通过但优化未获更小有效结果时，保留本次重建的初始化。'
        '连续需求求积与移动接触边界属于近似，未证明整个连续需求域全覆盖。', '',
        '| Pose set | 状态 | 初始化承载 | 最终材料 cm³ | 结果 |',
        '|---|---|---|---:|---|']
    cards=[]
    for group in groups:
        label=group['id'];directory=label+'/step4';out=directory+'/step4.2/continuous'
        row=by_id.get(label)
        if row is None:
            lines.append(f'| {label} | 运行中／待运行 | — | — | — |');continue
        initial=row.get('step41',{});force=initial.get('step41_force_passed')
        links=[]
        if (root/directory/'step4.1/final_result.png').exists():links.append(f'[初始化]({directory}/step4.1/final_result.png)')
        if row['passed']:links+=[f'[过程]({out}/process.png)',f'[最终]({out}/final_result.png)']
        else:links.append(f'[日志]({out}/data/run.log)')
        lines.append(f"| {label} | {row['status']} | {force} | {row.get('volume_cm3',0):.3f} | {' · '.join(links)} |")
        header=f'<section><h2>{html.escape(label)}</h2><p>{html.escape(row["status"])}'
        if row['passed']:
            header+=f" · {row['volume_cm3']:.3f} cm³"
            header+=' · 保留本次初始化' if row.get('baseline_retained') else ' · 接受优化更新'
        elif row.get('error'):header+=' · '+html.escape(row['error'])
        header+='</p>'
        for filename,title in [('step4.1/final_result.png','重新初始化'),
                ('step4.2/continuous/process.png','连续优化过程'),
                ('step4.2/continuous/final_result.png','最终真实支撑')]:
            if 'continuous' in filename and not row['passed']:continue
            if (root/directory/filename).exists():
                url=html.escape(directory+'/'+filename,quote=True)
                header+=f'<h3>{title}</h3><a href="{url}"><img loading="lazy" src="{url}" alt="{title}"></a>'
        cards.append(header+'</section>')
    lines+=['','预算、失败候选和回溯保存在每组新目录；未解决不冒充无解证明。完整夹具接地、连通与强度尚未验收。']
    page='\n'.join(lines)+'\n'
    for filename in ['continuous_step4_results.md','step4_results.md']:(root/filename).write_text(page)
    gallery='<!doctype html><meta charset="utf-8"><title>B continuous Step4</title><style>body{font:16px sans-serif;max-width:1500px;margin:30px auto;padding:0 20px}section{margin:40px 0;border-top:1px solid #ddd}img{width:100%;height:auto}h3{font-weight:normal}</style>'
    gallery+=f'<h1>B 连续 Step4</h1><p>完成 {len(rows)}/{len(groups)}；通过 {sum(r["passed"] for r in rows)}。粗细求积不是连续域包含证明。</p>'
    gallery+=''.join(cards)
    for filename in ['continuous_step4_index.html','step4_index.html']:(root/filename).write_text(gallery)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object',nargs='?',default='B');parser.add_argument('--jobs',type=int,default=6)
    parser.add_argument('--iterations',type=int,default=6);parser.add_argument('--jumps',type=int,default=2)
    parser.add_argument('--jump-trials',type=int,default=4);parser.add_argument('--jump-refine',type=int,default=1)
    parser.add_argument('--backtracks',type=int,default=4);parser.add_argument('--real-budget',type=int,default=3)
    args=parser.parse_args();root=output_root(args.object);groups=saved_groups(args.object)
    for group in groups:I.check_report(root/group['id']/'step3/step3.1/data/report.json')
    experiment='before_continuous_B_rerun_'+time.strftime('%Y%m%d_%H%M%S')
    archive,protected=archive_and_reset(root,groups,experiment)
    token=os.urandom(16).hex();os.environ['COOPT_WHOLE_RUN_TOKEN']=token
    save(root/'data/whole_step4_active_run.json',dict(object=args.object,run_token=token,experiment=experiment))
    options=vars(args)|dict(quadrature_level=1,contact_depth=1,resume=False)
    rows=[];began=time.monotonic()
    batch=dict(complete=False,object=args.object,experiment=experiment,requested_sets=len(groups),
        requested_pose_instances=sum(len(g['poses']) for g in groups),options=options,results=rows,
        initialization_rebuilt=True,old_solutions_used_as_optimizer_start=False,continuous_domain_certified=False)
    save(root/'data/continuous_step4_batch.json',batch);publish(root,groups,rows,batch)
    # Start larger sets early, so lengthy workers do not remain at the tail.
    ordered=sorted(groups,key=lambda g:-len(g['poses']))
    with ProcessPoolExecutor(max_workers=args.jobs,mp_context=multiprocessing.get_context('spawn')) as pool:
        scheduled={pool.submit(run_case,(args.object,g,options,token)):g for g in ordered}
        for future in as_completed(scheduled):
            group=scheduled[future]
            try:row=future.result()
            except Exception as error:row=dict(id=group['id'],poses=group['poses'],passed=False,status='worker_unresolved',error=str(error),seconds=0.)
            rows.append(row);batch.update(completed=len(rows),passed=sum(r['passed'] for r in rows),
                unresolved=sum(not r['passed'] for r in rows),seconds=time.monotonic()-began)
            save(root/'data/continuous_step4_batch.json',batch)
            save(root/'data/continuous_step42_batch.json',dict(complete=False,results=rows,options=options,experiment=experiment))
            publish(root,groups,rows,batch)
            print('CONTINUOUS B',len(rows),'/',len(groups),row['id'],row['status'],
                  round(row['seconds'],1),row.get('volume_cm3'),row.get('error',''),flush=True)
    for relative,digest in protected.items():assert I.sha256(ROOT/relative)==digest,relative
    manifest=json.loads((archive/'manifest.json').read_text())
    for relative,digest in manifest['original_step4_hashes'].items():
        assert I.sha256(ROOT/manifest['archived_paths'][relative])==digest,relative
    batch.update(complete=True,seconds=time.monotonic()-began,step3_unchanged=True,
                 old_step4_archive_unchanged=True,archived_file_count=len(manifest['original_step4_hashes']))
    save(root/'data/continuous_step4_batch.json',batch)
    save(root/'data/continuous_step42_batch.json',dict(complete=True,results=rows,options=options,experiment=experiment))
    publish(root,groups,rows,batch)
    print('COMPLETE',batch['passed'],'/',len(groups),'actual passes;',batch['unresolved'],'unresolved;',round(batch['seconds'],1),'seconds',flush=True)


if __name__=='__main__':main()
