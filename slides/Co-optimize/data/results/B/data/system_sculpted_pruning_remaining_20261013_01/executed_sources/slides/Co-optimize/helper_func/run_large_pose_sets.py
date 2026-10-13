"""Run the ten preserved pose_set_search sets through production whole search.

This has its own progress and comparison files. Existing experiments and their
galleries are read only; every new case starts from the current Step3 initializer.
"""
import _bootstrap
import argparse
import contextlib
import html
import multiprocessing
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from co_common import *
import run_all
import whole_pipeline
import run_compact_volume
from audit_compact_volume import audit as audit_compact
from audit_whole_step4 import audit_case, protected_files

EXPERIMENT = 'large_pose_sets_v1'
OLD = HERE.parent / 'pose_set_search'


def groups():
    cases = json.loads((OLD/'output/B/experiment.json').read_text())['cases']
    return [dict(id='pose'+'+'.join(map(str, ids)), poses=['pose_'+str(i) for i in ids],
                 source_case=alias) for alias, ids in cases.items()]


def record_phase(group, phase, row):
    path = run_all.output_root('B')/group['id']/'data'/EXPERIMENT/'progress.json'
    data = json.loads(path.read_text()) if path.exists() else dict(id=group['id'], phases={})
    data['phases'][phase] = row
    data['current_phase'] = phase
    save(path, data)
    print('LARGE_PHASE', group['id'], phase, row.get('status', 'complete'), flush=True)


def run_case(arguments):
    group, options = arguments
    root = run_all.output_root('B'); base = root/group['id']
    token = json.loads((root/'data/whole_step4_active_run.json').read_text())['run_token']
    os.environ['COOPT_WHOLE_RUN_TOKEN'] = token
    began = time.monotonic()
    try:
        path = base/'step3/data/report.json'
        if path.exists():
            report = I.check_report(path)
            assert report['poses'] == group['poses'] and report['passed']
            row = dict(status='initialized', passed=True, reused_completed_result=True)
        else:
            row = run_all.run_case(('B', group, .005))
        record_phase(group, 'step3', row)
        if not row['passed']: return dict(id=group['id'], status='unresolved', phase='step3', error=row.get('error'))
        path = base/'step4/step4.2/data/report.json'
        if path.exists():
            report = I.check_report(path)
            assert report['poses'] == group['poses'] and report['force_exit_work_passed']
            row = dict(status='pass', passed=True, final_volume_cm3=report['volume_cm3'], reused_completed_result=True)
        else:
            step41 = base/'step4/step4.1/data/report.json'
            stage = 'step4.2' if step41.exists() else 'both'
            row = whole_pipeline.run_case(('B', group, dict(stage=stage, iterations=8,
                finalists=3, branch_rounds=3, volume_rounds=3, screen_budget=96, seed=42,
                numerical_recovery_candidates=3, run_token=token)))
        record_phase(group, 'whole_feasibility', row)
        if not row['passed']: return dict(id=group['id'], status='unresolved', phase='whole_feasibility', error=row.get('error'))
        methods = []
        for method in options['methods']:
            row = run_compact_volume.run_case(('B', group, dict(mode=method, rounds=6,
                width=2, repair_width=1, screen_budget=96, full_budget=48, real_budget=3, seed=42)))
            methods.append(row); record_phase(group, method, row)
        audits = []
        if options['audit']:
            logs = base/'data'/EXPERIMENT; logs.mkdir(parents=True, exist_ok=True)
            with (logs/'audit.log').open('w', buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                audits.append(dict(method='baseline', result=audit_case('B', group)))
                for method in options['methods']:
                    if (base/f'step4/step4.2/compact/{method}/data/report.json').exists():
                        audits.append(dict(method=method, result=audit_compact('B', group, method)))
            save(logs/'audit.json', dict(complete=True, passed=True, audits=audits))
            record_phase(group, 'independent_audit', dict(status='pass', passed=True))
        return dict(id=group['id'], status='pass', methods=methods, seconds=time.monotonic()-began)
    except Exception as error:
        traceback.print_exc()
        row = dict(id=group['id'], status='unresolved', error=f'{type(error).__name__}: {error}', seconds=time.monotonic()-began)
        record_phase(group, 'exception', row)
        return row


def protection(root):
    """Freeze old search numerical files and existing production result files."""
    path = root/'data'/EXPERIMENT/'protection.json'
    if path.exists(): return json.loads(path.read_text())
    protected_files(root)
    files = []
    for parent in [OLD/'code', OLD/'output/B']:
        files += [p for p in parent.rglob('*') if p.is_file()
                  and p.suffix in ('.py', '.json', '.npz', '.obj') and '__pycache__' not in p.parts]
    for group in run_all.saved_groups('B'):
        base = root/group['id']
        for parent in [base/'step3', base/'step4']:
            files += [p for p in parent.rglob('*') if p.is_file()
                      and p.suffix in ('.json', '.npz', '.obj', '.png') and '_history' not in p.parts]
    data = dict(files={str(p.relative_to(ROOT)): I.sha256(p) for p in sorted(set(files))})
    save(path, data)
    return data


def publish(root, requested):
    rows=[]; cards=[]
    lines=['# 7–10 pose：旧 pose_set_search 与新版 whole 的对照', '',
           '旧 whole/incremental 为已完成实验的名义三角支撑体积；新版从新的整组注册初始化独立搜索。'
           '新版最终列为真实接触核及退出净空网格体积，原32,768载荷/pose、完整工作角度与退出检查不变。'
           '两列的几何验收口径不同，不能把体积差全部归因于搜索。所有体积均为实体材料，单位cm³。', '',
           '新版先做 whole 可行性修复，再从同一个真实可行解分别做 greedy 与 beam 各6轮/48个全载荷候选/3个真实候选；'
           '选择检查通过且实体体积最小者，并保留原可行解。两条路合计两个优化预算。', '',
           '| 集合 | 原whole名义 | 原incremental名义 | 新可行基线 | 新greedy真实 | 新beam真实 | 选中真实 | 状态/图 |',
           '|---|---:|---:|---:|---:|---:|---:|---|']
    for group in requested:
        old={}
        for method in ['whole','incremental']:
            path=OLD/'output/B'/group['id']/method/'mesh_geometry.json'
            old[method]=json.loads(path.read_text())['final_material_volume_cm3']
        base=root/group['id']; reports={}; paths={}
        choices=[('baseline',base/'step4/step4.2/data/report.json')]
        choices += [(m,base/f'step4/step4.2/compact/{m}/data/report.json') for m in ['greedy','beam']]
        for method,path in choices:
            if path.exists():
                report=I.check_report(path)
                if report.get('force_exit_work_passed'):
                    reports[method]=report;paths[method]=path
        best=min(reports,key=lambda m:reports[m]['volume_cm3']) if reports else None
        final=reports[best]['volume_cm3'] if best else None
        row=dict(id=group['id'],poses=group['poses'],source_case=group['source_case'],
                 old_nominal_cm3=old,new_actual_cm3={m:r['volume_cm3'] for m,r in reports.items()},
                 selected_method=best,selected_actual_cm3=final,
                 complete=all(m in reports for m in ['baseline','greedy','beam']),
                 original_loads_per_pose=32768,
                 report_paths={m:str(p.relative_to(root)) for m,p in paths.items()})
        if best:
            out=paths[best].parent.parent
            row['final_image']=str((out/'final_result.png').relative_to(root))
            row['process_image']=str((out/'process.png').relative_to(root))
            row['selected_report_sha256']=I.sha256(paths[best])
            save(base/'data'/EXPERIMENT/'best.json',row)
            cards.append('<section><h2>'+html.escape(group['id'])+'</h2><p>'
                +html.escape(f"原whole {old['whole']:.2f} / incremental {old['incremental']:.2f} cm³（名义）；新版 {final:.2f} cm³（真实），{best}。")
                +'</p><a href="'+row['process_image']+'">过程图</a><img loading="lazy" src="'+row['final_image']+'"></section>')
        def fmt(method):return f"{reports[method]['volume_cm3']:.2f}" if method in reports else '—'
        status=('完成' if row['complete'] else '进行中') if best else '未决'
        link=f"[{status}]({row['final_image']})" if best else status
        lines.append(f"| {group['id']} | {old['whole']:.2f} | {old['incremental']:.2f} | {fmt('baseline')} | {fmt('greedy')} | {fmt('beam')} | {final:.2f} | {link} |" if best else
            f"| {group['id']} | {old['whole']:.2f} | {old['incremental']:.2f} | — | — | — | — | {link} |")
        rows.append(row)
    save(root/'data'/EXPERIMENT/'comparison.json',dict(rows=rows,
        completed_sets=sum(r['complete'] for r in rows),requested_sets=len(rows),
        old_geometry_rule='nominal',new_geometry_rule='actual contact core and exit clearance',
        same_original_object_poses_work_angles_and_loads=True))
    (root/'large_pose_sets_comparison.md').write_text('\n'.join(lines)+'\n')
    (root/'large_pose_sets_index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>7–10 pose对照</title>'
        '<style>body{max-width:1500px;margin:24px auto;padding:20px;font-family:system-ui}img{max-width:100%}section{margin:40px 0}h2{overflow-wrap:anywhere}</style>'
        '<h1>原十个大集合：新版 whole</h1><p><a href="large_pose_sets_comparison.md">完整体积和验收口径表</a></p>'+''.join(cards)+'</html>')
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sets',nargs='*');parser.add_argument('--jobs',type=int,default=2)
    parser.add_argument('--methods',nargs='+',choices=['greedy','beam'],default=['greedy','beam'])
    parser.add_argument('--no-audit',action='store_true');parser.add_argument('--collect-only',action='store_true')
    args=parser.parse_args(); root=run_all.output_root('B'); requested=groups()
    frozen=protection(root)
    save(root/'data'/EXPERIMENT/'experiment.json',dict(groups=requested,
        whole=dict(iterations=8,finalists=3,branch_rounds=3,volume_rounds=3,screen_budget=96,seed=42,numerical_recovery_candidates=3),
        compact=dict(rounds=6,width=2,repair_width=1,screen_budget=96,full_budget=48,real_budget=3,seed=42),
        methods=args.methods,independent_current_initializations=True,
        old_results_read_only=True,code=I.hashes([Path(__file__)])))
    if not args.collect_only:
        selected=[g for g in requested if not args.sets or g['id'] in args.sets or g['source_case'] in args.sets]
        if not selected:raise ValueError('No matching old cases')
        options=dict(methods=args.methods,audit=not args.no_audit); rows=[]
        with ProcessPoolExecutor(max_workers=args.jobs,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1) as pool:
            for future in as_completed([pool.submit(run_case,(g,options)) for g in selected]):
                row=future.result();rows.append(row);print('LARGE_CASE',row,flush=True)
                save(root/'data'/EXPERIMENT/'progress.json',rows);publish(root,requested)
    publish(root,requested)
    for relative,digest in frozen['files'].items():
        if I.sha256(ROOT/relative)!=digest:raise RuntimeError('Changed protected existing artifact: '+relative)
    save(root/'data'/EXPERIMENT/'preservation_check.json',dict(passed=True,unchanged_files=len(frozen['files'])))


if __name__=='__main__':main()
