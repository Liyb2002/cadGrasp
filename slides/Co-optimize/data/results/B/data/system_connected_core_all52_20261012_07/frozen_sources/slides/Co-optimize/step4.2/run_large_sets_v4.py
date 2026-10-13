"""Run repaired continuous Step4.2 on saved 7+ pose sets, preserving initialization."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
import argparse
import html
import json
import multiprocessing
import os
import shutil
import signal
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from co_common import ROOT, I, save
from run_all import saved_groups, output_root
import optimizer_v4 as optimizer


def worker_session():
    # Geometry subprocesses remain in this group, so cancellation stops them too.
    os.setsid()


def run_group(arguments):
    name, group, options, token = arguments
    os.environ['COOPT_WHOLE_RUN_TOKEN'] = token
    started = time.monotonic()
    result = optimizer.run_case(name, group, options)
    row = dict(result, poses=group['poses'], seconds=time.monotonic()-started)
    row['status'] = 'pass' if row['passed'] else 'unresolved'
    out = output_root(name)/group['id']/'step4/step4.2'/options['output_name']
    if row['passed']:
        report = I.check_report(out/'data/report.json')
        row.update(counts=report['counts'], hosts=report['hosts'],
                   continuous_quadrature_gate_passed=report['continuous_quadrature_gate_passed'])
    attempts = out/'actual_validation.json'
    if attempts.exists():
        row['actual_validation_attempts'] = json.loads(attempts.read_text())
    return row


def publish(root, batch):
    rows = {row['id']: row for row in batch['results']}
    lines = ['# 修复梯度后的十组 7–10 pose 完整优化', '',
             '从各组保存的 Step4.1 开始，全部 pose 始终参与；历史优化答案未作为起点。', '',
             f"完成 {len(rows)}/{len(batch['groups'])}，实际通过 {sum(r['passed'] for r in rows.values())}。", '',
             '通过要求全部原始 32768 载荷/pose 与真实工作禁区、退出及接触几何检查通过。', '',
             '| Pose set | 状态 | 实体材料 cm³ | 结果 |', '|---|---|---:|---|']
    cards = []
    for group in batch['groups']:
        label = group['id']; row = rows.get(label)
        relative = label+'/step4/step4.2/'+batch['options']['output_name']
        if row is None:
            lines.append(f'| {label} | 待完成 | — | — |'); continue
        if row['passed']:
            links = f'[过程]({relative}/process.png) · [最终]({relative}/final_result.png)'
            volume = f"{row['volume_cm3']:.3f}"
        else:
            links = f'[日志]({relative}/data/run.log)'; volume = '—'
        lines.append(f"| {label} | {row['status']} | {volume} | {links} |")
        card = f"<section><h2>{html.escape(label)}</h2><p>{html.escape(row['status'])}"
        if row['passed']:
            card += f" · {volume} cm³</p>"
            for filename in ['process.png', 'final_result.png']:
                url = html.escape(relative+'/'+filename, quote=True)
                card += f'<a href="{url}"><img loading="lazy" src="{url}"></a>'
        else:
            card += '</p><p>'+html.escape(row.get('error', ''))+'</p>'
        cards.append(card+'</section>')
    lines += ['', '[批次与完整预算](data/continuous_large_sets_v4/batch.json)。', '',
              '连续求积属于搜索指导，未证明整个连续需求域。未解决不表示数学无解。']
    (root/'continuous_large_sets_results.md').write_text('\n'.join(lines)+'\n')
    page = '<!doctype html><meta charset="utf-8"><title>Continuous large sets</title>'
    page += '<style>body{font:16px sans-serif;max-width:1500px;margin:30px auto}img{width:100%}section{margin:40px 0}</style>'
    (root/'continuous_large_sets_index.html').write_text(page+''.join(cards))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object', nargs='?', default='B')
    parser.add_argument('--jobs', type=int, default=6)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--jumps', type=int, default=8)
    parser.add_argument('--jump-trials', type=int, default=8)
    parser.add_argument('--jump-refine', type=int, default=2)
    parser.add_argument('--jump-branches', type=int, default=2)
    parser.add_argument('--minimum-progress', type=float, default=.005)
    parser.add_argument('--backtracks', type=int, default=4)
    parser.add_argument('--real-budget', type=int, default=3)
    parser.add_argument('--quadrature-level', type=int, default=1)
    parser.add_argument('--contact-depth', type=int, default=1)
    parser.add_argument('--output-name', default='continuous_gradient_v4')
    parser.add_argument('--sets', nargs='+')
    args = parser.parse_args(); root = output_root(args.object)
    groups = [g for g in saved_groups(args.object) if len(g['poses']) >= 7]
    if args.sets: groups = [g for g in groups if g['id'] in args.sets]
    if not groups: parser.error('No saved large set matches')
    directory = root/'data/continuous_large_sets_v4'
    if directory.exists():
        parser.error('This batch already exists; preserve its outcome before a separate retry')
    directory.mkdir(parents=True)
    protected = {}
    for group in groups:
        base = root/group['id']
        for stage in [base/'step3', base/'step4/step4.1']:
            for path in stage.rglob('*'):
                if path.is_file(): protected[str(path.relative_to(ROOT))] = I.sha256(path)
        I.check_report(base/'step4/step4.1/data/report.json')
    for filename in ['continuous_step4_batch.json', 'continuous_step42_batch.json', 'batch_stop_followup.json']:
        path = root/'data'/filename
        if path.exists(): protected[str(path.relative_to(ROOT))] = I.sha256(path)
    save(directory/'protected.json', protected)
    code = I.hashes(optimizer.source_files()+[Path(__file__)])
    for relative in code:
        target = directory/'executed_sources'/relative
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/relative, target)
    save(directory/'source_manifest.json', dict(code=code))
    generation = root/'data/whole_step4_active_run.json'
    if generation.exists(): shutil.copy2(generation, directory/'previous_generation.json')
    token = os.urandom(16).hex()
    save(generation, dict(object=args.object, run_token=token, status='running_large_sets',
                         experiment='continuous_large_sets_v4', old_52_case_batch_remains_stopped=True))
    options = dict(vars(args), resume=False)
    batch = dict(complete=False, object=args.object, groups=groups, results=[], options=options,
                 old_solutions_used_as_start=False, initialization_rebuilt=False, code=code,
                 requested_pose_instances=sum(len(g['poses']) for g in groups))
    began = time.monotonic(); path = directory/'batch.json'
    save(path, batch); publish(root, batch)
    pool = ProcessPoolExecutor(max_workers=args.jobs, initializer=worker_session,
                               mp_context=multiprocessing.get_context('spawn'))
    def interrupted(signum, frame): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        futures = {pool.submit(run_group, (args.object, g, options, token)): g
                   for g in sorted(groups, key=lambda g: -len(g['poses']))}
        for future in as_completed(futures):
            group = futures[future]
            try: row = future.result()
            except Exception as error:
                traceback.print_exc()
                row = dict(id=group['id'], poses=group['poses'], passed=False, status='worker_unresolved', error=str(error))
            batch['results'].append(row); batch['seconds'] = time.monotonic()-began
            save(path, batch); publish(root, batch)
            print('LARGE SET', len(batch['results']), '/', len(groups), row, flush=True)
        for relative, digest in protected.items(): assert I.sha256(ROOT/relative) == digest, relative
        for relative, digest in code.items(): assert I.sha256(ROOT/relative) == digest, relative
        batch.update(complete=True, protected_artifacts_unchanged=True, seconds=time.monotonic()-began)
        save(generation, dict(object=args.object, run_token=token, status='large_sets_complete',
                             experiment='continuous_large_sets_v4', old_52_case_batch_remains_stopped=True))
    except KeyboardInterrupt:
        batch.update(status='stopped', seconds=time.monotonic()-began)
        save(generation, dict(object=args.object, run_token=os.urandom(16).hex(), status='stopped', previous_run_token=token))
        for process in list(pool._processes.values()):
            try:
                if os.getpgid(process.pid) == process.pid: os.killpg(process.pid, signal.SIGTERM)
                else: process.terminate()
            except ProcessLookupError: pass
        raise
    finally:
        save(path, batch); publish(root, batch); pool.shutdown(wait=True, cancel_futures=True)
    print('COMPLETE', sum(r['passed'] for r in batch['results']), '/', len(groups), flush=True)


if __name__ == '__main__': main()
