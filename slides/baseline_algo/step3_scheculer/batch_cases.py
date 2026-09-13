"""Batch execution with all persistent artifacts inside each case's stage."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent.parent
OUTPUTS = HERE / 'output'
CASES = [(name, f'pose_{i}') for i in range(1, 5) for name in ('A1-f', 'B', 'C5')]
STAGES = {1: 'step_1_needs', 2: 'step2_local_support', 3: 'step3_scheculer',
          4: 'step4_floor_contact', 5: 'step5_connect_support'}
REPORTS = {1: 'needs.json', 2: 'circles.json', 3: 'schedule.json',
           4: 'floor_contact.json', 5: 'connection.json'}


def stage_folder(case, step):
    name, pose = case
    if (name, pose) not in CASES:
        raise ValueError(f'Unknown batch case: {case}')
    return OUTPUTS / name / pose / STAGES[step]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


class ProposalFolder:
    """Route existing geometry writers to flat proposal_* files in Step5."""
    def __init__(self, folder):
        self.folder = Path(folder)

    def __truediv__(self, filename):
        if Path(filename).name != filename:
            raise ValueError('Proposal artifacts must be flat filenames')
        return self.folder / ('proposal.json' if filename == 'proposal.json' else 'proposal_' + filename)

    def mkdir(self, **kwargs):
        self.folder.mkdir(**kwargs)

    def iterdir(self):
        return (p for p in self.folder.iterdir()
                if p.name == 'proposal.json' or p.name.startswith('proposal_'))


def summarize_case(case, step):
    """Index current evidence; this does not rerun or replace the stage auditor."""
    folder = stage_folder(case, step)
    if not folder.is_dir():
        return dict(object=case[0], pose=case[1], step=step, status='not_run')
    batch = read(folder / 'batch_run.json')
    report = {} if batch.get('skipped') else read(folder / REPORTS[step])
    audit = read(folder / 'audit.json')
    state = (dict(complete=False, status=batch.get('reason', 'skipped')) if batch.get('skipped')
             else read(folder / 'status.json'))
    row = dict(object=case[0], pose=case[1], step=step,
               status=(state.get('status', 'incomplete') if state.get('complete') is False
                       else report.get('status', state.get('status', 'missing_report'))),
               validation='index_of_saved_reports_only',
               recorded_audit_passed=None if batch.get('skipped') else audit.get('passed'),
               skipped=bool(batch.get('skipped')),
               recorded_design_passed=report.get('passed'))
    for key in ('geometry_constructed', 'trajectory_verified', 'continuous_coverage_proved',
                'selected_ids', 'valid_count', 'load_count','covered_percent','contact_count','round_limit',
                'rest_equilibrium_verified','passive_support_no_uplift_verified'):
        if key in report:
            row[key] = report[key]
    if 'bearing' in report:
        row['recorded_continuous_bearing_passed'] = report['bearing'].get('continuous_passed')
    if step == 5 and 'geometry' in report:
        geometry=report['geometry']
        row['direction_candidate_count']=geometry.get('direction_search',{}).get('candidate_count')
        row['surviving_direction_count']=geometry.get('direction_search',{}).get('remaining_count')
        row['head_sweep_passed_count']=sum(a.get('head_sweep_passed',False) for a in geometry.get('attempts',[]))
        row['frame_constructed']=bool(geometry.get('belt',{}).get('passed'))
        row['actual_head_collision_witnesses']=audit.get('checks',{}).get('actual_head_collision_witnesses_replayed')
    files = [REPORTS[step], 'audit.json', 'status.json', 'batch_run.json', 'batch_run.log',
             'centers.png', 'circles.png', 'floor_diagnostic.png', 'connection.png',
             'support.stl', 'insertion.mp4', 'failure_viewer.html', 'failure.png', 'failure_directions.png',
             'failure.mp4', 'failed_shape_mm.stl', 'failure.json', 'bearing_failure.png',
             'bearing_failure.json', 'connectivity_failure.png', 'connectivity_failure.json',
             'support_mm.stl', 'proposal.json', 'proposal_insertion.mp4',
             'schedule.png','schedule_views.json','withdrawal_directions.png','withdrawal_direction_views.json']
    row['evidence'] = [name for name in files if (folder / name).is_file()]
    if step == 5:
        row['primary_failure_figure'] = next((name for name in
            ('connectivity_failure.png', 'bearing_failure.png', 'failure.png')
            if (folder/name).is_file()), None)
    save(folder / 'results.json', row)
    lines = [f'# {case[0]} / {case[1]} / Step{step}', '',
             '当前已保存报告的索引；不代表重新审计通过，也不将几何或轨迹成功等同于完整设计成功。', '',
             f'状态：{row["status"]}', '']
    if row.get('primary_failure_figure'):
        lines += [f"[查看主要失败原因图]({row['primary_failure_figure']})", '']
    if step == 3 and report:
        lines += [f"已选 {report['contact_count']} / {report['round_limit']} 块；覆盖 {report['covered_percent']:.3f}%；共同方向 {len(report['common_withdrawal_directions']['ids'])} 个。",'',
                  f"连续域覆盖通过：{report['continuous_coverage_proved']}。每轮硬条件为共同头部路径、纯重力平衡和不上抬约束；实际底座倾覆未验证。",'']
    if step == 5 and report:
        lines += [f"Step3 共同剩余方向：{row.get('surviving_direction_count', '—')}；Step5 本次候选：{row.get('direction_candidate_count', '—')}；头扫掠通过：{row.get('head_sweep_passed_count', '—')}；框架构造：{row.get('frame_constructed', False)}。", '',
                  f"整件轨迹通过：{row.get('trajectory_verified', False)}；完整设计通过：{row.get('recorded_design_passed', False)}。", '',
                  '仅完整轨迹通过才生成装入视频。未找到共同平移方向不代表已证明所有刚体路径都不存在。', '']
    lines += [f'- [{name}]({name})' for name in row['evidence']]
    (folder / 'results.md').write_text('\n'.join(lines) + '\n')
    return row


def summarize(step=5, cases=CASES):
    rows = [summarize_case(case, step) for case in cases]
    print(json.dumps(rows, ensure_ascii=False), flush=True)
    return rows


def run_case(case, first, last, resume=False, connection_edge_budget=2000,
             connection_workers=1):
    folder = stage_folder(case, last)
    folder.mkdir(parents=True, exist_ok=True)
    record = dict(object=case[0], pose=case[1], from_step=first, through_step=last,
                  complete=False, skipped=False, returncode=None)
    ledger = folder / 'batch_run.json'
    save(ledger, record)
    log = folder / 'batch_run.log'
    if first == 5 and not read(stage_folder(case, 3) / 'status.json').get('complete'):
        record.update(skipped=True, reason='Step3 incomplete')
        command=[sys.executable,str(HERE/'step5_connect_support/failure_visuals.py'),case[0],'--pose',case[1],'--static-only']
        with log.open('w') as stream:
            stream.write('Step3 incomplete; render the saved partial geometry only.\n');stream.flush()
            diagnostic=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        record['diagnostic_returncode']=diagnostic.returncode
        save(ledger, record)
        return record
    if resume and first == last == 5 and read(folder / 'status.json').get('complete'):
        from step3_scheculer import contacts as I
        try:
            report = I.check_report(folder / 'connection.json')
            audit = I.check_report(folder / 'audit.json')
            I.check_report(folder / 'views.json')
            if report['schema'] == 'direction_first_loose_frame_v1' and audit['passed']:
                record.update(complete=True, cached=True, returncode=0 if report['passed'] else 2)
                log.write_text('Reused completed Step5 after verifying report provenance.\n')
                save(ledger, record)
                return record
        except (OSError, AssertionError, KeyError, ValueError, RuntimeError):
            pass
    command = [sys.executable, str(HERE / 'step3_scheculer/run_all.py'), case[0], '--pose', case[1],
               '--from-step', str(first), '--through-step', str(last),
               '--continuous-retry', '--visibility-retry',
               '--connection-edge-budget', str(connection_edge_budget),
               '--connection-workers', str(connection_workers)]
    if resume:
        command.append('--resume')
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', PYTHONUNBUFFERED='1')
    start = time.monotonic()
    record.update(command=command, log=log.name,
                  execution_environment={key: env[key] for key in
                      ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                       'VECLIB_MAXIMUM_THREADS', 'CADGRASP_SCORE_WORKERS') if key in env})
    save(ledger, record)
    try:
        with log.open('w') as stream:
            result = subprocess.run(command, cwd=HERE.parents[1], env=env,
                                    stdout=stream, stderr=subprocess.STDOUT)
        record.update(returncode=result.returncode, complete=True)
        # A numerical Step3 stop must not erase inspectable partial geometry.
        # Floor demand is independent of head selection and can still be drawn.
        schedule_state=read(stage_folder(case,3)/'status.json')
        if last==5 and first<=3 and result.returncode not in (0,2) and schedule_state and not schedule_state.get('complete'):
            floor_command=[sys.executable,str(HERE/'step3_scheculer/run_all.py'),case[0],'--pose',case[1],
                           '--from-step','4','--through-step','4']
            diagnostic_command=[sys.executable,str(HERE/'step5_connect_support/failure_visuals.py'),
                                case[0],'--pose',case[1],'--static-only']
            with log.open('a') as stream:
                stream.write('\nStep3 incomplete: save independent floor demand and inspectable partial heads.\n');stream.flush()
                floor_result=subprocess.run(floor_command,cwd=HERE.parents[1],env=env,stdout=stream,stderr=subprocess.STDOUT)
                diagnostic=subprocess.run(diagnostic_command,cwd=HERE.parents[1],env=env,stdout=stream,stderr=subprocess.STDOUT)
            record.update(skipped=True,reason='Step3 incomplete; Step5 shows partial geometry only',
                          step5_skipped=True,step4_returncode=floor_result.returncode,
                          diagnostic_returncode=diagnostic.returncode,
                          diagnostic_commands=[floor_command,diagnostic_command])
    except OSError as error:
        record.update(error=str(error))
    record.update(elapsed_seconds=time.monotonic()-start,
                  finished_at=datetime.now(timezone.utc).isoformat())
    save(ledger, record)
    return record


def main(first=1, last=5, budget=2000, legacy_phase=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-step', type=int, choices=range(1, 6), default=first)
    parser.add_argument('--through-step', type=int, choices=range(1, 6), default=last)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--connection-edge-budget', '--shape-budget', type=int, default=budget)
    parser.add_argument('--connection-workers', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--case', action='append', choices=[f'{n}:{p}' for n, p in CASES])
    if legacy_phase:
        parser.add_argument('--phase', choices=['1-3', '4'], required=True)
        parser.add_argument('--exclude', action='append', default=[])
        parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    if legacy_phase:
        args.from_step, args.through_step = (1, 3) if args.phase == '1-3' else (4, 4)
        args.resume = args.resume or args.retry_failed
    if args.from_step > args.through_step or min(args.workers, args.connection_edge_budget, args.connection_workers) < 1:
        parser.error('Use ordered stages and positive worker counts and budgets')
    cases = CASES if not args.case else [tuple(c.split(':')) for c in dict.fromkeys(args.case)]
    if legacy_phase:
        cases = [c for c in cases if ':'.join(c) not in args.exclude]
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_case, c, args.from_step, args.through_step, args.resume,
                               args.connection_edge_budget, args.connection_workers) for c in cases]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps(record), flush=True)
    summarize(args.through_step, cases)
    return int(any(not r['skipped'] and r['returncode'] not in (0, 2) for r in records))
