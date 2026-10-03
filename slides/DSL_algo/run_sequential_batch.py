"""Baseline Step0–4: choose n compatible poses, then prepare loads, heads and bodies."""
import argparse
from pathlib import Path

from step3_scheculer import contacts as I
from step0_pose_selection.select_poses import select, select_many, TaskCache, materialize_step1


def run_pipeline(name, n, seed=20260929, particles=10, count=200, through_step=4):
    if through_step not in (0, 1, 2, 3, 4) or particles < 1 or count < 1:
        raise ValueError('Valid stage (0–4) and positive search sizes required')
    selection = select(name, n, seed)
    return run_selected(name, n, seed, particles, count, through_step, selection)


def run_selected(name, n, seed, particles, count, through_step, selection, *, from_step=0):
    result = dict(object=name, n=n, seed=seed, selected_poses=selection.report['selected_poses'],
        step0_passed=selection.report['passed'], completed_through=0,
        status=selection.report['status'], constructed=False,
        selection_report=str(selection.ledger_path.relative_to(I.ROOT)))
    # No Step1 group, candidate pool, head search or body construction on exhaustion.
    if not result['step0_passed'] or through_step == 0:
        return result
    poses = result['selected_poses']
    if from_step != 3:
        materialize_step1(name, poses, selection.check_path)
    result.update(completed_through=1, status='step1_inputs_ready')
    destination = selection.check_path.parent/'pipeline.json'
    I.save(destination, result)
    if through_step == 1:
        return result
    from step3_scheculer.run_sequential_k import SequentialKSearch
    from step3_scheculer.run_sequential import numerical_recovery
    search = SequentialKSearch(name, poses, seed, particles, count, reuse_candidates=from_step == 3)
    result.update(completed_through=2, status='step2_candidates_ready')
    I.save(destination, result)
    if through_step == 2:
        return result
    with numerical_recovery(search.out/'numerical_retries', search.recoveries):
        summary = search.run()
    result.update(completed_through=3, status=summary['result']['status'],
        step3_passed=summary['result']['passed'], heads=summary['result']['heads'],
        covered_counts=summary['result']['covered_counts'])
    I.save(destination, result)
    if through_step == 3:
        return result
    from step4_connect_support.run_sequential_k import run as construct
    from step4_connect_support.draw_selected_heads import draw as draw_heads
    body = construct(search.out)
    output = search.out.parents[2]/'step4'
    draw_heads(output)
    result.update(completed_through=4, status=body['status'], constructed=body['constructed'],
                  step4_report=str((output/'data/report.json').relative_to(I.ROOT)))
    I.save(destination, result)
    return result


def _run_batch_case(arguments):
    """One isolated process/log per accepted case; preserve other cases on error."""
    import contextlib
    import traceback
    selection = arguments[-1]
    group = selection.check_path.parent.parent
    data = group/'step4/data'
    data.mkdir(parents=True, exist_ok=True)
    with (data/'pipeline.log').open('w', buffering=1) as log, \
            contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            return run_selected(*arguments)
        except Exception as error:
            traceback.print_exc()
            result = dict(object=arguments[0], n=arguments[1], seed=arguments[2],
                selected_poses=selection.report['selected_poses'], step0_passed=True,
                status='pipeline_error', constructed=False,
                error=f'{type(error).__name__}: {error}')
            I.save(data/'pipeline_error.json', result)
            return result


def run_batch(name, sizes, groups=2, seed=20260929, particles=10, count=200, jobs=2):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    if len(set(sizes)) != len(sizes) or not sizes or min(groups, particles, count, jobs) < 1:
        raise ValueError('Distinct sizes and positive batch/search sizes required')
    cache = TaskCache(name)
    selections, ledgers = [], []
    for n in sizes:
        chosen, ledger = select_many(name, n, seed, groups, cache)
        selections.extend(chosen); ledgers.append(ledger)
    result = dict(object=name, sizes=list(sizes), groups_per_n=groups, seed=seed,
        requested_count=len(sizes)*groups, selected_count=len(selections),
        plan=[s.report['selected_poses'] for s in selections], selection=ledgers,
        complete=False, results=[])
    # The object-level selection record is also available when no set passes.
    path = cache.root/f'batch_seed{seed}.json'
    I.save(path, result)
    print('BATCH PLAN', result['plan'], flush=True)
    arguments = [(name, len(s.report['selected_poses']), seed, particles, count, 4, s)
                 for s in selections]
    with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(_run_batch_case, args) for args in arguments]
        for future in as_completed(futures):
            row = future.result()
            result['results'].append(row)
            I.save(path, result)
            print('CASE COMPLETE', row, flush=True)
    result['results'].sort(key=lambda r: result['plan'].index(r['selected_poses']))
    result['complete'] = True
    I.save(path, result)
    return result


def _rerun_case(arguments):
    import contextlib
    import traceback
    from types import SimpleNamespace
    name, poses, seed, particles, count = arguments
    from step3_scheculer.run_sequential_k import folder
    group = folder(name, poses, 'step3_scheculer').parents[2]
    check_path = group/'step0_pose_selection/report.json'
    check = I.check_report(check_path)
    if not check['passed'] or check['poses'] != poses:
        raise ValueError('Saved Step0 does not certify the requested group')
    selection = SimpleNamespace(check_path=check_path,
        ledger_path=check_path,
        report=dict(selected_poses=poses, passed=True, status='reuse_accepted_pose_set'))
    data = group/'step4/data'
    data.mkdir(parents=True, exist_ok=True)
    with (data/'pipeline.log').open('w', buffering=1) as log, \
            contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            return run_selected(name, len(poses), seed, particles, count, 4, selection, from_step=3)
        except Exception as error:
            traceback.print_exc()
            result = dict(object=name, n=len(poses), seed=seed, selected_poses=poses,
                step0_passed=True, status='pipeline_error', constructed=False,
                error=f'{type(error).__name__}: {error}')
            I.save(data/'pipeline_error.json', result)
            return result


def rebind_step0_inputs(group):
    """Replace missing cache references only with byte-identical local inputs."""
    import json
    path = group/'step0_pose_selection/report.json'
    report = json.loads(path.read_text())
    inputs = report['provenance']['inputs']
    changes = []
    for pose, old_folder in report['load_input_folders'].items():
        source, local = I.ROOT/old_folder, group/'step_1_needs'/pose
        if source.is_dir():
            continue
        for filename in ('needs.json', 'samples.json'):
            old = str((source/filename).relative_to(I.ROOT))
            new = str((local/filename).relative_to(I.ROOT))
            expected = inputs[old]
            if not (local/filename).is_file() or I.sha256(local/filename) != expected:
                raise ValueError('Missing Step0 source has no byte-identical Step1 replacement')
            changes.append(dict(old=old, new=new, sha256=expected))
    if changes:
        # Validate every replacement before editing the report.
        for change in changes:
            inputs[change['new']] = inputs.pop(change['old'])
        for pose, old_folder in list(report['load_input_folders'].items()):
            if not (I.ROOT/old_folder).is_dir():
                report['load_input_folders'][pose] = str((group/'step_1_needs'/pose).relative_to(I.ROOT))
        report['identical_input_reference_relocation'] = changes
        I.save(path, report)
    return I.check_report(path)


def rerun_batch(name, seed=20260929, particles=10, count=200, jobs=2):
    """Reuse the recorded groups and exact Step0–2 files, replace only Step3–4."""
    import json
    import multiprocessing
    import shutil
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from step3_scheculer.run_sequential_k import folder
    path = I.OUTPUTS/name/'step0_pose_selection'/f'batch_seed{seed}.json'
    if path.exists():
        previous = json.loads(path.read_text())
    else:
        # The shared ledger may have been removed during cleanup. Retained
        # group checks still identify the exact cases; never resample them.
        checks = [json.loads(p.read_text()) for p in
                  (I.OUTPUTS/name).glob('pose*+*/step0_pose_selection/report.json')]
        plan = sorted([c['poses'] for c in checks if c['object'] == name],
                      key=lambda poses:(len(poses), [int(p.split('_')[1]) for p in poses]))
        if not plan:
            raise ValueError('No saved groups to rerun')
        previous = dict(object=name, seed=seed, plan=plan,
                        requested_count=len(plan), selected_count=len(plan),
                        plan_source='retained group Step0 reports; removed selection ledger not reconstructed')
        path = folder(name, plan[-1], 'step3_scheculer').parents[2]/'step4/data'/f'batch_seed{seed}.json'
    if previous['object'] != name or not previous['plan'] or min(particles, count, jobs) < 1:
        raise ValueError('Invalid saved batch or search sizes')
    for poses in previous['plan']:
        group = folder(name, poses, 'step3_scheculer').parents[2]
        check = rebind_step0_inputs(group)
        if not check['passed'] or check['poses'] != poses:
            raise ValueError('Batch contains an unaccepted group')
        for pose in poses:
            candidate = folder(name, poses, 'step2_local_support')/f'candidates_{pose}.json'
            metadata = json.loads(candidate.read_text())
            I.check_hashes(metadata['provenance']['inputs'])
            if metadata['count'] != count or I.sha256(candidate.with_suffix('.npz')) != metadata['artifacts'][candidate.with_suffix('.npz').name]:
                raise ValueError('Saved candidates are stale or have a different count')
    # Clear replaced stages only after every group's inputs pass preflight.
    for poses in previous['plan']:
        group = folder(name, poses, 'step3_scheculer').parents[2]
        for stage in ('step3_scheculer', 'step4'):
            target = group/stage
            if target.exists():
                if target.is_symlink() or target.resolve().parent != group.resolve():
                    raise ValueError('Unexpected output path')
                shutil.rmtree(target)
    result = dict(previous, complete=False, results=[], rerun_from_step=3,
                  reused_step0_to_step2=True, inherited_whole_head_withdrawal=True)
    I.save(path, result)
    # Invalidate the previous summary before any result is replaced.
    (path.parent/'batch_report.md').write_text('八组正在从 Step3 重跑：每次加头继承全部 pose 的整组退出方向。完成后更新结果。\n')
    I.save(path.parent/'batch_review.json', dict(complete=False, status='rerunning_from_step3'))
    with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(_rerun_case, (name, poses, seed, particles, count)) for poses in result['plan']]
        for future in as_completed(futures):
            row = future.result()
            result['results'].append(row)
            I.save(path, result)
            print('CASE COMPLETE', row, flush=True)
    result['results'].sort(key=lambda r: result['plan'].index(r['selected_poses']))
    result['complete'] = True
    I.save(path, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object', nargs='?', default='B')
    parser.add_argument('--n', type=int, nargs='+', help='Pose count(s)')
    parser.add_argument('--from-step', type=int, choices=(0, 3), default=0,
                        help='3 reuses the saved batch plan and exact Step0–2 artifacts')
    parser.add_argument('--groups-per-n', type=int, default=1)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20260929)
    parser.add_argument('--particles', type=int, default=10)
    parser.add_argument('--candidates', type=int, default=200)
    parser.add_argument('--through-step', type=int, choices=range(5), default=4)
    args = parser.parse_args(argv)
    if args.from_step == 3:
        if args.n is not None or args.through_step != 4:
            parser.error('--from-step 3 uses the saved batch groups and runs through Step4')
        result = rerun_batch(args.object, args.seed, args.particles, args.candidates, args.jobs)
        print('BASELINE RERUN COMPLETE', result, flush=True)
        return 0 if all(r['status'] != 'pipeline_error' for r in result['results']) else 2
    if args.n is None:
        parser.error('--n is required when selecting new groups')
    batch = len(args.n) > 1 or args.groups_per_n != 1
    if batch:
        if args.through_step != 4:
            parser.error('Batch mode runs through Step4; use single-set mode for partial stages')
        result = run_batch(args.object, args.n, args.groups_per_n, args.seed,
                           args.particles, args.candidates, args.jobs)
    else:
        result = run_pipeline(args.object, args.n[0], args.seed, args.particles, args.candidates, args.through_step)
    import json
    print('BASELINE COMPLETE', json.dumps(result, ensure_ascii=False), flush=True)
    return (0 if result['selected_count'] == result['requested_count'] else 2) if batch else (0 if result['step0_passed'] else 2)


if __name__ == '__main__':
    raise SystemExit(main())
