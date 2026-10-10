"""Run the rotating-reuse-first correction; old search and results preserved."""
import argparse
import hashlib
import json
import shutil
import time
from common import *
from model import Model
from reuse_first import ReuseFirstSearch
from case_sets import CASES


def protected():
    # Published results are explicitly writable by this user request. Protect
    # the production source/docs and original inputs, not new output ledgers.
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in CO.rglob('*') if p.is_file() and p.suffix in ['.py', '.md']
            and not p.is_relative_to(CO/'output')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=['all']+list(CASES), default='pose1-10')
    parser.add_argument('--mode', choices=['joint', 'incremental', 'both'], default='both')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=8)
    parser.add_argument('--finalists', type=int, default=3)
    parser.add_argument('--branch-rounds', type=int, default=2)
    parser.add_argument('--volume-rounds', type=int, default=2)
    parser.add_argument('--start-result', type=Path)
    parser.add_argument('--start-layout', type=Path)
    args = parser.parse_args()
    if args.start_layout and (args.mode != 'joint' or args.case == 'all' or args.start_result):
        parser.error('--start-layout requires one joint case and no --start-result')
    if args.start_result and (args.case == 'all' or args.mode == 'both'):
        parser.error('--start-result requires one case/mode')
    args.out.mkdir(parents=True, exist_ok=False)
    before = protected()
    C.save(args.out/'protected_coopt_sources.json', before)
    runtime = args.out/'runtime_sources'
    runtime.mkdir()
    sources = list((HERE/'code').glob('*.py')) + C.code_sources() + [
        Path(C.J.__file__), Path(C.U.__file__), Path(C.WORK.__file__),
        Path(ConservativeExitClearance.sweep.__code__.co_filename), Path(C.S.swept_solid.__code__.co_filename)]
    hashes = {}
    for path in sorted(set(sources)):
        relative = path.resolve().relative_to(ROOT)
        destination = runtime/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    C.save(runtime/'manifest.json', dict(startup_sources=hashes, args={
        k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}, cases=CASES))
    rows = []
    cases = CASES if args.case == 'all' else {args.case:CASES[args.case]}
    modes = ['joint', 'incremental'] if args.mode == 'both' else [args.mode]
    for name, numbers in cases.items():
        model = Model([f'pose_{k}' for k in numbers])
        model.checkpoint_dir = args.out/name/'exact_states'
        model.worker_program = runtime/'slides/pose_set_search/code/exact_worker.py'
        for mode in modes:
            out = args.out/name/mode
            out.mkdir(parents=True)
            began = time.monotonic()
            try:
                search = ReuseFirstSearch(model, out, args.iterations, args.finalists, args.branch_rounds)
                search.volume_rounds = args.volume_rounds
                report = (search.resume_layout(args.start_layout, mode) if args.start_layout else
                          search.optimize_existing(args.start_result, mode) if args.start_result else
                          getattr(search, mode)())
                full_set = report['pose_count'] == len(numbers)
                passed = full_set and report['force_exit_work_passed'] and report.get('every_insertion_passed', True)
                rows.append(dict(case=name, strategy=mode, passed=passed, full_requested_set_evaluated=full_set,
                    report=str(out/'report.json'), counts=report['counts'], volume_cm3=report['volume_cm3'],
                    rotating_reuse_pose_count=report['rotating_reuse_pose_count'],
                    juxtaposed_pose_count=report['juxtaposed_pose_count'], seconds=time.monotonic()-began))
            except Exception as error:
                import traceback
                traceback.print_exc()
                row = dict(case=name, strategy=mode, passed=False, error=str(error), seconds=time.monotonic()-began)
                C.save(out/'error.json', row)
                rows.append(row)
            C.save(args.out/'batch.json', dict(complete=False, policy='rotate-first-selective-juxtapose', results=rows))
    assert before == protected(), 'Production Co-optimize source files changed'
    C.save(args.out/'batch.json', dict(complete=True, results=rows, coopt_sources_unchanged=True,
        policy='rotate-first-selective-juxtapose', passed=sum(r['passed'] for r in rows), case_count=len(rows)))
    print('REUSE-FIRST COMPLETE', sum(r['passed'] for r in rows), '/', len(rows), flush=True)


if __name__ == '__main__':
    main()
