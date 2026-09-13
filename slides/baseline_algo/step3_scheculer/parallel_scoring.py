"""Optional parallel precomputation of the unchanged Step3 candidate scores.

The original contribution.run still writes each result in candidate order.
Every worker executes the original classifier and independent LP verification
on the same complete reaction matrix. Existing serial checkpoints stay valid;
new rounds additionally record this wrapper's source hash.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from concurrent.futures import ProcessPoolExecutor
import hashlib
import multiprocessing
import os
from pathlib import Path
import runpy
import sys
import numpy as np

HERE = Path(__file__).resolve().parent.parent


def initialize(objects, pose, retry):
    os.environ['CADGRASP_POSE'] = pose
    if retry in (True, 'continuous_retry.py'):
        from step3_scheculer import continuous_retry
        continuous_retry.install(objects)
    elif retry == 'numerical_retry.py':
        from step3_scheculer import numerical_retry
        numerical_retry.install(objects)


def calculate(args):
    from step3_scheculer.stage_imports import load_stage
    C = load_stage('score', 'contribution')
    index, full, targets, base = args
    mask, classifier = C.J.classify(full, targets, known_covered=base)
    verification = C.verify_classification(full, targets, mask)
    return index, mask, classifier, verification


def key(full):
    return hashlib.sha256(np.ascontiguousarray(full).tobytes()).digest()


def install(objects, workers, retry=True):
    from step3_scheculer.stage_imports import load_stage
    from step3_scheculer import contacts as I
    from step1.cases import pose_name
    C = load_stage('score', 'contribution')
    original_run, original_hashes = C.run, C.code_hashes
    C.code_hashes = lambda: {**original_hashes(), **I.hashes([Path(__file__)])}
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                               initializer=initialize, initargs=(objects, pose_name(), retry))

    def run(name, round_number=1, state_path=None, problem=None):
        problem = problem or C.Problem(name)
        fixed, base, _ = problem.load_state(state_path)
        already = {p['candidate_index'] for p in fixed}
        fixed_full = problem.supply(fixed)
        jobs, keys = [], {}
        for index in range(len(problem.data.valid)):
            if index in already or not problem.data.valid[index]:
                continue
            full = I.merge_columns(fixed_full, C.columns(problem.domain, problem.data, index,
                                                       problem.floor, problem.scale))
            if not C.gravity_check(full,problem.domain,problem.scale)['passed']:
                continue
            keys[index] = key(full)
            jobs.append((index, full, problem.targets, base))
        cache = {}
        for count, (index, mask, classifier, verification) in enumerate(pool.map(calculate, jobs), 1):
            cache[keys[index]] = (mask, classifier, verification)
            if count % 10 == 0 or count == len(jobs):
                print(name, 'round', round_number, 'parallel exact candidate scores', count, '/', len(jobs), flush=True)
        classify, verify = C.J.classify, C.verify_classification
        hits = [0, 0]
        def cached_classify(full, targets, known_covered=None):
            row = cache.get(key(full))
            if row is None:
                return classify(full, targets, known_covered)
            np.testing.assert_array_equal(targets, problem.targets)
            np.testing.assert_array_equal(known_covered, base)
            hits[0] += 1
            return row[0].copy(), dict(row[1])
        def cached_verify(full, targets, accepted):
            row = cache[key(full)]
            np.testing.assert_array_equal(targets, problem.targets)
            np.testing.assert_array_equal(accepted, row[0])
            hits[1] += 1
            return row[2]
        C.J.classify, C.verify_classification = cached_classify, cached_verify
        try:
            result = original_run(name, round_number, state_path, problem)
            assert hits == [len(jobs), len(jobs)], hits
            return result
        finally:
            C.J.classify, C.verify_classification = classify, verify
    C.run = run
    return pool


if __name__ == '__main__':
    workers = int(os.environ.get('CADGRASP_SCORE_WORKERS', '6'))
    stage = Path(sys.argv[1]).resolve()
    objects = [s for s in sys.argv[2:] if s in ('A1-f', 'B', 'C5')]
    if len(objects) != 1 or workers < 1:
        raise ValueError('Expected one object and a positive score worker count')
    # runpy temporarily replaces __main__; spawned worker callables must be
    # importable from this module's stable name, not the displaced CLI module.
    from step3_scheculer import parallel_scoring
    pool = parallel_scoring.install(objects, workers, retry=stage.name)
    try:
        sys.argv = [str(stage)]+sys.argv[2:]
        runpy.run_path(str(stage), run_name='__main__')
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
