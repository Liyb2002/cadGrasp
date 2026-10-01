"""Optional time-bounded dual hull after the existing primal-first proofs.

A hull timeout is unresolved, never an infeasibility verdict. The child imports
the original cone implementation in a fresh spawned interpreter; no production
solver, geometry, tolerance, or proof is changed.
"""
import argparse
import math
import multiprocessing
from pathlib import Path
import runpy
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer import primal_first_retry

HERE = Path(__file__).resolve().parent.parent


def _hull_worker(connection, full, conditioned):
    try:
        from step3_scheculer.stage_imports import load_stage
        original = load_stage('score', 'wrench_cone')
        connection.send(('ok', original.cone(full, conditioned=conditioned)))
    except Exception as error:
        connection.send(('error', type(error).__name__ + ': ' + str(error)))
    finally:
        connection.close()


def _run_worker(target, args, timeout):
    """Receive before joining: a large hull must not fill a blocked pipe."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Hull timeout must be finite and positive')
    context = multiprocessing.get_context('spawn')
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=target, args=(send, *args))
    started = time.monotonic()
    try:
        process.start()
        send.close()
        remaining = max(0., timeout - (time.monotonic() - started))
        if not receive.poll(remaining):
            raise RuntimeError(f'Dual hull unresolved: timeout after {timeout:g} seconds')
        try:
            status, value = receive.recv()
        except EOFError as error:
            raise RuntimeError('Dual hull unresolved: worker exited without a result') from error
        if status != 'ok':
            raise RuntimeError('Dual hull unresolved: ' + str(value))
        return value
    finally:
        receive.close()
        send.close()
        if process.pid is not None:
            # The child may still be returning from its completed pipe write.
            process.join(timeout=.2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.)
            if process.is_alive():
                process.kill()
                process.join(timeout=2.)
            process.close()


def bounded_cone(full, conditioned=False, *, timeout=60.):
    return _run_worker(_hull_worker, (full, conditioned), timeout)


def install(objects, timeout=60.):
    timeout = float(timeout)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Hull timeout must be finite and positive')
    verification = primal_first_retry.install(objects)
    from step3_scheculer import contacts as I
    original_hashes = verification.C.code_hashes
    verification.C.code_hashes = lambda: {
        **original_hashes(), **I.hashes([Path(__file__)])}
    calls = []

    def cone(full, conditioned=False):
        started = time.monotonic()
        record = dict(timeout_seconds=timeout, ray_count=len(full),
                      conditioned=bool(conditioned))
        try:
            result = bounded_cone(full, conditioned, timeout=timeout)
            record.update(status='completed', facet_count=len(result))
            return result
        except RuntimeError as error:
            record.update(status='unresolved', reason=str(error))
            raise
        finally:
            record['elapsed_seconds'] = time.monotonic() - started
            calls.append(record)

    verification.C.W.cone = cone
    original_check = verification.continuous_check

    def check(problem, contacts, certificate_path=None):
        start = len(calls)
        result = original_check(problem, contacts, certificate_path)
        return dict(result, bounded_hull_protocol=dict(
            timeout_seconds=timeout, calls=calls[start:],
            code_sha256=I.sha256(__file__),
            timeout_verdict='unresolved',
            scope='Optional dual-hull computation is bounded; original primal proofs and strict counterexample checks are unchanged.'))

    verification.continuous_check = check
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hull-timeout', type=float, default=60.)
    parser.add_argument('stage')
    parser.add_argument('stage_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    stage = Path(args.stage).resolve()
    if HERE not in stage.parents:
        raise ValueError('Stage must be inside baseline_algo')
    from step1.registry import active_objects
    objects = [v for v in args.stage_args if v in active_objects()]
    if len(objects) != 1:
        raise ValueError('Use one object/pose per retry process')
    verification = install(objects, args.hull_timeout)
    sys.argv = [str(stage), *args.stage_args]
    if stage == HERE/'step3_scheculer/verification.py':
        verification.run(objects[0])
    else:
        runpy.run_path(str(stage), run_name='__main__')


if __name__ == '__main__':
    # runpy temporarily replaces __main__ with the selected stage. Keep the
    # spawned target in its stable importable module, never in that alias.
    from step3_scheculer.bounded_hull_retry import main as imported_main
    imported_main()
