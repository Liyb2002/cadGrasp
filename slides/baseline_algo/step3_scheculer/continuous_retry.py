"""Try direct primal enclosure certificates when the optional dual hull fails.

No contact geometry or sample decision is changed. A successful result still
has to pass the existing independent Step 3 certificate audit.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathlib import Path
import runpy
import sys
import numpy as np
from step3_scheculer import numerical_retry
HERE = Path(__file__).resolve().parent.parent


def install(objects):
    numerical_retry.install(objects)
    from step3_scheculer import verification as V, enclosure as E, contacts as I
    original = V.continuous_check
    original_hashes = V.C.code_hashes
    V.C.code_hashes = lambda: {**original_hashes(), **I.hashes([Path(__file__)])}
    def check(problem, contacts, certificate_path=None):
        result = original(problem, contacts, certificate_path)
        if result['status'] != 'unresolved' or certificate_path is None:
            return result
        full = problem.supply(contacts)
        attempts = []
        for sides, bands in [(8, 1), (16, 2), (32, 4)]:
            proof, arrays = E.contain(problem, full, sides, bands)
            attempts.append(dict(proof))
            print('Direct primal continuous retry:', problem.name, sides, bands, proof['status'], flush=True)
            if arrays is not None:
                path = Path(certificate_path); path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **arrays)
                proof.update(certificate_file=str(path.relative_to(I.ROOT)), certificate_sha256=I.sha256(path),
                    enclosure_code_sha256=I.sha256(E.__file__), enclosure_attempts=attempts,
                    preliminary_dual_test=result, direct_primal_retry_code_sha256=I.sha256(__file__))
                return proof
        return dict(result, direct_primal_retry_attempts=attempts)
    V.continuous_check = check
    return V


if __name__ == '__main__':
    stage = Path(sys.argv[1]).resolve()
    if HERE not in stage.parents: raise ValueError('Stage must be inside baseline_algo')
    objects = [v for v in sys.argv[2:] if v in ('A1-f', 'B', 'C5')]
    if len(objects) != 1: raise ValueError('Use one object/pose per retry process')
    verification = install(objects)
    sys.argv = [str(stage)]+sys.argv[2:]
    if stage == HERE/'step3_scheculer/verification.py':
        verification.run(objects[0])
    else:
        runpy.run_path(str(stage), run_name='__main__')
