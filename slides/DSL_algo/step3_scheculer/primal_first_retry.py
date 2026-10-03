"""Try the existing audited primal certificate before the optional dual hull."""
from pathlib import Path
import runpy
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer import continuous_retry

HERE = Path(__file__).resolve().parent.parent


def install(objects):
    V = continuous_retry.install(objects)
    from step3_scheculer import enclosure as E, contacts as I
    original = V.continuous_check
    original_hashes = V.C.code_hashes
    V.C.code_hashes = lambda: {**original_hashes(), **I.hashes([Path(__file__)])}

    def check(problem, contacts, certificate_path=None):
        if contacts and certificate_path is not None:
            full = problem.supply(contacts)
            attempts = []
            for sides, bands in [(8, 1), (16, 2), (32, 4)]:
                proof, arrays = E.contain(problem, full, sides, bands)
                attempts.append(dict(proof))
                print('Primal-first continuous check:', problem.name, sides, bands,
                      proof['status'], flush=True)
                if arrays is not None:
                    path = Path(certificate_path)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, **arrays)
                    proof.update(certificate_file=str(path.relative_to(I.ROOT)),
                                 certificate_sha256=I.sha256(path),
                                 enclosure_code_sha256=I.sha256(E.__file__),
                                 enclosure_attempts=attempts,
                                 primal_first_code_sha256=I.sha256(__file__))
                    return proof
        # An unsuccessful outer approximation is not a physical counterexample.
        # Retain the original proofs, counterexample search and retry behavior.
        return original(problem, contacts, certificate_path)

    V.continuous_check = check
    return V


if __name__ == '__main__':
    stage = Path(sys.argv[1]).resolve()
    if HERE not in stage.parents:
        raise ValueError('Stage must be inside baseline_algo')
    from step1.registry import active_objects
    objects = [v for v in sys.argv[2:] if v in active_objects()]
    if len(objects) != 1:
        raise ValueError('Use one object/pose per retry process')
    verification = install(objects)
    sys.argv = [str(stage)] + sys.argv[2:]
    if stage == HERE/'step3_scheculer/verification.py':
        verification.run(objects[0])
    else:
        runpy.run_path(str(stage), run_name='__main__')
