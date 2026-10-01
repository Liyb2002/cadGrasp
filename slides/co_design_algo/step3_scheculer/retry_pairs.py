"""Retry one explicit pose pair with recorded original-equation LP recovery.

Only solver exceptions enter recovery; normal solves and search rules are
unchanged. Keep this separate entry so completed pair provenance remains valid.
"""
import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I, strict_lp_retry
from step3_scheculer import run_pairs as R
from step3_scheculer.pair_tasks import completion_folder as pair_folder


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('object')
    parser.add_argument('--poses', nargs=2, required=True)
    args, _ = parser.parse_known_args()
    folder = pair_folder(args.object, args.poses, 'step3_scheculer')/'numerical_retries'
    original_solve, original_run = R.C.W.solve, R.PairSearch.run
    source_hashes = I.hashes([Path(__file__), Path(strict_lp_retry.__file__)])
    recoveries = {}

    def solve(full, target):
        try:
            return original_solve(full, target)
        except RuntimeError as error:
            target = R.C.U.target(target, full.shape[1])
            token = hashlib.sha256(full.tobytes()+target.tobytes()).hexdigest()
            folder.mkdir(parents=True, exist_ok=True)
            arrays = folder/(token+'.npz')
            np.savez_compressed(arrays, full=full, target=target)
            R.save(folder/(token+'.json'), dict(complete=False, original_error=str(error)))
            witness, evidence = strict_lp_retry.recover(full, target)
            if witness is None:
                separator = R.C.W.exact_separator(full, target)
                if separator is None:
                    raise RuntimeError('LP recovery has no primal witness or exact separator') from error
                evidence['exact_separator'] = separator
            record = dict(complete=True, original_error=str(error), same_equations=True,
                tolerance_relaxed=False, witness=witness, **evidence,
                provenance=dict(inputs={}, code=source_hashes),
                artifacts={arrays.name: I.sha256(arrays)})
            R.save(folder/(token+'.json'), record)
            recoveries[token] = evidence['status']
            print('Original-equation LP recovery:', evidence['status'], flush=True)
            return witness

    def run(search):
        result = original_run(search)
        result['provenance']['code'].update(source_hashes)
        result['numerical_recovery'] = dict(entry='retry_pairs.py', recovered_cases=recoveries,
            evidence_directory=str(folder.relative_to(I.ROOT)), tolerance_relaxed=False)
        R.save(search.out/'schedule.json', result)
        return result

    R.C.W.solve = solve
    R.PairSearch.run = run
    R.main()


if __name__ == '__main__':
    main()
