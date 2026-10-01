"""Run the unchanged two-pose search with a separate, explicit head budget."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I, run_pairs as R, retry_pairs
from step3_scheculer.pair_tasks import completion_folder as pair_folder, sample_pairs, canonical_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object')
    parser.add_argument('--poses', nargs=2)
    parser.add_argument('--pairs', type=int, default=1)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--particles', type=int, default=10)
    parser.add_argument('--candidates', type=int, default=200)
    parser.add_argument('--max-heads', type=int, required=True)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if min(args.particles, args.candidates, args.max_heads, args.workers) < 1:
        parser.error('Positive search sizes required')

    def folder(name, poses, stage):
        return pair_folder(name, poses, stage)/f'heads_{args.max_heads}'

    if not args.poses:
        pairs, seed = sample_pairs(args.object, args.pairs, args.seed)
        print('Head budget:', args.max_heads, 'seed:', seed, 'pairs:', pairs, flush=True)
        def launch(poses):
            out = folder(args.object, poses, 'step0_pose_selection')
            out.mkdir(parents=True, exist_ok=True)
            R.save(out/'batch_plan.json', dict(object=args.object, pair=poses, pairs=pairs,
                seed=seed, particles=args.particles, candidates=args.candidates, max_heads=args.max_heads))
            command = [sys.executable, str(Path(__file__).resolve()), args.object,
                '--poses', *poses, '--seed', str(seed), '--particles', str(args.particles),
                '--candidates', str(args.candidates), '--max-heads', str(args.max_heads)]
            with (out/'run.log').open('w') as log:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
            return poses, code
        errors = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(launch, p) for p in pairs]):
                poses, code = future.result()
                print('Finished', poses, 'exit', code, flush=True)
                if code:
                    errors.append(poses)
        if errors:
            raise RuntimeError(f'Head-budget execution errors: {errors}; see per-case logs')
        return

    poses = canonical_pair(args.poses)
    baseline = pair_folder(args.object, poses, 'step3_scheculer')/'schedule.json'
    baseline_hash = I.sha256(baseline) if baseline.exists() else None
    original_run = R.PairSearch.run
    def run(search):
        result = original_run(search)
        result['provenance']['code'].update(I.hashes([Path(__file__)]))
        result['head_budget_experiment'] = dict(max_heads=args.max_heads,
            comparison_report=str(baseline.relative_to(I.ROOT)) if baseline_hash else None,
            comparison_sha256=baseline_hash, other_search_rules_unchanged=True)
        R.save(search.out/'schedule.json', result)
        return result
    R.PairSearch.run = run
    R.pair_folder = folder
    retry_pairs.pair_folder = folder
    retry_pairs.main()


if __name__ == '__main__':
    main()
