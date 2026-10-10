"""Whole-gradient search with bounded own-state repair and mesh conditioning.

Both stages have explicit budgets and separate outputs. Conditioning starts
from this run's OWN optimized state; it never imports a historical answer.
"""
import argparse
import json
import sys
from pathlib import Path
import run_stable_gradient
import run_stable_conditioning
import run_fast_state_seat_gradient
from co_common import HERE, I, save


def own_guidance_counts(result, out):
    if result['passed']:
        return None
    report = out / 'search_report.json'
    if not report.exists() or not (out / 'sampled_layout.npz').exists():
        return None
    data = json.loads(report.read_text())
    return data['counts'] or None


def needs_conditioning(result, out):
    counts = own_guidance_counts(result, out)
    return counts is not None and all(n == 32768 for n in counts.values())


def invoke(main, arguments):
    original = sys.argv
    sys.argv = [original[0], *arguments]
    try:
        main()
    finally:
        sys.argv = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sets', nargs='+', required=True)
    parser.add_argument('--jobs', type=int, default=3)
    parser.add_argument('--worker-cpus', type=int, default=6)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--repair-iterations', type=int, default=8)
    parser.add_argument('--demand-nodes', type=int, default=32)
    parser.add_argument('--output-name', default='stable_gradient_v17')
    args = parser.parse_args()
    base = HERE / 'output/B'
    common = ['--sets', *args.sets, '--jobs', str(args.jobs),
              '--iterations', str(args.iterations), '--demand-nodes', str(args.demand_nodes)]
    invoke(run_stable_gradient.main, [*common, '--worker-cpus', str(args.worker_cpus),
                                     '--output-name', args.output_name])
    directory = base / 'data' / args.output_name
    first = json.loads((directory / 'batch.json').read_text())
    selected = {r['id']: dict(r, experiment=args.output_name) for r in first['results']}
    repair_guests = [r['id'] for r in first['results']
        if (counts := own_guidance_counts(r, base / r['id'] / 'step4/step4.2' / args.output_name))
        and any(n != 32768 for n in counts.values())]
    repaired = None
    if repair_guests:
        repair_name = args.output_name + '_repair'
        invoke(run_fast_state_seat_gradient.main, ['--sets', *repair_guests, '--jobs', str(args.jobs),
            '--worker-cpus', str(args.worker_cpus), '--iterations', str(args.repair_iterations),
            '--demand-nodes', str(args.demand_nodes), '--resume-experiment', args.output_name,
            '--output-name', repair_name])
        repaired = json.loads((base / 'data' / repair_name / 'batch.json').read_text())
        for r in repaired['results']:
            selected[r['id']] = dict(r, experiment=repair_name, additional_gradient_budget=True)
    # A mixed set of source experiments is handled separately; each fallback
    # loads only its own immediately preceding state's layout.
    conditioning_sources = {}
    for r in selected.values():
        if needs_conditioning(r, base / r['id'] / 'step4/step4.2' / r['experiment']):
            conditioning_sources.setdefault(r['experiment'], []).append(r['id'])
    pending = [label for labels in conditioning_sources.values() for label in labels]
    second = None
    conditioned = []
    for source, labels in conditioning_sources.items():
        name = source + '_conditioning'
        invoke(run_stable_conditioning.main, ['--sets', *labels, '--jobs', str(min(args.jobs, 2)),
            '--demand-nodes', str(args.demand_nodes), '--resume-experiment', source,
            '--output-name', name])
        second = json.loads((base / 'data' / name / 'batch.json').read_text())
        conditioned.append(second)
        for r in second['results']:
            if r['passed']:
                selected[r['id']] = dict(r, experiment=name, additional_conditioning_budget=True)
    save(directory / 'pipeline.json', dict(complete=True, policy='stable_whole_gradient_then_fast_state_seat_repair_then_joint_conditioning',
        groups=args.sets, base_budget_passes=sum(r['passed'] for r in first['results']),
        gradient_repair_passes=sum(r['passed'] for r in repaired['results']) if repaired else 0,
        conditional_passes=sum(r['passed'] for b in conditioned for r in b['results']),
        passed=sum(r['passed'] for r in selected.values()), selected=selected,
        gradient_repair_guests=repair_guests,conditioning_guests=pending,
        maximum_additional_gradient_rounds=args.repair_iterations,old_optimized_answers_used_as_start=False,
        physical_acceptance_unchanged=True, code=I.hashes([Path(__file__),
            Path(run_stable_gradient.__file__), Path(run_fast_state_seat_gradient.__file__),
            Path(run_stable_conditioning.__file__)])))
    print('PIPELINE', sum(r['passed'] for r in selected.values()), '/', len(args.sets), flush=True)


if __name__ == '__main__':
    main()
