"""Summarize saved proposal trajectories without changing numerical results."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from run_pso_campaign import BASE, POLICIES, read, save


def summarize(events):
    moves = [e for e in events if 'move' in e and 'raw' in e]
    result = dict(events=len(events), recorded_moves=len(moves),
        accepted=sum(e['accepted'] for e in events),
        errors=dict(Counter(e['error'] for e in events if 'error' in e)),
        raw_proxy_loss_decreased=sum(e['raw']['loss'] < e['before']['loss'] - 1e-12 for e in moves),
        raw_failed_demands_decreased=sum(e['raw']['failed_demands'] < e['before']['failed_demands'] for e in moves),
        raw_same_layout=sum(e['raw']['id'] == e['before']['id'] for e in moves),
        accepted_after_adjustment=sum(e['accepted'] and e['trial']['id'] != e['raw']['id'] for e in moves))
    if moves and moves[0]['branch'] == 'pso':
        norms = np.array([np.linalg.norm(e['move']['velocity_m']) for e in moves])
        result.update(zero_translation=int((norms < 1e-12).sum()),
            less_than_1mm=int((norms < .001).sum()),
            proposed_translation_mm=dict(min=float(norms.min() * 1000),
                median=float(np.median(norms) * 1000), max=float(norms.max() * 1000)),
            zero_components={key: sum(np.linalg.norm(e['move'][key + '_component_m']) < 1e-12
                for e in moves) for key in ['inertia', 'cognitive', 'social']})
        result['zero_components'] = {k: int(v) for k, v in result['zero_components'].items()}
    return result


def analyze(system_name):
    system = BASE / 'data' / ('system_' + system_name)
    batches = {p: read(BASE / 'data' / exp / 'batch.json') for p, exp in POLICIES.items()}
    cases = []
    for label in batches['pso']['sets']:
        row = dict(id=label)
        first_rounds = {}
        for policy, experiment in POLICIES.items():
            finished = next((r for r in batches[policy]['results'] if r['id'] == label), None)
            path = BASE / label / 'step4/step4.2' / experiment / 'population32_trace.json'
            trace = read(path) if path.exists() else []
            events = [e for r in trace for e in r['events']]
            row[policy] = dict(finished=finished is not None,
                passed=finished['passed'] if finished else None, rounds=len(trace),
                wall_seconds=finished.get('process_wall_seconds') if finished else None,
                branches={branch: summarize([e for e in events if e['branch'] == branch])
                    for branch in sorted({e['branch'] for e in events})})
            if trace:
                first_rounds[policy] = trace[0]['events']
        if len(first_rounds) == 2:
            row['first_round_events_identical'] = first_rounds['pso'] == first_rounds['differential']
        cases.append(row)
    paired = [r for r in cases if all(r[p]['finished'] for p in POLICIES)]
    summary = dict(complete=all(b['complete'] for b in batches.values()),
        paired_finished=len(paired), paired_passing=sum(all(r[p]['passed'] for p in POLICIES) for r in paired),
        pso_fewer_rounds=sum(r['pso']['rounds'] < r['differential']['rounds'] for r in paired),
        pso_more_rounds=sum(r['pso']['rounds'] > r['differential']['rounds'] for r in paired),
        pso_equal_rounds=sum(r['pso']['rounds'] == r['differential']['rounds'] for r in paired))
    result = dict(summary=summary, cases=cases,
        caveats=['Proxy loss alone is not the physical acceptance rank.',
                 'Recorded velocity precedes the unchanged floor projection and height repair.',
                 'Timing includes the same local adjustment and physical checks.',
                 'Incomplete traces are observations, not failed cases.'])
    save(system / 'proposal_diagnostics.json', result)
    print('PROPOSAL DIAGNOSTICS', summary, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--system', default='pso_comparison_all52_20261013_01')
    analyze(parser.parse_args().system)


if __name__ == '__main__':
    main()
