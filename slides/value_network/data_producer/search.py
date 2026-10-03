"""Budgeted, diverse completion labels using the baseline terminal verifier.

The two-pose mechanics/path checks factor by pose. Banks of certified local
completions are paired by exact finite-menu direction alignment; no head sharing.
These are best-found labels under a recorded search, not learned predictions.
"""
from collections import OrderedDict
from pathlib import Path
import argparse
import csv
import itertools
import json
import time
import numpy as np

from prepare import ROOT, BASELINE, I, prepare, compatible, intersect
from step3_scheculer.stage_imports import load_stage
J = load_stage('score', 'joint_samples')
from step3_scheculer.strict_lp_retry import recover


def install_recorded_recovery(output):
    original = J.C.W.solve
    def solve(full, target):
        try:
            return original(full, target)
        except RuntimeError as error:
            import hashlib
            digest = hashlib.sha256(full.tobytes()+np.asarray(target).tobytes()).hexdigest()
            record = dict(original_error=str(error), same_equations=True, tolerance_relaxed=False)
            try:
                witness, evidence = recover(full, target)
                record.update(evidence)
            except RuntimeError as recovery_error:
                record.update(status='numerical_unresolved', recovery_error=str(recovery_error))
                I.save(output/'numerical_retries'/f'{digest}.json', record)
                raise
            I.save(output/'numerical_retries'/f'{digest}.json', record)
            return witness
    J.C.W.solve = solve


class TerminalOracle:
    def __init__(self, problem, pool, cache_path):
        self.problem, self.pool, self.path = problem, pool, Path(cache_path)
        self.columns = [problem.supply([e['contact']]) if e['valid'] else None for e in pool]
        self.probes = np.linspace(0, len(problem.targets)-1, 96, dtype=int)
        self.records = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.calls = 0

    def check(self, indices):
        indices = tuple(sorted(set(indices)))
        key = ','.join(map(str, indices))
        if key in self.records:
            return self.records[key]['passed']
        if not compatible(self.pool, indices):
            return False
        full = I.merge_columns(*[self.columns[i] for i in indices])
        try:
            probe, probe_info = J.classify(full, self.problem.targets[self.probes])
            if not probe.all():
                result = dict(passed=False, method='actual_stored_load_counterexample',
                              failed_sample_indices=self.probes[~probe].tolist(), probe_info=probe_info)
            else:
                mask, info = J.classify(full, self.problem.targets)
                result = dict(passed=bool(mask.all()), sample_count=len(mask), covered_count=int(mask.sum()),
                              failed_sample_indices=np.flatnonzero(~mask)[:16].tolist(), classifier=info,
                              no_uplift_in_same_reaction_solve=True)
        except RuntimeError as error:
            result = dict(passed=None, method='numerical_unresolved', error=str(error))
        result['indices'] = list(indices)
        self.records[key] = result
        self.calls += 1
        if self.calls % 20 == 0:
            self.save()
        return result['passed']

    def save(self):
        I.save(self.path, self.records)


def prune(oracle, indices, forced, rng):
    """Delete only after an actual terminal pass, keeping the forced head."""
    result = sorted(set(indices))
    order = [i for i in result if i != forced]
    rng.shuffle(order)
    for i in order:
        trial = [j for j in result if j != i]
        if trial and oracle.check(trial):
            result = trial
    return tuple(result)


def seed_groups(name, pose, pool, oracle):
    folder = I.OUTPUTS/name/'independent_poses'/pose/'step3_scheculer'
    id_to_index = {e['id']: i for i,e in enumerate(pool)}
    groups = set()
    for path in sorted(folder.glob('particle_*/schedule.json')):
        report = json.loads(path.read_text())
        if report.get('passed'):
            ids = report['selected_ids']
            group = tuple(sorted(id_to_index[c] for c in ids))
            if oracle.check(group):
                groups.add(group)
    if not groups:
        raise RuntimeError(f'No replayed baseline warm-start solution for {pose}')
    return sorted(groups)


def grow_random(pool, forced, rng, cap):
    """Random compatible continuation, without intermediate no-uplift gates."""
    selected = [forced] if forced is not None else [int(rng.choice([i for i,e in enumerate(pool) if e['valid']]))]
    while len(selected) < cap:
        choices = [i for i,e in enumerate(pool) if e['valid'] and i not in selected
                   and compatible(pool, sorted(selected+[i]))]
        if not choices:
            break
        selected.append(int(rng.choice(choices)))
    return tuple(sorted(selected))


def make_bank(pool, oracle, seeds, attempts, rng, cap, output):
    """Explore dozens of complete sets by swaps and random compatible growth."""
    groups = set(seeds)
    traces = []
    valid = [i for i,e in enumerate(pool) if e['valid']]
    for trial in range(attempts):
        if trial % 4 == 3:
            proposed = grow_random(pool, None, rng, cap)
            strategy = 'random_compatible_growth'
        else:
            base = list(sorted(groups)[int(rng.integers(len(groups)))])
            remove = int(rng.choice(base))
            choices = [i for i in valid if i not in base and compatible(pool, sorted([j for j in base if j != remove]+[i]))]
            proposed = tuple(sorted([j for j in base if j != remove]+[int(rng.choice(choices))])) if choices else tuple(base)
            strategy = 'one_head_exchange'
        passed = oracle.check(proposed)
        final = prune(oracle, proposed, None, rng) if passed else None
        if final:
            groups.add(final)
        traces.append(dict(attempt=trial, strategy=strategy, proposed=list(proposed), passed=passed,
                           completion=list(final) if final else None))
        I.save(output, dict(attempts=traces, completions=[list(x) for x in sorted(groups)]))
        if (trial+1)%8 == 0:
            print('BANK', oracle.problem.pose, trial+1, 'attempts;', len(groups), 'certified sets', flush=True)
    return sorted(groups)


def force_head(pool, oracle, head, bank, attempts, rng, cap):
    """Equal attempt budget per legal action, with exact memoization of repeats."""
    completions = set()
    traces = []
    for trial in range(attempts):
        if trial % 4 == 3:
            proposed = grow_random(pool, head, rng, cap)
            strategy = 'forced_random_compatible_growth'
        else:
            base = bank[int(rng.integers(len(bank)))]
            proposed = tuple(sorted(set(base)|{head}))
            strategy = 'forced_seed_augmentation'
        path_ok = len(proposed) <= cap and compatible(pool, proposed)
        passed = path_ok and oracle.check(proposed)
        final = prune(oracle, proposed, head, rng) if passed else None
        if final:
            completions.add(final)
        traces.append(dict(attempt=trial, strategy=strategy, proposed=list(proposed), path_passed=path_ok,
                           passed=bool(passed), completion=list(final) if final else None,
                           terminal_status=oracle.records.get(','.join(map(str, proposed)), {}).get('method', 'checked_or_path_rejected')))
    return sorted(completions), traces


def object_directions(problem, catalogue):
    rotation = np.asarray(problem.domain.data['frame']['T_world_mesh'])[:3,:3]
    result = catalogue@rotation  # row-vector equivalent of R.T @ d_world
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1, atol=1e-12)
    return result


def pair_dispersion(pool_a, group_a, pool_b, group_b, pair_angles):
    da = sorted(intersect(pool_a, group_a, 'directions'))
    db = sorted(intersect(pool_b, group_b, 'directions'))
    sub = pair_angles[np.ix_(da,db)]
    flat = int(np.argmin(sub))
    i,j = np.unravel_index(flat, sub.shape)
    angle = float(sub[i,j])
    return (angle/np.pi)**2, (da[i],db[j]), float(np.rad2deg(angle))


def best_pair(own, other, task, pools, pair_angles, weight):
    best = None
    for local, counterpart in itertools.product(own, other):
        groups = [local,counterpart] if task == 0 else [counterpart,local]
        dispersion, direction_ids, degrees = pair_dispersion(pools[0],groups[0],pools[1],groups[1],pair_angles)
        remaining = sum(map(len,groups))-1
        score = remaining+weight*dispersion
        item = dict(value=score, remaining_heads=remaining, total_heads=remaining+1,
                    dispersion=dispersion, minimum_angle_deg=degrees, direction_ids=list(direction_ids),
                    completion_indices=[list(g) for g in groups])
        if best is None or (score,tuple(groups)) < (best['value'],tuple(tuple(g) for g in best['completion_indices'])):
            best = item
    return best


def run(args):
    output = ROOT/'slides/value_network/data'/args.object/'pose1+3'
    output.mkdir(parents=True, exist_ok=True)
    poses = ['pose_1','pose_3']
    problems,pools,catalogues,inputs = prepare(args.object,poses,output/'inputs')
    config = dict(object=args.object, poses=poses, attempts=args.attempts, seed=args.seed,
                  max_heads_per_pose=args.max_heads, lambda_direction=args.direction_weight,
                  state='empty head sets in both poses; force one head in its owner pose',
                  local_completion_search='baseline warm starts + random exchanges + augmentation + random growth + verified pruning',
                  failures='budget_unresolved; value=null', force_and_uplift='terminal only',
                  proposal_probe_loads=96, final_loads_per_pose=32768,
                  source_inputs=inputs, code=I.hashes([Path(__file__), Path(__file__).with_name('prepare.py'),Path(J.__file__),
                      Path(J.C.W.__file__), BASELINE/'step3_scheculer/passive_support.py',
                      BASELINE/'step3_scheculer/floor_support.py', BASELINE/'step3_scheculer/strict_lp_retry.py']))
    config_path = output/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Existing run has a different configuration or inputs; use a new output or remove only this generated run')
    I.save(config_path,config)
    install_recorded_recovery(output)
    oracles,banks=[],[]
    for k,(problem,pool) in enumerate(zip(problems,pools)):
        oracle=TerminalOracle(problem,pool,output/f'terminal_checks_{problem.pose}.json')
        oracles.append(oracle)
        seed=seed_groups(args.object,problem.pose,pool,oracle)
        bank_path=output/f'bank_{problem.pose}.json'
        if bank_path.exists() and len(json.loads(bank_path.read_text())['attempts']) == args.attempts:
            bank=[tuple(g) for g in json.loads(bank_path.read_text())['completions']]
        else:
            bank=make_bank(pool,oracle,seed,args.attempts,np.random.default_rng(args.seed+k),args.max_heads,bank_path)
        oracle.save();banks.append(bank)
    direction_vectors=[object_directions(p,c) for p,c in zip(problems,catalogues)]
    pair_angles=np.arccos(np.clip(direction_vectors[0]@direction_vectors[1].T,-1,1))
    rows=[]
    for k,(problem,pool,oracle) in enumerate(zip(problems,pools,oracles)):
        for index,entry in enumerate(pool):
            cid=entry['id']; path=output/'actions'/f'{cid}.json'
            if path.exists():
                row=json.loads(path.read_text())['summary']
            else:
                row=dict(id=cid,pose=problem.pose,index=index,valid=entry['valid'],value=None,
                         reason=entry['reason'],status='step2_rejected', attempts=0)
                traces=[]
                if entry['valid']:
                    found,traces=force_head(pool,oracle,index,banks[k],args.attempts,
                        np.random.default_rng(args.seed+1000*k+index+100),args.max_heads)
                    best=best_pair(found,banks[1-k],k,pools,pair_angles,args.direction_weight)
                    row.update(status='success_found' if best else 'budget_unresolved',attempts=args.attempts,
                               successful_attempts=sum(t['passed'] for t in traces),distinct_completions=len(found))
                    if best:
                        row.update(best)
                        row['completion_ids']=[[pools[t][i]['id'] for i in group]
                                               for t,group in enumerate(best['completion_indices'])]
                        row['object_exit_vectors']=[direction_vectors[t][i].tolist() for t,i in enumerate(best['direction_ids'])]
                I.save(path,dict(summary=row,continuation_attempts=traces))
            rows.append(row)
            if (index+1)%10 == 0:
                print('VALUES',problem.pose,index+1,'of 200;',sum(r['value'] is not None for r in rows),
                      'successful actions;',oracle.calls,'new terminal checks',flush=True)
                oracle.save()
                I.save(output/'progress.json',dict(complete=False,rows_completed=len(rows),rows=rows))
        oracle.save()
    I.save(output/'values.json',dict(complete=True,config=config,rows=rows,
        summary=dict(candidates=len(rows),valid=sum(r['valid'] for r in rows),
                     success_found=sum(r['value'] is not None for r in rows),
                     unresolved=sum(r['status']=='budget_unresolved' for r in rows)),
        label_scope='best found successful pair in the recorded conditional continuation searches; not global optimal value',
        complete_fixture_constructed=False))
    columns=['id','pose','index','valid','status','value','remaining_heads','total_heads','dispersion','minimum_angle_deg','successful_attempts','distinct_completions']
    with (output/'values.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    print('FINISHED',output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--object',default='B')
    parser.add_argument('--attempts',type=int,default=32)
    parser.add_argument('--seed',type=int,default=20261002)
    parser.add_argument('--max-heads',type=int,default=6)
    parser.add_argument('--direction-weight',type=float,default=1.)
    args=parser.parse_args()
    if args.attempts<1 or args.max_heads<1 or args.direction_weight<0:
        parser.error('Positive search sizes and nonnegative weight required')
    run(args)
