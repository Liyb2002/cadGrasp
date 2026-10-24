"""Independent top-k stochastic greedy chains. No resampling or replacement."""
import copy
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from step3_scheculer import contacts as I
from step3_scheculer import paths as PTH
from step3_scheculer import completion as Q


def configuration(mode='top5', trajectories=10, seed=0, top_k=5, sizing_sweeps=2, sizing_budget=96):
    if mode not in ('top5', 'greedy', 'smc'):
        raise ValueError('Unknown Step3 search mode')
    if trajectories < 1 or seed < 0 or top_k < 1 or sizing_sweeps < 1:
        raise ValueError('Positive chain count/top-k/sweeps and a nonnegative seed are required')
    if sizing_budget < 3*sizing_sweeps*Q.MAX_CONTACTS:
        raise ValueError('Sizing budget must allow three evaluations per coordinate per sweep')
    return dict(mode=mode, trajectories=trajectories, seed=seed, top_k=top_k,
                sizing_sweeps=sizing_sweeps, sizing_budget=sizing_budget)


def chain_seed(seed, index):
    return int(np.random.SeedSequence([seed, index]).generate_state(1, dtype=np.uint64)[0])


def result_key(result):
    area = result['rounds'][-1]['total_area_m2'] if result['rounds'] else 0.
    if result['continuous_coverage_proved']:
        return (0, area, result['contact_count'])
    # Compare the same original load set across chains with different counterexamples.
    return (1, -result['random_covered_count'], area)


def reuse_first_score(name, problem, source):
    from step3_scheculer import scheduler as S
    tick = time.monotonic()
    report = copy.deepcopy(I.check_report(source/'contributions.json'))
    assert report['object'] == name and report['round'] == 1
    assert report['sample_count'] == len(problem.targets) and not report['selected_indices']
    out = PTH.folder(name, S.C.OUTPUT_NAME, 1)
    out.mkdir(parents=True, exist_ok=True)
    for filename in report['artifacts']:
        shutil.copyfile(source/filename, out/filename)
    report['provenance']['inputs'] = I.hashes(problem.inputs+[source/'contributions.json'])
    report['provenance']['code'].update(I.hashes([Path(__file__)]))
    report['reused_from'] = str(source.relative_to(I.ROOT))
    report['reused_scoring_seconds'] = report['elapsed_seconds']
    report['elapsed_seconds'] = time.monotonic()-tick
    I.save(out/'contributions.json', report)
    I.save(out/'status.json', dict(object=name, complete=True, status='scored',
                                 contributions_sha256=I.sha256(out/'contributions.json')))
    return report


def run(name, config, draw=True, resume=False):
    from step3_scheculer import scheduler as S
    tick = time.monotonic()
    out = PTH.stage_folder(name, S.OUTPUT_NAME)
    out.mkdir(parents=True, exist_ok=True)
    I.save(out/'status.json', dict(object=name, complete=False, status='searching_independent_chains'))
    results, folders, reused = [], [], []
    first_score = {}
    try:
        for index in range(config['trajectories']):
            seed = chain_seed(config['seed'], index)
            expected = dict(seed=seed, top_k=config['top_k'],
                            sizing_sweeps=config['sizing_sweeps'], sizing_budget=config['sizing_budget'])
            with PTH.trajectory(index):
                folder = PTH.stage_folder(name, S.OUTPUT_NAME)
                folders.append(folder)
                result = None
                can_resume = False
                if resume:
                    try:
                        settings = json.loads((folder/'settings.json').read_text())
                        can_resume = settings == expected
                        if can_resume:
                            result = I.check_report(folder/'schedule.json')
                            assert result['search_mode'] == 'top5' and result['search_config'] == expected
                            if draw:
                                for row in result['rounds']:
                                    for stage, filename in [(S.M.OUTPUT_NAME, 'selection_views.json'),
                                                            (S.A.OUTPUT_NAME, 'adjustment_views.json')]:
                                        assert (PTH.folder(name, stage, row['round'])/filename).is_file()
                    except (OSError, RuntimeError, AssertionError, KeyError, ValueError):
                        result = None
                I.save(folder/'settings.json', expected)
                print(name, 'trajectory', index+1, '/', config['trajectories'], 'seed', seed, flush=True)
                if result is None:
                    result = S.run_chain(name, draw=draw, resume=can_resume,
                                         first_score=first_score, **expected)
                else:
                    reused.append(index)
                score_path = PTH.folder(name, S.C.OUTPUT_NAME, 1)
                if (score_path/'contributions.json').exists() and 'source' not in first_score:
                    first_score['source'] = score_path
                results.append(result)
        winner = min(range(len(results)), key=lambda i: (*result_key(results[i]), i))
        source = folders[winner]
        result = copy.deepcopy(results[winner])
        shutil.copyfile(source/'final_contacts.npz', out/'final_contacts.npz')
        direction = copy.deepcopy(I.check_report(source/'insertion_directions.json'))
        direction['actual_contacts_file'] = str((out/'final_contacts.npz').relative_to(I.ROOT))
        direction['provenance']['inputs'].update(I.hashes([source/'insertion_directions.json', out/'final_contacts.npz']))
        I.save(out/'insertion_directions.json', direction)
        summaries = [dict(index=i, seed=chain_seed(config['seed'], i), status=r['status'],
                          continuous_coverage_proved=r['continuous_coverage_proved'],
                          random_covered_count=r['random_covered_count'],
                          total_area_m2=r['rounds'][-1]['total_area_m2'] if r['rounds'] else 0.,
                          schedule=str((folders[i]/'schedule.json').relative_to(I.ROOT)))
                     for i,r in enumerate(results)]
        result.update(search_mode='top5_independent', search_config=config,
                      selected_trajectory=winner, trajectories=summaries,
                      reused_trajectories=reused, elapsed_seconds=time.monotonic()-tick,
                      wall_timings=[dict(trajectory=i, **row) for i,r in enumerate(results)
                                    if i not in reused for row in r['wall_timings']],
                      insertion_direction_record=str((out/'insertion_directions.json').relative_to(I.ROOT)))
        result['provenance']['inputs'].update(I.hashes([f/'schedule.json' for f in folders]))
        result['provenance']['code'].update(I.hashes([Path(__file__)]))
        # Winner's round certificates remain in its own stage directory.
        result['artifacts'] = {os.path.relpath(source/f, out): digest
                               for f,digest in result['artifacts'].items()}
        result['artifacts'].update({f:I.sha256(out/f) for f in ['final_contacts.npz','insertion_directions.json']})
        I.save(out/'schedule.json', result)
        I.save(out/'status.json', dict(object=name, complete=True, status=result['status'],
                                     continuous_domain_status=result['continuous_domain_status'],
                                     physical_supports_verified=False,
                                     schedule_sha256=I.sha256(out/'schedule.json')))
        print(name, 'selected trajectory', winner+1, 'of', len(results), result['status'], flush=True)
        return result
    except Exception as error:
        I.save(out/'status.json', dict(object=name, complete=False, status='search_error',
                                     error_type=type(error).__name__, error=str(error)))
        raise
