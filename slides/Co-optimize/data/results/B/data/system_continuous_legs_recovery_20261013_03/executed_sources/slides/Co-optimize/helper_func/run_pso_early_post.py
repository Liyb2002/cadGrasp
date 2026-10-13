"""Postprocess completed cases sequentially while the ALL52 Step4 queues run."""
import argparse
import fcntl
import time
from run_pso_campaign import ROOT, BASE, POLICIES, read, save, execute, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases', nargs='+')
    parser.add_argument('--system', default='pso_comparison_all52_20261013_01')
    args = parser.parse_args()
    system = BASE / 'data' / ('system_' + args.system)
    token = time.strftime('%Y%m%d_%H%M%S', time.gmtime())
    with (system / 'early_postprocess.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify(system)
        for policy in ['differential', 'pso']:
            experiment = POLICIES[policy]
            source = read(BASE / 'data' / experiment / 'batch.json')
            if source['complete']:
                raise RuntimeError('The main postprocessor must own completed Step4 batches')
            selected = {r['id']: r for r in source['results'] if r['passed']}
            folder = BASE / 'data' / (experiment + '_sculpted')
            batch_path = folder / 'batch.json'
            batch = read(batch_path)
            done = {r['id'] for r in batch['results']}
            cases = [selected[label] for label in args.cases if label not in done]
            selection = folder / ('early_selection_' + token + '.json')
            save(selection, dict(complete=True, group_count=len(cases), results=cases,
                passed=len(cases), scope='Completed own-campaign cases only'))
            for case in cases:
                safe = case['id'].replace('/', '__')
                output = folder / 'workers' / (safe + '.json')
                assert not output.exists(), 'Preserve previous attempts before retrying'
                row = execute(system / 'execution_scripts/cadgrasp_pso_post_worker.py',
                    (folder / 'payloads' / (safe + '.json'), dict(
                        frozen=str(system / 'frozen_sources'), case=case, selection=str(selection),
                        initial=experiment + '_stock', aesthetic=experiment + '_rounded',
                        final=experiment + '_sculpted', floor=experiment + '_floor', result=str(output))),
                    output, folder / 'logs' / (safe + '.log'))
                row['timing_context'] = 'One sequential early Step5 worker while both Step4 queues run'
                save(output, row)
                batch['results'].append(row)
                batch.update(passed=sum(r['passed'] for r in batch['results']),
                    seconds=batch['seconds'] + row['process_wall_seconds'], early_postprocess=True)
                batch.setdefault('early_execution_segments', []).append(dict(
                    token=token, id=case['id'], jobs=1, seconds=row['process_wall_seconds']))
                save(batch_path, batch)
                print('EARLY STEP5', policy, case['id'], row['passed'], row.get('error'), flush=True)
        verify(system)


if __name__ == '__main__':
    main()
