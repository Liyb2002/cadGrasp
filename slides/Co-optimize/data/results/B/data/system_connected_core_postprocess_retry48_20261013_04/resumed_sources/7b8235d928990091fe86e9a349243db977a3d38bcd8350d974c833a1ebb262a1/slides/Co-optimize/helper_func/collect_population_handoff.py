"""Keep running independent workers and collect a disjoint remaining batch."""
from pathlib import Path
import argparse,json,time
from run_population_soft_batch import ROOT,BASE,sha,save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment');parser.add_argument('--remaining',required=True)
    args=parser.parse_args();system=BASE/'data'/('system_'+args.experiment)
    handoff=json.loads((system/'handoff.json').read_text())
    folder=BASE/'data'/args.experiment;path=folder/'batch.json'
    batch=json.loads(path.read_text());original_ids=set(handoff['original_partition'])
    remaining_path=BASE/'data'/args.remaining/'batch.json'
    known={r['id']:r for r in batch['results']};printed=set(known)
    while True:
        for label in original_ids:
            result_path=BASE/label/'step4/step4.2'/args.experiment/'run_result.json'
            if label not in known and result_path.exists():
                row=json.loads(result_path.read_text())
                row.update(experiment=args.experiment,memory_limit_gib=16,
                    execution_log=str((folder/'logs'/(label.replace('/','__')+'.log')).relative_to(ROOT)))
                known[label]=row
        remaining=json.loads(remaining_path.read_text())if remaining_path.exists()else None
        if remaining is not None:
            other=dict(remaining['options']);other.pop('output_name')
            expected=dict(batch['options']);expected.pop('output_name');assert other==expected
            assert not original_ids&set(remaining['sets'])
            for row in remaining['results']:known[row['id']]=row
        assert set(known)<=set(batch['sets'])
        complete=(remaining is not None and remaining['complete']and
            original_ids<=set(known)and len(known)==batch['group_count'])
        batch.update(complete=complete,results=list(known.values()),passed=sum(r['passed']for r in known.values()),
            seconds=time.time()-handoff['started_epoch'],execution_partitions=handoff['partitions'])
        if complete:
            protected=json.loads((system/'protected_originals.json').read_text())
            changed=[p for p,h in protected.items()if not(ROOT/p).is_file()or sha(ROOT/p)!=h]
            files=json.loads((system/'frozen_sources.json').read_text())['files']
            changed_frozen=[r['frozen']for r in files if sha(ROOT/r['frozen'])!=r['frozen_sha256']]
            batch.update(protected_unchanged=not changed,frozen_sources_unchanged=not changed_frozen)
            save(system/'handoff_verification.json',dict(changed_originals=changed,changed_frozen=changed_frozen,
                original_files_checked=len(protected),frozen_files_checked=len(files)))
        save(path,batch)
        for label in set(known)-printed:
            print('CONNECTED STEP4',len(known),'/',batch['group_count'],'PASS',batch['passed'],label,
                known[label].get('error',''),flush=True)
        printed=set(known)
        if complete:break
        time.sleep(2.)
    print('CONNECTED STEP4 COMPLETE',len(known),'PASS',batch['passed'],flush=True)


if __name__=='__main__':main()
