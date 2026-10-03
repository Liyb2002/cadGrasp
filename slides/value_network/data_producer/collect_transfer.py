"""Twenty conditional states in each of five training groups; no test labels."""
import json
from pathlib import Path
import sys
import time
HERE=Path(__file__).resolve().parent
TRAIN=HERE.parent/'train'
sys.path.insert(0,str(TRAIN))
from transfer_experiment import plan,ROOT,TrainingContext
from bank_teacher import Context
from collect_states import sample_states
from train import save,digest

def run():
    start=time.perf_counter();d=plan();root=Path(d['state_root']);root.mkdir(parents=True,exist_ok=True)
    ctx=TrainingContext(ROOT/'label_evidence')
    certificates={p:json.loads((TRAIN.parent/'data/B/pilot20_shared'/f'terminal_checks_{p}.json').read_text()) for p in ctx.poses}
    results=[]
    for k,(name,poses) in enumerate(d['train']):
        folder=root/name/'pilot20';folder.mkdir(parents=True,exist_ok=True)
        states=sample_states([ctx.pools[p] for p in poses],[ctx.banks[p] for p in poses],20,20261006+k)
        save(folder/'states.json',dict(poses=poses,states=states))
        finite=0
        for si,state in enumerate(states):
            record=ctx.labels(poses,[list(g) for g in state]);record.update(state_id=si,complete=True)
            # Audit each claimed completion against the immutable all-load certificate.
            for row in record['rows']:
                if row['value'] is None:continue
                finite+=1
                for p,g in zip(poses,row['completion_indices']):
                    evidence=certificates[p][','.join(map(str,sorted(g)))]
                    assert evidence['passed'] and evidence['covered_count']==32768 and evidence['no_uplift_in_same_reaction_solve']
            save(folder/f'state_{si:03d}.json',record)
            print('DATA',name,si+1,'/20',flush=True)
        results.append(dict(group=name,states=20,finite_values=finite))
    config=dict(groups=d['train'],state_root=d['state_root'],states_per_group=20,
                source_inputs=ctx.cfg['source_inputs'],training_scope='only five designated training combinations; reusable per-pose physical completion banks',
                label_policy='same Q = remaining heads + exit dispersion; existing certified bank completions, null when no witness; no test label/state collection',
                seed=20261006)
    save(ROOT/'train_config.json',config)
    save(root/'summary.json',dict(complete=True,groups=results,states=100,elapsed_s=time.perf_counter()-start))

if __name__=='__main__':run()
