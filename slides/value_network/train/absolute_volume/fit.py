"""Train only on successful, actual Step4/Step5 continuation evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from model import AbsoluteValueNetwork

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = HERE.parents[1] / 'absolute_direction/data/successful_completion_targets.jsonl'
DATA = HERE.parents[1] / 'data/B/absolute_volume'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + '\n')


def prepare(rows, samples=64, seed=20261003):
    poses = sorted({p for r in rows for p in r['poses']}, key=lambda p:int(p.split('_')[1]))
    candidates = []
    for pose in poses:
        path = HERE.parents[1] / 'baseline_algo/output/B/independent_poses' / pose / 'step2_local_support' / f'candidates_{pose}.npz'
        with np.load(path) as a: candidates.extend(a['candidate_ids'].tolist())
    index = {ident:i for i,ident in enumerate(candidates)}
    pindex = {p:i for i,p in enumerate(poses)}
    h, p = len(candidates), len(poses)
    rng = np.random.default_rng(seed)
    states=[]; members=[]; labels=[]; masks=[]; costs=[]; orders=[]; offsets=[]; groups=[]
    for row in rows:
        member=np.zeros(p,np.float32); mask=np.zeros(h,np.float32)
        chosen=np.zeros(h,np.float32); order=np.zeros(h,np.float32); seating=np.zeros((p,2),np.float32)
        for k,pose in enumerate(row['poses']):
            member[pindex[pose]]=1
            mask[[i for i,ident in enumerate(candidates) if ident.startswith(pose+'_')]]=1
            ids=row['successful_completion']['selected_ids'][k]
            for j,ident in enumerate(ids):
                chosen[index[ident]]=1; order[index[ident]]=j/max(len(ids),1)
            seating[pindex[pose]]=np.asarray(row['successful_completion']['placement']['offsets'][k])[:2]/.2
        positive=np.flatnonzero(chosen)
        for s in range(samples):
            state=np.zeros(h,np.float32)
            if s:
                if s%3==0:
                    # Canonical prefixes as well as unordered partial sets.
                    for ids in row['successful_completion']['selected_ids']:
                        for ident in ids[:rng.integers(len(ids)+1)]:state[index[ident]]=1
                else:state[rng.choice(positive, rng.integers(len(positive)+1), replace=False)]=1
            states.append(state); members.append(member); labels.append(chosen*(1-state))
            masks.append(mask*(1-state)); costs.append(np.log1p(row['best_observed_continuation_cost']))
            orders.append(order); offsets.append(seating.ravel()); groups.append(row['group'])
    arrays=dict(state=np.array(states),membership=np.array(members),evidence=np.array(labels),
                active=np.array(masks),cost=np.array(costs,np.float32),order=np.array(orders),
                seating=np.array(offsets),groups=np.array(groups))
    return arrays, dict(candidates=candidates, poses=poses, width=256, objective='actual_step5_extra_occupied_volume',
        head_count_penalty=0, seating_grid_m=.005, q_transform='log1p(cost)',
        evidence_semantics='membership in a witnessed successful completion; absence is unknown feasibility')


def metrics(model, d):
    with torch.no_grad():
        e,q,o,xy=model(d['state'],d['membership'])
        active=d['active'].bool(); positive=d['evidence'].bool()
        errors=(q-d['cost'][:,None]).abs()[positive]
        return dict(evidence_accuracy=float(((e>0)==positive)[active].float().mean()),
            all_states_exact=bool((((e>0)==positive)|~active).all()),
            maximum_positive_log_cost_error=float(errors.max()),
            seating_maximum_error_m=float(((xy-d['seating']).abs()*d['membership'].repeat_interleave(2,dim=1)).max()*.2),
            order_maximum_error=float((o-d['order']).abs()[positive].max()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--epochs',type=int,default=2000);args=parser.parse_args()
    torch.manual_seed(20261003);torch.set_num_threads(4)
    rows=[json.loads(line) for line in SOURCE.read_text().splitlines()]
    for row in rows:
        for relative,digest in row['provenance'].items():
            assert sha(ROOT/relative)==digest, relative
    arrays,manifest=prepare(rows)
    validation,_=prepare(rows,16,seed=38114)
    DATA.mkdir(parents=True,exist_ok=True);np.savez_compressed(DATA/'states.npz',**arrays)
    save(DATA/'manifest.json',dict(**manifest,source_sha256=sha(SOURCE),successful_completions=len(rows),
        training_states=len(arrays['state']),evaluation_states=len(validation['state']),
        evaluation_scope='new partial states of the same training pose groups; no unseen-group claim',
        positive_action_labels=int(arrays['evidence'].sum()),negative_volume_labels=0))
    device='cuda' if torch.cuda.is_available() else 'cpu'
    d={k:torch.as_tensor(v,device=device) for k,v in arrays.items() if k!='groups'}
    v={k:torch.as_tensor(a,device=device) for k,a in validation.items() if k!='groups'}
    model=AbsoluteValueNetwork(len(manifest['candidates']),len(manifest['poses'])).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-6)
    start=time.monotonic();history=[]
    for epoch in range(args.epochs):
        if epoch == 600:
            for param in optimizer.param_groups:param['lr']=.0002
        model.train();optimizer.zero_grad()
        e,q,o,xy=model(d['state'],d['membership']);pos=d['evidence'];active=d['active']
        # Non-demonstrated actions are imitation negatives, never failed-volume labels.
        weights=active*(1+12*pos)
        evidence=(F.binary_cross_entropy_with_logits(e,pos,reduction='none')*weights).sum()/weights.sum()
        cost=(((q-d['cost'][:,None])**2)*pos).sum()/pos.sum()
        order=(((o-d['order'])**2)*pos).sum()/pos.sum()
        xy_mask=d['membership'].repeat_interleave(2,dim=1)
        layout=(((xy-d['seating'])**2)*xy_mask).sum()/xy_mask.sum()
        loss=evidence+10*cost+2*order+20*layout;loss.backward();optimizer.step()
        if epoch%100==0 or epoch==args.epochs-1:
            model.eval();train=metrics(model,d);val=metrics(model,v)
            row=dict(epoch=epoch,loss=float(loss.detach()),train=train,validation=val);history.append(row)
            print(json.dumps(row),flush=True)
            if epoch>=700 and train['all_states_exact'] and val['all_states_exact'] and val['seating_maximum_error_m']<.0005 and val['maximum_positive_log_cost_error']<.035 and val['order_maximum_error']<.02:
                break
    model.eval();train=metrics(model,d);val=metrics(model,v)
    checkpoint=HERE/'best.pt'
    torch.save(dict(state_dict=model.cpu().state_dict(),manifest=manifest),checkpoint)
    save(HERE/'training.json',dict(complete=True,device=device,elapsed_s=time.monotonic()-start,
        epochs=epoch+1,train=train,validation=val,history=history,checkpoint_sha256=sha(checkpoint),
        provenance={str(SOURCE.relative_to(ROOT)):sha(SOURCE),str((DATA/'states.npz').relative_to(ROOT)):sha(DATA/'states.npz')},
        neural_network_trained=True,unseen_group_generalization=False,negative_volume_labels=0))
    if not train['all_states_exact']:raise RuntimeError('Network has not fit the demonstrated contact sets')


if __name__=='__main__':main()
