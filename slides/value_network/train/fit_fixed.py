"""Fit all recorded fixed states; do not describe this as held-out generalization."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from fixed_model import FixedValueNetwork
from fixed_data import collect
from train import save,digest

OUT=Path(__file__).parent/'fixed_output'

def evaluate(pred,logits,target,known,observed):
    dif=(pred-target)[known];valid=known.any(axis=1);regrets=[];hits=[];masked_hits=[];masked_regrets=[]
    for i in np.flatnonzero(valid):
        good=known[i];best=float(target[i,good].min());j=int(np.where(good,pred[i],np.inf).argmin())
        regret=float(target[i,j]-best);regrets.append(regret);hits.append(regret<=1e-5)
        eligible=observed[i]&(logits[i]>0)
        j=int(np.where(eligible,pred[i],np.inf).argmin()) if eligible.any() else -1
        masked_hits.append(j>=0 and known[i,j] and target[i,j]-best<=.01)
        masked_regrets.append(float(target[i,j]-best) if j>=0 and known[i,j] else 100.)
    return dict(mae=float(np.abs(dif).mean()),rmse=float(np.sqrt(np.mean(dif**2))),
        exact_top1_fraction=float(np.mean(hits)),mean_top1_regret=float(np.mean(regrets)),max_top1_regret=float(max(regrets)),
        covered_policy_within_001_fraction=float(np.mean(masked_hits)),covered_policy_mean_regret=float(np.mean(masked_regrets)),
        coverage_accuracy=float(((logits>0)==known)[observed].mean()),states_with_values=int(valid.sum()),finite_values=int(known.sum()))

def run(a):
    start=time.perf_counter();torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed);device='cuda' if torch.cuda.is_available() else 'cpu'
    poses,arrays,keys,records,sources=collect(a.extra,getattr(a,'data_config',None));out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    model=FixedValueNetwork(len(poses)*200,len(poses),a.width,a.depth).to(device)
    if a.resume:
        c=torch.load(a.resume,map_location=device,weights_only=True);model.load_state_dict(c['model'])
    t={k:torch.tensor(v,device=device) for k,v in arrays.items()};optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=0.)
    history=[];best=float('inf');best_epoch=0;n=len(keys);positive=t['known'].float()
    print('FIT',n,'unique fixed states;',int(t['known'].sum()),'finite labels; parameters',sum(p.numel() for p in model.parameters()),flush=True)
    for epoch in range(1,a.epochs+1):
        model.train();order=torch.randperm(n,device=device)
        for ix in order.split(a.batch_size):
            v,c=model(t['state'][ix],t['membership'][ix],t['counts'][ix]);known=t['known'][ix];observed=t['observed'][ix]
            loss=((v-t['target'][ix])**2)[known].mean() if known.any() else v.sum()*0
            bce=torch.nn.functional.binary_cross_entropy_with_logits(c[observed],positive[ix][observed])
            # Relative values within each state drive selecting a good next head.
            minimum=t['target'][ix].masked_fill(~known,float('inf')).min(dim=1).values
            nonempty=known.any(dim=1)
            optimal=known&(t['target'][ix]<=minimum[:,None]+1e-5)
            rank=-v/a.temperature
            denominator=torch.logsumexp(rank.masked_fill(~known,-1e9),dim=1)
            numerator=torch.logsumexp(rank.masked_fill(~optimal,-1e9),dim=1)
            ranking=(denominator-numerator)[nonempty].mean() if nonempty.any() else v.sum()*0
            total=loss+a.coverage_weight*bce+a.rank_weight*ranking
            optimizer.zero_grad(set_to_none=True);total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        if epoch%25==0 or epoch==a.epochs:
            model.eval()
            with torch.no_grad():v,c=model(t['state'],t['membership'],t['counts'])
            m=evaluate(v.cpu().numpy(),c.cpu().numpy(),arrays['target'],arrays['known'],arrays['observed'])
            score=m['mae']+m['mean_top1_regret']+2*(1-m['covered_policy_within_001_fraction'])
            history.append(dict(epoch=epoch,**m));print('EPOCH',epoch,m,flush=True)
            if score<best:
                best=score;best_epoch=epoch
                torch.save(dict(kind='fixed_full_state',model=model.state_dict(),poses=poses,candidates=len(poses)*200,width=a.width,depth=a.depth,epoch=epoch),out/'best.pt')
            if m['mae']<a.target_mae and m['covered_policy_within_001_fraction']>=.995 and m['max_top1_regret']<.01:break
    c=torch.load(out/'best.pt',map_location=device,weights_only=True);model.load_state_dict(c['model']);model.eval()
    with torch.no_grad():v,l=model(t['state'],t['membership'],t['counts'])
    final=evaluate(v.cpu().numpy(),l.cpu().numpy(),arrays['target'],arrays['known'],arrays['observed'])
    report=dict(complete=True,scope='fit only groups authorized by data_config; training fit metrics, not held-out evaluation' if getattr(a,'data_config',None) else 'fit all recorded fixed states, including formerly held-out states; this is memorization, not held-out evaluation',
        metrics=final,best_epoch=best_epoch,epochs_run=epoch,parameters=sum(p.numel() for p in model.parameters()),device=device,
        elapsed_s=time.perf_counter()-start,args=vars(a),source_hashes=sources,
        code={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('fixed_model.py'),Path(__file__).with_name('fixed_data.py')]})
    save(out/'report.json',report);save(out/'history.json',history)
    print('FINISHED',final,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(OUT));p.add_argument('--extra');p.add_argument('--data-config');p.add_argument('--resume');p.add_argument('--width',type=int,default=512);p.add_argument('--depth',type=int,default=2);p.add_argument('--epochs',type=int,default=2500);p.add_argument('--batch-size',type=int,default=256);p.add_argument('--lr',type=float,default=.001);p.add_argument('--rank-weight',type=float,default=.01);p.add_argument('--coverage-weight',type=float,default=.5);p.add_argument('--temperature',type=float,default=.03);p.add_argument('--target-mae',type=float,default=.02);p.add_argument('--seed',type=int,default=20261005);run(p.parse_args())
