"""Train on certified labels; split whole states, including cross-group duplicates."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import torch
from model import ValueNetwork

ROOT=Path(__file__).resolve().parents[3]
DATA=ROOT/'slides/value_network/data/B'

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def load_data(seed):
    cfg=json.loads((DATA/'pilot20_shared/config.json').read_text())
    audit=json.loads((DATA/'pilot20_shared/audit.json').read_text());assert audit['passed']
    poses=sorted({p for _,ps in cfg['groups'] for p in ps},key=lambda p:int(p.split('_')[1]))
    pose_index={p:k for k,p in enumerate(poses)};width=len(poses)*200
    signatures={};states=[];members=[];counts=[];labels=[];actions=[];state_ids=[];groups=[];sources={}
    # Identical pose-set/selected-head states share a signature and hence one split.
    assignments={};rng=np.random.default_rng(seed)
    for name,ps in cfg['groups']:
        local=[]
        for sid in range(cfg['states_per_group']):
            path=DATA/name/'pilot20'/f'state_{sid:03d}.json';d=json.loads(path.read_text());sources[str(path.relative_to(ROOT))]=digest(path)
            signature=tuple((p,tuple(g)) for p,g in zip(ps,d['selected_indices']))
            if signature not in signatures:
                signatures[signature]=len(states)
                mask=np.zeros(width,np.float32);member=np.zeros(len(poses),np.float32);count=np.zeros(len(poses),np.float32)
                for p,g in signature:
                    k=pose_index[p];mask[k*200+np.asarray(g,dtype=int)]=1;member[k]=1;count[k]=len(g)
                states.append(mask);members.append(member);counts.append(count)
            uid=signatures[signature];local.append((uid,d))
        fresh=list(dict.fromkeys(uid for uid,_ in local if uid not in assignments));rng.shuffle(fresh)
        # Four validation and four test states per group, shared states stay assigned.
        for j,uid in enumerate(fresh):assignments[uid]='validation' if j<4 else 'test' if j<8 else 'train'
        for uid,d in local:
            for row in d['rows']:
                if row['value'] is None:continue
                assert row['status']=='success_found'
                labels.append(row['value']);actions.append(pose_index[row['pose']]*200+row['index']);state_ids.append(uid);groups.append(name)
    return dict(poses=poses,states=np.array(states),membership=np.array(members),counts=np.array(counts),
                labels=np.array(labels,np.float32),actions=np.array(actions,np.int64),state_ids=np.array(state_ids,np.int64),groups=groups,
                splits=np.array([assignments[i] for i in state_ids]),assignments=assignments,sources=sources)

def metrics(pred,target,state_ids):
    regrets=[];hits=[]
    for sid in np.unique(state_ids):
        mask=state_ids==sid;p=pred[mask];y=target[mask];best=y.min();choice=int(p.argmin())
        regrets.append(float(y[choice]-best));hits.append(bool(y[choice]<=best+1e-6))
    return dict(mae=float(np.abs(pred-target).mean()),rmse=float(np.sqrt(np.mean((pred-target)**2))),
                mean_top1_regret=float(np.mean(regrets)),top1_optimal_fraction=float(np.mean(hits)),
                states=len(regrets),records=len(target))

def run(args):
    start=time.perf_counter();torch.manual_seed(args.seed);np.random.seed(args.seed);torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');data=load_data(args.seed)
    out=Path(__file__).parent/'output';out.mkdir(exist_ok=True)
    model=ValueNetwork(data['states'].shape[1],len(data['poses'])).to(device)
    tensors={k:torch.tensor(data[k],device=device) for k in ['states','membership','counts','labels','actions','state_ids']}
    indices={s:torch.tensor(np.flatnonzero(data['splits']==s),device=device) for s in ['train','validation','test']}
    assert all(len(i) for i in indices.values())
    split_states={s:set(data['state_ids'][data['splits']==s].tolist()) for s in indices}
    assert not(split_states['train']&split_states['validation'] or split_states['train']&split_states['test'] or split_states['validation']&split_states['test'])
    def predict(idx):
        uid=tensors['state_ids'][idx]
        return model(tensors['states'][uid],tensors['membership'][uid],tensors['counts'][uid],tensors['actions'][idx])
    training=data['splits']=='train'
    prior_design=np.c_[data['membership'][data['state_ids']],data['counts'][data['state_ids']]]
    prior_coefficients=np.linalg.lstsq(prior_design[training],data['labels'][training],rcond=None)[0]
    with torch.no_grad():model.base.weight.copy_(torch.tensor(prior_coefficients,dtype=torch.float32,device=device)[None])
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=1e-4)
    partner_lists={int(sid):torch.tensor(np.flatnonzero((data['state_ids']==sid)&training),device=device) for sid in np.unique(data['state_ids'][training])}
    partner_table=torch.full((len(data['states']),max(len(v) for v in partner_lists.values())),0,dtype=torch.long,device=device)
    partner_lengths=torch.ones(len(data['states']),dtype=torch.long,device=device)
    for sid,items in partner_lists.items():partner_table[sid,:len(items)]=items;partner_lengths[sid]=len(items)
    best=float('inf');best_epoch=0;history=[]
    print('DEVICE',device,torch.cuda.get_device_name() if device.type=='cuda' else '', 'PARAMETERS',sum(p.numel() for p in model.parameters()),flush=True)
    for epoch in range(1,args.epochs+1):
        model.train();order=indices['train'][torch.randperm(len(indices['train']),device=device)];loss_sum=0.
        for batch in order.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True);pred=predict(batch)
            loss=torch.nn.functional.smooth_l1_loss(pred,tensors['labels'][batch])
            uid=tensors['state_ids'][batch]
            partner=partner_table[uid,(torch.rand(len(batch),device=device)*partner_lengths[uid]).long()]
            pair_loss=torch.nn.functional.smooth_l1_loss(pred-predict(partner),tensors['labels'][batch]-tensors['labels'][partner])
            loss=loss+args.ranking_weight*pair_loss
            loss.backward();optimizer.step();loss_sum+=float(loss.detach())*len(batch)
        model.eval()
        with torch.no_grad():vpred=predict(indices['validation']);vmae=float((vpred-tensors['labels'][indices['validation']]).abs().mean())
        vm=metrics(vpred.cpu().numpy(),data['labels'][data['splits']=='validation'],data['state_ids'][data['splits']=='validation'])
        score=vmae+vm['mean_top1_regret']
        history.append(dict(epoch=epoch,train_huber=loss_sum/len(order),validation_mae=vmae,validation_regret=vm['mean_top1_regret'],selection_score=score))
        if score<best-1e-5:
            best=score;best_epoch=epoch
            torch.save(dict(model=model.state_dict(),poses=data['poses'],candidates=data['states'].shape[1],epoch=epoch,seed=args.seed),out/'best.pt')
        if epoch%10==0:print('EPOCH',epoch,'validation MAE',round(vmae,4),'best',round(best,4),flush=True)
        if epoch-best_epoch>=args.patience:break
    checkpoint=torch.load(out/'best.pt',map_location=device,weights_only=True);model.load_state_dict(checkpoint['model']);model.eval()
    with torch.no_grad():prediction=predict(torch.arange(len(data['labels']),device=device)).cpu().numpy()
    # Baselines are fitted to training labels only, with no use of held-out states.
    train=data['splits']=='train';group_patterns,gi=np.unique(data['membership'][data['state_ids']],axis=0,return_inverse=True);group_names=list(range(len(group_patterns)))
    static={};group_mean={g:float(data['labels'][train&(gi==g)].mean()) for g in range(len(group_names))}
    for g,a,y in zip(gi[train],data['actions'][train],data['labels'][train]):static.setdefault((int(g),int(a)),[]).append(float(y))
    fixed=np.array([np.mean(static[g,a]) if (g,a) in static else group_mean[g] for g,a in zip(gi,data['actions'])])
    design=np.c_[np.eye(len(group_names))[gi],data['counts'][data['state_ids']].sum(axis=1)]
    coefficients=np.linalg.lstsq(design[train],data['labels'][train],rcond=None)[0];count_baseline=design@coefficients
    result={}
    for split in indices:
        mask=data['splits']==split
        result[split]=dict(network=metrics(prediction[mask],data['labels'][mask],data['state_ids'][mask]),
                           fixed_head_value=metrics(fixed[mask],data['labels'][mask],data['state_ids'][mask]),
                           group_and_head_count=metrics(count_baseline[mask],data['labels'][mask],data['state_ids'][mask]))
    np.savez_compressed(out/'predictions.npz',prediction=prediction,target=data['labels'],state_ids=data['state_ids'],actions=data['actions'],splits=data['splits'])
    report=dict(complete=True,device=str(device),gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,
                parameters=sum(p.numel() for p in model.parameters()),best_epoch=best_epoch,epochs_run=epoch,
                elapsed_s=time.perf_counter()-start,metrics=result,args=vars(args),
                selection_policy='minimum validation MAE plus validation mean top1 regret; test not used for checkpoint selection',
                split_policy='whole unique pose-set/selected-head states; identical states across repeated groups cannot cross splits',
                ranking_scope='only candidates with finite verified dataset labels; unknown actions excluded, no online mechanical rollout',
                sources=data['sources'],code={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('model.py')]})
    save(out/'report.json',report);save(out/'history.json',history);save(out/'split.json',dict(poses=data['poses'],assignments=data['assignments']))
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(11,4));ax[0].plot([h['epoch'] for h in history],[h['validation_mae'] for h in history]);ax[0].set(xlabel='Epoch',ylabel='Validation MAE',title='Whole-state holdout')
    mask=data['splits']=='test';ax[1].scatter(data['labels'][mask],prediction[mask],s=3,alpha=.2);lim=float(max(prediction[mask].max(),data['labels'][mask].max()));ax[1].plot([0,lim],[0,lim],'k--');ax[1].set(xlabel='Verified search label',ylabel='Network prediction',title='Test states');fig.tight_layout();fig.savefig(out/'fit.png',dpi=180);plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k in ['elapsed_s','best_epoch','epochs_run','metrics']},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=400);p.add_argument('--patience',type=int,default=50);p.add_argument('--batch-size',type=int,default=512);p.add_argument('--lr',type=float,default=.001);p.add_argument('--seed',type=int,default=20261004);p.add_argument('--ranking-weight',type=float,default=1.);run(p.parse_args())
