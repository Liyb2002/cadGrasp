"""Score every currently legal unselected head in a saved state."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from model import ValueNetwork
from fixed_model import FixedValueNetwork
from train import DATA

def predict(state_path,checkpoint_path):
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    poses=checkpoint['poses'];pindex={p:k for k,p in enumerate(poses)}
    data=json.loads(Path(state_path).read_text());member_poses=data['poses'];selected=data['selected_indices']
    state=torch.zeros(1,checkpoint['candidates']);membership=torch.zeros(1,len(poses));counts=torch.zeros(1,len(poses))
    pools={p:json.loads((DATA/'pilot20_shared/inputs'/f'paths_{p}.json').read_text())['candidates'] for p in member_poses}
    for p,g in zip(member_poses,selected):
        k=pindex[p];membership[0,k]=1;counts[0,k]=len(g)
        for i in g:state[0,k*200+i]=1
    fixed=checkpoint.get('kind')=='fixed_full_state'
    model=FixedValueNetwork(checkpoint['candidates'],len(poses),checkpoint['width'],checkpoint['depth']) if fixed else ValueNetwork(checkpoint['candidates'],len(poses))
    model.load_state_dict(checkpoint['model']);model.eval()
    rows=[]
    for p,g in zip(member_poses,selected):
        pool=pools[p]
        for i,e in enumerate(pool):
            if i in g or not e['valid']:continue
            trial=list(g)+[i]
            directions=set(pool[trial[0]]['directions']);components=set(pool[trial[0]]['components'])
            for j in trial[1:]:directions.intersection_update(pool[j]['directions']);components.intersection_update(pool[j]['components'])
            if not directions or not components:continue
            rows.append(dict(id=e['id'],pose=p,index=i,action=pindex[p]*200+i))
    if rows:
        with torch.no_grad():
            if fixed:
                values,logits=model(state,membership,counts)
                actions=[r['action'] for r in rows]
                scores=values[0,actions].numpy()
                coverage=torch.sigmoid(logits[0,actions]).numpy()
                for r,prob in zip(rows,coverage):r['predicted_bank_coverage']=float(prob);r['policy_eligible']=bool(prob>=.5)
            else:
                scores=model(state.expand(len(rows),-1),membership.expand(len(rows),-1),counts.expand(len(rows),-1),torch.tensor([r['action'] for r in rows])).numpy()
        for r,v in zip(rows,scores):r['predicted_value']=float(v);r.pop('action')
    return sorted(rows,key=lambda r:(not r.get('policy_eligible',True),r['predicted_value'] if r.get('policy_eligible',True) else -r['predicted_bank_coverage'],r['pose'],r['index']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--state',type=Path,required=True);p.add_argument('--checkpoint',type=Path,default=Path(__file__).parent/'current/best.pt');p.add_argument('--top',type=int,default=20);p.add_argument('--output',type=Path);a=p.parse_args()
    rows=predict(a.state,a.checkpoint)
    if a.output:a.output.write_text(json.dumps(dict(state=str(a.state),scores=rows),indent=2)+'\n')
    print(json.dumps(rows[:a.top],indent=2))
