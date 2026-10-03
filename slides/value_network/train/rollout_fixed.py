"""Network-only greedy actions with baseline hard masks and terminal acceptance."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from fixed_model import FixedValueNetwork
from fixed_data import features
from bank_teacher import Context
from train import DATA,save,digest

def run(a):
    torch.set_num_threads(4);start=time.perf_counter();checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    model=FixedValueNetwork(checkpoint['candidates'],len(checkpoint['poses']),checkpoint['width'],checkpoint['depth']);model.load_state_dict(checkpoint['model']);model.eval()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);ctx=Context(getattr(a, 'checks', Path(__file__).parent/'policy_checks'));results=[]
    groups=getattr(a,'groups',None) or ctx.cfg['groups']
    for name,poses in groups:
        state=[[] for _ in poses];trace=[];passed=False;coverage_fallbacks=0
        root_path=Path(getattr(a,'state_root',DATA))/name/'pilot20/state_000.json'
        root=json.loads(root_path.read_text()) if root_path.exists() else None
        reference=min((r['value']+1 for r in root['rows'] if r['value'] is not None),default=None) if root else None
        for step in range(6*len(poses)+1):
            passed=ctx.terminal(poses,state)
            if passed:break
            if step==6*len(poses):break
            if a.collect:ctx.save_label(poses,state,a.collect)
            legal=ctx.legal(poses,state)
            if not legal:break
            x,m,c=features(poses,state,checkpoint['poses'])
            with torch.no_grad():values,logits=model(torch.tensor(x)[None],torch.tensor(m)[None],torch.tensor(c)[None])
            values=values[0].numpy();logits=logits[0].numpy();pindex={p:k for k,p in enumerate(checkpoint['poses'])}
            candidates=[(k,i,pindex[poses[k]]*200+i) for k,i in legal];covered=[t for t in candidates if logits[t[2]]>=0]
            fallback=not covered
            if fallback:
                coverage_fallbacks+=1;chosen=max(candidates,key=lambda t:float(logits[t[2]]))
            else:chosen=min(covered,key=lambda t:(float(values[t[2]]),poses[t[0]],t[1]))
            k,i,j=chosen
            trace.append(dict(state=[list(g) for g in state],action=dict(pose=poses[k],index=i,id=ctx.pools[poses[k]][i]['id']),predicted_value=float(values[j]),predicted_bank_coverage=float(torch.sigmoid(torch.tensor(logits[j]))),coverage_fallback=fallback))
            state[k]=sorted(state[k]+[i])
        result=dict(group=name,poses=poses,passed=bool(passed),selected_indices=state,selected_ids=[[ctx.pools[p][i]['id'] for i in g] for p,g in zip(poses,state)],trace=trace,reference_cost=reference,coverage_fallbacks=coverage_fallbacks)
        if passed:
            result.update(ctx.terminal_cost(poses,state));result['cost_gap']=result['cost']-reference if reference is not None else None
            result['load_counts']=[len(ctx.problems[p].targets) for p in poses]
        ctx.save();save(out/f'{name}.json',result);results.append(result)
        print('ROLLOUT',name,'PASS',passed,'HEADS',sum(map(len,state)),'GAP',result.get('cost_gap'),flush=True)
    report=dict(complete=True,passed_groups=sum(r['passed'] for r in results),groups=len(results),results=results,
        elapsed_s=time.perf_counter()-start,action_selection='network values and learned bank-coverage logits only; exact candidate/no-repeat/path masks; no runtime teacher/bank action filtering',
        terminal_acceptance='baseline joint equilibrium and no-uplift on all 32768 original loads in each pose; legal common directions/components; no Step4 solid constructed',
        checkpoint_sha256=digest(Path(a.checkpoint)),code={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('fixed_model.py'),Path(__file__).with_name('bank_teacher.py')]})
    save(out/'report.json',report);print('COMPLETE',report['passed_groups'],'/',report['groups'],flush=True);return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default=str(Path(__file__).parent/'fixed_output/best.pt'));p.add_argument('--output',default=str(Path(__file__).parent/'fixed_rollout'));p.add_argument('--collect');run(p.parse_args())
