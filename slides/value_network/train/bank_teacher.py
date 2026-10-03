"""Offline bank labels for newly visited states; not a runtime action selector."""
import sys
from pathlib import Path
import json
import hashlib
import numpy as np
from train import DATA
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'data_producer'))
from prepare import prepare,compatible,I
from search import object_directions,TerminalOracle,install_recorded_recovery,J
from collect_states import joint_value

class Context:
    def __init__(self,output,mechanics=True):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.cfg=json.loads((DATA/'pilot20_shared/config.json').read_text());I.check_hashes(self.cfg['source_inputs'])
        self.poses=sorted({p for _,ps in self.cfg['groups'] for p in ps},key=lambda p:int(p.split('_')[1]))
        problems,pools,catalogues,_=prepare('B',self.poses,DATA/'pilot20_shared/inputs')
        self.problems=dict(zip(self.poses,problems));self.pools=dict(zip(self.poses,pools));self.vectors={p:object_directions(problem,c) for p,problem,c in zip(self.poses,problems,catalogues)}
        self.banks={p:[tuple(g) for g in json.loads((DATA/'pilot20_shared'/f'bank_{p}.json').read_text())['completions']] for p in self.poses}
        self.oracles={}
        if mechanics:
            install_recorded_recovery(self.output)
            for p in self.poses:
                cache=self.output/f'terminal_checks_{p}.json'
                if not cache.exists():cache.write_bytes((DATA/'pilot20_shared'/cache.name).read_bytes())
                self.oracles[p]=TerminalOracle(self.problems[p],self.pools[p],cache)
        self.angle_cache={}
    def angles(self,poses):
        key=tuple(poses)
        if key not in self.angle_cache:self.angle_cache[key]={(a,b):(np.arccos(np.clip(self.vectors[poses[a]]@self.vectors[poses[b]].T,-1,1))/np.pi)**2 for a in range(len(poses)) for b in range(a+1,len(poses))}
        return self.angle_cache[key]
    def check(self,p,g):
        if not g:
            oracle=self.oracles[p]
            if '' not in oracle.records:
                full=self.problems[p].supply([]);mask,info=J.classify(full,self.problems[p].targets)
                oracle.records['']=dict(passed=bool(mask.all()),indices=[],sample_count=len(mask),covered_count=int(mask.sum()),classifier=info,no_uplift_in_same_reaction_solve=True)
            return oracle.records['']['passed']
        return self.oracles[p].check(g)
    def terminal(self,poses,state):return all(self.check(p,g) for p,g in zip(poses,state))
    def terminal_cost(self,poses,state):
        result=joint_value([self.pools[p] for p in poses],[[tuple(g)] for g in state],self.angles(poses),sum(map(len,state))-1)
        return dict(total_heads=sum(map(len,state)),dispersion=result['dispersion'],cost=sum(map(len,state))+result['dispersion'],direction_ids=result['direction_ids'],object_exit_vectors=[self.vectors[p][d].tolist() for p,d in zip(poses,result['direction_ids'])])
    def legal(self,poses,state):
        return [(k,i) for k,p in enumerate(poses) for i,e in enumerate(self.pools[p]) if i not in state[k] and e['valid'] and compatible(self.pools[p],sorted(state[k]+[i]))]
    def labels(self,poses,state):
        pools=[self.pools[p] for p in poses];available=[[g for g in self.banks[p] if set(s).issubset(g)] for p,s in zip(poses,state)];rows=[]
        maps=[]
        for pool,groups in zip(pools,available):
            m={i:[] for i in range(len(pool))}
            for g in groups:
                for i in g:m[i].append(g)
            maps.append(m)
        for k,p in enumerate(poses):
            for i,e in enumerate(pools[k]):
                if i in state[k]:continue
                row=dict(pose=p,index=i,id=e['id'],value=None,status='step2_rejected')
                if e['valid']:
                    if not compatible(pools[k],sorted(state[k]+[i])):row['status']='no_path'
                    else:
                        choices=list(available);choices[k]=maps[k][i]
                        best=joint_value(pools,choices,self.angles(poses),sum(map(len,state)))
                        row['status']='success_found' if best else 'budget_unresolved'
                        if best:row.update(best)
                rows.append(row)
        return dict(poses=poses,selected_indices=state,rows=rows,label_scope='best found in the same recorded certified bank; unknown is not infeasible')
    def save_label(self,poses,state,folder):
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        key=hashlib.sha256(json.dumps([poses,state]).encode()).hexdigest();path=folder/f'{key}.json'
        if not path.exists():I.save(path,self.labels(poses,state))
        return path
    def save(self):
        for o in self.oracles.values():o.save()
