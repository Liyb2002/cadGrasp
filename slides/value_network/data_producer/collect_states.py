"""Fixed-group conditional Q pilot; shared certified local completion caches."""
import argparse
import itertools
import json
import time
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from prepare import ROOT, I, prepare, compatible, intersect
from search import TerminalOracle, seed_groups, make_bank, object_directions, install_recorded_recovery

OUT = ROOT/'slides/value_network/data/B'

def conditional(pool, oracle, required, bank, rng, attempts=8, cap=6):
    required=set(required)
    if not compatible(pool,sorted(required)):
        return []
    found={g for g in bank if required.issubset(g)}
    if found:return sorted(found)
    # Spend the proposal budget only when no eligible certified completion exists.
    for trial in range(attempts):
        if trial%4 != 3 and bank:
            proposed=sorted(required | set(sorted(bank)[int(rng.integers(len(bank)))]))
        else:
            proposed=sorted(required)
            while len(proposed)<cap:
                choices=[i for i,e in enumerate(pool) if e['valid'] and i not in proposed and compatible(pool,sorted(proposed+[i]))]
                if not choices:break
                proposed.append(int(rng.choice(choices)))
        proposed=sorted(proposed)
        if len(proposed)>cap or not compatible(pool,proposed) or not oracle.check(proposed):continue
        order=[i for i in proposed if i not in required];rng.shuffle(order)
        for i in order:
            reduced=[j for j in proposed if j!=i]
            if reduced and oracle.check(reduced):proposed=reduced
        found.add(tuple(proposed))
    bank.update(found)
    return sorted(found)


def options(pool, groups):
    # For a fixed exit direction, larger local completions are dominated.
    best={}
    for group in groups:
        for d in intersect(pool,group,'directions'):
            if d not in best or (len(group),group)<(len(best[d]),best[d]):best[d]=group
    return sorted(best),best


def joint_value(pools, groups, angles, selected_count, weight=1., starts=8):
    """Finite-menu coordinate search; exact for two poses, budgeted for >2."""
    opts=[options(p,g) for p,g in zip(pools,groups)]
    if any(not ds for ds,_ in opts):return None
    n=len(opts);pairs=n*(n-1)//2
    ids=[np.asarray(ds,int) for ds,_ in opts]
    sizes=[np.array([len(m[d]) for d in ds]) for ds,m in opts]
    def score(choice):
        count=sum(sizes[k][choice[k]] for k in range(n))
        dispersion=sum(angles[k,j][ids[k][choice[k]],ids[j][choice[j]]] for k in range(n) for j in range(k+1,n))/pairs
        return float(count+weight*dispersion),float(dispersion)
    best=None
    if n==2:
        cost=sizes[0][:,None]+sizes[1][None,:]+weight*angles[0,1][np.ix_(*ids)]
        choices=[list(np.unravel_index(int(np.argmin(cost)),cost.shape))]
    else:
        rng=np.random.default_rng(1729)
        choices=[]
        for trial in range(starts):
            choice=[int(np.argmin(s)) if trial==0 else int(rng.integers(len(s))) for s in sizes]
            for sweep in range(12):
                changed=False
                for k in range(n):
                    costs=sizes[k].astype(float).copy()
                    for j in range(n):
                        if j==k:continue
                        matrix=angles[min(k,j),max(k,j)]
                        costs+=weight/pairs*(matrix[ids[k],ids[j][choice[j]]] if k<j else matrix[ids[j][choice[j]],ids[k]])
                    new=int(np.argmin(costs));changed|=new!=choice[k];choice[k]=new
                if not changed:break
            choices.append(choice)
    for choice in choices:
        cost,d=score(choice)
        ds=[int(ids[k][choice[k]]) for k in range(n)]
        gs=[list(opts[k][1][ds[k]]) for k in range(n)]
        item=dict(value=cost-selected_count-1,remaining_heads=sum(map(len,gs))-selected_count-1,
                  total_heads=sum(map(len,gs)),dispersion=d,direction_ids=ds,completion_indices=gs)
        if best is None or (item['value'],ds)<(best['value'],best['direction_ids']):best=item
    return best


def sample_states(pools,banks,count,seed):
    rng=np.random.default_rng(seed);states=[tuple(() for _ in pools)];seen=set(states)
    # Diverse compatible prefixes, mostly from successes, with legal unverified prefixes.
    for attempt in range(10000):
        groups=[]
        for pool,bank in zip(pools,banks):
            if bank:
                groups.append(list(sorted(bank)[int(rng.integers(len(bank)))]))
            else:
                chosen=[]
                for _ in range(4):
                    legal=[i for i,e in enumerate(pool) if e['valid'] and i not in chosen and compatible(pool,sorted(chosen+[i]))]
                    if not legal:break
                    chosen.append(int(rng.choice(legal)))
                groups.append(chosen)
        actions=[(k,i) for k,g in enumerate(groups) for i in g];rng.shuffle(actions)
        depth=int(rng.integers(1,len(actions)))
        chosen=[[] for _ in pools]
        for k,i in actions[:depth]:chosen[k].append(i)
        if attempt%4==3:
            k=int(rng.integers(len(pools)))
            legal=[i for i,e in enumerate(pools[k]) if e['valid'] and i not in chosen[k] and compatible(pools[k],sorted(chosen[k]+[i]))]
            if legal:chosen[k].append(int(rng.choice(legal)))
        state=tuple(tuple(sorted(g)) for g in chosen)
        if state not in seen:states.append(state);seen.add(state)
        if len(states)==count:return states
    raise RuntimeError('Could not sample enough distinct states')


def run(args):
    start=time.monotonic();shared=OUT/'pilot20_shared';shared.mkdir(exist_ok=True)
    cache=shared if args.group is None else shared/'groups'/args.group
    cache.mkdir(parents=True,exist_ok=True)
    install_recorded_recovery(cache)
    manifest=json.loads((ROOT/'slides/baseline_algo/output/B/pose1+3/step5_evaluate/all_groups.json').read_text())
    all_specs=[(g['group'],g['poses']) for g in manifest['groups']]
    specs=all_specs if args.group is None else [s for s in all_specs if s[0]==args.group]
    if not specs:raise ValueError('Unknown group')
    if args.group:
        for _,ps in specs:
            for pose in ps:
                for prefix in ['bank_', 'terminal_checks_']:
                    src=shared/f'{prefix}{pose}.json';dst=cache/src.name
                    if src.exists() and not dst.exists():shutil.copy2(src,dst)
    poses=sorted({p for _,ps in specs for p in ps},key=lambda p:int(p.split('_')[1]))
    problems,pools,catalogues,inputs=prepare('B',poses,shared/'inputs')
    cfg=dict(states_per_group=args.states,attempts_per_conditional=args.attempts,max_heads_per_pose=6,seed=args.seed,
             source_inputs=inputs,input_policy='current independent_poses, same as accepted pose1+3 value experiment',
             continuation_policy='reuse eligible certified completions; otherwise at most eight proposals',
             groups=specs,worker_group=args.group,workers=args.workers,lambda_direction=1.,dispersion='mean squared normalized pairwise exit angle in object frame',
             direction_search='exact finite menu for 2 poses; 8 deterministic multistart coordinate searches for >2',
             label_scope='best found verified completion; null unknown is not infeasible',
             code=I.hashes([Path(__file__),Path(__file__).with_name('prepare.py'),Path(__file__).with_name('search.py')]))
    cf=cache/'config.json'
    if cf.exists() and json.loads(cf.read_text())!=json.loads(json.dumps(cfg)):raise ValueError('Pilot configuration changed')
    I.save(cf,cfg)
    oracles=[];banks=[]
    old=OUT/'pose1+3'
    for k,(p,pool) in enumerate(zip(problems,pools)):
        path=cache/f'terminal_checks_{p.pose}.json'
        legacy=old/f'terminal_checks_{p.pose}.json'
        if not path.exists() and legacy.exists():path.write_bytes(legacy.read_bytes())
        oracle=TerminalOracle(p,pool,path);oracles.append(oracle)
        bp=cache/f'bank_{p.pose}.json'
        if bp.exists():bank={tuple(g) for g in json.loads(bp.read_text())['completions']}
        else:
            try:
                seeds=seed_groups('B',p.pose,pool,oracle)
            except RuntimeError:
                seeds=[]
            if seeds:
                bank=set(make_bank(pool,oracle,seeds,16,np.random.default_rng(args.seed+k),6,bp))
            else:
                bank=set();rng=np.random.default_rng(args.seed+k)
                root=I.OUTPUTS/'B'/'independent_poses'/p.pose/'step3_scheculer'
                by_id={e['id']:i for i,e in enumerate(pool)}
                for schedule in sorted(root.glob('particle_*/schedule.json')):
                    chosen=[by_id[c] for c in json.loads(schedule.read_text())['selected_ids']]
                    if compatible(pool,sorted(chosen)):
                        conditional(pool,oracle,chosen,bank,rng,32)
                for _ in range(64):conditional(pool,oracle,[],bank,rng,1)
                print('FALLBACK BANK',p.pose,len(bank),'certified sets',flush=True)
            if p.pose in ['pose_1','pose_3']:
                data=json.loads((old/'completion_banks.json').read_text());bank.update(tuple(g) for g in data['groups'][data['poses'].index(p.pose)])
        banks.append(bank);oracle.save();I.save(bp,dict(completions=[list(g) for g in sorted(bank)]))
    if args.group is None:
        def launch(spec):
            name,_=spec
            with (shared/f'{name}.log').open('w') as log:
                subprocess.run([str(ROOT/'.venv/bin/python'),str(Path(__file__)), '--states',str(args.states),'--attempts',str(args.attempts),'--seed',str(args.seed),'--workers',str(args.workers),'--group',name],stdout=log,stderr=subprocess.STDOUT,check=True)
            return json.loads((OUT/name/'pilot20/summary.json').read_text())
        summaries=[]
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            jobs={executor.submit(launch,s):s[0] for s in specs}
            for future in as_completed(jobs):
                summaries.append(future.result())
                I.save(shared/'progress.json',dict(complete=False,groups=summaries,elapsed_s=time.monotonic()-start))
                print('GROUP COMPLETE',jobs[future],len(summaries),'/',len(specs),flush=True)
        # Merge worker certificates and banks after workers stop writing.
        for pose in poses:
            merged=dict(oracles[poses.index(pose)].records);bank=set(banks[poses.index(pose)])
            for name,ps in specs:
                if pose not in ps:continue
                worker=shared/'groups'/name
                merged.update(json.loads((worker/f'terminal_checks_{pose}.json').read_text()))
                bank.update(tuple(g) for g in json.loads((worker/f'bank_{pose}.json').read_text())['completions'])
            I.save(shared/f'terminal_checks_{pose}.json',merged)
            I.save(shared/f'bank_{pose}.json',dict(completions=[list(g) for g in sorted(bank)]))
        I.check_hashes(inputs)
        summaries.sort(key=lambda g:[s[0] for s in specs].index(g['group']))
        I.save(shared/'summary.json',dict(complete=True,groups=summaries,total_states=sum(g['states'] for g in summaries),total_actions=sum(g['actions'] for g in summaries),elapsed_s=time.monotonic()-start,source_inputs_unchanged=True))
        print('COMPLETE',shared/'summary.json',flush=True)
        return
    index={p:i for i,p in enumerate(poses)};summaries=[]
    for group_name,ps in specs:
        group_index=[s[0] for s in all_specs].index(group_name)
        gs=time.monotonic();ks=[index[p] for p in ps];local_pools=[pools[k] for k in ks]
        folder=OUT/group_name/'pilot20';folder.mkdir(parents=True,exist_ok=True)
        states_path=folder/'states.json'
        if states_path.exists():states=[tuple(tuple(g) for g in s) for s in json.loads(states_path.read_text())['states']]
        else:
            states=sample_states(local_pools,[banks[k] for k in ks],args.states,args.seed+group_index)
            I.save(states_path,dict(poses=ps,states=states))
        vectors=[object_directions(problems[k],catalogues[k]) for k in ks]
        angles={(a,b):(np.arccos(np.clip(vectors[a]@vectors[b].T,-1,1))/np.pi)**2 for a in range(len(ks)) for b in range(a+1,len(ks))}
        stats=[]
        for si,state in enumerate(states):
            sp=folder/f'state_{si:03d}.json'
            if sp.exists():stats.append(json.loads(sp.read_text())['summary']);continue
            ss=time.monotonic();rng=np.random.default_rng(args.seed+si+1000*group_index)
            local=[conditional(local_pools[t],oracles[k],state[t],banks[k],rng,args.attempts) for t,k in enumerate(ks)]
            action_sets=[]
            for t,k in enumerate(ks):
                candidates=[]
                for i,e in enumerate(local_pools[t]):
                    if i in state[t]:continue
                    row=dict(pose=ps[t],index=i,id=e['id'],value=None)
                    if not e['valid']:row['status']='step2_rejected';found=[]
                    elif not compatible(local_pools[t],sorted(set(state[t])|{i})):row['status']='no_path';found=[]
                    else:
                        found=conditional(local_pools[t],oracles[k],set(state[t])|{i},banks[k],rng,args.attempts)
                        row['status']='pending';local[t]=sorted(set(local[t])|set(found))
                    candidates.append((row,found))
                action_sets.append(candidates)
            rows=[];selected_count=sum(map(len,state))
            for t,candidates in enumerate(action_sets):
                for row,found in candidates:
                    if row['status']=='pending':
                        choices=list(local);choices[t]=found
                        best=joint_value(local_pools,choices,angles,selected_count)
                        row['status']='success_found' if best else 'budget_unresolved'
                        if best:
                            row.update(best)
                            row['completion_ids']=[[local_pools[j][i]['id'] for i in g] for j,g in enumerate(best['completion_indices'])]
                            row['object_exit_vectors']=[vectors[j][d].tolist() for j,d in enumerate(best['direction_ids'])]
                            # Every claimed completion contains state and action and passed all original loads.
                            for j,g in enumerate(best['completion_indices']):
                                assert set(state[j]).issubset(g) and compatible(local_pools[j],g)
                                assert oracles[ks[j]].check(g) is True
                            assert row['index'] in best['completion_indices'][t] and row['remaining_heads']>=0
                    rows.append(row)
            summary=dict(state_id=si,actions=len(rows),success=sum(r['value'] is not None for r in rows),
                         unknown=sum(r['status']=='budget_unresolved' for r in rows),
                         no_path=sum(r['status']=='no_path' for r in rows),rejected=sum(r['status']=='step2_rejected' for r in rows),elapsed_s=time.monotonic()-ss)
            I.save(sp,dict(complete=True,state_id=si,poses=ps,selected_indices=state,
                          selected_ids=[[local_pools[t][i]['id'] for i in g] for t,g in enumerate(state)],summary=summary,rows=rows))
            stats.append(summary)
            for k in ks:
                oracles[k].save();I.save(cache/f'bank_{poses[k]}.json',dict(completions=[list(g) for g in sorted(banks[k])]))
            print('STATE',group_name,si+1,'/',args.states,summary,flush=True)
        with (folder/'training_records.jsonl').open('w') as f:
            for si in range(len(states)):
                data=json.loads((folder/f'state_{si:03d}.json').read_text())
                for row in data['rows']:f.write(json.dumps(dict(group=group_name,state_id=si,poses=ps,selected_indices=data['selected_indices'],**row))+'\n')
        summary=dict(group=group_name,poses=ps,states=len(stats),actions=sum(s['actions'] for s in stats),success=sum(s['success'] for s in stats),unknown=sum(s['unknown'] for s in stats),elapsed_s=time.monotonic()-gs,state_compute_s=sum(s['elapsed_s'] for s in stats))
        I.save(folder/'summary.json',summary);summaries.append(summary)
        I.save(cache/'progress.json',dict(complete=False,groups=summaries,elapsed_s=time.monotonic()-start))
    I.check_hashes(inputs)
    I.save(cache/'summary.json',dict(complete=True,groups=summaries,total_states=sum(g['states'] for g in summaries),total_actions=sum(g['actions'] for g in summaries),elapsed_s=time.monotonic()-start,source_inputs_unchanged=True))
    print('COMPLETE',cache/'summary.json',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--states',type=int,default=20);p.add_argument('--attempts',type=int,default=8);p.add_argument('--seed',type=int,default=20261003);p.add_argument('--group');p.add_argument('--workers',type=int,default=4);run(p.parse_args())
