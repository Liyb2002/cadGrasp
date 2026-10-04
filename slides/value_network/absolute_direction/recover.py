"""Recover bounded beam failures by varying pose order and translation scale."""
import argparse
import inspect
import itertools
import re
import shutil
import numpy as np
import experiment as E
from step5_evaluate.evaluate import evaluate

NORMAL=E.joint_layouts
# Identical geometric/force tests; broaden only the search budget, not legality.
source=inspect.getsource(NORMAL).replace('def joint_layouts(', 'def broader_layouts(').replace('[.02,.04,.08,.12,.18]','[.03,.06,.10,.20,.30]').replace('[:3]]','[:8]]')
exec(source,E.__dict__)
BROAD=E.__dict__['broader_layouts']

def recover_layouts(group,problems,pools,exit_vector):
    m=len(problems);orders=[list(reversed(range(m)))]+[[k]+[j for j in range(m) if j!=k] for k in range(m)]
    for order in orders:
        try:layouts=BROAD(group,[problems[k] for k in order],[pools[k] for k in order],exit_vector)
        except RuntimeError as error:
            print('RECOVERY ORDER FAILED',group.name,order,str(error),flush=True);continue
        remapped=[]
        for offsets,allowed in layouts:
            inverse=np.argsort(order);off=offsets[inverse];off-=off[0]
            remapped.append((off,[allowed[k] for k in inverse]))
        E.save(group/'recovery_policy.json',dict(pose_order=[problems[k].pose for k in order],translation_radii_m=[.03,.06,.10,.20,.30],beam_width=8,hard_constraints_unchanged=True))
        return remapped
    raise RuntimeError('All broadened order/translation beams failed')
E.joint_layouts=recover_layouts

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--groups',nargs='+',required=True);args=p.parse_args();E.install_recorded_recovery(E.OUT)
    lookup=dict(E.specs())
    for name in args.groups:
        group=E.OUT/name;comparison=group/'comparison.json'
        previous=E.json.loads(comparison.read_text()) if comparison.exists() else dict(passed=False)
        incumbent=group/'history/recovery_incumbent';saved=['step3','step4/data/growing_support','step4/data/source_inputs','step5_evaluate','comparison.json']
        if previous.get('passed'):
            if incumbent.exists():shutil.rmtree(incumbent)
            for relative in saved:
                src=group/relative;dst=incumbent/relative
                if src.is_dir():shutil.copytree(src,dst)
                elif src.exists():dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        original_candidates=E.candidates;banned=set();succeeded=False
        def filtered(problem,exit_vector):
            entries,path=original_candidates(problem,exit_vector)
            return [e for e in entries if e['contact']['candidate_id'] not in banned],path
        E.candidates=filtered
        for attempt in range(6):
            try:
                E.run(name,lookup[name]);path=group/'step4/data/growing_support/report.json';report=E.json.loads(path.read_text())
                report['provenance']['code'].update(E.I.hashes([E.Path(__file__)]));report['direction_objective']['recovery_policy']=dict(E.json.loads((group/'recovery_policy.json').read_text()),blocked_growth_contacts=sorted(banned));E.save(path,report)
                evaluate(group);succeeded=True;break
            except Exception as error:
                import traceback;traceback.print_exc()
                match=re.search(r'No legal original contact growth start: (pose_\d+):(pose_\d+_C\d+)',str(error))
                if match:
                    banned.add(match[2]);print('RETRY WITHOUT BLOCKED GROWTH CONTACT',name,match[2],flush=True);continue
                E.save(comparison,dict(complete=True,passed=False,error=str(error)));break
        E.candidates=original_candidates
        current=E.json.loads(comparison.read_text()) if comparison.exists() else dict(passed=False)
        if previous.get('passed') and (not succeeded or current.get('step5',{}).get('object_and_support_poses',{}).get('box_volume_cm3',float('inf'))>=previous['step5']['object_and_support_poses']['box_volume_cm3']):
            for relative in saved:
                src=incumbent/relative;dst=group/relative
                if src.is_dir():shutil.copytree(src,dst,dirs_exist_ok=True)
                elif src.exists():shutil.copy2(src,dst)
            print('KEPT BETTER INCUMBENT',name,flush=True)
