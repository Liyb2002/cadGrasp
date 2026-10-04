"""Neural Step3, fresh actual Step4, unchanged Step5; no teacher-plan retrieval."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
from model import AbsoluteValueNetwork

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'absolute_direction'))
import experiment as E
import raw_sweep_carving as C
import refined_recovery as P
from step5_evaluate.evaluate import evaluate
_render_spec=importlib.util.spec_from_file_location('absolute_volume_render',E.HERE/'render.py')
V=importlib.util.module_from_spec(_render_spec)
_render_spec.loader.exec_module(V)

OUT=HERE/'output/B'


class Policy:
    def __init__(self):
        torch.set_num_threads(4)
        checkpoint=torch.load(HERE/'best.pt',map_location='cpu',weights_only=False)
        self.manifest=checkpoint['manifest'];m=self.manifest
        self.model=AbsoluteValueNetwork(len(m['candidates']),len(m['poses']),m['width'])
        self.model.load_state_dict(checkpoint['state_dict']);self.model.eval()
        self.index={ident:i for i,ident in enumerate(m['candidates'])}
        self.pindex={p:i for i,p in enumerate(m['poses'])}
        training=json.loads((HERE/'training.json').read_text())
        self.tie_band=2*training['validation']['maximum_positive_log_cost_error']

    def predict(self,poses,allowed):
        state=torch.zeros(1,len(self.index));membership=torch.zeros(1,len(self.pindex))
        for pose in poses:membership[0,self.pindex[pose]]=1
        with torch.no_grad():_,_,_,xy=self.model(state,membership)
        # A global, documented 5 mm seating grid, not a retrieved layout.
        grid=self.manifest['seating_grid_m']
        offsets=np.array([[*(xy.reshape(-1,2)[self.pindex[p]].numpy()*.2),0.] for p in poses])
        offsets=np.round(offsets/grid)*grid;offsets-=offsets[0]
        chosen={p:[] for p in poses};trace=[]
        candidate_ids=[ident for ident in self.manifest['candidates'] if any(ident.startswith(p+'_') for p in poses)]
        # Predict first, then test Step2/world geometry. Unknown actions are not
        # assigned fabricated failed-volume targets or silently replaced.
        with torch.no_grad():e,_,_,_=self.model(state,membership)
        predicted=[ident for ident in candidate_ids if e[0,self.index[ident]]>0]
        invalid=set(predicted)-set(allowed)
        if invalid:raise RuntimeError('Predicted contacts fail the geometric action mask: '+','.join(sorted(invalid)))
        for step in range(len(candidate_ids)+1):
            with torch.no_grad():e,q,order,_=self.model(state,membership)
            ids=[ident for ident in candidate_ids if ident in allowed and state[0,self.index[ident]]==0 and e[0,self.index[ident]]>0]
            if not ids:break
            lower=min(float(q[0,self.index[ident]]) for ident in ids)
            tied=[ident for ident in ids if float(q[0,self.index[ident]])<=lower+self.tie_band]
            ident=min(tied,key=lambda ident:(float(order[0,self.index[ident]]),ident))
            i=self.index[ident];pose=ident.split('_C')[0]
            chosen[pose].append(allowed[ident]);state[0,i]=1
            trace.append(dict(step=step,candidate=ident,log_cost=float(q[0,i]),
                predicted_cost=float(np.expm1(max(0,float(q[0,i])))),completion_evidence_probability=float(e[0,i].sigmoid()),
                tie_order=float(order[0,i]),tie_band_log_cost=self.tie_band))
        else:raise RuntimeError('Neural policy failed to terminate')
        if any(not chosen[p] for p in poses):raise RuntimeError('Neural policy produced an empty pose contact set')
        return [chosen[p] for p in poses],offsets,trace


def run(policy,name,poses):
    started=time.monotonic();group=OUT/name;group.mkdir(parents=True,exist_ok=True)
    problems=E.read_problems(name,poses);paths=[];allowed={}
    exit_vector=np.array([0.,0.,1.])
    for problem in problems:
        pool,source=E.candidates(problem,exit_vector)
        allowed.update({entry['contact']['candidate_id']:entry for entry in pool})
        paths.extend([source,source.with_suffix('.json')]+list(problem.inputs))
    rows,offsets,trace=policy.predict(poses,allowed)
    certificates=[]
    for problem,entries in zip(problems,rows):
        common=set(entries[0]['components'])
        for entry in entries:common&=set(entry['components'])
        if not common:raise RuntimeError('Neural contacts have no shared insertion component')
        cache={}
        if not E.verify_heads(problem,entries,cache):raise RuntimeError('Neural contact set fails original loads/no-uplift')
        certificate=next(iter(cache.values()));certificates.append(certificate)
        E.save(group/'step3'/f'forces_{problem.pose}.json',cache)
    placement=dict(bases=np.repeat(np.eye(3)[None],len(poses),axis=0).tolist(),offsets=offsets.tolist(),
        directions=np.repeat((-exit_vector)[None],len(poses),axis=0).tolist(),object_exit_mode=True,
        absolute_object_exit=np.repeat(exit_vector[None],len(poses),axis=0).tolist())
    contacts=[[e['contact'] for e in row] for row in rows];cells=[[e['cells'] for e in row] for row in rows]
    plan=dict(complete=True,object='B',poses=poses,head_count_optimized=False,head_count_penalty=0,
        selected_ids=[[c['candidate_id'] for c in row] for row in contacts],force_certificates=certificates,
        absolute_object_exit=exit_vector.tolist(),relative_support_direction=(-exit_vector).tolist(),
        coordinate_frame='original workstation world axes',object_withdrawal=True,placement=placement,
        selection='trained state/action volume network; witnessed-completion evidence gate',
        teacher_plan_loaded_at_inference=False,placement_predicted_by_network=True,
        neural_action_trace=trace,checkpoint=E.I.hashes([HERE/'best.pt']))
    plan_path=group/'step3/plan.json';E.save(plan_path,plan);paths.append(plan_path)
    case=SimpleNamespace(name='B',poses=poses,pair=group,output=group/'step4/data/growing_support',tasks=problems,
        groups=contacts,heads=cells,support_seeds=cells,
        root_solids=[E.F.union([E.S.solid(E.G.hull_mesh(v)) for e in row for v in e['cells']]) for row in rows],
        demands=[E.pressure_centers(p.targets/p.scale,p.domain.com)[0] for p in problems],paths=paths,
        object_exit_mode=True,schedule=dict(passed=True,covered_counts=[c['covered_count'] for c in certificates],head_model=E.Z.MODEL))
    for p in problems:
        original=next(Path(path) for path in p.inputs if Path(path).name=='needs.json')
        target=group/'step4/data/source_inputs'/p.pose/'needs.json';target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(original.read_bytes())
    growth=E.AbsoluteGrow(group,case,placement);report=growth.run(pitch=.008)
    report['direction_objective']=plan
    report['construction']['method']='Trained absolute-volume network contacts/seating; actual envelope growth'
    report['provenance']['code'].update(E.I.hashes([Path(__file__),HERE/'model.py',Path(E.__file__),Path(C.__file__),Path(P.__file__),HERE/'best.pt']))
    report['previous_material_volume_cm3']=None;report['material_reduction_percent']=None
    E.save(growth.out/'report.json',report)
    result=evaluate(group)
    E.save(group/'comparison.json',dict(complete=True,passed=True,group=name,step5=result['metrics'],
        elapsed_s=time.monotonic()-started,head_count_penalty=0,new_neural_network_used=True))
    V.render(group)
    print('NEURAL PASS',name,result['metrics']['object_and_support_poses']['box_volume_cm3'],flush=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--groups',nargs='+');args=parser.parse_args()
    E.OUT=OUT;E.install_recorded_recovery(OUT);policy=Policy();specs=dict(E.specs());rows=[]
    for name in args.groups or list(specs):
        try:
            result=run(policy,name,specs[name]);rows.append(dict(group=name,passed=True,volume_cm3=result['metrics']['object_and_support_poses']['box_volume_cm3']))
        except Exception as error:
            import traceback;traceback.print_exc();rows.append(dict(group=name,passed=False,error=str(error)))
            E.save(OUT/name/'comparison.json',dict(complete=True,passed=False,error=str(error),new_neural_network_used=True))
        E.save(OUT/'progress.json',dict(complete=False,results=rows))
    E.save(OUT/'progress.json',dict(complete=True,results=rows))


if __name__=='__main__':main()
