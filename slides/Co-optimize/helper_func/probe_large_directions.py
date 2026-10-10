"""Screen small OWN-state direction nudges; optionally check isolated geometry.

All probe artifacts remain separate from the active case. A sampled probe is
never published as a feasible fixture. Extra budgets and source layouts persist.
"""
import _bootstrap
import argparse
import os
import time
from co_common import *
from run_all import output_root
from run_large_pose_sets import groups
from whole_pipeline import load_layout
from whole_search.fast_search import FastModel
from whole_search.common import tangent_frame,legal_direction
from whole_search.search import failed_count
from whole_search.exact_worker import write_result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('case')
    parser.add_argument('--real-budget',type=int,default=0);args=parser.parse_args()
    group=next(g for g in groups() if g['id']==args.case or g['source_case']==args.case)
    root=output_root('B');base=root/group['id'];source=base/'step4/step4.2/sampled_layout.npz'
    layout=load_layout(source)
    out=base/'step4/step4.2/direction_probes'/time.strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True);model=FastModel(group['poses'],'B',initialization_report=base/'step3/step3.1/data/report.json')
    first=model.evaluate(layout);model.commit(first);rows=[]; candidates=[]
    for angle in [.03125,.125,.25,.5,1.]:
        trial=layout.copy()
        for k in layout.active:
            trial.directions[k]=legal_direction(layout.directions[k]+np.tan(np.radians(angle))*model.floor_normal(layout,k),model.floor_normal(layout,k))
        candidates.append(('world_up_margin',angle,None,trial))
    for axis in range(2):
        for angle in [-.5,-.125,.125,.5]:
            trial=layout.copy()
            for k in layout.active:
                trial.directions[k]=legal_direction(layout.directions[k]+np.tan(np.radians(angle))*tangent_frame(layout.directions[k])[:,axis],model.floor_normal(layout,k))
            candidates.append(('coherent_tangent',angle,axis,trial))
    feasible=[]
    record=dict(poses=group['poses'],source_layout_sha256=I.sha256(source),
        own_saved_state_only=True,original_layout_sampled_failed_loads=failed_count(first),
        original_constraints_unchanged=True,real_candidate_budget=args.real_budget,
        code=I.hashes([Path(__file__)]),candidates=rows)
    for index,(kind,angle,axis,trial) in enumerate(candidates):
        row=dict(index=index,kind=kind,degrees=angle,axis=axis)
        try:
            result=model.evaluate(trial);row.update(sampled_failed_loads=failed_count(result),estimated_volume_cm3=result['volume_cm3'])
            row['minimum_world_exit_z']=min(float((model.native[trial.hosts[k],:3,:3]@trial.directions[k])[2]) for k in trial.active)
            if not failed_count(result):
                path=out/f'layout_{index:02d}.npz'
                np.savez_compressed(path,placements=trial.placements,directions=trial.directions,hosts=trial.hosts,active=np.array(trial.active))
                row['layout']=path.name;feasible.append((result,row))
        except Exception as error:row['error']=str(error)
        rows.append(row);save(out/'report.json',record);print('PROBE',group['id'],row,flush=True)
    # Prefer a positive world-up margin before other passing perturbations.
    feasible.sort(key=lambda r:(r[1]['kind']!='world_up_margin',abs(r[1]['degrees']),r[0]['volume_cm3']))
    os.environ['COOPT_WHOLE_RUN_TOKEN']=json.loads((root/'data/whole_step4_active_run.json').read_text())['run_token']
    model.worker_program=HERE/'helper_func/whole_search/exact_worker.py';model.checkpoint_dir=out/'states'
    record['actual_attempts']=[]
    for result,row in feasible[:args.real_budget]:
        attempt=dict(probe_index=row['index'],passed=False)
        try:
            actual=model.exact(result['layout']);attempt.update(counts=actual['counts'],volume_cm3=actual['volume_cm3'],passed=failed_count(actual)==0)
            write_result(actual,out/f'result_{row["index"]:02d}')
        except Exception as error:attempt['error']=str(error)
        record['actual_attempts'].append(attempt);save(out/'report.json',record)
        print('PROBE_ACTUAL',group['id'],attempt,flush=True)
        if attempt['passed']:break
    record['complete']=True;save(out/'report.json',record)


if __name__=='__main__':main()
