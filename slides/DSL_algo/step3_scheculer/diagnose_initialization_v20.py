"""Separate candidate-force limits from finite six-head search failures."""
import json,time,contextlib,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from codes.precompute_objects import head_cache as HC
from step3_scheculer import contacts as I
from step3_scheculer.initialize_cached_v19 import Search,STAGE,compatibility
from step3_scheculer.pair_scoring import J
from step3_scheculer.run_sequential import numerical_recovery


def solve(case):
    name,pose=case['object'],case['pose'];out=I.OUTPUTS/name/STAGE/pose/'step3_scheculer/failure_diagnosis_v20';out.mkdir(exist_ok=True)
    with (out/'pipeline.log').open('w',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        start=time.monotonic();result=dict(object=name,pose=pose,original=case,complete=False)
        try:
            with numerical_recovery(out/'numerical_retries',{}):
                s=Search(name,pose);pools={a:[e for e in pool if e['valid']] for a,pool in s.pools.items()}
                def score(entries,targets=None):
                    full=I.merge_columns(s.floor,*[s.columns[e['contact']['candidate_id']] for e in entries])
                    mask,info=J.classify(full,s.task.targets if targets is None else targets)
                    return mask,info
                result['area_pools']={str(a):dict(valid=len(es),total=len(s.pools[a])) for a,es in pools.items()}
                everything=[e for es in pools.values() for e in es]
                mask,info=score(everything);result['all_valid_heads_force_upper_bound']=dict(covered=int(mask.sum()),all_passed=bool(mask.all()),heads=len(everything),exit_constraint_ignored=True,head_limit_ignored=True,lp=info)
                print('upper bound',result['all_valid_heads_force_upper_bound'],flush=True)
                old=I.check_report(s.out/'schedule.json');ids=set(old['result']['selected_ids']);selected=[e for e in everything if e['contact']['candidate_id'] in ids]
                ds,cs=compatibility(selected,s.catalogue);original=np.load(s.out/'coverage.npz')['mask'];missing=s.task.targets[~original]
                result['best_common_directions']=len(ds);result['best_common_ports']=len(cs)
                # First test one extra cached head, allowing mixed areas. This
                # probes six-head/greedy limits without changing exit legality.
                proposals=[]
                for e in everything:
                    if e['contact']['candidate_id'] in ids:continue
                    if not (ds&set(e['directions'][0]) and cs&set(e['path_components'])):continue
                    if any(np.linalg.norm(e['contact']['center_m']-v['contact']['center_m'])<s.scale*1e-6 for v in selected):continue
                    try:m,lp=score(selected+[e],missing)
                    except (RuntimeError,np.linalg.LinAlgError):continue
                    proposals.append(dict(id=e['contact']['candidate_id'],fixed_missing=int(m.sum()),area=e['target_area_fraction'] if 'target_area_fraction' in e else e['area_fraction']))
                    if m.all():
                        fullmask,lp=score(selected+[e]);assert fullmask.all()
                        final_entries=selected+[e];kind='seven_head_rescue'
                        # Remove one old head to recover the original six-head
                        # budget. Every removal checks every original load.
                        for remove in selected:
                            trial=[v for v in final_entries if v is not remove]
                            try:tm,ti=score(trial)
                            except (RuntimeError,np.linalg.LinAlgError):continue
                            if tm.all():final_entries=trial;kind='six_head_swap_rescue';break
                        dd,cc=compatibility(final_entries,s.catalogue);assert dd and cc
                        contacts=[v['contact'] for v in final_entries];fresh=[]
                        for c in contacts:
                            pts=c['triangles_m'].reshape(-1,3);ns=np.repeat(-s.task.domain.mesh.face_normals[c['source_faces']],3,axis=0)
                            np.testing.assert_allclose(np.c_[ns,np.cross(pts-s.task.domain.com,ns)],c['wrench_generators'],atol=1e-12,rtol=0)
                            fresh.append({k:v for k,v in c.items() if k not in ('wrench_generators','wrench_com_m')})
                        finalmask,finalinfo=J.classify(s.task.supply(fresh),s.task.targets);assert finalmask.all()
                        I.save_contacts(out/'rescue_contacts.npz',contacts)
                        result['rescue']=dict(kind=kind,heads=len(final_entries),selected_ids=[v['contact']['candidate_id'] for v in final_entries],common_direction_ids=sorted(dd),common_ports=sorted(cc),full_cpu_covered=int(finalmask.sum()),actual_triangle_generators_checked=True,mixed_areas=True,full_support_constructed=False)
                        print('RESCUE',result['rescue'],flush=True);break
                result['compatible_addition_tests']=proposals
                result['complete']=True
                result['provenance']=dict(inputs=I.hashes(s.inputs+[s.out/'schedule.json']),code=I.hashes([Path(__file__)])+old['provenance']['code'])
        except Exception as error:result['error']=repr(error);traceback.print_exc()
        result['seconds']=time.monotonic()-start;I.save(out/'report.json',result)
        return {k:v for k,v in result.items() if k not in ('provenance','compatible_addition_tests')}


def main():
    import multiprocessing
    failures=json.loads((I.OUTPUTS/'B'/STAGE/'review.json').read_text())['failures'];rows=[]
    with ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(solve,r) for r in failures]):
            r=f.result();rows.append(r);print(json.dumps(r),flush=True)
            I.save(I.OUTPUTS/'B'/STAGE/'failure_diagnosis_v20.json',dict(complete=False,results=rows))
    I.save(I.OUTPUTS/'B'/STAGE/'failure_diagnosis_v20.json',dict(complete=True,results=rows))

if __name__=='__main__':main()
