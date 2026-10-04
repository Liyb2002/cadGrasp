"""Fresh triangle/load acceptance of diagnostic six-head rescues."""
import json
from pathlib import Path
import numpy as np
from codes.precompute_objects import head_cache as HC
from codes.precompute_objects.heads.surface import FLOOR_CLEARANCE_M
from step3_scheculer import contacts as I
from step3_scheculer.initialize_cached_v19 import STAGE,compatibility
from step3_scheculer.initialize_current_v16 import current_task
from step3_scheculer.pair_scoring import J
from step3_scheculer.run_sequential import numerical_recovery


def main():
    data=json.loads((I.OUTPUTS/'B'/STAGE/'failure_diagnosis_v21.json').read_text());assert data['complete']
    rows=[]
    for d in data['results']:
        assert d['complete'] and 'error' not in d
        name,pose=d['object'],d['pose'];folder=I.OUTPUTS/name/STAGE/pose/'step3_scheculer/failure_diagnosis_v21'
        evidence=I.check_report(folder/'report.json');groups=d['directional_force_upper_bounds']
        row=dict(object=name,pose=pose,original_missing=d['original']['missing'],force_union_passed=d['all_valid_heads_force_upper_bound']['all_passed'],
                 directional_classes=len(groups),passing_classes=sum(g.get('all_passed',False) for g in groups),unknown_classes=sum('error' in g for g in groups),
                 best_directional_coverage=max([g.get('covered',0) for g in groups],default=0),rescued=False)
        rescue=d.get('backward_reduction')
        if rescue is not None:
            task=current_task(name,pose);cache=HC.load(HC.ROOT/'objects'/name/'poses'/pose,200,strict=True)
            ids=set(rescue['selected_ids']);entries=[e for es in cache['pools'].values() for e in es if e['contact']['candidate_id'] in ids]
            assert len(entries)==rescue['heads'] and len(ids)==len(entries)
            ds,cs=compatibility(entries,cache['catalogue']);assert ds and cs
            assert rescue['direction'] in ds and rescue['port'] in cs
            centers=np.array([e['contact']['center_m'] for e in entries]);distinct=all(np.linalg.norm(centers[i]-centers[j])>task.domain.mesh.extents.max()*1e-6 for i in range(len(centers)) for j in range(i))
            if not distinct:
                # Try dropping either co-located patch before reporting a
                # rescue under the original distinct-center policy.
                for i,e in enumerate(entries):
                    others=entries[:i]+entries[i+1:]
                    if not all(np.linalg.norm(a['contact']['center_m']-b['contact']['center_m'])>task.domain.mesh.extents.max()*1e-6 for j,a in enumerate(others) for b in others[:j]):continue
                    trial=task.supply([v['contact'] for v in others])
                    try:
                        with numerical_recovery(folder/'audit_numerical_retries',{}):tm,ti=J.classify(trial,task.targets)
                    except (RuntimeError,np.linalg.LinAlgError):continue
                    if tm.all():entries=others;distinct=True;break
            contacts=[]
            for e in entries:
                assert e['valid'] and e['local_clearance']['valid'];c=e['contact']
                assert not np.intersect1d(c['source_faces'],task.domain.work_ids).size
                assert c['triangles_m'][:,:,2].min()>=FLOOR_CLEARANCE_M-task.domain.mesh.extents.max()*1e-10
                pts=c['triangles_m'].reshape(-1,3);ns=np.repeat(-task.domain.mesh.face_normals[c['source_faces']],3,axis=0)
                np.testing.assert_allclose(np.c_[ns,np.cross(pts-task.domain.com,ns)],c['wrench_generators'],atol=1e-12,rtol=0)
                contacts.append({k:v for k,v in c.items() if k not in ('wrench_generators','wrench_com_m')})
            with numerical_recovery(folder/'audit_numerical_retries',{}):mask,info=J.classify(task.supply(contacts),task.targets)
            assert len(mask)==32768 and mask.all()
            row.update(rescued=bool(len(entries)<=6 and distinct),heads=len(entries),distinct_centers=distinct,
                       full_cpu_covered=int(mask.sum()),exit_world=(-np.array(cache['catalogue']['vectors'][rescue['direction']])).tolist(),
                       area_fractions=[e['area_fraction'] for e in entries])
        rows.append(row)
    out=I.OUTPUTS/'B'/STAGE
    I.save(out/'failure_diagnosis_audit_v22.json',dict(complete=True,original_passed=613,original_failed=17,newly_rescued=sum(r['rescued'] for r in rows),
        scope='Single-pose contacts and cached common withdrawal/port only; no full support construction',results=rows,
        provenance=dict(inputs=I.hashes([out/'failure_diagnosis_v21.json']),code=I.hashes([Path(__file__)]))))
    lines=['# Failed-pose diagnosis','', '| Object | Pose | Original missing loads | Best common-exit candidate union | Numerically unresolved direction classes | New six-head solution |', '|---|---|---:|---:|---:|---|']
    lines += [f"| {r['object']} | {r['pose']} | {r['original_missing']} | {r['best_directional_coverage']}/32768 | {r['unknown_classes']} | {'yes' if r['rescued'] else 'not found'} |" for r in sorted(rows,key=lambda r:(r['object'],int(r['pose'].split('_')[1])))]
    (out/'failure_diagnosis_v22.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
