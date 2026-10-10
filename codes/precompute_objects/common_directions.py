"""Audit saved pose sets and construct certified no-common-direction groups.

Only native-floor hemispheres are tested. Loads, poses and published sets are
immutable. Strict, boundary-only and absent nonzero intersections are distinct.
"""
import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog

ROOT=Path(__file__).resolve().parents[2]
TOL=1e-9
OPTIONS={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9}


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def classify(normals):
    normals=np.asarray(normals,float)
    if normals.ndim!=2 or normals.shape[1]!=3 or not np.isfinite(normals).all():raise ValueError('finite Nx3 normals required')
    np.testing.assert_allclose(np.linalg.norm(normals,axis=1),1.,atol=1e-10)
    lp=linprog([0.,0.,0.,-1.],A_ub=np.c_[-normals,np.ones(len(normals))],
               b_ub=np.zeros(len(normals)),bounds=[(-1.,1.)]*3+[(None,None)],options=OPTIONS,method='highs')
    if not lp.success:raise RuntimeError(lp.message)
    margin=float(lp.x[3])
    if margin>TOL:
        d=lp.x[:3]/np.linalg.norm(lp.x[:3])
        if np.min(normals@d)<=TOL:raise RuntimeError('strict witness failed')
        return dict(status='strict_common_direction',common_direction_exists=True,
                    direction=d.tolist(),minimum_upward_dot=float(np.min(normals@d)),lp_cube_margin=margin)
    # Exclude v=0 rigorously: every nonzero cone vector can be rescaled so one
    # of its coordinates is +1 or -1 and all coordinates lie in [-1,1].
    for axis in range(3):
        for sign in [-1.,1.]:
            bounds=[(-1.,1.)]*3;bounds[axis]=(sign,sign)
            edge=linprog(np.zeros(3),A_ub=-normals,b_ub=np.zeros(len(normals)),bounds=bounds,options=OPTIONS,method='highs')
            if edge.success:
                d=edge.x/np.linalg.norm(edge.x)
                if np.min(normals@d)<-TOL:raise RuntimeError('boundary witness failed')
                return dict(status='boundary_only_common_direction',common_direction_exists=True,
                            direction=d.tolist(),minimum_upward_dot=float(np.min(normals@d)),lp_cube_margin=margin)
            if edge.status!=2:raise RuntimeError('nonzero feasibility unresolved: '+edge.message)
    certificate=positive_span_certificate(normals)
    return dict(status='no_nonzero_common_direction',common_direction_exists=False,direction=None,
                lp_cube_margin=margin,certificate=certificate)


def positive_span_certificate(normals):
    normals=np.asarray(normals,float);count=len(normals)
    matrix=np.zeros((4,count+1));matrix[:3,:count]=normals.T;matrix[3,:count]=1.
    inequalities=np.c_[-np.eye(count),np.ones(count)]
    lp=linprog(np.r_[np.zeros(count),-1.],A_ub=inequalities,b_ub=np.zeros(count),
               A_eq=matrix,b_eq=[0.,0.,0.,1.],bounds=[(0.,None)]*(count+1),options=OPTIONS,method='highs')
    rank=int(np.linalg.matrix_rank(normals,tol=1e-9))
    if not lp.success or lp.x[-1]<=TOL or rank!=3:raise RuntimeError('no stable positive-span certificate')
    weights=lp.x[:count];residual=float(np.max(np.abs(weights@normals)))
    if residual>1e-9:raise RuntimeError('positive-span residual unresolved')
    return dict(type='strict_positive_dependence_and_rank_three',weights=weights.tolist(),
                minimum_weight=float(weights.min()),zero_sum_residual=residual,normal_rank=rank,
                explanation='N*d>=0 and positive weights with weights*N=0 imply N*d=0; rank(N)=3 forces d=0')


def floor_info(indices,violations):
    block=violations[np.ix_(indices,indices)]
    return dict(floor_demands_compatible=bool(np.max(block)==0),total_directed_violating_samples=int(block.sum()))


def build_counterexamples(normals,violations,per_size=4,excluded=()):
    whole=classify(normals)
    if whole['common_direction_exists']:
        return [],dict(status='impossible_from_saved_poses',reason='All saved poses already share a nonzero direction; every subset also shares it',all_pose_witness=whole)
    combos=np.array(list(itertools.combinations(range(len(normals)),4)),int)
    matrices=np.concatenate([np.transpose(normals[combos],(0,2,1)),np.ones((len(combos),1,4))],axis=1)
    valid=np.abs(np.linalg.det(matrices))>1e-9
    weights=np.zeros((len(combos),4))
    weights[valid]=np.linalg.solve(matrices[valid],np.broadcast_to(np.array([0.,0.,0.,1.]),(int(valid.sum()),4))[...,None])[...,0]
    valid &= weights.min(axis=1)>1e-6
    cores=combos[valid]
    if not len(cores):
        core=list(range(len(normals)))
        for i in list(core):
            trial=[k for k in core if k!=i]
            if len(trial)>=4 and not classify(normals[trial])['common_direction_exists']:core=trial
        cores=np.array([core],int)
    # Prefer floor-compatible counterexamples; otherwise label floor failure.
    core_rows=[]
    for c in cores:
        info=floor_info(c,violations)
        core_rows.append((not info['floor_demands_compatible'],info['total_directed_violating_samples'],tuple(map(int,c))))
    core_rows.sort()
    selected=[];used={tuple(sorted(g)) for g in excluded}
    for size in [4,5,6]:
        for require_floor in [True,False]:
            if sum(r['size']==size for r in selected)>=per_size:break
            for core_incompatible,_,core in core_rows:
                if len(core)>size or (require_floor and core_incompatible):continue
                rest=[k for k in range(len(normals)) if k not in core]
                if require_floor:
                    rest=[k for k in rest if not np.any(violations[k,list(core)]) and not np.any(violations[list(core),k])]
                for extra in itertools.combinations(rest,size-len(core)):
                    group=tuple(sorted(core+extra))
                    if group in used:continue
                    info=floor_info(group,violations)
                    if require_floor and not info['floor_demands_compatible']:continue
                    verdict=classify(normals[list(group)])
                    if verdict['common_direction_exists']:raise RuntimeError('counterexample certification failed')
                    used.add(group)
                    selected.append(dict(indices=list(group),size=size,common_direction=verdict,**info))
                    if sum(r['size']==size for r in selected)>=per_size:break
                if sum(r['size']==size for r in selected)>=per_size:break
    return selected,dict(status='constructed',certified_four_pose_core_count=len(combos[valid]),
                         floor_compatible_four_pose_core_count=sum(not r[0] for r in core_rows),all_pose_result=whole)


def run(per_size=4):
    began=time.perf_counter();objects=[]
    folders=sorted(p.parent for p in (ROOT/'objects').glob('*/poses.json') if (p.parent/'pose_sets.json').exists())
    for folder in folders:
        manifest=json.loads((folder/'poses.json').read_text());rows=manifest['poses'];names=[r['pose_id'] for r in rows]
        normals=np.array([np.asarray(r['T_world_mesh'])[:3,:3].T@np.array([0.,0.,1.]) for r in rows])
        lookup={name:i for i,name in enumerate(names)}
        data=json.loads((folder/'pose_sets.json').read_text());violations=np.asarray(data['directed_violating_counts'])
        evaluated=[];sources={}
        for source in sorted(folder.glob('*pose_sets.json')):
            content=json.loads(source.read_text())
            if source.name=='no_common_direction_pose_sets.json' and content.get('category')!='legal_without_common_direction':continue
            sources[source.name]=digest(source)
            for group in content['sets']:
                ids=[lookup[p] for p in group['poses']]
                evaluated.append(dict(id=group['id'],poses=group['poses'],size=len(ids),source=source.name,
                                      common_direction=classify(normals[ids]),**floor_info(ids,violations)))
        provenance=dict(pose_manifest_sha256=digest(folder/'poses.json'),pose_sets_sha256=sources,
                        code_sha256=digest(Path(__file__)))
        if data.get('category')=='legal_with_common_direction':
            existing=json.loads((folder/'no_common_direction_pose_sets.json').read_text())
            examples=[g for g in existing['sets'] if not g.get('canonical_precomputed_set',False)]
            construction=dict(status='already_categorized',new_counterexamples_preserved=len(examples))
        else:
            excluded=[tuple(lookup[p] for p in g['poses']) for g in evaluated]
            examples,construction=build_counterexamples(normals,violations,per_size,excluded)
            for group in examples:
                ids=group.pop('indices');group['poses']=[names[i] for i in ids]
                group['id']='pose'+'+'.join(str(i+1) for i in ids)
            write(folder/'no_common_direction_pose_sets.json',dict(schema='cadgrasp_no_common_direction_sets_v1',object=folder.name,
                  set_count=len(examples),sets=examples,construction=construction,provenance=provenance,
                  scope='native-floor hemisphere intersection only; no new poses/loads; floor compatibility separately recorded'))
        counts={status:sum(g['common_direction']['status']==status for g in evaluated) for status in
                ['strict_common_direction','boundary_only_common_direction','no_nonzero_common_direction']}
        result=dict(object=folder.name,pose_count=len(rows),existing_set_count=len(evaluated),counts=counts,
                    sets=evaluated,constructed_set_count=len(examples),construction=construction,provenance=provenance)
        write(folder/'common_direction_audit.json',result);objects.append(result)
        print(folder.name,counts,'constructed',len(examples),flush=True)
    summary=dict(schema='cadgrasp_common_direction_audit_v1',object_count=len(objects),
        existing_set_count=sum(r['existing_set_count'] for r in objects),
        counts={s:sum(r['counts'][s] for r in objects) for s in objects[0]['counts']},
        constructed_set_count=sum(r['constructed_set_count'] for r in objects),objects=objects,
        tolerance=TOL,seconds=time.perf_counter()-began,
        scope='nonzero exits satisfying every saved pose native floor hemisphere; no force/geometry feasibility claim')
    write(ROOT/'objects/common_direction_audit.json',summary)
    print('TOTAL',summary['counts'],'new',summary['constructed_set_count'],'seconds',summary['seconds'],flush=True)
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--per-size',type=int,default=4)
    args=parser.parse_args()
    if args.per_size<1:parser.error('per-size must be positive')
    run(args.per_size)
