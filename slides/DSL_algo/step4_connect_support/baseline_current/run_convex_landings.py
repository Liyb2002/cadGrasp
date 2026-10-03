"""Run independent-seating Step4 for every retained B pose combination."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step4_connect_support.baseline_current import reseating as R
from step4_connect_support.baseline_current.codesign_port import build_convex_landings as L
from step4_connect_support.baseline_current.run_sequential_k import foot_menu,plain


def groups(name):
    return sorted((p for p in (I.OUTPUTS/name).glob('pose*+*')
        if (p/'step3_scheculer/independent_poses_floor2mm').is_dir()),
        key=lambda p:(len(p.name.split('+')),[int(x) for x in p.name[4:].split('+')]))


def run(group,limit_ratio=2.2,count=18000,max_checks=6000,keep=6,max_attempts=2):
    start=time.perf_counter();output=group/'step4/convex_landing';case=R.load_case(output)
    previous=group/'step4/reseated'
    layout_file=previous/'data/search/layouts.json'
    search=I.check_report(layout_file)
    old_report=I.check_report(previous/'data/report.json')
    selected=old_report['attempts'][old_report['selected_attempt']]['layout_index']
    order=[selected]+[k for k in range(len(search['selected'])) if k!=selected]
    search=dict(search,selected=[search['selected'][k] for k in order])
    report=dict(schema='independent_seating_step4_v2',complete=False,object=case.name,poses=case.poses,
        constructed=False,passed=False,step3_passed=case.schedule['passed'],
        step3_covered_counts=case.schedule['covered_counts'],physical_head_count=sum(map(len,case.groups)),
        shared_head_count=0,head_model=case.head_model,contact_and_load_inputs_unchanged=True,
        source_schedule=str((case.source/'schedule.json').relative_to(I.ROOT)),
        construction_model='one_convex_landing_and_loft_per_head_floor',
        maximum_connections_per_head_floor=1,previous_placement_search=str(layout_file.relative_to(I.ROOT)),
        maximum_horizontal_span_m=limit_ratio*case.scale,max_span_ratio=limit_ratio,
        maximum_spatial_span_m=limit_ratio*case.scale,
        object_scale_m=case.scale,attempts=[],videos_generated=False,
        search_scope='Finite independent rigid-group placements; prism/corner/polyhedron families and saved local exit menus',
        floor_prefilter_scope='Restricted search family: original demand endpoints and object pivots above every floor',
        global_optimality_or_infeasibility_claim=False)
    best=None;accepted=0
    for li,layout in enumerate(search['selected']):
        bases=np.asarray(layout['bases']);offsets=np.asarray(layout['offsets'])
        case.support_seed_records=R.root_records(case,bases,offsets)
        feet=list(foot_menu(case,bases,offsets))[:2]
        # Compare layouts before spending the entire budget on one large
        # Cartesian product of otherwise independent direction menus.
        choices=list(itertools.islice(itertools.product(*layout['direction_menus']),1))
        for fi,ids in itertools.product(range(len(feet)),choices):
            if len(report['attempts'])>=max_attempts:break
            placement=dict(bases=bases,offsets=offsets,
                directions=np.array([cat[i] for cat,i in zip(case.catalogues,ids)]),direction_ids=list(ids))
            index=len(report['attempts']);work=output/'data/attempts'/f'{index:03d}'
            attempt=dict(index=index,layout_index=li,family=layout['family'],direction_ids=list(ids),feet_index=fi)
            report['attempts'].append(attempt)
            print('CONVEX LANDING BUILD',group.name,index,'layout',li,'feet',fi,flush=True)
            try:
                cache=output/'data/build_cache'/f'layout_{li}_dir_{"_".join(map(str,ids))}'
                old_cache=previous/'data/build_cache'/f'layout_{order[li]}_dir_{"_".join(map(str,ids))}'
                cache.mkdir(parents=True,exist_ok=True)
                for cached in old_cache.iterdir():
                    if cached.name!='construction_plan.json' and cached.is_file() and not (cache/cached.name).exists():
                        shutil.copyfile(cached,cache/cached.name)
                L.build(work,feet[fi],case=case,placement=placement,verify=True,allow_failed=True,
                    skip_unreachable=True,floor_policy='nearest',cache_dir=cache,export_stl=False)
                built=I.check_report(work/'report.json')
                mesh=trimesh.load(work/'fixture.obj',force='mesh',process=False)
                metrics=R.span_metrics(mesh.vertices,bases,offsets)
                compact=metrics['maximum_spatial_span_m']<=report['maximum_spatial_span_m']+1e-9
                # Actual full original-load verification is authoritative even
                # when an earlier Step3 solver stopped with partial coverage.
                passed=bool(built['passed'] and compact)
                loads=sum(c['coupled_equilibrium_passed'] for c in built['checks'])
                exits=sum(c['withdrawal']['clear'] for c in built['checks'])
                attempt.update(constructed=True,physical_passed=built['passed'],compactness_passed=compact,
                    passed=passed,load_tasks_passed=loads,exit_tasks_passed=exits,
                    volume_cm3=built['volume_cm3'],metrics=metrics)
                score=(not passed,not compact,-exits,-loads,built['volume_cm3'])
                if best is None or score<best:
                    best=score;shutil.copyfile(work/'fixture.obj',output/'shape.obj')
                    report.update(constructed=True,passed=passed,selected_attempt=index,
                        placement=plain(placement),construction=built,body_directory=str(work.relative_to(output)),
                        volume_cm3=built['volume_cm3'],metrics=metrics,compactness_passed=compact,
                        support_seed_records=case.support_seed_records,
                        status='accepted_fixture' if passed else 'constructed_candidate_failed_acceptance')
                accepted+=int(passed)
            except (RuntimeError,ValueError) as error:
                attempt.update(constructed=False,passed=False,error=str(error))
                print('CONVEX LANDING BUILD FAILED',group.name,index,str(error),flush=True)
            I.save(output/'data/progress.json',plain(report))
            if accepted>=2:break
        if len(report['attempts'])>=max_attempts or accepted>=2:break
    if not report['constructed']:
        fallback=(search['selected'] or [search['fallback']])[0]
        if fallback:
            ids=[m[0] for m in fallback.get('direction_menus',case.menus)]
            report['placement']=dict(bases=fallback['bases'],offsets=fallback['offsets'],direction_ids=ids,
                directions=[case.catalogues[k][i].tolist() for k,i in enumerate(ids)])
            report['preview_layout_passed']=bool(search['selected'])
        report['status']='construction_search_failed' if search['selected'] else 'no_layout_in_finite_search'
        # A rerun must not expose an earlier body as this run's result.
        (output/'shape.obj').unlink(missing_ok=True)
    report.update(complete=True,seconds=time.perf_counter()-start,layout_count=len(search['selected']),
        layouts_checked=search['exact_layout_checks'],
        provenance=dict(inputs=I.hashes(case.paths+[layout_file,previous/'data/report.json']),code=I.hashes([Path(__file__),Path(R.__file__),Path(L.__file__)])),
        artifacts={'../shape.obj':I.sha256(output/'shape.obj')} if report['constructed'] else {})
    I.save(output/'data/report.json',plain(report))
    print('CONVEX LANDING COMPLETE',group.name,report['status'],report.get('volume_cm3'),flush=True)
    return report


def batch(name,args):
    selected=groups(name)
    def worker(group):
        folder=group/'step4/convex_landing/data';folder.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,str(Path(__file__).resolve()),name,'--group',group.name,
            '--limit-ratio',str(args.limit_ratio),'--count',str(args.count),'--max-checks',str(args.max_checks),
            '--keep',str(args.keep),'--max-attempts',str(args.max_attempts)]
        with (folder/'run.log').open('w') as log:
            process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
        result=dict(group=group.name,returncode=process.returncode)
        if process.returncode==0:
            r=I.check_report(folder/'report.json');result.update(status=r['status'],passed=r['passed'],constructed=r['constructed'],poses=r['poses'])
        print('CONVEX LANDING BATCH',json.dumps(result),flush=True)
        return result
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:rows=list(executor.map(worker,selected))
    destination=I.OUTPUTS/name/'convex_landing_batch.json'
    I.save(destination,dict(complete=all(r['returncode']==0 for r in rows),groups=rows,
        videos_generated=False,provenance=dict(inputs=I.hashes([p/'step4/convex_landing/data/report.json' for p,r in zip(selected,rows) if r['returncode']==0]),code=I.hashes([Path(__file__)]))))
    if any(r['returncode'] for r in rows):raise RuntimeError('Some Step4 runs crashed; inspect group run.log files')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('object',nargs='?',default='B')
    p.add_argument('--group');p.add_argument('--jobs',type=int,default=2)
    p.add_argument('--limit-ratio',type=float,default=2.2);p.add_argument('--count',type=int,default=18000)
    p.add_argument('--max-checks',type=int,default=6000);p.add_argument('--keep',type=int,default=6)
    p.add_argument('--max-attempts',type=int,default=2);args=p.parse_args()
    if args.group:run(I.OUTPUTS/args.object/args.group,args.limit_ratio,args.count,args.max_checks,args.keep,args.max_attempts)
    else:batch(args.object,args)
