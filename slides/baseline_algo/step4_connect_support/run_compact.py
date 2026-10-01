"""Build and fully check compact, independently seated two-pose fixtures."""
import argparse
import itertools
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step4_connect_support import compact_layout as C
from step4_connect_support.codesign_port import build_local_bodies as L
from step4_connect_support.run_sequential_k import foot_menu,plain


def run(output,max_span_ratio=1.65,max_checks=2500,keep=8,max_attempts=8):
    start=time.perf_counter();case=C.load_case(output)
    layout_file=case.output/'data/search/layouts.json'
    config_file=case.output/'data/search/search_config.json'
    if layout_file.exists() and config_file.exists():
        saved=I.check_report(layout_file)
        config=json.loads(config_file.read_text())
        if config['max_span_ratio']!=max_span_ratio:
            raise ValueError('Existing compact search uses another footprint bound')
        layouts=saved['selected']
    else:layouts=C.search(case,max_span_ratio,max_checks,keep)
    report=dict(schema='compact_independent_seating_v1',complete=False,object=case.name,
        poses=case.poses,constructed=False,passed=False,step3_passed=case.schedule['passed'],
        covered_counts=case.schedule['covered_counts'],shared_head_count=0,
        physical_head_count=sum(map(len,case.groups)),head_model=case.head_model,
        contact_and_load_inputs_unchanged=True,maximum_horizontal_span_m=max_span_ratio*case.scale,
        max_span_ratio=max_span_ratio,object_scale_m=case.scale,
        search_scope='Finite rigid-group placements, sampled tilts/twists/translations and saved own-pose straight withdrawal directions',
        floor_prefilter_scope='Additional sufficient restriction: all demand endpoints and original object pivots above every floor',
        global_optimality_or_infeasibility_claim=False,attempts=[])
    best=None;accepted=0
    for li,layout in enumerate(layouts):
        bases=np.asarray(layout['bases']);offsets=np.asarray(layout['offsets'])
        case.support_seed_records=C.root_records(case,bases,offsets)
        feet=list(foot_menu(case,bases,offsets))
        for ids in itertools.product(*layout['direction_menus']):
            placement=dict(bases=bases,offsets=offsets,directions=np.array([cat[i] for cat,i in zip(case.catalogues,ids)]),direction_ids=list(ids))
            for fi,terminals in enumerate(feet[:2]):
                if len(report['attempts'])>=max_attempts:break
                index=len(report['attempts']);work=case.output/'data/attempts'/f'{index:03d}'
                attempt=dict(index=index,layout_index=li,search_candidate_index=layout['candidate_index'],direction_ids=list(ids),feet_index=fi)
                report['attempts'].append(attempt)
                print('COMPACT BUILD',case.pair.name,index,'layout',li,'feet',fi,flush=True)
                try:
                    cache=case.output/'data/build_cache'/f'layout_{li}_dir_{"_".join(map(str,ids))}'
                    L.build(work,terminals,case=case,placement=placement,verify=True,allow_failed=True,
                        skip_unreachable=True,floor_policy='nearest',cache_dir=cache,export_stl=False)
                    built=I.check_report(work/'report.json')
                    mesh=trimesh.load(work/'fixture.obj',force='mesh',process=False)
                    metrics=C.span_metrics(mesh.vertices,bases,offsets)
                    compact=metrics['maximum_horizontal_span_m']<=report['maximum_horizontal_span_m']+1e-9
                    passed=bool(built['passed'] and compact and case.schedule['passed'])
                    attempt.update(constructed=True,physical_passed=built['passed'],compactness_passed=compact,
                        passed=passed,volume_cm3=built['volume_cm3'],metrics=metrics)
                    # Full acceptance precedes material volume in the ranking.
                    score=(not passed,not compact,not bool(built['passed']),built['volume_cm3'])
                    if best is None or score<best:
                        best=score
                        shutil.copyfile(work/'fixture.obj',case.output/'shape.obj')
                        report.update(constructed=True,passed=passed,selected_attempt=index,
                            placement=plain(placement),construction=built,body_directory=str(work.relative_to(case.output)),
                            volume_cm3=built['volume_cm3'],metrics=metrics,compactness_passed=compact,
                            support_seed_records=case.support_seed_records,
                            status='accepted_compact_fixture' if passed else 'constructed_candidate_failed_acceptance')
                    accepted+=int(passed)
                except (RuntimeError,ValueError) as error:
                    attempt.update(constructed=False,passed=False,error=str(error))
                    print('COMPACT BODY FAILED',case.pair.name,index,str(error),flush=True)
                I.save(case.output/'data/progress.json',plain(report))
                # Compare a few fully accepted candidates instead of claiming
                # the first feasible shape minimizes material globally.
                if accepted>=2:break
            if len(report['attempts'])>=max_attempts or accepted>=2:break
        if len(report['attempts'])>=max_attempts or accepted>=2:break
    report.update(complete=True,seconds=time.perf_counter()-start,
        status=report.get('status','no_constructed_body_in_finite_search'),
        provenance=dict(inputs=I.hashes(case.paths+[layout_file]),code=I.hashes([Path(__file__),Path(C.__file__),Path(L.__file__)])),
        artifacts={'../shape.obj':I.sha256(case.output/'shape.obj')} if report['constructed'] else {})
    I.save(case.output/'data/report.json',plain(report))
    print('COMPACT FINISHED',case.pair.name,report['status'],report.get('volume_cm3'),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('outputs',type=Path,nargs='+')
    parser.add_argument('--max-span-ratio',type=float,default=1.65)
    parser.add_argument('--max-checks',type=int,default=2500)
    parser.add_argument('--keep',type=int,default=8)
    parser.add_argument('--max-attempts',type=int,default=8)
    args=parser.parse_args()
    for output in args.outputs:run(output,args.max_span_ratio,args.max_checks,args.keep,args.max_attempts)
