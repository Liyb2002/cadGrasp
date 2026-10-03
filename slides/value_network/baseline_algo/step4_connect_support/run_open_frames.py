"""Replace the current Step4 result with open frames; no historical folders."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step4_connect_support import reseating as R
from step4_connect_support.codesign_port import build_open_frames as L
from step4_connect_support.run_reseated import groups
from step4_connect_support.run_sequential_k import plain


def run(group):
    began=time.perf_counter();out=group/'step4/convex_landing'
    plan_path=out/'data/design_inputs.json';plan=I.check_report(plan_path)
    case=R.load_case(out);placement=plan['placement']
    b,o=np.asarray(placement['bases']),np.asarray(placement['offsets'])
    case.support_seed_records=R.root_records(case,b,o)
    report=dict(schema='independent_seating_step4_v2',complete=False,object=case.name,poses=case.poses,
        constructed=False,passed=False,step3_passed=case.schedule['passed'],
        step3_covered_counts=case.schedule['covered_counts'],physical_head_count=sum(map(len,case.groups)),
        shared_head_count=0,head_model=case.head_model,contact_and_load_inputs_unchanged=True,
        source_schedule=str((case.source/'schedule.json').relative_to(I.ROOT)),
        construction_model='open_ground_frames_single_slender_stem_per_head',
        ground_hull_interior_required_as_material=False,placement=placement,
        maximum_spatial_span_m=plan['maximum_spatial_span_m'],
        maximum_horizontal_span_m=plan['maximum_spatial_span_m'],max_span_ratio=plan['max_span_ratio'],
        object_scale_m=case.scale,support_seed_records=case.support_seed_records,
        videos_generated=False,previous_volume_cm3=plan['previous_volume_cm3'],
        search_scope='Current saved seating; finite narrow-frame, stem and thin-joint routing only',
        global_optimality_or_infeasibility_claim=False)
    work=out/'data/current_build';cache=out/'data/frame_cache'
    try:
        L.build(work,plan['ground_hulls_xy_m'],case=case,placement=placement,
            verify=True,allow_failed=True,skip_unreachable=True,cache_dir=cache,export_stl=False)
        built=I.check_report(work/'report.json')
        mesh=trimesh.load(work/'fixture.obj',force='mesh',process=False)
        metrics=R.span_metrics(mesh.vertices,b,o)
        compact=metrics['maximum_spatial_span_m']<=plan['maximum_spatial_span_m']+1e-9
        passed=bool(built['passed'] and compact)
        shutil.copyfile(work/'fixture.obj',out/'shape.obj')
        report.update(constructed=True,passed=passed,construction=built,
            body_directory=str(work.relative_to(out)),volume_cm3=built['volume_cm3'],
            metrics=metrics,compactness_passed=compact,
            status='accepted_fixture' if passed else 'constructed_candidate_failed_acceptance')
    except (RuntimeError,ValueError) as error:
        report.update(status='construction_search_failed',preview_layout_passed=True,error=str(error))
        (out/'shape.obj').unlink(missing_ok=True)
        (out/'shape.html').unlink(missing_ok=True)
        print('OPEN FRAME FAILED',group.name,str(error),flush=True)
    # Only the current geometry and its inputs remain. Old attempts and heavy
    # plate caches are removed, not renamed into another history directory.
    for name in ('attempts','build_cache','search'):
        path=out/'data'/name
        if path.is_dir():shutil.rmtree(path)
    for name in ('independent_review.json','visualization.json','progress.json'):
        (out/'data'/name).unlink(missing_ok=True)
    report.update(complete=True,seconds=time.perf_counter()-began,
        provenance=dict(inputs=I.hashes(case.paths),code=I.hashes([Path(__file__),Path(R.__file__),Path(L.__file__)])),
        artifacts={'../shape.obj':I.sha256(out/'shape.obj')} if report['constructed'] else {})
    I.save(out/'data/report.json',plain(report))
    print('OPEN FRAME COMPLETE',group.name,report['status'],report.get('volume_cm3'),flush=True)
    return report


def batch(name,jobs):
    def worker(group):
        with (group/'step4/convex_landing/data/run.log').open('w') as log:
            p=subprocess.run([sys.executable,str(Path(__file__).resolve()),name,'--group',group.name],stdout=log,stderr=subprocess.STDOUT)
        print(group.name,p.returncode,flush=True)
        return dict(group=group.name,returncode=p.returncode)
    with ThreadPoolExecutor(max_workers=jobs) as pool:rows=list(pool.map(worker,groups(name)))
    I.save(I.OUTPUTS/name/'convex_landing_batch.json',dict(complete=all(r['returncode']==0 for r in rows),groups=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('object',nargs='?',default='B');p.add_argument('--group');p.add_argument('--jobs',type=int,default=2)
    args=p.parse_args()
    if args.group:run(I.OUTPUTS/args.object/args.group)
    else:batch(args.object,args.jobs)
