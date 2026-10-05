"""Recompute every baseline stage and its figures, then audit the saved results."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse
import math
from contextlib import ExitStack
import fcntl
import json
from pathlib import Path
import subprocess
import sys

from step3_scheculer.completion import read_passed, IncompleteSchedule
from step1.cases import normalize_pose, selected_pose

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parents[1]
STAGES = (
    'step1/needs.py',
    'step2_local_support/circles.py',
    'step2_local_support/insertion_directions.py',
    'step2_local_support/audit.py',
    'step2_local_support/sampling_audit.py',
    'step2_local_support/draw.py',
    'step3_scheculer/scheduler.py',
    'step3_scheculer/verification.py',
    'step3_scheculer/draw_schedule.py',
    'step3_scheculer/draw_directions.py',
    'step3_scheculer/draw_result.py',
    'step4_floor_contact/floor_contact.py',
    'step4_floor_contact/audit.py',
    'step4_floor_contact/draw.py',
    'step5_base/base.py',
    'step5_base/draw.py',
    'step6_connect_support/connect.py',
    'step6_connect_support/audit.py',
    'step6_connect_support/draw.py',
)


def floor_verified(name):
    from step4_floor_contact.floor_contact import read
    return bool(read(name).get('continuous_demand_enclosure_proved'))


def base_verified(name):
    from step5_base.base import read
    return bool(read(name)[0]['passed'])


def connection_verified(name):
    from step3_scheculer import contacts as I
    from step1.cases import pose_name
    return bool(I.check_report(HERE/'output'/name/pose_name()/'step6_connect_support/connection.json')['passed'])


def completed_schedule(name):
    """Reuse a completed search with current inputs, code and saved artifacts."""
    from step3_scheculer import contacts as I
    from step1.cases import pose_name
    root=HERE/'output'/name/pose_name()/'step3_scheculer'
    try:
        schedule=I.check_report(root/'schedule.json')
        state=json.loads((root/'status.json').read_text())
        digest=I.sha256(root/'schedule.json')
        if state.get('complete') and state.get('schedule_sha256')==digest:
            return schedule
    except (OSError, RuntimeError, AssertionError, KeyError, ValueError):
        pass
    return None


def run(objects,from_step=1,resume=False,no_round_drawings=False,pose=None,through_step=None,lp_retry=False,continuous_retry=False,visibility_retry=False,connection_edge_budget=2000,connection_workers=1,timings=None,force_directions=False,search_mode='top5',trajectories=10,search_seed=0,top_k=5,sizing_sweeps=2,sizing_budget=96,primal_first_retry=False,hull_timeout=None):
    from step3_scheculer.random_search import configuration
    search = configuration(search_mode,trajectories,search_seed,top_k,sizing_sweeps,sizing_budget)
    through_step = 6 if through_step is None else through_step
    if through_step < from_step:
        raise ValueError('The last stage must not precede the first stage')
    if force_directions and not from_step <= 2 <= through_step:
        raise ValueError('Forced direction computation requires Step2 in the requested stages')
    if connection_edge_budget < 1:
        raise ValueError('The per-support connection edge budget must be positive')
    if connection_workers < 1:
        raise ValueError('The connection worker count must be positive')
    if hull_timeout is not None:
        if not math.isfinite(hull_timeout) or hull_timeout <= 0:
            raise ValueError('Hull timeout must be finite and positive')
        if resume:
            raise ValueError('A hull timeout requires a fresh run; timeout policies are not resume-compatible')
    if (lp_retry or continuous_retry or hull_timeout is not None) and len(objects) != 1:
        raise ValueError('Use --lp-retry for one object/pose per process')
    with selected_pose(pose):
        return _run(objects,from_step,resume,no_round_drawings,through_step,lp_retry,continuous_retry,visibility_retry,connection_edge_budget,connection_workers,timings,force_directions,search,primal_first_retry,hull_timeout)


def _run(objects,from_step=1,resume=False,no_round_drawings=False,through_step=6,lp_retry=False,continuous_retry=False,visibility_retry=False,connection_edge_budget=2000,connection_workers=1,timings=None,force_directions=False,search=None,primal_first_retry=False,hull_timeout=None):
    from step3_scheculer.random_search import configuration
    search = search or configuration()
    starts={1:'step1/needs.py',2:'step2_local_support/circles.py',3:'step3_scheculer/scheduler.py',
            4:'step4_floor_contact/floor_contact.py',5:'step5_base/base.py',6:'step6_connect_support/connect.py'}
    active = list(objects)
    blocked = []
    cached=[]
    cached_candidates=[]
    if resume and from_step<=2 and not force_directions:
        # Recheck after acquiring the case lock: another recovery process may
        # have completed Step 2 while this command was waiting for that lock.
        from step2_local_support import circles,insertion_directions,audit,sampling_audit
        for name in active:
            try:
                circles.read(name)
                insertion_directions.read(name)
                audit.run(name)
                sampling_audit.run(name)
            except (OSError,RuntimeError,AssertionError,KeyError,ValueError):
                continue
            cached_candidates.append(name)
            print(name,'reusing completed, audited Step 1/2 after acquiring case lock',flush=True)
    for stage in STAGES[STAGES.index(starts[from_step]):]:
        number = int(stage[4])
        if number > through_step:
            break
        # Step 4/5 also produce explicit partial diagnostics after a completed,
        # unsuccessful upstream search; feasibility remains a separate verdict.
        if not active:
            break
        if resume and stage=='step3_scheculer/scheduler.py':
            for name in active:
                checkpoint=completed_schedule(name)
                if checkpoint is not None:
                    if search['mode'] in ('top5','smc'):
                        if checkpoint.get('search_config')!=search: checkpoint=None
                    elif checkpoint.get('search_mode','greedy')!='greedy':
                        checkpoint=None
                if checkpoint is not None:
                    cached.append(name)
                    if not checkpoint['continuous_coverage_proved']:blocked.append(name)
                    print(name,'reusing completed Step 3 search; verifying final output:',checkpoint['status'],flush=True)
        stage_objects=[name for name in active if (number>2 or name not in cached_candidates)
                       and (name not in cached or stage!='step3_scheculer/scheduler.py')]
        if not stage_objects:
            continue
        print(f'[{", ".join(stage_objects)}] {stage}', flush=True)
        flags=[]
        if stage=='step2_local_support/insertion_directions.py' and force_directions:
            flags.append('--force')
        if stage=='step3_scheculer/scheduler.py':
            if resume:flags.append('--resume')
            if no_round_drawings:flags.append('--no-draw')
            flags.extend(['--search-mode',search['mode'],'--trajectories',str(search['trajectories']),
                          '--seed',str(search['seed']),'--top-k',str(search['top_k']),
                          '--sizing-sweeps',str(search['sizing_sweeps']),'--sizing-budget',str(search['sizing_budget'])])
        command = [sys.executable, str(HERE/stage), *stage_objects, *flags]
        if stage == 'step6_connect_support/connect.py':
            command.extend(['--edge-budget',str(connection_edge_budget)])
        from step2_local_support.support_policy import ENFORCE_PROCESS_ACCESS
        if stage == 'step2_local_support/circles.py' and ENFORCE_PROCESS_ACCESS:
            # Reuse only current, independently audited access geometry.
            # Changed physical inputs still fail the work-volume reader.
            from step2_local_support.work_volume import WorkVolume
            from step1.cases import pose_name
            reusable = True
            for name in stage_objects:
                try:
                    WorkVolume.read(HERE/'output'/name/pose_name()/'step2_local_support/work_volume.json')
                except (OSError, RuntimeError, AssertionError, KeyError, ValueError):
                    reusable = False
            if reusable: command.append('--reuse-work-volume')
        if visibility_retry and stage == 'step2_local_support/circles.py' and ENFORCE_PROCESS_ACCESS:
            command.insert(1, str(HERE/'step2_local_support/visibility_overlay_retry.py'))
        if hull_timeout is not None and stage.startswith('step3'):
            command[1:1] = [str(HERE/'step3_scheculer/bounded_hull_retry.py'), '--hull-timeout', str(hull_timeout)]
        elif primal_first_retry and stage.startswith('step3'):
            command.insert(1, str(HERE/'step3_scheculer/primal_first_retry.py'))
        elif continuous_retry and stage.startswith('step3'):
            command.insert(1, str(HERE/'step3_scheculer/continuous_retry.py'))
        elif lp_retry and stage.startswith('step3'):
            command.insert(1, str(HERE/'step3_scheculer/numerical_retry.py'))
        import os
        if stage == 'step3_scheculer/scheduler.py' and int(os.environ.get('CADGRASP_SCORE_WORKERS', '1')) > 1:
            command.insert(1, str(HERE/'step3_scheculer/parallel_scoring.py'))
        operation = lambda: subprocess.run(command, cwd=ROOT, check=False)
        result = timings.call(stage, stage_objects, operation) if timings is not None else operation()
        if stage == 'step3_scheculer/scheduler.py' and result.returncode == 2:
            for name in stage_objects:
                try:
                    read_passed(name)
                except IncompleteSchedule:
                    blocked.append(name)
        # Exit 2 is a completed but unsuccessful search; retain its diagnostics.
        if result.returncode and not (stage=='step3_scheculer/scheduler.py' and result.returncode==2):
            raise subprocess.CalledProcessError(result.returncode, command)
    if from_step <= 4 <= through_step:
        for name in active:
            if not floor_verified(name):
                blocked.append(name)
    if from_step <= 5 <= through_step:
        for name in active:
            if not base_verified(name): blocked.append(name)
    if through_step == 6:
        for name in active:
            if not connection_verified(name):blocked.append(name)
    return 2 if blocked else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('objects', nargs='*')
    parser.add_argument('--pose', type=normalize_pose, default=None,
                        help='Target case, e.g. pose_3 or 3 (default: CADGRASP_POSE or pose_1)')
    parser.add_argument('--from-step',type=int,choices=[1,2,3,4,5,6],default=1)
    parser.add_argument('--through-step',type=int,choices=[1,2,3,4,5,6],default=None,
                        help='Last stage (default 6; one connected rigid support and explicit failure diagnostics)')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--timing-name',default='timing.json',help='Timing JSON basename in the last requested stage')
    parser.add_argument('--force-directions',action='store_true',help='Recompute Step2 directions, including when unchanged inputs permit a cache hit')
    parser.add_argument('--lp-retry',action='store_true',help='Record an additional original-equation IPM retry if the usual solver reports an unresolved error')
    parser.add_argument('--continuous-retry',action='store_true',help='Also try direct primal enclosure certificates if an optional dual-hull test is inconclusive')
    parser.add_argument('--primal-first-retry',action='store_true',help='Try existing primal certificates before the optional dual hull; preserve original fallback and audit')
    parser.add_argument('--hull-timeout',type=float,default=None,help='Optional seconds per isolated dual-hull call after primal-first checks; timeout is unresolved; requires a fresh run')
    parser.add_argument('--visibility-retry',action='store_true',help='Use a conservative possible-shadow union recovery in Step 2')
    parser.add_argument('--no-round-drawings',action='store_true',help='Keep Step 4/5 figures; skip per-round selection/size figures')
    parser.add_argument('--search-mode',choices=['top5','greedy','smc'],default='top5')
    parser.add_argument('--trajectories',type=int,default=10)
    parser.add_argument('--search-seed',type=int,default=0)
    parser.add_argument('--top-k',type=int,default=5)
    parser.add_argument('--sizing-sweeps',type=int,default=2)
    parser.add_argument('--sizing-budget',type=int,default=96)
    parser.add_argument('--connection-edge-budget',type=int,default=2000,
                        help='Maximum new edge checks for the whole connected support (default 2000); exhaustion is inconclusive')
    parser.add_argument('--connection-workers',type=int,default=1,
                        help='Legacy option; one-body construction is sequential within each case')
    args = parser.parse_args()
    from step3_scheculer.random_search import configuration
    configuration(args.search_mode,args.trajectories,args.search_seed,args.top_k,args.sizing_sweeps,args.sizing_budget)
    if Path(args.timing_name).name != args.timing_name or not args.timing_name.startswith('timing') or not args.timing_name.endswith('.json'):
        parser.error('--timing-name must be a JSON basename starting with timing')
    from step1.registry import active_objects
    objects = args.objects or active_objects()
    if any(name not in active_objects() for name in objects):
        parser.error('objects must be active in objects/cases.json')
    if args.through_step is not None and args.through_step < args.from_step:
        parser.error('--through-step must be at least --from-step')
    if args.connection_edge_budget < 1:
        parser.error('--connection-edge-budget must be positive')
    if args.connection_workers < 1:
        parser.error('--connection-workers must be positive')
    if args.hull_timeout is not None and (not math.isfinite(args.hull_timeout) or args.hull_timeout <= 0 or args.resume):
        parser.error('--hull-timeout must be finite and positive and requires a fresh run (no --resume)')
    # Serialize writers to each case; separate object/pose cases remain parallel.
    from step1.cases import pose_name
    with selected_pose(args.pose), ExitStack() as locks:
        for name in sorted(objects):
            folder=HERE/'output'/name/pose_name();folder.mkdir(parents=True,exist_ok=True)
            handle=locks.enter_context((folder/'.pipeline.lock').open('a'))
            fcntl.flock(handle,fcntl.LOCK_EX)
        from step3_scheculer.timing import StageTimings
        import os
        last = args.through_step or 6
        directory = {1:'step_1_needs',2:'step2_local_support',3:'step3_scheculer',4:'step4_floor_contact',5:'step5_base',6:'step6_connect_support'}[last]
        timings = StageTimings([HERE/'output'/name/pose_name()/directory/args.timing_name for name in objects],
            dict(command=sys.argv, from_step=args.from_step, through_step=last, resume=args.resume, force_directions=args.force_directions,
                 environment={k:v for k,v in os.environ.items() if k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','CADGRASP_SCORE_WORKERS','CADGRASP_DIRECTION_WORKERS')}))
        try:
            code=run(objects,args.from_step,args.resume,args.no_round_drawings,args.pose,args.through_step,args.lp_retry,args.continuous_retry,args.visibility_retry,args.connection_edge_budget,args.connection_workers,timings,args.force_directions,args.search_mode,args.trajectories,args.search_seed,args.top_k,args.sizing_sweeps,args.sizing_budget,args.primal_first_retry,args.hull_timeout)
        except BaseException as error:
            timings.finish(error=error)
            raise
        timings.finish(returncode=code)
    sys.exit(code)
