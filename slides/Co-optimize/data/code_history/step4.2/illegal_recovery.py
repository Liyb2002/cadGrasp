"""Apply the existing fixed-position recovery to the separate illegal-set batch."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from co_common import *
from run_illegal import SOURCE
import step33,step41,step42_contact_consistency as contact_context
from step42_contact_consistency import motion_consistent_boundary
from step42_timed import TimeLimitedSearch
from step42_dispatch import SmallConnectedSearch
from step42_connected import finish_connected
import step42 as force_module,step42_connected as connected_module
import step42_geometry_cache as cache
import argparse,time,types,contextlib,multiprocessing,signal,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed

class SearchBudget(BaseException):pass

class BoundedForce(TimeLimitedSearch):
    def candidate(self,directions,*args,**kwargs):
        if time.monotonic()>self.stop:raise SearchBudget()
        contact_context._current_directions=np.asarray(directions)
        force_module.contact_boundary=motion_consistent_boundary
        try:return super().candidate(directions,*args,**kwargs)
        except (RuntimeError,ValueError,np.linalg.LinAlgError):return None

class BoundedConnected(SmallConnectedSearch):
    def candidate(self,*args,**kwargs):
        if time.monotonic()>self.stop:raise SearchBudget()
        return super().candidate(*args,**kwargs)

def prepare_alias():
    # The unmodified Step4.1 function hardcodes its cache root. Give that
    # function an isolated output alias, keeping ROOT/native inputs unchanged.
    alias=HERE/'output/B/illegal/data/runtime';alias.mkdir(parents=True,exist_ok=True)
    for name in ['helper_func','vis_func','step3.1','step3.2','step3.3','step4.1','step4.2']:
        path=alias/name
        if not path.exists():path.symlink_to(HERE/name,target_is_directory=True)
    target=alias/'output/B';target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():target.symlink_to(HERE/'output/B/illegal',target_is_directory=True)
    return alias

def initialize(original,alias):
    name='B';group=dict(original);folder=HERE/'output/B/illegal/data/step41';folder.mkdir(parents=True,exist_ok=True)
    seedpath=HERE/'output/B/illegal'/group['id']/'step3/step3.3/support_with_rings.obj'
    support=trimesh.load(seedpath,force='mesh',process=False)
    paths={};length=.5;states={p:state(name,p) for p in group['poses']}
    for pose,(task,T,mesh) in states.items():
        direction,native=step41.initialize_direction(T)
        length=max(length,float((support.vertices@direction).max()-(mesh.vertices@direction).min())+.02)
        paths[pose]=dict(direction_fixture=direction.tolist(),direction_world=native.tolist())
    # Each set owns its sweep cache, so concurrent jobs never overwrite inputs.
    local=folder/group['id'];local.mkdir(parents=True,exist_ok=True)
    for pose,(task,T,mesh) in states.items():
        path=local/pose;path.mkdir(exist_ok=True);direction=np.array(paths[pose]['direction_fixture'])
        for kind,distance in [('full',length),('display',.10)]:
            D.export_exact_obj(S.swept_solid(mesh,distance*direction),path/f'{kind}_sweep.obj')
    initialization=dict(complete=True,paths=paths,full_length_m=length,display_length_m=.10,initialization_kind='Each pose withdraws along its own native world +z',provenance=provenance([SOURCE,seedpath],[HERE/'step4.2/illegal_recovery.py',HERE/'step4.1/step41.py']))
    save(local/'initialization.json',initialization)
    # Substitute only the literal cache-folder expression in a copied function;
    # all construction, rendering and force checks stay those of Step4.1.
    import inspect
    source=inspect.getsource(step41.run)
    source=source.replace("folder=(HERE/'output'/name if name=='B' else HERE/'data/object_inputs'/name)/'data/step41'",'folder=illegal_cache_folder')
    globals_copy=dict(step41.run.__globals__,HERE=alias,illegal_cache_folder=local)
    exec(compile(source,str(HERE/'step4.2/illegal_recovery.py')+'::step41_adapter','exec'),globals_copy)
    globals_copy['run'](name,group,initialization)
    reportpath=HERE/'output/B/illegal'/group['id']/'step4/step4.1/data/report.json'
    report=json.loads(reportpath.read_text());report['provenance']['code'].update(I.hashes([HERE/'step4.2/illegal_recovery.py']));save(reportpath,report);I.check_report(reportpath)

def recover(arguments):
    original,seconds=arguments;group=dict(original,id='illegal/'+original['id']);base=HERE/'output/B'/group['id'];(base/'data').mkdir(exist_ok=True)
    with (base/'data/recovery.log').open('w',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        began=time.monotonic();row=dict(id=original['id'],original_floor_violation_count=original['total_directed_violating_sample_count'],passed=False,full_fixture_accepted=False)
        try:
            gate=I.check_report(base/'step3/step3.2/data/report.json')
            if not gate['passed']:row.update(status='stopped_at_wrap_gate');return row
            ring=step33.run('B',group)
            if not ring['passed']:row.update(status='ring_geometry_unresolved');return row
            initialize(original,prepare_alias());cache.install()
            search=BoundedForce('B',group);search.stop=time.monotonic()+seconds
            result=None
            try:result=search.search()
            except SearchBudget:pass
            row.update(force_proposals=search.evaluations,force_best_counts=None if search.best is None else search.best['counts'])
            save(search.out/'data/force_search.json',dict(complete=result is not None,seconds_budget=seconds,proposals=search.evaluations,best_counts=row['force_best_counts'],trace=search.trace))
            if result is None:
                row.update(status='force_recovery_search_unresolved');return row
            force_module.finish(search,result,began)
            reportpath=search.out/'data/report.json';report=json.loads(reportpath.read_text())
            report.update(passed=False,force_exit_passed=True,connectivity_required=True,status='force_exit_pass_connectivity_pending')
            report['provenance']['inputs'].update(I.hashes([SOURCE]));report['provenance']['code'].update(I.hashes([HERE/'step4.2/illegal_recovery.py',HERE/'step4.2/step42_contact_consistency.py',HERE/'step4.2/step42_timed.py',HERE/'helper_func/step42_geometry_cache.py']));save(reportpath,report);I.check_report(reportpath)
            connected=BoundedConnected('B',group);connected.stop=time.monotonic()+seconds
            result=None
            try:result=connected.connected_search()
            except SearchBudget:pass
            row.update(connected_proposals=connected.evaluations,force_exit_passed=True)
            if result is None:
                row.update(status='force_exit_pass_connectivity_unresolved',component_count=report['remaining_component_count']);return row
            finish_connected(connected,result,began)
            report=I.check_report(reportpath);report['motion_consistent_contacts_required']=True
            report['provenance']['inputs'].update(I.hashes([SOURCE]));report['provenance']['code'].update(I.hashes([HERE/f for f in ['step4.2/illegal_recovery.py','step4.2/step42_connected_safe.py','step4.2/step42_connected_local.py','step4.2/step42_connected_independent.py','step4.2/step42_contact_consistency.py','helper_func/step42_geometry_cache.py','step4.2/step42_dispatch.py']]))
            save(reportpath,report);I.check_report(reportpath)
            # Start with the overview canvas; the standard renderer retains size.
            import shutil
            shutil.copy2(search.out/'overview.png',search.out/'exit_motion.png')
            from render_exit_motion import render
            render(group)
            row.update(status='force_exit_and_connectivity_pass',passed=True,component_count=1)
            return row
        except Exception as error:
            traceback.print_exc();row.update(status='runtime_unresolved',error=str(error));return row
        finally:
            signal.setitimer(signal.ITIMER_REAL,0.);row['seconds']=time.monotonic()-began
            save(base/'data/result.json',dict(complete=True,**row,provenance=provenance([SOURCE],[HERE/'step4.2/illegal_recovery.py'])))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--jobs',type=int,default=2);parser.add_argument('--seconds-per-phase',type=float,default=180);parser.add_argument('--sets',nargs='+');args=parser.parse_args()
    groups=json.loads(SOURCE.read_text())['sets'];groups=[g for g in groups if not args.sets or g['id'] in args.sets];prepare_alias();rows=[]
    with ProcessPoolExecutor(max_workers=args.jobs,mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(recover,(g,args.seconds_per_phase)) for g in groups]):
            row=future.result();rows.append(row);print('ILLEGAL RECOVERY',row,flush=True)
    order={g['id']:i for i,g in enumerate(groups)};rows.sort(key=lambda r:order[r['id']])
    save(HERE/'output/B/illegal/data/recovery_batch.json',dict(complete=True,sets=len(rows),passed_sets=sum(r['passed'] for r in rows),seconds_per_phase=args.seconds_per_phase,results=rows,provenance=provenance([SOURCE],[HERE/'step4.2/illegal_recovery.py'])))

if __name__=='__main__':main()
