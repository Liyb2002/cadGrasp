"""Continue an unresolved large whole set from its OWN saved state.

An actual counterexample may conservatively exclude optimistic contact samples
from SEARCH GUIDANCE. Final exact physics and original load data are unchanged.
Old attempts are archived, and every extra search/evaluation is recorded.
"""
import _bootstrap
import argparse
import contextlib
import multiprocessing
import os
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from co_common import *
from run_all import output_root
from run_large_pose_sets import groups, record_phase, run_case as finish_case
from whole_pipeline import WholeSearch, load_layout, stage_record, SCHEMA
from whole_search.fast_search import FastModel
from whole_search.exact_worker import read_result
from whole_search.search import failed_count
from whole_step4_render import render_search


def excluded_samples(model, actual):
    """Find nominal query points absent from the actual contact triangles."""
    flags=model.contact_flags(actual['layout']); result={}
    for k in actual['layout'].active:
        if actual['masks'][k].all(): continue
        points=transform_points(model.points,actual['layout'].placements[k])
        triangles=actual['contacts'][k]['triangles']; sources=actual['contacts'][k]['sources']
        unsafe=[]
        for index in np.flatnonzero(flags[k]):
            tri=triangles[sources==model.sources[index]]
            if not len(tri): unsafe.append(int(index));continue
            # Signed edge distances avoid barycentric denominator underflow
            # on tiny clipped contact triangles. Tolerance is in metres.
            normal=model.point_normals[index] @ actual['layout'].placements[k,:3,:3].T
            edge=np.roll(tri,-1,axis=1)-tri
            length=np.linalg.norm(edge,axis=2)
            signed=np.einsum('tvc,c->tv',np.cross(edge,points[index]-tri),normal)
            inside=(signed>=-1e-10*length).all(axis=1)
            if not inside.any(): unsafe.append(int(index))
        if unsafe: result[k]=unsafe
    return result


class FeedbackModel(FastModel):
    def apply_feedback(self, unsafe):
        self.unsafe_samples=unsafe
        # Computing the counterexample itself queried the old availability.
        # Its cached state must not mask the newly excluded samples.
        from whole_search.delta_guidance import ContactDelta
        self.contact_delta=ContactDelta(self)
        for cache in [self.sample_cache,self.force_cache,self.proxy_force_cache,
                      self.worst_cache,self.complete_sampled_states]:
            cache.clear()

    def contact_allowed(self, layout, owner):
        allowed=super().contact_allowed(layout,owner)
        if owner in getattr(self,'unsafe_samples',{}):
            allowed[self.unsafe_samples[owner]]=False
        return allowed


def run_case(arguments):
    group,options=arguments;root=output_root('B');base=root/group['id'];out=base/'step4/step4.2'
    if (out/'data/report.json').exists():
        return finish_case((group,dict(methods=['greedy','beam'],audit=True)))
    began=time.monotonic();stamp=time.strftime('%Y%m%d_%H%M%S')
    archive=base/'step4/_history'/('before_large_continuation_'+stamp)
    shutil.copytree(out,archive)
    logpath=out/'data'/('continuation_'+stamp+'.log');logpath.parent.mkdir(parents=True,exist_ok=True)
    token=json.loads((root/'data/whole_step4_active_run.json').read_text())['run_token']
    os.environ['COOPT_WHOLE_RUN_TOKEN']=token
    request=dict(complete=False,group=group,source_layout_sha256=I.sha256(archive/'sampled_layout.npz'),
        own_saved_state_only=True,all_poses_active=True,additional_budget=options,
        original_attempt_archive=str(archive.relative_to(base)),original_work_and_force_model_unchanged=True)
    save(out/'data'/('continuation_'+stamp+'.json'),request)
    with logpath.open('w',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            source=base/'step3/step3.1/data/report.json';initial=I.check_report(base/'step4/step4.1/data/report.json')
            model=FeedbackModel(group['poses'],'B',initialization_report=source)
            layout=load_layout(archive/'sampled_layout.npz'); learned={};actual_source=None
            # These results were completed even though some loads failed.
            # They supply geometric counterexamples, not feasible incumbents.
            actual_paths=sorted(archive.glob('data/validation_continuations/*/result.npz'))
            for path in reversed(actual_paths):
                actual=read_result(path.with_suffix(''),1)
                if failed_count(actual):
                    learned=excluded_samples(model,actual);layout=actual['layout'];actual_source=path;break
            model.apply_feedback(learned)
            model.worker_program=HERE/'helper_func/whole_search/exact_worker.py'
            # Avoid overwriting previous isolated evaluation logs.
            model.checkpoint_dir=out/'data'/('continuation_exact_'+stamp)/'states'
            search=WholeSearch(model,out,iterations=options['iterations'],finalists=options['finalists'],
                branch_rounds=options['branch_rounds'],seed=options['seed'])
            search.screen_budget=options['screen_budget'];search.capture_process=True
            search.final_candidates=options['real_budget']
            current=search.initialize_exact(layout,'own_saved_continuation')
            search.baseline(current)
            request.update(counterexample_sample_exclusions={model.poses[k]:v for k,v in learned.items()},
                actual_counterexample_source=str(actual_source.relative_to(base)) if actual_source else None,
                start_counts=current['counts'])
            print('CONTINUE',group['id'],'start failed',failed_count(current),'unsafe',sum(map(len,learned.values())),flush=True)
            if failed_count(current):current=search.fine_refine(current,rounds=10)
            if failed_count(current):current=search.solve_frontier(current,anchor=None)
            current=search.restore_reuse(current)
            current=search.polish_volume(current,options['volume_rounds'])
            search.last_sampled_result=current
            # A known failed sampled state cannot become an actual solution;
            # do not spend a Boolean evaluation just to establish that again.
            search.search_only=failed_count(current)>0
            report=search.save(current,'whole',began,initial['initialization'],
                initial_counts=initial['counts'],fixture_placement_count_is_not_cost=True,
                all_pose_juxtapose_disabled=True,separated_fallback_used=False,
                continuation_request=request,search_sample_exclusions_are_guidance_only=True)
            if failed_count(current):
                render_search(model,out,None)
                raise RuntimeError('Own-state continuation still has '+str(failed_count(current))+' sampled failed loads')
            render_search(model,out,report)
            report.update(status='pass',passed=True,step41_volume_cm3=initial['volume_cm3'],step41_counts=initial['counts'])
            report['provenance']['code'].update(I.hashes([Path(__file__)]))
            stage_record(out,report,group,'step4.2')
            request.update(complete=True,passed=True,seconds=time.monotonic()-began,
                final_counts=report['counts'],volume_cm3=report['volume_cm3'],
                additional_full_load_evaluations=model.sample_calls,additional_actual_evaluations=model.exact_calls)
            save(out/'data'/('continuation_'+stamp+'.json'),request)
            save(base/'step4/data/report.json',dict(complete=True,schema=SCHEMA,id=group['id'],
                poses=group['poses'],status='pass',passed=True,final_volume_cm3=report['volume_cm3'],
                step41_force_passed=initial['force_and_path_initialization_gate'],
                separately_recorded_search_continuation=True,seconds=time.monotonic()-began))
            (base/'step4/data/failure.json').unlink(missing_ok=True)
        except Exception as error:
            traceback.print_exc();request.update(complete=True,passed=False,error=str(error),seconds=time.monotonic()-began)
            save(out/'data'/('continuation_'+stamp+'.json'),request)
            record_phase(group,'search_continuation',dict(status='unresolved',passed=False,error=str(error)))
            return dict(id=group['id'],status='unresolved',error=str(error))
    record_phase(group,'search_continuation',dict(status='pass',passed=True,volume_cm3=report['volume_cm3']))
    return finish_case((group,dict(methods=['greedy','beam'],audit=True)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--sets',nargs='+',required=True)
    parser.add_argument('--jobs',type=int,default=2);parser.add_argument('--iterations',type=int,default=12)
    parser.add_argument('--finalists',type=int,default=6);parser.add_argument('--branch-rounds',type=int,default=4)
    parser.add_argument('--volume-rounds',type=int,default=3);parser.add_argument('--screen-budget',type=int,default=128)
    parser.add_argument('--real-budget',type=int,default=3);parser.add_argument('--seed',type=int,default=43)
    args=parser.parse_args();options=vars(args).copy();options.pop('sets');options.pop('jobs')
    selected=[g for g in groups() if g['id'] in args.sets or g['source_case'] in args.sets]
    if not selected:raise ValueError('No matching requested cases')
    with ProcessPoolExecutor(max_workers=args.jobs,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1) as pool:
        for future in as_completed([pool.submit(run_case,(g,options)) for g in selected]):
            print('CONTINUED_CASE',future.result(),flush=True)


if __name__=='__main__':main()
