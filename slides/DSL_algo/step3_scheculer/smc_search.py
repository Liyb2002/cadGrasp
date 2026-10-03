"""SMC-inspired population optimization; no posterior or global-optimality claim.

Particles are immutable partial contact designs. Each tree node evaluates only
one additional contact. Resampling redistributes a fixed next-layer slot budget;
verified designs enter an archive and never receive more contacts.
"""
import copy
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from step3_scheculer import contacts as I, paths as PTH, random_search as R

POLICY = dict(name='smc_inspired_optimization', elite_slots=1, exploration_fraction=.2,
    ranking='absolute coverage on common original loads descending, then actual area ascending',
    weights='linear rank weights: best/worst ratio 2 after layer 1, 4 after layer 2',
    proposal='marginal-coverage top-k among candidates not already tried by a sibling',
    limitation='Finite stochastic heuristic; not a posterior sampler or global optimality certificate')


def resampling_weights(results, depth):
    """Gentle ordinal weights; counterexamples and marginal gain never enter."""
    if not results:
        return np.empty(0)
    totals = {r['random_sample_count'] for r in results}
    if len(totals) != 1:
        raise ValueError('Particles must share the same original load set')
    keys = [(-r['random_covered_count'], r['rounds'][-1]['total_area_m2']) for r in results]
    levels = sorted(set(keys))
    ranks = {key: len(levels)-1-i for i, key in enumerate(levels)}
    strength = 1. if depth == 1 else 3.
    weights = np.array([1+strength*ranks[k]/max(1,len(levels)-1) for k in keys])
    return weights/weights.sum()


def resample(ids, results, count, rng, depth):
    weights = resampling_weights(results, depth)
    elite = min(range(len(results)), key=lambda i: (*R.result_key(results[i]), ids[i]))
    exploration = min(count-1, max(1, int(np.ceil(.2*count)))) if count > 1 else 0
    chosen = [ids[elite]]
    kinds = ['elite']
    for _ in range(exploration):
        chosen.append(ids[int(rng.integers(len(ids)))])
        kinds.append('uniform_exploration')
    for _ in range(count-len(chosen)):
        chosen.append(ids[int(rng.choice(len(ids),p=weights))])
        kinds.append('weighted')
    return dict(depth=depth, eligible_particle_ids=ids, weights=weights.tolist(),
        effective_sample_size=float(1/np.dot(weights,weights)),
        selected_parent_ids=chosen, slot_kinds=kinds)


def reuse_score(name, number, problem, source):
    """Only called for siblings of the exact same immutable parent state."""
    from step3_scheculer import scheduler as S
    tick=time.monotonic()
    report=copy.deepcopy(I.check_report(source/'contributions.json'))
    assert report['object']==name and report['round']==number
    assert report['sample_count']==len(problem.targets)
    out=PTH.folder(name,S.C.OUTPUT_NAME,number)
    out.mkdir(parents=True,exist_ok=True)
    for filename in report['artifacts']:
        shutil.copyfile(source/filename,out/filename)
    report['provenance']['inputs'].update(I.hashes(problem.inputs+[source/'contributions.json']))
    report['provenance']['code'].update(I.hashes([Path(__file__)]))
    report.update(reused_from=str(source.relative_to(I.ROOT)),
        reused_scoring_seconds=report['elapsed_seconds'],elapsed_seconds=time.monotonic()-tick)
    I.save(out/'contributions.json',report)
    I.save(out/'status.json',dict(object=name,complete=True,status='scored',
        contributions_sha256=I.sha256(out/'contributions.json')))
    return report


def node_summary(index, parent, seed, result, folder, excluded, evaluations):
    return dict(index=index,parent_id=parent,seed=seed,depth=len(result['rounds']),
        status=result['status'],continuous_coverage_proved=result['continuous_coverage_proved'],
        random_sample_count=result['random_sample_count'],random_covered_count=result['random_covered_count'],
        total_area_m2=result['rounds'][-1]['total_area_m2'] if result['rounds'] else 0.,
        excluded_sibling_candidates=list(excluded),evaluations=evaluations,
        elapsed_seconds=result['elapsed_seconds'],schedule=str((folder/'schedule.json').relative_to(I.ROOT)))


def run(name, config, draw=True, resume=False):
    from step3_scheculer import scheduler as S
    started=time.monotonic()
    out=PTH.stage_folder(name,S.OUTPUT_NAME)
    out.mkdir(parents=True,exist_ok=True)
    if resume:
        try:
            saved=I.check_report(out/'schedule.json')
            assert saved['search_mode']=='smc' and saved['search_config']==config
            if draw:
                for node in saved['particles']:
                    r=I.check_report(I.ROOT/node['schedule'])
                    with PTH.trajectory(node['index']),PTH.round_owners(r['round_trajectories']):
                        for row in r['rounds']:
                            assert (PTH.folder(name,S.A.OUTPUT_NAME,row['round'])/'adjustment_views.json').is_file()
                            assert (PTH.folder(name,S.M.OUTPUT_NAME,row['round'])/'selection_views.json').is_file()
            print(name,'reusing completed SMC population',flush=True)
            return saved
        except (OSError,RuntimeError,AssertionError,KeyError,ValueError):
            pass
    I.save(out/'status.json',dict(object=name,complete=False,status='searching_smc_population'))
    results, folders, summaries, layers, archive = [],[],[],[],[]
    parents=[None]*config['trajectories']
    rng=np.random.default_rng(R.chain_seed(config['seed'],2**31))
    try:
        for depth in range(1,S.Q.MAX_CONTACTS+1):
            active=[]
            score_cache={}
            sibling_choices={}
            evaluated_ids=[]
            skipped=[]
            for slot,parent in enumerate(parents):
                prefix=results[parent] if parent is not None else None
                excluded=sibling_choices.setdefault(parent,[])
                source=score_cache.get(parent)
                if source is not None:
                    candidates=S.M.rank_candidates(I.check_report(source/'contributions.json')['contributions'])
                    if candidates and not set(candidates)-set(excluded):
                        skipped.append(dict(slot=slot,parent_id=parent,reason='all_sibling_candidates_already_evaluated'))
                        continue
                    if not candidates:
                        skipped.append(dict(slot=slot,parent_id=parent,reason='parent_has_no_eligible_candidates'))
                        continue
                index=len(results)
                seed=R.chain_seed(config['seed'],index)
                owners=dict(prefix['round_trajectories']) if prefix else {}
                owners[str(depth)]=index
                print(name,'SMC layer',depth,'slot',slot+1,'/',len(parents),'particle',index,'parent',parent,flush=True)
                with PTH.trajectory(index),PTH.round_owners(owners):
                    folder=PTH.stage_folder(name,S.OUTPUT_NAME)
                    result=S.run_chain(name,draw=draw,resume=False,seed=seed,top_k=config['top_k'],
                        sizing_sweeps=config['sizing_sweeps'],sizing_budget=config['sizing_budget'],
                        prefix=prefix,stop_after_round=depth,particle_id=index,
                        excluded_indices=tuple(excluded),score_source=source,population_seed=config['seed'])
                    score_folder=PTH.folder(name,S.C.OUTPUT_NAME,depth)
                    score=I.check_report(score_folder/'contributions.json')
                    score_cache.setdefault(parent,score_folder)
                    evaluations=dict(scoring_rounds=int(source is None),
                        scored_candidate_combinations=sum(r['status']=='scored_joint' for r in score['contributions']) if source is None else 0,
                        added_contact_optimizations=int(len(result['rounds'])==depth),
                        sizing_coverage_evaluations=0,continuous_checks=0)
                    if len(result['rounds'])==depth:
                        adjusted=S.A.read(name,depth)
                        evaluations['sizing_coverage_evaluations']=adjusted['selection']['evaluated_sizes']
                        evaluations['continuous_checks']=int('continuous_validation' in result['rounds'][-1])
                    summary=node_summary(index,parent,seed,result,folder,excluded,evaluations)
                    if len(result['rounds'])==depth:
                        excluded.append(result['rounds'][-1]['candidate_index'])
                results.append(result);folders.append(folder);summaries.append(summary);evaluated_ids.append(index)
                if result['continuous_coverage_proved']:
                    archive.append(index)
                elif result['status']=='population_partial':
                    active.append(index)
                I.save(out/'population_progress.json',dict(object=name,complete=False,config=config,
                    particles=summaries,verified_archive=archive,layers=layers,elapsed_seconds=time.monotonic()-started))
            layer=dict(depth=depth,requested_slots=len(parents),evaluated_particle_ids=evaluated_ids,
                skipped_slots=skipped,active_particle_ids=active,verified_archive=list(archive))
            if depth<S.Q.MAX_CONTACTS and active:
                record=resample(active,[results[i] for i in active],config['trajectories'],rng,depth)
                layer['resampling']=record
                parents=record['selected_parent_ids']
            else:
                parents=[]
            layers.append(layer)
            if not parents:
                break
        winner=min(range(len(results)),key=lambda i:(*R.result_key(results[i]),i))
        result=copy.deepcopy(results[winner]);source=folders[winner]
        shutil.copyfile(source/'final_contacts.npz',out/'final_contacts.npz')
        directions=copy.deepcopy(I.check_report(source/'insertion_directions.json'))
        directions['actual_contacts_file']=str((out/'final_contacts.npz').relative_to(I.ROOT))
        directions['provenance']['inputs'].update(I.hashes([source/'insertion_directions.json',out/'final_contacts.npz']))
        I.save(out/'insertion_directions.json',directions)
        counts={key:sum(row['evaluations'][key] for row in summaries) for key in summaries[0]['evaluations']}
        result['selected_particle_status']=result['status']
        if result['status']=='population_partial':
            result['status']='population_search_exhausted'
        result.update(search_mode='smc',search_config=config,smc_policy=POLICY,
            selected_trajectory=winner,particles=summaries,layers=layers,verified_archive=archive,
            particle_evaluations=len(results),evaluation_counts=counts,
            evaluation_count_scope='Explicit candidate classifications, added-contact optimizations, sizing coverage calls and continuous checks; not internal LP or geometry call counts.',
            resampling_seed=R.chain_seed(config['seed'],2**31),
            elapsed_seconds=time.monotonic()-started,
            wall_timings=[dict(particle=i,**row) for i,r in enumerate(results) for row in r['wall_timings']],
            insertion_direction_record=str((out/'insertion_directions.json').relative_to(I.ROOT)))
        result['provenance']['inputs'].update(I.hashes([p/'schedule.json' for p in folders]))
        result['provenance']['code'].update(I.hashes([Path(__file__)]))
        result['artifacts']={os.path.relpath(source/f,out):digest for f,digest in result['artifacts'].items()}
        result['artifacts'].update({f:I.sha256(out/f) for f in ['final_contacts.npz','insertion_directions.json']})
        I.save(out/'schedule.json',result)
        I.save(out/'status.json',dict(object=name,complete=True,status=result['status'],
            continuous_domain_status=result['continuous_domain_status'],physical_supports_verified=False,
            schedule_sha256=I.sha256(out/'schedule.json')))
        I.save(out/'population_progress.json',dict(object=name,complete=True,config=config,
            particles=summaries,verified_archive=archive,layers=layers,elapsed_seconds=result['elapsed_seconds']))
        print(name,'SMC selected particle',winner,'/',len(results),result['status'],counts,flush=True)
        return result
    except Exception as error:
        I.save(out/'status.json',dict(object=name,complete=False,status='search_error',
            error_type=type(error).__name__,error=str(error)))
        raise
