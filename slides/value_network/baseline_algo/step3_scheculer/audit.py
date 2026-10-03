"""Audit every round's rankings, fixed geometry and post-optimization state."""
import argparse
from pathlib import Path
import sys
import json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from step3_scheculer import scheduler as S
from step3_scheculer.stage_imports import load_stage
C = load_stage('score', 'contribution')
M = load_stage('select', 'choose')
A = load_stage('optimize', 'adjust')
from step3_scheculer import contacts as I
from step3_scheculer import paths as PTH
from step2_local_support import withdrawal as D
from step3_scheculer import directions as T
from step3_scheculer import connection as B
from step2_local_support import insertion_directions as K

from step1.cases import pose_name


def run(name, check_drawings=True, audited_rounds=None):
    out=PTH.stage_folder(name,S.OUTPUT_NAME)
    schedule=I.check_report(out/'schedule.json')
    if schedule.get('search_mode') == 'top5_independent':
        return audit_ensemble(name, schedule, out, check_drawings)
    if schedule.get('search_mode') == 'smc':
        return audit_population(name, schedule, out, check_drawings)
    with PTH.round_owners(schedule.get('round_trajectories', {})):
        return audit_chain(name, schedule, out, check_drawings, audited_rounds)


def audit_chain(name, schedule, out, check_drawings, audited_rounds=None):
    joint = schedule.get('search_mode') in ('top5', 'smc_particle')
    rng = np.random.default_rng(schedule['search_config']['seed']) if joint else None
    assert schedule['round_limit']==S.Q.MAX_CONTACTS
    assert len(schedule['rounds'])==schedule['contact_count']<=S.Q.MAX_CONTACTS
    assert schedule['passive_support_constraint']==C.U.description()
    assert schedule['passive_support_no_uplift_verified']==(schedule['status']=='continuous_contact_model_verified')
    problem=C.Problem(name)
    initial_areas=T.candidate_areas(problem)
    minimum_area=A.area_limit.MIN_AREA_FRACTION*float(problem.domain.mesh.area)
    assert schedule['minimum_contact_area_m2']==minimum_area
    assert schedule['all_contact_areas_above_minimum']
    assert schedule['process_access_enforced']==A.P.POLICY.ENFORCE_PROCESS_ACCESS
    assert schedule['all_contact_heads_clear_of_work_volume'] if A.P.POLICY.ENFORCE_PROCESS_ACCESS else schedule['all_contact_heads_clear_of_work_volume'] is None
    catalogue=K.read(name)
    direction_catalogue=catalogue['direction_catalogue']
    connection_checker=B.for_problem(problem,catalogue)
    common_allowed=direction_catalogue['global_allowed_directions']
    direction_analyzer=D.Analyzer(problem.domain.mesh,catalogue['normal_depth_m'],direction_catalogue)
    work=A.W.WorkVolume.read(C.OUTPUTS/name/pose_name()/A.W.STAGE/'work_volume.json') if A.P.POLICY.ENFORCE_PROCESS_ACCESS else None
    work_checker=A.WC.ContactClearance(problem.domain.mesh,catalogue['normal_depth_m'],work)
    fixed=[]
    reports=[]
    replayed_head_rays=set()
    head_geometry_cache={}
    for entry in schedule['rounds']:
        number=entry['round']
        owner_out=PTH.owner_stage(name,S.OUTPUT_NAME,number)
        round_key=(str(owner_out),number)
        if audited_rounds is not None and round_key in audited_rounds:
            # Already replayed this immutable ancestor once; reconstruct only the
            # state needed for its child's independent new-round audit.
            fixed=I.read_contacts(PTH.folder(name,A.OUTPUT_NAME,number)/'contacts.npz')
            d=I.check_report(PTH.folder(name,A.OUTPUT_NAME,number)/'insertion_directions.json')
            common_allowed=d['common_directions']
            if entry.get('continuous_validation')=='counterexample':
                validation=json.loads((owner_out/'verification'/f'round_{number:03d}_continuous.json').read_text())
                problem.add_counterexample(validation['counterexample'],
                    owner_out/'search_loads'/f'after_round_{number:03d}.json',write=False)
            reports.append(dict(round=number,reused_audited_ancestor=str(owner_out.relative_to(I.ROOT))))
            continue
        score=C.read(name,number)
        selected=M.read(name,number)
        adjusted=A.read(name,number)
        assert score['passive_support_constraint']==C.U.description()
        assert adjusted['passive_support_constraint']==C.U.description()
        score_folder=PTH.folder(name,C.OUTPUT_NAME,number)
        select_folder=PTH.folder(name,M.OUTPUT_NAME,number)
        adjust_folder=PTH.folder(name,A.OUTPUT_NAME,number)
        masks=I.load_npz(score_folder/'sample_coverage.npz')
        rows=score['contributions']
        assert score['selected_indices']==sorted(p['candidate_index'] for p in fixed)
        filtering=I.check_report(PTH.folder(name,S.OUTPUT_NAME,number)/'candidate_filter.json')
        assert filtering['mode']==D.MODE
        expected_eligible,expected_rows=T.candidate_filter(catalogue['candidates'],
            [p['candidate_index'] for p in fixed],problem.data.valid,initial_areas,minimum_area,common_allowed,
            connection_checker=connection_checker,selected_contacts=fixed,candidate=problem.candidate)
        assert filtering['candidates']==expected_rows
        np.testing.assert_array_equal(filtering['eligible'],expected_eligible)
        np.testing.assert_array_equal(masks['insertion_eligible'],expected_eligible)
        for index,row in enumerate(rows):
            if expected_eligible[index]:
                trial=problem.supply(fixed+[problem.candidate(index)])
                hard=C.gravity_check(trial,problem.domain,problem.scale)
                assert row['hard_feasibility']['passed']==hard['passed']
                if not hard['passed']:
                    assert row['status']=='rejected_rest_equilibrium'
                    assert row['covered_count'] is None and not masks['covered'][index].any()
                    expected_eligible[index]=False
                    continue
                assert row['covered_count']==int(masks['covered'][index].sum())
                assert row['covered_percent']==100*row['covered_count']/len(problem.targets)
                assert not np.any(masks['base'] & ~masks['covered'][index])
            else:
                assert row['covered_count'] is None and not masks['covered'][index].any()
        np.testing.assert_array_equal(masks['eligible'],expected_eligible)
        order=M.rank_candidates(rows)
        assert selected['ranking']==order
        chosen=selected['winner']['index']
        if joint:
            sampling=selected['sampling']
            assert sampling['top_k']==schedule['search_config']['top_k']
            indices,probabilities=M.top_k_distribution(rows,int(masks['base'].sum()),sampling['top_k'],sampling.get('excluded_indices',()))
            assert sampling['indices']==indices
            np.testing.assert_allclose(sampling['probabilities'],probabilities,rtol=0,atol=1e-15)
            if schedule.get('search_mode')=='smc_particle':
                owner=int(schedule['round_trajectories'][str(number)])
                from step3_scheculer.random_search import chain_seed
                assert sampling['proposal_seed']==chain_seed(schedule['population_seed'],owner)
                draw=float(np.random.default_rng(sampling['proposal_seed']).random())
            else:
                draw=float(rng.random())
            assert sampling['uniform_draw']==draw
            expected=indices[min(int(np.searchsorted(np.cumsum(probabilities),draw,side='right')),len(indices)-1)]
            assert chosen==expected
        else:
            assert chosen==order[0]
        before=I.read_contacts(select_folder/'contacts_before_optimization.npz')
        np.testing.assert_array_equal(before[-1]['triangles_m'],problem.candidate(chosen)['triangles_m'])
        assert before[-1]['radius_m']==problem.data.radius_m[chosen]
        for old,unchanged in zip(fixed,before[:-1]):
            for key in old: np.testing.assert_array_equal(old[key],unchanged[key])
        after=I.read_contacts(adjust_folder/'contacts.npz')
        directions=I.check_report(adjust_folder/'insertion_directions.json')
        assert directions['selected_ids']==[p['candidate_id'] for p in after]
        assert len(directions['contacts'])==len(after)
        for contact,record in zip(after,directions['contacts']):
            assert record['geometry_signature']==D.signature(contact,catalogue['normal_depth_m'])
            assert record['certified_directions']==D.normalize(r['direction_id'] for r in record['analysis']['checks'] if r['clear'])
            assert D.nonempty(record['certified_directions'])
            assert record['representative']==D.representative(record['certified_directions'],direction_catalogue)
        assert directions['mode']==D.MODE
        assert directions['direction_catalogue']==direction_catalogue
        assert directions['all_contacts_have_certified_direction']
        assert entry['insertion']['all_contacts_have_certified_direction']
        assert filtering['common_before_selection']==common_allowed
        expected_common=D.common(directions['contacts'],direction_catalogue['global_allowed_directions'])
        assert directions['common_directions']==expected_common and D.nonempty(expected_common)
        connection=connection_checker.check(after,expected_common)
        assert directions['connection']==connection
        assert entry['insertion']['connection']==connection
        if connection['passed']:
            # Materialize the witness and independently replay its whole sweep,
            # not merely the algebraic rear-plane test used by selection.
            parts=connection_checker.parts(after,connection['witness'])
            index=connection['directions']['ids'][0]
            assert direction_analyzer.test(parts,direction_catalogue['vectors'][index])['clear']
            assert work_checker.check_parts(parts)['passed']
        else:
            assert number==len(schedule['rounds']) and schedule['status']=='no_connection_witness_after_optimization'
        if not joint:
            assert set(expected_common['ids']) <= set(common_allowed['ids'])
        common_allowed=expected_common
        for contact,record in zip(after,directions['contacts']):
            signature=record['geometry_signature']
            for index in common_allowed['ids']:
                key=(signature,index)
                if key in replayed_head_rays: continue
                if signature not in head_geometry_cache:
                    head_geometry_cache[signature]=direction_analyzer.heads(contact)
                assert direction_analyzer.test(head_geometry_cache[signature],direction_catalogue['vectors'][index])['clear']
                replayed_head_rays.add(key)
        assert len(after)==number and len(before)==number
        for old,passed in zip(fixed,after[:-1]):
            for key in (('center_m','center_face','candidate_index','candidate_id') if joint else old):
                np.testing.assert_array_equal(old[key],passed[key])
        sample_masks=I.load_npz(adjust_folder/'sample_coverage.npz')
        np.testing.assert_array_equal(sample_masks['initial'],masks['covered'][chosen])
        np.testing.assert_array_equal(sample_masks['base'],masks['base'])
        assert adjusted['covered_count']==int(sample_masks['adjusted'].sum())
        if joint:
            audit_coordinates(adjusted, before, after, schedule['search_config'])
        else:
            assert adjusted['adjusted']['efficiency']>=adjusted['initial']['efficiency']
        assert adjusted['area_constraint']['minimum_area_m2']==minimum_area
        assert adjusted['work_volume_constraint']['enforced']==A.P.POLICY.ENFORCE_PROCESS_ACCESS
        assert all(v['work_volume_clearance']['passed'] for v in adjusted['verification'].values())
        assert all(row['area_m2']>minimum_area for row in adjusted['curve'])
        np.testing.assert_allclose(adjusted['adjusted']['total_area_m2'],I.area(after),rtol=1e-10)
        for contact in after:
            triangles=contact['triangles_m']
            actual_area=.5*np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],
                                                 triangles[:,2]-triangles[:,0]),axis=1).sum()
            assert actual_area>minimum_area
            _,connected=A.connectivity.components(problem.domain.mesh,contact['triangles_m'],contact['source_faces'])
            assert connected['components']==1
            assert C.P.normal_spread(problem.domain.mesh,np.unique(contact['source_faces']))['wrap_limit_satisfied']
        work_checks=[work_checker.check_parts(direction_analyzer.heads(p)) for p in (after if joint else after[-1:])]
        assert all(c['passed'] for c in work_checks),(name,number,'optimized head intersects work volume',work_checks)
        work_check=dict(passed=True,contacts=work_checks)
        full=problem.supply(after)
        assert C.gravity_check(full,problem.domain,problem.scale)['passed']
        assert adjusted['verification']['adjusted']['hard_feasibility']['passed']
        replay,_=C.J.classify(full,problem.targets)
        np.testing.assert_array_equal(replay,sample_masks['adjusted'])
        verification=C.verify_classification(full,problem.targets,replay)
        if check_drawings:
            for folder,filename,report_name in [(select_folder,'selection_views.json','selection.json'),
                    (adjust_folder,'adjustment_views.json','adjustment.json')]:
                views=json.loads((folder/filename).read_text())
                key='selection_sha256' if report_name=='selection.json' else 'adjustment_sha256'
                assert views[key]==C.sha256(folder/report_name)
                I.check_hashes(views['code'])
                for image,digest in views['artifacts'].items():
                    assert C.sha256(folder/image)==digest
        reports.append(dict(round=number,candidate=after[-1]['candidate_id'],candidate_count=len(order),
                            frozen_geometry_unchanged=not joint,centers_unchanged=True,force_checks=verification,
                            actual_contact_areas_above_minimum=True,
                            optimized_work_volume_clearance=work_check,
                            insertion_filter_replayed=True,optimized_direction_sets_replayed=True,
                            individual_head_sweeps_replayed=True))
        fixed=after
        if entry.get('continuous_validation') == 'verified':
            validation=json.loads((owner_out/'verification'/f'round_{number:03d}_continuous.json').read_text())
            assert validation['status']=='verified'
            assert validation['passive_support_constraint']==C.U.description()
            assert validation['geometry_sha256']==C.sha256(adjust_folder/'contacts.npz')
            if 'witnesses' in validation:
                from fractions import Fraction
                from step3_scheculer.verification import Supply
                supply=Supply(problem,after)
                for witness in validation['witnesses']:
                    coefficients=[Fraction(value) for value in witness['coefficients_exact']]
                    assert min(coefficients)>=0
                    target=[Fraction(float(value)) for value in witness['target']]
                    assert witness['passive_support_no_uplift']
                    assert len(target)==7 and target[-1]==0
                    for component in range(7):
                        assert sum(supply.exact(index)[component]*value for index,value in
                                   zip(witness['indices'],coefficients))==target[component]
            if 'certificate_file' in validation:
                from step3_scheculer import enclosure as E
                assert C.sha256(E.__file__)==validation['enclosure_code_sha256']
                path=C.ROOT/validation['certificate_file']
                assert C.sha256(path)==validation['certificate_sha256']
                certificate=I.load_npz(path)
                enclosed=E.targets(problem,validation['sides'],validation['bands'])
                assert len(enclosed)==len(certificate['assignment'])
                assert certificate['assignment'].min()>=0
                assert certificate['assignment'].max()<len(certificate['bases'])
                for i,basis in enumerate(certificate['bases']):
                    tested=enclosed[certificate['assignment']==i]
                    assert E.basis_membership(full[basis].T,tested)[0].all()
        if entry.get('continuous_validation') == 'counterexample':
            validation=json.loads((owner_out/'verification'/f'round_{number:03d}_continuous.json').read_text())
            problem.add_counterexample(validation['counterexample'],
                owner_out/'search_loads'/f'after_round_{number:03d}.json',write=False)
        if audited_rounds is not None:
            audited_rounds.add(round_key)
    final=I.read_contacts(out/'final_contacts.npz')
    for old,new in zip(fixed,final):
        for key in old:
            np.testing.assert_array_equal(old[key],new[key])
    assert len(final)==len(fixed)==schedule['contact_count']
    # Also covers zero selected contacts and a counterexample added after the last round.
    replay,_=C.J.classify(problem.supply(final),problem.targets)
    assert schedule['covered_count']==int(replay.sum())
    if 'random_covered_count' in schedule:
        assert schedule['random_covered_count']==int(replay[:problem.random_sample_count].sum())
    final_directions=I.check_report(out/'insertion_directions.json')
    assert final_directions['mode']==D.MODE
    assert final_directions['selected_ids']==schedule['selected_ids']
    assert len(final_directions['contacts'])==len(final)
    for contact,record in zip(final,final_directions['contacts']):
        assert record['geometry_signature']==D.signature(contact,catalogue['normal_depth_m'])
    valid=D.nonempty(D.common(final_directions['contacts'],direction_catalogue['global_allowed_directions']))
    assert final_directions['common_directions']==common_allowed
    assert schedule['common_withdrawal_directions']==common_allowed
    assert schedule['common_head_withdrawal_verified']==valid
    assert schedule['contact_heads_individually_insertable']==valid
    final_connection=connection_checker.check(final,common_allowed)
    assert final_directions['connection']==final_connection==schedule['connection']
    assert schedule['head_connection_witness_verified']==final_connection['passed']
    assert schedule['common_connected_directions']==final_connection['directions']
    for path in schedule['candidate_filter_records']:
        filtering=I.check_report(C.ROOT/path)
        if filtering['round']>len(schedule['rounds']):
            assert filtering['round']==len(schedule['rounds'])+1
            assert schedule['status']=='candidates_exhausted'
            assert filtering['mode']==D.MODE
            eligible,rows=T.candidate_filter(catalogue['candidates'],
                [p['candidate_index'] for p in final],problem.data.valid,initial_areas,minimum_area,common_allowed,
                connection_checker=connection_checker,selected_contacts=final,candidate=problem.candidate)
            exhausted=C.read(name,filtering['round'])
            assert not M.rank_candidates(exhausted['contributions'])
            for i in np.flatnonzero(eligible):
                assert not C.gravity_check(problem.supply(final+[problem.candidate(i)]),problem.domain,problem.scale)['passed']
            assert filtering['candidates']==rows
    assert schedule['rest_equilibrium_verified']==C.gravity_check(problem.supply(final),problem.domain,problem.scale)['passed']
    if schedule['status']=='population_partial':
        assert len(schedule['rounds'])<S.Q.MAX_CONTACTS and not schedule['continuous_coverage_proved']
    if schedule['status']=='contact_limit_reached':
        assert len(schedule['rounds'])==S.Q.MAX_CONTACTS
        assert not schedule['continuous_coverage_proved']
    if schedule['status'] in ['sampled_complete','continuous_contact_model_verified']:
        assert replay.all()
        assert valid
        assert final_connection['passed']
    if schedule['status']=='no_common_insertion_direction_after_optimization':
        assert not valid
    if schedule['status']=='no_connection_witness_after_optimization':
        assert not final_connection['passed']
    result=dict(object=name,passed=True,rounds=reports,unique_head_ray_replays=len(replayed_head_rays),
                sweep_reuse_rule='Only identical actual geometry signature, normal depth and shared direction ID within this case',schedule_sha256=C.sha256(out/'schedule.json'),
                audit_code_sha256=C.sha256(__file__),drawings_checked=check_drawings,
                scope='Stored load decisions, joint connection/insertion filters, actual-thickness rear-frame witness sweeps, round transitions and continuous primal certificates; no base or complete fixture certificate.')
    I.save(out/'audit.json',result)
    print(name,'scheduler audit passed',len(reports),'rounds',flush=True)
    return result


def audit_coordinates(adjusted, before, after, config):
    assert adjusted['optimization_mode']=='coordinate_all_contacts'
    updates=adjusted['coordinate_updates']
    assert 0<len(updates)<=config['sizing_sweeps']*len(after)
    assert len(updates)%len(after)==0
    order=[len(after)-1,*range(len(after)-1)]
    radii=[p['radius_m'] for p in before]
    areas=[I.area([p]) for p in before]
    count=adjusted['initial']['covered_count']
    full=adjusted['sample_count']
    for i,update in enumerate(updates):
        index=order[i%len(order)]
        assert update['coordinate_index']==index and update['sweep']==i//len(order)+1
        assert update['candidate_id']==before[index]['candidate_id']
        assert update['initial_radius_m']==radii[index]
        assert update['before']['covered_count']==count
        np.testing.assert_allclose(update['before']['total_area_m2'],sum(areas),rtol=1e-10)
        row=update['after']
        if count==full:
            assert row['covered_count']==full
            assert row['total_area_m2']<=sum(areas)*(1+1e-10)
        elif row['covered_count']!=full:
            assert row['efficiency']>=update['before']['efficiency']*(1-1e-12)
        count=row['covered_count'];radii[index]=update['radius_m'];areas[index]=row['area_m2']
        np.testing.assert_allclose(row['total_area_m2'],sum(areas),rtol=1e-10)
    np.testing.assert_allclose(radii,[p['radius_m'] for p in after],rtol=0,atol=0)
    assert count==adjusted['covered_count']
    used=sum(u['search']['evaluated_sizes'] for u in updates)
    assert used==adjusted['selection']['evaluated_sizes']<=config['sizing_budget']


def audit_ensemble(name, schedule, out, check_drawings):
    from step3_scheculer import random_search as R
    assert len(schedule['trajectories'])==schedule['search_config']['trajectories']
    results=[]
    for index,row in enumerate(schedule['trajectories']):
        assert row['index']==index and row['seed']==R.chain_seed(schedule['search_config']['seed'],index)
        with PTH.trajectory(index):
            folder=PTH.stage_folder(name,S.OUTPUT_NAME)
            result=I.check_report(folder/'schedule.json')
            assert row['schedule']==str((folder/'schedule.json').relative_to(I.ROOT))
            assert row['status']==result['status']
            assert row['random_covered_count']==result['random_covered_count']
            assert row['continuous_coverage_proved']==result['continuous_coverage_proved']
            assert row['seed']==result['search_config']['seed']
            for key in ('top_k','sizing_sweeps','sizing_budget'):
                assert result['search_config'][key]==schedule['search_config'][key]
            run(name,check_drawings)
            results.append(result)
    winner=min(range(len(results)),key=lambda i:(*R.result_key(results[i]),i))
    assert schedule['selected_trajectory']==winner
    with PTH.trajectory(winner):
        source=PTH.stage_folder(name,S.OUTPUT_NAME)
        assert I.sha256(out/'final_contacts.npz')==I.sha256(source/'final_contacts.npz')
        original_directions=I.check_report(source/'insertion_directions.json')
        published_directions=I.check_report(out/'insertion_directions.json')
        for key in ('contacts','common_directions','selected_ids','all_contacts_have_certified_direction','connection'):
            assert published_directions[key]==original_directions[key]
    for key in ('status','rounds','selected_ids','contact_count','covered_count','random_covered_count',
                'continuous_coverage_proved','common_withdrawal_directions',
                'connection','common_connected_directions','head_connection_witness_verified'):
        assert schedule[key]==results[winner][key]
    result=dict(object=name,passed=True,trajectory_count=len(results),selected_trajectory=winner,
                schedule_sha256=I.sha256(out/'schedule.json'),audit_code_sha256=I.sha256(__file__),
                drawings_checked=check_drawings,
                scope='Every independent trajectory audited; minimum-area continuously verified winner, or highest original-sample coverage partial result.')
    I.save(out/'audit.json',result)
    return result


def audit_population(name, schedule, out, check_drawings):
    from step3_scheculer import random_search as R, smc_search as SM
    assert schedule['smc_policy']==SM.POLICY
    assert schedule['particle_evaluations']==len(schedule['particles'])
    config=schedule['search_config']
    rng=np.random.default_rng(R.chain_seed(config['seed'],2**31))
    assert schedule['resampling_seed']==R.chain_seed(config['seed'],2**31)
    results=[]
    audited_rounds=set()
    expected_archive=[]
    for index,row in enumerate(schedule['particles']):
        assert row['index']==index and row['seed']==R.chain_seed(config['seed'],index)
        parent=row['parent_id']
        assert parent is None or 0<=parent<index
        with PTH.trajectory(index):
            folder=PTH.stage_folder(name,S.OUTPUT_NAME)
            result=I.check_report(folder/'schedule.json')
            assert row['schedule']==str((folder/'schedule.json').relative_to(I.ROOT))
            assert result['particle_id']==index and result['population_seed']==config['seed']
            for key in ('status','continuous_coverage_proved','random_sample_count','random_covered_count'):
                assert row[key]==result[key]
            assert result['search_config']['seed']==row['seed']
            for key in ('top_k','sizing_sweeps','sizing_budget'):
                assert result['search_config'][key]==config[key]
            prefix=results[parent] if parent is not None else None
            depth=len(prefix['rounds'])+1 if prefix else 1
            if prefix:
                assert prefix['status']=='population_partial'
                assert result['rounds'][:depth-1]==prefix['rounds']
                for key,value in prefix['round_trajectories'].items():
                    assert result['round_trajectories'][key]==value
            with PTH.round_owners(result['round_trajectories']):
                score=C.read(name,depth)
                selected=M.read(name,depth)
                previous_siblings=[j for j in range(index) if schedule['particles'][j]['parent_id']==parent]
                expected_excluded=[results[j]['rounds'][-1]['candidate_index'] for j in previous_siblings
                    if len(results[j]['rounds'])==depth]
                assert row['excluded_sibling_candidates']==expected_excluded
                assert selected['sampling']['excluded_indices']==expected_excluded
                if selected['winner'] is not None:
                    assert selected['winner']['index'] not in expected_excluded
                counts=dict(scoring_rounds=int('reused_from' not in score),
                    scored_candidate_combinations=sum(r['status']=='scored_joint' for r in score['contributions']) if 'reused_from' not in score else 0,
                    added_contact_optimizations=int(len(result['rounds'])==depth),
                    sizing_coverage_evaluations=A.read(name,depth)['selection']['evaluated_sizes'] if len(result['rounds'])==depth else 0,
                    continuous_checks=int(len(result['rounds'])==depth and 'continuous_validation' in result['rounds'][-1]))
                assert row['evaluations']==counts
            run(name,check_drawings,audited_rounds)
            results.append(result)
            if result['continuous_coverage_proved']:
                expected_archive.append(index)
    assert schedule['verified_archive']==expected_archive
    assert schedule['evaluation_counts']=={k:sum(p['evaluations'][k] for p in schedule['particles']) for k in schedule['evaluation_counts']}
    expected_parents=[None]*config['trajectories']
    for layer in schedule['layers']:
        ids=layer['evaluated_particle_ids']
        assert layer['requested_slots']==len(expected_parents)
        skipped={row['slot']:row for row in layer['skipped_slots']}
        expected_evaluated=[parent for i,parent in enumerate(expected_parents) if i not in skipped]
        assert [schedule['particles'][i]['parent_id'] for i in ids]==expected_evaluated
        active=[i for i in ids if results[i]['status']=='population_partial']
        assert layer['active_particle_ids']==active
        if layer['depth']<S.Q.MAX_CONTACTS and active:
            record=SM.resample(active,[results[i] for i in active],config['trajectories'],rng,layer['depth'])
            assert layer['resampling']==record
            expected_parents=record['selected_parent_ids']
        else:
            assert 'resampling' not in layer
            expected_parents=[]
    winner=min(range(len(results)),key=lambda i:(*R.result_key(results[i]),i))
    assert schedule['selected_trajectory']==winner
    with PTH.trajectory(winner):
        source=PTH.stage_folder(name,S.OUTPUT_NAME)
        assert I.sha256(out/'final_contacts.npz')==I.sha256(source/'final_contacts.npz')
        original=I.check_report(source/'insertion_directions.json')
        published=I.check_report(out/'insertion_directions.json')
        for key in ('contacts','common_directions','selected_ids','all_contacts_have_certified_direction','connection'):
            assert original[key]==published[key]
    assert schedule['selected_particle_status']==results[winner]['status']
    expected_status='population_search_exhausted' if results[winner]['status']=='population_partial' else results[winner]['status']
    assert schedule['status']==expected_status
    for key in ('rounds','selected_ids','contact_count','covered_count','random_covered_count',
                'continuous_coverage_proved','common_withdrawal_directions','connection',
                'common_connected_directions','head_connection_witness_verified','round_trajectories'):
        assert schedule[key]==results[winner][key]
    result=dict(object=name,passed=True,particle_count=len(results),selected_trajectory=winner,
        unique_rounds_audited=len(audited_rounds),schedule_sha256=I.sha256(out/'schedule.json'),
        audit_code_sha256=I.sha256(__file__),drawings_checked=check_drawings,
        scope='Every unique particle round audited; ancestor states reused only after audit; proposals, resampling, archive and winning contact certificate replayed. No robot grasp or complete fixture certificate.')
    I.save(out/'audit.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('objects',nargs='*')
    parser.add_argument('--no-drawings',action='store_true')
    args=parser.parse_args()
    for name in args.objects or C.OBJECTS:
        run(name,not args.no_drawings)
