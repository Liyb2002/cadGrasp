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
from step2_local_support import withdrawal as D
from step3_scheculer import directions as T
from step2_local_support import insertion_directions as K

from step1.cases import pose_name


def run(name, check_drawings=True):
    out=C.OUTPUTS/name/pose_name()/S.OUTPUT_NAME
    schedule=I.check_report(out/'schedule.json')
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
        score=C.read(name,number)
        selected=M.read(name,number)
        adjusted=A.read(name,number)
        assert score['passive_support_constraint']==C.U.description()
        assert adjusted['passive_support_constraint']==C.U.description()
        score_folder=I.folder(name,C.OUTPUT_NAME,number)
        select_folder=I.folder(name,M.OUTPUT_NAME,number)
        adjust_folder=I.folder(name,A.OUTPUT_NAME,number)
        masks=I.load_npz(score_folder/'sample_coverage.npz')
        rows=score['contributions']
        assert score['selected_indices']==sorted(p['candidate_index'] for p in fixed)
        filtering=I.check_report(out/f'round_{number:03d}'/'candidate_filter.json')
        assert filtering['mode']==D.MODE
        expected_eligible,expected_rows=T.candidate_filter(catalogue['candidates'],
            [p['candidate_index'] for p in fixed],problem.data.valid,initial_areas,minimum_area,common_allowed)
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
        assert selected['ranking']==order and selected['winner']['index']==order[0]
        before=I.read_contacts(select_folder/'contacts_before_optimization.npz')
        np.testing.assert_array_equal(before[-1]['triangles_m'],problem.candidate(order[0])['triangles_m'])
        assert before[-1]['radius_m']==problem.data.radius_m[order[0]]
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
            for key in old:
                np.testing.assert_array_equal(old[key],passed[key])
        sample_masks=I.load_npz(adjust_folder/'sample_coverage.npz')
        np.testing.assert_array_equal(sample_masks['initial'],masks['covered'][order[0]])
        np.testing.assert_array_equal(sample_masks['base'],masks['base'])
        assert adjusted['covered_count']==int(sample_masks['adjusted'].sum())
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
        work_check=work_checker.check_parts(direction_analyzer.heads(after[-1]))
        assert work_check['passed'],(name,number,'optimized head intersects work volume',work_check)
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
                            frozen_geometry_unchanged=True,force_checks=verification,
                            actual_contact_areas_above_minimum=True,
                            optimized_work_volume_clearance=work_check,
                            insertion_filter_replayed=True,optimized_direction_sets_replayed=True,
                            individual_head_sweeps_replayed=True))
        fixed=after
        if entry.get('continuous_validation') == 'verified':
            validation=json.loads((out/'verification'/f'round_{number:03d}_continuous.json').read_text())
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
            validation=json.loads((out/'verification'/f'round_{number:03d}_continuous.json').read_text())
            problem.add_counterexample(validation['counterexample'],
                out/'search_loads'/f'after_round_{number:03d}.json',write=False)
    final=I.read_contacts(out/'final_contacts.npz')
    for old,new in zip(fixed,final):
        for key in old:
            np.testing.assert_array_equal(old[key],new[key])
    assert len(final)==len(fixed)==schedule['contact_count']
    # Also covers zero selected contacts and a counterexample added after the last round.
    replay,_=C.J.classify(problem.supply(final),problem.targets)
    assert schedule['covered_count']==int(replay.sum())
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
    for path in schedule['candidate_filter_records']:
        filtering=I.check_report(C.ROOT/path)
        if filtering['round']>len(schedule['rounds']):
            assert filtering['round']==len(schedule['rounds'])+1
            assert schedule['status']=='candidates_exhausted'
            assert filtering['mode']==D.MODE
            eligible,rows=T.candidate_filter(catalogue['candidates'],
                [p['candidate_index'] for p in final],problem.data.valid,initial_areas,minimum_area,common_allowed)
            exhausted=C.read(name,filtering['round'])
            assert not M.rank_candidates(exhausted['contributions'])
            for i in np.flatnonzero(eligible):
                assert not C.gravity_check(problem.supply(final+[problem.candidate(i)]),problem.domain,problem.scale)['passed']
            assert filtering['candidates']==rows
    assert schedule['rest_equilibrium_verified']==C.gravity_check(problem.supply(final),problem.domain,problem.scale)['passed']
    if schedule['status']=='contact_limit_reached':
        assert len(schedule['rounds'])==S.Q.MAX_CONTACTS
        assert not schedule['continuous_coverage_proved']
    if schedule['status'] in ['sampled_complete','continuous_contact_model_verified']:
        assert replay.all()
        assert valid
    if schedule['status']=='no_common_insertion_direction_after_optimization':
        assert not valid
    result=dict(object=name,passed=True,rounds=reports,unique_head_ray_replays=len(replayed_head_rays),
                sweep_reuse_rule='Only identical actual geometry signature, normal depth and shared direction ID within this case',schedule_sha256=C.sha256(out/'schedule.json'),
                audit_code_sha256=C.sha256(__file__),drawings_checked=check_drawings,
                scope='Software, stored load decisions, geometry, insertion filtering, common 3-D direction intersections and every surviving whole-head sweep, round transitions, and saved continuous primal certificates; floor and connectors are not certified.')
    I.save(out/'audit.json',result)
    print(name,'scheduler audit passed',len(reports),'rounds',flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('objects',nargs='*')
    parser.add_argument('--no-drawings',action='store_true')
    args=parser.parse_args()
    for name in args.objects or C.OBJECTS:
        run(name,not args.no_drawings)
