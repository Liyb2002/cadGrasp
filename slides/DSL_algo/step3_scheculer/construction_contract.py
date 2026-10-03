"""Constructive Step3 acceptance and deterministic Step4 materialization.

A contact proposal is never a Step3 success. Success requires an exported,
construction-validated full solid, including ground coverage, all installed roots, working
surfaces and complete continuous object paths. The witness is the fallback
output: Step4 may publish it without solving another construction problem.
"""
from pathlib import Path
import shutil
import json
import numpy as np
import trimesh
from step3_scheculer import contacts as I


def sources():
    from step4_connect_support import try_dsl_growth as B
    from step4_connect_support import coverage_growth as C
    from step3_scheculer import pair_geometry as P
    return list(dict.fromkeys([Path(__file__),Path(B.__file__),Path(C.__file__),Path(P.__file__),
        Path(B.H.__file__),Path(B.L.__file__),Path(B.D.__file__),Path(B.F.__file__),Path(B.S.__file__),
        Path(B.B.__file__),Path(B.G.__file__),Path(B.P.__file__)]+B.PATHS.sources()+B.A.sources()))


def accept_witness(report):
    """No amount of contact coverage can substitute for a real geometry witness."""
    return bool(report.get('complete') and report.get('passed') and report.get('constructed')
                and report.get('export_roundtrip_bit_exact')
                and report.get('minimum_branch_thickness',{}).get('all_complete_cores_preserved')
                and report.get('selected_exit_paths'))


def certify(group,proposal,trials=3):
    from step4_connect_support.try_dsl_growth import construct_witness
    out=group/'step3_scheculer/dsl_shared';witness=out/'construction_witness'
    # Immutable proposal inputs prevent a report -> witness -> report hash cycle.
    proposal=dict(proposal,passed=False,passed_scope='contact_proposal_only',
                  complete_fixture_verified=False)
    I.save(out/'contact_proposal.json',proposal)
    result=dict(proposal,complete=False,status='constructing_step3_witness')
    I.save(out/'report.json',result)
    if proposal['local_contact_passed']:
        construct_witness(group,trials=trials,output=witness,proposal=True,render=False)
        body=json.loads((witness/'report.json').read_text())
        if body['passed']:
            I.check_report(witness/'report.json')
        else:
            body['provenance']=dict(inputs=I.hashes([out/'contact_proposal.json']),code=I.hashes(sources()))
            I.save(witness/'report.json',body)
        accepted=accept_witness(body)
    else:
        for name in ('shape.obj','geometry_certificate.npz','overview.png'):
            (witness/name).unlink(missing_ok=True)
        body=dict(complete=True,passed=False,constructed=False,status='contact_proposal_not_feasible',
                  provenance=dict(inputs=I.hashes([out/'contact_proposal.json']),code=I.hashes(sources())))
        I.save(witness/'report.json',body)
        accepted=False
    result.update(complete=True,passed=accepted,passed_scope='force_and_complete_constructible_fixture',
        status='step3_constructive_witness_accepted' if accepted else 'step3_no_accepted_constructive_witness',
        complete_fixture_verified=accepted,construction_passed=accepted,
        construction_witness=str((witness/'report.json').relative_to(I.ROOT)),
        step4_requires_new_construction_search=False,
        negative_verdict=None if accepted else 'No verified construction in the finite proposal/path/placement search; not physical infeasibility')
    result['provenance']=dict(inputs=I.hashes([out/'contact_proposal.json',witness/'report.json']),
                              code=I.hashes(sources()))
    if accepted:
        result['construction_witness_artifacts']=I.hashes([witness/'shape.obj',witness/'geometry_certificate.npz'])
        result['selected_exit_paths']=body['selected_exit_paths']
    I.save(out/'report.json',result)
    return result


def replay(group):
    from step4_connect_support.try_dsl_growth import DSLGrow
    schedule=I.check_report(group/'step3_scheculer/dsl_shared/report.json')
    if not schedule['passed'] or not schedule.get('complete_fixture_verified'):
        raise ValueError('Step3 has no accepted full construction witness')
    I.check_hashes(schedule['construction_witness_artifacts'])
    witness=I.ROOT/schedule['construction_witness'];body=I.check_report(witness)
    if not accept_witness(body):raise ValueError('Invalid full construction witness')
    grow=DSLGrow(group,out=witness.parent/'replay_inputs',selected_option_ids=body['selected_exit_option_ids'])
    mesh=trimesh.load(witness.parent/'shape.obj',force='mesh',process=False)
    geometry=dict(validation_policy='Step3 construction acceptance reused; no independent geometry replay',
        physical_head_count=body.get('physical_head_count'),
        installed_shared_solid_max_error_m=body.get('installed_shared_solid_max_error_m'))
    return schedule,body,grow,mesh,geometry


def publish(group):
    """Step4 consumes an existing Step3 witness; it cannot reject it by a new search."""
    out=group/'step4/data/dsl_shared_support';out.mkdir(parents=True,exist_ok=True)
    schedule=I.check_report(group/'step3_scheculer/dsl_shared/report.json')
    if not schedule['passed']:
        for name in ('shape.obj','geometry_certificate.npz','overview.png'):(out/name).unlink(missing_ok=True)
        result=dict(complete=True,group=group.name,passed=False,constructed=False,
            status='not_applicable_step3_rejected',new_construction_search=False,
            provenance=dict(inputs=I.hashes([group/'step3_scheculer/dsl_shared/report.json']),code=I.hashes(sources())))
        I.save(out/'report.json',result)
        return result
    schedule,body,grow,mesh,geometry=replay(group)
    witness=I.ROOT/schedule['construction_witness']
    for name in ('shape.obj','geometry_certificate.npz'):shutil.copy2(witness.parent/name,out/name)
    assert I.sha256(out/'shape.obj')==I.sha256(witness.parent/'shape.obj')
    report=dict(body,schema='step4_materialized_step3_witness_v4',group=group.name,
        status='step4_step3_witness_materialized',new_construction_search=False,
        step3_witness_retained_as_fallback=True,published_model_bit_identical_to_step3_witness=True,
        step3_construction_acceptance=geometry)
    report['provenance']=dict(inputs=I.hashes([group/'step3_scheculer/dsl_shared/report.json',witness,
        witness.parent/'shape.obj',witness.parent/'geometry_certificate.npz']),code=I.hashes(sources()))
    I.save(out/'report.json',report)
    grow.case.output=out
    from step4_connect_support import boxed_support as F
    F.draw(out/'overview.png',grow.case,mesh,grow.bases,grow.offsets)
    print('DSL STEP4 MATERIALIZED',group.name,'identical Step3 witness',flush=True)
    return dict(group=group.name,passed=True,constructed=True,volume_cm3=report['volume_cm3'],space_budget=report['space_budget'])
