"""Publish exact accepted V6 witnesses, preserving previous public results."""
from pathlib import Path
import shutil
from step3_scheculer import contacts as I,operation_dsl as F


def publish_all():
    rows=[]
    for group in sorted((I.OUTPUTS/'B').glob('pose*+*')):
        stage=group/'step3_scheculer'/F.STAGE
        r=I.check_report(stage/'report.json');assert r['passed']
        root=group/'step4';body=root/'data'/F.BODY_STAGE
        accepted=I.check_report(body/'report.json');assert accepted['passed']
        witness=I.check_report(I.ROOT/accepted['witness'])
        assert witness['construction']['minimum_branch_thickness']['all_complete_cores_preserved']
        assert witness['validation_policy']['export_recheck'] is False
        backup=stage/'previous_public_step4';backup.mkdir(exist_ok=True)
        for name in ('shape.obj','geometry_certificate.npz','overview.png','construction_steps.png','report.json'):
            if (root/name).exists() and not (backup/name).exists():shutil.copy2(root/name,backup/name)
        for source,target in (('shape.obj','shape.obj'),('geometry_certificate.npz','geometry_certificate.npz'),('overview.png','overview.png'),('construction.png','construction_steps.png')):
            shutil.copy2(body/source,root/target)
            assert I.sha256(body/source)==I.sha256(root/target)
        report=dict(accepted,step4_executed=True,latest_baseline_envelope_growth=True,construction_search_rerun=True,exported_geometry_replay=False,
            placement=dict(accepted['placement'],directions=[p['support_withdrawal_world'] for p in accepted['selected_exit_paths']]),
            provenance=dict(inputs=I.hashes([body/'report.json']),code=I.hashes([Path(__file__)])),
            artifacts={name:I.sha256(root/name) for name in ('shape.obj','geometry_certificate.npz','overview.png','construction_steps.png')})
        I.save(root/'report.json',report)
        I.save(root/'data/report.json',dict(report,artifacts={'../'+k:v for k,v in report['artifacts'].items()}))
        I.check_report(root/'report.json');I.check_report(root/'data/report.json')
        (root/'README.md').write_text(f"# Current Step4: operation DSL\n\n{group.name}: {accepted['physical_head_count']} physical heads, {accepted['shared_head_count']} shared. All original loads and independent exits pass. The latest copied baseline EnvelopeGrow constructed this fixture with one in-memory acceptance.\n\n[Model](shape.obj) · [Overview](overview.png) · [Construction](construction_steps.png) · [Report](report.json)\n")
        rows.append(dict(group=group.name,passed=True,physical_heads=accepted['physical_head_count'],shared_heads=accepted['shared_head_count'],shape_sha256=I.sha256(root/'shape.obj')))
        print('PUBLIC OPERATION STEP4',group.name,'PASS',flush=True)
    out=I.OUTPUTS/'B/pose2+9+13+15+17/step4/data'
    I.save(out/'dsl_operations_publication.json',dict(complete=True,passed=True,results=rows,passed_count=len(rows),provenance=dict(inputs=I.hashes([I.OUTPUTS/'B'/r['group']/'step4/report.json' for r in rows]),code=I.hashes([Path(__file__)]))))
