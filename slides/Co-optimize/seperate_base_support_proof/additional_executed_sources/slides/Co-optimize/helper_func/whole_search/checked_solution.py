"""Validate a stored actual solution under its original execution sources."""
import json
from pathlib import Path
import numpy as np
from .common import ROOT,C


def check_solution(path):
    path=Path(path);report=json.loads(path.read_text())
    if not report.get('complete') or not report.get('force_exit_work_passed'):
        raise ValueError(f'No actual feasible certificate: {path}')
    if not report.get('poses') or any(n!=32768 for n in report['counts'].values()):
        raise ValueError(f'Incomplete original-load coverage: {path}')
    work=report['actual_work_surface_checks']
    if len(work)!=len(report['poses']) or not all(r['passed'] for r in work):
        raise ValueError(f'Incomplete original work-access checks: {path}')
    C.I.check_hashes(report['provenance']['inputs'])
    code=report['provenance']['code'];snapshots={}
    for relative,digest in code.items():
        if '/executed_sources/' in relative:
            if C.I.sha256(ROOT/relative)!=digest:raise RuntimeError(f'Changed executed source: {relative}')
            original=relative.split('/executed_sources/',1)[1]
            snapshots.setdefault(original,set()).add(digest)
    for relative,digest in code.items():
        current=ROOT/relative
        if current.is_file() and C.I.sha256(current)==digest:continue
        if digest in snapshots.get(relative,set()):continue
        # Original initializers are sometimes included only by their source.
        historical=ROOT/'slides/Co-optimize/data/code_history/before_xyz_translation_20261010'/relative
        if historical.is_file() and C.I.sha256(historical)==digest:continue
        raise RuntimeError(f'Missing checked solution execution source: {relative}')
    for filename,digest in report['artifacts'].items():
        if C.I.sha256(path.parent/filename)!=digest:raise RuntimeError(f'Changed solution artifact: {filename}')
    out=path.parent.parent
    with np.load(out/'layout.npz') as saved:
        heights={pose:float((saved['native_world'][saved['hosts'][k]]@saved['placements'][k])[2,3]
                 -saved['native_world'][k,2,3]) for k,pose in enumerate(report['poses'])}
    for pose,height in heights.items():
        if height < -1e-9:raise ValueError('Saved object crosses the floor')
        if height>1e-9:
            if report.get('workpiece_floor_contact_allowed',{}).get(pose,True):
                raise ValueError('Airborne incumbent lacks a no-floor certificate')
            with np.load(out/f'{pose}_force.npz') as force:
                supply=force['supply_7d'];heads=np.max(abs(supply[:,:6]),axis=1)>1e-12
                np.testing.assert_allclose(supply[heads,6],supply[heads,2],atol=1e-12,rtol=0)
    if not np.isfinite(report['volume_cm3']) or report['volume_cm3']<=0:raise ValueError('Invalid certified volume')
    return report


def check_force_result(path):
    """Read saved masks and fingerprints, without solving any demands again."""
    path=Path(path);report=json.loads(path.read_text())
    if report.get('acceptance')!='all_original_force_torque_demands_on_search_contacts':
        return check_solution(path)
    if not report.get('complete') or not report.get('force_passed'):
        raise ValueError(f'No full force/torque pass: {path}')
    if set(report['counts'])!=set(report['poses']) or any(n!=32768 for n in report['counts'].values()):
        raise ValueError(f'Incomplete original-load coverage: {path}')
    C.I.check_hashes(report['provenance']['inputs'])
    code=report['provenance']['code'];snapshots={}
    for relative,digest in code.items():
        if '/executed_sources/' in relative:
            if C.I.sha256(ROOT/relative)!=digest:raise RuntimeError(f'Changed executed source: {relative}')
            snapshots.setdefault(relative.split('/executed_sources/',1)[1],set()).add(digest)
    for relative,digest in code.items():
        if (ROOT/relative).is_file() and C.I.sha256(ROOT/relative)==digest:continue
        if digest in snapshots.get(relative,set()):continue
        raise RuntimeError(f'Missing force result execution source: {relative}')
    for filename,digest in report['artifacts'].items():
        if C.I.sha256(path.parent/filename)!=digest:raise RuntimeError(f'Changed result artifact: {filename}')
    out=path.parent.parent
    for pose in report['poses']:
        with np.load(out/f'{pose}_force.npz') as saved:
            mask=saved['mask']
            if mask.shape!=(32768,) or not mask.all():raise ValueError('Saved force mask is incomplete')
    return report


def read_incumbents(base,labels,summaries):
    choices={};records=[]
    for filename in summaries:
        path=Path(filename)
        if not path.is_absolute():
            repository_path=ROOT/path
            path=repository_path if repository_path.exists() else base/path
        if not path.exists():continue
        data=json.loads(path.read_text())
        for label in labels:
            row=data.get('selected',{}).get(label)
            if not row or not row.get('passed'):continue
            report_path=base/label/'step4/step4.2'/row['experiment']/'data/report.json'
            report=check_force_result(report_path)
            expected=['pose_'+s.removeprefix('pose') for s in label.split('+')]
            if report['poses']!=expected:raise ValueError('Incumbent belongs to another pose set')
            if report.get('acceptance')=='all_original_force_torque_demands_on_search_contacts' and not report.get('mesh_exported'):
                continue
            row=dict(row,volume_cm3=report['volume_cm3'],checked_incumbent=True,
                     certificate=str(report_path.relative_to(ROOT)),incumbent_summary=str(path))
            records.append(dict(id=label,experiment=row['experiment'],volume_cm3=row['volume_cm3']))
            if label not in choices or row['volume_cm3']<choices[label]['volume_cm3']:choices[label]=row
    return choices,records


def select_no_worse(candidate,incumbent):
    """No tolerance admits more material; unresolved proposals never replace."""
    if incumbent is None:return dict(candidate,retained_incumbent=False)
    comparable=candidate.get('volume_measure')!='nominal_material_occupancy_estimate'
    improvement=candidate.get('passed',False) and comparable and candidate['volume_cm3']<incumbent['volume_cm3']
    chosen=candidate if improvement else incumbent
    return dict(chosen,retained_incumbent=not improvement,
                incumbent_experiment=incumbent['experiment'],incumbent_volume_cm3=incumbent['volume_cm3'],
                attempted_experiment=candidate['experiment'],attempted_passed=candidate.get('passed',False),
                attempted_volume_cm3=candidate.get('volume_cm3'),material_non_regression=True)
