"""Immutable saved-layout inputs and provenance for Step5.2 and Step5.3."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/cadgrasp-stage5-matplotlib')

COOPT = Path(__file__).resolve().parents[1]
ROOT = Path('/home/yli581/Desktop/cadGrasp')
for folder in [ROOT, COOPT/'helper_func', COOPT/'vis_func', COOPT/'step5.1']:
    sys.path.insert(0, str(folder))

import numpy as np
import trimesh
import system_floor as F
import step51_render as V
import hull_render as H
import artifact_store as A


def digest(path):
    return A.digest(path)


def connected_core_policy(data):
    # data['record'] is Step5.1's ground-demand report. The policy belongs to
    # the selected Step4 certificate, whose bytes Inputs already protects.
    record=json.loads((data['selected']/'data/report.json').read_text())
    return bool(record.get('step4_contact_core_connected'))


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def posed(mesh, transform):
    result = mesh.copy()
    result.apply_transform(transform)
    return result


def constructed_reference(inputs, data, args):
    """Read a certified reference belonging to this exact saved Step4 layout."""
    stage = getattr(args, 'reference_stage', 'step5.2')
    if stage == 'step4':
        folder = data['selected']
        report_path = folder/'preconstructed_base.json'
        report = json.loads(report_path.read_text())
        if not report['complete'] or not report['construction']['passed']:
            raise ValueError('The selected Step4 has no certified constructed reference')
        expected = report.get('source_sha256', {})
        for name, digest_value in expected.items():
            path = folder/name
            if digest(path) != digest_value:
                raise ValueError('Preconstructed Step4 reference changed: '+name)
            inputs.protected[relative(path)] = digest_value
    else:
        folder = inputs.base/data['case']['id']/'step5'/stage/args.reference_experiment
        report_path = folder/'data/report.json'
        report = json.loads(report_path.read_text())
        if (not report['complete'] or not report['passed'] or
                report['selected_step4_experiment'] != data['case']['experiment']):
            raise ValueError('Constructed reference belongs to another or failed Step4 layout')
    support_path = folder/report['support']
    for path in [report_path, support_path]:
        inputs.protected[relative(path)] = digest(path)
    return support_path, report


def occupied_bounds(body, fixture, rows):
    points = []
    for row in rows:
        points.extend([F.transform_points(body.vertices, row['object_transform']),
                       F.transform_points(fixture.vertices, row['fixture_transform'])])
    points = np.vstack(points)
    bounds = np.array([points.min(axis=0), points.max(axis=0)])
    return bounds


def add_bounds(bounds, vertices, states):
    points = np.vstack([F.transform_points(vertices, state) for state in states])
    return np.array([np.minimum(bounds[0], points.min(axis=0)),
                     np.maximum(bounds[1], points.max(axis=0))])


def volume(bounds):
    return float(np.prod(np.asarray(bounds)[1]-np.asarray(bounds)[0]))


class Inputs:
    def __init__(self, name, selection=None, experiment=None):
        self.name = name
        self.base = A.result_root(name)
        self.selection_path = A.resolve_artifact(selection or self.base/'data/base_oracle_results/summary.json')
        selection_record = json.loads(self.selection_path.read_text())
        self.cases = selection_record.get('selected', selection_record.get('cases', selection_record.get('results', [])))
        self.experiment = experiment
        self.mesh_path = ROOT/'objects'/name/'mesh.stl'
        self.mesh = trimesh.load(self.mesh_path, force='mesh', process=False)
        self.protected = {relative(p): digest(p) for p in [self.selection_path, self.mesh_path]}

    def load(self, case):
        out = self.base/case['id']/'step5/step5.1'
        if self.experiment:
            out = out/self.experiment
        report_path = out/'data/report.json'
        record = json.loads(report_path.read_text())
        selected = A.resolve_artifact(ROOT/record['selected_step4_source'])
        if not case['passed'] or not record['complete'] or case['experiment'] != record['selected_step4_experiment']:
            raise ValueError('Expected the current passed selection and saved Step5.1 results')
        self.protected.update(record['provenance']['inputs'])
        for p in [report_path, selected/'support.obj', selected/'layout.npz', selected/'data/report.json']:
            self.protected[relative(p)] = digest(p)
        fixture = trimesh.load(selected/'support.obj', force='mesh', process=False)
        with np.load(selected/'layout.npz', allow_pickle=False) as layout:
            directions = layout['directions'].copy()
        rows = []
        for k, previous in enumerate(record['per_pose']):
            path = out/previous['arrays']
            self.protected[relative(path)] = digest(path)
            with np.load(path, allow_pickle=False) as saved:
                cloud = saved['floor_demands_world_m'].copy()
                hull = saved['hull_indices'].copy()
                ft = saved['T_world_fixture'].copy()
                ot = saved['T_world_object'].copy()
            object_world = F.transform_points(self.mesh.vertices, ot)
            ground = object_world[np.abs(object_world[:, 2]) <= 1e-9, :2]
            rows.append(dict(pose=previous['pose'], color=previous['color'],
                fixture_transform=ft, object_transform=ot, cloud=cloud, hull=cloud[hull],
                object_ground=ground, direction=directions[k],
                world_exit_direction=ft[:3, :3]@directions[k],
                common_exit_direction=directions[k]))
        if len(rows) != case['pose_count']:
            raise ValueError('Every pose must remain in the process')
        return dict(case=case, record=record, fixture=fixture, body=self.mesh, rows=rows,
                    states=[r['fixture_transform'] for r in rows], selected=selected,
                    bounds=occupied_bounds(self.mesh, fixture, rows))

    def verify(self):
        changed = [p for p, h in self.protected.items() if digest(ROOT/p) != h]
        if changed:
            raise ValueError(f'Saved inputs changed: {changed[:5]}')


def arguments(step, default_batch):
    parser = argparse.ArgumentParser(description=f'Deterministic {step} from current saved results')
    parser.add_argument('object', nargs='?', default='B')
    parser.add_argument('--sets', nargs='+')
    parser.add_argument('--selection-summary', type=Path)
    parser.add_argument('--batch-name', default=default_batch)
    args = parser.parse_args()
    inputs = Inputs(args.object, args.selection_summary)
    cases = inputs.cases
    if args.sets:
        missing = set(args.sets)-{c['id'] for c in cases}
        if missing: parser.error(f'Unknown selected sets: {sorted(missing)}')
        cases = [c for c in cases if c['id'] in args.sets]
    if not cases: parser.error('No selected sets')
    for case in cases:
        out = inputs.base/case['id']/'step5'/step
        if out.exists(): parser.error(f'Existing result is protected: {out}')
    directory = inputs.base/'data'/args.batch_name
    directory.mkdir(parents=True, exist_ok=False)
    return args, inputs, cases, directory


def freeze(directory):
    paths = {V.CPU_PATH.resolve(), V.BASELINE/'step4_connect_support/translucent_raster.cpp'}
    for module in list(sys.modules.values()):
        file = getattr(module, '__file__', None)
        if file:
            path = Path(file).resolve()
            if path.is_relative_to(ROOT) and not path.is_relative_to(ROOT/'.venv') and path.is_file():
                paths.add(path)
    records = {}
    for path in sorted(paths):
        target = directory/'executed_sources'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        records[relative(path)] = digest(path)
        if digest(target) != records[relative(path)]: raise ValueError('Source snapshot mismatch')
    return records


def begin(directory, step, inputs, cases):
    code = freeze(directory)
    save(directory/'execution.json', dict(step=step, started_utc=datetime.now(timezone.utc).isoformat(),
        object=inputs.name, groups=[c['id'] for c in cases], selection=relative(inputs.selection_path),
        executed_code=code, step4_rerun=False, step51_rerun=False, force_solver_run=False))
    return code


def finish(directory, inputs, code, summary):
    inputs.verify()
    for path, expected in code.items():
        if digest(ROOT/path) != expected or digest(directory/'executed_sources'/path) != expected:
            raise ValueError(f'Executed code changed: {path}')
    save(directory/'summary.json', dict(summary, complete=True, upstream_files_unchanged=True,
        source_and_execution_snapshot_hashes_match=True, protected_input_count=len(inputs.protected)))
    save(directory/'verification.json', dict(complete=True, protected_inputs=inputs.protected,
        executed_code=code, upstream_files_unchanged=True, source_and_snapshot_hashes_match=True))
    A.finish_batch(directory)
