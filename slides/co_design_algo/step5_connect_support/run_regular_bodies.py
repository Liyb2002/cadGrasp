"""Reconstruct an accepted unified material network as a few planar bodies."""
import argparse
from contextlib import redirect_stdout
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step1.needs import ROOT, OUTPUTS
from step3_scheculer import contacts as I
from step5_connect_support import material_graph as M, material_network as N
from step5_connect_support import regular_body_geometry as B, build_coupled_saddle as S
from step5_connect_support import run_unified_growth as R
from step5_connect_support.run_greedy import read_case
from step5_connect_support.run_local_bodies import Tee

SCHEMA = 'unified_material_regular_bodies_v1'


def preserve_grid(out):
    """Keep the exact previous network as reproducible input and comparison."""
    report = json.loads((out/'report.json').read_text())
    if report['schema'] == SCHEMA and (out/'grid_report.json').exists(): return
    if report['schema'] != R.SCHEMA or not report['passed']:
        raise ValueError('Run and accept unified growth before regular-body reconstruction')
    if (out/'grid_report.json').exists() and I.sha256(out/'grid_report.json') == I.sha256(out/'report.json'): return
    for name, digest in report['artifacts'].items():
        if I.sha256(out/name) != digest: raise ValueError(f'Modified seed artifact: {name}')
    names = list(report['artifacts'])+['report.json','overview.png','fixture.png','index.html',
        'independent_audit.json','viewer_check.json']
    for name in names:
        if (out/name).exists(): shutil.copy2(out/name, out/f'grid_{name}')


def run_pair(pair, max_fill_ratio=8.):
    out = pair/'step5'; preserve_grid(out)
    source = json.loads((out/'grid_report.json').read_text())
    reference = json.loads((out/'reference_report.json').read_text())
    case = read_case((ROOT/source['source_schedule']).parent)
    work = Path(tempfile.gettempdir())/'cadgrasp_unified_growth'/pair.name
    work.mkdir(parents=True, exist_ok=True)
    context = M.prepare(case, source['placement'], work)
    dimensions = source['body_design']['graph']
    graph = M.build_graph(case, context, work, dimensions['step_m'], dimensions['padding_m'], dimensions['minimum_joint_area_m2'])
    seed_files = [out/f'grid_{name}' for name in ('report.json','material_graph.npz','geometry.npz')]
    for path in seed_files[1:]:
        assert I.sha256(path) == source['artifacts'][path.name.removeprefix('grid_')]
    with np.load(out/'grid_material_graph.npz') as saved:
        np.testing.assert_array_equal(saved['edges'], graph['edges'])
        np.testing.assert_allclose(saved['cost_cm3'], graph['cost'], rtol=0, atol=1e-10)
        selected = saved['selected'].copy()
    full, shape = B.reconstruct(graph, selected, context, max_fill_ratio)
    mesh = S.unpack(full)
    solid = dict(component_count=len(full.decompose()), watertight=bool(mesh.is_watertight),
        consistently_wound=bool(mesh.is_winding_consistent), volume_m3=float(mesh.volume))
    solid['one_solid'] = solid['component_count'] == 1 and solid['watertight'] and solid['consistently_wound'] and mesh.volume > 0
    if not solid['one_solid']: raise RuntimeError(f'Regular reconstruction is not one closed solid: {solid}')
    floor_checks = []
    for k in (0, 1):
        world = (mesh.vertices-context['offsets'][k])@context['bases'][k].T
        floor_checks.append(N.containment(world[np.abs(world[:, 2]) < 1e-9, :2], case.demands[k]))
    if not all(c['passed'] for c in floor_checks): raise RuntimeError('Regular bodies lost original ground coverage')
    print('REGULAR', pair.name, 'grid cm3', source['volume_cm3'], 'regular cm3', mesh.volume*1e6, flush=True)
    checks, certificate = S.verify(case.tasks, case.groups, case.heads, context['directions'],
        context['bases'], context['offsets'], mesh, context['sweeps'])
    result = dict(full=full, mesh=mesh, solid=solid, checks=checks, certificate=certificate,
        floor_checks=floor_checks, passed=True, selected=selected, regularization=shape['export_regularization'])
    design = copy.deepcopy(source['body_design'])
    design.update(method=SCHEMA, regular_shape=shape, seed_source=dict(
        schema=source['schema'], artifacts=I.hashes(seed_files),
        generation_code_hashes=source['provenance']['code'], volume_cm3=source['volume_cm3']),
        strict_five_head_identity_passed=False,
        duplicate_yellow_explanation='Two pose-specific physical patches have the same Step3 selected_id. They are not one registered shared head.')
    case.paths += seed_files
    stage = work/'regular'; stage.mkdir(exist_ok=True)
    report = R.publish(out, stage, case, context, source['placement'], graph, result, design, reference,
        schema=SCHEMA, extra_code=[Path(__file__), Path(B.__file__)],
        presentation_description='统一生长确定连接与落脚，再按头的图测地区域重建少量直面凸包身体；完整避障裁切、保留原网络并重新验收。六块物理接触面，两个黄色是同编号的两份实体，并未实现严格五头共享。')
    for name in ('comparison.png','batch_summary.json'): (out/name).unlink(missing_ok=True)
    print('RESULT', pair.name, report['volume_cm3'], report['passed'], flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair', nargs='+', default=['pose1+3','pose1+6','pose2+8'])
    parser.add_argument('--max-fill-ratio', type=float, default=8., help='Split a local hull when its volume exceeds this multiple of its seed material')
    args = parser.parse_args()
    if args.max_fill_ratio < 1: parser.error('--max-fill-ratio must be at least 1')
    for name in args.pair:
        pair = OUTPUTS/'B'/name
        with (pair/'step5/regular_growth.log').open('w') as log, redirect_stdout(Tee(sys.stdout, log)):
            run_pair(pair, args.max_fill_ratio)
