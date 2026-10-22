"""Step5: the merged support of step4/merge.py standing on its plate, certified for every task.

Step4 merged the seats into one support with Juxtaposition and Rotational reuse, every merge
saving material by the gate and audited in full.  With one pose (one face of the support always
down) the accepted support stands on the table: the merged membrane, an upright post under its
lowest part and a plate under the post and every task's pressure hull (step4/merge.stand).  A task
is installed by turning the support on the table about the vertical and sliding its slot to the
station; the arm rises and falls with the object.

Here the standing support is rebuilt from the saved membrane and slots, compared with Step4's, and
every task is audited on it: loads (32,768 paired samples and the continuous outer targets), the
continuous exit of every source face, no overlap with the body or the task's keep-out (needle, head
and KUKA), the KUKA at every stored joint set clear of the whole support (step1.arm's exact test),
and on the table the actual ground faces enclose the task's pressure hull.  Validation rebuilds
everything from disk and audits again.
"""
import argparse
import json
import time
import numpy as np
import trimesh

from whole_task_2.step1.common import ROOT, DEFAULT_OUTPUT, sha256, write_json
from whole_task_2.step3 import independent_geometry as G
from whole_task_2.step4.ground_oracle import pressure_rows, ground_enclosure
from whole_task_2.step4.seats import context, read_mesh
from whole_task_2.step4.merge import FOLDER as MERGE, stand, downs, footprint

FOLDER = 'step5_joined'


def inputs(object_name):
    c = context(object_name); data = ROOT/'whole_task_2/data'/object_name
    report = json.loads((data/MERGE/'report.json').read_text())
    files = [data/MERGE/'report.json', ROOT/report['group']['file'], c.folder/'sets.json', c.folder/'turns.json',
             data/'step4_keepout/report.json', data/'step3_tool_membranes/report.json',
             *[ROOT/'whole_task_2'/p for p in ('step4/merge.py', 'step4/join.py', 'step4/seats.py', 'step4/keepout.py', 'step5/joined.py')]]
    return {str(p.relative_to(ROOT)): sha256(p) for p in files}


def model_from_disk(object_name):
    """The merged support standing in its table frame, rebuilt from Step4's membrane and slots and
    checked against Step4's own standing support; the model with one seat per slot and its layout."""
    c = context(object_name); report = json.loads((ROOT/'whole_task_2/data'/object_name/MERGE/'report.json').read_text())
    group = report['group']
    if len(downs(c, group)) != 1: raise ValueError('Step5 stands one-pose supports only')
    with np.load(ROOT/group['file']) as z:
        membrane = trimesh.Trimesh(z['vertices'], z['faces'], process=False)
        saved = trimesh.Trimesh(z['standing_vertices'], z['standing_faces'], process=False)
    support, _, model, q = stand(c, group['slots'], membrane)
    difference = G.volume(G.solid(support)-G.solid(saved))+G.volume(G.solid(saved)-G.solid(support))
    if difference > c.base.tolerance: raise ValueError(f'Standing support differs from Step4: {difference} m^3')
    return c, model, support, q, report


def certify(model, q, support):
    """Full audit of every task on the support, and its stance on the table."""
    candidate = dict(layout=q, support=support, tasks=model.prepare(q), construction={})
    proof = model.audit(candidate)
    ground = ground_enclosure(support, pressure_rows(model, q, candidate['tasks']))
    if not all(r['passed'] for r in ground): raise ValueError('A pressure hull leaves the ground faces: '+str(
        [(r['id'], r['pressure_hull_gap_m'], r['minimum_world_z_m']) for r in ground if not r['passed']]))
    return proof, ground


def run(object_name='B', output_root=DEFAULT_OUTPUT):
    began = time.monotonic(); c, model, support, q, step4 = model_from_disk(object_name)
    proof, ground = certify(model, q, support)
    folder = ROOT/'whole_task_2/data'/object_name/FOLDER; folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder/'support.npz', vertices=support.vertices, faces=support.faces)
    (folder/'support.obj').write_text(trimesh.exchange.obj.export_obj(support, digits=17, include_normals=False))
    group = step4['group']; names = [c.name(s['members']) for s in group['slots']]
    result = dict(status='passed', object=object_name, slots=names, poses=1, volume_cm3=float(support.volume*1e6),
                  membrane_cm3=group['volume_cm3'], posts_and_plate_cm3=float(support.volume*1e6)-group['volume_cm3'],
                  merges=group['chain'], footprint_cm2=1e4*footprint(support), bounds_m=support.bounds.tolist(),
                  tasks=[dict(id=t['id'], slot=names[model.seat_of[k]], turn_deg=model.records[k].get('angle_deg', 0.),
                              force=dict((key, t['force'][key]) for key in ('passed', 'original_sample_count', 'passed_original_sample_count',
                                         'continuous_outer_target_count', 'passed_continuous_outer_target_count', 'contact_area_m2')),
                              exit_length_m=t['exit_length_m'], overlap_m3=t['overlap_m3'], arm=t['arm'])
                         for k, t in enumerate(proof['tasks'])],
                  ground=[dict(id=r['id'], pressure_hull_gap_m=r['pressure_hull_gap_m'], minimum_world_z_m=r['minimum_world_z_m'],
                               ground_triangles=r['actual_ground_triangle_count']) for r in ground],
                  continuous_paths=proof['continuous_paths'], input_sha256=inputs(object_name),
                  support_sha256=sha256(folder/'support.npz'), seconds=time.monotonic()-began)
    write_json(folder/'report.json', result)
    print('STEP5 JOINED PASS', round(result['volume_cm3'], 1), 'cm3, footprint', round(result['footprint_cm2']), 'cm2', flush=True)
    validate(object_name)
    from .render_joined import render
    render(object_name, output_root)
    return result


def validate(object_name='B'):
    """Disk replay: same inputs; the support rebuilt from Step4's membrane and slots equals the saved one;
    every task audited again."""
    folder = ROOT/'whole_task_2/data'/object_name/FOLDER; report = json.loads((folder/'report.json').read_text())
    if report['input_sha256'] != inputs(object_name): raise ValueError('Step5 inputs changed since the build')
    if sha256(folder/'support.npz') != report['support_sha256']: raise ValueError('Changed support file')
    c, model, rebuilt, q, _ = model_from_disk(object_name); support = read_mesh(folder/'support.npz')
    difference = G.volume(G.solid(support)-G.solid(rebuilt))+G.volume(G.solid(rebuilt)-G.solid(support))
    if difference > c.base.tolerance: raise ValueError(f'Saved support differs from its rebuild: {difference} m^3')
    proof, ground = certify(model, q, support)
    result = dict(status='passed', report_sha256=sha256(folder/'report.json'), rebuilt_symmetric_difference_m3=difference,
                  tasks=[dict(id=t['id'], samples=t['force']['passed_original_sample_count'],
                              outer=t['force']['passed_continuous_outer_target_count'], arm_faces_hit=len(t['arm']['faces_hit']))
                         for t in proof['tasks']],
                  ground=[dict(id=r['id'], passed=r['passed']) for r in ground])
    write_json(folder/'validation.json', result); print('STEP5 JOINED DISK REPLAY PASS', flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--object', default='B'); p.add_argument('--validate-only', action='store_true')
    a = p.parse_args()
    validate(a.object) if a.validate_only else run(a.object)
