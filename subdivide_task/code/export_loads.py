"""Export the existing Step3 load domains without changing the load algorithm."""
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from subdivide_task.step1.common import ROOT, sha256, write_json
from subdivide_task.step3.tool_membranes import load_model, stem


folder = ROOT/'subdivide_task/data/B/step3_tool_membranes'
data = ROOT/'subdivide_task/data/B/step2'; data.mkdir(parents=True, exist_ok=True)
output = ROOT/'subdivide_task/output/B/step2'; output.mkdir(parents=True, exist_ok=True)
model, layout, initialization = load_model(folder)
proof = json.loads((folder/'validation.json').read_text())
assert proof['status'] == 'passed' and proof['report_sha256'] == sha256(folder/'report.json')
tasks = []; sample_sets = []; max_residual = 0.
for i, record in enumerate(model.records):
    row = model.world_data(record, layout.poses[i])
    domain, endpoints, samples = row['domain'], row['endpoints'], row['samples']
    name = stem(record['id'])
    screened = np.isin(samples['source_face_id'], record['certified_face_ids'])
    samples['face_passes_pose_geometry'] = screened
    samples['geometrically_screened'] = screened.copy()
    np.savez_compressed(data/f'{name}.npz', **samples,
                        **{'continuous_'+key:value for key,value in endpoints.items()})
    domain.data['geometry']['face_passes_pose_geometry'] = np.isin(domain.work_ids, record['certified_face_ids']).tolist()
    domain.data['geometry']['filled_face_ids'] = np.setdiff1d(domain.work_ids, record['certified_face_ids']).tolist()
    domain.data['source_pose_status'] = 'validated_step3_initialization'
    write_json(data/f'{name}_domain.json', domain.data)
    # Verify the serialized six-dimensional pairs from their physical inputs.
    with np.load(data/f'{name}.npz') as saved:
        required = np.c_[-(saved['force_push_mg']+saved['gravity_force_mg']),
                         -np.cross(saved['pt_m']-saved['moment_origin_m'], saved['force_push_mg'])]
        np.testing.assert_allclose(required, saved['need_wrench'], atol=1e-13, rtol=0)
        np.testing.assert_allclose(domain.forward(saved['work_face_index'], saved['parameters']), required, atol=1e-13, rtol=0)
        np.testing.assert_allclose(saved['need_wrench_SI'], required*domain.weight_N, atol=1e-13, rtol=0)
        assert np.isin(saved['source_face_id'], record['working_face_ids']).all()
        assert len(saved['need_wrench']) == initialization['task_results'][i]['audit']['tasks'][0]['force']['original_sample_count']
        assert len(saved['continuous_endpoint_need_wrench']) == initialization['task_results'][i]['audit']['tasks'][0]['force']['continuous_outer_target_count']
        max_residual = max(max_residual, float(np.abs(required-saved['need_wrench']).max()))
        sample_sets.append(saved['need_wrench'].copy())
    tasks.append(dict(id=record['id'], arrays=f'{name}.npz', domain=f'{name}_domain.json',
                      working_face_count=len(record['working_face_ids']),
                      certified_face_count=len(record['certified_face_ids']),
                      sample_count=len(samples['need_wrench']),
                      continuous_outer_target_count=len(endpoints['endpoint_need_wrench']),
                      world_from_object=layout.poses[i].tolist(), moment_origin_world_m=domain.com.tolist(),
                      weight_N=domain.weight_N))

# The force cone is shared in world coordinates; moments depend on each pose
# and its own cleaned pose region. These are projections of the saved paired data.
plt.rcParams.update({'font.size':10, 'axes.titleweight':'medium', 'savefig.facecolor':'white'})
panel_rows = (len(sample_sets)+2)//2
fig = plt.figure(figsize=(14, 5.3*panel_rows))
fig.subplots_adjust(left=.04, right=.94, bottom=.05, top=.97, wspace=.12, hspace=.30)
pink = '#d83c88'
panels = [(sample_sets[0][:, :3], 'Required force (shared world cone)', ('Fx [mg]', 'Fy [mg]', 'Fz [mg]'))]
panels += [(sample[:, 3:], task['id']+' | Required moment about COM', ('tx [mg m]', 'ty [mg m]', 'tz [mg m]'))
           for task, sample in zip(tasks, sample_sets)]
torque_limit = max(float(np.abs(sample[:, 3:]).max()) for sample in sample_sets)*1.08
for i, (values, title, labels) in enumerate(panels):
    ax = fig.add_subplot(panel_rows, 2, i+1, projection='3d')
    ids = np.linspace(0, len(values)-1, 2400).astype(int)
    ax.scatter(*values[ids].T, s=2.6, color=pink, alpha=.32, edgecolors='none', rasterized=True)
    ax.set_title(title, pad=14)
    ax.set_xlabel(labels[0], labelpad=8); ax.set_ylabel(labels[1], labelpad=8); ax.set_zlabel(labels[2], labelpad=8)
    if i:
        for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):setter(-torque_limit, torque_limit)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=22, azim=-55)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill=False
        axis.set_major_locator(MaxNLocator(4))
fig.savefig(output/'force_moment.png', dpi=160, bbox_inches='tight', pad_inches=.3)
plt.close(fig)

exporter = data/'export_loads.py'
if Path(__file__).resolve() != exporter.resolve():shutil.copy2(__file__, exporter)
inputs = {str(p):sha256(p) for p in [folder/'report.json', folder/'validation.json', folder/'initialization.npz',
                                   ROOT/'subdivide_task/common/wrenches.py', ROOT/'subdivide_task/step3/independent_physics.py', exporter]}
report = dict(status='exported_and_replayed', schema='step2_current_initialization_loads_v1', object='B',
              computation='existing Step3 world_data / demands; no separate force algorithm',
              input_pose_count=initialization['input_task_count'], task_count=len(tasks),
              skipped_empty_task_ids=initialization['skipped_empty_task_ids'], tasks=tasks,
              paired_order=['Fx','Fy','Fz','tau_x','tau_y','tau_z'],
              required_reaction='[-g-P, -cross(q-COM,P)]', force_units='mg', moment_units='mg*m',
              SI_conversion='multiply all six components by task weight_N',
              nominal_axis_world=model.axis.tolist(), half_angle_deg=model.policy['direction_half_angle_deg'],
              processing_force_range_mg=[0.,model.policy['maximum_processing_force_mg']],
              sample_count=sum(t['sample_count'] for t in tasks),
              continuous_outer_target_count=sum(t['continuous_outer_target_count'] for t in tasks),
              plot=dict(file='force_moment.png', definition='3D projections of paired 6D reaction demands',
                        shown_samples_per_panel=2400, sha256=sha256(output/'force_moment.png')),
              input_sha256=inputs,
              artifact_sha256={p.name:sha256(p) for p in sorted(data.iterdir()) if p.suffix in ('.npz','.json') and p.name not in ('report.json','validation.json')})
write_json(data/'report.json', report)
write_json(data/'validation.json', dict(status='passed', report_sha256=sha256(data/'report.json'),
    paired_mapping_replayed_from_serialized_data=True, maximum_paired_mapping_residual=max_residual,
    sample_count=report['sample_count'], continuous_outer_target_count=report['continuous_outer_target_count']))
shutil.copy2(data/'report.json', output/'report.json')
shutil.copy2(data/'validation.json', output/'validation.json')
print(json.dumps(dict(status='STEP2 EXPORT PASS', tasks=len(tasks), samples=report['sample_count'],
                      continuous_targets=report['continuous_outer_target_count'], output=str(output/'force_moment.png')), indent=2))
