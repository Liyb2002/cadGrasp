"""Fresh disk replay of the actual optimized mesh and every task load domain."""
from pathlib import Path
import json
import numpy as np
import trimesh
from subdivide_task.step1.common import DEFAULT_OUTPUT, sha256, write_json
from subdivide_task.operations import Layout
from subdivide_task.operations.construction import CompactConstructionModel


def validate(folder, output_root=DEFAULT_OUTPUT, name='B'):
    folder = Path(folder)
    current = json.loads((folder/'current.json').read_text())
    path = folder/current['state']
    if sha256(path) != current['sha256']:
        raise ValueError('Current saved geometry changed')
    model = CompactConstructionModel(output_root, name)
    if sha256(model.folder/'report.json') != current['initial_report_sha256']:
        raise ValueError('Optimization no longer binds the accepted Step3.2')
    previous = model.report['material_volume_cm3']
    checkpoints = []
    previous_groups = [[i] for i in range(len(model.ids))]
    previous_path = None
    for index in range(current['accepted']+1):
        state = folder/f'state_{index:03}.npz'
        record = json.loads(state.with_suffix('.json').read_text())
        if sha256(state) != record['mesh_sha256']:
            raise ValueError('Changed accepted checkpoint: '+str(state))
        if index:
            audit = record['audit']
            assert audit['passed'] and audit['full_loads'] and audit['source_face_prisms_checked']
            assert len(audit['tasks']) == len(model.ids)
            if audit.get('acceptance_scope')=='merged_family_with_unchanged_family_certificates':
                assert audit['parent_state_sha256']==sha256(previous_path)
                fresh=set(audit['freshly_checked_tasks'])
                retained=set(audit['retained_tasks'])
                assert fresh.isdisjoint(retained) and fresh|retained==set(model.ids)
                owners=sorted(model.ids.index(name) for name in fresh)
                inputs=[g for g in previous_groups if any(k in owners for k in g)]
                assert len(inputs)==2 and sorted(k for g in inputs for k in g)==owners
                previous_groups=[g for g in previous_groups if g not in inputs]+[owners]
                assert {r['task'] for r in audit['unchanged_family_guards']}==retained
                with np.load(previous_path) as old, np.load(state) as new:
                    outside=[model.ids.index(n) for n in retained]
                    for key in ('poses','material','exits'):
                        np.testing.assert_array_equal(old[key][outside],new[key][outside])
            for row in audit['tasks']:
                proof = row['force']
                assert proof['passed'] and not proof.get('preflight_only',False)
                assert proof['original_sample_count'] == proof['passed_original_sample_count'] == 32768
                assert proof['continuous_outer_target_count'] == proof['passed_continuous_outer_target_count']
            assert record['volume_cm3'] < previous-1e-5
        else:
            with np.load(state) as z:
                np.testing.assert_array_equal(z['vertices'],model.initial_support.vertices)
                np.testing.assert_array_equal(z['faces'],model.initial_support.faces)
        previous = record['volume_cm3']
        previous_path=state
        checkpoints.append(dict(index=index,volume_cm3=previous,operation=record['operation'],sha256=sha256(state),
            freshly_checked_tasks=record['audit'].get('freshly_checked_tasks'),
            retained_task_certificates=record['audit'].get('retained_tasks')))
    if any(c['freshly_checked_tasks'] is not None for c in checkpoints):
        assert current['groups']==previous_groups
    with np.load(path) as z:
        layout = Layout.read(z)
        support = trimesh.Trimesh(z['vertices'],z['faces'],process=False)
    ribs=record.get('construction',{}).get('ribbed_stock')
    if ribs:model.stock_grid=(ribs['pitch_m'],ribs['post_width_m'])
    tasks = model.prepare(layout)
    candidate = dict(layout=layout,support=support,tasks=tasks,volume_cm3=float(support.volume*1e6))
    audit = model.audit(candidate,full=True,continuous=True)
    assert abs(candidate['volume_cm3']-current['volume_cm3'])<1e-7
    assert candidate['volume_cm3']<model.report['material_volume_cm3']-1e-5
    result = dict(status='passed',fresh_disk_replay=True,all_tasks_on_actual_final_material=True,
        task_count=len(tasks),original_sample_count=sum(r['force']['original_sample_count']for r in audit['tasks']),
        continuous_outer_target_count=sum(r['force']['continuous_outer_target_count']for r in audit['tasks']),
        initial_volume_cm3=model.report['material_volume_cm3'],final_volume_cm3=candidate['volume_cm3'],
        volume_reduction_fraction=1-candidate['volume_cm3']/model.report['material_volume_cm3'],
        monotone_accepted_checkpoints=checkpoints,audit=audit,source_sha256=sha256(__file__),
        source_step32_sha256=sha256(model.folder/'report.json'),final_state_sha256=sha256(path),
        geometry_tolerance_m3=model.tolerance,force_tolerance=2e-9,
        scope='original frictionless unilateral unbounded normal-reaction model; one rigid fixture with separately audited SO(3) installations and actual floor patches')
    write_json(folder/'validation.json',result)
    print('STEP4 DISK REPLAY PASS',len(tasks),'tasks',round(candidate['volume_cm3'],3),'cm3',flush=True)
    return model,candidate,result
