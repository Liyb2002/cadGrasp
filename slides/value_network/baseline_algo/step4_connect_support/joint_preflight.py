"""Check saved N-pose Step3 contacts before constructing a shared fixture.

This entry diagnoses prerequisites only; it does not build a fixture or repair
an incomplete contact search. All loads are the original saved Step1 samples.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step3_scheculer.pair_tasks import read_task
from step0_pose_selection.floor_points import pressure_centers
from step4_connect_support import head_registration as H


def run(group):
    group = Path(group).resolve()
    source = group/'step3_scheculer/joint_weighted'
    schedule = json.loads((source/'schedule.json').read_text())
    assert schedule['complete']
    I.check_hashes(schedule['provenance']['inputs'])
    poses, result = schedule['poses'], schedule['result']
    tasks = [read_task(schedule['object'], p, folder=group/'step_1_needs'/p) for p in poses]
    particle = source/f'particle_{schedule["winner"]:03d}'
    for name, digest in result['artifacts'].items():
        assert I.sha256(particle/name) == digest
    with np.load(particle/'coverage.npz') as masks:
        assert [int(masks[p].sum()) for p in poses] == result['covered_counts']
    groups = [I.read_contacts(particle/f'contacts_{p}.npz') for p in poses]
    bases, offsets = H.fixed_placements(tasks)
    seen, registration = {}, []
    adjacency = [set() for _ in poses]
    for k, contacts in enumerate(groups):
        for contact in contacts:
            ident = contact['candidate_id']
            patch = contact['triangles_m'].reshape(-1, 3)@bases[k]+offsets[k]
            assert np.linalg.matrix_rank(patch-patch.mean(0), tol=1e-10) >= 2
            if ident in seen:
                owner, original = seen[ident]
                error = H.cloud_error(original, patch)
                assert error <= H.POSITION_TOL
                adjacency[k].add(owner); adjacency[owner].add(k)
                registration.append(dict(id=ident, poses=[poses[owner], poses[k]], error_m=error))
            else:
                seen[ident] = k, patch
    reached, pending = set(), [0]
    while pending:
        k = pending.pop()
        if k not in reached:
            reached.add(k); pending.extend(adjacency[k]-reached)
    if len(reached) != len(poses):
        raise ValueError('Shared-head graph is disconnected; relative component placements remain free')
    demands = [pressure_centers(t.targets/t.scale, t.domain.com)[0] for t in tasks]
    assert all(len(xy) == 32768 for xy in demands)
    rows = []
    for k in range(len(poses)):
        for j in range(len(poses)):
            if k == j:
                continue
            check = H.floor_compatibility([tasks[k], tasks[j]], [demands[k], demands[j]],
                                         bases[[k, j]], offsets[[k, j]])['per_pose'][0]
            # Independently check the affine certificate by transforming every
            # original ground demand into the other pose's world coordinates.
            points = np.c_[demands[k], np.zeros(len(demands[k]))]@bases[k]+offsets[k]
            direct_heights = ((points-offsets[j])@bases[j].T)[:, 2]
            coefficients = np.asarray(check['floor_xy_halfplane_coefficients'])
            np.testing.assert_allclose(direct_heights, demands[k]@coefficients[:2]+coefficients[2],
                                       atol=1e-13, rtol=0)
            assert int((direct_heights < -H.FLOOR_TOL).sum()) == check['violating_sample_count']
            rows.append(check)
    floor_passed = all(r['passed'] for r in rows)
    ready = bool(result['passed'] and floor_passed)
    report = dict(schema='joint_step5_preflight_v1', object=schedule['object'], poses=poses,
        complete=False, constructed=False, passed=False if not ready else None,
        status='blocked_before_body_growth' if not ready else 'prerequisites_passed',
        step3=dict(passed=result['passed'], status=result['status'], heads=result['heads'],
                   covered_counts=result['covered_counts'], sample_counts=result['sample_counts']),
        registration=dict(shared_contact_surfaces_coincide=True, connected_shared_head_graph=True,
                          physical_contact_ids=len(seen), shared_instances=registration,
                          bases=bases.tolist(), offsets=offsets.tolist(),
                          scope='Original complete contact correspondence; head solids not rechecked here'),
        floor_compatibility=dict(passed=floor_passed, per_ordered_pose_pair=rows,
            scope='This fixed shared-contact layout, massless support and unilateral planar floors only'),
        independent_affine_coordinate_check=True, original_sample_count_per_pose=32768,
        extra_loads_added=False, full_verification_performed=False, search_rerun=False,
        source_schedule=str((source/'schedule.json').relative_to(I.ROOT)),
        provenance=dict(inputs=I.hashes([source/'schedule.json', particle/'schedule.json']+
            [particle/f'contacts_{p}.npz' for p in poses]+[particle/'coverage.npz']+
            [p for t in tasks for p in t.inputs]),
            code=I.hashes([Path(__file__), Path(H.__file__)])))
    out = group/'step4'
    data = out/'data'; data.mkdir(parents=True, exist_ok=True)
    I.save(data/'report.json', report)
    draw(out, poses, demands, rows, result)
    (data/'README.md').write_text(
        '本次只运行 Step5 构造前置检查，未生成实体。详见 report.json。\n'
        'Step3 原六头预算结果未全覆盖；同一共享接触配准后的地面必要条件也未通过。\n'
        '未重新搜索、替换头、追加样本或运行完整实体审计。overview.png 是失败诊断图。\n')
    print(json.dumps(dict(output=str(out), status=report['status'], step3=report['step3'],
        floor_conflicts=[r for r in rows if not r['passed']]), indent=2))
    return report


def draw(out, poses, demands, rows, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(poses), figsize=(5*len(poses), 5), squeeze=False)
    for k, (pose, xy, ax) in enumerate(zip(poses, demands, axes[0])):
        worst = max((r for r in rows if r['pose'] == pose), key=lambda r:r['violating_sample_count'])
        co = np.asarray(worst['floor_xy_halfplane_coefficients'])
        failed = xy@co[:2]+co[2] < -H.FLOOR_TOL
        for mask, color, label in ((~failed, '#619087', 'Inside'), (failed, '#d35b4c', 'Outside')):
            points = xy[mask][::max(1, int(mask.sum())//1500)]*1000
            ax.scatter(*points.T, s=4, color=color, alpha=.45, label=label)
        ax.set(xlabel='Floor x (mm)', ylabel='Floor y (mm)', aspect='equal',
               title=f'{pose}: Step3 {result["covered_counts"][k]:,}/32,768\n'
                     f'Below {worst["other_pose"]} floor: {worst["violating_sample_count"]:,}')
    fig.suptitle('Step5 preflight | '+', '.join(poses), fontsize=16)
    fig.text(.5, .025, 'Original sampled ground demands under one shared-head registration. No fixture was constructed.',
             ha='center', fontsize=11)
    fig.tight_layout(rect=(0,.07,1,.90))
    fig.savefig(out/'overview.png', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', type=Path)
    run(parser.parse_args().group)
