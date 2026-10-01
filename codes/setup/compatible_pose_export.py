"""Validate and publish pose-only tasks without claiming a robot trajectory."""
import itertools
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import trimesh

from sequence_export import ROOT, WORK_AREA_FRACTION, digest, write
from compatible_pose_search import local_demands


def verify(folder):
    """Replay all loads and check saved witnesses independently of coarse search."""
    folder = Path(folder)
    record = json.loads((folder/'poses.json').read_text())
    assert record['schema'] == 'cadgrasp_pose_set_v1'
    count = record['pose_count']
    names = [f'pose_{i}' for i in range(1, count+1)]
    assert [p['pose_id'] for p in record['poses']] == names
    assert json.loads((folder/'tasks.json').read_text())['poses'] == names
    assert record['placement_trajectory_verified'] is False
    assert not any((folder/f).exists() for f in ('trajectory.npz', 'video.mp4', 'trajectories'))
    raw = trimesh.load(folder/'mesh.stl', force='mesh')
    assert record['mesh_sha256'] == digest(folder/'mesh.stl')
    meshes = {0: raw}
    frames, clouds, floors = [], [], []
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    for name, pose in zip(names, record['poses']):
        task = folder/'tasks'/name
        report = json.loads((task/'setup.json').read_text())
        assert report['source_snapshot_sha256'] == digest(task/'setup.npz')
        with np.load(task/'setup.npz') as arrays:
            transform = arrays['T_world_mesh'].copy()
            mask = arrays['work_faces'].copy()
            pivot, com = arrays['floor_contact_m'].copy(), arrays['com_m'].copy()
            assert str(arrays['poses_sha256']) == digest(folder/'poses.json')
            assert str(arrays['mesh_sha256']) == digest(folder/'mesh.stl')
            assert float(arrays['K']) == .5 and float(arrays['cone_half_deg']) == 30.
        np.testing.assert_array_equal(transform, pose['T_world_mesh'])
        np.testing.assert_allclose(transform[:3, :3].T@transform[:3, :3], np.eye(3), atol=1e-12, rtol=0)
        assert abs(np.linalg.det(transform[:3, :3])-1) < 1e-12
        np.testing.assert_array_equal(transform[3], [0., 0., 0., 1.])
        vertices = raw.vertices@transform[:3, :3].T+transform[:3, 3]
        assert abs(vertices[:, 2].min()) < 1e-9 and np.count_nonzero(vertices[:, 2] < 1e-8) == 1
        np.testing.assert_allclose(pivot, vertices[np.argmin(vertices[:, 2])], atol=1e-9, rtol=0)
        np.testing.assert_allclose(com, transform[:3, :3]@raw.center_mass+transform[:3, 3], atol=1e-12, rtol=0)
        assert np.linalg.norm(com[:2]-pivot[:2]) >= .001
        rounds = report['uniform_subdivision_rounds']
        for level in range(1, rounds+1):
            if level not in meshes:
                meshes[level] = meshes[level-1].subdivide()
        mesh = meshes[rounds]
        assert mask.shape == (len(mesh.faces),) and mask.dtype == bool
        fraction = float(mesh.area_faces[mask].sum()/mesh.area)
        assert WORK_AREA_FRACTION[0] <= fraction <= WORK_AREA_FRACTION[1]
        np.testing.assert_allclose(fraction, report['checks']['work_area_fraction'], atol=1e-12, rtol=0)
        world = mesh.vertices@transform[:3, :3].T+transform[:3, 3]
        assert world[mesh.faces[mask], 2].min() > .0015
        assert (mesh.face_normals[mask]@transform[2, :3]).min() > .35
        ids = np.flatnonzero(mask)
        remap = np.full(len(mask), -1); remap[ids] = np.arange(len(ids))
        pairs = mesh.face_adjacency[mask[mesh.face_adjacency].all(axis=1)]
        edges = remap[pairs]
        graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(ids), len(ids)))
        assert connected_components(graph, directed=False, return_labels=False) == 1
        _, local, samples = local_demands(mesh, transform, mask, record['object'])
        # Independent balance about world origin from raw applied positions/forces.
        force = np.array([0., 0., 1.])-np.asarray(samples['force_push_mg'])
        moment = np.cross(com, [0., 0., 1.])-np.cross(samples['pt_m'], samples['force_push_mg'])
        ground = np.c_[-moment[:, 1]/force[:, 2], moment[:, 0]/force[:, 2], np.zeros(len(force))]
        independent = (ground-transform[:3, 3])@transform[:3, :3]
        np.testing.assert_allclose(local, independent, atol=1e-12, rtol=0)
        with np.load(folder/'pose_search'/f'{name}.npz') as saved:
            np.testing.assert_array_equal(saved['floor_demands_object_m'], local)
            np.testing.assert_array_equal(saved['load_wrenches'], samples['need_wrench'])
        frames.append(transform); clouds.append(independent); floors.append(transform[2])
    # Demand points stay attached to the object; every target floor is an affine halfspace.
    heights = np.asarray([[cloud@f[:3]+f[3] for f in floors] for cloud in clouds])
    violations = (heights < -1e-9).sum(axis=2)
    adjacency = [set(np.flatnonzero((violations[i] == 0) & (violations[:, i] == 0)))-{i} for i in range(count)]
    pair_angles = [np.degrees(np.arccos(np.clip(a[2, :3]@b[2, :3], -1., 1.))) for a, b in itertools.combinations(frames, 2)]
    assert min(pair_angles) >= record['pose_selection']['minimum_pairwise_gravity_direction_deg']-1e-8
    result = dict(passed=True, object=record['object'], pose_count=count,
        sample_count_per_pose=32768, tolerance_m=1e-9,
        minimum_pairwise_gravity_direction_deg=float(min(pair_angles)),
        directed_violating_counts=violations.tolist(), minimum_heights_m=heights.min(axis=2).tolist(),
        witnesses={}, compatible_group_counts={}, placement_trajectory_verified=False,
        complete_fixture_verified=False,
        scope='Grounded target poses and original sampled floor demands only; robot motion and fixture feasibility unverified')
    for size in range(2, record['pose_selection']['compatible_size']+1):
        groups = [g for g in itertools.combinations(range(count), size)
                  if all(j in adjacency[i] for i, j in itertools.combinations(g, 2))]
        assert groups, f'Missing compatible {size}-pose group'
        result['compatible_group_counts'][str(size)] = len(groups)
        result['witnesses'][str(size)] = [names[i] for i in groups[0]]
        planned = record['pose_selection']['witnesses'].get(str(size))
        if planned:
            indices = [names.index(p) for p in planned]
            assert all(j in adjacency[i] for i, j in itertools.combinations(indices, 2))
    result['provenance'] = dict(poses_sha256=digest(folder/'poses.json'),
        task_inputs={name: digest(folder/'tasks'/name/'setup.npz') for name in names},
        verifier_sha256=digest(__file__))
    return result


def publish(name, plan_folder):
    plan_folder = Path(plan_folder)
    plan = json.loads((plan_folder/'plan.json').read_text())
    folder = ROOT/'objects'/name
    assert plan['object'] == name and plan['mesh_sha256'] == digest(folder/'mesh.stl')
    assert plan['generator_sha256'] == digest(ROOT/plan['generator'])
    old_hash = digest(folder/'poses.json')
    revision = f'before_compatible_poses_{old_hash[:12]}'
    history = folder/'history'/revision
    if history.exists():
        raise FileExistsError(f'History destination exists: {history}')
    with tempfile.TemporaryDirectory(prefix=f'.{name}_pose_set_', dir=folder.parent) as tmp:
        stage = Path(tmp)
        shutil.copy2(folder/'mesh.stl', stage/'mesh.stl')
        shutil.copytree(plan_folder, stage/'pose_search')
        names = [row['pose_id'] for row in plan['poses']]
        record = dict(schema='cadgrasp_pose_set_v1', object=name, pose_count=len(names),
            coordinate_system='z_up_xy_floor', mesh_sha256=plan['mesh_sha256'], rest=plan['rest'],
            poses=[dict(pose_id=r['pose_id'], index=i, T_world_mesh=r['T_world_mesh'],
                        grounded=True, target_pose_only=True) for i, r in enumerate(plan['poses'], 1)],
            order=['rest']+names, placement_trajectory_verified=False, sequence_continuous=False,
            pose_selection=dict(method='Seeded orientation pool and exact compatible-subset search',
                seed=plan['seed'], compatible_size=plan['compatible_size'], witnesses=plan['witnesses'],
                minimum_pairwise_gravity_direction_deg=18.,
                candidate_count=plan['candidate_count'], attempted_count=plan['attempted_count'],
                sample_count_per_pose=32768, sample_seed=plan['sample_seed']),
            floor_compatibility_report='floor_compatibility.json',
            verification_scope='Pose geometry and sampled floor-demand compatibility. No robot trajectory or complete fixture certificate.',
            generator='codes/setup/compatible_pose_search.py', generator_sha256=digest(ROOT/'codes/setup/compatible_pose_search.py'),
            previous_pose_manifest_sha256=old_hash, previous_inputs='history/'+revision)
        write(stage/'poses.json', record)
        raw = trimesh.load(stage/'mesh.stl', force='mesh')
        for row in plan['poses']:
            path = plan_folder/row['arrays']
            assert digest(path) == row['sha256']
            with np.load(path) as arrays:
                transform, mask = arrays['T_world_mesh'].copy(), arrays['work_faces'].copy()
            pose = row['pose_id']; target = stage/'tasks'/pose; target.mkdir(parents=True)
            vertices = raw.vertices@transform[:3, :3].T+transform[:3, 3]
            point = vertices[np.argmin(vertices[:, 2])].copy(); point[2] = 0.
            com = transform[:3, :3]@raw.center_mass+transform[:3, 3]
            np.savez_compressed(target/'setup.npz', object=name, pose_id=pose, T_world_mesh=transform,
                com_m=com, work_faces=mask, floor_contact_m=point, mesh_sha256=plan['mesh_sha256'],
                poses_sha256=digest(stage/'poses.json'), K=.5, cone_half_deg=30., tip=-1)
            write(target/'setup.json', dict(schema_version=1, object=name, pose_id=pose,
                target_pose_only=True, placement_trajectory_verified=False, support_search_run=False,
                T_world_mesh=transform.tolist(), floor_contact_m=point.tolist(),
                work_face_ids=np.flatnonzero(mask).tolist(), uniform_subdivision_rounds=plan['uniform_subdivision_rounds'],
                K=.5, cone_half_deg=30., source_snapshot='setup.npz', source_snapshot_sha256=digest(target/'setup.npz'),
                checks=dict(**row['area'], ground_min_z_m=float(vertices[:, 2].min()),
                    floor_contact_raw_vertex_ids=np.flatnonzero(vertices[:, 2] < 1e-8).tolist(), com_world_m=com.tolist()),
                work_region_method='Seeded connected 6-10% visible upward patch, fixed before compatibility selection',
                work_region_key=f'{name}/compatible_{plan["seed"]}_{row["candidate_index"]}',
                generator='codes/setup/compatible_pose_search.py'))
        write(stage/'tasks.json', dict(schema='cadgrasp_tasks_v1', object=name, poses=names,
            rest='poses.json:rest', sequence='poses.json', placement_trajectory_verified=False,
            definition=f'{len(names)} grounded target poses with certified sampled floor-compatible subsets; no robot trajectory'))
        checked = verify(stage)
        write(stage/'floor_compatibility.json', checked)
        # All validation precedes replacement. Preserve prior files and derived
        # outputs so identical pose labels cannot silently refer to old tasks.
        history.mkdir(parents=True)
        replaced, installed = [], []
        filenames = ('poses.json', 'tasks.json', 'tasks', 'trajectory.npz', 'trajectories',
                     'video.mp4', 'overview.png', 'pose_search', 'floor_compatibility.json')
        output = ROOT/'slides/baseline_algo/output'/name
        old_output = [p for p in output.iterdir() if p.name != 'history'] if output.exists() else []
        output_history = output/'history'/revision
        if old_output:
            output_history.mkdir(parents=True, exist_ok=False)
        try:
            for filename in filenames:
                source = folder/filename
                if source.exists():
                    source.rename(history/filename); replaced.append(filename)
            for filename in ('poses.json', 'tasks.json', 'tasks', 'pose_search', 'floor_compatibility.json'):
                (stage/filename).rename(folder/filename); installed.append(filename)
            for path in old_output:
                path.rename(output_history/path.name)
        except Exception:
            for filename in installed:
                source = folder/filename
                if source.exists():
                    if source.is_dir(): shutil.rmtree(source)
                    else: source.unlink()
            for filename in replaced:
                (history/filename).rename(folder/filename)
            for path in old_output:
                if (output_history/path.name).exists():
                    (output_history/path.name).rename(path)
            raise
        write(history/'migration.json', dict(reason='Active object poses regenerated with floor-compatible subset selection',
            prior_pose_manifest_sha256=old_hash, current_pose_manifest_sha256=digest(folder/'poses.json'),
            baseline_output_history=str(output_history.relative_to(ROOT)) if old_output else None))
    print('PUBLISHED', name, len(names), 'poses', checked['compatible_group_counts'], flush=True)
    return checked
