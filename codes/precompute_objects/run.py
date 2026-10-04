"""Precompute reusable object poses, immutable loads and compatible task sets."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from codes.precompute_objects.work_regions import digest, write
from codes.precompute_objects.search import search, local_demands
from codes.precompute_objects import loads as L
from codes.precompute_objects.floor_points import check_floor_points, validate_frames, validate_task_geometry
from codes.precompute_objects.cases import selected_pose
import numpy as np
import trimesh


def compatible_sets(frames, clouds, per_size=4, seed=20261003):
    """Every published set passes every directed pair of full sampled demands."""
    result = check_floor_points(clouds, frames, [f'pose_{i+1}' for i in range(len(frames))])
    counts = np.array([[(h[:, j] < -1e-9).sum() for j in range(len(frames))] for h in result.heights])
    adjacent = (counts == 0) & (counts.T == 0)
    rng = np.random.default_rng(seed)
    sets, totals = [], {}
    for size in range(2, 7):
        # Enumerate cliques through their common neighbors, avoiding C(30,6) scans.
        found = []
        def visit(prefix, candidates):
            if len(prefix) == size:
                found.append(tuple(prefix)); return
            for k, vertex in enumerate(candidates):
                remaining = [j for j in candidates[k+1:] if adjacent[vertex, j]]
                if len(prefix)+1+len(remaining) >= size:
                    visit(prefix+[vertex], remaining)
        visit([], list(range(len(frames))))
        totals[str(size)] = len(found)
        if len(found) < per_size:
            raise ValueError(f'Only {len(found)} compatible size-{size} groups; need {per_size}')
        # Spread choices across the pool before random deterministic tie breaking.
        usage = np.zeros(len(frames), int)
        rng.shuffle(found)
        for ordinal in range(per_size):
            index = min(range(len(found)), key=lambda i: (sum(usage[list(found[i])]), max(usage[list(found[i])]), i))
            group = found.pop(index); usage[list(group)] += 1
            names = [f'pose_{i+1}' for i in group]
            sets.append(dict(id='pose'+'+'.join(str(i+1) for i in group), size=size, poses=names, passed=True))
    return sets, totals, counts


def precompute(name, seed=20261003, budget=12000, clean=True, plan_source=None, mesh_source=None, mesh_metadata=None):
    began = time.monotonic()
    folder = ROOT/'objects'/name
    mesh_path = Path(mesh_source) if mesh_source is not None else folder/'mesh.stl'
    with tempfile.TemporaryDirectory(prefix=f'.precompute_{name}_', dir=ROOT/'objects') as tmp:
        stage = Path(tmp)
        plan_folder = stage/'plan'
        if plan_source is None:
            plan = search(name, plan_folder, 30, 7, seed, budget, mesh_source=mesh_source)
        else:
            shutil.copytree(plan_source, plan_folder)
            plan = json.loads((plan_folder/'plan.json').read_text())
            seed = plan['seed']
        if plan['pose_count'] != 30 or plan['object'] != name or plan['mesh_sha256'] != digest(mesh_path):
            raise ValueError('Dataset publication requires a matching 30-pose plan')
        shutil.copy2(mesh_path, stage/'mesh.stl')
        if mesh_metadata is not None:
            write(stage/'meta.json', mesh_metadata)
        raw = trimesh.load(stage/'mesh.stl', force='mesh')
        mesh = raw
        for _ in range(plan['uniform_subdivision_rounds']): mesh = mesh.subdivide()
        manifest = dict(schema='cadgrasp_precomputed_poses_v1', object=name, pose_count=30,
            coordinate_system='z_up_xy_floor', mesh_sha256=digest(stage/'mesh.stl'), rest=plan['rest'],
            poses=[dict(pose_id=r['pose_id'], index=i, T_world_mesh=r['T_world_mesh'], grounded=True,
                        input_folder=f'poses/{r["pose_id"]}') for i,r in enumerate(plan['poses'],1)],
            placement_trajectory_verified=False, complete_fixture_verified=False,
            sample_count_per_pose=32768, sample_seed=L.DEFAULT_SAMPLE_SEED,
            pose_selection=dict(seed=seed, candidate_count=plan['candidate_count'], attempted_count=plan['attempted_count'],
                minimum_pairwise_gravity_direction_deg=18., work_area_fraction=[.06,.10]),
            generator='codes/precompute_objects/run.py', generator_sha256=digest(__file__))
        write(stage/'poses.json', manifest)
        frames, clouds = [], []
        for row in plan['poses']:
            pose = row['pose_id']; target = stage/'poses'/pose; target.mkdir(parents=True)
            with np.load(plan_folder/row['arrays']) as arrays:
                transform, mask = arrays['T_world_mesh'].copy(), arrays['work_faces'].copy()
                original = arrays['load_wrenches'].copy()
                local_cloud = arrays['floor_demands_object_m'].copy()
            world = raw.vertices@transform[:3,:3].T+transform[:3,3]
            pivot = world[np.argmin(world[:,2])].copy(); pivot[2] = 0.
            com = transform[:3,:3]@raw.center_mass+transform[:3,3]
            np.savez_compressed(target/'setup.npz', object=name, pose_id=pose, T_world_mesh=transform,
                com_m=com, work_faces=mask, floor_contact_m=pivot, mesh_sha256=manifest['mesh_sha256'],
                poses_sha256=digest(stage/'poses.json'), K=.5, cone_half_deg=30., tip=-1)
            write(target/'setup.json', dict(object=name, pose_id=pose, uniform_subdivision_rounds=plan['uniform_subdivision_rounds'],
                checks=row['area'], source_snapshot='setup.npz', source_snapshot_sha256=digest(target/'setup.npz'),
                placement_trajectory_verified=False, generator='codes/precompute_objects/run.py'))
            with selected_pose(pose):
                domain_data = L.setup_geometry(name, source=target/'setup.npz', object_path=stage)
            domain_data['provenance']['setup_snapshot'] = str((folder/'poses'/pose/'setup.npz').relative_to(ROOT))
            L.save_json(target/'needs.json', domain_data)
            domain = L.ContinuousNeeds(domain_data)
            samples = L.sample_needs(domain)
            np.testing.assert_array_equal(samples['need_wrench'], original)
            samples['provenance'] = dict(physical_domain_file='needs.json', physical_domain_sha256=digest(target/'needs.json'),
                generator='codes/precompute_objects/loads.py', generator_sha256=digest(Path(L.__file__)))
            L.save_json(target/'samples.json', samples)
            from codes.precompute_objects.floor_points import pressure_centers
            xy, normal = pressure_centers(original, com)
            # Independent raw applied-force balance about world origin.
            force = np.array([0.,0.,1.])-np.asarray(samples['force_push_mg'])
            moment = np.cross(com,[0.,0.,1.])-np.cross(samples['pt_m'],samples['force_push_mg'])
            independent = np.c_[-moment[:,1]/force[:,2],moment[:,0]/force[:,2]]
            np.testing.assert_allclose(xy, independent, atol=1e-12, rtol=0)
            np.testing.assert_allclose((np.c_[xy,np.zeros(len(xy))]-transform[:3,3])@transform[:3,:3], local_cloud, atol=1e-12, rtol=0)
            np.savez_compressed(target/'floor_contact.npz', load_wrenches=original, floor_demands_xy_m=xy,
                total_floor_normal_mg=normal, original_pivot_m=pivot, moment_origin_m=com)
            frames.append(transform); clouds.append(xy)
        frames = validate_frames(frames)
        angles = [np.degrees(np.arccos(np.clip(a[2,:3]@b[2,:3],-1,1))) for a,b in itertools.combinations(frames,2)]
        if min(angles) < 18.-1e-8: raise ValueError('Pose separation failed')
        sets, totals, counts = compatible_sets(frames, clouds, seed=seed)
        write(stage/'pose_sets.json', dict(schema='cadgrasp_precomputed_pose_sets_v1', object=name, passed=True,
            pose_manifest_sha256=digest(stage/'poses.json'), pose_count=30, set_count=20, sets=sets,
            compatible_group_counts=totals, directed_violating_counts=counts.tolist(),
            minimum_pairwise_gravity_direction_deg=min(angles), tolerance_m=1e-9,
            sample_count_per_pose=32768, complete_fixture_verified=False,
            artifacts={str(p.relative_to(stage)):digest(p) for p in sorted((stage/'poses').rglob('*')) if p.is_file()},
            scope='Fixed object-relative registration; sampled floor compatibility only',
            seconds=round(time.monotonic()-began,3)))
        from codes.precompute_objects.draw_sets import render
        render(name, folder=stage)
        # Stage complete and independently checked before replacing any live data.
        backup = stage/'previous'; backup.mkdir()
        installed = []
        try:
            entries = ('poses','poses.json','pose_sets.json','sets.png')
            if mesh_source is not None:
                entries = ('mesh.stl','meta.json') + entries
            for entry in entries:
                if (folder/entry).exists(): (folder/entry).rename(backup/entry)
                (stage/entry).rename(folder/entry); installed.append(entry)
        except BaseException:
            for entry in installed:
                path=folder/entry
                if path.is_dir(): shutil.rmtree(path)
                else: path.unlink()
            for path in backup.iterdir(): path.rename(folder/path.name)
            raise
        removed=[]
        if clean:
            keep={'mesh.stl','meta.json','poses','poses.json','pose_sets.json','sets.png'}
            for path in folder.iterdir():
                if path.name in keep: continue
                removed.append(path.name)
                if path.is_dir(): shutil.rmtree(path)
                else: path.unlink()
        print('PRECOMPUTE COMPLETE', name, '30 poses / 20 sets', 'removed='+','.join(removed), flush=True)
        return dict(object=name, passed=True, pose_count=30, set_count=20, removed=removed, seconds=round(time.monotonic()-began,3))


def publish_plan(name, plan_folder):
    return precompute(name, plan_source=Path(plan_folder))


def refresh_registry():
    """Keep object-level catalogs consistent after a full or partial publication."""
    path = ROOT/'objects/cases.json'
    cases = json.loads(path.read_text())
    manifests = {name: json.loads((ROOT/'objects'/name/'poses.json').read_text()) for name in cases['active_objects']}
    count = sum(len(record['poses']) for record in manifests.values())
    set_count = sum(20 for name in manifests if (ROOT/'objects'/name/'pose_sets.json').is_file())
    cases.update(task_count=count, pose_set_count=set_count,
        selection_basis='Precomputed grounded target poses: 30 per completed object; 20 fixed sampled-floor-compatible sets per object, four each of sizes 2 through 6. No robot trajectory.')
    write(path, cases)
    index_path = ROOT/'objects/index.json'
    if index_path.exists():
        index = json.loads(index_path.read_text())
        index.update(target_count=count, pose_set_count=set_count,
            selection='Seeded pose/work-patch pool and full original-load floor compatibility; robot motion and fixtures are separate.')
        for row in index['objects']:
            name=row['name']
            if not (ROOT/'objects'/name/'pose_sets.json').is_file(): continue
            for key in ('sequence','trajectory','tasks','video','trajectory_segments','gripper_model'): row.pop(key,None)
            meta=json.loads((ROOT/'objects'/name/'meta.json').read_text())
            for key in ('extents_m','volume_m3','mass_kg','n_faces','watertight','hull_volume_ratio'):
                if key in meta: row[key]=meta[key]
            row.update(target_count=30, pose_set_count=20, poses=f'{name}/poses.json',
                pose_sets=f'{name}/pose_sets.json', pose_inputs=f'{name}/poses/', placement_trajectory_verified=False)
        write(index_path,index)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('objects', nargs='*'); p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--seed',type=int,default=20261003); p.add_argument('--budget',type=int,default=12000)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--skip-heads',action='store_true',help='Build only pose/load inputs; defer Step2')
    args=p.parse_args(); names=args.objects or json.loads((ROOT/'objects/cases.json').read_text())['active_objects']
    records=[]
    previous_path=ROOT/'codes/precompute_objects/batch_report.json'
    previous={r['object']:r for r in json.loads(previous_path.read_text()).get('cases',[])} if previous_path.exists() else {}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures={}
        for name in names:
            done=ROOT/'objects'/name/'pose_sets.json'
            if args.resume and done.exists() and json.loads(done.read_text()).get('set_count')==20:
                from codes.precompute_objects.dataset import read_sets, verify_files
                read_sets(name)
                for i in range(1,31): verify_files(name,f'pose_{i}')
                if not (ROOT/'objects'/name/'sets.png').is_file():
                    from codes.precompute_objects.draw_sets import render
                    render(name)
                records.append(dict(previous.get(name,dict(object=name,passed=True,pose_count=30,set_count=20)),reused=True))
                print('REUSE',name,flush=True); continue
            futures[pool.submit(precompute,name,args.seed,args.budget)]=name
        for future in as_completed(futures):
            name=futures[future]
            try: records.append(future.result())
            except Exception as error:
                records.append(dict(object=name,passed=False,error=repr(error)))
                print('PRECOMPUTE FAILED', name, repr(error),flush=True)
    refresh_registry()
    write(ROOT/'codes/precompute_objects/batch_report.json',dict(complete=True,passed=all(r['passed'] for r in records),cases=records))
    if any(not r['passed'] for r in records): raise SystemExit(2)
    if not args.skip_heads:
        from codes.precompute_objects.precompute_heads import build as build_heads
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            jobs=[(n,f'pose_{i}') for n in names for i in range(1,31)]
            futures=[pool.submit(build_heads,job) for job in jobs]
            for future in as_completed(futures):print('STEP2',json.dumps(future.result()),flush=True)

if __name__=='__main__': main()
