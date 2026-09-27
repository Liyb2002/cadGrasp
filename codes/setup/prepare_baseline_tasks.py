"""Prepare declared target tasks in objects/, without running support search."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'slides/baseline_algo'))
sys.path.insert(0, str(ROOT / 'slides/setup/poses'))
import numpy as np
import trimesh
import target_poses as T
from step1.registry import active_objects, object_folder

# Same orientations for every newly added object; selected before support search.
DEFAULT_GRAVITIES = ([1., .7, .4], [-.4, 1., -.7])


def prepare(name):
    folder = object_folder(name)
    if (folder / 'tasks.json').exists():
        raise FileExistsError(f'{name}: task manifest already exists; existing task inputs are immutable here')
    raw = trimesh.load(folder / 'mesh.stl', force='mesh')
    mesh, rounds = T.G.refine(raw)
    previous, poses = [], []
    for index, gravity in enumerate(DEFAULT_GRAVITIES, 1):
        pose = f'pose_{index}'
        out = folder / 'tasks' / pose
        if out.exists():
            raise FileExistsError(f'{out}: refusing to overwrite an existing task')
        transform, vertex = T.target_transform(raw, gravity)
        region = T.choose_region(mesh, transform, name + '_' + pose, previous)
        mask = region['take']
        checks = T.inspect(mesh, transform, mask, raw)
        previous.append(mask)
        contact = transform[:3, :3] @ raw.vertices[vertex] + transform[:3, 3]
        contact[2] = 0.
        out.mkdir(parents=True)
        np.savez_compressed(out / 'setup.npz', object=name, pose_id=pose,
            T_world_mesh=transform, com_m=checks['com_world_m'], work_faces=mask,
            floor_contact_m=contact, mesh_sha256=T.digest(folder / 'mesh.stl'),
            poses_sha256=T.digest(folder / 'poses.json'), K=T.K,
            cone_half_deg=T.CONE_HALF_DEG, tip=-1)
        report = dict(schema_version=1, object=name, pose_id=pose,
            label='Declared target tilt ' + str(index), target_pose_only=True,
            placement_trajectory_verified=False, support_search_run=False,
            contact_anatomy='lowest mesh vertex', T_world_mesh=transform.tolist(),
            floor_contact_m=contact.tolist(), work_face_ids=np.flatnonzero(mask).tolist(),
            work_seed=int(region['seed']), uniform_subdivision_rounds=rounds, checks=checks,
            K=T.K, cone_half_deg=T.CONE_HALF_DEG, source_snapshot='setup.npz',
            source_snapshot_sha256=T.digest(out / 'setup.npz'),
            generator=str(Path(__file__).relative_to(ROOT)), generator_sha256=T.digest(__file__),
            helper_sha256=T.digest(T.__file__),
            selection_rule='Two common predeclared gravity directions; one connected 6-10% work patch; no support results consulted',
            work_region_method='Connected geodesic patch, outward normal z > 0.35, normal-ray visible, clear of floor',
            reachability_scope='Normal rays only; Step1 checks each sampled cone direction')
        T.save(out / 'setup.json', report)
        T.draw_pose(mesh, transform, mask, report).save(out / 'pose.png')
        poses.append(pose)
        print(f'{name}/{pose}: work area {checks["work_area_fraction"]:.2%}, source {out.relative_to(ROOT)}', flush=True)
    T.save(folder / 'tasks.json', dict(schema='cadgrasp_tasks_v1', object=name, poses=poses,
        definition='Held target poses, not natural stable placements; independently solved by baseline'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('objects', nargs='+', choices=active_objects())
    args = parser.parse_args()
    for name in args.objects:
        prepare(name)
