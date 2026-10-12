"""Shrink saved work patches to 6-10% without changing poses or trajectories.

Retains each patch's seed and grows only inside its previous face mask. Writes
only task setup.npz/setup.json and, by default, the object's overview.png.
"""
import sys
sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path
import tempfile
import zlib

import numpy as np
import trimesh

from sequence_export import ROOT, B, WORK_AREA_FRACTION, digest, write

REVISION = 'same_seed_subset_6_10_percent_v1'


def resize(name, overview=True):
    folder = ROOT / 'objects' / name
    record = json.loads((folder / 'poses.json').read_text())
    raw = trimesh.load(folder / 'mesh.stl', force='mesh')
    meshes = {0: raw}
    graphs = {}
    fractions = []
    with tempfile.TemporaryDirectory(prefix='cadgrasp-work-area-') as temp:
        stage = Path(temp)
        for pose in record['poses']:
            task = folder / 'tasks' / pose['pose_id']
            report = json.loads((task / 'setup.json').read_text())
            if report.get('work_region_update', {}).get('revision') == REVISION:
                fractions.append(report['checks']['work_area_fraction'])
                continue
            rounds = report['uniform_subdivision_rounds']
            for level in range(1, rounds + 1):
                if level not in meshes:
                    meshes[level] = meshes[level - 1].subdivide()
            mesh = meshes[rounds]
            with np.load(task / 'setup.npz') as z:
                fields = dict(z)
            T = np.asarray(pose['T_world_mesh'])
            np.testing.assert_array_equal(fields['T_world_mesh'], T)
            old_mask = fields['work_faces']
            old_fraction = float(mesh.area_faces[old_mask].sum() / mesh.area)
            seed = int(report['checks']['work_seed'])
            original_seed = seed
            original_rounds = rounds
            seed_position = mesh.triangles_center[seed].copy()
            assert old_mask[seed]
            minimum, maximum = WORK_AREA_FRACTION
            assert old_fraction >= minimum
            # Some A2 source triangles alone occupy 14% of its surface.
            # Split uniformly without changing geometry; the central child
            # retains the exact original seed centroid and mask membership.
            while mesh.area_faces[old_mask].max() / mesh.area > minimum:
                rounds += 1
                if rounds not in meshes:
                    meshes[rounds] = mesh.subdivide()
                mesh = meshes[rounds]
                old_mask = np.repeat(old_mask, 4)
                seed = 4 * seed + 3
                np.testing.assert_allclose(mesh.triangles_center[seed], seed_position, atol=1e-12)
            np.testing.assert_allclose(mesh.area_faces[old_mask].sum() / mesh.area,
                                       old_fraction, atol=1e-12)
            if rounds not in graphs:
                adjacency = [[] for _ in mesh.faces]
                weights = [[] for _ in mesh.faces]
                pairs = mesh.face_adjacency
                distances = np.linalg.norm(
                    mesh.triangles_center[pairs[:, 0]] - mesh.triangles_center[pairs[:, 1]], axis=1)
                for (i, j), distance in zip(pairs, distances):
                    adjacency[i].append(j); weights[i].append(distance)
                    adjacency[j].append(i); weights[j].append(distance)
                graphs[rounds] = adjacency, weights
            key = f'{name}/{pose["pose_id"]}'
            rng = np.random.default_rng(20260919 ^ zlib.crc32(key.encode()))
            target_fraction = float(rng.uniform(minimum, min(maximum, old_fraction)))
            mask, _ = B.grow(mesh, seed, target_fraction * mesh.area,
                             maximum * mesh.area, ~old_mask, *graphs[rounds])
            fraction = float(mesh.area_faces[mask].sum() / mesh.area)
            assert minimum <= fraction <= maximum, (key, fraction)
            assert np.all(~mask | old_mask) and mask[seed]
            assert B.components(mesh, mask) == 1, key
            world = trimesh.transform_points(mesh.vertices, T)
            min_z = float(world[mesh.faces[mask], 2].min())
            min_normal_z = float((mesh.face_normals[mask] @ T[2, :3]).min())
            assert min_z > .0015 and min_normal_z > .35
            # The subset inherits the previous normal-ray clearance. Recheck
            # selected faces against the geometry before publishing the update.
            assert not mesh.ray.intersects_any(
                mesh.triangles_center[mask] + 1e-5 * mesh.face_normals[mask],
                mesh.face_normals[mask]).any(), key
            fields['work_faces'] = mask
            destination = stage / pose['pose_id']
            destination.mkdir()
            np.savez_compressed(destination / 'setup.npz', **fields)
            report['work_face_ids'] = np.flatnonzero(mask).tolist()
            report['uniform_subdivision_rounds'] = rounds
            report['checks'].update(work_area_fraction=fraction, work_components=1, work_seed=seed,
                work_normal_rays_clear=True, minimum_work_vertex_z_m=min_z,
                minimum_work_outward_normal_z=min_normal_z)
            report['source_snapshot_sha256'] = digest(destination / 'setup.npz')
            report['work_region_method'] = (
                'Same original seed; connected geodesic growth restricted to the previous '
                'work patch, with a seeded random 6-10% total-surface-area target '
                'capped by the previous patch area. Pose and trajectory unchanged.')
            report['work_region_update'] = dict(
                revision=REVISION, generator='codes/setup/resize_work_areas.py',
                generator_sha256=digest(__file__),
                previous_work_area_fraction=old_fraction,
                target_work_area_fraction=target_fraction,
                allowed_work_area_fraction=list(WORK_AREA_FRACTION),
                retained_original_seed=True, seed_location_mesh_m=seed_position.tolist(),
                previous_work_seed=original_seed, previous_uniform_subdivision_rounds=original_rounds,
                subset_of_previous_work_faces=True)
            write(destination / 'setup.json', report)
            fractions.append(fraction)
        # All ten patches pass before any task for this object is replaced.
        for staged in stage.iterdir():
            for path in staged.iterdir():
                (folder / 'tasks' / staged.name / path.name).write_bytes(path.read_bytes())
    from verify_sequences import verify
    verify(folder)
    if overview:
        from overview import render
        render(name)
    print(json.dumps(dict(object=name,poses=len(fractions),
        minimum_percent=100 * min(fractions), maximum_percent=100 * max(fractions))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('objects', nargs='*')
    parser.add_argument('--no-overview', action='store_true')
    args = parser.parse_args()
    names = args.objects or json.loads((ROOT / 'objects/cases.json').read_text())['active_objects']
    for name in names:
        resize(name, overview=not args.no_overview)
