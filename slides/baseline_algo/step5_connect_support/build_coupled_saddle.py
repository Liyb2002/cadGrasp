"""A concrete two-sided saddle specimen; not a general Step5 optimizer.

The two landing faces and six retained contact patches bound one carved body.
Task samples/poses are unchanged. The orange group has two distinct patches.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from shapely.geometry import MultiPoint

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step1.needs import ROOT
from step2_local_support import geometry as G, circles as P, withdrawal as W
from step3_scheculer import contacts as I
from step3_scheculer.pair_tasks import read_task, pair_folder
from step4_floor_contact import equilibrium as Q
from step4_floor_contact.whole_assembly import pressure_centers
from step5_base.bearing import bearing_rays, grounded_matrix
from step5_connect_support import ground as FLOOR, solids as SOL
from step5_connect_support.build_shared_geometry import cells_for, local_to_world, export_viewer
from step5_connect_support.surface_check import surface_distances
from step5_connect_support.translation_sweep import swept_solid

POSES = ['pose_1', 'pose_3']
PARTICLE = 9
ROTATION = np.array([
    [-.350218249588062, -.024589534164246057, -.9363453062119012],
    [-.35467650541405515, -.9217364769227027, .15686441157062206],
    [-.8669208465383373, .38703646071101516, .3140876054822018]])
TRANSLATION = np.array([76.06352957354306, 14.694317503725609, 57.567454295631926])/1000
DIRECTION_IDS = [60, 68]
FLOOR_BUFFERS = [.036, .020]
ROUNDING = .007
RELIEF = .0004
SWEEP_LENGTH = .5
SCALE = .2


def inputs():
    source = pair_folder('B', POSES, 'step3_scheculer')/'sequential_3plus2/from_pose_1/terminal_expansion/particle_009'
    schedule = json.loads((source/'schedule.json').read_text())
    if not schedule['passed']:
        raise ValueError('The source contact schedule must pass')
    tasks = [read_task('B', pose, POSES) for pose in POSES]
    contacts = [I.read_contacts(source/f'contacts_{pose}.npz') for pose in POSES]
    paths = [source/'schedule.json']+[source/f'contacts_{pose}.npz' for pose in POSES]
    directions = []
    for pose, ident in zip(POSES, DIRECTION_IDS):
        path = pair_folder('B', POSES, 'step2_local_support')/'sequential_3plus2/from_pose_1/terminal_expansion'/f'candidates_{pose}.json'
        directions.append(np.asarray(json.loads(path.read_text())['direction_catalogue']['vectors'][ident]))
        paths.append(path)
    frames = [np.asarray(p.domain.data['frame']['T_world_mesh']) for p in tasks]
    owned, heads = {}, []
    for k, (p, group) in enumerate(zip(tasks, contacts)):
        offsets = G.vertex_offsets(p.domain.mesh, P.DEPTH_FRACTION*p.domain.mesh.extents.max())[0]
        cells = []
        for c in group:
            ident = c['candidate_id']
            if ident not in owned:
                owned[ident] = k, cells_for(c, p.domain, offsets)
            owner, pieces = owned[ident]
            transform = frames[k]@np.linalg.inv(frames[owner])
            cells.append([v@transform[:3, :3].T+transform[:3, 3] for v in pieces])
        heads.append(cells)
    paths += [path for p in tasks for path in p.inputs]
    return tasks, contacts, heads, directions, source, schedule, paths


def solid(mesh):
    return FLOOR.solid64(mesh, np.zeros(3), SCALE)


def unpack(value):
    data = value.to_mesh64()
    return trimesh.Trimesh(np.asarray(data.vert_properties[:, :3])*SCALE,
                           np.asarray(data.tri_verts), process=False)


def construct(tasks, heads, directions, bases, offsets):
    cells = [G.hull_mesh(v@b+o) for hs, b, o in zip(heads, bases, offsets) for group in hs for v in group]
    points = [np.concatenate([m.vertices for m in cells])]
    for k, p in enumerate(tasks):
        cop = pressure_centers(p.targets/p.scale, p.domain.com)[0]
        outline = MultiPoint(cop).convex_hull.buffer(FLOOR_BUFFERS[k], quad_segs=8)
        xy = np.asarray(outline.exterior.coords)[:-1]
        points.append(np.c_[xy, np.zeros(len(xy))]@bases[k]+offsets[k])
    vertices = np.concatenate(points)
    vertices = vertices[ConvexHull(vertices).vertices]
    bead = trimesh.creation.icosphere(subdivisions=2, radius=ROUNDING).vertices
    blank = solid(G.hull_mesh((vertices[:, None, :]+bead).reshape(-1, 3)))
    for basis, offset in zip(bases, offsets):
        normal = basis[2, :]
        blank = blank.trim_by_plane(normal.tolist(), float(normal@offset/SCALE))
    sweeps = []
    for k, p in enumerate(tasks):
        obj = trimesh.Trimesh(p.domain.mesh.vertices@bases[k]+offsets[k], p.domain.mesh.faces, process=False)
        sweep = swept_solid(obj, -SWEEP_LENGTH*directions[k]@bases[k])
        sweeps.append(sweep)
        # Relief belongs only to the added body; the original head cells below
        # restore the exact contact faces, without an enlarged collision model.
        box = FLOOR.manifold.Manifold.cube([2*RELIEF/SCALE]*3).translate([-RELIEF/SCALE]*3)
        blank = blank-solid(sweep).minkowski_sum(box)
        print('Carved continuous object sweep:', p.pose, flush=True)
    joined = FLOOR.manifold.Manifold.batch_boolean([blank]+[solid(c) for c in cells], FLOOR.manifold.OpType.Add)
    mesh, record = SOL.union_parts([unpack(joined)], SCALE)
    if not record['one_solid']:
        raise RuntimeError(f'The carved body is disconnected: {record}')
    return mesh, sweeps, record


def verify(tasks, groups, heads, directions, bases, offsets, mesh, sweeps):
    checks, certificate = [], {}
    body = solid(mesh)
    tolerance = 1e-11*SCALE**3
    for k, (p, contacts, basis, offset) in enumerate(zip(tasks, groups, bases, offsets)):
        world = local_to_world(mesh.vertices, basis, offset)
        xy = np.unique(world[np.abs(world[:, 2]) < 1e-9, :2], axis=0)
        footprint = xy[ConvexHull(xy).vertices]
        points, normals, owners = bearing_rays(p.domain, contacts, p.floor, 64.)
        matrix, ground = grounded_matrix(points, normals, owners, p.domain.com, p.scale,
                                         [dict(pads_xy_m=[footprint])], 64.)
        targets = Q.padded_targets(p.targets/p.scale, p.scale, 12)
        solver = Q.BatchSolver(matrix)
        result = solver.solve(targets)
        residual = 0.
        for j, columns in enumerate(solver.bases):
            rows = np.flatnonzero(result['assignment'] == j)
            if len(rows):
                residual = max(residual, float(np.max(np.abs(result['weights'][rows]@matrix[:, columns].T-targets[rows]))))
        fitted = trimesh.Trimesh(world, mesh.faces, process=False)
        contact_points = np.unique(np.concatenate([c['triangles_m'].reshape(-1, 3) for c in contacts]), axis=0)
        gaps = surface_distances(fitted, contact_points)
        missing = [abs(float((solid(G.hull_mesh(v@basis+offset))-body).volume()))*SCALE**3
                   for group in heads[k] for v in group]
        overlap = abs(float((body^solid(sweeps[k])).volume()))*SCALE**3
        separation = float((p.domain.mesh.vertices@(-directions[k])).min()+SWEEP_LENGTH-(world@(-directions[k])).max())
        withdrawal = dict(clear=bool(overlap <= tolerance and separation > 0),
            complete_sweep_overlap_m3=overlap, volume_tolerance_m3=tolerance,
            length_m=SWEEP_LENGTH, terminal_projection_separation_m=separation,
            direction=directions[k].tolist(), method='Initial object plus every forward-facing boundary triangle prism; continuous Mesh64 Boolean')
        # A separate implementation checks original convex head cells. The full
        # nonconvex body is checked against its original, unpadded object sweep.
        local_cells = [v@b+o for hs, b, o in zip(heads, bases, offsets) for group in hs for v in group]
        analyzer = W.Analyzer(p.domain.mesh, P.DEPTH_FRACTION*p.domain.mesh.extents.max(), dict(vectors=[directions[k].tolist()]))
        head_check = analyzer.test([SimpleNamespace(vertices=local_to_world(v, basis, offset)) for v in local_cells], directions[k])
        check = dict(pose=p.pose, original_sample_count=len(targets),
            verified_sample_count=int((result['assignment'] >= 0).sum()),
            coupled_equilibrium_passed=bool(result['passed']), actual_ground_hull_xy_m=footprint.tolist(),
            min_fixture_z_m=float(world[:, 2].min()), max_active_contact_gap_m=float(gaps.max()),
            maximum_missing_head_cell_volume_m3=max(missing), maximum_equilibrium_residual=residual,
            minimum_reaction_coefficient=float(result['weights'].min()), withdrawal=withdrawal,
            independent_head_withdrawal=head_check, lp_count=result['lp_count'], friction_coefficient=64.)
        checks.append(check)
        for key, value in dict(matrix=matrix, assignment=result['assignment'], weights=result['weights'],
                               bases=np.asarray(solver.bases), ground_points_m=ground['points_m']).items():
            certificate[f'{p.pose}_{key}'] = value
        print('Final solid:', {key: value for key, value in check.items() if key != 'actual_ground_hull_xy_m'}, flush=True)
    passed = all(c['coupled_equilibrium_passed'] and c['withdrawal']['clear'] and c['independent_head_withdrawal']['clear']
        and c['min_fixture_z_m'] >= -1e-9 and c['max_active_contact_gap_m'] < 1e-8
        and c['maximum_missing_head_cell_volume_m3'] <= tolerance for c in checks)
    if not passed:
        raise RuntimeError('The actual body did not pass; existing outputs were not replaced')
    return checks, certificate


def local_reuse_probes(mesh, groups, bases, offsets):
    """Evidence on exact rays through patch-triangle centroids, not an area proof."""
    records = []
    for k, contacts in enumerate(groups):
        normal = bases[1-k][2, :]
        plane = normal@offsets[1-k]
        for c in contacts:
            triangles = c['triangles_m']
            centers = triangles.mean(1)@bases[k]+offsets[k]
            depth = centers@normal-plane
            destination = centers-depth[:, None]*normal
            start = centers-2e-7*normal
            locations, indices, _ = mesh.ray.intersects_location(start, np.tile(-normal, (len(start), 1)), multiple_hits=True)
            travel = np.einsum('ij,j->i', locations-start[indices], -normal)
            interrupted = np.zeros(len(start), bool)
            interrupted[indices[(travel > 2e-7) & (travel < depth[indices]-4e-7)]] = True
            inside = mesh.contains((centers+destination)/2)
            gap = surface_distances(mesh, destination)
            good = (~interrupted) & inside & (gap < 1e-8) & (depth > 0)
            areas = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1)/2
            records.append(dict(pose=POSES[k], id=c['candidate_id'],
                opposite_floor_depth_quantiles_mm=(np.quantile(depth, [0, .5, 1])*1000).tolist(),
                area_weighted_centroid_probe_fraction=float(areas[good].sum()/areas.sum()),
                probe_count=len(good), passed_probe_count=int(good.sum())))
    return records


def run(output=None):
    tasks, groups, heads, directions, source, schedule, paths = inputs()
    bases = [np.eye(3), ROTATION]
    offsets = [np.zeros(3), -TRANSLATION@ROTATION]
    mesh, sweeps, solid_check = construct(tasks, heads, directions, bases, offsets)
    checks, certificate = verify(tasks, groups, heads, directions, bases, offsets, mesh, sweeps)
    reuse = local_reuse_probes(mesh, groups, bases, offsets)
    out = Path(output) if output else pair_folder('B', POSES, 'step3_scheculer').parent/'step5'
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='cadgrasp_saddle_') as temporary:
        stage = Path(temporary)
        mesh.export(stage/'fixture.obj', file_type='obj', digits=17, include_normals=False)
        mm = mesh.copy(); mm.apply_scale(1000)
        (stage/'fixture_mm.stl').write_text(trimesh.exchange.stl.export_stl_ascii(mm))
        reloaded = trimesh.load(stage/'fixture_mm.stl', force='mesh')
        if not reloaded.is_watertight or not reloaded.is_winding_consistent:
            raise RuntimeError('ASCII STL failed the closed-solid reload check')
        np.testing.assert_allclose(reloaded.extents, mm.extents, atol=1e-10, rtol=0)
        np.savez_compressed(stage/'geometry.npz', vertices_m=mesh.vertices, faces=mesh.faces,
                            rotations=bases, local_offsets_m=offsets)
        np.savez_compressed(stage/'equilibrium.npz', **certificate)
        report = dict(object='B', poses=POSES, particle=PARTICLE, complete=True,
            status='geometry_and_fixed_sample_equilibrium_passed', design='two_sided_carved_saddle',
            same_rigid_solid_in_both_poses=True, contact_groups=5, contact_patch_count=6,
            shared_branch_definition='two distinct orange contact patches in one continuous body; not one shared patch',
            original_task_poses_changed=False, original_active_contacts_preserved=True,
            support_mass_ignored=True, extra_loads_added=False, structural_strength_verified=False,
            dimensions_mm=(mesh.extents*1000).tolist(), volume_cm3=float(mesh.volume*1e6),
            solid=solid_check, checks=checks,
            source_schedule=str((source/'schedule.json').relative_to(ROOT)),
            task_fixture_transforms=[dict(pose=p.pose, rotation=b.tolist(), translation_m=(-b@o).tolist()) for p, b, o in zip(tasks, bases, offsets)],
            body_design=dict(method='rounded convex blank carved by two complete withdrawal sweeps; source head cells retained',
                floor_demand_initialization_buffers_m=FLOOR_BUFFERS, exterior_rounding_m=ROUNDING,
                unselected_surface_relief_box_halfwidth_m=RELIEF, dedicated_floor_rails=False,
                layout_selection='manually selected finite-search specimen; no general optimizer or optimality claim'),
            local_reuse=dict(probes=reuse, interpretation='Area-weighted patch triangle centroid rays through continuous material to the other landing plane; sampled geometric evidence, not an area or strength certificate'),
            presentation_description='双面鞍形实体：彩色保留原接触头及附近区域，灰白色是连续身体。换姿态后，身体的另一面直接落地；橙色仍对应两块不同接触面。',
            exports=dict(obj_units='m', stl_units='mm', stl_encoding='ASCII', reloaded_stl_watertight=True),
            artifacts={name: I.sha256(stage/name) for name in ('fixture.obj', 'fixture_mm.stl', 'geometry.npz', 'equilibrium.npz')},
            provenance=dict(inputs=I.hashes(paths), code=I.hashes([Path(__file__), Path(__file__).with_name('translation_sweep.py'),
                Path(__file__).with_name('surface_check.py'), Path(__file__).with_name('build_shared_geometry.py'),
                Path(__file__).with_name('shared_geometry_viewer.html'), Path(Q.__file__), Path(W.__file__), Path(G.__file__), Path(SOL.__file__)])))
        I.save(stage/'report.json', report)
        shared = schedule['shared_head']['selected_id']
        colors = {shared: '#dc9d47'}
        colors.update(zip([i for i in schedule['selected_ids'] if i != shared], ['#ac7098', '#7196c0', '#50a59b', '#77a76a']))
        visual = [(c['candidate_id'], G.hull_mesh(np.concatenate([v@b+o for v in cells])))
                  for group, hs, b, o in zip(groups, heads, bases, offsets) for c, cells in zip(group, hs)]
        export_viewer(stage, mesh, visual, report, tasks, bases, offsets, colors)
        for path in stage.iterdir():
            os.replace(path, out/path.name)
    print('EXPORTED', out, report['dimensions_mm'], report['volume_cm3'], flush=True)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    run(parser.parse_args().output)
