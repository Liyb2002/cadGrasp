"""Common solid operations and unchanged original-load/withdrawal acceptance.

Extracted from the archived specimen; no fixed pose, contact assignment, feet or
placement constants belong to this module. All lengths are metres.
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
from step0_pose_selection import equilibrium as Q
from step0_pose_selection.floor_points import pressure_centers
from step5_base.bearing import bearing_rays, grounded_matrix
from step4_connect_support.baseline_current import ground as FLOOR, solids as SOL
from step4_connect_support.baseline_current.fixture_view import cells_for, local_to_world, export_viewer
from step4_connect_support.baseline_current.surface_check import surface_distances
from step4_connect_support.baseline_current.translation_sweep import swept_solid
from step4_connect_support.baseline_current import head_registration as H


RELIEF = .0004
SWEEP_LENGTH = .5
SCALE = .2

def solid(mesh):
    return FLOOR.solid64(mesh, np.zeros(3), SCALE)


def unpack(value):
    data = value.to_mesh64()
    return trimesh.Trimesh(np.asarray(data.vert_properties[:, :3])*SCALE,
                           np.asarray(data.tri_verts), process=False)


def verify(tasks, groups, heads, directions, bases, offsets, mesh, sweeps):
    registered, registration = H.register(groups, heads, bases, offsets)
    if not registration['all_heads_above_both_floors']:
        raise ValueError('Original five-head layout penetrates a floor')
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
        result['weights'] = np.maximum(result['weights'], 0.)
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
        local_cells = [v for head in registered for v in head.cells]
        analyzer = W.Analyzer(p.domain.mesh, P.DEPTH_FRACTION*p.domain.mesh.extents.max(), dict(vectors=[directions[k].tolist()]))
        head_check = analyzer.test([SimpleNamespace(vertices=local_to_world(v, basis, offset)) for v in local_cells], directions[k])
        check = dict(pose=p.pose, original_sample_count=len(targets),
            physical_head_count=len(registered), shared_head_registration_passed=True,
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
        error = RuntimeError('Final geometry or original-sample equilibrium failed')
        error.checks = checks
        raise error
    return checks, certificate
