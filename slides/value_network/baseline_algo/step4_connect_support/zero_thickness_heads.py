"""Exact contact surfaces and constructor-owned support roots.

The input head is the saved surface, with zero mandatory thickness. The solid
constructor needs a small outside collar to carry that surface through its
non-contact relief. These collars are new support material, not retained
Step3 probe solids. Every final body check still applies to that material.
"""
import numpy as np
import trimesh

from step2_local_support import geometry as G
from step4_connect_support.fixture_view import cells_for
from step4_connect_support import head_registration as H

MODEL = 'zero_thickness_contact_surface'


def patch_mesh(points):
    """Keep the exact saved triangles, including nonplanar patch boundaries."""
    points = np.asarray(points, float).reshape(-1, 3)
    if len(points) % 3:
        raise ValueError('Contact patch must contain complete saved triangles')
    return trimesh.Trimesh(points, np.arange(len(points)).reshape(-1, 3), process=False)


def prepare(case, relief_m, *, bases=None, offsets=None):
    """Replace required old probe volumes with patches and bounded new roots.

    Root normal depth is at most twice the non-contact relief, and no generated
    root vertex moves farther than half the minimum patch/floor clearance.
    Actual transformed root vertices are independently checked on every floor.
    """
    if bases is None:bases, offsets = H.fixed_placements(case.tasks)
    case.probe_heads = case.heads
    case.heads = [[list(np.asarray(c['triangles_m'], float)) for c in row] for row in case.groups]
    seeds, records = [], []
    for owner, (task, group) in enumerate(zip(case.tasks, case.groups)):
        unit_offsets, valid = G.vertex_offsets(task.domain.mesh, 1.)
        row = []
        for contact in group:
            patch = np.asarray(contact['triangles_m']).reshape(-1, 3)
            local_patch = patch@bases[owner]+offsets[owner]
            clearances = [float(((local_patch-o)@b.T)[:, 2].min()) for b, o in zip(bases, offsets)]
            if min(clearances) <= H.FLOOR_TOL:
                raise ValueError(f"Contact patch {contact['candidate_id']} has no positive all-pose floor clearance")
            vertices = np.unique(task.domain.mesh.faces[np.unique(contact['source_faces'])])
            if not np.all(valid[vertices]):
                raise ValueError(f"Contact patch {contact['candidate_id']} has no valid outward support root")
            offset_ratio = float(np.linalg.norm(unit_offsets[vertices], axis=1).max())
            depth = min(2*relief_m, .5*min(clearances)/offset_ratio)
            cells = cells_for(contact, task.domain, depth*unit_offsets)
            points = np.vstack(cells)@bases[owner]+offsets[owner]
            heights = [float(((points-o)@b.T)[:, 2].min()) for b, o in zip(bases, offsets)]
            if min(heights) < -H.FLOOR_TOL:
                raise ValueError(f"Generated support root {contact['candidate_id']} crosses a floor")
            row.append(cells)
            records.append(dict(candidate_id=contact['candidate_id'], owner_pose=task.pose,
                mandatory_input_head_thickness_m=0., constructor_root_normal_depth_m=float(depth),
                maximum_root_vertex_displacement_bound_m=float(depth*offset_ratio),
                minimum_contact_height_by_pose_m=clearances, minimum_root_height_by_pose_m=heights,
                all_generated_root_vertices_above_all_floors=True,
                original_probe_volume_retained=False))
        seeds.append(row)
    case.support_seeds = seeds
    case.head_model = MODEL
    case.support_seed_records = records
    case.schedule.update(head_model=MODEL, mandatory_head_thickness_m=0.,
        constructor_generates_support_roots=True, original_probe_volumes_required=False)
    return case
