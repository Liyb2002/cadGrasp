"""Whole contact-surface clearance in every requested object pose.

Only this floor margin is shared between the independent searches. Forces,
working surfaces, local paths and withdrawal still belong to the owner task.
The existing finite-thickness local checks are retained as conservative probes;
the intended head input for the new body construction is the contact surface.
"""
from pathlib import Path
import json

import numpy as np

from step3_scheculer.joint_geometry import JointGeometry
from step3_scheculer.pair_tasks import input_hashes

SCHEMA = 'independent_single_pose_floor_margin_v1'
HEAD_MODEL = 'zero_thickness_contact_surface'
DEFAULT_CLEARANCE_M = .002
TOLERANCE_M = 1e-10


def transforms_from_owner(owner, problems):
    inverse = np.linalg.inv(np.asarray(owner.domain.data['frame']['T_world_mesh'], float))
    transforms = []
    for problem in problems:
        transform = np.asarray(problem.domain.data['frame']['T_world_mesh'], float)@inverse
        np.testing.assert_array_equal(owner.domain.mesh.faces, problem.domain.mesh.faces)
        np.testing.assert_allclose(owner.domain.mesh.vertices@transform[:3, :3].T+transform[:3, 3],
                                   problem.domain.mesh.vertices, atol=1e-12, rtol=0)
        transforms.append(transform)
    return transforms


def contact_check(contact, poses, transforms, clearance_m=DEFAULT_CLEARANCE_M):
    """An affine plane height attains its minimum at a triangle vertex."""
    if not np.isfinite(clearance_m) or clearance_m < 0:
        raise ValueError('Floor clearance must be finite and nonnegative')
    if not poses or len(poses) != len(transforms):
        raise ValueError('Every supplied pose requires its rigid transform')
    vertices = np.asarray(contact['triangles_m'], float).reshape(-1, 3)
    if not len(vertices) or not np.isfinite(vertices).all():
        return dict(passed=False, reason='empty_or_invalid_contact_surface', per_pose=[])
    rows = []
    for pose, transform in zip(poses, transforms):
        transform = np.asarray(transform, float)
        heights = vertices@transform[2, :3]+transform[2, 3]
        index = int(np.argmin(heights))
        height = float(heights[index])
        rows.append(dict(pose=pose, minimum_contact_height_m=height,
            minimum_vertex_index=index, passed=bool(height >= clearance_m-TOLERANCE_M)))
    passed = all(row['passed'] for row in rows)
    return dict(passed=passed, reason='passed' if passed else 'all_pose_contact_floor_margin',
        clearance_m=clearance_m, tolerance_m=TOLERANCE_M, per_pose=rows,
        surface='all vertices of all finite contact triangles; exact affine-height minimum',
        finite_head_thickness_included=False)


class FloorMarginGeometry(JointGeometry):
    def __init__(self, problem, floor_problems, count=200, clearance_m=DEFAULT_CLEARANCE_M):
        if not floor_problems or problem.pose not in [p.pose for p in floor_problems]:
            raise ValueError('Floor context must include the owner pose')
        if not np.isfinite(clearance_m) or clearance_m < 0:
            raise ValueError('Floor clearance must be finite and nonnegative')
        self.floor_problems = list(floor_problems)
        self.floor_poses = [p.pose for p in floor_problems]
        self.floor_transforms = transforms_from_owner(problem, floor_problems)
        self.floor_clearance_m = float(clearance_m)
        super().__init__([problem], count=count, global_head_exclusions=False)
        for entry in self.pools[0]:
            check = self.check_contact(entry['contact'])
            entry['all_pose_floor_margin'] = check
            entry['native']['all_pose_floor_margin'] = check
            if not check['passed']:
                entry['valid'] = entry['native']['valid'] = False
                entry['reason'] = entry['native']['reason'] = check['reason']

    def check_contact(self, contact):
        return contact_check(contact, self.floor_poses, self.floor_transforms, self.floor_clearance_m)

    def floor_check(self, entries):
        rows = [dict(candidate_id=e['contact']['candidate_id'], **self.check_contact(e['contact']))
                for e in entries]
        return dict(passed=all(row['passed'] for row in rows), heads=rows,
            poses=self.floor_poses, clearance_m=self.floor_clearance_m,
            tolerance_m=TOLERANCE_M, head_model=HEAD_MODEL,
            scope='Entire finite contact surfaces in every supplied pose, including inactive heads')

    def group_check(self, entries):
        result = super().group_check(entries)
        floor = self.floor_check(entries)
        result.update(passed=result['passed'] and floor['passed'], all_pose_contact_floor_margin=floor,
            inactive_contact_surfaces_checked=True, inactive_head_solids_checked=False,
            local_geometry_probe='original finite-thickness owner-pose head; conservative for the surface model')
        return result

    def save(self, folder, save):
        super().save(folder, save)
        path = Path(folder)/f'candidates_{self.problems[0].pose}.json'
        result = json.loads(path.read_text())
        result.update(head_model=HEAD_MODEL,
            local_geometry_probe_depth_m=float(self.local[0].depth),
            all_pose_contact_floor_margin=dict(poses=self.floor_poses, clearance_m=self.floor_clearance_m,
                tolerance_m=TOLERANCE_M, inputs=input_hashes(self.floor_problems),
                candidates=[dict(candidate_id=e['contact']['candidate_id'], **e['all_pose_floor_margin'])
                            for e in self.pools[0]]))
        save(path, result)
