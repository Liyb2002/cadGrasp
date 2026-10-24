"""Fit one immutable two-window shell to two uses of a stationary needle."""
import numpy as np
from scipy.spatial.transform import Rotation

from whole_task.step1.needle_geometry import geometric_mask
from whole_task.step3 import independent_geometry as G
from whole_task.step3.exit_geometry import continuous_checks, whole_object_sweep
from whole_task.step3.independent_physics import demands, certificate
from whole_task.step3.render_independent import head_context
from whole_task.step3.run_independent import make_domain
from .geometry import moved_mesh
from .pose_change import pose_change
from .state import transform


def axial_pose_angles(model, layout, turn_degrees=90.):
    """Use the free axial task rotation to select the complete fixture turn."""
    axis = model.axis
    cross = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
    relative = layout.poses[1, :3, :3] @ layout.poses[0, :3, :3].T
    parallel = np.outer(axis, axis)
    cosine = np.trace((np.eye(3) - parallel) @ relative)
    sine = np.trace(cross @ relative)
    constant = np.trace(parallel @ relative)
    phase = np.arctan2(sine, cosine)
    value = (1. + 2. * np.cos(np.deg2rad(turn_degrees)) - constant) / np.hypot(cosine, sine)
    if abs(value) <= 1.:
        delta = np.arccos(value)
        return np.rad2deg([phase + delta, phase - delta])
    return np.rad2deg([phase])


def build_fixed_needle_merge(model):
    reference = model.initial.copy()
    tool = G.union([G.solid(mesh) for mesh, _, _ in head_context(model.workspace)])
    body_solid, outer = G.solid(model.source), G.solid(model.outer)
    n = np.arange(96)
    z = 1. - 2. * (n + .5) / len(n)
    azimuth = n * np.pi * (3. - np.sqrt(5.))
    sphere = np.c_[np.sqrt(1. - z*z) * np.cos(azimuth), np.sqrt(1. - z*z) * np.sin(azimuth), z]
    for angle in axial_pose_angles(model, reference):
        layout = pose_change(reference, 1, float(angle), model.axis, model.head)
        layout.material = np.array([np.linalg.inv(pose) for pose in layout.poses])
        rows = []
        for k, pose in enumerate(layout.poses):
            record = model.records[k]
            head = transform(model.head, layout.material[k])
            axis = pose[:3, :3].T @ model.axis
            assert geometric_mask(model.source, head, axis, model.workspace)[record['working_face_ids']].all()
            assert not model.source.contains(head[None])[0]
            workspace = dict(model.workspace)
            workspace['reference_head_front_world_m'] = head.tolist()
            workspace['nominal_axis_world'] = axis.tolist()
            work, _ = G.work_sector(workspace, model.policy)
            domain = make_domain(model.source, pose, record['working_face_ids'], model.ids[k],
                model.workspace, model.physical, model.policy)
            endpoints, samples = demands(domain, model.policy)
            body = moved_mesh(model.source, pose)
            rows.append(dict(body=body, domain=domain, endpoints=endpoints, samples=samples, work=work))
        up = [pose[:3, :3].T @ np.array([0., 0., 1.]) for pose in layout.poses]
        directions = [up[0] + up[1], *sphere]
        directions = [direction / np.linalg.norm(direction) for direction in directions
            if min(direction @ up[0], direction @ up[1]) > .05 * np.linalg.norm(direction)]
        directions.sort(key=lambda direction: -min(direction @ up[0], direction @ up[1]))
        for index, direction in enumerate(directions):
            world_directions = [pose[:3, :3] @ direction for pose in layout.poses]
            blocked = False
            for k, row in enumerate(rows):
                body = G.solid(row['body'])
                if any(G.volume(tool ^ body.translate((distance * world_directions[k] / G.SCALE).tolist()))
                        > model.tolerance for distance in np.linspace(0., .35, 15)):
                    blocked = True
                    break
            if blocked:
                continue
            sweep, _ = whole_object_sweep(model.source, direction, .7)
            raw = outer - G.union([body_solid, rows[0]['work'], rows[1]['work'], sweep])
            if raw.is_empty():
                continue
            final = G.mesh(max(raw.decompose(), key=lambda part: float(part.volume())) ^ raw)
            forces = []
            for k, row in enumerate(rows):
                force, _ = certificate(row['body'], moved_mesh(final, layout.poses[k]),
                    row['domain'], row['endpoints'], row['samples'], search=True)
                forces.append(force)
                if not force['passed']:
                    break
            if len(forces) != 2 or not all(force['passed'] for force in forces):
                continue
            tool_checks = []
            for k, row in enumerate(rows):
                checks, _ = continuous_checks(tool,
                    [(model.ids[k], row['body'], world_directions[k], .35)], model.tolerance)
                tool_checks.extend(checks)
            exits, _ = continuous_checks(G.solid(final),
                [(name, model.source, direction, .7) for name in model.ids], model.tolerance)
            if not all(row['passed'] for row in tool_checks + exits):
                continue
            layout.exits = np.array(world_directions)
            geometry = dict(outside_fitted_outer_m3=G.volume(G.solid(final) - outer),
                body_overlap_m3=G.volume(G.solid(final) ^ body_solid),
                work_overlap_m3=[G.volume(G.solid(final) ^ row['work']) for row in rows])
            assert max([geometry['outside_fitted_outer_m3'], geometry['body_overlap_m3'],
                *geometry['work_overlap_m3']]) <= model.tolerance
            turn = np.rad2deg(Rotation.from_matrix(layout.poses[1, :3, :3] @ layout.poses[0, :3, :3].T).magnitude())
            print('FIXED NEEDLE MERGE PASS: task pose', round(float(angle), 6),
                'degrees; fixture turn', round(float(turn), 6), 'degrees', flush=True)
            return layout, final, dict(contact_loads=forces, continuous_exits=exits,
                stationary_tool_extraction_checks=tool_checks, geometry=geometry,
                reference_task_poses_world=reference.poses.tolist(),
                selected_task_poses_world=layout.poses.tolist(),
                task_pose_adjustments_deg=[0., float(angle)], fixture_turn_deg=float(turn),
                working_region_inside_fixed_sector=[True, True],
                fixed_needle_position_world_m=model.head.tolist(), fixed_needle_axis_world=model.axis.tolist(),
                local_extraction_direction=direction.tolist(), accepted_direction_index=index)
    raise RuntimeError('No certified merge pose with a stationary needle in the finite candidate menu')
