"""Step5.4: +Z seating, updated demands, and floor-safe empty transfers.

Only world +Z translations change poses. Relative placements, rotations,
native work areas, loads and exit directions remain fixed. Exact floor
contact is allowed; penetration is forbidden. Existing files are read-only.
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

import stage5_common as C
import base_seating as B
import base_geometry as G
from base_reactions import ground_coverage


def move(data, heights):
    heights = np.asarray(heights)
    if np.any(heights < -1e-12):
        raise ValueError('Step5.4 permits +Z only')
    delta = np.zeros((len(heights), 3)); delta[:, 2] = heights
    return B.shifted(data, dict(fixture_lift_m=heights.tolist(),
        object_relative_world_translation_m=np.zeros_like(delta).tolist(),
        total_object_world_translation_m=delta.tolist()))


def minimum_lifts(mesh, data):
    values = []
    for row in data['rows']:
        fixture_z = C.F.transform_points(mesh.vertices, row['fixture_transform'])[:, 2].min()
        object_z = C.F.transform_points(data['body'].vertices, row['object_transform'])[:, 2].min()
        values.append(max(0., -float(fixture_z), -float(object_z)))
    return np.asarray(values)


def check(mesh, moved):
    heights = []
    for row in moved['rows']:
        material = C.F.transform_points(mesh.vertices, row['fixture_transform'])
        body = C.F.transform_points(moved['body'].vertices, row['object_transform'])
        heights.append(dict(pose=row['pose'], fixture_minimum_z_m=float(material[:, 2].min()),
                            object_minimum_z_m=float(body[:, 2].min())))
    coverage, points = ground_coverage(mesh, moved)
    floor = all(min(r['fixture_minimum_z_m'], r['object_minimum_z_m']) >= -1e-9 for r in heights)
    return dict(passed=floor and coverage['passed'], floor_passed=floor,
                minimum_heights=heights, updated_ground_coverage=coverage,
                updated_original_demands_per_pose=32768), points


def design_envelope(builder, moved):
    """Step5.4 feedback: a finite envelope with the requested seated faces.

    Feed this shape back to Step5.3, which reselects connected material and
    applies its unchanged work/body exclusions. A returned fixture receives
    translations only; an already selected fixture is never silently trimmed.
    """
    stock = builder.stock
    for transform in moved['states']:
        stock = stock.trim_by_plane(tuple(transform[2, :3]), float(-transform[2, 3]*1000))
    return stock


def height_proposals(data):
    """Pure assembly Z proposals; LP floors are not a solid acceptance."""
    records = []
    for width, extra in [(0., 0.), (.005, 0.), (.005, .1), (.005, .3), (.03, .3)]:
        solver = B.Seating(data, width=width, preserve_material=True,
                           protect_full_wrap=width > 0)
        if width == 0:
            solver.head_bounds = []
        result = solver.solve('z', extra_lift=extra)
        records.append(dict(label=f'z_width{round(width*1000)}_extra{round(extra*1000)}mm', result=result))
        # This finite proposal family requires positive h in every state.
        # Infeasibility is NOT a proof over grounded-object branches.
        if not result['passed'] and width == 0:
            break
    return records


def transfer(mesh, moved):
    """Analytic clearance for a defined empty-fixture transfer path.

    Remove object, lift the fixture frame origin to radius+1mm, shortest
    SO(3) interpolation at that height while translating XY, then lower.
    At rotation all points have z>=origin_z-radius, for ALL intermediate
    angles. Sampling is for visualization only, not the certificate.
    """
    radius = float(np.linalg.norm(mesh.vertices, axis=1).max())
    states = moved['states']; transitions = []; frames = []
    if any(C.F.transform_points(mesh.vertices, state)[:, 2].min() < -1e-9 for state in states):
        raise ValueError('Empty transfer needs floor-legal seated endpoints')
    for i, (first, last) in enumerate(zip(states, states[1:])):
        safe_z = max(radius+.001, float(first[2, 3]), float(last[2, 3]))
        rotations = Rotation.from_matrix(np.array([first[:3, :3], last[:3, :3]]))
        interpolate = Slerp([0., 1.], rotations)
        path = []
        for fraction in np.linspace(0., 1., 5):
            matrix = first.copy(); matrix[2, 3] += fraction*(safe_z-first[2, 3]); path.append(matrix)
        for fraction in np.linspace(0., 1., 13):
            matrix = np.eye(4); matrix[:3, :3] = interpolate(fraction).as_matrix()
            matrix[:2, 3] = (1-fraction)*first[:2, 3]+fraction*last[:2, 3]
            matrix[2, 3] = safe_z; path.append(matrix)
        for fraction in np.linspace(0., 1., 5):
            matrix = last.copy(); matrix[2, 3] = safe_z+fraction*(last[2, 3]-safe_z); path.append(matrix)
        points = np.vstack([C.F.transform_points(mesh.vertices, matrix) for matrix in path])
        transitions.append(dict(from_pose=moved['rows'][i]['pose'], to_pose=moved['rows'][i+1]['pose'],
            object_removed_before_transfer=True, safe_frame_origin_height_mm=safe_z*1000,
            lift_from_first_mm=(safe_z-first[2, 3])*1000,
            lower_to_last_mm=(safe_z-last[2, 3])*1000,
            exact_rotation_minimum_z_bound_mm=(safe_z-radius)*1000,
            sampled_minimum_z_mm=float(points[:, 2].min())*1000))
        frames.extend(path)
    # Conservative bound over the entire continuous path, not a sampled BBX.
    origins = np.array([t[:3, 3] for t in frames]) if frames else np.array([states[0][:3, 3]])
    low, high = origins.min(0)-radius, origins.max(0)+radius
    low[2] = 0.
    return dict(passed=True, bounding_radius_mm=radius*1000, transitions=transitions,
                continuous_floor_clearance_proved_for_defined_path=True,
                process_bbx_upper_bound_m=[low.tolist(), high.tolist()],
                robot_tools_and_object_loading_trajectory_included=False), np.asarray(frames)
