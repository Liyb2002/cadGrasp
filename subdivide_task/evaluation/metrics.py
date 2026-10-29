"""Overlapping unsupported poses versus actual supported static layouts."""
import numpy as np

METRIC_VERSION = 'static_occupancy_v2_overlapping_objects'
SUPPORT_POSE_METRIC_VERSION = 'installed_support_pose_membership_v1'
BOX_KEYS = ('object_static', 'with_step4_support', 'with_full_support')
STATE_KEYS = ('objects_in_final_layout',) + BOX_KEYS[1:]


def transformed(vertices,matrix):
    return np.asarray(vertices)@matrix[:3,:3].T+matrix[:3,3]


def bbox(clouds):
    if not clouds:raise ValueError('An occupied space cannot be empty')
    low=np.min([np.asarray(p).min(0) for p in clouds],axis=0)
    high=np.max([np.asarray(p).max(0) for p in clouds],axis=0)
    dimensions=high-low
    return dict(min_m=low.tolist(),max_m=high.tolist(),dimensions_m=dimensions.tolist(),
        dimensions_cm=(dimensions*100).tolist(),volume_cm3=float(np.prod(dimensions)*1e6),
        volume_litres=float(np.prod(dimensions)*1000))


def unsupported_transforms(body, layout):
    """Smallest world-axis union box with fixed rotations and free translations.

    Tasks run sequentially, so their object volumes may overlap. Align each
    rotated object's horizontal box center and place its lowest point at z=0.
    The union then attains max(pose extent) on every axis, a lower bound for
    any translation assignment. Never borrow seats/hosts from the fixture.
    """
    matrices = {}
    for k in layout['active']:
        k = int(k)
        T = np.eye(4)
        T[:3, :3] = layout['native_world'][k, :3, :3]
        points = transformed(body, T)
        low, high = points.min(0), points.max(0)
        T[:2, 3] = -(low[:2] + high[:2])/2
        T[2, 3] = -low[2]
        matrices[k] = T
    return matrices


def static_bounds(body, layout, step4_vertices=None, full_vertices=None, pose_names=None, execution=None):
    """Compare one overlapping no-fixture box to actual supported layouts.

    A state cell reserves the static union of every task served in that fixture
    placement, although those tasks run one at a time. Different state cells
    are summed as separate reserved spaces, without imposing arbitrary offsets.
    State allocations and final-layout objects are separate diagnostics;
    neither defines the no-fixture baseline. No swept endpoint is included.
    """
    active = [int(k) for k in layout['active']]
    if not active:
        raise ValueError('An occupied layout cannot be empty')
    names = pose_names if pose_names is not None else [f'pose_{k + 1}' for k in range(len(layout['hosts']))]
    unsupported = unsupported_transforms(body, layout)
    groups = {}; poses = []; objects = []; baseline_objects = []
    for k in active:
        T = (np.asarray(execution[k]['world_from_support']) if execution is not None
             else layout['native_world'][layout['hosts'][k]])
        Q = layout['placements'][k]
        vertices = transformed(body, T @ Q)
        objects.append(vertices)
        baseline_vertices = transformed(body, unsupported[k])
        baseline_objects.append(baseline_vertices)
        registered = int(layout['hosts'][k]) == k and np.allclose(Q, np.eye(4), atol=1e-10, rtol=0)
        poses.append(dict(pose=names[k], layout_index=k, host_pose=names[int(layout['hosts'][k])],
            juxtaposed=not registered, rotating_reuse=bool(registered),
            unsupported_world_transform=unsupported[k].tolist(),
            object_static=bbox([baseline_vertices]), objects_in_final_layout=bbox([vertices])))
        key = np.round(T, 11).tobytes()
        group = groups.setdefault(key, dict(transform=T, indices=[], objects=[]))
        group['indices'].append(k); group['objects'].append(vertices)

    fixture_clouds = {key: [] for key in BOX_KEYS[1:]}; states = []
    for group in groups.values():
        T = group['transform']
        state = dict(state=len(states) + 1, fixture_transform=T.tolist(),
            poses=[names[k] for k in group['indices']], pose_count=len(group['indices']),
            objects_in_final_layout=bbox(group['objects']))
        for key, vertices in [('with_step4_support', step4_vertices), ('with_full_support', full_vertices)]:
            if vertices is None:
                state[key] = None
            else:
                fixture = transformed(vertices, T)
                fixture_clouds[key].append(fixture)
                state[key] = bbox(group['objects'] + [fixture])
        states.append(state)

    result = dict(metric_version=METRIC_VERSION,
        coordinate_frame="centered_execution" if execution is not None else "search_layout",
        support_pose_metric_version=SUPPORT_POSE_METRIC_VERSION, object_static=bbox(baseline_objects),
        objects_in_final_layout=bbox(objects),
        object_baseline='fixed_native_rotations; overlapping_xy_box_centers; lowest_point_z0',
        object_baseline_translation_optimal=True, same_final_object_placements_in_all_columns=False,
        pose_count=len(active), state_count=len(states), per_pose=poses, fixture_states=states,
        support_pose_count=len(states),
        object_pose_counts_per_support_pose=[s['pose_count'] for s in states],
        independent_pose_object_sum_litres=sum(p['object_static']['volume_litres'] for p in poses),
        juxtaposed_pose_count=sum(p['juxtaposed'] for p in poses),
        rotating_reuse_pose_count=sum(p['rotating_reuse'] for p in poses),
        multi_pose_fixture_state_count=sum(s['pose_count'] > 1 for s in states),
        singleton_fixture_state_count=sum(s['pose_count'] == 1 for s in states),
        poses_in_multi_pose_fixture_states=sum(s['pose_count'] for s in states if s['pose_count'] > 1),
        state_space_sums_litres={})
    for key in BOX_KEYS[1:]:
        result[key] = bbox(objects + fixture_clouds[key]) if fixture_clouds[key] else None
    for key in STATE_KEYS:
        result['state_space_sums_litres'][key] = (sum(s[key]['volume_litres'] for s in states)
            if all(s[key] is not None for s in states) else None)
    return result
