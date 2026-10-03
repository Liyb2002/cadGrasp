"""One physical head per Step3 ID, under the original contact correspondence.

All arrays use metres and row vectors: local = world @ basis + offset.
Independent reseating and duplicating a shared patch are not admissible here.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree

POSITION_TOL = 1e-8
FLOOR_TOL = 1e-9


class SharedHeadRegistrationError(ValueError):
    """A saved construction changed the identity of the shared physical head."""

    def __init__(self, message, checks):
        super().__init__(message)
        self.registration_checks = checks


@dataclass
class RegisteredHead:
    ident: str
    pose: int
    active_poses: tuple
    cells: list
    contact_points: np.ndarray


def cloud_error(a, b):
    return float(max(cKDTree(a).query(b)[0].max(), cKDTree(b).query(a)[0].max()))


def fixed_placements(tasks):
    """Use the first task frame as a gauge; other transforms are not free."""
    frames = [np.asarray(p.domain.data['frame']['T_world_mesh'], float) for p in tasks]
    for transform in frames:
        if (transform.shape != (4, 4) or not np.isfinite(transform).all()
                or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-12, rtol=0)
                or not np.allclose(transform[:3, :3].T@transform[:3, :3], np.eye(3), atol=1e-12, rtol=0)
                or abs(np.linalg.det(transform[:3, :3])-1) > 1e-12):
            raise ValueError('Task frame must be a proper rigid transform')
    transforms = [t@np.linalg.inv(frames[0]) for t in frames]
    bases = np.asarray([t[:3, :3] for t in transforms])
    offsets = np.asarray([-t[:3, 3]@b for t, b in zip(transforms, bases)])
    return bases, offsets


def register(groups, heads, bases, offsets, *, general_layout=False):
    """Reject a separated shared head BEFORE deduplicating anything."""
    ids = [set(c['candidate_id'] for c in row) for row in groups]
    if not general_layout and (len(ids) != 2 or any(len(row) != 3 for row in groups)
            or any(len(row) != 3 for row in ids)
            or len(ids[0] | ids[1]) != 5 or len(ids[0] & ids[1]) != 1):
        raise ValueError('Expected two active triples, five IDs and exactly one shared head')
    count = len(groups)
    if (count < 2 or len(heads) != count or
            any(not group or len(group) != len(unique_ids) or len(cells) != len(group)
                for group, unique_ids, cells in zip(groups, ids, heads))):
        raise ValueError('Head cells and unique IDs must match every nonempty active group')
    bases, offsets = np.asarray(bases), np.asarray(offsets)
    if bases.shape != (count, 3, 3) or offsets.shape != (count, 3) or not np.isfinite(bases).all() or not np.isfinite(offsets).all():
        raise ValueError('Invalid fixture transforms')
    for b in bases:
        if not np.allclose(b@b.T, np.eye(3), atol=1e-12, rtol=0) or abs(np.linalg.det(b)-1) > 1e-12:
            raise ValueError('Fixture transform is not a proper rotation')
    unique, errors = {}, []
    for k, (contacts, cells, basis, offset) in enumerate(zip(groups, heads, bases, offsets)):
        for contact, pieces in zip(contacts, cells):
            ident = contact['candidate_id']
            local = [np.asarray(v)@basis+offset for v in pieces]
            patch = np.asarray(contact['triangles_m']).reshape(-1, 3)@basis+offset
            if ident in unique:
                original = unique[ident]
                surface_error = cloud_error(original.contact_points, patch)
                solid_error = cloud_error(np.concatenate(original.cells), np.concatenate(local))
                if max(surface_error, solid_error) > POSITION_TOL:
                    raise ValueError(f'Shared head {ident} is split: surface error {surface_error:.9g} m; solid error {solid_error:.9g} m')
                original.active_poses += (k,)
                errors.append(dict(candidate_id=ident, contact_surface_error_m=surface_error,
                                   head_solid_vertex_error_m=solid_error))
            else:
                unique[ident] = RegisteredHead(ident, k, (k,), local, patch)
    records = list(unique.values())
    minimum_heights = []
    vertices = np.concatenate([v for h in records for v in h.cells])
    for b, o in zip(bases, offsets):
        minimum_heights.append(float(((vertices-o)@b.T)[:, 2].min()))
    check = dict(passed=True, physical_head_count=len(records),
        shared_head_count=sum(len(h.active_poses)>1 for h in records), shared_instance_checks=len(errors),
        shared_heads=errors, active_head_ids=[sorted(row) for row in ids],
        physical_heads=[dict(candidate_id=h.ident, active_pose_indices=list(h.active_poses)) for h in records],
        minimum_head_height_m=minimum_heights,
        all_heads_above_both_floors=all(z >= -FLOOR_TOL for z in minimum_heights),
        all_heads_above_all_floors=all(z >= -FLOOR_TOL for z in minimum_heights),
        tolerance_m=POSITION_TOL)
    return records, check


def floor_compatibility(tasks, demands, bases, offsets):
    """Necessary halfplane test on ALL original CoPs, including object pivot.

Any admissible contact on floor k must also lie above the other floor.
Nonnegative floor normals make the total CoP a convex combination of these
contacts and the object's existing pivot. A negative affine halfplane value
therefore disproves feasibility under this fixed contact registration.
"""
    rows = []
    if len(tasks) < 2 or not (len(tasks) == len(demands) == len(bases) == len(offsets)):
        raise ValueError('Matching task, demand and transform counts required')
    for k, (task, xy) in enumerate(zip(tasks, demands)):
        for j in range(len(tasks)):
            if j == k:
                continue
            normal = bases[j][2]
            coefficients = np.r_[(bases[k]@normal)[:2], (offsets[k]-offsets[j])@normal]
            heights = np.asarray(xy)@coefficients[:2]+coefficients[2]
            pivot_height = float((np.asarray(task.floor)@bases[k]+offsets[k]-offsets[j])@normal)
            if pivot_height < -FLOOR_TOL:
                raise ValueError('Object pivot violates the other floor: the halfplane certificate is inapplicable')
            worst = int(np.argmin(heights))
            count = int(np.count_nonzero(heights < -FLOOR_TOL))
            rows.append(dict(pose=task.pose, other_pose=tasks[j].pose, passed=count == 0,
                original_sample_count=len(xy), violating_sample_count=count,
                floor_xy_halfplane_coefficients=coefficients.tolist(),
                minimum_other_floor_height_m=float(heights[worst]),
                object_pivot_other_floor_height_m=pivot_height,
                worst_sample_index=worst, worst_demand_xy_m=np.asarray(xy[worst]).tolist(),
                tolerance_m=FLOOR_TOL))
    return dict(passed=all(r['passed'] for r in rows), per_pose=rows,
        scope='Fixed original complete contact correspondence and task poses; massless support; unilateral planar ground contacts',
        extra_loads_added=False)


def require_saved_registration(case, placement):
    """Reject split saved layouts; record what exact registration would imply.

    This is a mandatory model check even when final load audits are disabled.
    Correcting the transforms invalidates the old feet and bodies, so this never
    silently substitutes new transforms into an old construction recipe.
    """
    try:
        _, check = register(case.groups, case.heads, placement['bases'], placement['offsets'])
    except ValueError as error:
        checks = dict(saved_layout=dict(passed=False, reason=str(error)))
        bases, offsets = fixed_placements(case.tasks)
        _, registered = register(case.groups, case.heads, bases, offsets)
        floor = floor_compatibility(case.tasks, case.demands, bases, offsets)
        checks['exact_registration'] = dict(bases=bases.tolist(), offsets=offsets.tolist(),
            registration=registered, floor_compatibility=floor,
            constructed=False, full_verification_performed=False)
        failures = ', '.join(f"{r['pose']}: {r['violating_sample_count']}/{r['original_sample_count']}"
                             for r in floor['per_pose'] if not r['passed'])
        message = str(error) + '. Saved six-patch construction is invalid for shared-head 3+2.'
        if failures:
            message += ' Exact registration also fails the floor necessary condition (' + failures + ').'
        raise SharedHeadRegistrationError(message, checks) from error
    return check
