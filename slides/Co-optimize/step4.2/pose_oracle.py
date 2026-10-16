"""Pose-owned work exclusions and shared rigid-material floor constraints.

Each task checks its own work cone in its own working state. Shared fixture
material must pass every task after the appropriate coordinate transform.
No work cone is copied untransformed from another task's world state.
"""
from dataclasses import dataclass
import numpy as np
from fixture_state import installed_state

SCHEMA = 'pose_specific_oracle_v2'


def transform_points(points, matrix):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def transform_planes(planes, target_to_source):
    """Pull back source inequalities a*x+b<=0 into target coordinates."""
    planes = np.asarray(planes)
    return np.c_[planes[:, :3] @ target_to_source[:3, :3],
                 planes[:, :3] @ target_to_source[:3, 3] + planes[:, 3]]


def inside_union(points, families, tolerance=1e-10):
    points = np.asarray(points).reshape(-1, 3)
    result = np.zeros(len(points), bool)
    for planes in families:
        result |= np.all(points @ planes[:, :3].T + planes[:, 3] <= tolerance, axis=1)
    return result


@dataclass(frozen=True)
class PoseConstraint:
    index: int
    pose: str
    T_world_fixture: np.ndarray
    Q_fixture_object: np.ndarray
    T_world_object: np.ndarray
    work_local: tuple
    work_common: tuple
    work_world: tuple
    work_face_ids: np.ndarray
    half_angle_deg: float
    work_access: object


class PoseOracleConstraints:
    def __init__(self, model, layout, work_access=None):
        self.poses = []
        work_access=model.work_rays if work_access is None else work_access
        for k in layout.active:
            T = installed_state(model.native, layout, layout.hosts[k])
            Q = layout.placements[k].copy()
            U = T @ Q
            work = work_access[k]
            fixed = getattr(getattr(model, 'base_oracle', None), 'fixed_work_planes', None)
            inverse_Q = fixed.inverse(Q) if fixed is not None else np.linalg.inv(Q)
            inverse_U = fixed.inverse(U) if fixed is not None else np.linalg.inv(U)
            local = (fixed.local[k] if fixed is not None else
                     tuple(p[:n].copy() for p, n in zip(work.planes, work.counts)))
            self.poses.append(PoseConstraint(int(k), model.poses[k], T, Q, U, local,
                fixed.transformed(k, inverse_Q) if fixed is not None else
                    tuple(transform_planes(p, inverse_Q) for p in local),
                fixed.transformed(k, inverse_U) if fixed is not None else
                    tuple(transform_planes(p, inverse_U) for p in local),
                np.asarray(work.face_ids, int).copy(), float(work.half_angle_deg),work))
        # h(q)>=0: a physical material point must be above every final floor.
        # Keep each final object-frame floor, including the object's Z move.
        # Pull it back through that task's own seat; U_i Q_i^-1 = T_i.
        self.floor_planes_object = np.array([p.T_world_object[2] for p in self.poses])
        self.floor_planes_common = np.array([transform_planes(p.T_world_object[2:3],
            np.linalg.inv(p.Q_fixture_object))[0] for p in self.poses])
        self.support_contact_clearance_m = float(getattr(model,'support_floor_clearance_m',0.))
        self.support_contact_planes_common = self.floor_planes_common.copy()
        self.support_contact_planes_common[:,3] -= self.support_contact_clearance_m
        self.by_index = {p.index: p for p in self.poses}

    def support_contact_allowed(self, points_common):
        """Contacts obey final floors; the active geometric margin is zero."""
        planes = self.support_contact_planes_common
        return np.all(np.asarray(points_common)@planes[:,:3].T+planes[:,3]>=-1e-10,axis=1)

    def classify_world(self, index, points):
        """Separate global floor exclusion from ONLY the selected task's cone.

        `current_pose_work` is the yellow visualization and per-task query.
        `shared_material_work` tests the permanent material across all tasks;
        it does not mean those tasks' world cones are simultaneously present.
        Exit sweeps and object interiors are separate BaseOracle constraints.
        """
        pose = self.by_index[index]
        points = np.asarray(points).reshape(-1, 3)
        common = transform_points(points, np.linalg.inv(pose.T_world_fixture))
        ground = np.any(common @ self.floor_planes_common[:, :3].T +
                        self.floor_planes_common[:, 3] < -1e-10, axis=1)
        own = inside_union(points, pose.work_world)
        shared = np.zeros(len(points), bool)
        for task in self.poses:
            shared |= inside_union(common, task.work_common)
        return dict(global_floor=ground, current_pose_work=own, shared_material_work=shared,
            support_contact_floor=~self.support_contact_allowed(common))

    def definition(self):
        return dict(schema=SCHEMA,
            task_condition='T_i S intersects only its own world work cone W_i in task i',
            shared_material_condition='S avoids union_i inverse(T_i) W_i in fixture material coordinates',
            floor_condition='h_i(q)=(T_i q)_z>=0 in every final installed state',
            support_contact_condition='(T_j q)_z >= contact clearance for every final state j; base feet use z>=0',
            support_contact_clearance_m=self.support_contact_clearance_m,
            all_support_contact_planes_fixture=self.support_contact_planes_common.tolist(),
            object_frame_floor_condition='(U_i p)_z>=0, with U_i including final XYZ translation',
            floor_pullback='U_i Q_i^-1 = T_i; object-seat Translation changes Q_i, installation Translation changes T_i and its floor offset',
            fixture_installation_translation_axes='world_xyz',
            all_floor_planes_object=self.floor_planes_object.tolist(),
            all_floor_planes_fixture=self.floor_planes_common.tolist(),
            work_cones_pose_specific=True,other_task_world_cones_not_superimposed=True,
            shared_rigid_material_must_pass_all_tasks=True,intermediate_transfer_states_constrained=False,
            poses=[dict(index=p.index,pose=p.pose,T_world_fixture=p.T_world_fixture.tolist(),
                Q_fixture_object=p.Q_fixture_object.tolist(),T_world_object=p.T_world_object.tolist(),
                work_face_ids=p.work_face_ids.tolist(),half_angle_deg=p.half_angle_deg,
                total_opening_deg=2*p.half_angle_deg,
                work_access_definition=(p.work_access.definition() if hasattr(p.work_access,'definition') else None)) for p in self.poses])
