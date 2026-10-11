"""The user's Step5.3/5.4 contract; never inherit the old exit exclusion."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BasePolicy:
    forbid_entire_work_cones: bool = True
    forbid_stationary_object_interiors: bool = True
    forbid_exit_paths: bool = False
    floor_owner: str = 'step5.4'
    pose_change: str = 'independent task-world +Z of object and fixture together'
    keep_relative_object_fixture_placements: bool = True
    allow_floor_contact: bool = True
    fixture_weight_ratio: float = 0.0

    def record(self):
        return asdict(self)


POLICY = BasePolicy()
