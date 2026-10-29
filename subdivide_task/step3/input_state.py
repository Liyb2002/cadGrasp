"""State of the transition from frozen task inputs to joint initialization."""
from subdivide_task.step1.common import ROOT

FAMILY_ROOT = ROOT/'subdivide_task/data/continuous13'


class JointInitializationRequired(RuntimeError):
    """The removed frozen-pose stage cannot supply search state anymore."""


def require_joint_initialization():
    raise JointInitializationRequired(
        'Independent Step2 and its frozen pose/load inputs have been deleted. '
        'Step3 must initialize poses and supports jointly from Step1 before '
        'this legacy downstream entry point can run. No old inputs were read '
        'or regenerated.')
