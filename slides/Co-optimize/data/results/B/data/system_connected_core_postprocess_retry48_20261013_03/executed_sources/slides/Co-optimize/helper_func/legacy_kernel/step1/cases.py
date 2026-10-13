"""Select one target pose without mixing artifacts from different cases."""
from contextlib import contextmanager
import os
import re


def normalize_pose(value):
    value = str(value)
    if value.isdecimal():
        value = 'pose_' + value
    if not re.fullmatch(r'pose_[1-9][0-9]*', value):
        raise ValueError('pose must be pose_<positive integer>, for example pose_3')
    return value


def pose_name():
    return normalize_pose(os.environ.get('CADGRASP_POSE', 'pose_1'))


@contextmanager
def selected_pose(value=None):
    previous = os.environ.get('CADGRASP_POSE')
    os.environ['CADGRASP_POSE'] = normalize_pose(value) if value is not None else pose_name()
    try:
        yield pose_name()
    finally:
        if previous is None:
            os.environ.pop('CADGRASP_POSE', None)
        else:
            os.environ['CADGRASP_POSE'] = previous
