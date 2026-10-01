"""Trajectory-local Step3 paths, without changing shared upstream geometry code."""
from contextlib import contextmanager
from contextvars import ContextVar
from step3_scheculer import contacts as I
from step1.cases import pose_name

_trajectory = ContextVar('step3_trajectory', default=None)
_round_owners = ContextVar('step3_round_owners', default={})


@contextmanager
def trajectory(number):
    if number is not None and (not isinstance(number, int) or number < 0):
        raise ValueError('Expected a nonnegative trajectory index')
    token = _trajectory.set(number)
    try:
        yield
    finally:
        _trajectory.reset(token)


def stage_folder(name, step):
    path = I.OUTPUTS/name/pose_name()/step
    number = _trajectory.get()
    return path if number is None else path/f'trajectory_{number:03d}'


@contextmanager
def round_owners(owners):
    """Resolve immutable ancestor rounds without copying or recomputing them."""
    token = _round_owners.set({int(k): int(v) for k, v in owners.items()})
    try:
        yield
    finally:
        _round_owners.reset(token)


def owner_stage(name, step, round_number):
    owner = _round_owners.get().get(round_number, _trajectory.get())
    with trajectory(owner):
        return stage_folder(name, step)


def folder(name, step, round_number):
    return owner_stage(name, step, round_number)/f'round_{round_number:03d}' 
