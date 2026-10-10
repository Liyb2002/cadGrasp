"""Deterministic system-floor demands at the final object/fixture placement.

The input is the unchanged COM wrench in its native task-world orientation.
No reaction solve, resampling, Boolean or Step4 model is needed here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from codes.precompute_objects.floor_points import pressure_centers, validate_frames


def transform_points(points, transform):
    points = np.asarray(points, dtype=float)
    transform = np.asarray(transform, dtype=float)
    return points @ transform[:3, :3].T + transform[:3, 3]


def final_task_frame(native_frames, placements, hosts, index):
    """Return fixture and object world frames without changing task orientation."""
    fixture = np.asarray(native_frames[int(hosts[index])], dtype=float)
    obj = fixture @ np.asarray(placements[index], dtype=float)
    validate_frames(np.asarray([fixture, obj]))
    if not np.allclose(obj[:3, :3], native_frames[index, :3, :3], atol=1e-10, rtol=0):
        raise ValueError('Final object orientation differs from its native task; COM loads cannot be reused unchanged')
    return fixture, obj


def system_floor_demands(wrench_com, object_com_world_m, *,
                         fixture_weight_ratio=0., fixture_com_world_m=None):
    """Project the whole-system reaction demand onto z=0.

    Forces are in object-weight units; moments are in object-weight * metres.
    fixture_weight_ratio = fixture mass / object mass. Default zero keeps the
    existing objects/ demand model; no material density is silently assumed.
    Object-fixture reactions are internal to the whole system. Object-floor
    and fixture-floor reactions together provide its external ground reaction.
    """
    loads = np.asarray(wrench_com, dtype=float)
    com = np.asarray(object_com_world_m, dtype=float)
    ratio = float(fixture_weight_ratio)
    if (loads.ndim != 2 or loads.shape[1] != 6 or len(loads) == 0
            or not np.isfinite(loads).all() or com.shape != (3,)
            or not np.isfinite(com).all()):
        raise ValueError('Finite nonempty Nx6 COM demands and one 3D COM required')
    if not np.isfinite(ratio) or ratio < 0:
        raise ValueError('Fixture weight ratio must be finite and nonnegative')
    world = loads.copy()
    world[:, 3:] += np.cross(com, loads[:, :3])
    if ratio:
        fixture_com = np.asarray(fixture_com_world_m, dtype=float)
        if fixture_com.shape != (3,) or not np.isfinite(fixture_com).all():
            raise ValueError('Fixture COM required when including fixture weight')
        fixture_reaction = np.array([0., 0., ratio])
        world[:, :3] += fixture_reaction
        world[:, 3:] += np.cross(fixture_com, fixture_reaction)
    xy, normal = pressure_centers(world, np.zeros(3))
    return dict(wrench_world_origin=world, floor_demands_xy_m=xy,
                total_floor_normal_mg=normal,
                floor_demands_world_m=np.c_[xy, np.zeros(len(xy))])
