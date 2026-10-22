"""Bounded shared-installation proposals before the original population search.

All poses retain their task orientation, height, force family and exit direction
in world coordinates. A common fixture installation removes conflicting floor
planes; small relative seat translations can recover lost contact material.
Every accepted proposal passes the same full-solid gate as population search.
"""
import numpy as np

from .search import context  # Load the original reference modules through adapter.
from whole_task_2.step1.common import write_json
from population32 import centered_seat
from fixture_state import installed_state
from translation.layout_update import shift


def common_installation(model, layout):
    q = layout.copy()
    anchor = q.active[0]  # Coordinate reference only; no anatomical pose choice.
    for k in q.active:
        if k != anchor:
            q = centered_seat(model, q, k, anchor)
    return q


def planar_offset(model, layout, owner, delta_xy):
    T = installed_state(model.native, layout, int(layout.hosts[owner]))
    delta = T[:3, :3].T @ np.r_[delta_xy, 0.]
    return shift(model, layout, owner, delta)


def precheck(search, initial, maximum_trials=25):
    """Return a certified solid, or leave the original cold search as fallback."""
    model = search.model
    folder = search.search.out.parent / 'station_precheck'
    folder.mkdir(parents=True, exist_ok=True)
    rows, seen = [], set()

    def check(q, operation, detail=None, result=None):
        if q.key() in seen or len(rows) >= maximum_trials:
            return None
        seen.add(q.key())
        index = len(rows)
        np.savez_compressed(folder / f'candidate_{index:02d}.npz', **q.arrays(model.native))
        row = dict(index=index, operation=operation, detail=detail or {}, completed=False)
        rows.append(row)
        try:
            item = search.checked(search.preview_cached(q, operation, detail=detail, result=result))
            # Includes retained-core contact recertification, connected legal
            # space, construction and the continuous-force/full-access gate.
            search.certify([item], connections=1, constructions=1)
            row.update(counts=item.result['counts'], oracle_status=item.oracle['status'],
                       completed=bool(item.completed), construction=item.construction,
                       errors=search.errors[-3:])
        except (ValueError, RuntimeError, AssertionError) as error:
            row['error'] = str(error)
            item = None
        write_json(folder / 'report.json', dict(
            method='shared_installation_then_failure_ranked_planar_seat_offsets',
            maximum_trials=maximum_trials, extent_m=float(model.extent),
            offset_fractions=[.25, .5, .75], world_xy_directions=[[1,0],[0,1],[-1,0],[0,-1]],
            physical_gates_unchanged=True, external_warm_layout_used=False,
            complete=bool(item is not None and item.completed), candidates=rows))
        print('STATION PRECHECK', index, operation, row.get('counts'), row.get('oracle_status'),
              'complete=', row['completed'], flush=True)
        return item

    item = check(initial['layout'], 'registered-initial', result=initial)
    if item is not None and item.completed:
        return item, rows
    q = common_installation(model, initial['layout'])
    centered = check(q, 'shared-installation-centered')
    if centered is not None and centered.completed:
        return centered, rows
    # Order depends on actual failed loads, never a prescribed ear/mouth ID.
    failures = {k: int(np.count_nonzero(~centered.result['masks'][k]))
                if centered is not None else 1 for k in q.active}
    owners = sorted(q.active, key=lambda k: (-failures[k], k))
    for fraction in (.25, .5, .75):
        for owner in owners:
            for direction in ((1.,0.), (0.,1.), (-1.,0.), (0.,-1.)):
                if len(rows) >= maximum_trials:
                    return None, rows
                delta = model.extent * fraction * np.array(direction)
                try:
                    trial = planar_offset(model, q, owner, delta)
                except (ValueError, RuntimeError, AssertionError):
                    continue
                item = check(trial, 'shared-installation-planar-offset',
                             dict(owner=int(owner), delta_world_xy_m=delta.tolist(),
                                  fraction_of_mesh_extent=fraction))
                if item is not None and item.completed:
                    return item, rows
    return None, rows
