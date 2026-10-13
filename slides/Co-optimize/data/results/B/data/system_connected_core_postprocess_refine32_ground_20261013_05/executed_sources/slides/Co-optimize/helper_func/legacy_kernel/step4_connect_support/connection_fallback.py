"""Connect by widening a carved envelope when thin local bridges fail.

Lengths are metres; Manifold solids are in the caller's normalized coordinates.
Instead of another finite set of curves, the last attempt adds ALL legal volume
inside a saved construction envelope. If that maximal addition is disconnected,
no smaller additive connector within that envelope can connect the input. This
is a bounded-domain guarantee, not a claim about physically disconnected space,
minimum wall thickness, or exact arithmetic in floating-point mesh Booleans.
"""
import itertools
import time

import manifold3d as md
import numpy as np
from scipy.spatial import cKDTree


def vertices(value, scale):
    return np.asarray(value.to_mesh64().vert_properties[:, :3])*scale


def box(bounds_m, scale):
    low, high = np.asarray(bounds_m, float)
    return md.Manifold.cube(((high-low)/scale).tolist()).translate((low/scale).tolist())


def connector_in_envelope(full, bounds_m, carve, scale):
    """Keep only legal components that overlap at least two existing bodies."""
    components = full.decompose()
    legal = carve(box(bounds_m, scale))
    useful = []
    for part in legal.decompose():
        hits = sum(abs(float((part ^ component).volume()))*scale**3 > 1e-12
                   for component in components)
        if hits >= 2:
            useful.append(part)
    if not useful:
        return None
    bridge = md.Manifold.batch_boolean(useful, md.OpType.Add)
    joined = full+bridge
    if len(joined.decompose()) >= len(components):
        return None
    return joined, bridge


def connect(full, carve, scale, obstacle_bounds_m, initial_padding_m=.004):
    started = time.perf_counter()
    components = full.decompose()
    if len(components) < 2:
        raise ValueError('Fallback requires at least two disconnected components')
    clouds = [vertices(c, scale) for c in components]
    points = np.vstack(clouds)
    pairs = []
    for i, j in itertools.combinations(range(len(components)), 2):
        distances, ids = cKDTree(clouds[j]).query(clouds[i])
        k = int(np.argmin(distances))
        pairs.append((float(distances[k]), i, j, clouds[i][k], clouds[j][ids[k]]))
    pairs.sort(key=lambda p:p[:3])
    attempts = []

    def attempt(bounds, phase, padding):
        result = connector_in_envelope(full, bounds, carve, scale)
        attempts.append(dict(phase=phase, padding_m=padding, connected=result is not None))
        if result is None:
            return None
        joined, bridge = result
        record = dict(kind='carved_connection_envelope', phase=phase,
            bounds_m=np.asarray(bounds).tolist(), padding_m=padding,
            component_count_before=len(components), component_count_after=len(joined.decompose()),
            added_volume_cm3=float((joined.volume()-full.volume())*scale**3*1e6),
            attempts=attempts.copy(), search_seconds=time.perf_counter()-started,
            search_scope='all legal solid inside the recorded envelope',
            strength_or_minimum_thickness_verified=False)
        return joined, bridge, record

    # Grow all nearby envelopes monotonically; allow routes on either side of
    # an obstacle rather than forcing one upward midpoint.
    for padding in initial_padding_m*2.**np.arange(5):
        for _, i, j, a, b in pairs:
            result = attempt([np.minimum(a,b)-padding, np.maximum(a,b)+padding],
                             'local_expansion', float(padding))
            if result is not None:
                return result

    # Complete the construction in a conservative domain containing every
    # original body and every exclusion solid, including room around its rim.
    # The caller also clips floors and the terminal withdrawal halfspaces.
    obstacle_bounds_m = np.asarray(obstacle_bounds_m)
    low = np.minimum(points.min(0), obstacle_bounds_m[0])
    high = np.maximum(points.max(0), obstacle_bounds_m[1])
    margin = max(initial_padding_m*4, float(np.max(high-low))*.05)
    for multiplier in (1., 2., 4.):
        padding = margin*multiplier
        result = attempt([low-padding, high+padding], 'full_free_envelope', padding)
        if result is not None:
            return result
    error = RuntimeError('Even the full legal construction envelope cannot join the remaining components')
    error.connection_diagnostic = dict(attempts=attempts,
        bounds_m=[(low-padding).tolist(),(high+padding).tolist()],
        scope='No additive connection in this bounded envelope under the current clipping and Boolean tolerances')
    raise error


def replay(full, record, carve, scale):
    result = connector_in_envelope(full, record['bounds_m'], carve, scale)
    if result is None:
        raise RuntimeError('Cached fallback envelope no longer connects the input bodies')
    return result
