"""Final fixture installations; native task rotations remain immutable."""
import numpy as np


def installed_states(native, layout):
    states = np.asarray(native).copy()
    offsets = getattr(layout, 'state_offsets', None)
    if offsets is not None:
        states[:, :3, 3] += offsets
    return states


def installed_state(native, layout, host):
    state = np.asarray(native[int(host)]).copy()
    offsets = getattr(layout, 'state_offsets', None)
    if offsets is not None:
        state[:3, 3] += offsets[int(host)]
    return state
