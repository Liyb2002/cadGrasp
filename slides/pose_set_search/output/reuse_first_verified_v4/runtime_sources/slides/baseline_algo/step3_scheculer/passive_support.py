"""One shared no-uplift equation; force signs are ON the workpiece.

Keep the physical demand in six coordinates. Lift head generators by their
vertical force, floor generators by zero, and add one nonnegative slack N:
sum(head Fz) - N = 0. All heads share N, so downward heads may cooperate with
upward heads. N is not an independent force acting on the workpiece.
"""
import numpy as np

MODE = 'connected_massless_support_no_uplift'


def target(values, dimension=7):
    values = np.asarray(values, float)
    if values.shape[-1] == dimension:
        return values
    if values.shape[-1] != 6 or dimension != 7:
        raise ValueError('Expected six physical demand coordinates or seven lifted coordinates')
    return np.concatenate([values, np.zeros(values.shape[:-1]+(1,))], axis=-1)


def heads(wrenches, scale):
    raw = np.asarray(wrenches, float).reshape(-1, 6)
    return np.c_[raw*np.asarray(scale), raw[:, 2]]


def floor(wrenches, scale):
    raw = np.asarray(wrenches, float).reshape(-1, 6)
    return np.vstack([np.c_[raw*np.asarray(scale), np.zeros(len(raw))],
                      np.r_[np.zeros(6), -1.]])


def description():
    return dict(mode=MODE, enforced=True, inequality='sum(head_force_on_workpiece_z) >= 0',
        excluded_contact='original_workpiece_floor_contact', support_self_weight_included=False,
        shared_by_all_heads=True, reaction_allocation='reoptimized_for_each_load_with_equilibrium',
        scope='Necessary no-uplift condition only; footprint, friction and tipping need final checks.')
