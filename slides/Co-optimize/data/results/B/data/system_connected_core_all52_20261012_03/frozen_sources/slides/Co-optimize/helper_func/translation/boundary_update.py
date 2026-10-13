"""Use the repaired boundary-aware descent for the translation operation."""
from continuous_support.boundary_descent import block_descent


def translation(objective, current, *, difference_fraction=1/512,
                step_fraction=1/32, backtracks=4):
    return block_descent(objective, current, 'translation', difference_fraction,
                         step_fraction, backtracks)
