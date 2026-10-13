"""Continuous world-horizontal changes to explicitly juxtaposed placements."""
from continuous_support.objective import descent


def translation(objective,current,*,difference_fraction=1/512,step_fraction=1/32,backtracks=4):
    return descent(objective,current,'translation',difference_fraction,step_fraction,backtracks)
