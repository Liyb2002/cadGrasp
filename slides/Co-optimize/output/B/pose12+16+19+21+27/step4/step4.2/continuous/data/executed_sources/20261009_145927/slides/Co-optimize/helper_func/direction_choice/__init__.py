"""Continuous exit-direction update for all currently active object poses."""
import numpy as np
from continuous_support.objective import descent


def direction_choice(objective,current,*,difference_degrees=.75,step_degrees=2.,backtracks=4):
    return descent(objective,current,'direction',np.radians(difference_degrees),
                   np.radians(step_degrees),backtracks)
