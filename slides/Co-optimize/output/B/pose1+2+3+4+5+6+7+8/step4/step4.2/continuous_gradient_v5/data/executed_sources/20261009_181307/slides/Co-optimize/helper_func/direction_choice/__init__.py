"""Continuous exit-direction update for all currently active object poses."""
import numpy as np
from continuous_support.objective import descent


def direction_choice(objective,current,*,difference_degrees=.75,step_degrees=2.,backtracks=4):
    updated,record=descent(objective,current,'direction',np.radians(difference_degrees),
                          np.radians(step_degrees),backtracks)
    if not record['accepted'] and any(len(group)>1 for group in record.get('coordinate_groups',[])):
        updated,individual=descent(objective,current,'direction',np.radians(difference_degrees),
                                  np.radians(step_degrees),backtracks,coupled=False)
        individual['shared_direction_attempt']=record
        return updated,individual
    return updated,record
