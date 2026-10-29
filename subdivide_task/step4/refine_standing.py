"""Reduce material in a fully certified standing design, preserving its uses."""
import argparse
from pathlib import Path
from subdivide_task.step1.common import DEFAULT_OUTPUT
from subdivide_task.operations.construction import CompactConstructionModel
from subdivide_task.operations.installations import standing_pairs
from .optimize import Optimizer
from .spatial_translation import sweep


def run(folder):
    model=CompactConstructionModel(DEFAULT_OUTPUT,'B')
    optimizer=Optimizer(model,folder,resume=True)
    if not standing_pairs(optimizer.layout):raise ValueError('An actual standing installation is required')
    optimizer.try_operation(optimizer.layout,'support_refinement',
        refinement='replace full side wall with a narrow physical post')
    for pitch,width in((.035,.012),(.025,.012),(.02,.012),(.015,.01)):
        model.stock_grid=(pitch,width)
        if optimizer.try_operation(optimizer.layout,'support_refinement',
                refinement='retain fitted contact skin and floor plates, replace casting fill by real posts',
                post_pitch_m=pitch,post_width_m=width):
            sweep(optimizer,rounds=2)
            break
    return optimizer


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workdir',type=Path,required=True)
    run(p.parse_args().workdir)
