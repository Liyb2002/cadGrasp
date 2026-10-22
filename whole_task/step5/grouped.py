"""Opt-in five-installation adapters; physical Step5 guards stay unchanged."""
import copy
import json
from pathlib import Path
import numpy as np
from whole_task.step1.common import write_json
from whole_task.step4.compact_grouped import shared_static_bounds
from .video.workflow import Workflow


def installation_groups(rows):
    """Group equal physical transforms, including signed-zero roundoff."""
    groups = []
    for k, row in rows.items():
        matrix = np.asarray(row['fixture'])
        for group in groups:
            if np.allclose(matrix, rows[group[0]]['fixture'], atol=1e-10, rtol=0):
                group.append(k)
                break
        else:
            groups.append([k])
    return groups


class SharedWorkflow(Workflow):
    def __init__(self, data):
        groups = installation_groups(data['rows'])
        # Mutate the shared order list so the caller records the same order.
        data['order'][:] = [k for group in groups for k in group]
        super().__init__(data)
        moves = sum(s['label'] == 'seat_fixture' for s in self.segments)
        if moves != len(groups):
            raise ValueError('Fixture was repositioned inside a shared installation')


def canonical_presentation(layout, result):
    """Use the already-counted physical states for display grouping only."""
    result = copy.deepcopy(result)
    for group in result['fixture_states']:
        for pose in group['poses']:
            k = next(p['layout_index'] for p in result['per_pose'] if p['pose'] == pose)
            result['execution'][k]['world_from_support'] = group['fixture_transform']
    return ORIGINAL_PRESENTATION(layout, result)


def grouped_prepare(out, duration=48., fps=24):
    plan = ORIGINAL_PREPARE(out, duration, fps)
    rows = {k: dict(fixture=matrix) for k, matrix in enumerate(plan['fixture_poses'])}
    groups = installation_groups(rows)
    plan.update(unique_support_installations=len(groups),
                support_installation_groups=groups,
                support_grouping_tolerance=1e-10,
                support_fixed_within_each_group=True)
    if plan['execution_order'] != [k for group in groups for k in group]:
        raise ValueError('Movie order does not keep shared installations together')
    write_json(Path(out)/'video/plan.json', plan)
    return plan


ORIGINAL_PRESENTATION = None
ORIGINAL_PREPARE = None


def configure():
    """Install counting/presentation adapters only, never force tolerances."""
    global ORIGINAL_PRESENTATION, ORIGINAL_PREPARE
    from whole_task.evaluation import run as evaluation
    from . import validate
    from .video import prepare
    if ORIGINAL_PRESENTATION is None:
        ORIGINAL_PRESENTATION = evaluation.presentation
        ORIGINAL_PREPARE = prepare.prepare
    evaluation.static_bounds = shared_static_bounds
    evaluation.presentation = canonical_presentation
    validate.static_bounds = shared_static_bounds
    prepare.Workflow = SharedWorkflow
    prepare.prepare = grouped_prepare
