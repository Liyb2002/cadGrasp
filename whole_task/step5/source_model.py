"""Restore the exact task model and layout carried by a Step4/Step5 report."""
from pathlib import Path
import numpy as np


def restore(report, layout_path):
    if report.get('schema') == 'whole_task_continuous_axial_juxtaposition_v1' or report.get('pose_model') == 'continuous_axial':
        from whole_task.step4.angular_model import AngularTaskModel, AngularLayout
        model = AngularTaskModel(name=report['object'], output_root=Path(report['input_root']))
        with np.load(layout_path) as arrays:
            layout = AngularLayout.from_arrays(arrays)
        model.require_connected_core = False
        model.activate(layout)
    else:
        from whole_task.step4.mobile import MobileTaskModel
        from whole_search.model import Layout
        model = MobileTaskModel(report['poses'], report['object'], report['output_root'])
        with np.load(layout_path) as arrays:
            layout = Layout.from_arrays(arrays)
    if model.poses != report['poses']:
        raise ValueError('Step5 task names differ from the certified Step4 input')
    return model, layout
