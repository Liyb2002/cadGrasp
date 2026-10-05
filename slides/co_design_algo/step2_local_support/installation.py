"""Initial lying installation expressed in the task's coordinates.

Loads remain in the task frame. Only module assembly uses the declared rest pose.
Horizontal placement of that rest pose can change in the presentation scene.
"""
import json
from pathlib import Path
import numpy as np
from step1.needs import ROOT

CONTACT_CLEARANCE_M = .0015


def scene(name, domain):
    path = ROOT/'objects'/name/'poses.json'
    data = json.loads(path.read_text())
    initial = np.asarray(data['rest']['T_world_mesh'], float)
    task = np.asarray(domain.data['frame']['T_world_mesh'], float)
    transform = initial @ np.linalg.inv(task)
    plane = transform[2].copy()
    return dict(kind='rest_pose_module_installation_v1', initial_pose='rest',
                T_initial_from_task=transform.tolist(), floor_plane=plane.tolist(),
                contact_floor_clearance_m=CONTACT_CLEARANCE_M,
                direction_coordinate_frame='task world; transform with T_initial_from_task to display at rest',
                initial_pose_source=str(path.relative_to(ROOT)),
                motion='blue module moves; object remains in its initial lying pose',
                task_floor_static_only=True, dock_direction_independent=True)


def inputs(name):
    return [ROOT/'objects'/name/'poses.json']
