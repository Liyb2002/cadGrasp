"""Random work stations for whole_task_2: robots_config/random/<NN>_<model>.json.

A station is a robot model (models/<model>.json: KUKA LBR Med 14 or SO101), a nominal needle direction
drawn uniformly on the northern hemisphere (the tool comes from above: the surface-to-tool direction has
z > 0), a tilt-cone half-angle from TILTS and the work arm's bbx (the edge of the surface work cube, from
EDGES; null: unbounded).  The base stands at the origin facing the work window, whose centre is the
comfortable work point of robots_config.place: the arm has the most room around the whole job there (first
guess: the model's wrist home plus the tool, TOOL, along the needle).  Every file is a full schema-1 config
(constraints.load_config); whole_task_2 Step1 reads it for the run <object>+<NN>_<model>.

A draw of `count` stations holds both models and exactly one unbounded bbx.

  python -m robots_config.draw --count 5 --seed 20261022
"""
import argparse
import json
from pathlib import Path

import numpy as np

from robots_config.constraints import load_config
from robots_config.place import placed

HERE = Path(__file__).resolve().parent
MODELS = ('kuka_lbr_med14_r820', 'so101')
TILTS = (15, 25, 35)            # deg, the needle's tilt-cone half-angle
EDGES = (.075, .15)             # m, the bounded work-arm bbx edges; one station per draw is unbounded
TOOL = .17                      # m, needle 100 mm + its root 70 mm from the flange or the jaws


def station(index, model, up, tilt, edge, seed):
    m = json.loads((HERE/'models'/f'{model}.json').read_text()); s = m['station']
    a = -np.asarray(up, float)                                     # tool -> surface
    u = np.array([a[0], a[1], 0.]); u /= np.linalg.norm(u)        # the work window, seen from the base
    z = np.array([0., 0., 1.]); R = np.column_stack([u, np.cross(z, u), z])
    centre = s['wrist_home_forward_m']*u+s['wrist_home_height_m']*z+TOOL*a
    name = f"{index:02d}_{m['short']}"
    edge_note = 'unbounded: no bbx' if edge is None else f'{edge*1000:.0f} mm cube'
    return dict(schema_version=1, id=name, name=f"{m['name']}, random station {index:02d}", robot_model=model,
        researched_on='2026-10-22', robot_specification=m['robot_specification'],
        task_constraint=dict(frame='world_z_up', direction_semantics='tool_to_surface', working_direction_world=a.tolist(),
            cone_half_angle_deg=tilt, surface_incidence_half_angle_deg=10,
            direction_label_en=f'drawn on the northern hemisphere: needle {np.degrees(np.arcsin(-a[2])):.1f} deg below horizontal',
            provenance='user_editable_estimate_not_ik_certified',
            surface_workspace=dict(shape='cube', edge_m=edge, applies_to='surface_process_point', label=edge_note,
                                   note='The work arm bbx: surface points it may process; whole_task_2 Step1 reads it, it does not compute it.'),
            world_position_constraint=dict(type='planar_disk', applies_to='surface_points_world', center_world_m=centre.tolist(),
                outward_normal_world=(-a).tolist(), radius_m=.1, normal_tolerance_m=.005,
                base_to_center_distance_m=float(np.linalg.norm(centre)), provenance='estimated_work_window_not_ik_certified',
                rule='wrist home of the model plus the tool along the needle direction; Step1 slides the work point forward from here'),
            tcp_to_surface_distance_m=dict(min=0, nominal=0, max=0, measurement='along_tool_axis_from_process_tcp_to_surface',
                                           provenance='user_editable_process_estimate')),
        feasibility=dict(direction_only=False, ik_verified=False,
            base_pose_world=dict(position_m=[0., 0., 0.], rotation_matrix=R.tolist(), provenance='base +X toward the work window'),
            tcp_transform=m['tcp_transform'], table_half_size_m=s['table_half_size_m'], work_point_search=s['work_point_search'],
            arm_module=m['arm_module']),
        draw=dict(seed=seed, index=index, hemisphere='north: the surface-to-tool direction has z > 0, uniform on the hemisphere',
                  models=list(MODELS), tilts_deg=list(TILTS), edges_m=[*EDGES, None]),
        sources=m['sources'])


def draw(count=5, seed=20261022, folder=HERE/'random'):
    rng = np.random.default_rng(seed)
    while True:
        models = rng.choice(len(MODELS), count)
        if len(set(models)) == len(MODELS): break
    unbounded = int(rng.integers(count)); out = []
    for k in range(count):
        z = rng.uniform(0, 1); phi = rng.uniform(0, 2*np.pi)    # uniform on the hemisphere
        up = np.array([np.sqrt(1-z*z)*np.cos(phi), np.sqrt(1-z*z)*np.sin(phi), z])
        edge = None if k == unbounded else float(EDGES[rng.integers(len(EDGES))])
        config = placed(station(k+1, MODELS[models[k]], up, int(TILTS[rng.integers(len(TILTS))]), edge, seed))
        path = folder/f"{config['id']}.json"; folder.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, indent=2)+'\n'); load_config(path); out.append(path)
        t = config['task_constraint']
        print(config['id'], 'needle', [round(x, 3) for x in t['working_direction_world']], 'tilt', t['cone_half_angle_deg'],
              'bbx', t['surface_workspace']['label'])
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--count', type=int, default=5); p.add_argument('--seed', type=int, default=20261022)
    a = p.parse_args(); draw(a.count, a.seed)
