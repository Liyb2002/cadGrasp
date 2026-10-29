"""Copy a measured request with the station search profile; optional explicit pad radius and upright yaw target."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from bunny_movement import HERE, load_request, write
from robot import points


def yaw_target(source, robot, degrees):
    pan = robot.base @ robot.origins[0]
    axis = pan[:3, :3] @ robot.axes[0]
    if abs(abs(axis[2]) - 1.) > 1e-7:
        raise ValueError('An upright yaw target requires a vertical pan axis')
    if axis[2] < 0:
        axis = -axis
    # URDF rpy uses rounded pi, which leaves a microradian tilt. A table-preserving yaw is exactly world-vertical.
    axis = np.array([0., 0., 1.])
    rotation = Rotation.from_rotvec(axis * np.radians(degrees)).as_matrix()
    turn = np.eye(4)
    turn[:3, :3] = rotation
    turn[:3, 3] = pan[:3, 3] - rotation @ pan[:3, 3]
    return turn @ source


def prepare(source_path, output, profile, pad_radius_mm=None, target_yaw_deg=None):
    source_path, output = Path(source_path).resolve(), Path(output).resolve()
    if source_path == output:
        raise ValueError('Write a new request; the measured input is kept intact')
    raw = json.loads(source_path.read_text())
    tuning = json.loads(Path(profile).read_text())
    raw.setdefault('planner', {}).update(tuning['planner'])
    if pad_radius_mm is not None:
        if not np.isfinite(pad_radius_mm) or pad_radius_mm <= 0:
            raise ValueError('Pad radius must be a positive measured value')
        raw['gripper']['contact_patch_radius_m'] = pad_radius_mm * .001
    if target_yaw_deg is not None:
        resolved, robot, body, _ = load_request(source_path)
        source = np.asarray(resolved['source']['world_from_object'])
        if abs(points(body.vertices, source)[:, 2].min() - resolved['table_top_z_m']) > .001:
            raise ValueError('Yaw-only target helper requires a source resting on the table; fixture relocation is not planned')
        target = yaw_target(source, robot, target_yaw_deg)
        raw['target'] = dict(world_from_object=target.tolist(), exit_direction_world=[0, 0, 1],
                             travel_m=resolved['source'].get('travel_m', .04))
    specs = [(raw['object'], 'mesh'), (raw['robot'], 'urdf')]
    specs += [(raw[n], 'package') for n in ('source', 'target') if 'package' in raw[n]]
    specs += [(o, 'mesh') for o in raw.get('obstacles', []) if 'mesh' in o]
    for spec, key in specs:
        spec[key] = os.path.relpath((source_path.parent / spec[key]).resolve(), output.parent)
    raw['station_update'] = dict(profile=str(Path(profile).name),
        target_yaw_deg=target_yaw_deg, explicit_pad_radius_mm=pad_radius_mm,
        source_pose_changed=False, robot_calibration_changed=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, raw)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--profile', type=Path, default=HERE / 'station_search_profile.json')
    parser.add_argument('--pad-radius-mm', type=float, help='Only set when confirmed by physical pad dimensions')
    parser.add_argument('--target-yaw-deg', type=float, help='Explicitly replace target by source rotated CCW about the actual vertical pan axis')
    args = parser.parse_args()
    print(prepare(args.request, args.output, args.profile, args.pad_radius_mm, args.target_yaw_deg))
