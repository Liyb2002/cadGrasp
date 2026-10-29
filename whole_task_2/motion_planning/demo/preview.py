"""Draw the saved transfer and joint curves; this does not change or replan the motion."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from bunny_movement import HERE, load_request
from robot import points


def draw(request, output):
    req, robot, body, _ = load_request(request)
    with np.load(Path(output) / 'trajectory.npz') as z:
        times, q = z['times_s'], z['joints_model_deg']
        poses, hand = z['world_from_object'], z['world_from_gripper']
    program = json.loads((Path(output) / 'program.json').read_text())
    contact_points = np.asarray(program['solution']['grasp']['contacts_object_m'])
    fig = plt.figure(figsize=(16, 7))
    ax = fig.add_subplot(121, projection='3d')
    all_points = [robot.base[:3, 3][None]]
    for pose, color, label in [(poses[0], '#73a3c9', 'Source B'), (poses[-1], '#db7795', 'Target B')]:
        v = points(body.vertices, pose)
        all_points.append(v)
        triangles = v[body.faces]
        light = np.array([-.4, -.3, .8])
        light /= np.linalg.norm(light)
        shade = .55 + .45 * np.maximum(0., (body.face_normals @ pose[:3, :3].T) @ light)
        colors = np.asarray(to_rgb(color))[None] * shade[:, None]
        ax.add_collection3d(Poly3DCollection(triangles, facecolors=colors, edgecolors='none', alpha=.45))
        ax.scatter(*points(body.centroid, pose), color=color, label=label)
        ax.scatter(*points(contact_points, pose).T, color='#164e36', s=16, marker='x')
    centres = np.array([points(body.centroid, p) for p in poses])
    ax.plot(*centres.T, color='#cb365a', linewidth=2, label='B centre')
    ax.plot(*hand[:, :3, 3].T, color='#376d98', linewidth=1, alpha=.7, label='Gripper')
    _, frames = robot.frames(np.radians(q[0, :5]), np.radians(q[0, 5]))
    for part in robot.parts:
        v = points(part['table_vertices'], frames[part['link']])
        ax.scatter(*v[::max(1, len(v) // 80)].T, s=.5, color='#666666', alpha=.45)
    all_points.extend([centres, hand[:, :3, 3]])
    bounds = np.vstack(all_points)
    low, high = bounds.min(0) - .04, bounds.max(0) + .04
    x, y = np.meshgrid([low[0], high[0]], [low[1], high[1]])
    ax.plot_surface(x, y, np.full_like(x, req['table_top_z_m']), color='#bdb7ac', alpha=.15)
    ax.scatter(*robot.base[:3, 3], marker='s', color='black', label='Carrier base')
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(min(low[2], req['table_top_z_m']), high[2]),
           xlabel='world x (m)', ylabel='world y (m)', zlabel='world z (m)', title='Model-verified bunny transfer')
    ax.set_box_aspect(high - low)
    ax.legend(loc='upper left', fontsize=8)
    ax.view_init(elev=28, azim=-60)
    joints = fig.add_subplot(122)
    names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
    for i, name in enumerate(names):
        joints.plot(times, q[:, i], label=name, linewidth=1.2)
    joints.set(xlabel='time (s)', ylabel='URDF model angle (deg)', title='Five arm joints + native rotating jaw')
    joints.yaxis.set_label_position('right')
    joints.yaxis.tick_right()
    joints.grid(alpha=.2)
    joints.legend(fontsize=8)
    fig.tight_layout()
    path = Path(output) / 'preview.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, default=HERE / 'example_request.json')
    parser.add_argument('--output', type=Path, default=HERE / 'example_output')
    args = parser.parse_args()
    draw(args.request, args.output)
