"""Render each object's ten saved target poses into one overview.png.

Uses exact target transforms and saved work-face masks, not video screenshots.
The camera and scale are fixed across all ten panels of an object. Robot and
gripper are omitted for clarity; these targets are robot-held, not self-stable.
"""
import sys
sys.dont_write_bytecode = True

import argparse
import gc
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
BODY = np.array([.32, .43, .53])
WORK = np.array([1., .59, .20])


def render(name):
    folder = ROOT / 'objects' / name
    record = json.loads((folder / 'poses.json').read_text())
    poses = record['poses']
    assert [p['pose_id'] for p in poses] == [f'pose_{i}' for i in range(1, 11)]
    raw = trimesh.load(folder / 'mesh.stl', force='mesh')
    transforms = np.asarray([p['T_world_mesh'] for p in poses])
    all_vertices = np.concatenate([
        trimesh.transform_points(raw.vertices, T) for T in transforms])
    lower, upper = all_vertices.min(axis=0), all_vertices.max(axis=0)
    span = float((upper - lower).max()) * 1.12
    center = (lower + upper) / 2
    xlim = center[0] + np.array([-.5, .5]) * span
    ylim = center[1] + np.array([-.5, .5]) * span
    zlim = np.array([-.06, .94]) * span
    meshes = {0: raw}
    light = np.array([-.4, -.5, 1.])
    light /= np.linalg.norm(light)

    fig = plt.figure(figsize=(20, 9.2), facecolor='white')
    fig.subplots_adjust(left=.015, right=.985, bottom=.015, top=.90,
                        wspace=.015, hspace=.09)
    fig.text(.025, .95, name, fontsize=23, weight='bold', color='#243444')
    fig.text(.025, .919, '10 target poses', fontsize=11, color='#687785')
    fig.legend(handles=[Patch(facecolor=WORK, label='Working area')],
               loc='upper right', bbox_to_anchor=(.98, .967),
               frameon=False, fontsize=11)
    try:
        for index, (pose, T) in enumerate(zip(poses, transforms)):
            task = folder / 'tasks' / pose['pose_id']
            setup = json.loads((task / 'setup.json').read_text())
            rounds = setup['uniform_subdivision_rounds']
            for level in range(1, rounds + 1):
                if level not in meshes:
                    meshes[level] = meshes[level - 1].subdivide()
            mesh = meshes[rounds]
            with np.load(task / 'setup.npz') as data:
                mask = data['work_faces'].copy()
                np.testing.assert_allclose(data['T_world_mesh'], T, atol=1e-12)
            assert mask.dtype == bool and mask.shape == (len(mesh.faces),)
            vertices = trimesh.transform_points(mesh.vertices, T)
            assert abs(vertices[:, 2].min()) < 1e-8
            normals = mesh.face_normals @ T[:3, :3].T
            shade = .66 + .34 * np.maximum(0., normals @ light)
            colors = np.tile(BODY, (len(mesh.faces), 1))
            colors[mask] = WORK
            colors *= shade[:, None]

            ax = fig.add_subplot(2, 5, index + 1, projection='3d',
                                 computed_zorder=False)
            ax.set_proj_type('ortho')
            ax.view_init(elev=28, azim=125)
            ax.set(xlim=xlim, ylim=ylim, zlim=zlim)
            ax.set_box_aspect((1, 1, 1), zoom=1.25)
            ax.set_axis_off()
            ax.text2D(.06, .94, f'POSE {index + 1:02d}', transform=ax.transAxes,
                      fontsize=11, weight='bold', color='#334858')
            floor_z = -.001 * span
            floor = np.array([[xlim[0], ylim[0], floor_z],
                              [xlim[1], ylim[0], floor_z],
                              [xlim[1], ylim[1], floor_z],
                              [xlim[0], ylim[1], floor_z]])
            ax.add_collection3d(Poly3DCollection(
                [floor], facecolor='#eef1f3', edgecolor='#d9e0e5', linewidth=.6,
                zorder=1))
            ax.add_collection3d(Poly3DCollection(
                vertices[mesh.faces], facecolors=colors, edgecolors='none',
                linewidth=0, antialiased=False, zsort='average', zorder=2))
        destination = folder / 'overview.png'
        fig.savefig(destination, dpi=160, facecolor='white',
                    metadata={'Description':
                        'Ten nominal robot-held target poses, fixed view and scale; '
                        'orange faces are saved working areas. Robot omitted.'})
    finally:
        plt.close(fig)
    gc.collect()
    print(destination, flush=True)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('objects', nargs='*')
    args = parser.parse_args()
    names = args.objects or json.loads((ROOT / 'objects/cases.json').read_text())['active_objects']
    for name in names:
        render(name)
