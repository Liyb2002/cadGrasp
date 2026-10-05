"""Show the selected shared heads and the two independent floor-demand clouds."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer.pair_tasks import read_task, pair_folder, fixed_area_folder, completion_folder
from step3_scheculer import contacts as I
from step3_scheculer.sample_acceptance import read_report


def draw(name, poses, max_heads=None, fixed_area=False, terminal_expansion=False):
    folder = completion_folder if terminal_expansion else fixed_area_folder if fixed_area else pair_folder
    step3 = folder(name, poses, 'step3_scheculer')
    step4 = folder(name, poses, 'step4_floor_contact')
    if max_heads is not None:
        step3 = step3/f'heads_{max_heads}'
        step4 = step4/f'heads_{max_heads}'
    report = read_report(step3)
    poses = report['poses']
    fig = plt.figure(figsize=(12, 10), facecolor='white')
    colors = ['#387cad', '#ca8841', '#936dac', '#b54b55', '#29999e']
    result = report['result']
    area_label = ' | fixed 1% heads' if report.get('area_optimization_performed') is False else ''
    if 'terminal_expansion_policy' in report:
        area_label = ' | fixed 1% selection + terminal expansion'
    fig.suptitle(f'{name}: {poses[0]} + {poses[1]} | shared object-attached heads\n'
                 f'{report["successful_particles"]}/{report["particles"]} passed | 32,768 fixed loads per pose{area_label}', fontsize=15)
    for k, pose in enumerate(poses):
        problem = read_task(name, pose, poses)
        mesh = problem.domain.mesh
        contacts = I.read_contacts(step3/report.get('selected_contact_files', {}).get(pose, f'final_contacts_{pose}.npz'))
        ax = fig.add_subplot(2, 2, k+1, projection='3d')
        ax.add_collection3d(Poly3DCollection(mesh.triangles, facecolors='#c6c9c9', edgecolors='none', alpha=.3))
        ax.add_collection3d(Poly3DCollection(mesh.triangles[problem.domain.work_ids], facecolors='#789375', edgecolors='none'))
        for i, contact in enumerate(contacts):
            ax.add_collection3d(Poly3DCollection(contact['triangles_m'], facecolors=colors[i % len(colors)], edgecolors='none'))
            ax.text(*contact['center_m'], contact['candidate_id'], fontsize=8)
        center = mesh.bounds.mean(axis=0)
        radius = mesh.extents.max()*.6
        ax.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius),
               zlim=(center[2]-radius, center[2]+radius))
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=19, azim=-55)
        ax.set_axis_off()
        ax.set_title(f'{pose}: {100*result["covered_fractions"][k]:.3f}% search-load coverage')
        ax = fig.add_subplot(2, 2, k+3)
        with np.load(step4/f'floor_contact_{pose}.npz') as data:
            cloud = data['floor_demands_xy_m']*1000
            pivot = data['original_pivot_m'][:2]*1000
        ax.scatter(*cloud[::max(1, len(cloud)//2500)].T, s=3, color='#cb8c43', alpha=.35, label='Sampled demand')
        ax.scatter(*pivot, marker='x', color='black', s=45, label='Original object-floor contact')
        ax.set_aspect('equal')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title(f'{pose}: floor demand, no base constructed')
        ax.legend(fontsize=8)
    fig.text(.5, .02, 'Green: working area. Colored patches: selected heads. No connected support or fixture bearing certificate.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .045, 1, .92))
    path = step4/'pair_overview.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(path, flush=True)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object')
    parser.add_argument('poses', nargs=2)
    parser.add_argument('--max-heads', type=int)
    parser.add_argument('--fixed-area', action='store_true', help='Read the current fixed 1%%-area run')
    parser.add_argument('--terminal-expansion', action='store_true', help='Read fixed selection with terminal expansion')
    args = parser.parse_args()
    draw(args.object, args.poses, args.max_heads, args.fixed_area, args.terminal_expansion)
