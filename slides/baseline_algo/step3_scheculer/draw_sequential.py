"""Draw the two active triples, keeping the shared head the same color."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from step3_scheculer.pair_tasks import read_task
from step3_scheculer import contacts as I


def draw(schedule, folder):
    schedule, folder = Path(schedule), Path(folder)
    report = json.loads(schedule.read_text())
    result = report['result']
    palette = ['#387cad', '#ca8841', '#936dac', '#b54b55', '#29999e']
    colors = dict(zip(result['selected_ids'], palette))
    labels = {key: f'H{i+1}' for i, key in enumerate(result['selected_ids'])}
    fig = plt.figure(figsize=(12, 10), facecolor='white')
    fig.suptitle(f'{report["object"]}: {report["poses"][0]} (3) -> share 1 -> {report["poses"][1]} (+2)\n'
                 f'{report["successful_particles"]}/{report["particles"]} contact sets passed; full fixture pending', fontsize=15)
    for k, pose in enumerate(report['poses']):
        problem = read_task(report['object'], pose, report['poses'])
        mesh = problem.domain.mesh
        contacts = I.read_contacts(schedule.parent/f'final_contacts_{pose}.npz')
        ax = fig.add_subplot(2, 2, k+1, projection='3d')
        ax.add_collection3d(Poly3DCollection(mesh.triangles, facecolors='#c6c9c9', edgecolors='none', alpha=.3))
        ax.add_collection3d(Poly3DCollection(mesh.triangles[problem.domain.work_ids], facecolors='#789375', edgecolors='none'))
        legend = []
        for contact in contacts:
            key = contact['candidate_id']
            ax.add_collection3d(Poly3DCollection(contact['triangles_m'], facecolors=colors[key], edgecolors='none'))
            label = f'{labels[key]}: {key}'+(' (shared)' if key == result['shared_head']['selected_id'] else '')
            legend.append(Patch(color=colors[key], label=label))
        if legend:
            ax.legend(handles=legend, loc='upper left', fontsize=8, frameon=False)
        center = mesh.bounds.mean(axis=0)
        radius = mesh.extents.max()*.6
        ax.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius),
               zlim=(center[2]-radius, center[2]+radius))
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=19, azim=-55)
        ax.set_axis_off()
        ax.set_title(f'{pose}: {result["covered_counts"][k]:,} / {result["sample_counts"][k]:,} covered')
        ax = fig.add_subplot(2, 2, k+3)
        with np.load(folder/f'floor_contact_{pose}.npz') as data:
            cloud = data['floor_demands_xy_m']*1000
            pivot = data['original_pivot_m'][:2]*1000
        ax.scatter(*cloud[::max(1, len(cloud)//2500)].T, s=3, color='#cb8c43', alpha=.35)
        ax.scatter(*pivot, marker='x', color='black', s=45)
        ax.set(xlabel='x (mm)', ylabel='y (mm)', title=f'{pose}: floor demand')
        ax.set_aspect('equal')
    fig.text(.5, .02, 'Same color = same head. Only active contacts shown; feet and full-body insertion await Step5.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .045, 1, .92))
    path = folder/'sequential_overview.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
