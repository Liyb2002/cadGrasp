"""Show actual active contacts and original floor demands for any task count."""
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from step3_scheculer import contacts as I


def draw(search, report, folder):
    result, count = report['result'], len(search.poses)
    colors = {head: plt.get_cmap('tab20')(i % 20) for i, head in enumerate(result['selected_ids'])}
    uses = Counter(head for group in result['active_ids_by_pose'] for head in group)
    columns = min(count, 5)
    blocks = (count+columns-1)//columns
    fig = plt.figure(figsize=(5*columns, 8*blocks+1), facecolor='white')
    fig.suptitle(f'{search.name}: simultaneous uncovered-weighted contact search\n'
                 f'{report["successful_particles"]}/{report["particles"]} particles passed; '
                 f'{result["heads"]} distinct heads, {sum(n > 1 for n in uses.values())} shared', fontsize=14)
    for k, problem in enumerate(search.problems):
        mesh = problem.domain.mesh
        group = I.read_contacts(search.out/f'final_contacts_{problem.pose}.npz')
        position = (k//columns)*2*columns+k % columns+1
        ax = fig.add_subplot(2*blocks, columns, position, projection='3d')
        ax.add_collection3d(Poly3DCollection(mesh.triangles, facecolors='#c6c9c9', edgecolors='none', alpha=.3))
        ax.add_collection3d(Poly3DCollection(mesh.triangles[problem.domain.work_ids], facecolors='#789375', edgecolors='none'))
        for contact in group:
            ax.add_collection3d(Poly3DCollection(contact['triangles_m'], facecolors=colors[contact['candidate_id']], edgecolors='none'))
        center, radius = mesh.bounds.mean(axis=0), mesh.extents.max()*.6
        ax.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius),
               zlim=(center[2]-radius, center[2]+radius))
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=19, azim=-55)
        ax.set_axis_off()
        ax.set_title(f'{problem.pose}: {len(group)} heads\n'
                     f'{result["covered_counts"][k]:,}/{result["sample_counts"][k]:,} loads')
        ax = fig.add_subplot(2*blocks, columns, position+columns)
        with np.load(folder/f'floor_contact_{problem.pose}.npz') as data:
            cloud = data['floor_demands_xy_m']*1000
            pivot = data['original_pivot_m'][:2]*1000
        ax.scatter(*cloud[::max(1, len(cloud)//2500)].T, s=3, color='#cb8c43', alpha=.35)
        ax.scatter(*pivot, marker='x', color='black', s=45)
        ax.set(xlabel='x (mm)', ylabel='y (mm)', title=f'{problem.pose}: original floor demand')
        ax.set_aspect('equal')
    fig.text(.5, .02, 'Active contacts only. Full fixture geometry, feet and inactive material await Step5.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .045, 1, .92))
    path = folder/'joint_overview.png'
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
