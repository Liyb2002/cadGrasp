"""Show the actual base material against Step4 floor demands."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from step1.needs import OUTPUTS, OBJECTS
from step1.cases import pose_name
from step5_base import base as C
from step5_connect_support import whole_assembly as A


def draw(name):
    domain, _, _, floor, _, _, _ = A.read_inputs(name)
    report, _ = C.read(name)
    if report.get('schema') == 'stationary_base_removable_module_v1':
        from step6_connect_support.modular_workflow import preview
        return preview(name)
    out = OUTPUTS/name/pose_name()/C.STAGE
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    points = floor['floor_demands_xy_m']
    ax.scatter(points[:, 0]*1000, points[:, 1]*1000, s=1, alpha=.2,
               color='#547b90', label='Step4 demands')
    required = np.asarray(report['required_hull_xy_m'])*1000
    required = np.vstack([required, required[0]])
    ax.plot(*required.T, color='#d45c47', lw=1.5, label='Continuous demand hull')
    if report['passed']:
        polygons = np.asarray(report['base']['pads_xy_m'])*1000
        ax.add_collection(PolyCollection(polygons, facecolor='#469488', edgecolor='none', alpha=.8))
        direction = np.asarray(report['withdrawal_direction'])[:2]
        center = np.asarray(report['base']['outer_xy_m']).mean(axis=0)*1000
        size = float(domain.mesh.extents.max())*150
        ax.arrow(*center, *(direction*size), width=.6, head_width=3, color='#282c33')
        ax.text(*center, '  withdrawal', fontsize=9)
        ax.set_title(f'{name}/{pose_name()} - Step5 base\n'
                     f'Footprint {report["base"]["footprint_area_m2"]*1e4:.1f} cm²; '
                     f'direction {report["direction_id"]}')
    else:
        ax.set_title(report['status'])
    ax.scatter(*np.asarray(floor['original_pivot_m'])[:2]*1000, color='black', s=25, label='Object floor contact')
    ax.set(xlabel='World X (mm)', ylabel='World Y (mm)', aspect='equal')
    ax.autoscale_view(); ax.legend(loc='best'); fig.tight_layout()
    fig.savefig(out/'base.png', dpi=180); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('objects', nargs='*')
    for name in parser.parse_args().objects or OBJECTS:
        draw(name)
