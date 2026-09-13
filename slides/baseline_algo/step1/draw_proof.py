"""Draw three separate input loads and their paired demands for each object.

Static Matplotlib/Agg figures in output/<object>/<pose>/step_1_needs, with no simulator.
"""
from __future__ import annotations
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from needs import COORD, ContinuousNeeds, OBJECTS, OUTPUTS, sha256, save_json
from step1.cases import pose_name

PAPER = '#ffffff'
INK = '#24323b'
BLUE = '#2269b5'
ORANGE = '#cd651d'
GREY = '#9b9e9e'


def projected_arrow(ax, start, end, color, label=None, width=2.8, dashed=False):
    """Project world coordinates using the final camera; preserve arrow sign."""
    if np.linalg.norm(np.asarray(end)-start) == 0:
        return
    a = np.array(proj3d.proj_transform(*start, ax.get_proj()))[:2]
    b = np.array(proj3d.proj_transform(*end, ax.get_proj()))[:2]
    ax.annotate('', xy=b, xytext=a, annotation_clip=False,
                arrowprops=dict(arrowstyle='-|>', color=color, lw=width,
                                linestyle='--' if dashed else '-', mutation_scale=16))
    if label:
        ax.annotate(label, a, xytext=(5, 5), textcoords='offset points',
                    color=color, fontsize=11, weight='bold', annotation_clip=False)


def camera(domain, case):
    i = case['work_face_index']
    outward = -domain.normals[i]
    q = np.asarray(case['pt_m'])
    d = np.asarray(case['d'])
    # Oblique, visible surface view so the arrow does not point into the screen.
    for a, b in [(1.0, .6), (-1.0, .6), (.7, -.7), (-.7, -.7), (.3, .2), (0., 0.)]:
        view = outward + a*domain.e1[i] + b*domain.e2[i]
        view /= np.linalg.norm(view)
        blocked = domain.mesh.ray.intersects_any(
            (q + domain.ray_offset*outward)[None], view[None])[0]
        if not blocked:
            return view
    return -d  # The actual escape ray was verified by the needs exporter.


def object_panel(ax, domain, case):
    mesh = domain.mesh
    vertices = mesh.vertices*1000
    triangles = vertices[mesh.faces]
    colors = np.tile([.73, .76, .78, 1.], (len(mesh.faces), 1))
    colors[domain.work_ids] = [.51, .69, .46, 1.]
    view = camera(domain, case)
    # A light from the camera makes folds legible with a single opaque surface.
    shade = .68 + .32*np.maximum(0, mesh.face_normals@view)
    colors[:, :3] *= shade[:, None]
    ax.add_collection3d(Poly3DCollection(triangles, facecolors=colors, edgecolors='none',
                                        linewidths=0, rasterized=True))
    q = 1000*np.asarray(case['pt_m'])
    com = 1000*domain.com
    d = np.asarray(case['d'])
    extent = float(np.ptp(vertices, axis=0).max())
    magnitude = float(np.linalg.norm(case['force_push_mg']))
    tail = q - .26*extent*(magnitude/domain.k)*d
    all_points = np.vstack([vertices, tail])
    low, high = all_points.min(axis=0), all_points.max(axis=0)
    center = (low+high)/2
    span = 1.04*float((high-low).max())
    for setlim, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        setlim(c-span/2, c+span/2)
    ax.set_box_aspect((1, 1, 1), zoom=1.15)
    ax.set_proj_type('ortho')
    COORD.matplotlib_view(ax, elev=np.rad2deg(np.arcsin(view[1])), azim=np.rad2deg(np.arctan2(view[2], view[0])))
    ax.set_axis_off()
    projected_arrow(ax, tail, q, BLUE, r'$F_{push}$')
    # COM and the moment arm are an x-ray reference, not extra applied forces.
    cq = np.array(proj3d.proj_transform(*np.stack([com, q]).T, ax.get_proj()))[:2]
    ax.add_artist(Line2D(cq[0], cq[1], transform=ax.transData, color=INK, ls=':', lw=1.3, zorder=20))
    ax.annotate('c', cq[:, 0], xytext=(-12, -12), textcoords='offset points',
                fontsize=12, color=INK, weight='bold')
    ax.annotate(f'pt{case["id"]}', cq[:, 1], xytext=(6, -12), textcoords='offset points',
                fontsize=11, color=BLUE, weight='bold')
    ax.add_artist(Line2D([cq[0, 0]], [cq[1, 0]], transform=ax.transData,
                        marker='x', color=INK, markersize=7, markeredgewidth=2, zorder=20))
    ax.add_artist(Line2D([cq[0, 1]], [cq[1, 1]], transform=ax.transData,
                        marker='o', color=BLUE, markersize=5, zorder=20))
    # World-coordinate triad; these axes remain tied to the object camera.
    origin = np.array([-.12, -.075])
    for j, label in enumerate('xyz'):
        end3 = com + np.eye(3)[j]*.14*extent
        end2 = np.array(proj3d.proj_transform(*end3, ax.get_proj()))[:2]
        displacement = end2-cq[:, 0]
        ax.annotate('', origin+displacement, origin, annotation_clip=False,
                    arrowprops=dict(arrowstyle='->', color='#727978', lw=1))
        ax.annotate(label, origin+displacement, color='#727978', fontsize=9,
                    annotation_clip=False)
    return {'view_toward_camera': view.tolist(), 'arrow_tail_m': (tail/1000).tolist(),
            'arrow_head_m': (q/1000).tolist(), 'arrow_draw_length_m': float(np.linalg.norm(q-tail)/1000),
            'force_arrow_scale': 'display length proportional to magnitude; 0.26 object extents at the upper bound'}


def vector_panel(ax, primary, original, limit, color, title):
    ax.set_proj_type('ortho')
    COORD.matplotlib_view(ax, elev=23, azim=-58)
    ax.set_box_aspect((1, 1, 1), zoom=1.12)
    for setlim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        setlim(-limit, limit)
    ax.set_axis_off()
    for j, label in enumerate('xyz'):
        axis = np.eye(3)[j]*limit*.8
        ax.plot(*np.stack([-axis, axis]).T, color='#c9cecd', lw=.9)
        ax.text(*axis, label, fontsize=11, color='#727978')
    projected_arrow(ax, np.zeros(3), original, GREY, width=2.0, dashed=True)
    projected_arrow(ax, np.zeros(3), primary, color, width=3.2)
    ax.text2D(.5, .96, title, transform=ax.transAxes, ha='center', va='top',
              color=INK, fontsize=14)


def triple(vector, digits):
    return '(' + ', '.join(f'{x:+.{digits}f}' for x in vector) + ')'


def draw(name):
    folder = OUTPUTS/name/pose_name()/'step_1_needs'
    path = folder/'needs.json'
    domain = ContinuousNeeds.read(path)
    examples = json.loads((folder/'examples.json').read_text())
    if examples['needs_sha256'] != sha256(path):
        raise ValueError('Proof examples refer to a different demand export')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': INK,
                         'figure.facecolor': PAPER, 'axes.facecolor': PAPER})
    fig = plt.figure(figsize=(16, 15), dpi=145)
    fig.text(.035, .966, f'{name}  /  Step 1: from a surface push to its 6D need',
             fontsize=23, weight='bold')
    fig.text(.035, .936, 'Three separate loads  ·  Green = work surface  ·  c = center of mass  ·  '
             f'0 ≤ |Fpush| ≤ 0.5 mg  ·  local inward cone = {domain.data["load"]["cone_half_deg"]:g}°', fontsize=12)
    fig.text(.225, .902, 'INPUT: position + force direction', fontsize=14, ha='center', weight='bold')
    fig.text(.604, .902, 'NEED: force', fontsize=14, ha='center', weight='bold', color=BLUE)
    fig.text(.848, .902, 'NEED: moment about c', fontsize=14, ha='center', weight='bold', color=ORANGE)
    torque_limit = 1.3*max(np.linalg.norm(np.asarray(c['need_wrench'])[3:])*1000
                          for c in examples['cases'])
    views = []
    for row, case in enumerate(examples['cases']):
        bottom = .65-row*.275
        fig.text(.035, bottom+.192, f'{case["id"]:02d}', fontsize=22, color=BLUE, weight='bold')
        obj = fig.add_axes([.048, bottom-.005, .386, .235], projection='3d')
        views.append(object_panel(obj, domain, case))
        f = np.asarray(case['need_wrench'])[:3]
        tau = np.asarray(case['need_wrench'])[3:]*1000
        force_ax = fig.add_axes([.483, bottom+.005, .237, .215], projection='3d')
        torque_ax = fig.add_axes([.727, bottom+.005, .237, .215], projection='3d')
        vector_panel(force_ax, f, -f, 1.6, BLUE, r'$F_{need}=mg\,\hat{z}-F_{push}$')
        vector_panel(torque_ax, tau, -tau, torque_limit, ORANGE,
                     r'$\tau_{need}=-(\mathrm{pt}-c)\times F_{push}$')
        fig.text(.078, bottom-.01, 'pt = '+triple(np.asarray(case['pt_m'])*1000, 2)+' mm', fontsize=10.5)
        fig.text(.078, bottom-.03, 'Fpush = '+triple(case['force_push_mg'], 3)+' mg', fontsize=10.5, color=BLUE)
        fig.text(.725, bottom-.012, 'One paired demand:', ha='center', fontsize=11)
        fig.text(.603, bottom-.033, triple(f, 3)+' mg', ha='center', fontsize=11, color=BLUE)
        fig.text(.848, bottom-.033, triple(tau, 2)+' mg·mm', ha='center', fontsize=11, color=ORANGE)
        if row < 2:
            fig.add_artist(Line2D([.035, .965], [bottom-.045]*2, transform=fig.transFigure,
                                  color='#dddeda', linewidth=1))
    fig.text(.035, .036, 'Flip the complete external wrench:  '
             '[Fpush + (0, −mg, 0) ; (pt − c) × Fpush]  →  [Fneed ; τneed]', fontsize=13)
    fig.text(.035, .019, 'Gray dashed = external resultant; colored = its negative.  '
             'World axes; literal vector directions.  Figures use mg·mm; JSON uses mg·m.', fontsize=10.5,
             color='#687270')
    image = folder/'proof.png'
    fig.savefig(image, facecolor=PAPER)
    plt.close(fig)
    save_json(folder/'proof.json', {
        'object': name, 'needs_sha256': sha256(path),
        'examples_sha256': sha256(folder/'examples.json'),
        'draw_script_sha256': sha256(__file__), 'views': views,
        'cases': examples['cases'], 'figure_moment_multiplier_from_json': 1000})
    print(f'{name}: {image}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('objects', nargs='*')
    names = parser.parse_args().objects or OBJECTS
    if any(name not in OBJECTS for name in names):
        parser.error('objects must be A1-f, B or C5')
    for name in names:
        draw(name)
