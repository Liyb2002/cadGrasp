"""Show common-height clearances and the binding cross-installation floor."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from subdivide_task.operations.state import transform
from subdivide_task.step1.common import ROOT, sha256, write_json
from subdivide_task.step3.run_membranes import load_model
from subdivide_task.step4.ground_oracle import pressure_rows
from subdivide_task.step4.membrane_search import read_group
from .height_reference import lower_layout, lower_model


def render(folder, output=None):
    folder = Path(folder)
    report = json.loads((folder / 'report.json').read_text())
    plan = report['height_reference']
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(),
                         'axes.unicode_minus': False, 'font.size': 11})
    fig = plt.figure(figsize=(15, 9), dpi=180, facecolor='white')
    fig.suptitle('Step5｜地面固定，只统一降低一次针头', fontsize=22, color='#29475c', y=.965)
    old = plan['original_head_world_m'][2] * 1000
    new = plan['head_world_m'][2] * 1000
    geo = plan['geometric_head_height_m'] * 1000
    fig.text(.5, .905, f'原针高 {old:.2f} mm → 已验收针高 {new:.2f} mm；13 个 task 共用这个固定位置',
             ha='center', fontsize=13, color='#405d70')
    ax = fig.add_axes([.08, .19, .37, .63])
    rows = sorted(plan['tasks'], key=lambda r: r['id'])
    y = np.arange(len(rows))
    actual = np.array([r['shell_to_ground_at_selected_reference_m'] for r in rows]) * 1000
    geometric = np.array([r['shell_to_ground_at_geometric_reference_m'] for r in rows]) * 1000
    ax.barh(y - .17, actual, height=.31, color='#507fa1', label=f'已验收公共针高 {new:.2f} mm')
    ax.barh(y + .17, geometric, height=.31, color='#dcaa91', label=f'仅几何贴地 {geo:.2f} mm（不可行）')
    for i, value in enumerate(actual):
        ax.text(value + 2, y[i] - .17, f'{value:.1f}', va='center', fontsize=9, color='#365871')
    ax.set_yticks(y, [f"{r['id']} · 组 {r['set']}" for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel('完整薄壳最低点离地距离（mm）')
    ax.set_xlim(0, max(actual) * 1.2)
    ax.grid(axis='x', color='#e2e8ed', linewidth=.6)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper left', bbox_to_anchor=(-.02, -.14), frameon=False, fontsize=10)
    ax.set_title('每个姿态需要补多少高度', fontsize=15, pad=17, color='#29475c')

    model, _, _ = load_model(ROOT / 'subdivide_task/data' / report['object'] / 'step3_membranes')
    bound = plan['limiting_pressure_pair']
    group = report['groups'][bound['set'] - 1]['core']
    q, core = read_group(group)
    part = model.subset(group['members'])
    i = part.ids.index(bound['task'])
    j = part.ids.index(bound['other_task'])
    clouds = []
    for distance in (plan['lowering_m'], plan['geometric_max_lowering_m']):
        shifted = lower_layout(q, distance)
        rows_p = pressure_rows(lower_model(part, distance), shifted)
        clouds.append(transform(rows_p[i]['ground_material'], np.linalg.inv(shifted.material[j])) * 1000)
    vertices = transform(core.vertices, np.linalg.inv(lower_layout(q, plan['lowering_m']).material[j])) * 1000
    bx = fig.add_axes([.49, .23, .48, .59], projection='3d', proj_type='ortho')
    bx.add_collection3d(Poly3DCollection(vertices[core.faces], facecolor='#acbdc8',
                                        edgecolor='none', alpha=.3, rasterized=True))
    for points, color, label in zip(clouds, ('#27865d', '#c45440'), ('已验收压力凸包', '几何贴地后的压力凸包')):
        bx.add_collection3d(Poly3DCollection([points], facecolor=color, edgecolor=color, alpha=.3))
        line = np.vstack([points, points[0]])
        bx.plot(*line.T, color=color, linewidth=2, label=label)
    all_points = np.vstack([vertices, *clouds])
    low, high = all_points.min(0), all_points.max(0)
    floor = np.array([[low[0], low[1], 0], [high[0], low[1], 0],
                      [high[0], high[1], 0], [low[0], high[1], 0]])
    bx.add_collection3d(Poly3DCollection([floor], facecolor='#a4afb4', alpha=.17, edgecolor='#8b979e'))
    center = (low + high) / 2
    span = np.max(high - low) * 1.13
    bx.set_xlim(center[0] - span / 2, center[0] + span / 2)
    bx.set_ylim(center[1] - span / 2, center[1] + span / 2)
    bx.set_zlim(center[2] - span / 2, center[2] + span / 2)
    bx.set_box_aspect([1, 1, 1])
    bx.view_init(elev=20, azim=-70)
    bx.set_xlabel('X (mm)', fontsize=9)
    bx.set_ylabel('Y (mm)', fontsize=9)
    bx.set_zlabel('离地 Z (mm)', fontsize=9)
    bx.tick_params(labelsize=8)
    bx.set_title(f"限制案例：{bound['task']} 的承压范围 vs {bound['other_task']} 的地面", fontsize=13, pad=17)
    bx.legend(loc='upper right', bbox_to_anchor=(1., .98), frameon=False, fontsize=10)
    fig.text(.735, .16,
             f"针高降至 {geo:.2f} mm 时，此压力凸包最低点穿地 {-clouds[1][:, 2].min():.1f} mm。\n"
             '同一支撑不能长出会穿过另一姿态地面的接触点。',
             ha='center', va='top', fontsize=11, color='#9b4e42', linespacing=1.8)
    fig.text(.5, .023,
             f"保持 Step4 的实体、座位和旋转不变，受力必要条件最多允许降针 {plan['maximum_lowering_necessary_conditions_m']*1000:.3f} mm；"
             f"本次降针 {plan['lowering_m']*1000:.3f} mm，并重新构造、全量验收地脚。",
             ha='center', fontsize=10, color='#617684')
    output = Path(output or folder / 'height_reference.png')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor='white')
    plt.close(fig)
    write_json(folder / 'height_figure.json', dict(figure=str(output.resolve()), sha256=sha256(output),
               report_sha256=sha256(folder / 'report.json'), source_sha256=sha256(__file__)))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    render(args.folder, args.output)
