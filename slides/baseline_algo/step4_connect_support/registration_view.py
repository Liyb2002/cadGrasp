"""Show registered heads and a failed floor precondition, never a fake fixture."""
from pathlib import Path
import numpy as np
import trimesh

from step1.needs import ROOT
from step2_local_support import geometry as G
from step3_scheculer import contacts as I
from step4_connect_support import greedy_geometry as GG, geometry_kernel as K
from step4_connect_support.fixture_view import pack, export_viewer
from step4_connect_support.refresh_shared_geometry_view import write_viewer


def plot_conflict(out, case, floor_check):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), constrained_layout=True)
    for ax, task, xy, check in zip(axes, case.tasks, case.demands, floor_check['per_pose']):
        xy = np.asarray(xy)*1000
        a, b, c = check['floor_xy_halfplane_coefficients']
        c *= 1000
        low, high = xy.min(0)-12, xy.max(0)+12
        xx, yy = np.meshgrid(np.linspace(low[0], high[0], 220), np.linspace(low[1], high[1], 220))
        z = a*xx+b*yy+c
        ax.contourf(xx, yy, (z < 0).astype(float), levels=[-.5, .5, 1.5], colors=['#f1f5f7', '#fce5e4'])
        if z.min() < 0 < z.max():
            ax.contour(xx, yy, z, levels=[0], colors=['#a34840'], linewidths=1.4)
        outside = xy@np.array([a, b])+c < -1e-6
        ax.scatter(*xy[~outside].T, s=1, alpha=.13, color='#406b86', rasterized=True)
        if outside.any():
            ax.scatter(*xy[outside].T, s=2, alpha=.25, color='#bf3c34', rasterized=True)
            worst = check['worst_sample_index']
            ax.scatter(*xy[worst], marker='*', s=110, color='#a42320', zorder=5)
        hull = ConvexHull(xy)
        rim = xy[np.r_[hull.vertices, hull.vertices[0]]]
        ax.plot(*rim.T, color='#334d5c', lw=.8)
        ax.scatter(*(task.floor[:2]*1000), s=45, facecolors='white', edgecolors='#1f3544', zorder=5)
        ax.set(title=f"{task.pose.replace('pose_', 'Pose ')}: {check['violating_sample_count']:,} / {len(xy):,} outside",
               xlabel='Floor x (mm)', ylabel='Floor y (mm)', xlim=(low[0], high[0]), ylim=(low[1], high[1]))
        ax.set_aspect('equal')
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Fixed shared contact: original floor demands vs. the other floor\nRed region is forbidden for material; star = worst original demand', fontsize=12)
    fig.savefig(out/'floor_conflict.png', dpi=180)
    plt.close(fig)


def export_failure(out, report, case, bases, offsets, heads, registration, floor_check):
    pieces = [K.unpack(GG.union(K.solid(G.hull_mesh(v)) for v in h.cells)) for h in heads]
    mesh = trimesh.util.concatenate(pieces)
    mesh.export(out/'head_layout.obj', file_type='obj', digits=17, include_normals=False)
    np.savez_compressed(out/'head_layout.npz', vertices_m=mesh.vertices, faces=mesh.faces,
        rotations=bases, local_offsets_m=offsets,
        physical_head_ids=np.asarray([h.ident for h in heads]),
        head_vertex_offsets=np.cumsum([0]+[len(p.vertices) for p in pieces]),
        head_face_offsets=np.cumsum([0]+[len(p.faces) for p in pieces]))
    plot_conflict(out, case, floor_check)
    report.update(particle=case.schedule['particle'], source_schedule=str((case.source/'schedule.json').relative_to(ROOT)),
        diagnostic_only=True, diagnostic_geometry_kind='registered_original_heads_without_body',
        registration=registration, floor_compatibility=floor_check,
        dimensions_mm=(mesh.extents*1000).tolist(),
        task_fixture_transforms=[dict(pose=p, rotation=b.tolist(), translation_m=(-b@o).tolist())
                                for p, b, o in zip(case.poses, bases, offsets)],
        presentation_description='未通过：五个头已合并为唯一实体，共享橙色面保持重合。这里仅显示原始头的位置，没有生成完整支架。下图红色区域违反另一姿态地面约束。')
    report['artifacts'].update({name:I.sha256(out/name) for name in ('head_layout.obj', 'head_layout.npz', 'floor_conflict.png')})
    report['provenance']['code'].update(I.hashes([Path(__file__), Path(__file__).with_name('fixture_view.py'),
        Path(__file__).with_name('refresh_shared_geometry_view.py'), Path(__file__).with_name('shared_geometry_viewer.html')]))
    I.save(out/'report.json', report)
    shared = case.schedule['shared_head']['selected_id']
    colors = {shared:'#dc9d47'}
    colors.update(zip([i for i in case.schedule['selected_ids'] if i != shared],
                      ['#ac7098', '#7196c0', '#50a59b', '#77a76a']))
    export_viewer(out, mesh, [(h.ident, p) for h, p in zip(heads, pieces)],
                  report, case.tasks, bases, offsets, colors)
