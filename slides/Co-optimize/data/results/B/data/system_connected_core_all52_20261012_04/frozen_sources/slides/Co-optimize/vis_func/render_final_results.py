"""Export saved Step4.2 support and show every installation with its actual exit."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from co_common import read_sets, HERE, ROOT, I, np, save, state, transform_points, trimesh

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Patch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def surface(ax, mesh, color):
    light = np.array([.6, -.8, 1.4])
    light /= np.linalg.norm(light)
    shade = .62 + .38 * np.maximum(mesh.face_normals @ light, 0)
    rgb = np.asarray(matplotlib.colors.to_rgb(color))
    ax.add_collection3d(Poly3DCollection(
        mesh.triangles * 1000, facecolors=shade[:, None] * rgb,
        edgecolors='none', zsort='average'))


def draw(group, support, obj, transforms, directions, passed, length, image):
    count = len(transforms)
    cols = 2 if count == 4 else min(3, count)
    rows = (count + cols - 1) // cols
    fig = plt.figure(figsize=(cols * 6.5, rows * 6.1 + 1.1), dpi=150,
                     facecolor='white')
    display_length = min(.10, length)
    pending = []
    for i, (pose, transform, direction) in enumerate(zip(
            group['poses'], transforms, directions)):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        world_obj = obj.copy()
        world_obj.apply_transform(transform)
        world_support = support.copy()
        world_support.apply_transform(transform)
        surface(ax, world_obj, '#a4a8ac')
        surface(ax, world_support, '#279bd7')
        origin = transform_points(obj.bounding_box.centroid[None, :], transform)[0] * 1000
        world_direction = transform[:3, :3] @ direction
        end = origin + display_length * 1000 * world_direction
        points = np.vstack([world_obj.vertices * 1000,
                            world_support.vertices * 1000, end])
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        radius = float(np.ptp(points, axis=0).max()) * .56
        for axis, value in zip('xyz', center):
            getattr(ax, 'set_' + axis + 'lim')(value - radius, value + radius)
        ax.set_box_aspect((1, 1, 1), zoom=1.10)
        ax.set_proj_type('ortho')
        ax.view_init(elev=np.degrees(np.arctan(1 / np.sqrt(2))), azim=-45)
        ax.set_axis_off()
        # Straight translations are the saved paths. The solid is not rebuilt.
        ax.plot(*np.vstack([origin, end]).T, color='#d69d00',
                linewidth=2.5, linestyle='--')
        lo = points[:, :2].min(axis=0) - 8
        hi = points[:, :2].max(axis=0) + 8
        floor = np.array([[lo[0], lo[1], 0], [hi[0], lo[1], 0],
                          [hi[0], hi[1], 0], [lo[0], hi[1], 0],
                          [lo[0], lo[1], 0]])
        ax.plot(*floor.T, color='#d5dade', linewidth=.7)
        ax.set_title(pose.replace('_', ' ').title(), fontsize=14, pad=0,
                     color='#233544')
        pending.append((ax, origin, end))
    status = 'Force / exit PASS' if passed else 'UNRESOLVED — saved best candidate'
    fig.suptitle(f"B / {group['id'].removeprefix('illegal/')}  |  {status}",
                 fontsize=17, color='#233544' if passed else '#b44b28', y=.985)
    fig.legend(handles=[Patch(color='#279bd7', label='Final shared support'),
                        Patch(color='#a4a8ac', label='Object'),
                        Patch(color='#f3c823', label='Saved exit path')],
               loc='lower center', bbox_to_anchor=(.5, .035), ncol=3,
               frameon=False, fontsize=12)
    fig.text(.5, .012,
             f"Same support in every pose · first {display_length * 1000:.0f} mm of each straight path shown · full exit {length * 1000:.0f} mm",
             ha='center', fontsize=10, color='#55636e')
    fig.subplots_adjust(left=.015, right=.985, bottom=.11,
                        top=.91, wspace=.02, hspace=.10)
    fig.canvas.draw()
    # Project actual path endpoints, keeping arrows visible over the geometry.
    for ax, origin, end in pending:
        def screen(point):
            x, y, _ = proj3d.proj_transform(*point, ax.get_proj())
            return ax.transAxes.inverted().transform(ax.transData.transform([x, y]))
        start, stop = screen(origin), screen(end)
        if np.linalg.norm(stop - start) < 1e-5:
            ax.text2D(.08, .12, 'Exit points toward camera', transform=ax.transAxes,
                      fontsize=10, color='#9c7600')
        else:
            fig.add_artist(FancyArrowPatch(
                start, stop, transform=ax.transAxes,
                arrowstyle='-|>,head_length=0.7,head_width=0.4',
                mutation_scale=24, linewidth=2.5, facecolor='#f3c823',
                edgecolor='#9c7600', zorder=100, clip_on=True))
    fig.savefig(image, facecolor='white')
    plt.close(fig)
    return display_length


def render(group):
    folder = HERE / 'output/B' / group['id'] / 'step4/step4.2'
    report_path = folder / 'data/report.json'
    source = folder / 'data/remaining_support.obj'
    report = json.loads(report_path.read_text())
    digest = I.sha256(source)
    if digest != report['artifacts']['remaining_support.obj']:
        raise RuntimeError(f'Saved model hash mismatch: {source}')
    support = trimesh.load(source, force='mesh', process=False)
    if not len(support.faces):
        raise RuntimeError(f'No saved support material: {source}')
    states = [state('B', pose) for pose in group['poses']]
    transforms = [transform for _, transform, _ in states]
    obj = states[0][2]
    directions = np.asarray(report['directions'])
    if directions.shape != (len(states), 3):
        raise RuntimeError('Saved pose/direction order mismatch')
    initialization = json.loads((folder / 'data/initialization_input.json').read_text())
    length = float(initialization['initialization']['full_length_m'])
    final = folder / 'final_results'
    final.mkdir(exist_ok=True)
    stl = support.copy()
    stl.apply_scale(1000)
    stl.export(final / 'support.stl', file_type='stl')
    display_length = draw(group, support, obj, transforms, directions,
                          report['passed'], length, final / 'final_results.png')
    record = dict(
        complete=True, pose_set=group['id'], passed=report['passed'],
        presentation_only=True, design_rerun=False,
        source_model=str(source.relative_to(ROOT)), source_model_sha256=digest,
        source_report_sha256=I.sha256(report_path),
        poses=group['poses'], directions_fixture=directions.tolist(),
        transforms_fixture_to_world=[t.tolist() for t in transforms],
        full_exit_length_m=length, displayed_path_length_m=display_length,
        stl_units='millimeters', stl_coordinate_frame='original object mesh',
        stl_scale_from_source=1000, triangle_count=len(support.faces),
        images_use_exact_saved_support=True,
        artifacts={name: I.sha256(final / name)
                   for name in ['final_results.png', 'support.stl']})
    save(folder / 'data/final_results.json', record)
    print('FINAL RESULTS', group['id'], 'PASS' if report['passed'] else 'UNRESOLVED', flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sets', nargs='+')
    args = parser.parse_args()
    groups = read_sets('B')['sets']
    groups += [dict(g, id='illegal/' + g['id']) for g in json.loads(
        (ROOT / 'objects/B/illegal_pose_sets.json').read_text())['sets']]
    if args.sets:
        unknown = set(args.sets) - {g['id'] for g in groups}
        if unknown:
            parser.error('Unknown sets: ' + ', '.join(sorted(unknown)))
        groups = [g for g in groups if g['id'] in args.sets]
    records = [render(g) for g in groups]
    save(HERE / 'output/B/data/final_results_batch.json', dict(
        complete=True, presentation_only=True, generated_groups=len(records),
        results=[dict(pose_set=r['pose_set'], passed=r['passed'],
                      artifacts=r['artifacts']) for r in records]))


if __name__ == '__main__':
    main()
