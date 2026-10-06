"""Show saved endpoint geometry and real sampling/gradient records, without rerunning."""
import argparse
import json
from pathlib import Path

from render_final_results import (
    HERE, ROOT, I, np, save, state, trimesh, plt, FancyArrowPatch, proj3d, surface)


def render(group):
    folder = HERE/'output/B'/group['id']/'step4/step4.2'
    data = folder/'data'
    report = json.loads((data/'report.json').read_text())
    proposals = json.loads((data/'global_proposals.json').read_text())
    trace = json.loads((data/'optimization_trace.json').read_text())
    initial = np.load(data/'initial_directions.npz')['directions']
    final = np.asarray(report['directions'])
    seed_path = folder.parents[1]/'step3/step3.3/support_with_rings.obj'
    seed = trimesh.load(seed_path, force='mesh', process=False)
    support = trimesh.load(data/'remaining_support.obj', force='mesh', process=False)
    obj = state('B', group['poses'][0])[2]
    fig = plt.figure(figsize=(24, 8.3), dpi=150, facecolor='white')
    grid = fig.add_gridspec(2, 4, height_ratios=[1, 1], left=.015,
                           right=.985, top=.83, bottom=.17, wspace=.14, hspace=.45)
    palette = plt.get_cmap('tab10').colors
    origin = obj.bounding_box.centroid*1000
    endpoints = origin + 80*np.vstack([initial, final])
    points = np.vstack([obj.vertices*1000, seed.vertices*1000, endpoints])
    center = (points.min(axis=0)+points.max(axis=0))/2
    radius = float(np.ptp(points, axis=0).max())*.56
    arrows = []
    for col, mesh, directions, title, detail in [
            (0, seed, None, '1  Shared initial material', 'Step3.3 seed; common object coordinates'),
            (1, seed, initial, '2  Initialize exit directions', 'Native-up exits; seed shown before cutting'),
            (3, support, final, '4  Saved final material',
             'Force / exit PASS' if report['passed'] else 'UNRESOLVED: saved best candidate')]:
        ax = fig.add_subplot(grid[:, col], projection='3d')
        surface(ax, obj, '#a4a8ac')
        surface(ax, mesh, '#279bd7')
        for axis, value in zip('xyz', center):
            getattr(ax, 'set_'+axis+'lim')(value-radius, value+radius)
        ax.set_proj_type('ortho')
        ax.set_box_aspect((1, 1, 1), zoom=1.05)
        ax.view_init(elev=np.degrees(np.arctan(1/np.sqrt(2))), azim=-45)
        ax.set_axis_off()
        ax.set_title(title, fontsize=14, pad=0, color='#233544')
        ax.text2D(.5, -.01, detail, ha='center', transform=ax.transAxes,
                  fontsize=10, color='#55636e')
        if directions is not None:
            for i, direction in enumerate(directions):
                arrows.append((ax, origin, origin+80*direction, palette[i]))
    sampling = fig.add_subplot(grid[0, 2])
    sampling.set_title('3  Sampling + local gradient', fontsize=14, color='#233544', pad=20)
    built = [p for p in proposals if p.get('status') == 'constructed' and p.get('counts')]
    x = [p['proposal'] for p in built]
    worst = [min(p['counts'])/32768*100 for p in built]
    mean = [np.mean(p['counts'])/32768*100 for p in built]
    initial_counts = report['initial_counts']
    if initial_counts is not None:
        x.insert(0, 0)
        worst.insert(0, min(initial_counts)/32768*100)
        mean.insert(0, np.mean(initial_counts)/32768*100)
    sampling.plot(x, mean, color='#83bcdc', marker='.', label='Mean pose')
    sampling.plot(x, worst, color='#237fae', marker='.', label='Worst pose')
    selected_x = max([p.get('proposal', 0) for p in proposals]+[0])
    sampling.scatter([selected_x], [min(report['final_counts'])/32768*100],
                     color='#209861' if report['passed'] else '#bc6337',
                     marker='*', s=100, label='Saved result', zorder=5)
    sampling.set_ylim(-5, 105)
    sampling.set_ylabel('Original loads passed (%)', fontsize=10)
    sampling.set_xlabel('Sampling proposal number', fontsize=10)
    sampling.grid(alpha=.18)
    sampling.legend(loc='lower right', fontsize=8)
    gradient = fig.add_subplot(grid[1, 2])
    trials = [t for iteration in trace for t in iteration.get('trials', []) if t.get('counts')]
    if trials:
        values = [min(t['counts'])/32768*100 for t in trials]
        gradient.plot(range(1, len(values)+1), values, color='#dd9a28', marker='.')
        accepted = [(i+1, values[i]) for i, t in enumerate(trials) if t.get('accepted')]
        if accepted:
            gradient.scatter(*np.asarray(accepted).T, color='#209861', s=25,
                             label='Accepted local step', zorder=5)
            gradient.legend(fontsize=8, loc='lower right')
        gradient.set_ylim(-5, 105)
        gradient.set_xlabel('Recorded gradient trial order', fontsize=10)
        gradient.set_ylabel('Worst pose passed (%)', fontsize=10)
        gradient.grid(alpha=.18)
    else:
        gradient.set_axis_off()
        text = ('Initial state already passes; no search needed.' if not proposals else
                'No gradient steps recorded.\nThis saved result comes from sampling.')
        if not report['passed']:
            text = 'No gradient steps recorded.\nBounded sampling did not resolve this group.'
        gradient.text(.5, .7, text, ha='center', va='center', fontsize=11,
                      color='#55636e', transform=gradient.transAxes)
    rejected = sum(p.get('status') == 'optimistic cone separator rejected' for p in proposals)
    details = f"{len(built)} constructed sampling candidates · {rejected} certified prefilter rejections · {len(trials)} recorded gradient trials"
    if initial_counts is None:
        details += ' · initial geometry unresolved'
    fig.suptitle(f"B / {group['id']}  |  Step4.2 process", fontsize=20,
                 y=.96, color='#233544')
    fig.text(.5, .905, details, ha='center', fontsize=11, color='#55636e')
    fig.text(.5, .075, 'Exit colors:  '+ '   '.join(
        pose.replace('_', ' ') for pose in group['poses']), ha='center', fontsize=11)
    # Color labels tie pose IDs to arrows without inventing intermediate geometry.
    fig.canvas.draw()
    for ax, start, end, color in arrows:
        def screen(point):
            a, b, _ = proj3d.proj_transform(*point, ax.get_proj())
            return ax.transAxes.inverted().transform(ax.transData.transform([a, b]))
        fig.add_artist(FancyArrowPatch(screen(start), screen(end), transform=ax.transAxes,
                                     arrowstyle='-|>', mutation_scale=18, linewidth=2,
                                     color=color, zorder=100, clip_on=True))
    # A compact true-color pose legend applies to both direction panels.
    for i, pose in enumerate(group['poses']):
        fig.text(.30+i*.40/max(1, len(group['poses'])-1), .045,
                 pose.replace('_', ' '), color=palette[i], ha='center', fontsize=11)
    process = folder/'process'
    process.mkdir(exist_ok=True)
    image = process/'process.png'
    fig.savefig(image, facecolor='white')
    plt.close(fig)
    record = dict(complete=True, presentation_only=True, pose_set=group['id'],
                  passed=report['passed'], initial_geometry_shown='immutable Step3.3 seed, not an inferred initial Boolean result',
                  geometry_sources=[str(seed_path.relative_to(ROOT)),
                                    str((data/'remaining_support.obj').relative_to(ROOT))],
                  intermediate_geometry_inferred=False,
                  sampling_constructed=len(built), gradient_trials=len(trials),
                  source_files={str(p.relative_to(ROOT)): I.sha256(p) for p in [
                      seed_path, data/'remaining_support.obj', data/'report.json',
                      data/'global_proposals.json', data/'optimization_trace.json',
                      data/'initial_directions.npz']}, image_sha256=I.sha256(image))
    save(data/'process_render.json', record)
    print('PROCESS', group['id'], flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sets', nargs='+')
    args = parser.parse_args()
    groups = json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets']
    groups += [dict(g, id='illegal/'+g['id']) for g in json.loads(
        (ROOT/'objects/B/illegal_pose_sets.json').read_text())['sets']]
    if args.sets:
        unknown = set(args.sets)-{g['id'] for g in groups}
        if unknown:
            parser.error('Unknown sets: '+', '.join(sorted(unknown)))
        groups = [g for g in groups if g['id'] in args.sets]
    for group in groups:
        render(group)


if __name__ == '__main__':
    main()
