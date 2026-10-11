"""Try the user's partial-rim idea, then allow every unbounded legal foot.

Saved Step4 placements and Step5.1 loads remain byte-for-byte unchanged.
Results are a new side experiment inside Step5.3, not replacement bases.
"""
from __future__ import annotations

import html
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'step5.2'))
import stage5_common as C
sys.path.insert(0, str(HERE))
import partial_ground as P

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

STAGE = 'step5.3/partial_ground_v1'


def clip_region(vertices, a, b):
    vertices = np.asarray(vertices).reshape(-1, 2)
    for normal, offset in zip(a, b):
        output = []
        for first, last in zip(vertices, np.roll(vertices, -1, axis=0)):
            x, y = float(first @ normal + offset), float(last @ normal + offset)
            if x >= -P.TOL:
                output.append(first)
            if (x >= -P.TOL) != (y >= -P.TOL):
                output.append(first + np.clip(x/(x-y), 0., 1.) * (last-first))
        vertices = np.asarray(output).reshape(-1, 2)
        if not len(vertices):
            break
    return vertices


def draw_polygon(axis, points, color, fill=None, width=1.7, style='-'):
    points = np.asarray(points).reshape(-1, 2)
    if not len(points):
        return
    if len(points) > 2:
        if fill is not None:
            axis.fill(points[:, 0], points[:, 1], color=fill, zorder=1)
        points = np.vstack([points, points[0]])
    axis.plot(points[:, 0], points[:, 1], color=color, linewidth=width,
              linestyle=style, zorder=2)


def render_trial(row, result, path):
    hull, own = row['hull'][:, :2]*1000, row['object_ground']*1000
    contacts = np.asarray(result['retained_contact_convex_hull_xy_m']).reshape(-1, 2)*1000
    segments = np.asarray(result['retained_segments_xy_m']).reshape(-1, 2, 2)*1000
    joined = np.vstack([hull, own])
    center = (joined.min(0)+joined.max(0))/2
    span = max(float(np.ptp(joined, axis=0).max())*1.35, 1.)
    low, high = center-span/2, center+span/2
    rectangle = np.array([[low[0], low[1]], [high[0], low[1]],
                          [high[0], high[1]], [low[0], high[1]]])
    a, b = np.asarray(result['floor_normal_coefficients_xy']), np.asarray(result['floor_offsets_m'])*1000
    # This rectangle is ONLY the figure's viewport. Neither experiment nor
    # its linear program imposes it as a contact-location or BBX restriction.
    region = clip_region(rectangle, a, b)
    figure, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
    for axis in axes:
        axis.set_xlim(low[0], high[0]); axis.set_ylim(low[1], high[1])
        axis.set_aspect('equal'); axis.set_axis_off()
        draw_polygon(axis, hull, '#cf4057')
        if len(own):
            axis.scatter(own[:, 0], own[:, 1], s=25, color='#397aa0', zorder=5)
    draw_polygon(axes[0], contacts, '#bca45c', fill='#f4eed8', style='--')
    # Redraw demand outline above the virtual contact hull, then show only
    # retained legal arcs as green material candidates, with separated ends.
    draw_polygon(axes[0], hull, '#cf4057')
    for segment in segments:
        axes[0].plot(segment[:, 0], segment[:, 1], color='#328758', linewidth=4., zorder=4)
        axes[0].scatter(segment[:, 0], segment[:, 1], s=22, color='#328758', zorder=5)
    draw_polygon(axes[1], region, '#628c46', fill='#e4edd7', width=1.)
    draw_polygon(axes[1], hull, '#cf4057')
    witness = result['strongest_separating_floor']
    if witness:
        normal = np.asarray(witness['normal_xy'])
        offset = (witness['offset_m']-witness['attainable_lower_bound_m'])*1000
        norm2 = float(normal @ normal)
        if norm2 > 1e-20:
            origin = center-((center @ normal+offset)/norm2)*normal
            direction = np.array([-normal[1], normal[0]])/np.sqrt(norm2)
            ends = np.array([origin-direction*span*2, origin+direction*span*2])
            axes[1].plot(ends[:, 0], ends[:, 1], color='#24384a', linewidth=1.8, zorder=6)
        point = np.asarray(witness['demand_xy_m'])*1000
        for axis in axes:
            axis.scatter(*point, marker='x', s=80, color='#d92342', linewidths=2., zorder=7)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=.03)
    figure.savefig(path, facecolor='white')
    plt.close(figure)


def main():
    _, inputs, cases, directory = C.arguments(STAGE, 'step53_partial_ground_v1')
    code = C.begin(directory, STAGE, inputs, cases)
    summaries = []
    for case in cases:
        start = time.perf_counter()
        data = inputs.load(case)
        out = inputs.base/case['id']/'step5'/STAGE
        out.mkdir(parents=True, exist_ok=False)
        results = [P.analyze(row, data['states'], i) for i, row in enumerate(data['rows'])]
        bad = [r for r in results if not r['unlimited_ground_necessary_coverage_passed']]
        rim_bad = [r for r in results if not r['retained_parts_cover_entire_hull']]
        choices = [r for r in results if r['strongest_separating_floor']]
        chosen = max(choices, key=lambda r:r['strongest_separating_floor']['gap_m']) if choices else max(results, key=lambda r:r['retained_parts_uncovered_demand_count'])
        index = next(i for i, r in enumerate(results) if r is chosen)
        render_trial(data['rows'][index], chosen, out/'partial_boundary_trial.png')
        witness = chosen['strongest_separating_floor']
        if witness:
            witness['other_task'] = data['rows'][witness['other_task_index']]['pose']
        summary = dict(id=case['id'], pose_count=len(results), demand_count=sum(r['demand_count'] for r in results),
                       partial_boundary_failed_pose_count=len(rim_bad),
                       unlimited_legal_ground_failed_pose_count=len(bad),
                       partial_boundary_uncovered_demand_count=sum(r['retained_parts_uncovered_demand_count'] for r in results),
                       fixed_layout_can_cover_all_demands_in_generous_unlimited_model=not bad,
                       shown_task=chosen['pose'], strongest_separating_floor=witness,
                       shown_task_retained_boundary_fraction=chosen['retained_boundary_fraction'],
                       shown_task_partial_boundary_uncovered_demand_count=chosen['retained_parts_uncovered_demand_count'],
                       seconds=time.perf_counter()-start,
                       figure=C.relative(out/'partial_boundary_trial.png'), report=C.relative(out/'data/report.json'))
        report = dict(summary, complete=True, per_pose=results,
                      no_continuous_material_rim_required=True, no_bounding_box_limit=True,
                      retains_current_fixture_states_and_object_positions=True,
                      includes_actual_object_floor_contacts=True,
                      ignores_connection_thickness_body_work_and_exit_obstacles=True,
                      original_nominal_mesh_floor_penetration_is_not_used_to_reject_this_trial=True,
                      geometric_necessary_condition_only=True, successful_base_constructed=False,
                      step4_force_pass_rechecked_or_revoked=False,
                      interpretation='Partial legal boundary is tested first; independent LP then permits every legal foot location, with unlimited extent. Infeasibility is stronger than a finite candidate search failure.')
        C.save(out/'data/report.json', report)
        (out/'index.html').write_text('<!doctype html><meta charset="utf-8">'
            f'<h1>{html.escape(case["id"])} · Partial ground contact trial</h1>'
            f'<p>Task {chosen["pose"]}. Continuous rim material is not required. '
            f'{len(bad)} poses fail even with unrestricted legal feet.</p>'
            '<p>Left: green segments are the legal part of the demand boundary; cream is the virtual convex hull '
            'of their endpoints and blue object-floor contacts. Red is the required hull. '
            'Right: green is the unbounded legal fixture-ground region, cropped only for display; '
            'the dark line includes object-floor contribution. The red cross is a separating demand. '
            'There is no BBX or contact-radius constraint in the independent linear program.</p>'
            '<img style="max-width:100%" src="partial_boundary_trial.png">\n')
        summaries.append(summary)
        print('PARTIAL GROUND', case['id'], f'partial-rim failures {len(rim_bad)}, unlimited-ground failures {len(bad)}', flush=True)
    C.finish(directory, inputs, code, dict(step=STAGE, cases=summaries,
        group_count=len(summaries), pose_count=sum(r['pose_count'] for r in summaries),
        original_demand_count=sum(r['demand_count'] for r in summaries),
        partial_boundary_all_pose_pass_group_count=sum(r['partial_boundary_failed_pose_count']==0 for r in summaries),
        unlimited_legal_ground_all_pose_pass_group_count=sum(r['unlimited_legal_ground_failed_pose_count']==0 for r in summaries),
        no_continuous_material_rim_required=True, no_bounding_box_limit=True))
    cards = ''.join(f'<section><h2><a href="{r["id"]}/step5/{STAGE}/index.html">{html.escape(r["id"])}</a></h2>'
                   f'<p>{r["shown_task"]}; legal boundary retained {r["shown_task_retained_boundary_fraction"]:.1%}; '
                   f'{r["unlimited_legal_ground_failed_pose_count"]}/{r["pose_count"]} poses fail even with unlimited legal feet.</p>'
                   f'<img src="{r["id"]}/step5/{STAGE}/partial_boundary_trial.png"></section>' for r in summaries)
    gallery = inputs.base/'step5.3_partial_ground_index.html'
    if gallery.exists():
        raise ValueError('Existing partial-ground gallery is protected')
    gallery.write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;max-width:1700px;margin:24px auto}'
        'img{width:100%}section{margin:32px 0}</style><h1>Step5.3 · Partial boundary and unlimited legal ground</h1>'
        '<p>Step4 layouts and original demand points unchanged. No continuous material rim or BBX restriction.</p>'
        '<p>Left: green = retained legal boundary segments; cream = their virtual support hull including blue object-floor contacts; '
        'red = required hull. Right: green = unbounded legal ground region, cropped ONLY for display; '
        'dark = separating line; red cross = impossible demand. An independent linear program permits every legal foot location.</p>'+cards+'\n')
    print('GALLERY', C.relative(gallery), flush=True)


if __name__ == '__main__':
    main()
