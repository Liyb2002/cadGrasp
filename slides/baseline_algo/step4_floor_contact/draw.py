"""Draw current whole-assembly floor demand; retain historical drawing helpers."""
import argparse
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step1.needs import COORD
from step4_floor_contact import floor_contact as F
from step2_local_support import render as R
from step1.cases import pose_name

PALETTE = np.array([[41, 139, 147], [210, 128, 44], [106, 120, 187],
                    [162, 90, 142], [99, 151, 94], [164, 126, 89]], float)
VIEW = R.axes([.68, -1., .8])
OBJECT_OPACITY = .72
OCCLUDED_CONTACT_OPACITY = .60


def color(j):
    return PALETTE[j % len(PALETTE)]/255


def boundary(points):
    points = np.unique(points, axis=0)
    if len(points) < 3 or np.linalg.matrix_rank(points-points.mean(axis=0)) < 2:
        return points
    return points[ConvexHull(points).vertices]


def plan_view(domain, contacts, report, arrays, out):
    fig, ax = plt.subplots(figsize=(10, 9), dpi=160)
    fig.patch.set_facecolor('#ffffff'); ax.set_facecolor('#ffffff')
    envelope = boundary(COORD.floor(domain.mesh.vertices))*1000
    ax.fill(*envelope.T, color='#bdc4c7', alpha=.3)
    for j, contact in enumerate(contacts):
        xy = boundary(COORD.floor(contact['triangles_m'].reshape(-1, 3)))*1000
        if len(xy) >= 3: ax.fill(*xy.T, color=color(j), alpha=.55)
    for j, foot in enumerate(report['ground_footprints']):
        for pad in foot['pads_xy_m']:
            xy = np.asarray(pad)*1000
            ax.fill(*xy.T, color=color(j), alpha=.95)
        hull = np.asarray(foot['hull_xy_m'])*1000
        ax.plot(*np.vstack([hull, hull[0]]).T, color=color(j), lw=1.3, ls='--', alpha=.65)
        points = arrays['pressure_centers_xy_m'][:, j]
        if 'continuous_pressure_centers_xy_m' in arrays:
            points = np.vstack([points, arrays['continuous_pressure_centers_xy_m'][:, j]])
        points = points[np.isfinite(points).all(axis=1)]
        if len(points):
            region = boundary(points)*1000
            if len(region) >= 3: ax.fill(*region.T, color=color(j), alpha=.16)
            ax.plot(*np.vstack([region, region[0]]).T, color=color(j), lw=2)
    ax.scatter(*(COORD.floor(arrays['original_pivot_m'])*1000), s=45, marker='x', c='#33424a')
    ax.set_aspect('equal'); ax.set_axis_off(); ax.margins(.1)
    fig.subplots_adjust(left=.025, right=.975, bottom=.025, top=.975)
    fig.savefig(out/'floor_contact.png'); plt.close(fig)


def object_view(domain, contacts, report, out, size=1500):
    feet = report['ground_footprints']
    pads = []
    pad_colors = []
    for j, foot in enumerate(feet):
        for xy in foot['pads_xy_m']:
            corners = COORD.lift_floor(xy)
            pads.append(corners[[[0, 1, 2], [0, 2, 3]]])
            pad_colors.append(np.tile(PALETTE[j % len(PALETTE)], (2, 1)))
    patches = [c['triangles_m'] for c in contacts]
    all_points = np.concatenate([domain.mesh.vertices]+[p.reshape(-1, 3) for p in pads])
    margin = .08*domain.mesh.extents.max()
    lo = COORD.floor(all_points).min(axis=0)-margin; hi = COORD.floor(all_points).max(axis=0)+margin
    floor = np.array([[lo[0], lo[1], 0], [hi[0], lo[1], 0],
                      [hi[0], hi[1], 0], [lo[0], hi[1], 0]])[[[0, 1, 2], [0, 2, 3]]]
    projected = np.vstack([all_points, floor.reshape(-1, 3)])@VIEW.T
    low, high = projected.min(axis=0), projected.max(axis=0)
    focus = ((low+high)/2)@VIEW
    width = 1.1*max(high[:2]-low[:2])
    under, _ = R.raster(np.concatenate([floor]+pads), np.concatenate([np.tile(R.FLOOR, (2, 1))]+pad_colors),
                        focus, VIEW, width, size, overlay=np.arange(2, 2+sum(map(len, pads))), unlit=range(2+sum(map(len, pads))))
    body_colors = np.tile(R.GREY, (len(domain.mesh.faces), 1)); body_colors[domain.work_ids] = R.GREEN
    patch_colors = [np.tile(PALETTE[j % len(PALETTE)], (len(p), 1)) for j, p in enumerate(patches)]
    body, ids = R.raster(np.concatenate([domain.mesh.triangles]+patches),
                         np.concatenate([body_colors]+patch_colors), focus, VIEW, width, size,
                         overlay=np.arange(len(domain.mesh.faces), len(domain.mesh.faces)+sum(map(len, patches))))
    blended = Image.blend(under, body, OBJECT_OPACITY)
    under.paste(blended, mask=Image.fromarray(np.uint8(ids >= 0)*255))
    # Keep the actual fixed contact faces opaque in front of the posed body.
    patch_mask = ids >= len(domain.mesh.faces)
    under.paste(body, mask=Image.fromarray(np.uint8(patch_mask)*255))
    if patches:
        # Show rear heads through the translucent body so their ownership is
        # visible in the same view as their feet, without moving the geometry.
        heads, head_ids = R.raster(np.concatenate(patches), np.concatenate(patch_colors),
            focus, VIEW, width, size, overlay=np.arange(sum(map(len, patches))))
        hidden = (head_ids >= 0) & ~patch_mask
        under.paste(Image.blend(under, heads, OCCLUDED_CONTACT_OPACITY),
                    mask=Image.fromarray(np.uint8(hidden)*255))
    under.save(out/'floor_contact_object.png')
    return dict(basis=VIEW.tolist(), focus_m=focus.tolist(), width_m=float(width),
                object_opacity=OBJECT_OPACITY, occluded_contact_opacity=OCCLUDED_CONTACT_OPACITY,
                rear_contacts_shown_through_body=True, pad_z_m=0., connectors_drawn=False)


def diagnostic_page(report, out):
    picture = Image.open(out/'floor_contact_object.png').convert('RGB')
    fig, (ax, info) = plt.subplots(1, 2, figsize=(15, 7.8), gridspec_kw={'width_ratios': [1.2, 1]}, dpi=150)
    fig.patch.set_facecolor('#ffffff')
    ax.imshow(picture); ax.set_axis_off(); info.set_axis_off()
    title = report['object']+' / '+report['pose'].replace('_', ' ')+' / independent feet'
    fig.text(.035, .94, title, fontsize=20, color='#23383c')
    status = report['status']
    failure = report.get('failure', {})
    good = report['sampled_independent_equilibrium_verified']
    info.text(0, .91, 'Fixed bearing footprints' if good else 'Fixed foot candidates / not certified', fontsize=17,
              color='#298b93' if good else '#ad4b3d', transform=info.transAxes)
    info.text(0, .84, status.replace('_', ' '), fontsize=10, wrap=True, transform=info.transAxes)
    columns = min(5, max(2, int(np.ceil(len(report['selected_ids'])/6))))
    rows = max(1, int(np.ceil(len(report['selected_ids'])/columns)))
    spacing = min(.055, .24/max(1, rows-1))
    for j, cid in enumerate(report['selected_ids']):
        x, y = (j%columns)/columns, .73-(j//columns)*spacing
        info.scatter(x+.015, y, color=color(j), s=55, transform=info.transAxes)
        info.text(x+.05, y, cid, va='center', fontsize=11 if columns < 4 else 9, transform=info.transAxes)
    if failure:
        label = failure['load']
        load_names = {'stored_reachable_sample': 'Stored reachable sample',
                      'validated_step3_counterexample': 'Validated Step 3 counterexample'}
        text = ('Zero process force (gravity only)' if label['kind'] == 'zero_process_force' else
                load_names.get(label['kind'], label['kind'])+' '+str(label.get('sample_index', label.get('index', '')))+' (allowed load)')
        info.text(0, .43, text, fontsize=11, transform=info.transAxes)
        share = failure.get('workpiece_only_allocation', {}).get('required_floor_normal_mg', [])
        lines = [f'{cid}: required floor normal = {value:+.5f} mg' for cid, value in zip(report['selected_ids'], share)]
        if len(lines) > 4:
            lines = lines[:4]+[f'{len(lines)-4} further allocations in floor_contact.json']
        info.text(0, .38, '\n'.join(lines), fontsize=10, va='top', linespacing=1.7, transform=info.transAxes)
        exact = failure.get('exact_separator') is not None
        info.text(0, .12, ('Exact separation proves no reallocation works.' if exact else 'Joint LP result and diagnostics saved in JSON.')+
                  '\nNegative floor normal would require anchoring.\nDisplayed allocation is illustrative; it is not the proof.',
                  fontsize=10, linespacing=1.6, transform=info.transAxes)
    elif good:
        info.text(0, .36, 'Solid squares: actual ground-bearing pads\nDashed boundary: each support\'s footprint hull\nEach colour belongs to one independent support.\nNo shared ring and no connectors.',
                  fontsize=11, linespacing=1.8, transform=info.transAxes)
        info.text(0, .13, f"Finite sufficient floor friction: mu = {report['sufficient_friction_coefficient']:g}\nContinuous domain: {report['continuous_verification']['status']}",
                  fontsize=11, transform=info.transAxes)
    else:
        info.text(0, .35, report.get('reason', ''), fontsize=11, wrap=True, transform=info.transAxes)
    fig.text(.035, .035, 'Step 4: fixed contact surfaces and individual ground bearing. Connections and installation remain Step 5.', fontsize=10, color='#59676a')
    fig.subplots_adjust(left=.015, right=.98, top=.89, bottom=.08, wspace=.06)
    fig.savefig(out/'floor_diagnostic.png'); plt.close(fig)


def run(name):
    report = F.read(name); out = F.output_folder(name)
    arrays = F.I.load_npz(out/report['arrays_file'])
    source = F.OUTPUTS/name/pose_name()/'step_1_needs/needs.json'
    domain = F.ContinuousNeeds.read(source)
    contact_path = out.parent/'step3_scheculer/final_contacts.npz'
    contacts = F.I.read_contacts(contact_path) if contact_path.exists() else []
    plan_view(domain, contacts, report, arrays, out)
    view = object_view(domain, contacts, report, out)
    diagnostic_page(report, out)
    images = ['floor_contact.png', 'floor_contact_object.png', 'floor_diagnostic.png']
    F.I.save(out/'views.json', dict(complete=True, object=name, pose=pose_name(),
        text_in_primary_figures=False, diagnostic_figure_has_status_text=True,
        whole_assembly_demand_cloud_drawn=False, shared_ground_ring_drawn=False,
        view=view, status=report['status'],
        provenance=dict(inputs=F.I.hashes([source, out/'floor_contact.json', out/report['arrays_file']]),
                        code=F.I.hashes([Path(__file__), Path(R.__file__)])),
        artifacts={p: F.sha256(out/p) for p in images}))
    print(name, pose_name(), 'Step 4 independent bearing figures written', flush=True)


# Current one-body entry; earlier independent-body helpers remain for regressions.
from step4_floor_contact.whole_assembly import draw as run


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('objects', nargs='*')
    for name in parser.parse_args().objects or F.OBJECTS: run(name)
