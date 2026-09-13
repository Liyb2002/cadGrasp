"""Draw saved final contacts and their shared, certified withdrawal directions."""
import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step1.cases import pose_name, selected_pose
from step1.needs import ContinuousNeeds
from step2_local_support import render as R, withdrawal as W
from step3_scheculer import contacts as I

COLORS = np.array([[49, 127, 195], [245, 139, 37], [146, 101, 180]], float)
TEAL = '#258778'
MUTED = '#65706c'
VIEW = R.axes([.68, .8, -1.])


def arrow(ink, start, end, width=5):
    delta = end-start
    length = np.linalg.norm(delta)
    if length < 1e-8:
        return
    along = delta/length
    across = np.array([-along[1], along[0]])
    ink.line([tuple(start), tuple(end)], fill=TEAL, width=width)
    ink.polygon([tuple(end), tuple(end-17*along+7*across),
                 tuple(end-17*along-7*across)], fill=TEAL)


def object_panel(domain, contacts, direction, basis, size):
    scale = float(domain.mesh.extents.max())
    centers = np.array([c['center_m'] for c in contacts])
    ends = centers+.24*scale*direction if direction is not None else centers
    floor = R.floor_triangles(domain)
    bounds = np.vstack([domain.mesh.vertices, floor.reshape(-1, 3), ends])@basis.T
    focus = ((bounds.min(axis=0)+bounds.max(axis=0))/2)@basis
    width = 1.12*max(np.ptp(bounds, axis=0)[:2])
    base, _ = R.raster(floor, np.tile(R.FLOOR, (len(floor), 1)),
                       focus, basis, width, size, unlit=range(len(floor)))
    body_colors = np.tile(R.GREY, (len(domain.mesh.faces), 1))
    body_colors[domain.work_ids] = R.GREEN
    body, ids = R.raster(domain.mesh.triangles, body_colors, focus, basis, width, size)
    base.paste(Image.blend(base, body, .50), mask=Image.fromarray(np.uint8(ids >= 0)*255))
    patches = np.concatenate([c['triangles_m'] for c in contacts])
    colors = np.concatenate([np.tile(COLORS[j % len(COLORS)], (len(c['triangles_m']), 1))
                             for j, c in enumerate(contacts)])
    overlay, ids = R.raster(patches, colors, focus, basis, width, size,
                            unlit=range(len(patches)))
    base.paste(overlay, mask=Image.fromarray(np.uint8(ids >= 0)*255))
    if direction is not None:
        ink = ImageDraw.Draw(base)
        for center, end in zip(centers, ends):
            start = center+.045*scale*direction
            a, b = R.project(np.array([start, end]), focus, basis, width, size)[:, :2]
            arrow(ink, a, b)
    return base, dict(basis=basis.tolist(), focus_m=focus.tolist(), width_m=float(width))


def direction_panel(vectors, representative, size):
    paper = Image.new('RGB', (size, size), R.PAPER)
    ink = ImageDraw.Draw(paper)
    center = np.array([size*.5, size*.5])
    radius = size*.36

    def project(points):
        p = np.asarray(points)@VIEW.T
        return center+p[..., :2]*[radius, -radius]

    t = np.linspace(0, 2*np.pi, 241)
    for axis in range(3):
        circle = np.zeros((len(t), 3))
        circle[:, (axis+1) % 3] = np.cos(t)
        circle[:, (axis+2) % 3] = np.sin(t)
        ink.line([tuple(p) for p in project(circle)], fill='#d7ddd9', width=2)
    ink.ellipse((*tuple(center-radius), *tuple(center+radius)), outline='#d7ddd9', width=2)
    for axis, label in zip(np.eye(3), ('x', 'y', 'z')):
        end = project(axis*1.14)
        ink.line([tuple(center), tuple(end)], fill='#b1bbb7', width=2)
        ink.text(tuple(project(axis*1.24)), label, font=R.font(23), fill=MUTED, anchor='mm')
    for end in project(vectors):
        ink.line([tuple(center), tuple(end)], fill='#b5d5cb', width=1)
        x, y = end
        ink.ellipse((x-4, y-4, x+4, y+4), fill=TEAL)
    if representative is not None:
        arrow(ink, center, project(representative), width=5)
    return paper


def run(name):
    out = I.OUTPUTS/name/pose_name()/'step3_scheculer'
    source = out.parent/'step_1_needs/needs.json'
    schedule = I.check_report(out/'schedule.json')
    directions = I.check_report(out/'insertion_directions.json')
    contacts = I.read_contacts(out/'final_contacts.npz')
    if not contacts:
        raise ValueError('No final contacts are available to highlight')
    selected = [c['candidate_id'] for c in contacts]
    assert selected == schedule['selected_ids'] == directions['selected_ids']
    for contact, record in zip(contacts, directions['contacts']):
        assert record['candidate_id'] == contact['candidate_id']
        assert record['geometry_signature'] == W.signature(contact, directions['normal_depth_m'])
    catalogue = directions['direction_catalogue']
    common = W.common(directions['contacts'], catalogue['global_allowed_directions'])
    assert common == directions['common_directions'] == schedule['common_withdrawal_directions']
    all_vectors = np.asarray(catalogue['vectors'])
    vectors = all_vectors[common['ids']]
    record = directions['common_representative']
    representative = None
    if record is not None:
        assert record['direction_id'] in common['ids']
        representative = all_vectors[record['direction_id']]
        np.testing.assert_array_equal(representative, record['vector'])
    domain = ContinuousNeeds.read(source)
    page = Image.new('RGB', (2460, 1120), R.PAPER)
    ink = ImageDraw.Draw(page)
    ink.text((35, 24), 'Selected contacts and common withdrawal', font=R.font(38), fill=R.INK)
    ink.text((35, 83), 'Colored patches: final contact surfaces. Teal arrows: the same certified withdrawal direction.',
             font=R.font(25), fill=MUTED)
    views = []
    for x, basis, label in [(10, VIEW, 'Overall view'),
                            (850, R.axes([1., .45, .35]), 'Contact-side view')]:
        panel, view = object_panel(domain, contacts, representative, basis, 840)
        page.paste(panel, (x, 170))
        ink.text((x+25, 143), label, font=R.font(27), fill=R.INK)
        views.append(view)
    page.paste(direction_panel(vectors, representative, 730), (1710, 230))
    ink.text((1735, 143), 'Common feasible directions', font=R.font(27), fill=R.INK)
    ink.text((1750, 875), 'Each dot is a certified direction.\nSame orientation as the overall view.',
             font=R.font(22), fill=MUTED, spacing=10)
    ink.text((35, 1018), 'Contact faces are shown through the translucent workpiece. Insertion reverses the arrows.',
             font=R.font(25), fill=R.INK)
    ink.text((35, 1060), 'Directions apply to the selected heads; frame and base checks belong to the connection stage.',
             font=R.font(23), fill=MUTED)
    page.save(out/'selected_contacts_directions.png')
    inputs = [source, out/'schedule.json', out/'insertion_directions.json', out/'final_contacts.npz']
    I.save(out/'selected_contacts_directions_views.json', dict(
        complete=True, object=name, pose=pose_name(), selected_ids=selected,
        common_directions=common, displayed_direction_vectors=vectors.tolist(),
        arrow_withdrawal_direction=record, insertion_reverses_arrows=True,
        contacts_match_saved_direction_geometry=True, views=views,
        contact_colors_rgb={c['candidate_id']: COLORS[j % len(COLORS)].tolist() for j, c in enumerate(contacts)},
        contact_labels_drawn=False, hidden_contacts_shown=True,
        continuous_direction_region_claimed=False, connectors_checked=False,
        provenance=dict(inputs=I.hashes(inputs), code=I.hashes([Path(__file__), Path(R.__file__), Path(W.__file__)])),
        artifacts={'selected_contacts_directions.png': I.sha256(out/'selected_contacts_directions.png')}))
    print(name, pose_name(), 'final contact and direction figure written', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('objects', nargs='*', default=['A1-f', 'B', 'C5'])
    parser.add_argument('--pose', default=None)
    args = parser.parse_args()
    with selected_pose(args.pose):
        for name in args.objects: run(name)
