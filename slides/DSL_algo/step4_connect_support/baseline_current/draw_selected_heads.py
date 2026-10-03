"""Render saved physical heads without rerunning search or constructing a fixture.

Writes heads.png, head_details.png and data/head_preview.json beside an existing
Step4 result. Existing overview images, reports and meshes are never replaced.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step1.needs import ROOT, OUTPUTS
from step2_local_support import geometry as G, render as R
from step3_scheculer import contacts as I
from step4_connect_support.baseline_current import head_registration as H, visual_details as V
from step4_connect_support.baseline_current.fixture_view import local_to_world

PALETTE = ['#2879b9', '#e78b2f', '#339b78', '#c94d58', '#8762ae', '#946849',
           '#cb72a4', '#697984', '#a59b24', '#27a9b0', '#5c8fcb', '#88ab53']
INK = '#25333e'
MUTED = '#65737e'


def rgb(color):
    return np.array([int(color[i:i+2], 16) for i in (1, 3, 5)], float)


def pose_label(pose):
    return pose.replace('pose_', 'Pose ')


def read_case(out):
    report_path = out/'data/report.json'
    report = json.loads(report_path.read_text())
    if report.get('schema') in ('sequential_k_step5_v1', 'sequential_k_step4_v2'):
        from step4_connect_support.baseline_current.run_sequential_k import read_case as read
        case = read((ROOT/report['source_schedule']).parent.parent)
        assert case.source/'schedule.json' == ROOT/report['source_schedule']
    else:
        from step4_connect_support.baseline_current.run_greedy import read_case as read
        from step4_connect_support.baseline_current.run_fast_local_bodies import reference_path
        reference = json.loads(reference_path(out).read_text())
        case = read((ROOT/reference['source_schedule']).parent)
        expected = report['registration_checks']['exact_registration']['registration']['active_head_ids']
        assert [sorted(c['candidate_id'] for c in row) for row in case.groups] == expected
    assert case.pair.resolve() == out.parent.resolve()
    bases, offsets = H.fixed_placements(case.tasks)
    if all(case.groups):
        registered, registration = H.register(case.groups, case.heads, bases, offsets, general_layout=True)
    else:
        # An unfinished search may not have reached later poses. Still show the
        # heads it actually selected; do not weaken the constructor's nonempty
        # active-group requirement or invent contacts for the empty groups.
        unique = {}
        for k, group in enumerate(case.groups):
            for contact, cells in zip(group, case.heads[k]):
                ident = contact['candidate_id']
                patch = contact['triangles_m'].reshape(-1,3)@bases[k]+offsets[k]
                local = [v@bases[k]+offsets[k] for v in cells]
                if ident in unique:
                    head = unique[ident]
                    if max(H.cloud_error(head.contact_points, patch),
                           H.cloud_error(np.concatenate(head.cells), np.concatenate(local))) > H.POSITION_TOL:
                        raise ValueError(f'Shared head {ident} has inconsistent preview geometry')
                    head.active_poses += (k,)
                else:
                    unique[ident] = H.RegisteredHead(ident, k, (k,), local, patch)
        registered = list(unique.values())
        registration = dict(passed=True, physical_head_count=len(registered), complete_active_groups=False,
            scope='Identity of selected heads only; unfinished pose groups remain empty')
    lookup = {head.ident: head for head in registered}
    identifiers = case.schedule['selected_ids']
    assert set(identifiers) == set(lookup)
    heads = []
    for i, ident in enumerate(identifiers):
        head = lookup[ident]
        contact = next(c for c in case.groups[head.pose] if c['candidate_id'] == ident)
        surface = contact['triangles_m']@bases[head.pose]+offsets[head.pose]
        solid = np.concatenate([G.hull_mesh(cell).triangles for cell in head.cells])
        normal = case.tasks[head.pose].domain.mesh.face_normals[contact['center_face']]@bases[head.pose]
        normal /= np.linalg.norm(normal)
        heads.append(dict(id=ident, label=f'H{i+1}', color=PALETTE[i % len(PALETTE)],
                          active_poses=list(head.active_poses), surface=surface, solid=solid,
                          center=surface.reshape(-1, 3).mean(0), normal=normal))
    return case, heads, bases, offsets, registration, report_path


def head_arrays(heads, basis=None, offset=None):
    triangles, colors, overlays = [], [], []
    count = 0
    for head in heads:
        solid = head['solid']
        # Contact triangles arrive with the object's outward normal; reverse it
        # for the inward-facing boundary of the actual support head.
        surface = head['surface'][:, [0, 2, 1]]
        if basis is not None:
            solid = local_to_world(solid, basis, offset)
            surface = local_to_world(surface, basis, offset)
        triangles.extend([solid, surface])
        color = rgb(head['color'])
        colors.extend([np.tile(color*.78, (len(solid), 1)),
                       np.tile(color, (len(surface), 1))])
        overlays.extend(range(count+len(solid), count+len(solid)+len(surface)))
        count += len(solid)+len(surface)
    return np.concatenate(triangles), np.concatenate(colors), overlays, []


def label_heads(image, heads, view, basis, offset):
    ink = ImageDraw.Draw(image)
    focus, width, camera = view
    size = image.width
    used = []
    for head in heads:
        point = local_to_world(head['center'], basis, offset)
        x, y, _ = R.project(point, focus, camera, width, size)
        choices = [(x+35, y-24), (x-78, y-24), (x+30, y+28), (x-78, y+28), (x+35, y-65)]
        def box(q):
            tx, ty = np.clip(q[0], 8, size-72), np.clip(q[1], 8, size-38)
            return (tx, ty, tx+62, ty+32)
        def overlaps(a, b):
            return a[0] < b[2]+5 and a[2]+5 > b[0] and a[1] < b[3]+5 and a[3]+5 > b[1]
        rect = min((box(q) for q in choices), key=lambda a: sum(overlaps(a, b) for b in used))
        used.append(rect)
        ink.line([(x, y), ((rect[0]+rect[2])/2, (rect[1]+rect[3])/2)], fill=head['color'], width=2)
        ink.ellipse((x-4, y-4, x+4, y+4), fill=head['color'], outline='white', width=1)
        ink.rounded_rectangle(rect, radius=8, fill='white', outline=head['color'], width=2)
        text = head['label']+('*' if len(head['active_poses']) > 1 else '')
        ink.text(((rect[0]+rect[2])/2, (rect[1]+rect[3])/2), text,
                 anchor='mm', font=R.font(19), fill=INK)


def draw_layout(case, heads, bases, offsets, path):
    cols = min(2, len(case.poses))
    rows = math.ceil(len(case.poses)/cols)
    card_w, card_h, gap = 770, 870, 24
    page = Image.new('RGB', (cols*card_w+(cols+1)*gap, rows*card_h+(rows+1)*gap+125), '#f3f5f7')
    ink = ImageDraw.Draw(page)
    ink.text((gap, 24), f'{case.pair.name}  |  Selected heads', font=R.font(36), fill=INK)
    ink.text((gap, 78), 'Active heads in each pose. Same color / H number = same physical head.  * = shared.',
             font=R.font(21), fill=MUTED)
    ink.text((gap, 106), 'Translucent object for visibility; original head geometry. Heads only, no connecting body.',
             font=R.font(20), fill=MUTED)
    for k, task in enumerate(case.tasks):
        x = gap+(k % cols)*(card_w+gap)
        y = 150+(k//cols)*(card_h+gap)
        ink.rounded_rectangle((x, y, x+card_w, y+card_h), radius=18, fill='white')
        active = [h for h in heads if k in h['active_poses']]
        ink.text((x+25, y+21), f'{pose_label(case.poses[k])}  /  {len(active)} active heads', font=R.font(30), fill=INK)
        points = np.vstack([task.domain.mesh.vertices]+[
            local_to_world(h['solid'].reshape(-1, 3), bases[k], offsets[k]) for h in active])
        camera = V.camera(task.domain, [], points=points)
        size = 710
        # Use a neutral translucent object, and explicitly overlay the real head
        # solids to keep heads on the back of the workpiece visible.
        object_image, _ = V.render((task.domain.mesh.triangles,
            np.tile(np.array([153., 163., 173.]), (len(task.domain.mesh.faces), 1)), [], []), camera, size)
        picture = Image.blend(Image.new('RGB', (size, size), 'white'), object_image, .28)
        if active:
            front, ids = V.render(head_arrays(active, bases[k], offsets[k]), camera, size)
            picture.paste(front, mask=Image.fromarray(np.uint8(ids >= 0)*255))
            label_heads(picture, active, camera, bases[k], offsets[k])
        page.paste(picture, (x+30, y+69))
        labels = '   '.join(h['label']+('*' if len(h['active_poses']) > 1 else '') for h in active) or 'No heads selected for this pose'
        ink.text((x+25, y+790), labels, font=R.font(24), fill=INK)
        covered = case.schedule.get('covered_counts')
        if covered is not None:
            ink.text((x+25, y+831), f'Step3 load coverage: {covered[k]:,} / 32,768', font=R.font(18), fill=MUTED)
    page.save(path)


def scale_bar(picture, width_m):
    ink = ImageDraw.Draw(picture)
    target_mm = width_m*1000*.23
    unit = 10**math.floor(math.log10(target_mm))
    mm = max(v*unit for v in (1, 2, 5) if v*unit <= target_mm)
    x, y = 20, picture.height-35
    end = x+mm/1000/width_m*picture.width
    ink.line((x, y, end, y), fill=INK, width=3)
    for xx in (x, end):
        ink.line((xx, y-4, xx, y+4), fill=INK, width=2)
    ink.text((x, y+6), f'{mm:g} mm', font=R.font(16), fill=MUTED)


def draw_details(case, heads, path):
    if not heads:
        page = Image.new('RGB', (1600, 400), 'white')
        ink = ImageDraw.Draw(page)
        ink.text((35, 40), case.pair.name, font=R.font(38), fill=INK)
        ink.text((35, 130), 'No heads selected; Step3 did not produce a complete contact layout.', font=R.font(28), fill=INK)
        page.save(path)
        return []
    cols = min(3, len(heads))
    rows = math.ceil(len(heads)/cols)
    card_w, card_h, gap = 720, 568, 24
    page = Image.new('RGB', (cols*card_w+(cols+1)*gap, rows*card_h+(rows+1)*gap+110), '#f3f5f7')
    ink = ImageDraw.Draw(page)
    ink.text((gap, 22), f'{case.pair.name}  |  {len(heads)} unique physical heads', font=R.font(37), fill=INK)
    ink.text((gap, 76), 'Each head appears once. Left: contact face. Right: back / thickness. Shared heads list every active pose.',
             font=R.font(22), fill=MUTED)
    measurements = []
    for i, head in enumerate(heads):
        x = gap+(i % cols)*(card_w+gap)
        y = 130+(i//cols)*(card_h+gap)
        ink.rounded_rectangle((x, y, x+card_w, y+card_h), radius=18, fill='white')
        ink.rounded_rectangle((x+20, y+20, x+101, y+61), radius=9, fill=head['color'])
        ink.text((x+60, y+40), head['label'], anchor='mm', font=R.font(26), fill='white')
        ink.text((x+115, y+25), head['id'], font=R.font(27), fill=INK)
        used = ' + '.join(case.poses[k].split('_')[1] for k in head['active_poses'])
        subtitle = f'SHARED: poses {used}' if len(head['active_poses']) > 1 else f'Pose {used} only'
        ink.text((x+23, y+79), subtitle, font=R.font(22), fill=MUTED)
        normal = head['normal']
        tangent = R.axes(normal)[0]
        binormal = np.cross(normal, tangent)
        points = head['solid'].reshape(-1, 3)
        data = head_arrays([head])
        views = [R.axes(-normal+.52*tangent+.28*binormal), R.axes(normal+.90*tangent+.4*binormal)]
        fitted = [V.fit(points, view, margin=1.38) for view in views]
        width = max(v[1] for v in fitted)
        for j, fit in enumerate(fitted):
            camera = (fit[0], width, fit[2])
            picture, _ = V.render(data, camera, 330)
            scale_bar(picture, width)
            page.paste(picture, (x+20+j*350, y+125))
            ink.text((x+185+j*350, y+119), ['Contact face', 'Back / thickness'][j],
                     anchor='mt', font=R.font(17), fill=MUTED)
        local = points@np.stack([tangent, binormal, normal]).T
        extents = np.ptp(local, axis=0)*1000
        ink.text((x+23, y+477), 'Local size: '+' x '.join(f'{v:.1f}' for v in extents)+' mm', font=R.font(21), fill=INK)
        ink.text((x+23, y+518), 'Saved head geometry; dimensions in millimeters.', font=R.font(18), fill=MUTED)
        measurements.append(dict(id=head['id'], label=head['label'], color=head['color'],
            active_poses=[case.poses[k] for k in head['active_poses']],
            local_axes=[tangent.tolist(), binormal.tolist(), normal.tolist()], extents_mm=extents.tolist()))
    page.save(path)
    return measurements


def draw(out):
    # Snapshot only existing top-level results and the verdict. These are user
    # artifacts; a head illustration must never replace them.
    protected = [p for p in out.iterdir() if p.is_file() and p.name not in {'heads.png', 'head_details.png'}]
    protected += [out/'data/report.json']
    previous = {p: I.sha256(p) for p in protected}
    case, heads, bases, offsets, registration, report_path = read_case(out)
    draw_layout(case, heads, bases, offsets, out/'heads.png')
    measurements = draw_details(case, heads, out/'head_details.png')
    assert all(I.sha256(p) == digest for p, digest in previous.items())
    inputs = list(dict.fromkeys(case.paths+[report_path]))
    I.save(out/'data/head_preview.json', dict(kind='original_selected_heads_only',
        poses=case.poses, physical_head_count=len(heads), source_schedule=str((case.source/'schedule.json').relative_to(ROOT)),
        active_head_ids=[[c['candidate_id'] for c in group] for group in case.groups],
        shared_registration=registration, heads=measurements,
        constructed_fixture=False, contact_or_solid_geometry_changed=False,
        original_result_files_preserved=True,
        provenance=dict(inputs=I.hashes(inputs), code=I.hashes([Path(__file__), Path(R.__file__), Path(G.__file__), Path(H.__file__)])),
        artifacts={name:I.sha256(out/name) for name in ('heads.png', 'head_details.png')}))
    print(f'{case.pair.name}: {len(heads)} heads -> heads.png, head_details.png', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object', default='B')
    parser.add_argument('--pair', action='append', help='Existing group directory; repeatable.')
    args = parser.parse_args()
    root = OUTPUTS/args.object
    outputs = [root/name/'step4' for name in args.pair] if args.pair else [p.parent.parent for p in sorted(root.glob('*/step4/data/report.json'))]
    if not outputs:
        raise FileNotFoundError('No existing Step5 results found')
    for out in outputs:
        draw(out)


if __name__ == '__main__':
    main()
