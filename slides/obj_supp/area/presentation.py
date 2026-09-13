"""B/pose_2 contact-area comparison using actual surface patches and joint loads.

This presentation compares a small first patch, three times that contact area,
then two and three final baseline patches. It does not rerun greedy selection.
Percentages are explicitly finite-sample coverage under the Step3 contact model.
"""
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'tools'))
import slide_scene as S
from step3_scheculer import contacts as I, floor_support as F, passive_support as U
from step3_scheculer.stage_imports import load_stage
from step2_local_support import circles as P
C = load_stage('score', 'contribution')


def small_patch(domain, original):
    _, polygons = P.eligible_polygons(domain)
    surface = P.SurfaceCircles(domain.mesh, polygons)
    center, seed = original['center_m'], original['center_face']
    target = I.area([original])/3.
    low, high = 0., original['radius_m']
    for _ in range(32):
        radius = (low+high)/2
        clipped, area = surface.at_radius(center, seed, radius)
        if area < target:
            low = radius
        else:
            high = radius
    clipped, area = surface.at_radius(center, seed, high)
    assert abs(area/target-1.) < 1e-5
    triangles, faces = [], []
    for face, polygon in clipped.items():
        tri = P.fan(polygon)
        triangles.extend(tri); faces.extend([face]*len(tri))
    triangles = np.asarray(triangles)
    return dict(original, radius_m=high, triangles_m=triangles,
                source_faces=np.asarray(faces), triangle_areas_m2=P.areas(triangles))


def supply(domain, contacts):
    scale = np.r_[np.ones(3), np.ones(3)/domain.mesh.extents.max()]
    with np.load(S.CASE/'step4_floor_contact/floor_contact.npz') as data:
        pivot = data['original_pivot_m'].copy()
    groups = [U.floor(F.columns(pivot, domain.com), scale)]
    for patch in contacts:
        points = patch['triangles_m'].reshape(-1, 3)
        normals = np.repeat(-domain.mesh.face_normals[patch['source_faces']], 3, axis=0)
        groups.append(U.heads(np.c_[normals, np.cross(points-domain.com, normals)], scale))
    return I.merge_columns(*groups), scale


def main():
    domain = S.load()
    samples = S.samples(domain)
    original = I.read_contacts(S.CASE/'step3_scheculer/final_contacts.npz')
    assert [c['candidate_id'] for c in original] == ['C139', 'C024', 'C011']
    detail_widths = {p['candidate_id']: 2.8*float(p['radius_m']) for p in original}
    small = small_patch(domain, original[0])
    plans = [[small], original[:1], original[:2], original]
    titles = ['One small contact', 'Three times its contact area', 'Two contacts', 'Three contacts']
    colors = [S.ORANGE, S.BLUE, (151, 102, 160)]
    canvas = Image.new('RGB', (2400, 2480), S.PAPER)
    draw = ImageDraw.Draw(canvas)
    S.text(draw, (1200, 62), 'Contact area and contact combinations', 46)
    S.text(draw, (1200, 119), 'B / pose 2  |  The same 32,768 paired loads in every comparison', 28, S.MUTED)
    records = []
    masks = []
    for i, (patches, title) in enumerate(zip(plans, titles)):
        full, scale = supply(domain, patches)
        mask, diagnostics = C.J.classify(full, samples['wrench']*scale)
        # Independently replay representative primal/dual cases in the exact
        # matrix used for classification, as the baseline scorer itself does.
        checks = C.verify_classification(full, samples['wrench']*scale, mask)
        if masks:
            assert not np.any(masks[-1] & ~mask), 'Nested contact sets lost sample coverage'
        masks.append(mask)
        row, column = divmod(i, 2)
        x, y = 30+1200*column, 175+1110*row
        picture, cam, _ = S.render(domain, size=1050,
            patches=[(patch['triangles_m'], color) for patch, color in zip(patches, colors)])
        # Small patches need a locator; numbers identify the matching close-ups.
        ink = ImageDraw.Draw(picture)
        for number, (patch, color) in enumerate(zip(patches, colors), 1):
            point = cam.project(patch['center_m'])[:2]
            ink.ellipse((point[0]-15, point[1]-15, point[0]+15, point[1]+15), outline=color, width=3)
            S.text(ink, point+[25, -18], str(number), 25, color)
        canvas.paste(picture, (x+60, y+50))
        for number, (patch, color) in enumerate(zip(patches, colors), 1):
            normal = domain.mesh.face_normals[patch['center_face']]
            detail_cam = S.Camera(np.asarray(patch['center_m']), S.R.axes(normal),
                                  detail_widths[patch['candidate_id']], 240)
            # A below-contact camera must omit the floor to see the underside.
            # Keep the same local scale for a contact across every comparison.
            detail, _, _ = S.render(domain, cam=detail_cam, ground=False,
                                    patches=[(patch['triangles_m'], color)])
            px, py = x+875, y+160+(number-1)*285
            canvas.paste(detail, (px, py))
            draw.rectangle((px, py, px+239, py+239), outline='#dedede', width=2)
            S.text(draw, (px+120, py+258), f'Contact {number}', 22, color)
        S.text(draw, (x+570, y+18), title, 32)
        percent = 100*float(mask.mean())
        S.text(draw, (x+570, y+1050), f'Sampled joint coverage: {percent:.2f}%', 30)
        records.append(dict(title=title, candidate_ids=[p['candidate_id'] for p in patches],
            radii_m=[float(p['radius_m']) for p in patches], total_area_m2=I.area(patches),
            covered_count=int(mask.sum()), sample_count=len(mask),
            covered_percent=percent, diagnostics=diagnostics, verification=checks))
        print(title, f'{percent:.6f}%', flush=True)
    S.text(draw, (1200, 2420), 'Joint force + moment balance and no uplift; these percentages do not certify a complete support.', 25, S.MUTED)
    canvas.save(HERE/'area_B.png')
    S.record(HERE/'area_B_pose2.json', rows=records,
             area_ratio=I.area(original[:1])/I.area([small]),
             floor_model=F.description(), passive_support=U.description(),
             source_contacts_sha256=I.hashes([S.CASE/'step3_scheculer/final_contacts.npz']),
             legacy_experiment_reused=False, continuous_coverage_claimed=False)
    np.savez_compressed(HERE/'area_B_pose2.npz', covered=np.asarray(masks))
    print(HERE/'area_B.png', flush=True)


if __name__ == '__main__':
    main()
