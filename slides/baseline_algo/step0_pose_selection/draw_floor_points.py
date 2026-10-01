"""One 3D sheet: filled demand hulls split by each target floor, for display only."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step2_local_support import render as R
from step3_scheculer import contacts as I
from step0_pose_selection import floor_points as C

FONT = '/System/Library/Fonts/STHeiti Light.ttc'
IMAGE = 'floor_point_conflicts.png'


def text(ink, label, at, size=24, color='#394b46', anchor='mm'):
    ink.text(at, label, font=ImageFont.truetype(FONT, size), fill=color, anchor=anchor)


def clip_at_floor(polygon, above=True):
    """Clip a planar display polygon by z=0, without moving its vertices."""
    polygon = np.asarray(polygon, float).reshape(-1, 3)
    result = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        va, vb = (a[2], b[2]) if above else (-a[2], -b[2])
        if va >= 0:
            result.append(a)
        if (va < 0 < vb) or (vb < 0 < va):
            result.append(a+(b-a)*va/(va-vb))
    return np.asarray(result).reshape(-1, 3)


def area_vector(polygon):
    if len(polygon) < 3:
        return np.zeros(3)
    p = polygon-polygon[0]
    return np.cross(p, np.roll(p, -1, axis=0)).sum(axis=0)/2


def panel(vertices, faces, frames, clouds, result, target, size=900):
    mesh = vertices@frames[target, :3, :3].T+frames[target, :3, 3]
    mapped = [C.map_points(C.ground_points(xy), frames[i], frames[target])
              for i, xy in enumerate(clouds)]
    bad = [p[result.heights[i][:, target] < -C.TOL] for i, p in enumerate(mapped)]
    red = np.concatenate(bad)
    # Hulls enclose ALL original samples. They are an illustration, not a new
    # requirement for solid material or a replacement for the pointwise check.
    hulls = [points[ConvexHull(xy).vertices] for points, xy in zip(mapped, clouds)]
    parts = []
    for i, hull in enumerate(hulls):
        if i == target:
            parts.append((i, hull, np.empty((0, 3))))
        elif len(bad[i]):
            parts.append((i, clip_at_floor(hull), clip_at_floor(hull, above=False)))
        else:
            parts.append((i, hull, np.empty((0, 3))))
    center = mesh.mean(axis=0)
    delta = red.mean(axis=0)-center if len(red) else np.array([1., -1., 0.])
    direction = delta[:2]/max(np.linalg.norm(delta[:2]), 1e-12)
    # Avoid viewing the red demand face edge-on. Keep world vertical upright.
    emphasized = [area_vector(below) for _, _, below in parts if len(below) >= 3]
    if not emphasized:
        emphasized = [area_vector(h) for h in hulls]
    normals = np.array([a/max(np.linalg.norm(a), 1e-20) for a in emphasized])
    candidates = [np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])
                  for e in (.38, .55) for a in np.linspace(-np.pi, np.pi, 48, endpoint=False)]
    view = max(candidates, key=lambda v: np.abs(normals@v).mean()+.12*(v[:2]@direction))
    basis = R.axes(view)
    all_points = np.vstack([mesh, *mapped])
    span = max(float(np.ptp(all_points, axis=0).max()), .001)
    low = all_points[:, :2].min(axis=0)-span*.06
    high = all_points[:, :2].max(axis=0)+span*.06
    ground = np.array([[low[0], low[1], 0], [high[0], low[1], 0],
                       [high[0], high[1], 0], [low[0], high[1], 0]])
    limits = np.vstack([all_points, ground])@basis.T
    lo, hi = limits.min(axis=0), limits.max(axis=0)
    focus = ((lo+hi)/2)@basis
    width = 1.17*max(hi[0]-lo[0], hi[1]-lo[1])
    project = lambda p: R.project(np.asarray(p), focus, basis, width, size)
    picture = Image.new('RGB', (size, size), 'white')
    ink = ImageDraw.Draw(picture)
    outline = [tuple(p[:2]) for p in project(ground)]
    ink.polygon(outline, fill='#edf0ef', outline='#bcc6c1')
    object_image, face_ids = R.raster(mesh[faces], np.tile([182., 191., 187.], (len(faces), 1)),
                                    focus, basis, width, size)
    picture.paste(object_image, mask=Image.fromarray((face_ids >= 0).astype('uint8')*255))
    # X-ray faces show their complete extent through the object and ground.
    # Red is exactly the below-floor portion; no individual samples are drawn.
    def filled_face(polygon, fill, edge):
        if len(polygon) < 3:
            return
        overlay = Image.new('RGBA', picture.size)
        painter = ImageDraw.Draw(overlay)
        projected = [tuple(p[:2]) for p in project(polygon)]
        painter.polygon(projected, fill=fill)
        painter.line(projected+[projected[0]], fill=edge, width=3)
        picture.paste(overlay, (0, 0), overlay)
    for i, above, _ in sorted(parts, key=lambda p: hulls[p[0]].mean(axis=0)@basis[2]):
        filled_face(above, (64, 133, 182, 125) if i == target else (131, 177, 162, 80),
                    '#367fac' if i == target else '#7caa99')
    for _, _, below in parts:
        filled_face(below, (219, 75, 68, 190), '#ac3937')
    ink = ImageDraw.Draw(picture)
    # Each red face names its source pose, so two different demands remain legible.
    placed = []
    for i, _, below in parts:
        if len(below) < 3:
            continue
        anchor = project([below.mean(axis=0)])[0, :2]
        tx = float(np.clip(anchor[0]+(135 if anchor[0] > size/2 else -135), 115, size-115))
        ty = float(np.clip(anchor[1]+45, 35, size-35))
        for old_x, old_y in placed:
            if abs(tx-old_x) < 220 and abs(ty-old_y) < 48:
                ty = min(size-30, ty+52)
        ink.line([tuple(anchor), (tx, ty)], fill='#a33c38', width=2)
        label = result.rows[i]['pose'].replace('pose_', 'Pose ')+' 需求面'
        font = ImageFont.truetype(FONT, 24)
        box = ink.textbbox((tx, ty), label, font=font, anchor='mm')
        ink.rounded_rectangle((box[0]-7, box[1]-5, box[2]+7, box[3]+5), radius=4, fill='white')
        text(ink, label, (tx, ty), 24, '#a33c38')
        placed.append((tx, ty))
    if len(red):
        worst = red[np.argmin(red[:, 2])]
        on_floor = worst.copy(); on_floor[2] = 0
        bottom, top = project([worst, on_floor])[:, :2]
        for t in np.arange(0., 1., .12):
            a, b = bottom+(top-bottom)*t, bottom+(top-bottom)*min(t+.065, 1.)
            ink.line([tuple(a), tuple(b)], fill='#962f31', width=2)
        ink.ellipse((bottom[0]-4, bottom[1]-4, bottom[0]+4, bottom[1]+4), fill='#962f31')
    return picture


def draw(name, poses, vertices, faces, frames, clouds, result, folder):
    n = len(poses); columns = min(3, n) if n != 4 else 2
    rows = int(np.ceil(n/columns)); cell_w, cell_h = 960, max(1120, 1040+34*(n-1))
    sheet = Image.new('RGB', (columns*cell_w, rows*cell_h+170), 'white')
    ink = ImageDraw.Draw(sheet)
    text(ink, name+' · 完整地面需求面：换 pose 后，哪一片穿过地面', (sheet.width/2, 40), 32)
    panels = []
    for j, pose in enumerate(poses):
        x, y = (j % columns)*cell_w, (j//columns)*cell_h+75
        picture = panel(vertices, faces, frames, clouds, result, j)
        sheet.paste(picture, (x+30, y+45))
        text(ink, pose.replace('pose_', 'Pose ')+' 的地面', (x+cell_w/2, y+24), 30)
        rejected = []
        for row in result.rows:
            if row['pose'] == pose:
                continue
            pair = next(p for p in row['per_other_pose'] if p['other_pose'] == pose)
            if pair['violating_sample_count']:
                rejected.append(dict(source_pose=row['pose'], **pair))
        for k, pair in enumerate(rejected):
            label = (pair['source_pose'].replace('pose_', 'Pose ')+
                     f" 需求面穿地；最低 {pair['minimum_height_m']*1000:.2f} mm")
            text(ink, label, (x+cell_w/2, y+980+k*34), 24, '#b0423f')
        if not rejected:
            text(ink, '其他 pose 的需求面都未穿过此地面', (x+cell_w/2, y+980), 25, '#457965')
        panels.append(dict(floor_pose=pose, rejected_sources=rejected))
    text(ink, '蓝面：本 pose 的需求   浅绿面：其他 pose 未穿地的部分   红面：穿地部分',
         (sheet.width/2, sheet.height-65), 25)
    text(ink, '完整凸包面由全部原始撒点围成，透视显示；灰色为地面，横向无限延伸。',
         (sheet.width/2, sheet.height-25), 23, '#7a8982')
    output = Path(folder)/IMAGE
    sheet.save(output)
    return dict(image=IMAGE, panels=panels, legal_region_drawn=False,
                ground_extent_is_display_only=True, filled_demand_hulls=True,
                hull_uses_all_original_samples=True, faces_drawn_in_xray=True,
                individual_points_drawn=False, below_floor_faces_clipped_at_z_m=0.,
                renderer_sources=I.hashes([Path(__file__), Path(R.__file__)]))


def redraw(folder):
    folder = Path(folder)
    report = json.loads((folder/'report.json').read_text())
    with np.load(folder/'data/scene.npz') as data:
        vertices, faces, frames = (data[key].copy() for key in
            ('object_vertices_m', 'object_faces', 'T_world_mesh'))
    clouds = []
    for pose in report['poses']:
        with np.load(folder/f'data/floor_contact_{pose}.npz') as data:
            clouds.append(data['floor_demands_xy_m'].copy())
    result = C.check_floor_points(clouds, frames, report['poses'])
    if result.rows != report['step0_2']['per_pose']:
        raise ValueError('Saved report differs from current floor-point check; rebuild Step0')
    report['presentation'] = draw(report['object'], report['poses'], vertices, faces,
                                  frames, clouds, result, folder)
    report['artifacts'][IMAGE] = I.sha256(folder/IMAGE)
    renderer_key = str(Path(__file__).resolve().relative_to(I.ROOT))
    old_hash = report['provenance']['code'].get(renderer_key)
    new_hash = I.sha256(Path(__file__))
    if old_hash and old_hash != new_hash:
        report.setdefault('presentation_history', []).append(dict(renderer_sha256=old_hash,
            replaced_with='filled demand hull faces; numerical results unchanged'))
    report['provenance']['code'][renderer_key] = new_hash
    I.save(folder/'report.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object', default='B')
    parser.add_argument('--group', required=True)
    args = parser.parse_args()
    if Path(args.group).name != args.group:
        parser.error('Use an existing group directory name')
    redraw(I.OUTPUTS/args.object/args.group/'step0_pose_selection')
