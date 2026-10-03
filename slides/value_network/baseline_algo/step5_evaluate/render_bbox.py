"""Three-dimensional CAD views of the two measured workstation bounding boxes."""
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from step4_connect_support import clean_render as V
from step4_connect_support.publish_compact import CPU

VIEW = np.array([.68, -1., .75])
COLORS = ['#467f9e', '#c38346']
SOURCES = [Path(__file__), Path(V.__file__), Path(CPU.__file__),
           Path(V.__file__).with_name('translucent_raster.cpp')]


def corners(box):
    return np.array(list(product(*zip(box['min_m'], box['max_m']))))


def project(vertices, camera, size):
    focus, basis, span = camera
    v = (vertices-focus) @ basis.T
    return np.c_[size/2+v[:, 0]*size/span, size/2-v[:, 1]*size/span]


def wire_box(picture, box, camera, color):
    """Draw all twelve true 3D box edges; the three rear edges are dashed."""
    xyz = corners(box); xy = project(xyz, camera, picture.width)
    back = int(np.argmin(xyz @ camera[1][2]))
    edges = [(i, i ^ bit) for bit in (1, 2, 4) for i in range(8) if i < (i ^ bit)]
    ink = ImageDraw.Draw(picture)
    for rear in (True, False):
        for i, j in edges:
            if (back in (i, j)) != rear:
                continue
            a, b = xy[i], xy[j]
            if rear:
                length = np.linalg.norm(b-a)
                if length < 1e-9:
                    continue
                for start in np.arange(0, length, 16):
                    ink.line([tuple(a+(b-a)*start/length),
                              tuple(a+(b-a)*min(start+8, length)/length)], fill=color, width=2)
            else:
                ink.line([tuple(a), tuple(b)], fill=color, width=4)


def part(vertices, faces, color, alpha=1., smooth=False):
    data = CPU.piece(dict(v=vertices, f=faces), color, smooth=smooth)
    data['opacity'] = alpha
    return CPU.placed(data)


def draw(output, objects, supports, fixture_faces, metrics):
    renderer = V.Renderer()
    boxes = [metrics['object_poses'], metrics['object_and_support_poses']]
    camera = CPU.fit(corners(boxes[1]), VIEW, 1., padding=1.22)
    canvas = Image.new('RGB', (2200, 1200), 'white')
    ink = ImageDraw.Draw(canvas)
    font_path = '/System/Library/Fonts/Supplemental/Arial.ttf'
    if not Path(font_path).is_file():
        font_path = '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
    title = ImageFont.truetype(font_path, 38)
    label = ImageFont.truetype(font_path, 32)
    small = ImageFont.truetype(font_path, 28)
    ink.text((1100, 30), 'All poses · 3D bounding boxes · same scale',
             font=title, anchor='mt', fill='#303c43')
    object_parts = [part(m.vertices, m.faces, '#8eb3c9', .68, smooth=True) for m in objects]
    support_parts = [part(v, fixture_faces, '#b4beb9') for v in supports]
    for k, box in enumerate(boxes):
        lo, hi = np.asarray(box['min_m']), np.asarray(box['max_m'])
        floor = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
                          [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]]])
        floor[:, 2] -= .00002
        base = part(floor, np.array([[0, 1, 2], [0, 2, 3]]), ['#edf4f8', '#f9f0e6'][k])
        base[0]['unlit'] = True
        parts = [base] + object_parts + (support_parts if k else [])
        picture = renderer.render(parts, camera, 900)
        wire_box(picture, box, camera, COLORS[k])
        x = k*1100
        canvas.paste(picture, (x+100, 170))
        text = 'Object poses' if k == 0 else 'Object + support poses'
        ink.text((x+550, 98), text, font=title, anchor='mt', fill='#303c43')
        ink.text((x+550, 145), f'XY footprint: {box["xy_area_cm2"]:.2f} cm²',
                 font=label, anchor='mt', fill=COLORS[k])
        dimensions = ' × '.join(f'{v:.1f}' for v in box['extents_mm'])
        ink.text((x+550, 1060), f'{dimensions} mm  (X × Y × Z)',
                 font=small, anchor='mt', fill=COLORS[k])
    ink.text((1100, 1140), f'Extra XY footprint: {metrics["extra_area_cm2"]:.2f} cm² '
             f'(+{metrics["extra_area_percent"]:.2f}%)', font=title, anchor='mt', fill='#303c43')
    canvas.save(output)
    return dict(view='3D orthographic', bounding_box_dimensions=3, metric_dimensions=2,
                same_camera_and_scale=True, all_pose_meshes_shown=True,
                object_color='#8eb3c9', object_opacity=.68, support_color='#b4beb9',
                object_box_color=COLORS[0], supported_box_color=COLORS[1],
                camera=dict(focus_m=camera[0].tolist(), basis=camera[1].tolist(), span_m=float(camera[2])),
                image_size=list(canvas.size))
