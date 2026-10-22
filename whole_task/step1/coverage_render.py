"""Small rasterized views of complete regions, plus a coverage curve."""
from pathlib import Path
import math
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .common import ROOT, sha256


def render(mesh, masks, transforms, report, output):
    # Share the existing studio lighting without generating animation or paths.
    from whole_task.step5.video import raster as r
    import moderngl

    start = time.perf_counter()
    output = Path(output)
    width, height = 576, 470
    columns = min(7, len(masks))
    rows = math.ceil(len(masks)/columns)
    margin, heading, label_height = 12, 62, 42
    canvas = Image.new('RGB', (columns*width+2*margin, rows*(height+label_height)+heading), 'white')
    draw = ImageDraw.Draw(canvas)
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 22)
    heading_font = ImageFont.truetype(font_path, 28)
    draw.text((margin+12, 12), f'{len(masks)} working areas   /   100% surface coverage',
              font=heading_font, fill=(55, 61, 68))
    ctx = r.context()
    studio = r.Studio(ctx, width, height)
    shader = ctx.program(vertex_shader=r.VERTEX, fragment_shader=r.FRAGMENT)
    shadow_shader = ctx.program(vertex_shader=r.SHADOW_VERTEX, fragment_shader=r.SHADOW_FRAGMENT)
    body = r.Shape(ctx, shader, mesh.vertices, mesh.faces, r.GRAY, dynamic=True,
                   metallic=.03, roughness=.63)
    center = mesh.bounds.mean(axis=0)
    key = np.array([-.65, -.9, 1.45]); key /= np.linalg.norm(key)
    span = float(np.linalg.norm(mesh.extents)*1.15)
    light_vp = r.orthographic(span, span) @ r.view_matrix(key, center)
    for program in [shader, shadow_shader]:
        program['light_projection'].write(np.asarray(light_vp, dtype='f4').T.tobytes())
    shader['shadow_map'].value = 3
    try:
        for i, mask in enumerate(masks):
            # Look from this region's needle side, so all selected first-layer
            # faces can be seen; keep the object geometry in its original frame.
            direction = np.asarray(report['working_areas'][i]['outward_axis_object'])
            direction = direction + np.array([.025, -.015, .015])
            direction /= np.linalg.norm(direction)
            right = np.cross([0, 0, 1], direction); right /= np.linalg.norm(right)
            up = np.cross(direction, right)
            projected = mesh.vertices @ np.stack([right, up], axis=1)
            lo, hi = projected.min(axis=0), projected.max(axis=0)
            target = right*((lo[0]+hi[0])/2)+up*((lo[1]+hi[1])/2)+direction*(center @ direction)
            scale = max(hi[0]-lo[0], (hi[1]-lo[1])*width/height)*1.13
            vp = r.orthographic(scale, scale*height/width) @ r.view_matrix(direction, target)
            shader['view_projection'].write(np.asarray(vp, dtype='f4').T.tobytes())
            shader['eye'].value = tuple(target+direction*4)
            colors = np.tile(r.GRAY, (len(mesh.faces), 1)); colors[mask] = r.PINK
            body.paint(colors)
            ctx.enable(moderngl.DEPTH_TEST)
            studio.shadow_buffer.use(); ctx.viewport = (0, 0, 1536, 1536)
            studio.shadow_buffer.clear(depth=1)
            body.shadow(ctx, shadow_shader, np.eye(4))
            studio.gbuffer.use(); ctx.viewport = (0, 0, *studio.size)
            studio.gbuffer.clear(1, 1, 1, 1, depth=1); studio.shadow.use(3)
            body.draw(shader, np.eye(4))
            tile = Image.frombytes('RGB', (width, height), studio.finish(vp, direction))
            tile = tile.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            x = margin+(i % columns)*width; y = heading+(i//columns)*(height+label_height)
            canvas.paste(tile, (x, y))
            row = report['working_areas'][i]
            draw.text((x+20, y+height+3), f"{row['id']}   {row['area_fraction']*100:.2f}%", font=font, fill=(55, 61, 68))
        canvas.save(output/'working_areas_overview.png')
        device = ctx.info['GL_RENDERER']
    finally:
        ctx.release()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = np.arange(len(masks)+1)
    coverage = np.r_[0., [row['cumulative_coverage_fraction']*100 for row in report['working_areas']]]
    fig, ax = plt.subplots(figsize=(10, 4.5), layout='constrained')
    ax.plot(x, coverage, color='#e963a1', linewidth=2.4, marker='o', markersize=4)
    ax.fill_between(x, coverage, color='#fde3ef')
    ax.set(xlabel='Working areas selected', ylabel='Surface covered (%)',
           xlim=(0, len(masks)+.5), ylim=(0, 103), yticks=np.arange(0, 101, 20))
    ax.set_xticks(sorted(set([0, *range(5, len(masks), 5), len(masks)])))
    ax.spines[['right', 'top']].set_visible(False)
    ax.grid(axis='y', color='#e5e8ec', linewidth=.6)
    for threshold in [.95, .99, 1.]:
        index = int(np.flatnonzero(coverage >= threshold*100-1e-10)[0])
        ax.annotate(f'{threshold*100:g}%: {index}', (index, coverage[index]),
                    xytext=(0, -20 if threshold < 1 else -34), textcoords='offset points',
                    ha='center', fontsize=10, color='#4d5661')
    fig.savefig(output/'coverage_progress.png', dpi=180, facecolor='white')
    plt.close(fig)
    return dict(engine='OpenGL rasterization with existing studio lighting', device=device,
        color='pink', region_views='look_from_each_screening_needle_direction_in_object_frame',
        labels='complete_region_fraction_of_total_surface_not_newly_covered_fraction',
        overview_size=list(canvas.size), elapsed_s=time.perf_counter()-start,
        source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in
                       [Path(__file__), Path(r.__file__)]})
