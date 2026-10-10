"""One PNG of the two actual exported ten-pose supports, using the demo renderer."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/pose_set_search_numba')
from common import *
from PIL import Image, ImageDraw


def main():
    path = CO / 'operation_demo/combined/code/geometry.py'
    spec = importlib.util.spec_from_file_location('pose_set_existing_renderer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    renderer_module = module.R
    results = [
        ('Joint', HERE / 'output/ten_v10/pose1-10/joint'),
        ('Incremental', HERE / 'output/ten_incremental_v15/pose1-10/incremental'),
    ]
    meshes = [C.trimesh.load(folder / 'support.obj', process=False) for _, folder in results]
    reports = [json.loads((folder / 'report.json').read_text()) for _, folder in results]
    points = np.vstack([mesh.vertices for mesh in meshes])
    center = (points.min(0) + points.max(0)) / 2
    azimuths = [-55., 35., 125., 215.]
    elevation = np.arctan(1 / np.sqrt(2))
    cameras = []
    for degrees in azimuths:
        azimuth = np.radians(degrees)
        camera = np.array([np.cos(elevation)*np.cos(azimuth),
                           np.cos(elevation)*np.sin(azimuth), np.sin(elevation)])
        right = np.array([-np.sin(azimuth), np.cos(azimuth), 0.])
        cameras.append((camera, right, np.cross(camera, right)))
    scale = min(min(410 / np.ptp(points @ right), 330 / np.ptp(points @ up))
                for _, right, up in cameras)
    canvas = Image.new('RGB', (1800, 950), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 24), 'B | pose 1–10 | actual exported support',
              font=renderer_module.font(25, True), fill='#273340')
    draw.text((30, 62), 'Objects removed. Four isometric views per method. Same scale.',
              font=renderer_module.font(18), fill='#46505a')
    for row, ((label, folder), mesh, report) in enumerate(zip(results, meshes, reports)):
        slots = dict(initial=mesh, final=mesh, host=mesh, guest=mesh)
        renderer = renderer_module.Renderer(slots, dict(guest_direction_world=[0., 0., 1.]),
                                            width=450, height=360, annotations=False, guest_only=True)
        renderer.center, renderer.scale = center, scale
        y = 116 + row*420
        count = len(set(report['hosts'].values()))
        heading = (f'{label}   |   {report["volume_cm3"]:.1f} cm³ material   |   '
                   f'{1e4*report["maximum_projected_footprint_m2"]:.1f} cm² max projection   |   '
                   f'{count} fixture placement' + ('s' if count != 1 else ''))
        draw.text((30, y), heading, font=renderer_module.font(19, True), fill='#273340')
        for column, (camera, right, up) in enumerate(cameras):
            renderer.camera, renderer.right, renderer.up = camera, right, up
            frame = renderer.frame([(mesh, 'support')], 0., '', guest_visible=False)
            canvas.paste(Image.fromarray(frame), (column*450, y+30))
    draw.text((30, 923), 'Force / exit / work verified. Connectivity, installation and strength not certified.',
              font=renderer_module.font(16), fill='#46505a')
    out = HERE / 'vis'
    out.mkdir(exist_ok=True)
    canvas.save(out / 'ten_pose_supports.png')
    C.save(out / 'preview_sources.json', dict(
        supports={str(folder.relative_to(ROOT)): hashlib.sha256((folder/'support.obj').read_bytes()).hexdigest()
                  for _, folder in results},
        renderer=str((CO/'operation_demo/juxtapose/code/render.py').relative_to(ROOT)),
        viewpoints_azimuth_degrees=azimuths, common_pixels_per_m=scale))
    print(out / 'ten_pose_supports.png')


if __name__ == '__main__':
    main()
