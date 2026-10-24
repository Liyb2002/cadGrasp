"""Reuse the original Juxtapose motion and material sequence, with pink work faces."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image, ImageColor

from whole_task.step1.common import ROOT, sha256, write_json


SOURCE = ROOT / 'slides/Co-optimize/operation_demo/juxtapose'
DATA = ROOT / 'whole_task/data/B/operation_videos/juxtapose'


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('NUMBA_CACHE_DIR', str(DATA / 'numba_cache'))
    geometry = load_source('operation_reference_juxtapose_geometry', SOURCE / 'code/geometry.py')
    old_geometry = sys.modules.get('geometry')
    sys.modules['geometry'] = geometry
    try:
        reference = load_source('operation_reference_juxtapose_render', SOURCE / 'code/render.py')
    finally:
        if old_geometry is None:
            sys.modules.pop('geometry', None)
        else:
            sys.modules['geometry'] = old_geometry
    meshes, saved_report = reference.load()
    masks = {}
    for name, pose in [('host', 'pose_1'), ('guest', 'pose_2')]:
        source = ROOT / 'objects/B/poses' / pose / 'needs.json'
        assert sha256(source) == saved_report['provenance']['inputs'][str(source.relative_to(ROOT))]
        domain = json.loads(source.read_text())
        np.testing.assert_array_equal(domain['geometry']['faces'], meshes[name].faces)
        masks[name] = np.zeros(len(meshes[name].faces), bool)
        masks[name][domain['geometry']['work_face_ids']] = True

    class PinkRenderer(reference.Renderer):
        def frame(self, materials, guest_distance, caption, guest_visible=True):
            pixels = self.base.copy()
            depth = np.full((self.height, self.width), -np.inf)
            for panel in range(self.panel_count):
                is_guest = self.guest_only or panel == 1
                name = 'guest' if is_guest else 'host'
                body = self.meshes[name]
                shift = self.direction * guest_distance if is_guest else np.zeros(3)
                layers = [(mesh, color, np.zeros(3)) for mesh, color in materials if color == 'removed']
                if not is_guest or guest_visible:
                    layers += [(body, 'object', shift)]
                layers += [(mesh, color, np.zeros(3)) for mesh, color in materials if color != 'removed']
                projected, shades = [], []
                for mesh, color, move in layers:
                    if not len(mesh.faces):
                        continue
                    triangles, normals = self.project(mesh, move, panel)
                    rgb = np.tile(np.asarray(ImageColor.getrgb(reference.COLORS[color])), (len(normals), 1))
                    if color == 'object':
                        visible = mesh.face_normals @ self.camera > 0
                        rgb[masks[name][visible]] = [255, 133, 189]
                    brightness = .60 + .40 * np.maximum(normals @ self.light, 0)
                    projected.append(triangles)
                    shades.append(np.asarray(brightness[:, None] * rgb, dtype=np.uint8))
                if projected:
                    reference.raster(np.concatenate(projected), np.concatenate(shades), pixels, depth,
                        int(panel * self.width / self.panel_count) + 5,
                        20 if self.guest_only else 124,
                        int((panel + 1) * self.width / self.panel_count) - 5,
                        self.height - (20 if self.guest_only else 58))
            return pixels

    (DATA / 'data').mkdir(exist_ok=True)
    for filename in ('geometry.npz', 'geometry.json', 'work_surface_checks.json'):
        shutil.copy2(SOURCE / 'vis/data' / filename, DATA / 'data' / filename)
    geometry.OUT = DATA
    reference.Renderer = PinkRenderer
    arguments = sys.argv
    sys.argv = ['juxtapose']
    try:
        reference.main()
    finally:
        sys.argv = arguments
    result = ROOT / 'whole_task/operations/juxtaposition.mp4'
    (DATA / 'pose1_pose2_juxtapose.mp4').replace(result)
    source_animation = json.loads((SOURCE / 'vis/data/animation.json').read_text())
    animation = json.loads((DATA / 'data/animation.json').read_text())
    assert source_animation['timeline'] == animation['timeline']
    metadata = dict(path=str(result), sha256=sha256(result), width=720, height=720,
        source_video=str(SOURCE / 'vis/pose1_pose2_juxtapose.mp4'),
        original_motion_and_material_sequence_preserved=True, working_area_color='pink',
        original_working_face_ids={name: np.flatnonzero(mask).tolist() for name, mask in masks.items()},
        frames=animation['frame_count'], fps=animation['fps'], text_overlays=False,
        support_representation='wrapped_shell', no_base_plate=True, no_vertical_backing=True)
    write_json(DATA / 'video.json', metadata)
    index = DATA.parent / 'videos.json'
    videos = json.loads(index.read_text())
    videos['juxtaposition.mp4'] = metadata
    write_json(index, videos)
    print('JUXTAPOSE PINK PASS', result, flush=True)
    return metadata


if __name__ == '__main__':
    main()
