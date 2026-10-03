"""Render each saved object pose together with the actual connected fixture."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I

RENDER_SOURCE = I.ROOT/'slides/reuse/code/render_cpu.py'
spec = importlib.util.spec_from_file_location('fixture_cpu_renderer', RENDER_SOURCE)
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def font(size):
    return ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', size)


def render(output, renderer):
    report_path = output/'data/report.json'
    report = I.check_report(report_path)
    assert report['schema'] == 'independent_pose_connected_shape_v1' and report['constructed']
    inputs = [report_path, output/'shape.obj', output/'shape.html', output/'data/independent_body/geometry.npz']
    before = I.hashes(inputs)
    body = trimesh.load(output/'shape.obj', force='mesh', process=False)
    text = (output/'shape.html').read_text()
    offset = text.index('const DATA=')+len('const DATA=')
    data, _ = json.JSONDecoder().raw_decode(text[offset:])
    fixture = R.piece(dict(v=body.vertices.ravel(), f=body.faces.ravel()), '#839aab')
    pages, views, artifacts = [], [], []
    with np.load(inputs[-1]) as geometry:
        np.testing.assert_allclose(body.vertices, geometry['vertices_m'], atol=1e-14, rtol=0)
        np.testing.assert_array_equal(body.faces, geometry['faces'])
        for k, pose in enumerate(data['poses']):
            name = pose['name']
            assert name == report['poses'][k]
            basis, shift = geometry['rotations'][k], geometry['local_offsets_m'][k]
            np.testing.assert_allclose(pose['rotation'], basis, atol=1e-14, rtol=0)
            np.testing.assert_allclose(pose['translation'], -basis@shift, atol=1e-14, rtol=0)
            original = np.asarray(pose['object']['vertices']).reshape(-1, 3)
            object_vertices = original@basis+shift
            np.testing.assert_allclose((object_vertices-shift)@basis.T, original, atol=1e-14, rtol=0)
            obj = R.piece(dict(v=object_vertices.ravel(), f=pose['object']['faces']), '#e6a04b', smooth=True)
            points = np.vstack([body.vertices, object_vertices])
            low, high = points.min(0), points.max(0)
            floor = R.box(np.r_[high[:2]-low[:2]+.035, .001],
                          np.r_[(low[:2]+high[:2])/2, -.0007], '#f0f2f3')
            floor['unlit'] = True
            parts = [R.placed(floor), R.placed(fixture), R.placed(obj)]
            full_camera = R.fit(points, [-1., -.45, .8], 1120/750, padding=1.13)
            close_camera = R.fit(object_vertices, [-1., -.45, .8], 650/750, padding=1.45)
            page = Image.new('RGB', (1880, 960), 'white')
            draw = ImageDraw.Draw(page)
            draw.text((40, 25), f'{output.parent.name}  /  Pose {name.split("_")[1]}', font=font(34), fill='#253d4c')
            draw.text((40, 87), 'Complete shape + object', font=font(25), fill='#536b7a')
            draw.text((1190, 87), 'Object / support contact — close-up', font=font(24), fill='#536b7a')
            page.paste(renderer.render(parts, full_camera, 1120, 750, 1.4), (30, 125))
            page.paste(renderer.render(parts, close_camera, 650, 750, 1.4), (1190, 125))
            draw.line((1167, 130, 1167, 875), fill='#e1e6e9', width=2)
            draw.rectangle((42, 902, 64, 924), fill='#839aab')
            draw.text((75, 898), 'Connected shape', font=font(22), fill='#405865')
            draw.rectangle((295, 902, 317, 924), fill='#e6a04b')
            draw.text((328, 898), 'Object', font=font(22), fill='#405865')
            status = 'FIXTURE ACCEPTANCE: PASS' if report['passed'] else 'FIXTURE ACCEPTANCE: NOT PASSED'
            draw.text((970, 898), status, font=font(23), fill='#287754' if report['passed'] else '#a64538')
            filename = f'shape_{name}.png'
            page.save(output/filename)
            pages.append(page); artifacts.append(filename)
            views.append(dict(pose=name, image=filename, object_to_fixture_basis=basis.tolist(),
                object_to_fixture_offset_m=shift.tolist(),
                whole_shape_camera=dict(focus_m=full_camera[0].tolist(), axes=full_camera[1].tolist(), height_m=full_camera[2]),
                closeup_camera=dict(focus_m=close_camera[0].tolist(), axes=close_camera[1].tolist(), height_m=close_camera[2]),
                display_floor_only=True, closeup_crops_context_not_geometry=True))
    # A readable vertical sheet for the group's poses; individual images retain
    # the full-resolution view. No deleted head-only or overview figure returns.
    sheet = Image.new('RGB', (1880, 960*len(pages)), 'white')
    for k, page in enumerate(pages):
        sheet.paste(page, (0, 960*k))
    sheet.save(output/'shape_all_poses.png'); artifacts.append('shape_all_poses.png')
    assert I.hashes(inputs) == before
    result = dict(complete=True, poses=report['poses'], physical_geometry_unchanged=True,
        fixture_acceptance_passed=report['passed'], views=views,
        provenance=dict(inputs=before, code=I.hashes([Path(__file__), RENDER_SOURCE, RENDER_SOURCE.with_name('raster.cpp')])),
        artifacts={f'../{name}':I.sha256(output/name) for name in artifacts})
    I.save(output/'data/pose_shape_images.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('outputs', type=Path, nargs='+')
    args = parser.parse_args()
    renderer = R.Renderer()
    for output in args.outputs:
        result = render(output.resolve(), renderer)
        print(output.parent.name, len(result['poses']), 'pose images saved', flush=True)
