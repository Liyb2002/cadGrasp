"""Text-free, single-view deletion illustration ending at the exact Step5 OBJ.

This is a geometric explanation, not a replay of the optimizer. The starting
display stock combines the original body with the saved round stock and final
body. Protecting the final body allows every visible operation to remove
material; no contact markers or reconstruction appear.
All derived meshes stay in the movie assets, never in numerical result folders.
"""
from pathlib import Path
import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import time

if hasattr(os, 'sched_getaffinity'):
    task_cpus = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, task_cpus[-4:])

import render01 as R
import render02 as M
from deterministic_base import box
import numpy as np
import trimesh
from PIL import Image

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data/video02_minimal'
MOVIE = HERE.parent / '02_pruning_and_bbx.mp4'
FPS, FRAMES, WIDTH, HEIGHT = 24, 360, 1600, 900
START, STOCK_END, PRUNE_END = .6, 9.1, 13.


def read(path):
    return json.loads(Path(path).read_text())


def mesh_bounds(mesh, transform):
    world = R.transform(mesh.vertices, transform)
    return np.array([world.min(0), world.max(0)])


def freeze():
    paths = {Path(__file__).resolve(), Path(R.__file__).resolve(), Path(M.__file__).resolve(),
             R.V.CPU_PATH, R.V.BASELINE / 'step4_connect_support/translucent_raster.cpp'}
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            path = Path(name).resolve()
            if path.is_relative_to(R.ROOT) and not path.is_relative_to(R.ROOT / '.venv') and path.is_file():
                paths.add(path)
    sources = {}
    for path in paths:
        key = R.relative(path)
        target = DATA / 'executed_sources' / key
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        sources[key] = R.digest(path)
    R.save(DATA / 'sources.json', sources)


def prepare():
    DATA.mkdir(parents=True, exist_ok=True)
    original = read(HERE / 'data/video02/stages.json')
    protected = dict(original['protected_inputs'])
    protected[R.relative(HERE / 'data/video02/stages.json')] = R.digest(HERE / 'data/video02/stages.json')
    rows = original['stages']
    meshes, solids = {}, {}
    for i in (0, 1, 20):
        row = rows[i]
        path = R.ROOT / row['support']
        if R.digest(path) != row['support_sha256']:
            raise ValueError('Saved source changed: ' + str(path))
        mesh = trimesh.load(path, force='mesh', process=False)
        meshes[i] = mesh
        solids[i] = R.solid(mesh)
    ref = np.asarray(original['reference_fixture_transform'])
    # These are visualization stocks, NOT certified optimizer intermediates.
    # Keep the final reinforcement from the beginning, so there is no growth.
    final = solids[20]
    target = solids[1] + final
    current = solids[0] + target
    records = []

    def save(value, phase, source_index=None, cut=None):
        mesh = meshes[20] if source_index == 20 else R.unpack(value)
        if not mesh.is_watertight:
            raise ValueError('Open display stock')
        bounds = mesh_bounds(mesh, ref)
        material = abs(float(mesh.volume)) * 1e6
        if records:
            old = records[-1]
            if material > old['material_cm3'] + 1e-6:
                raise ValueError('Display stock gained material')
            old_bounds = np.asarray(old['bounds_world_m'])
            if np.any(bounds[0] < old_bounds[0] - 1e-9) or np.any(bounds[1] > old_bounds[1] + 1e-9):
                raise ValueError('Display BBX expanded')
        missing = R.volume(final - value)
        if missing > 1e-12:
            raise ValueError('A cut deleted final geometry')
        index = len(records)
        path = DATA / 'meshes' / f'{index:03d}.npz'
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces))
        records.append(dict(index=index, phase=phase, mesh=R.relative(path), mesh_sha256=R.digest(path),
            material_cm3=material, bounds_world_m=bounds.tolist(),
            bbx_l=float(np.prod(bounds[1] - bounds[0]) * 1000),
            protected_final_missing_m3=missing, source_stage=source_index, cut_cell=cut,
            certified_optimizer_intermediate=False,
            exact_final_source_sha256=rows[-1]['support_sha256'] if source_index == 20 else None))

    save(current, 'whole_blue_stock')
    low, high = mesh_bounds(R.unpack(current), ref)
    edges = [np.linspace(low[j] - 1e-6, high[j] + 1e-6, 6) for j in range(3)]
    center, extent = (low + high) / 2, (high - low) / 2
    cells = []
    for ijk in itertools.product(range(5), repeat=3):
        a = np.array([edges[j][ijk[j]] for j in range(3)])
        b = np.array([edges[j][ijk[j] + 1] for j in range(3)])
        score = float(np.linalg.norm(((a + b) / 2 - center) / extent))
        cells.append((-score, ijk, a, b))
    cells.sort(key=lambda row: (row[0], row[1]))
    inverse = np.linalg.inv(ref)
    cutters = [(ijk, R.moved(box(a, b), inverse)) for _, ijk, a, b in cells]
    cut_end = None
    for phase, keep in [('outside_in_cut', target), ('round_stock_cut', final)]:
        surplus = current - keep
        for ijk, cutter in cutters:
            remaining = surplus - cutter
            if R.volume(surplus) - R.volume(remaining) < 1e-11:
                continue
            surplus = remaining
            current = keep + surplus
            save(current, phase, cut=list(ijk))
            print(json.dumps(dict(cut=len(records)-1, phase=phase,
                remaining_cm3=round(records[-1]['material_cm3'], 3))), flush=True)
        if R.volume(surplus) > 1e-10:
            raise ValueError('Exterior cuts did not remove the surplus')
        current = keep
        if phase == 'outside_in_cut':
            records[-1]['source_stage'] = 1
            cut_end = len(records) - 1
    save(final, 'exact_final', source_index=20)
    for path, sha in protected.items():
        if R.digest(R.ROOT / path) != sha:
            raise ValueError('Protected original changed: ' + path)
    R.save(DATA / 'inputs.json', protected)
    R.save(DATA / 'geometry.json', dict(complete=True, records=records, exterior_cut_end_index=cut_end,
        reference_transform=ref.tolist(), protected_inputs=protected,
        final_obj_source=rows[-1]['support'], final_obj_sha256=rows[-1]['support_sha256'],
        starting_stock_definition='union(original Step5 input, first saved round stock, exact final)',
        illustrative_geometry_only=True, optimizer_replay=False, new_physical_certification_claimed=False,
        cuts_are_monotone=True, final_protected_throughout=True,
        numerical_models_modified=False, force_solver_rerun=False))
    freeze()
    print(json.dumps(dict(prepared=len(records), exterior_cuts=cut_end)), flush=True)


class Movie:
    def __init__(self):
        self.geometry = read(DATA / 'geometry.json')
        self.records = self.geometry['records']
        self.ref = np.asarray(self.geometry['reference_transform'])
        self.renderer = R.V.CPU.Renderer()
        first_bounds = np.asarray(self.records[0]['bounds_world_m'])
        points = M.box_vertices(first_bounds)
        focus, basis, _ = R.V.camera_for(points)
        projected = (points - focus) @ basis.T
        size = np.ptp(projected, axis=0)
        self.camera = (focus, basis, max(size[1] / .84, size[0] / (WIDTH / HEIGHT) / .84))
        self.cache = {}
        self.frames = []

    def at(self, t):
        end = self.geometry['exterior_cut_end_index']
        if t < START:
            return 0
        if t < STOCK_END:
            return min(end, 1 + int((t - START) / (STOCK_END - START) * end))
        if t < PRUNE_END:
            count = len(self.records) - end - 1
            return min(len(self.records) - 1, end + 1 + int((t - STOCK_END) / (PRUNE_END - STOCK_END) * count))
        return len(self.records) - 1

    def scene(self, index):
        if index in self.cache:
            return self.cache[index]
        row = self.records[index]
        path = R.ROOT / row['mesh']
        if R.digest(path) != row['mesh_sha256']:
            raise ValueError('Display mesh changed')
        with np.load(path) as arrays:
            mesh = trimesh.Trimesh(arrays['vertices'], arrays['faces'], process=False)
        parts = [R.placed(R.V.piece(mesh, R.V.BLUE), self.ref)]
        original = np.asarray(self.records[0]['bounds_world_m'])
        current = np.asarray(row['bounds_world_m'])
        # Complete boxes, no axes, hulls, markers, labels, or side panels.
        old_box = M.box_piece(original, self.camera, '#D3DBDF', True)
        new_box = M.box_piece(current, self.camera, '#238E7B')
        parts += [R.V.CPU.placed(old_box), R.V.CPU.placed(new_box)]
        image = self.renderer.render(parts, self.camera, WIDTH, HEIGHT, scale=1.5)
        self.cache[index] = image
        return image

    def frame(self, frame):
        index = self.at(frame / FPS)
        row = self.records[index]
        self.frames.append(dict(frame=frame, display_stock_index=index,
            phase=row['phase'], material_cm3=row['material_cm3'], bbx_l=row['bbx_l'],
            mesh_sha256=row['mesh_sha256'], exact_final_source_sha256=row['exact_final_source_sha256'],
            text_draw_calls=0, point_markers=0, side_panels=0, fixed_camera=True))
        return self.scene(index)

    def verify(self):
        for path, sha in self.geometry['protected_inputs'].items():
            if R.digest(R.ROOT / path) != sha:
                raise ValueError('Protected original changed: ' + path)
        for path, sha in read(DATA / 'sources.json').items():
            if R.digest(R.ROOT / path) != sha or R.digest(DATA / 'executed_sources' / path) != sha:
                raise ValueError('Executed render source changed: ' + path)
        with np.load(R.ROOT / self.records[-1]['mesh']) as arrays:
            final = trimesh.load(R.ROOT / self.geometry['final_obj_source'], force='mesh', process=False)
            if not np.array_equal(arrays['vertices'], final.vertices) or not np.array_equal(arrays['faces'], final.faces):
                raise ValueError('Last display mesh differs from the exact final OBJ')
        return dict(passed=True, protected_files=len(self.geometry['protected_inputs']),
            display_stocks=len(self.records), exterior_cuts=self.geometry['exterior_cut_end_index'],
            all_text_removed=True, all_side_panels_removed=True, all_point_markers_removed=True,
            hulls_removed=True, object_overlays_removed=True, single_fixed_view=True,
            blue_support_and_actual_bounding_boxes_only=True, final_geometry_byte_array_exact=True,
            final_obj_sha256=self.geometry['final_obj_sha256'], deletion_illustration_only=True,
            optimizer_replay=False, all_deletion_stocks_nested=True,
            original_physical_models_unchanged=True, first_video_unchanged=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--encode', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    movie = Movie()
    times = [0., 2., 4., 6., 8., 9.3, 11.2, 14.2]
    sheet = Image.new('RGB', (1600, 1800), 'white')
    for i, t in enumerate(times):
        frame = movie.scene(movie.at(t))
        frame.save(DATA / f'preview_{i:02d}.png')
        sheet.paste(frame.resize((800, 450), Image.Resampling.LANCZOS), ((i % 2) * 800, (i // 2) * 450))
    sheet.save(DATA / 'storyboard.png')
    R.save(DATA / 'verification.json', movie.verify())
    if not args.encode:
        return
    staging = DATA / '02_pruning_and_bbx.staging.mp4'
    encoder = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', f'{WIDTH}x{HEIGHT}', '-r', str(FPS), '-i', 'pipe:0', '-an',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-profile:v', 'main', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(staging)], stdin=subprocess.PIPE)
    began = time.monotonic()
    try:
        for i in range(FRAMES):
            encoder.stdin.write(movie.frame(i).tobytes())
            if i % FPS == 0:
                print(json.dumps(dict(frame=i, total=FRAMES, seconds=round(time.monotonic()-began, 1))), flush=True)
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError('Encoding failed')
    except BaseException:
        encoder.kill()
        encoder.wait()
        raise
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
        'format=duration:stream=width,height,nb_frames,r_frame_rate', '-of', 'json', str(staging)]))
    stream = probe['streams'][0]
    if (stream['width'], stream['height'], int(stream['nb_frames'])) != (WIDTH, HEIGHT, FRAMES):
        raise ValueError('Wrong encoding')
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(staging), '-f', 'null', '-'], check=True)
    verification = movie.verify()
    verification.update(frames=FRAMES, duration_seconds=FRAMES/FPS, full_decode_passed=True)
    R.save(DATA / 'frame_checks.json', movie.frames)
    R.save(DATA / 'verification.json', verification)
    # Retain old explanatory metadata, without keeping an extra public movie.
    R.save(DATA / 'previous_version.json', dict(manifest=read(HERE / 'data/video02/manifest.json'),
        reason='Replaced at user request by a text-free single-view blue-support deletion illustration'))
    staging.replace(MOVIE)
    shutil.copy2(DATA / 'storyboard.png', HERE / '02_storyboard.png')
    R.save(DATA / 'manifest.json', dict(complete=True, movie=R.relative(MOVIE), movie_sha256=R.digest(MOVIE),
        frames=FRAMES, fps=FPS, duration_seconds=FRAMES/FPS, width=WIDTH, height=HEIGHT,
        pose_set=R.CASE, text=False, side_panels=False, points=False, hulls=False,
        fixed_camera=True, geometric_deletion_illustration=True, optimizer_replay=False,
        final_obj_sha256=movie.geometry['final_obj_sha256'], numerical_results_modified=False,
        seconds=time.monotonic()-began))
    print(json.dumps(verification), flush=True)


if __name__ == '__main__':
    main()
