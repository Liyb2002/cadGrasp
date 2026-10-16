"""Replace current Step5.3 presentation PNGs from the saved current solids."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
import argparse
import importlib.util
import os
import shutil
import sys
import time

from wrap_pipeline import ROOT, COOPT, DEFAULT_EXPERIMENT, read, save, sha
sys.path[:0] = [str(COOPT/'helper_func'), str(COOPT/'step5.3')]
import stage5_common as C
from PIL import Image, ImageDraw, ImageFont


def render_case(task):
    row, directory, cpus = task
    os.sched_setaffinity(0, cpus)
    out = ROOT/row['output']
    report = read(out/'data/report.json')
    protected = {}

    def load(path, expected=None):
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError('Saved model changed: '+str(path))
        protected[C.relative(path)] = digest
        return C.trimesh.load(path, force='mesh', process=False)

    final = load(out/'support.obj', report['support_sha256'])
    initial = load(out/'initial_support.obj', report['initial_support_sha256'])
    states = read(out/'states.json')['states']
    # bbx_trim is the actual end of Step5.2, before spatial block pruning.
    middle = states[0]
    for state in states[1:]:
        if state['operation'] != 'outside_bbx_face':
            break
        middle = state
    trimmed = load(ROOT/middle['mesh'], middle['mesh_sha256'])
    body = load(ROOT/'objects/B/mesh.stl')
    rows = []
    for i, entry in enumerate(report['demands']):
        path = ROOT/entry['arrays']
        protected[C.relative(path)] = sha(path)
        if protected[C.relative(path)] != entry['sha256']:
            raise ValueError('Demand arrays changed')
        with C.np.load(path) as z:
            cloud = z['floor_demands_world_m'].copy()
            rows.append(dict(pose=entry['pose'], color=C.V.COLORS[i],
                fixture_transform=z['T_world_fixture'].copy(),
                object_transform=z['T_world_object'].copy(),
                cloud=cloud, hull=cloud[z['hull_indices']]))
    data = dict(fixture=final, body=body, rows=rows,
                states=[r['fixture_transform'] for r in rows])
    images = Path(directory)/'staging'/row['id']
    images.mkdir(parents=True)
    spec = importlib.util.spec_from_file_location('current_step53_renderer', COOPT/'step5.3/render.py')
    renderer_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer_module)
    renderer = C.V.Renderer()
    renderer_module.reference(renderer, data, images/'overview.png', fixture=final)
    renderer_module.reference(renderer, data, images/'bbx_trim.png', fixture=trimmed)
    tiles = []
    for row_pose in rows:
        tile, _ = C.H.render_pose(renderer,
            C.posed(body, row_pose['object_transform']),
            C.posed(final, row_pose['fixture_transform']), row_pose['cloud'], row_pose['color'],
            images/(row_pose['pose']+'.png'), hull=row_pose['hull'], size=700)
        tiles.append(tile)
    C.V.sheet(tiles, images/'all_poses.png')
    compared = [C.posed(mesh, data['states'][0]) for mesh in (initial, final)]
    camera = C.V.camera_for(C.np.vstack([mesh.vertices for mesh in compared]))
    C.V.sheet([renderer.render([C.V.CPU.placed(C.V.piece(mesh, C.V.BLUE))], camera, 800)
               for mesh in compared], images/'comparison.png')
    return dict(id=row['id'], source=row['output'], support_sha256=report['support_sha256'],
        initial_support_sha256=report['initial_support_sha256'],
        bbx_trim_checkpoint=middle['index'], protected_inputs=protected,
        images={p.name:sha(p) for p in images.glob('*.png')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', default=DEFAULT_EXPERIMENT)
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    ledger = COOPT/'data/results/B/data'/('step5_'+args.experiment)
    summary = read(ledger/'summary.json')
    assert summary['complete'] and summary['step5_failed'] == 0
    directory = ledger/('stage53_images_'+str(time.time_ns()))
    directory.mkdir()
    code = C.freeze(directory)
    # Include the small reference renderer before it is loaded in child processes.
    source = COOPT/'step5.3/render.py'
    target = directory/'executed_sources'/source.relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    code[C.relative(source)] = sha(source)
    save(directory/'render_sources.json', code)
    rows = [r for r in summary['results'] if r['passed']]
    cpus = sorted(os.sched_getaffinity(0))
    groups = [cpus[max(0, len(cpus)-2*args.jobs)+2*i:max(0, len(cpus)-2*args.jobs)+2*i+2]
              or cpus[-2:] for i in range(args.jobs)]
    results = []
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=get_context('spawn')) as pool:
        futures = [pool.submit(render_case, (row, str(directory), groups[i%args.jobs]))
                   for i, row in enumerate(rows)]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print('RENDERED STEP5.3', len(results), '/', len(rows), result['id'], flush=True)
    protected = {p:h for row in results for p,h in row['protected_inputs'].items()}
    for path, digest in protected.items():
        assert sha(ROOT/path) == digest, path
    for path, digest in code.items():
        assert sha(ROOT/path) == digest and sha(directory/'executed_sources'/path) == digest, path
    removed, published = {}, {}
    for case in summary['results']:
        folder = COOPT/'output/B'/case['id']/'step5.3'
        for old in folder.glob('*.png'):
            removed[C.relative(old)] = sha(old)
    save(directory/'removed_old_images.json', removed)
    for row in results:
        for name, digest in row['images'].items():
            path = directory/'staging'/row['id']/name
            with Image.open(path) as image:
                image.verify()
            assert sha(path) == digest
    # Replace derived PNGs only after every current result has rendered successfully.
    for path in removed:
        (ROOT/path).unlink()
    for row in results:
        folder = COOPT/'output/B'/row['id']/'step5.3'
        folder.mkdir(parents=True, exist_ok=True)
        for name, digest in row['images'].items():
            target = folder/name
            (directory/'staging'/row['id']/name).replace(target)
            assert sha(target) == digest
            published[C.relative(target)] = digest
    originals = read(ledger/'protected_originals.json')
    changes = {p:dict(before=originals[p], after=h) for p,h in published.items() if p in originals}
    save(ledger/'authorized_step53_updates.json', dict(complete=True,
        user_requested_old_step53_pngs_deleted_and_replaced=True, changes=changes,
        publication=C.relative(directory/'completion.json')))
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    gallery = Image.new('RGB', (1440, 264*((len(results)+3)//4)), 'white')
    for i, row in enumerate(sorted(results, key=lambda r:r['id'])):
        path = COOPT/'output/B'/row['id']/'step5.3/overview.png'
        with Image.open(path) as image:
            tile = image.convert('RGB').resize((240,240), Image.Resampling.LANCZOS)
        x,y = (i%4)*360, (i//4)*264
        gallery.paste(tile, (x+60,y+24))
        ImageDraw.Draw(gallery).text((x+4,y+5), row['id'], font=font, fill='#296B81')
    gallery.save(directory/'gallery.png')
    save(directory/'completion.json', dict(complete=True, experiment=args.experiment,
        rendered_groups=len(results), original_upstream_failures=summary['upstream_step4_fail'],
        removed_old_pngs=removed, old_png_backup_created=False, published=published,
        protected_inputs=protected, render_sources=C.relative(directory/'render_sources.json'),
        results=results, numerical_models_and_certificates_unchanged=True))
    print('STEP5.3 IMAGES COMPLETE', len(results), len(published), directory, flush=True)


if __name__ == '__main__':
    main()
