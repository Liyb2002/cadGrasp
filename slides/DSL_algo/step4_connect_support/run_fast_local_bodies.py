"""Fixed-layout nearest-floor construction; persistent exclusion cache, no audit.

Run the five saved B layouts without searching poses or terminal menus. A cache
miss is timed as part of construction, never hidden as untimed preparation.
All public case folders contain only overview.png, shape.obj and data/.
"""
import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step1.needs import ROOT, OUTPUTS
from step3_scheculer import contacts as I
from step4_connect_support import build_local_bodies as L
from step4_connect_support.run_local_bodies import original_case, foot_menu, plain
from step4_connect_support.run_greedy import read_case
from step4_connect_support.refresh_shared_geometry_view import write_viewer
from step4_connect_support import head_registration as H

HERE = Path(__file__).resolve().parent


def reference_path(out):
    for path in (out/'reference_report.json', out/'data/history/reference_report.json'):
        if path.exists():
            return path
    raise FileNotFoundError(f'No saved reference layout in {out}')


def recipe(out):
    path = reference_path(out)
    reference = json.loads(path.read_text())
    case = read_case((ROOT/reference['source_schedule']).parent)
    placement = dict(reference['placement'])
    for key in ('bases', 'offsets', 'directions'):
        placement[key] = np.asarray(placement[key])
    # Select the already retained terminal recipe, not a new parameter search.
    kind = reference['parameter_generation']
    if kind == 'original_eight_fixed_terminals':
        fixed = json.loads((HERE/'local_body_case.json').read_text())
        parameters = dict(kind=kind, feet=fixed['feet_xy_m'], assignments=fixed['assignments'])
    else:
        original = original_case() if kind == 'transferred_original_terminal_layout' else None
        parameters = next(p for p in foot_menu(case, placement, original) if p['kind'] == kind)
    return case, placement, parameters, reference


def preserve_previous(out):
    """Keep prior evidence, including failed results, behind the two-file surface."""
    data = out/'data'; history = data/'history'
    history.mkdir(parents=True, exist_ok=True)
    old = history/'before_fast_construction'
    old.mkdir(exist_ok=True)
    for path in list(out.iterdir()):
        if not path.is_file() or path.name == '.DS_Store':
            continue
        target = history/path.name if path.name.startswith('reference_') else old/path.name
        if target.exists():
            if I.sha256(path) == I.sha256(target):
                path.unlink()
                continue
            target = old/(I.sha256(path)[:12]+'_'+path.name)
        shutil.move(str(path), str(target))
    # Existing nearest-floor evidence in pose1+3 also needs to remain historical.
    for path in list(data.iterdir()):
        if path.is_file():
            target = old/path.name
            if target.exists() and I.sha256(target) != I.sha256(path):
                target = old/(I.sha256(path)[:12]+'_'+path.name)
            if target.exists():
                path.unlink()
            else:
                shutil.move(str(path), str(target))
    return data


def geometry_comparison(current, previous):
    """One regression comparison of the regenerated mesh, not a load audit."""
    a, b = L.S.solid(current), L.S.solid(previous)
    extra = abs(float((a-b).volume()))*L.S.SCALE**3
    missing = abs(float((b-a).volume()))*L.S.SCALE**3
    return dict(extra_volume_m3=extra, missing_volume_m3=missing,
                unchanged_within_boolean_tolerance=max(extra, missing) <= 8e-14,
                tolerance_m3=8e-14)


def run_pair(pair, repeat=1, verify=False):
    pipeline_start = time.perf_counter()
    out = pair/'step4'; out.mkdir(parents=True, exist_ok=True)
    before = None
    oldreport = out/'data/report.json'
    if (out/'shape.obj').exists() and oldreport.exists():
        old = json.loads(oldreport.read_text())
        if old.get('body_design', {}).get('initial_floor_policy') == 'nearest':
            before = trimesh.load(out/'shape.obj', force='mesh', process=False)
    records = []
    with tempfile.TemporaryDirectory(prefix='cadgrasp_fast_step5_') as tmp:
        stage = Path(tmp)
        with (stage/'run.log').open('w') as log, redirect_stdout(log):
            for iteration in range(repeat):
                start = time.perf_counter()
                case, placement, parameters, reference = recipe(out)
                registration = H.require_saved_registration(case, placement)
                input_seconds = time.perf_counter()-start
                L.build(stage, parameters['feet'], parameters['assignments'],
                    case=case, placement=placement, floor_policy='nearest',
                    skip_unreachable=parameters['kind'] != 'original_eight_fixed_terminals',
                    verify=verify, allow_failed=True, cache_dir=out/'data/cache')
                seconds = time.perf_counter()-start
                report = json.loads((stage/'report.json').read_text())
                records.append(dict(run=iteration+1, cache_hit=report['withdrawal_cache']['hit'],
                    input_seconds=input_seconds, construction_and_export_seconds=seconds,
                    stages_seconds=report['timings_seconds'],
                    obj_sha256=I.sha256(stage/'fixture.obj'), volume_cm3=report['volume_cm3']))
                if iteration == 0 and repeat > 1:
                    first = trimesh.load(stage/'fixture.obj', force='mesh', process=False)
            current = trimesh.load(stage/'fixture.obj', force='mesh', process=False)
            comparison = geometry_comparison(current, before) if before is not None else None
            repeat_comparison = geometry_comparison(current, first) if repeat > 1 else None
            if repeat_comparison and not repeat_comparison['unchanged_within_boolean_tolerance']:
                raise RuntimeError('Cache replay changed the constructed solid: '+json.dumps(repeat_comparison))
            report.update(schema='fast_nearest_floor_local_bodies_v1',
                placement=plain(placement), parameter_generation=parameters['kind'],
                fixed_layout_reused=True, layout_search_performed=False,
                reference_construction_retained=True, five_physical_shared_heads_claimed=True,
                registration=registration,
                complete_fixture_verified=report['passed'] is True,
                historical_reference_passed=reference['passed'],
                historical_reference_volume_cm3=reference['volume_cm3'],
                previous_nearest_geometry_comparison=comparison,
                repeat_geometry_comparison=repeat_comparison,
                external_area_cm2=float(current.area*1e4), benchmark_runs=records)
            report['provenance']['code'].update(I.hashes([Path(__file__)]))
            I.save(stage/'report.json', report)
            view, _ = json.JSONDecoder().raw_decode((stage/'index.html').read_text().split('const DATA=', 1)[1])
            view['report'] = report
            write_viewer(stage, view)
            start = time.perf_counter()
            subprocess.run(['node', str(HERE/'export_shared_geometry.cjs'), str(stage), '--render-only'],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            render_seconds = time.perf_counter()-start
        report['render_seconds'] = render_seconds
        report['output_layout'] = dict(public_files=['overview.png', 'shape.obj'], supporting_directory='data')
        report['artifacts']['../shape.obj'] = report['artifacts'].pop('fixture.obj')
        report['artifacts']['../overview.png'] = I.sha256(stage/'overview.png')
        I.save(stage/'report.json', report)
        view['report'] = report
        write_viewer(stage, view)
        data = preserve_previous(out)
        for path in stage.iterdir():
            target = out/'shape.obj' if path.name == 'fixture.obj' else out/'overview.png' if path.name == 'overview.png' else data/path.name
            shutil.copy2(path, target)
        I.save(data/'files.json', dict(files={str(p.relative_to(out)):I.sha256(p)
            for p in [out/'shape.obj', out/'overview.png', data/'report.json', data/'geometry.npz', data/'design.json']}))
        (data/'README.md').write_text('当前构造记录见 report.json；退出禁入区缓存在 cache/；历史证据在 history/。\n'
            '默认不运行完整几何/受力验收、独立反力审计或交互界面审计。passed=null 表示未重新验收，不代表失败或通过。\n'
            '时间包含读取输入、缓存未命中时的禁入区构建、身体构造和几何/HTML导出；PNG 渲染单独计时。\n')
    return dict(pair=pair.name, constructed=True, passed=report['passed'],
                volume_cm3=report['volume_cm3'], one_solid=report['solid']['one_solid'],
                runs=records, render_seconds=render_seconds,
                pipeline_seconds=time.perf_counter()-pipeline_start,
                historical_reference_passed=reference['passed'],
                previous_nearest_geometry_comparison=comparison)


def publish_batch(rows, destination):
    from PIL import Image, ImageDraw, ImageFont
    width, rowheight = 1800, 925
    canvas = Image.new('RGB', (width, 90+rowheight*len(rows)), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 26)
    draw.text((24, 24), 'Nearest-floor local bodies | Construction only | Final audit disabled', font=font, fill='#354b5b')
    for i, row in enumerate(rows):
        y = 90+i*rowheight
        if row['constructed']:
            runs = row['runs']
            times = ' / '.join(f'{r["construction_and_export_seconds"]:.2f}s ({"cache" if r["cache_hit"] else "first"})' for r in runs)
            draw.text((24,y), f'{row["pair"]} | {row["volume_cm3"]:.2f} cm3 | One connected solid | {times}', font=font, fill='#354b5b')
            picture = Image.open(OUTPUTS/'B'/row['pair']/'step4/overview.png').convert('RGB')
            canvas.paste(picture.resize((1800,863), Image.Resampling.LANCZOS), (0,y+45))
        else:
            draw.text((24,y), f'{row["pair"]} | Construction failed', font=font, fill='#a63b2e')
            import textwrap
            for j, line in enumerate(textwrap.wrap(row['error'],100)):
                draw.text((24,y+50+j*35),line,font=font,fill='#735d57')
    destination.mkdir(parents=True, exist_ok=True)
    canvas.save(destination/'batch.png')
    I.save(destination/'batch_summary.json', dict(results=rows, independent_audit_performed=False,
        image_sha256=I.sha256(destination/'batch.png')))


def publish_failure(pair, row):
    from PIL import Image, ImageDraw, ImageFont
    import textwrap
    out = pair/'step4'
    data = preserve_previous(out)
    canvas = Image.new('RGB', (2400,1150), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',36)
    draw.text((70,80),pair.name+' | Construction did not complete',font=font,fill='#a63b2e')
    for i,line in enumerate(textwrap.wrap(row['error'],105)):
        draw.text((70,175+55*i),line,font=font,fill='#354b5b')
    draw.text((70,1050),'No new connected model exported. Previous evidence is in data/history/.',font=font,fill='#647782')
    canvas.save(out/'overview.png')
    I.save(data/'report.json',dict(row,
        status='shared_head_registration_failed' if 'registration_checks' in row else 'construction_failed',
        complete=False, complete_fixture_verified=False,
        verification=dict(performed=False,independent_audit_performed=False)))


def failure_record(pair, error):
    row = dict(pair=pair.name, constructed=False, passed=None, error=str(error))
    if isinstance(error, H.SharedHeadRegistrationError):
        row.update(passed=False, registration_checks=error.registration_checks)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair', nargs='+')
    parser.add_argument('--repeat', type=int, default=1, help='Use 2 to measure cold construction and cached replay')
    parser.add_argument('--verify', action='store_true', help='Opt in to full final geometry/load verification')
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error('--repeat must be positive')
    pairs = [OUTPUTS/'B'/p for p in args.pair] if args.pair else sorted((OUTPUTS/'B').glob('pose*+*'))
    rows = []
    for pair in pairs:
        print('START',pair.name,flush=True)
        started = time.perf_counter()
        try:
            row = run_pair(pair, args.repeat, args.verify)
        except Exception as error:
            row = failure_record(pair, error)
            row['elapsed_seconds'] = time.perf_counter()-started
            publish_failure(pair,row)
        rows.append(row)
        print(json.dumps(row),flush=True)
    publish_batch(rows, pairs[0]/'step4/data')
    return 0 if all(r['constructed'] for r in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
