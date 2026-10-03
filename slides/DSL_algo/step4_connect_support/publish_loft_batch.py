"""Publish the latest all-pair run, including explicit failed-search rows."""
import json
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont

from run_loft_batch import OUTPUT, digest
from publish_unified_results import compare_loft_shapes


def publish():
    fontpath = '/System/Library/Fonts/Supplemental/Arial.ttf'
    font = ImageFont.truetype(fontpath, 32)
    small = ImageFont.truetype(fontpath, 23)
    large = ImageFont.truetype(fontpath, 48)
    rows = []
    for pair in sorted(OUTPUT.glob('pose*+*')):
        out = pair / 'step4'
        if not (out / 'latest_run.json').exists():
            continue
        run = json.loads((out / 'latest_run.json').read_text())
        assert run['status'] in ('passed', 'failed'), pair
        reference = json.loads((out / 'reference_report.json').read_text())
        for name, sha in run['references'].items():
            assert digest(out / name) == sha, name
        if run['passed']:
            report = json.loads((out / 'report.json').read_text())
            audit = json.loads((out / 'independent_audit.json').read_text())
            viewer = json.loads((out / 'viewer_check.json').read_text())
            assert report['passed'] and report['solid']['one_solid'] and audit['passed']
            assert run['report_sha256'] == audit['reviewed_report_sha256'] == digest(out / 'report.json')
            assert run['overview_sha256'] == digest(out / 'overview.png')
            assert not viewer['browser_errors']
            assert all(c['passed'] and c['samples'] == 32768 for c in viewer['geometry_checks'])
            for name, sha in report['artifacts'].items():
                assert digest(out / name) == sha, name
            png = Image.open(out / 'overview.png').convert('RGB')
            compare_loft_shapes(out, report, font, small)
            (out / 'failure.png').unlink(missing_ok=True)
        else:
            report = None
            png = Image.new('RGB', (2400, 1150), '#fff9f7')
            draw = ImageDraw.Draw(png)
            draw.text((100, 240), 'NO ACCEPTED FIXTURE IN THIS RUN', font=large, fill='#a63b2e')
            reason = run.get('error', 'Search did not complete').splitlines()[-1]
            for i, line in enumerate(textwrap.wrap(reason, 100)):
                draw.text((100, 350 + i * 46), line, font=font, fill='#573b37')
            draw.text((100, 600), 'Finite candidate search at the reference placement; this does not prove impossibility.',
                      font=small, fill='#735d57')
            png.save(out / 'failure.png')
        rows.append((pair.name, out, run, report, reference, png))

    width = 1800
    rowheights = [925 if row[2]['passed'] else 230 for row in rows]
    batch = Image.new('RGB', (width, 80 + sum(rowheights) + 60), 'white')
    comparison = Image.new('RGB', (2400, 80 + 635 * len(rows) + 60), 'white')
    bd, cd = ImageDraw.Draw(batch), ImageDraw.Draw(comparison)
    bd.text((28, 20), 'NEW RESULTS | Unified whole-loft greedy | All B pose pairs', font=font, fill='#264b51')
    cd.text((28, 20), 'LEFT: REFERENCE                                      RIGHT: NEW WHOLE-LOFT GREEDY', font=font, fill='#264b51')
    manifest = []; y = 80
    for i, (pair, out, run, report, reference, png) in enumerate(rows):
        label = pair.replace('pose', 'Pose ').replace('+', ' + ')
        passed = run['passed']
        status = f'PASS | {report["volume_cm3"]:.2f} cm³ | One connected solid' if passed else 'FAILED SEARCH | No accepted new fixture'
        bd.text((28, y + 14), f'{label} | {status}', font=font,
                fill='#264b51' if passed else '#a63b2e')
        if passed:
            batch.paste(png.resize((1800, 863), Image.Resampling.LANCZOS), (0, y + 60))
        else:
            reason = run.get('error', 'Search did not complete').splitlines()[-1]
            for line, text in enumerate(textwrap.wrap(reason, 120)):
                bd.text((28, y + 78 + line * 34), text, font=small, fill='#735d57')
        y += rowheights[i]
        refstatus = 'PASS' if reference['passed'] else 'FAILED CANDIDATE'
        cd.text((20, 94 + i * 635), f'{label} | Reference {reference["volume_cm3"]:.2f} cm³ | {refstatus}', font=small, fill='#65767c')
        cd.text((1220, 94 + i * 635), f'{label} | NEW | {status}', font=small, fill='#264b51' if passed else '#a63b2e')
        refpng = Image.open(out / 'reference_overview.png').convert('RGB')
        comparison.paste(refpng.resize((1200, 575), Image.Resampling.LANCZOS), (0, 140 + i * 635))
        comparison.paste(png.resize((1200, 575), Image.Resampling.LANCZOS), (1200, 140 + i * 635))
        manifest.append(dict(pair=pair, passed=passed, latest_run_sha256=digest(out / 'latest_run.json'),
                             volume_cm3=run.get('volume_cm3'), external_area_cm2=run.get('external_area_cm2'),
                             reference_volume_cm3=reference['volume_cm3'], reference_passed=reference['passed'],
                             comparison=run.get('comparison'), error=run.get('error')))
    for canvas, draw in ((batch, bd), (comparison, cd)):
        draw.text((25, canvas.height - 40), '6 physical patches / 5 IDs | Original loads only | Reference placements retained', font=small, fill='#65767c')
    destination = rows[0][1]
    batch.save(destination / 'batch.png')
    comparison.save(destination / 'comparison.png')
    (destination / 'batch_summary.json').write_text(json.dumps(dict(results=manifest,
        images={name: digest(destination / name) for name in ('batch.png', 'comparison.png')},
        publisher_sha256=digest(Path(__file__))), indent=2) + '\n')
    print(destination / 'batch.png')


if __name__ == '__main__':
    publish()
