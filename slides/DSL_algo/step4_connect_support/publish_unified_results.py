"""Assemble existing, audited Step5 PNGs into flat comparison sheets."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[3]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_loft_shapes(out,report,font,small):
    """Identical crop/scale of the fixture-only panel; retain all body pixels."""
    parts=[Image.open(out/name).convert('RGB').crop((1600,65,2400,1115))
           for name in ('reference_overview.png','overview.png')]
    boxes=[ImageOps.invert(p.convert('L')).point(lambda x:255 if x>3 else 0).getbbox() for p in parts]
    box=(max(0,min(b[0] for b in boxes)-25),max(0,min(b[1] for b in boxes)-25),
         min(800,max(b[2] for b in boxes)+25),min(1050,max(b[3] for b in boxes)+25))
    height=round(900*(box[3]-box[1])/(box[2]-box[0]))
    sheet=Image.new('RGB',(1800,height+140),'white');draw=ImageDraw.Draw(sheet)
    comparison=report['comparison']
    labels=[('Reference',comparison['reference_volume_cm3'],comparison['reference_external_area_cm2']),
            ('Unified whole-loft greedy',report['volume_cm3'],report['external_area_cm2'])]
    for i,(part,(label,volume,area)) in enumerate(zip(parts,labels)):
        draw.text((i*900+25,12),label,font=font,fill='#264b51')
        draw.text((i*900+25,52),f'Volume {volume:.2f} cm³  |  Surface {area:.2f} cm²',font=small,fill='#65767c')
        sheet.paste(part.crop(box).resize((900,height),Image.Resampling.LANCZOS),(i*900,90))
    draw.text((25,height+104),'Same heads / placements / original loads · 6 physical patches, 5 IDs · Strength not evaluated',font=small,fill='#65767c')
    sheet.save(out/'shape_comparison.png')


def publish(pairs):
    rows = []
    for pair in pairs:
        out = ROOT/'slides/DSL_algo/output/B'/pair/'step4'
        report = json.loads((out/'report.json').read_text())
        audit = json.loads((out/'independent_audit.json').read_text())
        reference = json.loads((out/'reference_report.json').read_text())
        viewer = json.loads((out/'viewer_check.json').read_text())
        assert report['schema'] in ('unified_material_group_growth_v1', 'unified_material_regular_bodies_v1',
                                    'unified_reference_style_loft_growth_v1')
        assert report['passed'] and report['solid']['one_solid'] and audit['passed']
        assert audit['reviewed_report_sha256'] == digest(out/'report.json')
        for file, expected in report['artifacts'].items():
            assert digest(out/file) == expected, file
        assert not viewer['browser_errors']
        assert all(check['passed'] and check['samples'] == 32768 for check in viewer['geometry_checks'])
        rows.append((pair, out, report, reference, audit))

    fontpath = '/System/Library/Fonts/Supplemental/Arial.ttf'
    font = ImageFont.truetype(fontpath, 30)
    small = ImageFont.truetype(fontpath, 23)
    width = 1800; image_height = round(width*1150/2400); row_height = image_height+58
    batch = Image.new('RGB', (width, row_height*len(rows)+50), 'white')
    comparison = Image.new('RGB', (2400, (575+58)*len(rows)+50), 'white')
    grid_comparison = Image.new('RGB', comparison.size, 'white') if all((r[1]/'grid_overview.png').exists() for r in rows) else None
    bdraw, cdraw = ImageDraw.Draw(batch), ImageDraw.Draw(comparison)
    manifest = []
    for i, (pair, out, report, reference, audit) in enumerate(rows):
        label = pair.replace('pose', 'Pose ').replace('+', ' + ')
        title = ('Unified whole lofts' if report['schema']=='unified_reference_style_loft_growth_v1'
                 else 'Regular bodies' if 'regular_shape' in report['body_design'] else 'Unified growth')
        bdraw.text((28, row_height*i+14), f'{label}   |   {report["volume_cm3"]:.3f} cm³   |   PASS · One connected solid', font=font, fill='#264b51')
        batch.paste(Image.open(out/'overview.png').convert('RGB').resize((width, image_height), Image.Resampling.LANCZOS), (0, row_height*i+58))
        for column, (name, value, title) in enumerate([
            ('reference_overview.png', reference['volume_cm3'], 'Original local bodies'),
            ('overview.png', report['volume_cm3'], title)]):
            x, y = column*1200, i*633
            cdraw.text((x+20, y+15), f'{label} · {title} · {value:.3f} cm³', font=font, fill='#264b51')
            comparison.paste(Image.open(out/name).convert('RGB').resize((1200, 575), Image.Resampling.LANCZOS), (x, y+58))
        if grid_comparison is not None:
            grid_report = json.loads((out/'grid_report.json').read_text())
            gd = ImageDraw.Draw(grid_comparison)
            for column, (name, volume, title) in enumerate([
                ('grid_overview.png', grid_report['volume_cm3'], 'Grid network'),
                ('overview.png', report['volume_cm3'], title)]):
                x, y = column*1200, i*633
                gd.text((x+20, y+15), f'{label} · {title} · {volume:.3f} cm³', font=font, fill='#264b51')
                grid_comparison.paste(Image.open(out/name).convert('RGB').resize((1200, 575), Image.Resampling.LANCZOS), (x, y+58))
        manifest.append(dict(pair=pair, passed=True, volume_cm3=report['volume_cm3'],
            reference_volume_cm3=reference['volume_cm3'], one_solid=True,
            original_loads_per_pose=32768, report_sha256=digest(out/'report.json'),
            audit_sha256=digest(out/'independent_audit.json'), png_sha256=digest(out/'overview.png'),
            reference_png_sha256=digest(out/'reference_overview.png')))
    canvases = [(batch, bdraw), (comparison, cdraw)]
    if grid_comparison is not None: canvases.append((grid_comparison, ImageDraw.Draw(grid_comparison)))
    for canvas, draw in canvases:
        draw.text((25, canvas.height-36), '6 physical contact patches / 5 IDs · Yellow: two separate heads · Same original loads · Strength not evaluated', font=small, fill='#65767c')
    destination = rows[0][1]
    batch.save(destination/'batch.png'); comparison.save(destination/'comparison.png')
    images = ['batch.png', 'comparison.png']
    if grid_comparison is not None:
        grid_comparison.save(destination/'grid_comparison.png'); images.append('grid_comparison.png')
    if len(rows)==1 and rows[0][2]['schema']=='unified_reference_style_loft_growth_v1':
        compare_loft_shapes(destination,rows[0][2],font,small)
        images.append('shape_comparison.png')
    (destination/'batch_summary.json').write_text(json.dumps(dict(results=manifest,
        images={name:digest(destination/name) for name in images},
        publisher_sha256=digest(Path(__file__))), indent=2)+'\n')
    print(destination/'batch.png')
    print(destination/'comparison.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair', nargs='+', default=['pose1+3', 'pose1+6', 'pose2+8'])
    publish(parser.parse_args().pair)
