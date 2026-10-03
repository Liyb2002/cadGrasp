"""Evaluate every existing pose group and publish one BBX comparison sheet."""
import argparse
import math
import os
import re
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I
from step5_evaluate.evaluate import evaluate


def group_key(path):
    ids = tuple(map(int,re.findall(r'\d+',path.name)))
    return len(ids), ids, path.name


def run(name='B'):
    root = I.OUTPUTS/name
    groups = sorted([p.parents[2] for p in root.glob('pose*/step4/data/report.json')], key=group_key)
    if not groups:
        raise ValueError(f'No existing Step4 pose groups in {root}')
    results = []
    for group in groups:
        results.append(evaluate(group))
    anchor = next((g for g in groups if g.name == 'pose1+3'), groups[0])/'step5_evaluate'
    columns = min(3, len(groups)); width = 1100; height = 600; header = 64
    rows = math.ceil(len(groups)/columns)
    canvas = Image.new('RGB', (columns*width, rows*(height+header)), 'white')
    ink = ImageDraw.Draw(canvas)
    font_path = '/System/Library/Fonts/Supplemental/Arial.ttf'
    if not Path(font_path).is_file():
        font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 30)
    records = []; inputs = []
    for i, (group, result) in enumerate(zip(groups, results)):
        out = group/'step5_evaluate'
        image_path = out/'bbox.png'; report_path = out/'report.json'
        inputs.extend([image_path, report_path])
        x = (i % columns)*width; y = (i // columns)*(height+header)
        title = group.name
        if result['source_step4_status']['status'] == 'historical_shape_reconstructed_for_requested_walkthrough':
            title += ' (reference poses)'
        ink.text((x+width/2, y+header/2), title, anchor='mm', font=font, fill='#303c43')
        with Image.open(image_path) as pic:
            canvas.paste(pic.resize((width, height), Image.Resampling.LANCZOS), (x, y+header))
        records.append(dict(group=group.name, poses=result['poses'], metrics=result['metrics'],
                            source_step4_status=result['source_step4_status'],
                            figure=os.path.relpath(image_path, anchor), report=os.path.relpath(report_path, anchor)))
    canvas.save(anchor/'all_groups.png')
    table = [f'# {name}: Step5 occupied-space comparison', '',
             f'{len(groups)} groups / {sum(r["pose_count"] for r in results)} installed poses. '
             'Aggregate XYZ box volume and XY area in saved workstation coordinates; motion sweeps are excluded.', '',
             '[All groups](all_groups.png)', '',
             '| Group | Object XYZ (mm) | Supported XYZ (mm) | Object volume (cm3) | Supported volume (cm3) | Object area (cm2) | Supported area (cm2) | Extra XY area |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in records:
        m = row['metrics']; a = m['object_poses']; b = m['object_and_support_poses']
        dimensions=lambda box:' x '.join(f'{v:.1f}' for v in box['extents_mm'])
        table.append(f'| [{row["group"]}]({row["figure"]}) | {dimensions(a)} | {dimensions(b)} | '
                     f'{a["box_volume_cm3"]:.2f} | {b["box_volume_cm3"]:.2f} | '
                     f'{a["xy_area_cm2"]:.2f} | {b["xy_area_cm2"]:.2f} | +{m["extra_area_percent"]:.2f}% |')
    table += ['', 'pose1+3 remains the historical reference. pose1+3copied uses the same archived object poses with new support. '
              'Other groups use current growing models and placements. Measurements preserve all upstream acceptance states; '
              'occupied box volume is distinct from support material volume.', '']
    (anchor/'all_groups.md').write_text('\n'.join(table))
    report = dict(complete=True, schema='multipose_bbox_footprint_batch_v1', object=name,
                  group_count=len(groups), pose_placements=sum(r['pose_count'] for r in results),
                  groups=records, metric='XYZ volume and XY area of two aggregate AABBs per group',
                  visualization='3D orthographic bounding boxes; actual XYZ volume and XY area',
                  source_geometry_changed=False,
                  provenance=dict(inputs=I.hashes(inputs), code=I.hashes([Path(__file__)])),
                  artifacts={p:I.sha256(anchor/p) for p in ('all_groups.png', 'all_groups.md')})
    I.save(anchor/'all_groups.json', report)
    for group in groups:
        I.check_report(group/'step5_evaluate/report.json')
    I.check_report(anchor/'all_groups.json')
    print(f'ALL GROUPS COMPLETE: {len(groups)} groups / {report["pose_placements"]} placements', flush=True)
    print(anchor/'all_groups.png', flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object', default='B')
    run(parser.parse_args().object)
