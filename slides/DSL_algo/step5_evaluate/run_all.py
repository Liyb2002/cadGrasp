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
    table = [f'# {name}：全部组合的 Step5 BBX 评价', '',
             f'共 {len(groups)} 组、{sum(r["pose_count"] for r in results)} 个组合内姿态。'
             '每组分别用一个 BBX 包住原物体的所有 pose，另一个包住物体加支撑的所有 pose；'
             '图中展示完整三维包围盒，数值仍使用实际工位 XY 占地面积和静态工作摆放，不计运动扫掠。', '',
             '[全部组合对比图](all_groups.png)', '',
             '| 组合 | 原 BBX（mm） | 加支撑 BBX（mm） | 原面积（cm²） | 加支撑面积（cm²） | 额外占地 |',
             '|---|---:|---:|---:|---:|---:|']
    for row in records:
        m = row['metrics']; a = m['object_poses']; b = m['object_and_support_poses']
        aw, ad = a['extents_mm'][:2]; bw, bd = b['extents_mm'][:2]
        table.append(f'| [{row["group"]}]({row["figure"]}) | {aw:.1f} × {ad:.1f} | {bw:.1f} × {bd:.1f} | '
                     f'{a["xy_area_cm2"]:.2f} | {b["xy_area_cm2"]:.2f} | +{m["extra_area_percent"]:.2f}% |')
    table += ['', 'pose1+3 使用历史 co-design shape；pose1+3copied 使用相同历史姿态的新支撑。其他组使用当前 growing 支撑及对应摆放。'
              '以上仅为占地测量，保留原有各组的验收状态，不把评价完成视为新的可行性证明。', '']
    (anchor/'all_groups.md').write_text('\n'.join(table))
    report = dict(complete=True, schema='multipose_bbox_footprint_batch_v1', object=name,
                  group_count=len(groups), pose_placements=sum(r['pose_count'] for r in results),
                  groups=records, metric='Two aggregate XY AABBs per group',
                  visualization='3D orthographic bounding boxes with unchanged XY footprint metrics',
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
