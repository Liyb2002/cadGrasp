"""Publish completed supports directly in each set's step5.3 directory."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import shutil
import textwrap

from PIL import Image, ImageDraw, ImageFont, ImageOps

COOPT = Path(__file__).resolve().parents[1]
ROOT = COOPT.parents[1]
RESULT = COOPT / 'data/results/B'
VISUAL = COOPT / 'output/B'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_sources(label, experiment, include_evaluation):
    stage = RESULT / label / 'step5/step5.3' / experiment
    if not (stage/'data/report.json').exists():
        stage = RESULT / label / 'step5/step5.2' / experiment
    result = json.loads((stage / 'data/report.json').read_text())
    if (not result['complete'] or not result['passed'] or
            result.get('stage_type')=='support_pruning'):
        raise ValueError('A completed final support is required: ' + label)
    source = (VISUAL / label / 'step5.3' if 'published_images' in result else
              VISUAL / label / 'step5/step5.2' / experiment)
    names = ['overview.png', 'all_poses.png'] + [p + '.png' for p in result['poses']]
    images = {name: source / name for name in names}
    if include_evaluation:
        evaluation = RESULT / label / 'evaluation' / experiment / 'report.json'
        record = json.loads(evaluation.read_text())
        if record.get('step53_source',record.get('step52_source')) != str(stage.relative_to(ROOT)):
            raise ValueError('Evaluation uses another final support: ' + label)
        direct = VISUAL / label / 'evaluation/static_occupancy.png'
        images['evaluation.png'] = (direct if record.get('step53_source') and direct.exists() else
            VISUAL / label / 'evaluation' / experiment / 'static_occupancy.png')
    for path in images.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return images


def publish_case_images(label, experiment, include_evaluation=False):
    images = image_sources(label, experiment, include_evaluation)
    destination = VISUAL / label / 'step5.3'
    destination.mkdir(parents=True, exist_ok=True)
    # Only this publisher's files are replaced; experiment archives remain intact.
    for old in destination.glob('pose_*.png'):
        if old.name not in images:
            old.unlink()
    if not include_evaluation:
        (destination / 'evaluation.png').unlink(missing_ok=True)
    records = []
    for name, source in images.items():
        target = destination / name
        if source.resolve()!=target.resolve():
            shutil.copy2(source, target)
        expected = digest(source)
        if digest(target) != expected:
            raise ValueError('Image copy differs: ' + str(target))
        records.append(dict(source=str(source.relative_to(ROOT)),
                            path=str(target.relative_to(ROOT)), sha256=expected))
    return dict(id=label, images=records)


def remove_html():
    removed = []
    for path in sorted(VISUAL.rglob('*.html')):
        removed.append(str(path.relative_to(ROOT)))
        path.unlink()
    return removed


def write_report(experiment, summary):
    rows = summary['cases']
    sums = summary['independent_group_bbox_sums_litres']
    totals = summary['layout_totals']
    comparison = ''
    previous = RESULT / 'data/evaluation_clearance_compact_v1/summary.json'
    if previous.exists():
        old = json.loads(previous.read_text())
        old_cases = {row['id']: row for row in old['cases']}
        if set(old_cases) == {row['id'] for row in rows}:
            old_volume = old['independent_group_bbox_sums_litres']['with_full_support']
            improvement = 100 * (1 - sums['with_full_support'] / old_volume)
            decreased = sum(row['with_full_support']['volume_litres'] <
                            old_cases[row['id']]['with_full_support']['volume_litres'] - 1e-7 for row in rows)
            comparison = (f'与保存的旧 `clearance_compact_v1` 对比：完整 BBX 统计和从 '
                          f'{old_volume:.2f} L 降到 {sums["with_full_support"]:.2f} L，降低 '
                          f'{improvement:.2f}%，{decreased}/{len(rows)} 组减小。'
                          '这是使用已记录暖启动的保存结果对比，不是冷启动算法性能对照。')
    report = RESULT / ('evaluation_' + experiment + '.md')
    csv_path = report.with_suffix('.csv')
    lines = [
        '# 全部 B 集合的 Evaluation 与 pose 数量', '',
        f'批次 `{experiment}`：{len(rows)}/{len(rows)} 组完成，'
        f'{summary["pose_instance_count"]} 个 object pose 实例，'
        f'{totals["support_pose_count"]} 个支撑安装 pose。每组是一件共享支撑实体，任务依次执行。', '',
        '每组图片直接放在 `slides/Co-optimize/output/B/[set]/step5.3/`，'
        '`illegal/` 集合保留其目录前缀。`overview.png` 是完整支撑，'
        '`all_poses.png` 汇总全部任务姿态，`pose_N.png` 是单个物体姿态，'
        '`evaluation.png` 从左到右是无支撑、加 Step4、加完整 Step5.3；不使用 HTML。', '',
        f'独立集合的 BBX 统计和：**{sums["object_static"]:.2f} / '
        f'{sums["with_step4_support"]:.2f} / {sums["with_full_support"]:.2f} L**。'
        f'完整实体材料统计和 **{summary["material_full_sum_cm3"]:.2f} cm³**。'
        'BBX 是空间包围盒；材料体积是实体用量。跨集合统计和不代表同一工作站的总体积。', '',
        comparison, '',
        '无支撑基线保留原任务朝向，将水平包围盒中心对齐、最低点放在 z=0，'
        '取全部依次使用的姿态的最小 XYZ 包围盒。有支撑两列使用保存的真实布局及全部最终安装姿态。'
        'Evaluation 只计最终静态工作位置，不计退出、搬运和机器人运动。', '',
        '支撑 pose 按保存的实际世界位置和朝向分组。每个支撑 pose 服务的物体 pose 数量之和'
        '等于该集合的 object pose 数量；Juxtapose 与原位复用是物体落座分类，不是支撑 pose 数。', '',
        '| Set（打开全部姿态图） | object pose | 支撑 pose | 每个支撑 pose 服务数量 | 无支撑 BBX (L) | Step4 BBX (L) | Step5.3 BBX (L) | 材料 (cm³) | Evaluation 图 |',
        '|---|---:|---:|---|---:|---:|---:|---:|---|',
    ]
    fields = ['set', 'object_pose_count', 'support_pose_count', 'object_pose_counts_per_support_pose',
              'object_pose_membership', 'no_support_bbx_l', 'step4_bbx_l', 'step53_bbx_l',
              'material_cm3', 'juxtaposed_pose_count', 'rotating_reuse_pose_count',
              'all_poses_png', 'evaluation_png']
    with csv_path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            folder = VISUAL / row['id'] / 'step5.3'
            all_url = os.path.relpath(folder / 'all_poses.png', RESULT)
            eval_url = os.path.relpath(folder / 'evaluation.png', RESULT)
            counts = ', '.join(map(str, row['object_pose_counts_per_support_pose']))
            membership = '; '.join('+'.join(p.removeprefix('pose_') for p in s['poses'])
                                   for s in row['fixture_states'])
            values = [row[k]['volume_litres'] for k in
                      ['object_static', 'with_step4_support', 'with_full_support']]
            lines.append(f'| [{row["id"]}]({all_url}) | {row["pose_count"]} | '
                         f'{row["support_pose_count"]} | {counts} | {values[0]:.2f} | '
                         f'{values[1]:.2f} | {values[2]:.2f} | {row["material_full_cm3"]:.2f} | [PNG]({eval_url}) |')
            writer.writerow(dict(set=row['id'], object_pose_count=row['pose_count'],
                                 support_pose_count=row['support_pose_count'],
                                 object_pose_counts_per_support_pose=counts, object_pose_membership=membership,
                                 no_support_bbx_l=values[0], step4_bbx_l=values[1], step53_bbx_l=values[2],
                                 material_cm3=row['material_full_cm3'],
                                 juxtaposed_pose_count=row['juxtaposed_pose_count'],
                                 rotating_reuse_pose_count=row['rotating_reuse_pose_count'],
                                 all_poses_png=all_url, evaluation_png=eval_url))
    lines.extend(['', '每个支撑安装 pose 服务的 object pose 成员：', '',
                  '| Set | 支撑 pose → object pose 成员 | Juxtapose | 原位复用 |',
                  '|---|---|---:|---:|'])
    for row in rows:
        membership = '; '.join(f'S{s["state"]}: ' + ', '.join(s['poses']) for s in row['fixture_states'])
        lines.append(f'| {row["id"]} | {membership} | {row["juxtaposed_pose_count"]} | '
                     f'{row["rotating_reuse_pose_count"]} |')
    lines.extend(['', f'Juxtapose 总数 {totals["juxtaposed_pose_count"]}，'
                  f'原位复用总数 {totals["rotating_reuse_pose_count"]}。'
                  f'{totals["multi_pose_fixture_state_count"]} 个支撑 pose 服务多个物体姿态，'
                  f'{totals["singleton_fixture_state_count"]} 个只服务一个。', '',
                  f'[完整 CSV]({csv_path.name})；'
                  f'[保存的完整 Evaluation JSON](data/evaluation_{experiment}/summary.json)。', '',
                  '本次发布只复制保存的结果图并整理报告，没有重跑受力、搜索或实体 Boolean。'])
    report.write_text('\n'.join(lines) + '\n')


def contact_sheet(experiment, rows):
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 15)
    titlefont = ImageFont.truetype(font_path, 28)
    width, height, cols = 480, 430, 4
    poster = Image.new('RGB', (cols * width, 60 + ((len(rows) + cols - 1) // cols) * height), 'white')
    draw = ImageDraw.Draw(poster)
    draw.text((20, 12), f'B - complete supports for all {len(rows)} sets', font=titlefont, fill='#253b49')
    for index, row in enumerate(rows):
        x, y = (index % cols) * width, 60 + (index // cols) * height
        title = '\n'.join(textwrap.wrap(row['id'], width=46))
        title += f'\nObject poses: {row["pose_count"]} | Support poses: {row["support_pose_count"]}'
        draw.multiline_text((x + 10, y + 5), title, font=font, fill='#253b49', spacing=2)
        with Image.open(VISUAL / row['id'] / 'step5.3/overview.png') as im:
            tile = ImageOps.contain(im.convert('RGB'), (width - 12, height - 85), Image.Resampling.LANCZOS)
        poster.paste(tile, (x + (width - tile.width) // 2, y + 80 + (height - 85 - tile.height) // 2))
    poster_path = VISUAL / ('all_supports_' + experiment + '.png')
    poster.save(poster_path)
    poster.save(VISUAL / 'all_supports.png')
    return poster_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment')
    args = parser.parse_args()
    summary = json.loads((RESULT / 'data' / ('evaluation_' + args.experiment) / 'summary.json').read_text())
    if not summary['complete']:
        raise ValueError('Evaluation batch is incomplete')
    # Check every source before replacing the currently published pictures.
    for row in summary['cases']:
        image_sources(row['id'], args.experiment, True)
    cases = [publish_case_images(row['id'], args.experiment, True) for row in summary['cases']]
    write_report(args.experiment, summary)
    poster = contact_sheet(args.experiment, summary['cases'])
    removed = remove_html()
    system = RESULT / 'data' / ('system_' + args.experiment)
    system.mkdir(parents=True, exist_ok=True)
    manifest = dict(complete=True, experiment=args.experiment, group_count=len(cases),
                    image_count=sum(len(c['images']) for c in cases), cases=cases,
                    poster=str(poster.relative_to(ROOT)), poster_sha256=digest(poster),
                    removed_html=removed, remaining_html=len(list(VISUAL.rglob('*.html'))),
                    force_solver_run=False, geometry_boolean_run=False)
    (system / 'direct_images.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print('DIRECT IMAGES', len(cases), 'sets;', manifest['image_count'], 'PNGs;',
          len(removed), 'HTML pages removed;', poster, flush=True)


if __name__ == '__main__':
    main()
