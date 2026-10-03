"""Publish actual five-head results; never reconstruct an old six-patch fallback."""
import argparse
import html
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step1.needs import OUTPUTS
from step3_scheculer import contacts as I


def summary(name):
    folders = sorted((OUTPUTS/name).glob('pose*+*/step4'))
    cards, rows = [], []
    for folder in folders:
        report = json.loads((folder/'report.json').read_text())
        if report.get('physical_head_definition') != 'five_unique_heads_one_shared_patch':
            raise ValueError(f'{folder}: rerun with physical shared-head registration first')
        assignments = report.get('body_design', {}).get('assignments', [])
        cross = sum(a['floor_pose_index'] not in a['active_pose_indices'] for a in assignments)
        checks = report.get('floor_compatibility', {}).get('per_pose', [])
        row = dict(pair=folder.parent.name, passed=report.get('passed', False),
            status=report['status'], particle=report.get('particle'),
            physical_head_count=report['contact_patch_count'],
            original_sample_count_per_pose=32768,
            successful_step3_particles_tried=report.get('successful_step3_particles_tried'),
            body_construction_attempted=report.get('body_construction_attempted', True),
            volume_cm3=report.get('volume_cm3'),
            floor_conflict_counts=[c['violating_sample_count'] for c in checks],
            cross_pose_terminals=cross, terminals=len(assignments))
        rows.append(row)
        base = f'../../{folder.parent.name}/step4/'
        status = '通过全部验收' if row['passed'] else ('固定接触关系与地面冲突' if report.get('infeasible_under_fixed_contact_correspondence') else '构造未通过')
        color = '#24694c' if row['passed'] else '#a44435'
        links = [f'<a href="{base}report.json">验收记录</a>']
        if (folder/'index.html').exists():
            links.insert(0, f'<a href="{base}index.html">五头位置与失败原因</a>' if report.get('diagnostic_only') else f'<a href="{base}index.html">交互模型</a>')
        if row['passed'] and (folder/'fixture_mm.stl').exists():
            links.append(f'<a href="{base}fixture_mm.stl">STL</a>')
        text = f"检查 {row['successful_step3_particles_tried']} 个 Step3 成功粒子；五个物理头、一个共享面。"
        if checks:
            text += ' 原始地面需求越界数：'+'；'.join(f"{c['pose'].replace('pose_', 'Pose ')}：{c['violating_sample_count']:,}/{c['original_sample_count']:,}" for c in checks)+'。'
        if report.get('diagnostic_only'):
            text += ' 图中仅为注册后的头，不是已完成的支架；没有可用 STL。'
        image = f'<img src="{base}overview.png" alt="五个头的注册位置">' if (folder/'index.html').exists() else ''
        conflict = f'<details><summary>查看地面冲突</summary><img src="{base}floor_conflict.png" alt="全部原始地面需求与合法半平面"></details>' if (folder/'floor_conflict.png').exists() else ''
        cards.append(f'<article><h2>{html.escape(row["pair"])} <span style="color:{color}">{status}</span></h2>{image}<p>{text}</p>{conflict}<p>{" · ".join(links)}</p></article>')
        I.save(folder/'case_summary.json', row)
    passed = sum(r['passed'] for r in rows)
    tried = sum(r['successful_step3_particles_tried'] or 0 for r in rows)
    page = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>{html.escape(name)} · Step5 五头重跑</title>
<style>body{{font:16px system-ui;color:#293e4b;background:#f5f7f8;margin:auto;max-width:1500px;padding:28px}}h1{{font-size:28px}}p{{line-height:1.65}}article{{background:white;border:1px solid #d9e1e5;border-radius:12px;padding:18px;margin:22px 0}}h2{{font-size:22px}}h2 span{{font-size:16px;margin-left:20px}}img{{width:100%;display:block}}a{{color:#285e80}}summary{{cursor:pointer}}</style>
<h1>{html.escape(name)} · 五个实体头，一个共享面</h1>
<p>{len(rows)} 组配对、{tried} 个 Step3 成功粒子已重跑；完整支架通过 {passed}/{len(rows)}。共享头只生成一次，两个 pose 的变换保持原完整接触面的对应关系。</p>
<p>地面相容性检查使用每 pose 原有的 32,768 个需求。若需求压力中心落在另一姿态地面所禁止的半平面内，添加身体不能补救，构造在此停止。该结论限于当前固定接触对应关系，不排除重新设计接触或改变问题设定。旧六面模型已从当前结果中移除。</p>
{''.join(cards)}</html>'''
    (folders[0]/'batch.html').write_text(page)
    print(json.dumps(dict(passed=passed, total=len(rows), results=rows), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object', default='B')
    summary(parser.parse_args().object)
