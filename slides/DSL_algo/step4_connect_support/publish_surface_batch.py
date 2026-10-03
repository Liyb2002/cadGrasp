"""Publish only the current surface-head batch; never substitute old shapes."""
import argparse
import html
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step3_scheculer import contacts as I


def failure_description(report):
    if report['constructed']:
        failed = []
        for row in report['construction'].get('checks', []):
            reasons = []
            if not row.get('coupled_equilibrium_passed', False):
                reasons.append('载荷')
            if not row.get('withdrawal', {}).get('clear', row.get('withdrawal_passed', False)):
                reasons.append('退出')
            if reasons:
                failed.append(row.get('pose', '?')+'：'+'／'.join(reasons))
        return ('完整验收通过' if report['passed'] else '已连接，完整验收未通过'), '；'.join(failed)
    errors = [a.get('error', '') for a in report.get('attempts', [])]
    if any('cannot reach a legal floor' in e for e in errors):
        return '未生成实体', '初始局部身体在已尝试的构造中没有连通的合法脚面；尚未完成整体连接。'
    return '未生成实体', report.get('status', 'unknown')+'；'+(errors[-1] if errors else '')


def publish(name):
    base = I.OUTPUTS/name
    groups = sorted((p for p in base.glob('pose*+*') if (p/'step4').is_dir()),
                    key=lambda p: (len(p.name.split('+')), [int(i) for i in p.name.removeprefix('pose').split('+')]))
    last = groups[-1]/'step4/data'
    cards, rows = [], []
    for group in groups:
        output = group/'step4'
        pipeline = json.loads((output/'data/surface_pipeline.json').read_text())
        relative = lambda path: os.path.relpath(output/path, last)
        current = output/'data/report.json'
        report = I.check_report(current) if current.exists() and pipeline.get('complete') and 'pipeline_error' not in pipeline.get('status', '') else None
        if report is not None:
            assert report['head_model'] == 'zero_thickness_contact_surface'
            status, detail = failure_description(report)
            if not report['step3_passed']:
                detail += ' 本组 Step3 尚未覆盖全部原载荷。'
        else:
            status, detail = pipeline.get('status', '运行中'), pipeline.get('error', '')
        view = ''
        if report is not None and report['constructed'] and pipeline.get('publication_passed'):
            I.check_report(output/'data/codesign_visualization.json')
            I.check_report(output/'data/codesign_replay.json')
            view += '<p><a href="shape.html">旋转查看实体</a> · <a href="shape.obj">OBJ</a></p><a href="shape.png"><img src="shape.png" alt="淡蓝色支撑与灰色物体"></a>'
        if report is not None:
            I.check_report(output/'data/common_object_heads.json')
            if (output/'data/exit_replay.json').exists():
                I.check_report(output/'data/exit_replay.json')
                view += '<p><a href="exit_paths.html">本组所有 pose：支撑、退出路径与碰撞视频</a> · <a href="exit_support.html">旋转查看退出候选支撑</a></p>'
            if (output/'data/exit_directions.json').exists():
                I.check_report(output/'data/exit_directions.json')
                view += '<p><a href="exit_directions.png">退出方向与 C020 冲突图</a> · <a href="exit_directions.html">3D 播放物体退出</a></p>'
            view += '<p><a href="all_heads.html">旋转查看接触面</a></p><a href="all_heads.png"><img src="all_heads.png" alt="同一物体周围的零厚度接触面"></a>'
        body = f'<h2>{group.name} · {html.escape(status)}</h2><p>{html.escape(detail)}</p>'+view
        style = '<style>body{font:16px system-ui;max-width:1700px;margin:28px auto;padding:0 20px;color:#304956}p{line-height:1.65}img{width:100%}a{color:#216c95}section{padding:20px;margin:20px 0;background:#f5f8fa;border-radius:12px}</style>'
        (output/'vis.html').write_text('<!doctype html><meta charset="utf-8">'+style+body)
        for filename in ['shape.html', 'shape.obj', 'shape.png', 'all_heads.html', 'all_heads.png', 'exit_directions.png', 'exit_directions.html', 'exit_paths.html', 'exit_support.html']:
            body = body.replace('"'+filename+'"', '"'+relative(filename)+'"')
        cards.append('<section>'+body+'</section>')
        rows.append(dict(group=group.name, status=status, detail=detail,
                         complete=bool(pipeline.get('complete')),
                         constructed=bool(report and report['constructed']), passed=bool(report and report['passed']),
                         step3=pipeline.get('step3', []), seconds=pipeline.get('seconds'),
                         vis=relative('vis.html'), shape=relative('shape.png') if report and report['constructed'] and pipeline.get('publication_passed') else None))
    counts = f'已处理 {sum(r["complete"] for r in rows)}/{len(rows)} 组；{sum(r["constructed"] for r in rows)} 组生成连接实体，{sum(r["passed"] for r in rows)} 组完整验收通过。'
    introduction = '<h1>零厚度接触面＋所有 pose 下 2 mm 地面间距</h1><p>Step3 各 pose 独立选头；完整接触面均按组内所有地面筛选。Step4 从接触面长出实体，实际材料继续检查地面、碰撞、退出与原载荷。淡蓝色为支撑，灰色为物体。</p><p>'+counts+'</p>'
    (last/'codesign_vis.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'+style+introduction+''.join(cards)+'</html>')
    I.save(last/'surface_results.json', dict(complete=all(json.loads((g/'step4/data/surface_pipeline.json').read_text()).get('complete') for g in groups),
                                         model='zero_thickness_contact_surface', floor_clearance_m=.002, results=rows))
    lines = ['# B：零厚度接触面与 2 mm 全组地面筛选', '', counts, '', '[查看八组新图](codesign_vis.html)', '',
             '| 组合 | 结果 | 原因／检查结果 |', '|---|---|---|']
    lines += [f'| [{r["group"]}]({r["vis"]}) | {r["status"]} | {r["detail"]} |' for r in rows]
    (last/'batch_report.md').write_text('\n'.join(lines)+'\n')
    print(counts, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object', nargs='?', default='B')
    publish(parser.parse_args().object)
