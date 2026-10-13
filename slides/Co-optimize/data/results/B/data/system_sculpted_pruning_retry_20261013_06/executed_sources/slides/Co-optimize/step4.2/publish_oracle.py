"""Aggregate only new force+base-oracle experiments, preserving all attempts."""
from pathlib import Path
import argparse
import json
import html
import os
import sys

COOPT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(COOPT/'helper_func'))
import artifact_store as A
BASE=A.result_root('B')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments',nargs='+',required=True)
    parser.add_argument('--name',default='base_oracle_results')
    args=parser.parse_args();selected={};attempts=[]
    for name in args.experiments:
        batch=json.loads((BASE/'data'/name/'batch.json').read_text())
        if not batch['complete']:raise ValueError('Finish batch before publishing: '+name)
        for row in batch['results']:
            attempts.append(row)
            old=selected.get(row['id'])
            if old is None or (row['passed'] and not old['passed']) or (
                row['passed']==old['passed'] and row.get('volume_cm3',1e100)<old.get('volume_cm3',1e100)):
                selected[row['id']]=row
    rows=sorted(selected.values(),key=lambda r:(-r.get('pose_count',0),r['id']))
    report=dict(complete=True,experiments=args.experiments,group_count=len(rows),
        passed=sum(r['passed'] for r in rows),force_passed=sum(r['force_passed'] for r in rows),
        base_oracle_passed=sum(r['base_oracle_passed'] for r in rows),selected=rows,attempts=attempts,
        old_force_only_incumbents_accepted=False,
        acceptance='original workpiece force demands + legal pressure enclosure + shared connected material space',
        actual_base_material_optimization_completed=False,whole_ground_friction_yaw_completed=False)
    directory=BASE/'data'/args.name;directory.mkdir(parents=True,exist_ok=False)
    (directory/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    lines=['# 动态base Oracle：全B结果','',
        f"{report['passed']}/{len(rows)}组同时通过全部原始工件需求、合法地脚包围与共同连接查询。",'',
        '体积是名义支撑实体材料；不包括尚未优化的base。连通空间见证单独保存，不作为成品支撑。只约束最终工作姿态。','',
        '| Pose set | Pose数 | 整体通过 | 支撑材料cm³ | 支撑state数 | 最大离地高度mm | 耗时s |',
        '|---|---:|:---:|---:|---:|---:|---:|']
    cards=[]
    for row in rows:
        label=row['id'];folder=BASE/label/'step4/step4.2'/row.get('experiment',args.experiments[-1])
        url=os.path.relpath(folder,BASE)
        value=row.get('volume_cm3');volume='—' if value is None else f'{value:.2f}'
        height=row.get('maximum_object_height_m');height='—' if height is None else f'{height*1000:.1f}'
        state=str(row.get('state_count','—'));passed='PASS' if row['passed'] else '未通过'
        lines.append(f"| {label} | {row.get('pose_count','—')} | {passed} | {volume} | {state} | {height} | {row['seconds']:.1f} |")
        picture=folder/'final_result.png'
        content=f'<img loading="lazy" src="{html.escape(url)}/final_result.png">' if A.resolve_artifact(picture).exists() else ''
        cards.append(f'<section><h2>{html.escape(label)} · {passed}</h2>'
            f'<p>{volume} cm³ · {state} states · maximum object height {height} mm · {row["seconds"]:.1f}s</p>'
            f'<p><a href="{html.escape(url)}/report.json">report</a> · '
            f'<a href="{html.escape(url)}/base_oracle.json">oracle</a> · '
            f'<a href="{html.escape(url)}/process.png">process</a> · '
            f'<a href="{html.escape(url)}/oracle_connected_witness.obj">connection-space witness</a></p>{content}</section>')
    (BASE/f'{args.name}.md').write_text('\n'.join(lines)+'\n')
    text='<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    text+='<style>body{font:16px system-ui;max-width:1450px;margin:32px auto;padding:0 24px;color:#294553}img{width:100%}video{width:min(800px,100%)}section{padding:22px 0;border-top:1px solid #ddd}p{line-height:1.6}</style>'
    text+=f'<h1>Base oracle · {report["passed"]}/{len(rows)}</h1>'
    text+='<p>PASS: original workpiece loads, legal ground enclosure, and one shared connected free-space component. Material volume excludes the unfinished base. Robot transfer rotations are unconstrained.</p>'
    text+='<p>Video: pose2 to pose20. The first pink floor constraint travels with the grey object and blue saved fixture. Pose20 shows both floor constraints and only its current yellow work cone. Transfer motion is illustrative; only installed states constrain material.</p>'
    text+='<video controls loop src="../../seperate_base_support_proof/videos/05_pose2_to_pose20_oracle.mp4"></video>'
    text+=''.join(cards)
    (BASE/f'{args.name}_index.html').write_text(text)
    A.publish_gallery(BASE/f'{args.name}_index.html')
    print(report['passed'],'/',len(rows))


if __name__=='__main__':main()
