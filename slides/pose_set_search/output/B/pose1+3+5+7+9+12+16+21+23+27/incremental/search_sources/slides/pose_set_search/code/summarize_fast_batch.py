"""Local comparison and navigation for the quick B experiment."""
import argparse
import html
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from common import HERE,C
from case_sets import CASES,case_directory,POSE_SET_NAMES


def font(size,bold=False):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/'+
        ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'),size)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=HERE/'output/B')
    args=parser.parse_args()
    root=args.root
    batch=json.loads((root/'batch.json').read_text())
    assert batch['complete'] and batch['completed_runs']==20
    reports={}
    comparisons={}
    for case in CASES:
        directory=case_directory(root,case)
        comparisons[case]=json.loads((directory/'comparison.json').read_text())
        for method in ['whole','incremental']:
            out=directory/method
            report=json.loads((out/'search_report.json').read_text())
            assert not report['final_acceptance_run'] and report['exact_evaluations_during_search']==0
            assert (out/'process.png').exists() and (out/'final_result.png').exists()
            reports[case,method]=report
        geometry=[json.loads((directory/method/'mesh_geometry.json').read_text()) for method in ['whole','incremental']]
        comparisons[case]['mesh_material_volumes_cm3']={method:data['final_material_volume_cm3'] for method,data in zip(['whole','incremental'],geometry)}
    lines=['# B：10 个集合，两种快速搜索', '',
        '**加入完整工作锥后，20 次独立搜索已完成。** 每个 pose 在采样支撑接触上检查全部原始32,768个载荷；不做完整压力证书验收。', '',
        '[对比图](comparison.png) · [可点击浏览](index.html) · [搜索清单](batch.json) · [预算与修复记录](experiment.json) · [完整工作禁区检查](work_access_audit.md)', '',
        'B原始设定为30°半角、60°总开角。工作面积沿整个角度范围向外形成的半无限区域，从每个候选开始就是硬排除；'
        '接触、材料占据和蓝色三角支撑使用同一定义，重建后另查全部工作锥的连续区域交叠。'
        '体积列使用实际mesh材料体积；括号保留共同131,072点采样估计。', '',
        '**每个 set 两个视频**：`process.mp4` 展示已选操作和材料增减，`result.mp4` 展示所有 pose 依次装入、取出，最后查看空支撑。'
        '固定单个等轴测视角，先 whole、后 incremental，中间短暂白场分隔，白底无文字。'
        '橙色面片为该 pose 的原始工作面积；放稳后停顿 1 秒，橙红色小箭头沿工作面内法线，表示可能施加的力。'
        '半透明琥珀色为当前pose的工作禁区，画面仅截取有限长度，算法排除完整半无限区域。'
        '过程图每步只有一个等轴测视角；具体选择和步幅仍在各方法 README。', '',
        '| 集合 | whole：mesh cm³（采样估计） / 搜索 s / 转动+J | incremental：mesh cm³（采样估计） / 搜索 s / 转动+J | 视频与过程图 |',
        '|---|---|---|---|']
    maxima=max(volume for comp in comparisons.values() for volume in comp['mesh_material_volumes_cm3'].values())
    image=Image.new('RGB',(1600,190+len(CASES)*144+85),'white')
    draw=ImageDraw.Draw(image)
    draw.text((25,17),'B | 10 pose sets | Whole vs incremental',font=font(34,True),fill='#253441')
    draw.text((25,65),'Independent starts, same base budget; local repairs recorded. Actual mesh material volume.',font=font(23),fill='#55636c')
    draw.text((25,104),'Sampled forces; full work-access geometry checked; no pressure certificate.',font=font(22),fill='#657581')
    draw.rectangle((26,152,45,170),fill='#319cd7');draw.text((55,148),'whole',font=font(20),fill='#253441')
    draw.rectangle((180,152,199,170),fill='#37a77d');draw.text((210,148),'incremental',font=font(20),fill='#253441')
    draw.text((1105,148),'cm3    search sec    rotating + J',font=font(20),fill='#55636c')
    whole_wins=incremental_wins=0
    browser=[]
    for index,(case,poses) in enumerate(CASES.items()):
        name=case_directory(root,case).name
        wr,ir=reports[case,'whole'],reports[case,'incremental']
        wv,iv=[comparisons[case]['mesh_material_volumes_cm3'][method] for method in ['whole','incremental']]
        ws,isample=[comparisons[case]['volumes_cm3'][method] for method in ['whole','incremental']]
        whole_wins+=wv<iv
        incremental_wins+=iv<wv
        def summary(report,volume,sampled=None):
            status='采样通过' if report['sampled_force_passed'] and report['pose_count']==len(poses) else '未完成 / 采样未通过'
            estimate=f'（估计 {sampled:.2f}）' if sampled is not None else ''
            return f"{volume:.2f}{estimate} / {report['search_seconds']:.1f} / {report['rotating_reuse_pose_count']}+{report['juxtaposed_pose_count']} ({status})"
        lines.append(f'| {name} | {summary(wr,wv,ws)} | {summary(ir,iv,isample)} | '
                     f'[操作视频]({name}/process.mp4) · [使用视频]({name}/result.mp4) · '
                     f'[whole]({name}/whole/process.png) · [incremental]({name}/incremental/process.png) |')
        y=190+index*144
        label='+'.join(map(str,poses))
        draw.text((25,y+5),label,font=font(16 if len(label)>25 else 18,True),fill='#253441')
        draw.text((25,y+41),f'{len(poses)} poses',font=font(21),fill='#657581')
        for j,(method,report,volume,color) in enumerate([('whole',wr,wv,'#319cd7'),('incremental',ir,iv,'#37a77d')]):
            top=y+12+j*52
            draw.rectangle((310,top,310+volume/maxima*750,top+33),fill=color)
            draw.text((1110,top+1),f'{volume:6.1f}     {report["search_seconds"]:6.1f}          {report["rotating_reuse_pose_count"]}+{report["juxtaposed_pose_count"]}',font=font(22),fill='#253441')
        draw.line((25,y+132,1575,y+132),fill='#e2e8ed',width=1)
        cards=[]
        for method,report,volume in [('whole',wr,wv),('incremental',ir,iv)]:
            prefix=f'{name}/{method}'
            cards.append(f'<article><h3>{method}</h3><p>{html.escape(summary(report,volume))}</p>'
                         f'<a href="{prefix}/process.png">每一步过程图</a> · '
                         f'<a href="{prefix}/README.md">候选选择明细</a> · '
                         f'<a href="{prefix}/search_report.json">统计</a>'
                         f'<a href="{prefix}/final_result.png"><img loading="lazy" src="{prefix}/final_result.png" alt="{case} {method} final" /></a>'
                         f'<details><summary>展开过程图</summary><a href="{prefix}/process.png"><img loading="lazy" src="{prefix}/process.png" alt="process" /></a></details></article>')
        media=json.loads((case_directory(root,case)/'videos.json').read_text())
        def ranges(kind):
            movie=media[kind]
            return ' · '.join(f'{part["method"]}：{part["start_frame"]/movie["fps"]:.1f}–{(part["end_frame"]+1)/movie["fps"]:.1f} 秒'
                              for part in movie['method_segments'])
        videos=f'<div class="pair"><article><h3>操作过程</h3><p>{ranges("process")}</p><video controls muted playsinline preload="metadata" src="{name}/process.mp4"></video></article><article><h3>所有 pose 依次使用</h3><p>{ranges("result")}</p><video controls muted playsinline preload="metadata" src="{name}/result.mp4"></video></article></div>'
        browser.append(f'<section id="{name}"><h2>{name} · {len(poses)} poses</h2><p>固定单个等轴测视角，先whole、后incremental。绿色为新增材料，红色为删除材料；橙色是工作面积，半透明琥珀色为完整30°工作禁区的有限显示片段。停顿1秒时朝内的小箭头表示可能施加的力。</p>'+videos+'<div class="pair">'+''.join(cards)+'</div></section>')
    draw.text((25,image.height-60),f'Smaller mesh material volume: whole {whole_wins} / incremental {incremental_wins}. Placement count is not the objective.',font=font(23),fill='#55636c')
    image.save(root/'comparison.png')
    lines += ['',f'重建 mesh 材料体积更小：whole {whole_wins} 组，incremental {incremental_wins} 组。'
              '这是一次相同基础预算的快速比较；未完成时额外的小步修复和数值重试单独留档，没有全局最优结论。', '',
              '每个方法目录的 `process.png` 无文字，仅显示依次选中的新支撑；`README.md` 保留操作步幅、拒绝回合、最难载荷和各 finalist 结果；'
              '`final_result.png` 显示同一蓝色支撑在每个 pose 中的摆放。每步布局在 `process_states/`，所有原始候选在 `trace.json`。', '',
              f'20 次纯搜索累计 {sum(report["search_seconds"] for report in reports.values())/60:.1f} 分钟；'
              f'双进程主批次墙钟 {batch["elapsed_seconds"]/60:.1f} 分钟；数值重试、后续局部修复及图像生成另留记录。', '',
              '保持 Co-optimize 数值代码和既有验收结果不变；本批次独立保存在这里。']
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    menu=' · '.join(f'<a href="#{POSE_SET_NAMES[case]}">{POSE_SET_NAMES[case]}</a>' for case in CASES)
    (root/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>B 快速搜索对比</title>'
        '<style>body{margin:25px auto;padding:0 20px;max-width:1500px;font-family:system-ui,sans-serif;color:#253441}'
        'a{color:#287da8}h1{font-size:30px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:22px}'
        'article{border:1px solid #dae2e8;border-radius:12px;padding:18px}img,video{width:100%;margin-top:12px}h2{overflow-wrap:anywhere}'
        'section{margin-top:36px}summary{cursor:pointer;padding:10px 0}nav{line-height:2}'
        '@media(max-width:800px){.pair{grid-template-columns:1fr}}</style>'
        '<h1>B：10个集合，whole / incremental；完整工作锥</h1><p>从每个候选开始排除完整30°工作禁区，全部独立初始化、相同基础预算，额外修复单独记录。载荷为快速采样评估，不做完整压力证书验收。'
        '图与视频使用直接重建的三角 mesh。每组两个视频：固定单个等轴测视角，先 whole、后 incremental。'
        '工作面积标橙色，放稳停顿 1 秒时展示朝内的可能施力箭头。</p><nav>'+menu+'</nav>'
        '<p><a href="comparison.png">整体对比图</a> · <a href="README.md">结果表</a></p>'+''.join(browser)+'</html>')
    C.save(root/'summary.json',dict(complete=True,run_count=20,final_acceptance_run=False,
        sampled_pass_count=batch['sampled_passes'],whole_smaller_count=whole_wins,
        incremental_smaller_count=incremental_wins,comparison_volumes={POSE_SET_NAMES[case]:value for case,value in comparisons.items()},
        triangle_mesh_presentation=True,videos_per_set=2,process_png_text=False,
        video_view_count=1,method_presentation='sequential',work_areas_from_original_tasks=True,
        seated_pause_seconds=1,possible_force_arrows='sampled working-surface inward normals',
        work_access_enforced_during_search=True,work_cone_half_angle_deg=30,
        forbidden_access_regions_drawn=True,continuous_work_access_geometry_checked=True,
        search_seconds_sum=sum(report['search_seconds'] for report in reports.values()),
        batch_wall_seconds=batch['elapsed_seconds']))
    print('PUBLISHED QUICK COMPARISON',root,whole_wins,incremental_wins,flush=True)


if __name__=='__main__':
    main()
