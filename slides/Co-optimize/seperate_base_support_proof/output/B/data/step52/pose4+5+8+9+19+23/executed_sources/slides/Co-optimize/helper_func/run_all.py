"""Whole-set initialization of existing groups: merged Step3.1 + work-volume views."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import _bootstrap
import argparse,contextlib,multiprocessing,shutil,time,traceback,html
from concurrent.futures import ProcessPoolExecutor,as_completed
from co_common import *
import step31,step32,step31_render


def output_root(name):return HERE/'output'/name if name=='B' else HERE/'data/object_inputs'/name


def saved_groups(name,scope='existing'):
    root=output_root(name)
    if scope=='selected' or name!='B':return read_selected_pose_groups(name)
    manifest=root/'data/whole_initialization_manifest.json'
    groups={g['id']:g for g in json.loads(manifest.read_text())['groups']} if manifest.exists() else {}
    for path in root.rglob('step3/step3.1/data/report.json'):
        if any(part in ('_history','history') for part in path.parts):continue
        report=json.loads(path.read_text());groups[report['pose_set']]=dict(id=report['pose_set'],poses=report['poses'])
    if not groups:raise RuntimeError('No existing B pose-set outputs found')
    return list(groups.values())


def prepare_existing(name,group):
    root=output_root(name);out=root/group['id']/'step3';report=out/'step3.1/data/report.json'
    if report.exists() and json.loads(report.read_text()).get('schema')!=step31.SCHEMA:
        backup=root/'_history/before_whole_initialize_20261008'/group['id']/'step3'
        if not backup.exists():backup.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(out,backup)
        for path in out.rglob('*'):
            if path.is_file():
                archived=backup/path.relative_to(out)
                if not archived.exists() or I.sha256(path)!=I.sha256(archived):
                    raise RuntimeError('Legacy stage differs from its archive: '+str(path))
        shutil.rmtree(out)
    (out/'data').mkdir(parents=True,exist_ok=True)
    return out


def run_case(arguments):
    name,group,thickness=arguments;out=prepare_existing(name,group)
    with (out/'data/pipeline.log').open('w',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        began=time.monotonic()
        try:
            states,mesh,report=step31.run(name,group,out/'step3.1',thickness)
            step31_render.render(name,group,out)
            visual=step32.run(name,group,out)
            row=dict(id=group['id'],poses=group['poses'],status='initialized',passed=True,step4_ready=True,
                reference_pose=report['reference_pose'],initial_force_status=report['initial_force_status'],
                initial_force_passed=report['force_gate_passed'],working_exclusions_clear=True,
                common_nonworking_faces=report['common_nonworking_face_count'],
                contact_area_cm2=report['actual_contact_area_m2']*1e4,material_volume_cm3=report['material_volume_cm3'],
                seconds=time.monotonic()-began)
            lines=[f"# {group['id']}：whole 初始化",'',
                f"全部 pose 注册到参考 {report['reference_pose']}。内部使用参考物体 mesh 坐标保存，世界注册变换逐 pose 留档。",'',
                '[Step3.1：真实包裹支撑](step3.1/overview.png) · [Step3.2：整组禁区环绕图](step3.2/overview.png)', '',
                'Step3.1 合并注册和贴合包裹，扣除所有完整工作角度禁区；Step3.2 只展示禁区并集。', '',
                f"支撑材料 {report['material_volume_cm3']:.3f} cm³，实际接触面积 {row['contact_area_cm2']:.3f} cm²。",'',
                '| Pose | 原始载荷通过数 | 当前接触受力诊断 |','|---|---:|---|']
            for state_row in report['state_results']:
                lines.append(f"| {state_row['pose']} | {state_row['covered']}/{state_row['load_count']} | {state_row['status']} |")
            lines+=['','INITIALIZED 表示注册和完整禁区切除完成，不表示最终受力或可装卸支撑已求解。'
                '受力失败仅针对当前重合布局，继续保留给后续 whole 的 Direction/Juxtapose 调整。'
                '本轮没有生成接地环、没有挖退出路径、没有运行 Step4。', '',
                '旧阶段结果保存在 B 根目录的 `_history/before_whole_initialize_20261008/`。'
                '原 Step4 文件仍是旧初始化下的历史结果，其数值与图片未重算。']
            (out/'README.md').write_text('\n'.join(lines)+'\n')
            record=dict(complete=True,**row,object=name,schema=step31.SCHEMA,
                full_fixture_accepted=False,downstream_requires_rebuild=True,
                provenance=provenance([out/'step3.1/data/report.json',out/'step3.2/data/report.json'],
                                      [Path(__file__),HERE/'vis_func/step31_render.py',HERE/'vis_func/step32_render.py']),
                artifacts={'../step3.1/overview.png':I.sha256(out/'step3.1/overview.png'),
                           '../step3.2/overview.png':I.sha256(out/'step3.2/overview.png')})
            save(out/'data/report.json',record);I.check_report(out/'data/report.json')
            return row
        except Exception as error:
            traceback.print_exc()
            row=dict(id=group['id'],poses=group['poses'],status='unresolved',passed=False,step4_ready=False,
                     error=f'{type(error).__name__}: {error}',seconds=time.monotonic()-began)
            save(out/'data/failure.json',dict(complete=True,**row));return row


def publish(name,rows,batch):
    root=output_root(name)
    lines=['# B：whole 注册与完整工作禁区初始化','',
        'Step3.1 合并注册、贴合包裹和整组工作禁区切除；Step3.2 展示整组禁区并集的8个环绕等轴测视角。'
        '参考 pose 为每组保存顺序的第一个。沿用已有集合；原始物体、工作面和载荷不变。','',
        '[图片浏览](index.html) · [初始化记录](data/batch.json) · [旧结果](_history/before_whole_initialize_20261008/)', '',
        'INITIALIZED 是几何初始化完成；受力列仅是当前重合布局的诊断。后续仍需 whole 优化、退出和接地处理。'
        '旧 Step3.3/Step4 数值与图片属于旧初始化，本轮没有重新求解。', '',
        '| Pose set | 初始化 | 当前受力诊断 | 材料 cm³ | 图片 |','|---|---|---|---:|---|']
    cards=[]
    for row in rows:
        label=html.escape(row['id']);directory=row['id']+'/step3'
        lines.append(f"| {row['id']} | {row['status']} | {row.get('initial_force_status','未完成')} | "
                     f"{row.get('material_volume_cm3',0):.3f} | [支撑]({directory}/step3.1/overview.png) · [禁区]({directory}/step3.2/overview.png) |")
        if row['passed']:
            cards.append(f'<section><h2>{label}</h2><p>参考 {html.escape(row["reference_pose"])}；材料 {row["material_volume_cm3"]:.3f} cm³；当前受力诊断 {row["initial_force_status"]}。'
                         f'<a href="{directory}/README.md">说明</a></p><div class="pair">'
                         f'<article><h3>Step3.1：真实支撑</h3><a href="{directory}/step3.1/overview.png"><img loading="lazy" src="{directory}/step3.1/overview.png"></a></article>'
                         f'<article><h3>Step3.2：整组工作禁区</h3><a href="{directory}/step3.2/overview.png"><img loading="lazy" src="{directory}/step3.2/overview.png"></a></article></div></section>')
    lines+=['',f"共 {len(rows)} 个已有集合，{batch['initialized_sets']} 个初始化完成，{batch['unresolved_sets']} 个几何未决。",
            '圆弧为球形显示截断边界；算法切除完整半无限区域。蓝色为实际三角支撑；未使用voxel平滑。']
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    (root/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>B whole 初始化</title>'
        '<style>body{max-width:1500px;margin:24px auto;padding:0 20px;font-family:system-ui;color:#253441}'
        '.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}img{width:100%}article{border:1px solid #dde4e9;padding:15px}'
        'section{margin:35px 0}h2{overflow-wrap:anywhere}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style>'
        '<h1>B：whole 初始化</h1><p>所有 pose 注册到每组首个参考 pose，生成贴合包裹并扣除完整工作禁区。'
        'Step3.1 沿用蓝色支撑、灰色物体的原画法；Step3.2 将全部禁区并集从8个环绕角度展示，圆弧只用于截断显示。'
        '当前受力为初始化诊断；退出、接地和后续优化尚未运行。</p><p><a href="README.md">结果表</a></p>'+''.join(cards)+'</html>')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object',nargs='?',default='B');parser.add_argument('--sets',nargs='+')
    parser.add_argument('--jobs',type=int,default=2);parser.add_argument('--wrap-thickness-mm',type=float,default=5.)
    parser.add_argument('--scope',choices=['existing','selected'],default='existing')
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if args.wrap_thickness_mm<=0:parser.error('Wrap thickness must be positive')
    groups=saved_groups(args.object,args.scope)
    if args.sets:
        groups=[g for g in groups if g['id'] in args.sets or g['id'].removeprefix('illegal/') in args.sets]
    if not groups:parser.error('No matching existing pose sets')
    root=output_root(args.object);rows=[];pending=[];began=time.monotonic()
    for group in groups:
        path=root/group['id']/'step3/data/report.json'
        record=json.loads(path.read_text()) if path.exists() else {}
        if args.resume and record.get('schema')==step31.SCHEMA and record.get('passed'):
            I.check_report(path)
            rows.append({key:record[key] for key in ['id','poses','status','passed','step4_ready','reference_pose',
                'initial_force_status','initial_force_passed','working_exclusions_clear','common_nonworking_faces',
                'contact_area_cm2','material_volume_cm3','seconds']})
        else:pending.append(group)
    with ProcessPoolExecutor(max_workers=args.jobs,mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(run_case,(args.object,group,args.wrap_thickness_mm/1000)) for group in pending]):
            row=future.result();rows.append(row);print('INITIALIZED',row['id'],row['status'],round(row['seconds'],2),flush=True)
            save(root/'data/whole_initialization_progress.json',rows)
    order={group['id']:i for i,group in enumerate(groups)};rows.sort(key=lambda row:order[row['id']])
    batch=dict(complete=True,object=args.object,schema=step31.SCHEMA,scope=args.scope,
        pose_sets=len(rows),pose_instances=sum(len(row['poses']) for row in rows),
        initialized_sets=sum(row['passed'] for row in rows),unresolved_sets=sum(not row['passed'] for row in rows),
        initial_force_passed_sets=sum(row.get('initial_force_passed',False) for row in rows),
        force_diagnostics_are_acceptance_gates=False,downstream_requires_rebuild=True,
        step4_run=False,full_fixture_accepted=False,results=rows,seconds=time.monotonic()-began,
        provenance=provenance([ROOT/'objects'/args.object/'pose_sets.json'],[Path(__file__),HERE/'step3.1/step31.py',HERE/'helper_func/work_access.py']))
    save(root/'data/batch.json',batch);publish(args.object,rows,batch)
    print('TOTAL',batch['initialized_sets'],'initialized',batch['unresolved_sets'],'unresolved',flush=True)
    if batch['unresolved_sets']:raise SystemExit(2)


if __name__=='__main__':main()
