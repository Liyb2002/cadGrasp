"""Continue checked whole Step4 solutions; preserve the complete baseline."""
import _bootstrap
import argparse,contextlib,multiprocessing,os,shutil,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from co_common import *
from run_all import output_root,saved_groups
from compact_volume import CompactSearch,load_incumbent
from whole_search.fast_search import FastModel
from whole_step4_render import render_search


def run_case(arguments):
    name,group,options=arguments;root=output_root(name);base=root/group['id']
    source=base/'step4/step4.2';out=source/'compact'/options['mode']
    if (out/'data/report.json').exists():
        report=I.check_report(out/'data/report.json')
        return dict(id=group['id'],method=options['mode'],status='pass',
            baseline_volume_cm3=report['baseline_volume_cm3'],volume_cm3=report['volume_cm3'],
            saved_fraction=report['material_saved_fraction'],reused_completed_result=True)
    out.mkdir(parents=True,exist_ok=True);(out/'data').mkdir(exist_ok=True)
    token=json.loads((root/'data/whole_step4_active_run.json').read_text())['run_token']
    os.environ['COOPT_WHOLE_RUN_TOKEN']=token
    sources=[Path(__file__),Path(__file__).with_name('compact_volume.py')]
    frozen=out/'data/executed_sources';frozen.mkdir(exist_ok=True)
    for path in sources:shutil.copy2(path,frozen/path.name)
    save(out/'data/source_manifest.json',dict(code=I.hashes(sources),
        baseline_report_sha256=I.sha256(source/'data/report.json'),options=options,
        upstream_physics_sources_are_frozen_by_baseline=True))
    began=time.monotonic()
    with (out/'data/run.log').open('w',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            model=FastModel(group['poses'],name,initialization_report=base/'step3/step3.1/data/report.json')
            baseline,baseline_report=load_incumbent(model,source)
            model.worker_program=HERE/'helper_func/whole_search/exact_worker.py'
            model.checkpoint_dir=out/'exact_states'
            model.exact_timeout=180
            search=CompactSearch(model,out,mode=options['mode'],rounds=options['rounds'],
                width=options['width'],repair_width=options['repair_width'],
                screen_budget=options['screen_budget'],full_budget=options['full_budget'],seed=options['seed'])
            nodes=search.run(baseline)
            report=search.validate_and_save(nodes,baseline,baseline_report,source,options['real_budget'])
            render_search(model,out,report)
            report.update(pose_set=group['id'],stage='step4.2',seconds=time.monotonic()-began)
            report['provenance']['code'].update(I.hashes(sources))
            report['artifacts']={p.name:I.sha256(p) for p in out.iterdir()
                if p.is_file() and p.name not in ['report.json','README.md']}
            report['artifacts'].update({'data/render.json':I.sha256(out/'data/render.json'),
                'data/source_manifest.json':I.sha256(out/'data/source_manifest.json')})
            save(out/'report.json',report)
            data=report.copy();data['artifacts']={'../'+k:v for k,v in report['artifacts'].items()}
            save(out/'data/report.json',data);I.check_report(out/'data/report.json')
            (out/'README.md').write_text('# Whole 实体体积优化\n\n'
                f"方法 `{options['mode']}`；{baseline['volume_cm3']:.3f} → {report['volume_cm3']:.3f} cm³，"
                f"减少 {100*report['material_saved_fraction']:.2f}%。\n\n"
                '[过程](process.png) · [最终各pose](final_result.png) · [搜索树](tree.json) · [真实验收](actual_validation.json)\n\n'
                '全组从始至终参与。材料/接触增删用于快速筛选；所有原始32,768载荷/pose用于候选和最终承载检查。'
                '最终接受原完整工作禁区、退出、接触核及1%净空检查的真实三角支撑，体积取实体本身。'
                '原可行支撑保留为真实体积上界。修复分支暂时失效不代表最终通过。\n\n'
                '图内没有文字，固定等轴测相机；过程绿增红减蓝保留，末格与最终图是真实验收的支撑。'
                '中间过程网格是名义几何，不能用作净空证书。接地、连通和强度尚未验收。\n')
            return dict(id=group['id'],method=options['mode'],status='pass',
                baseline_volume_cm3=baseline['volume_cm3'],volume_cm3=report['volume_cm3'],
                saved_fraction=report['material_saved_fraction'],full_checks=search.full_checks,
                seconds=report['seconds'],baseline_retained=report['baseline_retained'])
        except Exception as error:
            traceback.print_exc();row=dict(id=group['id'],method=options['mode'],status='unresolved',
                error=f'{type(error).__name__}: {error}',seconds=time.monotonic()-began)
            save(out/'data/failure.json',row);return row


def collect(name):
    root=output_root(name);rows=[]
    for group in saved_groups(name):
        for mode in ['greedy','structural-greedy','beam']:
            out=root/group['id']/'step4/step4.2/compact'/mode
            path=out/'data/report.json'
            if not path.exists():continue
            report=I.check_report(path)
            rows.append(dict(id=group['id'],method=mode,baseline_cm3=report['baseline_volume_cm3'],
                volume_cm3=report['volume_cm3'],saved_fraction=report['material_saved_fraction'],
                baseline_retained=report['baseline_retained'],budget=report['budget'],
                full_checks=report['full_load_evaluations_used'],seconds=report['seconds'],
                rotating_reuse_pose_count=report['rotating_reuse_pose_count'],
                juxtaposed_pose_count=report['juxtaposed_pose_count'],
                report=str(path.relative_to(root))))
    save(root/'data/whole_compact_comparison.json',dict(rows=rows,
        objective='actual support solid volume',actual_force_exit_work_checked=True,
        full_fixture_accepted=False,all42_baseline_preserved=True))
    lines=['# Whole 实体体积优化对照','',
        '所有方法从各自同一个真实可行Step4.2结果继续；体积为支撑实体，而非包围盒。'
        '新布局保留全组原载荷及完整工作/退出约束；实际候选不通过时保留原结果。'
        '方法启发式，不能证明全局最优。','',
        '| Pose set | 方法 | 基线 cm³ | 最终 cm³ | 减少 | 图 |',
        '|---|---|---:|---:|---:|---|']
    for row in rows:
        directory=f"{row['id']}/step4/step4.2/compact/{row['method']}"
        lines.append(f"| {row['id']} | {row['method']} | {row['baseline_cm3']:.3f} | {row['volume_cm3']:.3f} | "
            f"{100*row['saved_fraction']:.2f}% | [过程]({directory}/process.png) · [最终]({directory}/final_result.png) |")
    (root/'compact_results.md').write_text('\n'.join(lines)+'\n')
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object',nargs='?',default='B')
    parser.add_argument('--sets',nargs='*')
    parser.add_argument('--mode',choices=['greedy','structural-greedy','beam','both'],default='beam')
    parser.add_argument('--rounds',type=int,default=6)
    parser.add_argument('--width',type=int,default=2)
    parser.add_argument('--repair-width',type=int,default=1)
    parser.add_argument('--screen-budget',type=int,default=96)
    parser.add_argument('--full-budget',type=int,default=48)
    parser.add_argument('--real-budget',type=int,default=3)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--jobs',type=int,default=1)
    parser.add_argument('--collect-only',action='store_true')
    args=parser.parse_args()
    if args.collect_only:print('COLLECT',len(collect(args.object)));return
    groups=saved_groups(args.object)
    if args.sets:
        wanted=set(args.sets);groups=[g for g in groups if g['id'] in wanted]
        if len(groups)!=len(wanted):raise ValueError('Unknown saved group')
    if min(args.width,args.rounds,args.screen_budget,args.full_budget,args.real_budget,args.jobs)<1:
        raise ValueError('Search budgets/width must be positive')
    methods=['greedy','beam'] if args.mode=='both' else [args.mode]
    jobs=[]
    for group in groups:
        for mode in methods:
            options={key:getattr(args,key) for key in ['rounds','width','repair_width','screen_budget',
                'full_budget','real_budget','seed']};options['mode']=mode
            jobs.append((args.object,group,options))
    if args.jobs==1:
        for job in jobs:print('COMPACT CASE',run_case(job),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(run_case,job) for job in jobs]
            for future in as_completed(futures):print('COMPACT CASE',future.result(),flush=True)
    print('COLLECT',len(collect(args.object)),flush=True)


if __name__=='__main__':main()
