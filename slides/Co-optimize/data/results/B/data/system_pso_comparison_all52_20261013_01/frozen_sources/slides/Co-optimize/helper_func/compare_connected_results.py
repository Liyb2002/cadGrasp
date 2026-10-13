"""Compare saved results by identical sets; no search, force solve or Boolean."""
from pathlib import Path
import csv
import hashlib
import json
import os
import statistics

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COOPT=Path(__file__).resolve().parents[1]
ROOT=Path('/home/yli581/Desktop/cadGrasp')
BASE=COOPT/'data/results/B'
NAME='comparison_v8_connected_core_20261013'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths=dict(old=BASE/'data/system_dynamic_postprocess_all52_20261012_01/evaluation_results.json',
        new=BASE/'data/system_connected_core_postprocess_all52_20261013_final/evaluation_results.json',
        old_search=BASE/'data/system_population_soft_all52_20261012_final/step4_results.json',
        new_first=BASE/'data/connected_core_all52_20261012_05/batch.json',
        accepted=BASE/'data/evaluation_material_output_all52_20261012_final/summary.json')
    before={str(p.relative_to(ROOT)):sha(p)for p in paths.values()}
    data={k:json.loads(p.read_text())for k,p in paths.items()}
    assert all(d['complete']for d in data.values())
    old={r['id']:r for r in data['old']['cases']}
    new={r['id']:r for r in data['new']['cases']}
    assert old.keys()==new.keys()and len(new)==52
    rows=[]
    for label in sorted(new,key=lambda k:(new[k]['object_pose_count'],k)):
        o,n=old[label],new[label]
        assert o['object_pose_count']==n['object_pose_count']
        assert abs(o['no_support_bbx_l']-n['no_support_bbx_l'])<=1e-12
        assert n['step53_passed']and n['reuse']+n['juxtaposed']==n['object_pose_count']
        paired=o['step53_passed']and n['step53_passed']
        row=dict(id=label,object_pose_count=n['object_pose_count'],old_status=o['status'],new_status=n['status'],
            paired_final_success=paired,no_support_bbx_l=n['no_support_bbx_l'])
        for stage in ['step4','pruned','final']:
            row['old_'+stage+'_bbx_l']=o.get(stage+'_bbx_l')
            row['new_'+stage+'_bbx_l']=n[stage+'_bbx_l']
        for key in ['reuse','juxtaposed','support_pose_count']:
            row['old_'+key]=o[key]if paired else None
            row['old_candidate_'+key]=o[key]
            row['new_'+key]=n[key]
        row['final_delta_l']=n['final_bbx_l']-o['final_bbx_l']if paired else None
        row['final_delta_percent']=100*(n['final_bbx_l']/o['final_bbx_l']-1)if paired else None
        rows.append(row)
    paired=[r for r in rows if r['paired_final_success']]
    assert len(paired)==49 and sum(r['object_pose_count']for r in paired)==238
    totals={prefix+key:sum(r[prefix+key]for r in paired)
        for prefix in ['old_','new_']for key in ['step4_bbx_l','pruned_bbx_l','final_bbx_l','reuse','juxtaposed','support_pose_count']}
    summary=dict(complete=True,all_groups=52,paired_groups=49,paired_object_poses=238,paired_totals=totals,
        smaller=sum(r['final_delta_l'] < -1e-8 for r in paired),
        larger=sum(r['final_delta_l'] > 1e-8 for r in paired),
        unchanged=sum(abs(r['final_delta_l'])<=1e-8 for r in paired),
        median_per_group_change_percent=statistics.median(r['final_delta_percent']for r in paired),
        recovered=[r['id']for r in rows if not r['paired_final_success']],
        old_first_step4_passed=data['old_search']['passed_groups'],new_first_step4_passed=data['new_first']['passed'],
        old_first_rounds=data['old_search']['actual_rounds'],
        new_first_rounds=sum(r['frontier_search']['rounds']for r in data['new_first']['results']),
        old_final_passed=data['old']['step53_passed'],new_final_passed=data['new']['step53_passed'],
        accepted_reference=dict(bbx=data['accepted']['independent_group_bbox_sums_litres'],
            roles=data['accepted']['layout_totals'],warm_start_reference=True),
        source_sha256=before,force_solver_run=False,boolean_run=False,cases=rows)
    output=BASE/'data'/NAME;output.mkdir(exist_ok=True)
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    with(BASE/(NAME+'.csv')).open('w',newline='')as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

    fig,(ax,bx)=plt.subplots(1,2,figsize=(18,20),gridspec_kw={'width_ratios':[2.2,1]})
    y=np.arange(len(rows));height=.32
    ax.barh(y-height/2,[r['old_final_bbx_l']or 0 for r in rows],height,label='Previous V8 certified final',color='#c58a39')
    ax.barh(y+height/2,[r['new_final_bbx_l']for r in rows],height,label='Connected-core certified final',color='#319cd7')
    ax.set_yticks(y,[r['id']for r in rows],fontsize=8);ax.invert_yaxis()
    ax.set_xlabel('Final static BBX (L)');ax.legend(loc='lower right');ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    for k,r in enumerate(rows):
        if not r['paired_final_success']:
            ax.text(.1,k-height/2,'Old: no certified final',color='#b33434',va='center',fontsize=8)
        def count(value):return '—'if value is None else str(value)
        text=f"O{r['object_pose_count']}  J{count(r['old_juxtaposed'])} → {r['new_juxtaposed']}  S{count(r['old_support_pose_count'])} → {r['new_support_pose_count']}"
        bx.text(0,k,text,va='center',fontsize=9)
        delta='RECOVERED'if r['final_delta_percent']is None else f"{r['final_delta_percent']:+.1f}%"
        color='#319cd7'if r['final_delta_percent']is None else('#348351'if r['final_delta_percent']<0 else'#b33434')
        bx.text(1,k,delta,ha='right',va='center',fontsize=9,color=color)
    bx.set_ylim(ax.get_ylim());bx.set_xlim(0,1);bx.axis('off');bx.set_title('O: object | J: Juxtapose | S: support\nPrevious → current | final BBX change',fontsize=10)
    fig.suptitle('52 sets | previous V8 versus connected-core pipeline\nMatched 49 certified finals: 413.720 → 386.941 L (−6.47%); three recovered sets',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.965));picture=COOPT/'output/B'/(NAME+'.png');fig.savefig(picture,dpi=150);plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,4));x=np.arange(3);width=.36
    for prefix,label,color,offset in [('old_','Previous V8','#c58a39',-width/2),('new_','Connected core','#319cd7',width/2)]:
        values=[totals[prefix+k]for k in ['step4_bbx_l','pruned_bbx_l','final_bbx_l']]
        ax.bar(x+offset,values,width,label=label,color=color)
        for px,value in zip(x+offset,values):ax.text(px,value,f'{value:.3f}',ha='center',va='bottom',fontsize=9)
    ax.set_xticks(x,['Step4','Step5.2 deletion','Final Step5']);ax.set_ylabel('BBX sum (L)');ax.set_ylim(0,470)
    ax.set_title('Same 49 sets / 238 object poses');ax.legend();fig.tight_layout()
    stage_picture=picture.with_name(NAME+'_stages.png');fig.savefig(stage_picture,dpi=150);plt.close(fig)

    pct=lambda a,b:100*(b/a-1)
    warm=data['accepted']['independent_group_bbox_sums_litres'];current=data['new']['passed_totals']
    lines=['# 旧 V8 与连通主体流程的全部组对比','',
        '以紧接本轮之前的 V8 动态换座及其完整后处理为主对照。全部52个集合相同，无支撑基线逐组相同；成品BBX只在两版均通过的49组／238个object pose上比较。所有数字从保存的报告读取，没有重新搜索、求解受力或做Boolean。','',
        '| 同一49组 | 旧 V8 | 新版本 | 变化 |','|---|---:|---:|---:|']
    for key,label in [('step4_bbx_l','Step4 BBX L'),('pruned_bbx_l','删减后 BBX L'),('final_bbx_l','成品 BBX L'),('support_pose_count','支撑安装 pose'),('reuse','复用 pose'),('juxtaposed','Juxtapose pose')]:
        a,b=totals['old_'+key],totals['new_'+key]
        lines.append(f'| {label} | {a:.3f} | {b:.3f} | {b-a:+.3f} / {pct(a,b):+.2f}% |')
    lines+=['',f"共同49组：{summary['smaller']}组变小，{summary['larger']}组变大，{summary['unchanged']}组在1e-8L比较精度内不变；逐组变化中位数{summary['median_per_group_change_percent']:+.2f}%。",'',
        '旧版第一轮Step4为50/52，新版同为32槽位、8轮上限的第一轮为43/52；实际完成轮次87→143。旧版最终49/52，新版经独立32／48槽位重试及本轮保存候选精修后52/52。接受条件更严格，完整计算预算也不同，不能将最终52/52解释为单轮搜索更快或更容易。','',
        '搜索仍使用原位初始化、优化中离散换座、候选间差分和短调整，Juxtapose身份均不预先指定。新版本先在Step4检查连通实际接触主体与全部原始需求，Step5.2只删并优先缩小BBX，Step5.3继续长脚。布局也随之改变，BBX差异包含换座、主体选择、删减和长脚的共同影响。','',
        '共同49组的Step4→删减→成品：旧版324.424→315.041→413.720L，新版328.262→309.985→386.941L。删减带来的BBX降低9.383→18.278L，长脚增加98.679→76.956L，净增幅27.52%→17.88%。','',
        '新恢复的三组是pose1+2+4+5+6+7（旧后处理失败）、pose1+4+7+9+12+16+21+23+27及pose1+2+3+4+5+6+7+8+9+10（旧Step4失败）。新增成品BBX合计45.526L，因此不能直接拿新版52组成品432.467L与旧版49组成品413.720L判断大小。','',
        f'若对照更早已满意的暖启动52/52版本：成品BBX {warm["with_full_support"]:.3f}→{current["final_bbx_l"]:.3f}L（{pct(warm["with_full_support"],current["final_bbx_l"]):+.2f}%），Step4 {warm["with_step4_support"]:.3f}→{current["step4_bbx_l"]:.3f}L（{pct(warm["with_step4_support"],current["step4_bbx_l"]):+.2f}%）；支撑安装pose112→123，复用108→123，Juxtapose155→140。这是另一种执行来源的结果对照，不是同预算冷启动性能对照。','',
        f'[全部组图]({os.path.relpath(picture,BASE)}) · [相同49组三阶段图]({os.path.relpath(stage_picture,BASE)}) · [完整CSV]({NAME}.csv) · [原始比较数据](data/{NAME}/summary.json)。','',
        '旧版没有通过的三组，不填写旧成品BBX或旧成品角色数；CSV另存旧失败候选的诊断角色数。变化百分比以旧成品为基准，完整精度在CSV。','',
        '| Set | object | 复用/支撑：旧→新 | Juxtapose：旧→新 | 无支撑 L | Step4 L：旧→新 | 成品 L：旧→新 | 成品 Δ L / % |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    def number(value):return '—'if value is None else f'{value:.3f}'
    def integer(value):return '—'if value is None else str(value)
    for r in rows:
        delta='恢复通过'if r['final_delta_l']is None else f"{r['final_delta_l']:+.3f} / {r['final_delta_percent']:+.2f}%"
        lines.append(f"| {r['id']} | {r['object_pose_count']} | {integer(r['old_reuse'])}/{integer(r['old_support_pose_count'])} → {r['new_reuse']}/{r['new_support_pose_count']} | {integer(r['old_juxtaposed'])} → {r['new_juxtaposed']} | {r['no_support_bbx_l']:.3f} | {number(r['old_step4_bbx_l'])} → {r['new_step4_bbx_l']:.3f} | {number(r['old_final_bbx_l'])} → {r['new_final_bbx_l']:.3f} | {delta} |")
    (BASE/(NAME+'.md')).write_text('\n'.join(lines)+'\n')
    assert all(sha(ROOT/path)==digest for path,digest in before.items())
    (output/'verification.json').write_text(json.dumps(dict(passed=True,source_sha256=before,sources_unchanged=True,
        all_sets=52,paired_sets=49,unsupported_baselines_identical=True,force_solver_run=False,boolean_run=False,
        pictures={str(p.relative_to(ROOT)):sha(p)for p in [picture,stage_picture]}),indent=2)+'\n')
    print('COMPARISON VERIFIED',len(rows),'sets; matched',len(paired),'final BBX',totals['old_final_bbx_l'],'→',totals['new_final_bbx_l'],flush=True)


if __name__=='__main__':main()
