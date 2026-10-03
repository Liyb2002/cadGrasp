"""Audit the labels, export 32 partial-state examples, and show every head."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from prepare import ROOT, I, prepare, compatible
from search import J, object_directions, pair_dispersion, install_recorded_recovery, best_pair


def refine_pair_values(output, results, problems, pools, catalogues):
    # Share all discovered counterpart completions fairly across both poses.
    banks = [set(), set()]
    own = {}
    for k, pose in enumerate(results['config']['poses']):
        banks[k].update(tuple(g) for g in json.loads((output/f'bank_{pose}.json').read_text())['completions'])
        for path in sorted((output/'actions').glob(f'{pose}_C*.json')):
            data = json.loads(path.read_text())
            groups = {tuple(t['completion']) for t in data['continuation_attempts'] if t['completion'] is not None}
            own[data['summary']['id']] = sorted(groups)
            banks[k].update(groups)
    vectors = [object_directions(p,c) for p,c in zip(problems,catalogues)]
    angles = np.arccos(np.clip(vectors[0]@vectors[1].T,-1,1))
    for row in results['rows']:
        if not row['valid']:
            continue
        k = results['config']['poses'].index(row['pose'])
        best = best_pair(own[row['id']], sorted(banks[1-k]), k, pools, angles, results['config']['lambda_direction'])
        if best:
            row.setdefault('value_before_shared_counterpart_pool', row['value'])
            row.update(best)
            row['completion_ids'] = [[pools[t][i]['id'] for i in g] for t,g in enumerate(best['completion_indices'])]
            row['object_exit_vectors'] = [vectors[t][i].tolist() for t,i in enumerate(best['direction_ids'])]
    results['postprocessing'] = dict(counterpart_pool='all certified local completions discovered in the recorded run',
        local_bank_sizes=[len(b) for b in banks], code=I.hashes([Path(__file__)]),
        own_action_pool='only successful conditional attempts for that forced head')
    I.save(output/'values.json',results)
    I.save(output/'completion_banks.json',dict(poses=results['config']['poses'], groups=[[list(g) for g in sorted(b)] for b in banks]))
    columns=['id','pose','index','valid','status','value','remaining_heads','total_heads','dispersion','minimum_angle_deg','successful_attempts','distinct_completions']
    with (output/'values.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(results['rows'])


def audit(output, results, problems, pools, catalogues):
    install_recorded_recovery(output)
    cfg=results['config'];I.check_hashes(cfg['source_inputs']);I.check_hashes(cfg['code'])
    warm_starts=[p for pose in cfg['poses'] for p in (ROOT/'slides/baseline_algo/output'/cfg['object']/'independent_poses'/pose/'step3_scheculer').glob('particle_*/schedule.json')]
    results['postprocessing']['warm_start_inputs']=I.hashes(warm_starts)
    I.save(output/'values.json',results)
    vectors=[object_directions(p,c) for p,c in zip(problems,catalogues)]
    angles=np.arccos(np.clip(vectors[0]@vectors[1].T,-1,1))
    groups=[set(),set()]
    checked=0
    for row in results['rows']:
        k=cfg['poses'].index(row['pose']); entry=pools[k][row['index']]
        assert row['valid']==entry['valid']
        if row['value'] is None:
            assert row['status']!='success_found'
            continue
        gs=[tuple(g) for g in row['completion_indices']]
        assert row['index'] in gs[k]
        for t,g in enumerate(gs):
            assert len(g)==len(set(g)) and len(g)<=cfg['max_heads_per_pose']
            assert compatible(pools[t],g)
            groups[t].add(g)
        d,ids,degrees=pair_dispersion(pools[0],gs[0],pools[1],gs[1],angles)
        assert row['direction_ids']==list(ids)
        np.testing.assert_allclose(row['dispersion'],d,atol=1e-12)
        assert row['remaining_heads']==sum(map(len,gs))-1
        np.testing.assert_allclose(row['value'],row['remaining_heads']+cfg['lambda_direction']*d,atol=1e-12)
        checked+=1
    replays=[]
    for k in range(2):
        for i,g in enumerate(sorted(groups[k])):
            full=problems[k].supply([pools[k][j]['contact'] for j in g])
            mask,info=J.classify(full,problems[k].targets)
            if not mask.all():
                raise RuntimeError(f'Fresh complete-load replay failed: {cfg["poses"][k]} {g}')
            evidence=J.C.verify_classification(full,problems[k].targets,mask)
            replays.append(dict(pose=cfg['poses'][k],indices=list(g),passed=True,
                                load_count=len(mask),classifier=info,independent_lp_checks=evidence))
            if (i+1)%10==0:
                print('REPLAY',cfg['poses'][k],i+1,'of',len(groups[k]),flush=True)
    retries=[json.loads(p.read_text()) for p in (output/'numerical_retries').glob('*.json')]
    I.save(output/'audit.json',dict(passed=True,labels_checked=checked,
        unique_local_completions=len(replays),fresh_terminal_replays=replays,
        numerical_retries=len(retries),numerical_unresolved=sum(r['status']=='numerical_unresolved' for r in retries),
        source_files_unchanged=True,complete_fixture_constructed=False,
        report_code=I.hashes([Path(__file__)])))


def state_examples(output,results,limit=32):
    """Back up one verified continuation at diverse prefixes; not all-action Q."""
    rng=np.random.default_rng(results['config']['seed']+9000)
    rows=[r for r in results['rows'] if r['value'] is not None]
    examples=[];seen=set()
    for row in sorted(rows,key=lambda r:r['value']):
        actions=[(k,cid) for k,group in enumerate(row['completion_ids']) for cid in group]
        rng.shuffle(actions)
        chosen=[[],[]]
        for offset,(k,cid) in enumerate(actions):
            signature=(tuple(sorted(chosen[0])),tuple(sorted(chosen[1])),k,cid)
            if signature not in seen and offset>0 and len(examples)<limit:
                seen.add(signature)
                remaining=len(actions)-offset-1
                examples.append(dict(state_id=len(examples),selected_ids=[sorted(x) for x in chosen],
                    action=dict(pose=results['config']['poses'][k],id=cid),
                    completion_ids=row['completion_ids'],remaining_heads=remaining,
                    dispersion=row['dispersion'],value=remaining+results['config']['lambda_direction']*row['dispersion'],
                    target_kind='cost of this known verified continuation, not minimum over all completions',
                    source_action=row['id']))
            chosen[k].append(cid)
        if len(examples)>=limit:break
    I.save(output/'state_samples.json',dict(count=len(examples),records=examples,
        scope='32 partial-state/action training examples backed up from fully verified terminal pairs'))
    with (output/'training_records.jsonl').open('w') as f:
        for r in results['rows']:
            record=dict(kind='root_action',selected_ids=[[],[]],**r)
            f.write(json.dumps(record,allow_nan=False)+'\n')
        for r in examples:
            f.write(json.dumps(dict(kind='partial_state_action',**r),allow_nan=False)+'\n')


def render(output,results,problems,pools):
    rows=results['rows']; values=[r['value'] for r in rows if r['value'] is not None]
    cmap=plt.get_cmap('viridis_r');norm=colors.Normalize(vmin=min(values),vmax=max(values))
    fig=plt.figure(figsize=(25,15),dpi=160,facecolor='white')
    grid=fig.add_gridspec(2,2,width_ratios=[1,2.4],left=.035,right=.97,bottom=.11,top=.87,wspace=.1,hspace=.28)
    fig.suptitle('B / Pose 1 + Pose 3: conditional completion value of every candidate',fontsize=26,y=.975)
    cfg=results['config']
    fig.text(.5,.935, r'$C=N_{\rm remaining}+\lambda D_{\rm final}$'+
             f'   |   lower is better   |   lambda = {cfg["lambda_direction"]:g}   |   current state: no selected heads',
             ha='center',fontsize=19)
    fig.text(.5,.899,f'{cfg["attempts"]} continuation attempts per legal head; at most {cfg["max_heads_per_pose"]} heads per pose. '
             'Finite scores require all 32,768 loads in BOTH poses to pass.',ha='center',fontsize=16)
    gray='#dddddd';unresolved='#f4b55e'
    for k in range(2):
        problem=problems[k]; pool=pools[k]; local=[r for r in rows if r['pose']==problem.pose]
        transform=np.asarray(problem.domain.data['frame']['T_world_mesh']);rotation=transform[:3,:3];translation=transform[:3,3]
        mesh=(problem.domain.mesh.vertices-translation)@rotation
        ax=fig.add_subplot(grid[k,0],projection='3d')
        ax.add_collection3d(Poly3DCollection(mesh[problem.domain.mesh.faces],facecolor='#bcc7d0',edgecolor='none',alpha=.17))
        coordinates=[];facecolors=[]
        for r,e in zip(local,pool):
            if e['contact'] is None:continue
            coordinates.append((e['contact']['center_m']-translation)@rotation)
            facecolors.append(colors.to_rgba(cmap(norm(r['value'])) if r['value'] is not None else unresolved if r['valid'] else gray))
        coordinates=np.asarray(coordinates)
        ax.scatter(*coordinates.T,c=facecolors,s=32,depthshade=False,edgecolors='#333333',linewidths=.3)
        lo=mesh.min(axis=0);hi=mesh.max(axis=0);center=(lo+hi)/2;extent=(hi-lo).max()*.58
        ax.set_xlim(center[0]-extent,center[0]+extent);ax.set_ylim(center[1]-extent,center[1]+extent);ax.set_zlim(center[2]-extent,center[2]+extent)
        ax.set_box_aspect([1,1,1],zoom=1.5);ax.view_init(elev=25,azim=-55);ax.set_axis_off()
        successes=[r for r in local if r['value'] is not None]
        ax.set_title(problem.pose.replace('_',' ').title()+f': {len(successes)}/{sum(r["valid"] for r in local)} legal actions completed',fontsize=20,pad=6)
        for r in sorted(successes,key=lambda r:r['value'])[:3]:
            point=(pool[r['index']]['contact']['center_m']-translation)@rotation
            ax.text(*point,f' C{r["index"]+1:03d}',fontsize=10,color='black')
        table=fig.add_subplot(grid[k,1]);table.set_xlim(0,20);table.set_ylim(0,10);table.invert_yaxis();table.set_aspect('equal');table.axis('off')
        for r in local:
            i=r['index'];x=i%20;y=i//20
            color=cmap(norm(r['value'])) if r['value'] is not None else unresolved if r['valid'] else gray
            table.add_patch(plt.Rectangle((x,y),.96,.96,facecolor=color,edgecolor='white',linewidth=.5))
            lum=.2126*color[0]+.7152*color[1]+.0722*color[2] if not isinstance(color,str) else 1.
            ink='white' if lum<.5 else '#222222'
            table.text(x+.48,y+.27,f'C{i+1:03d}',ha='center',va='center',fontsize=9,color=ink)
            label=f'{r["value"]:.2f}' if r['value'] is not None else '?' if r['valid'] else 'X'
            table.text(x+.48,y+.64,label,ha='center',va='center',fontsize=12,color=ink,fontweight='bold')
        table.set_title('All 200 candidates: ID above, searched value below',fontsize=19,pad=13)
    cax=fig.add_axes([.3,.058,.4,.017]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal')
    cb.set_label('Best cost found by this search (not proven optimal)',fontsize=14)
    fig.text(.025,.071,'X / grey: Step2 rejected',fontsize=15)
    fig.text(.75,.071,'? / orange: no success found in budget',fontsize=15)
    fig.savefig(output/'values.png',facecolor='white');plt.close(fig)
    best=sorted([r for r in rows if r['value'] is not None],key=lambda r:(r['value'],r['id']))
    summary=results['summary']
    lines=['# B / pose1+3 的候选 value','', '![全部候选](values.png)','',
        f'每个 pose 各 200 个候选，共 {summary["candidates"]} 个动作；Step2 合法 {summary["valid"]} 个。',
        f'找到成功补全 {summary["success_found"]} 个，预算内未找到 {summary["unresolved"]} 个；其余由 Step2 排除。','',
        '状态为两个 pose 都尚未选头。每个合法 head 固定先选，再尝试 32 次补全；每 pose 最多 6 个头，lambda=1。',
        '各动作使用自己的 32 次条件补全；另一个 pose 可从本轮所有已验证成功补全中选择，统一比较方向和总代价。',
        '数值是离线搜索找到的最好代价，不是已训练网络的预测，也不是全局最少头数。',
        '数值越低越好。灰色 X 为原候选非法；橙色 ? 为本预算内未找到成功补全，不表示无解。','',
        '| 候选 | Value | 还需头数 | 最终总头数 | 退出方向最小夹角 |','|---|---:|---:|---:|---:|']
    for r in best[:12]:
        lines.append(f'| {r["id"]} | {r["value"]:.6f} | {r["remaining_heads"]} | {r["total_heads"]} | {r["minimum_angle_deg"]:.2f}° |')
    lines+=['','- `values.csv` / `values.json`：所有动作的分数、分解项和完整成功头组。',
        '- `actions/`：每个合法动作的 32 次补全尝试，包含失败。',
        '- `state_samples.json`：32 个部分状态及其已验证补全标签；每个状态只提供一个动作标签，不是全候选最优值。',
        '- `training_records.jsonl`：根状态动作记录与部分状态记录。',
        '- `audit.json`：分数公式、方向变换和完整原始载荷的重新验收。','',
        '复用当前 `baseline_algo/output/B/independent_poses/` 的 pose1、pose3 输入；没有使用历史 pose1+3 展示的旧 pose。',
        '补全搜索使用已有成功头组作起点，随机换头、强制加头、随机合法扩展与逐头删除。它有起点偏好，属于首次有限预算标签实验。',
        '两个 pose 的受力与局部路径独立，方向分散度对完成方案配对后共同计算。不强制共享头。',
        'Step3 路径资格与全部样本的受力通过不等于完整有限厚度实体已经构造；本次不运行 Step4。']
    (output/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--object',default='B');args=parser.parse_args()
    output=ROOT/'slides/value_network/data'/args.object/'pose1+3'
    results=json.loads((output/'values.json').read_text())
    problems,pools,catalogues,_=prepare(args.object,results['config']['poses'],output/'inputs')
    refine_pair_values(output,results,problems,pools,catalogues)
    audit(output,results,problems,pools,catalogues)
    state_examples(output,results)
    render(output,results,problems,pools)
    print('REPORT',output/'values.png',flush=True)


if __name__=='__main__':main()
