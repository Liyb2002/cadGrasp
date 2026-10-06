"""Step4.2 joint path/equilibrium/connectivity recovery with verified pruning."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from step42_timed import *
import shutil,signal,multiprocessing,traceback
from concurrent.futures import ProcessPoolExecutor

class ConnectedSearch(Search):
    def __init__(self,name,group):
        super().__init__(name,group);self.timeouts=0;self.connection_trace=[];self.connected_best=None;self.connected_common=None
        warm_report=self.out/'data/report.json'
        original_report=json.loads(warm_report.read_text()) if warm_report.exists() else self.initial
        # Historical directions only; reconstruct every candidate with current clearance.
        self.warm=np.array([row['direction_fixture'] for row in original_report['state_results']])
        self.archive=self.out/'data/before_connectivity'
        if not self.archive.exists():
            (self.archive/'data').mkdir(parents=True)
            for f in ['remaining_support.obj','removed_support.obj','restored_support.obj','README.md']:
                if (self.out/f).exists():shutil.copy2(self.out/f,self.archive/f)
            for f in (self.out/'data').iterdir():
                if f.is_file():shutil.copy2(f,self.archive/'data'/f.name)
        self.out=self.base/'step4/step4.2/data/connected_run';(self.out/'data').mkdir(parents=True,exist_ok=True)
        np.savez(self.out/'data/warm_paths.npz',directions=self.warm)

    def component_evidence(self,part):
        mesh=S.unpack(part);tri,src=contact_boundary(self.mesh,mesh,self.allowed)
        masks,infos,supplies=self.classify(tri,src)
        return dict(part=part,triangles=tri,sources=src,masks=masks,infos=infos,supplies=supplies,counts=[int(m.sum()) for m in masks],volume_cm3=material_volume(part)*1e6)

    def candidate(self,directions,common=None,label='proposal',proxy=True):
        signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,max(30.,10.*len(self.states)))
        try:
            result=Search.candidate(self,directions,common,label,proxy)
            if result is None:return None
            components=result['remaining'].decompose();components.sort(key=lambda part:-material_volume(part))
            evidence=[];winner=None
            for k,part in enumerate(components):
                # A retained solid must contain actual positive material, not
                # a zero-volume coplanar remnant of a floating-point Boolean.
                if material_volume(part)<1e-12:continue
                e=self.component_evidence(part);evidence.append(e)
                if all(mask.all() for mask in e['masks']):winner=e;break
                if k>=2:break
            if not evidence:return None
            ranked=max(evidence,key=lambda e:(sum(m.all() for m in e['masks']),min(e['counts']),sum(e['counts'])))
            fractions=np.array(ranked['counts'])/32768.
            score=(int(sum(m.all() for m in ranked['masks'])),float(fractions.min()),float(fractions.sum()),float(ranked['volume_cm3']))
            row=dict(proposal=self.evaluations,label=label,component_count=len(components),component_checks=[dict(volume_cm3=e['volume_cm3'],covered_counts=e['counts']) for e in evidence],best_single_component_counts=ranked['counts'])
            self.connection_trace.append(row)
            if self.connected_best is None or score>self.connected_best['connection_score']:
                result['connection_score']=score;self.connected_best=result;self.connected_common=common
                np.savez(self.out/'data/best_connected_paths.npz',directions=result['directions'],counts=ranked['counts'])
                print('CONNECTED RECOVERY',self.group['id'],ranked['counts'],'proposal',self.evaluations,flush=True)
            if winner is None:return None
            original=result['remaining'];result.update(remaining=winner['part'],removed=self.seed-winner['part'],triangles=winner['triangles'],sources=winner['sources'],masks=winner['masks'],infos=winner['infos'],supplies=winner['supplies'],counts=winner['counts'],pre_prune_component_count=len(components),pre_prune_volume_cm3=material_volume(original)*1e6,pruned_volume_cm3=(material_volume(original)-material_volume(winner['part']))*1e6,pruning_evidence=row)
            result['overlap']=max(material_volume(winner['part']^sweep) for sweep in result['sweeps'])
            result['partition']=abs(material_volume(self.seed)-material_volume(result['remaining'])-material_volume(result['removed']))
            if result['overlap']>=1e-10 or result['partition']>=1e-10 or len(result['remaining'].decompose())!=1:return None
            return result
        except EvaluationDeadline:
            self.timeouts+=1;self.connection_trace.append(dict(proposal=self.evaluations,label=label,status='numerically_unresolved_deadline'));return None
        finally:signal.setitimer(signal.ITIMER_REAL,0.)

    def connected_search(self):
        result=self.candidate(self.warm,label='existing force-feasible paths',proxy=False)
        if result is not None:return result
        lp=linprog([0,0,0,-1],A_ub=np.c_[-self.normals,np.ones(len(self.normals))],b_ub=np.zeros(len(self.normals)),bounds=[(-1,1)]*3+[(None,None)],method='highs')
        if lp.success and np.linalg.norm(lp.x[:3])>1e-8:
            a=lp.x[:3]/np.linalg.norm(lp.x[:3]);result=self.candidate(project_common(a,self.normals),a,label='common floor cone')
            if result is not None:return result
        for count in [160,512,2048]:
            for common in fibonacci(count):
                result=self.candidate(project_common(common,self.normals),common,label=f'connected common tendency {count}')
                if result is not None:return result
            if self.connected_common is not None:
                center=self.connected_common.copy();axis=np.eye(3)[np.argmin(abs(center))];u=np.cross(center,axis);u/=np.linalg.norm(u);v=np.cross(center,u)
                for angle in [20,10,5,2,.5]:
                    for theta in np.linspace(0,2*np.pi,24,endpoint=False):
                        common=np.cos(np.deg2rad(angle))*center+np.sin(np.deg2rad(angle))*(np.cos(theta)*u+np.sin(theta)*v)
                        for lift in [.0001,.003,.01,.03]:
                            result=self.candidate(project_common(common,self.normals,lift),common,label='connected common local refinement')
                            if result is not None:return result
            save(self.out/'data/connection_checkpoint.json',dict(complete=False,proposals=self.evaluations,trace=self.connection_trace))
        if self.connected_best is not None:
            for sweep in range(6):
                previous=self.connected_best['connection_score']
                for k in range(len(self.states)):
                    center=self.connected_best['directions'][k].copy();n=self.normals[k];tangent=center-(center@n)*n
                    if np.linalg.norm(tangent)<1e-8:
                        axis=np.eye(3)[np.argmin(abs(n))];tangent=axis-(axis@n)*n
                    tangent/=np.linalg.norm(tangent);side=np.cross(n,tangent)
                    for angle in [1,-1,2,-2,5,-5,10,-10,20,-20,40,-40,80,-80]:
                        for lift in [.001,.01,.05,.2]:
                            directions=self.connected_best['directions'].copy();theta=np.deg2rad(angle);directions[k]=np.cos(theta)*tangent+np.sin(theta)*side+lift*n;directions[k]/=np.linalg.norm(directions[k])
                            result=self.candidate(directions,label=f'connected independent pose {k}',proxy=True)
                            if result is not None:return result
                if self.connected_best['connection_score']==previous:break
        return None

def finish_connected(search,result,began):
    row=finish(search,result,began);out=search.out;p=out/'data/report.json';report=json.loads(p.read_text())
    # The original force-only result remains reviewable in before_connectivity.
    report.update(status='force_exit_and_connectivity_pass',connectivity_required=True,remaining_component_count=1,pre_prune_component_count=result['pre_prune_component_count'],pre_prune_volume_cm3=result['pre_prune_volume_cm3'],pruned_unnecessary_volume_cm3=result['pruned_volume_cm3'],component_value_policy='Deleting all discarded components together is certified by every original demand passing on the retained connected component alone; ground coverage and strength are deferred',pruning_evidence=result['pruning_evidence'],candidate_timeouts_unresolved=search.timeouts,search_method='Co-optimize floor-legal exit directions against full equilibrium and single-component equilibrium; remove collectively dispensable fragments only after all-demand validation',validation_policy='One construction acceptance: retained positive-volume component satisfies all original demands, full continuous exits, north-hemisphere paths, and one connected solid. No exported-model replay. Ground reconstruction, installed support floor legality and strength deferred.')
    report['provenance']['code'].update(I.hashes([HERE/'step4.2/step42_connected.py']))
    report['provenance']['inputs'].update(I.hashes([search.archive/'data/report.json',search.archive/'remaining_support.obj',search.out/'data/warm_paths.npz']))
    save(out/'data/connectivity_search.json',dict(complete=True,proposals=search.evaluations,timeouts=search.timeouts,trace=search.connection_trace));report['artifacts']['connectivity_search.json']=I.sha256(out/'data/connectivity_search.json')
    save(p,report);I.check_report(p)
    lines=[f"# {search.group['id']}：Step4.2 承载、退出与连通共同通过",'', '**PASS：一个连通的正体积支撑，所有 pose 的全部 32,768 个原始力／力矩需求通过，退出不穿自身地面并完整脱离。**', '', f"初始 {result['pre_prune_component_count']} 个分量，保留一个能独立满足全部需求的材料块；删除当前承载条件下无需保留的材料 {result['pruned_volume_cm3']:.3f} cm³。", '', '不添加连接杆：有价值的分离块通过修改退出路径、恢复原始材料来处理；无需保留的块通过完整载荷检查后删除。接地材料覆盖重建和强度仍待处理，当前可删不等于所有后续条件下永远无用。', '', '只保留用户指定的退出方向变化视频与 poster；本求解入口不生成静态图片。原承载结果保存在内部 `data/before_connectivity/`，其无余量记录属于历史结果。']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    target=search.base/'step4/step4.2'
    for file in out.iterdir():
        if file.is_dir():
            for item in file.iterdir():shutil.copy2(item,target/'data'/item.name)
        else:shutil.copy2(file,target/file.name)
    I.check_report(target/'data/report.json');row.update(connected=True,component_count=1,pruned_volume_cm3=result['pruned_volume_cm3']);print('CONNECTED STEP4.2 PASS',row,flush=True);return row

def run_case(name,group):
    search=ConnectedSearch(name,group);began=time.monotonic()
    try:
        result=search.connected_search()
        if result is None:
            save(search.out/'data/unresolved.json',dict(complete=False,proposals=search.evaluations,trace=search.connection_trace));return dict(id=group['id'],passed=False,status='connected_search_unresolved')
        return finish_connected(search,result,began)
    except Exception as error:
        traceback.print_exc();save(search.out/'data/error.json',dict(error=str(error),trace=traceback.format_exc()));return dict(id=group['id'],passed=False,error=str(error))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--sets',nargs='+');parser.add_argument('--jobs',type=int,default=2);args=parser.parse_args();manifest=ROOT/'objects/B/pose_sets.json';groups=json.loads(manifest.read_text())['sets'];groups=[g for g in groups if not args.sets or g['id'] in args.sets]
    with ProcessPoolExecutor(max_workers=args.jobs,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(run_case,'B',g) for g in groups];rows=[f.result() for f in futures]
    save(HERE/'output/B/data/step42_connected_batch.json',dict(complete=all(r['passed'] for r in rows),sets=len(rows),passed_sets=sum(r['passed'] for r in rows),results=rows));print('CONNECTED STEP4.2 TOTAL',sum(r['passed'] for r in rows),'/',len(rows),flush=True)
if __name__=='__main__':main()
