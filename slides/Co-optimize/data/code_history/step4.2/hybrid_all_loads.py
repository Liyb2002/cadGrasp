"""Sampling plus gradients using computed gaps for every original failed load."""
from hybrid_clearance_fast import *
from hybrid_stable import StableProjection
from physics_guided_batch_projection import all_projection_losses
import physics_guided_all_load_search as scheduling

class AllLoadGap(StableProjection):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.all_target_arrays=[U.target(task.targets) for task,T in self.states]
        self.all_gap_trace=[];self.relative_gap_limit=None

    def all_gaps(self,result):
        if 'all_original_projection_losses' not in result:
            rows=[];info=[];began=time.monotonic()
            for k in range(len(self.states)):
                loss,record=all_projection_losses(result['supplies'][k],self.all_target_arrays[k],failed=~result['masks'][k])
                rows.append(loss);info.append(record)
            result['all_original_projection_losses']=rows
            self.all_gap_trace.append(dict(serial=result['serial'],per_pose=info,
                computed_worst_gap=max(float(a.max()) for a in rows),seconds=time.monotonic()-began))
        return result['all_original_projection_losses']

    def choose_loads(self,result):
        losses=self.all_gaps(result)
        for k,(loss,mask) in enumerate(zip(losses,result['masks'])):
            failed=np.flatnonzero(~mask)
            indices=failed[np.argsort(loss[failed])[-6:]].tolist() if len(failed) else []
            self.load_bank.update((k,int(i)) for i in indices)
            self.proxy_indices[k]=set(indices)
            self.selection_trace.append(dict(serial=result['serial'],pose=self.group['poses'][k],indices=indices,
                scanned=len(mask),projected=int((~mask).sum()),selection='all original failed loads; individually or active-region projected and all-ray KKT checked'))
        return sorted(self.load_bank)

    def actual_loss(self,result,loads):
        losses=self.all_gaps(result)
        return max(float(a.max()) for a in losses),[float(losses[k][i]) for k,i in loads]

    def score(self,result):
        fractions=np.array(result['counts'])/32768
        return (bool(all(m.all() for m in result['masks'])),
                -max(float(a.max()) for a in self.all_gaps(result)),float(fractions.sum()))

    def remember(self,result,common=None):
        RefinedHybridSearch.remember(self,result,common)

class AllLoadGradientBranch(AllLoadGap,GradientBranch):
    pass

class AllLoadHybridSearch(AllLoadGap,ClearanceFastHybridSearch):
    optimize=scheduling.all_load_search

    def __init__(self,*args,**kwargs):
        scheduling.GradientBranch=AllLoadGradientBranch
        super().__init__(*args,**kwargs)
        self.additional_code += [Path(__file__),HERE/'step4.2/hybrid_stable.py',HERE/'helper_func/physics_guided_cone_svd.py',
            HERE/'helper_func/physics_guided_batch_projection.py',HERE/'helper_func/physics_guided_all_load_search.py']
        self.additional_artifacts += ['all_load_gap_trace.json']

    def finish(self,result,*args,**kwargs):
        save(self.out/'all_load_gap_trace.json',self.all_gap_trace)
        self.report_extra.update(gap_policy='every original failed load projected with original rays; all-ray KKT checks; no load sampling for gap evaluation',
            computed_worst_gap=max(float(a.max()) for a in self.all_gaps(result)),
            gradient_eligibility_policy='diverse actual states ranked by computed all-original-load worst gap; no passed-load-count gate',
            cone_projection_backend='SVD active sets with verified batched regions and individual fallbacks')
        return super().finish(result,*args,**kwargs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object',default='B');parser.add_argument('--set',required=True)
    parser.add_argument('--directions',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=12);parser.add_argument('--max-proposals',type=int,default=1200)
    args=parser.parse_args();source=ROOT/'objects'/args.object
    groups=json.loads((source/'pose_sets.json').read_text())['sets']
    if (source/'illegal_pose_sets.json').exists():
        groups += [dict(g,id='illegal/'+g['id']) for g in json.loads((source/'illegal_pose_sets.json').read_text())['sets']]
    group=next(g for g in groups if g['id']==args.set)
    search=AllLoadHybridSearch(args.object,group,out=args.out,directions=args.directions,max_proposals=args.max_proposals)
    try:search.optimize(args.iterations)
    except (RuntimeError,ValueError) as error:
        save(search.out/'unresolved.json',dict(status='numerically_unresolved',passed=False,pose_set=group['id'],error=str(error)));raise

if __name__=='__main__':main()
