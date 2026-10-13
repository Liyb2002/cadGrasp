"""First acceptable gradient line step, with sampled plateau alternatives.

Whole-demand distance ranks candidates inexpensively. The original full
sampled-load check runs for the first improving gradient, then alternatives
only if needed. This avoids checking three accepted line steps per round.
Final original geometry and full actual loads remain mandatory.
"""
from .descent_fast_gradient import CachedGradientModel,DescentGradientSearch
from whole_search.search import failed_count


class EfficientGradientSearch(DescentGradientSearch):
    def refine(self,current,rounds,anchor=None,translation_guests=(),phase='direction'):
        # Infeasible equal-count steps must improve demand distance. Material
        # alone cannot justify spending another round on a flat error plateau.
        from whole_search.search import lost_protected
        for iteration in range(rounds):
            if failed_count(current)==0:break
            targets,worst,scan=self.targets(current)
            proposals,base=self.local_proposals(current,targets,worst,translation_guests)
            selected,screened=self.shortlist(proposals,targets)
            selected.sort(key=lambda r:(0 if 'gradient' in r[3] and r[0]<base['loss']-max(1e-30,base['loss']*1e-8) else 1,r[0]))
            row=dict(phase=phase,iteration=iteration+1,before_counts=current['counts'],
                     hardest=worst,hardest_scan=scan,proxy_before=base,screened_candidates=screened,trials=[])
            best=current;before_failed=failed_count(current)
            for _,_,_,kind,layout,detail,proxy in selected:
                record=dict(operation=kind,detail=detail,proxy=proxy,accepted=False)
                try:
                    trial=self.model.evaluate(layout);count=failed_count(trial)
                    loss_progress=proxy['loss']<base['loss']-max(1e-30,base['loss']*1e-8)
                    eligible=count<before_failed or (count==before_failed and loss_progress)
                    record.update(counts=trial['counts'],rank=self.score(trial,anchor),serial=trial['serial'],
                                  lost_protected_loads=lost_protected(anchor,trial),loss_progress=loss_progress)
                    if eligible and self.score(trial,anchor)<self.score(best,anchor):best=trial;record['eligible']=True
                except (RuntimeError,ValueError,AssertionError) as error:record['error']=str(error)
                row['trials'].append(record)
                if best is not current and lost_protected(anchor,best)==0:break
            accepted=best is not current
            if accepted:
                current=best;self.checkpoint(current,phase)
                for record in row['trials']:record['accepted']=record.get('serial')==best['serial']
            row.update(accepted=accepted,after_counts=current['counts']);self.record(row)
            print('GRADIENT REFINE',phase,iteration+1,'accepted',accepted,'failed',failed_count(current),flush=True)
            if not accepted:break
        self.model.commit(current)
        return current
