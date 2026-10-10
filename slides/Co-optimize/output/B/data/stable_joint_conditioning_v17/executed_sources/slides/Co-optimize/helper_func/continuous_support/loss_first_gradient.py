"""Continuous whole-loss descent before discrete full-load feasibility.

The original sample count is a diagnostic during infeasible descent: a lower
integral may temporarily uncover more near-boundary samples. Feasible states
always dominate; material reduction keeps every original sample feasible.
"""
import numpy as np
from whole_search.search import failed_count, lost_protected
from whole_search.reuse_first import juxtaposed_count
from .expanded_gradient import ExpandedGradientModel, ExpandedGradientSearch


class LossFirstGradientSearch(ExpandedGradientSearch):
    def score(self,result,anchor=None):
        missing=failed_count(result)
        loss=self.model.proxy(result['layout'])['loss'] if missing else 0.
        if self.model.volume_delta is not None:self.model.refresh_volume(result)
        return (lost_protected(anchor,result),int(missing>0),loss,missing,
                result['volume_cm3'],result['maximum_projected_footprint_m2'],juxtaposed_count(result['layout']))

    def refine(self, current, rounds, anchor=None, translation_guests=(), phase='direction'):
        for iteration in range(rounds):
            if failed_count(current) == 0: break
            targets, worst, scan = self.targets(current)
            proposals, base = self.local_proposals(current, targets, worst, translation_guests)
            for kind, _, detail in proposals:
                if kind.startswith('translation'): detail['body_extent_m'] = self.model.extent
            row = dict(phase=phase, iteration=iteration+1, before_counts=current['counts'],
                       hardest=worst, hardest_scan=scan, proxy_before=base,
                       screened_candidates=0, skipped_candidates=0, trials=[], groups=[])
            best = current; before_failed = failed_count(current); seen = set()
            groups = self.proposal_groups(proposals, getattr(self, 'fine_resolution', False))
            for label, group in groups:
                ranked = []
                for kind, layout, detail in group:
                    key = layout.key()
                    if key in seen or key == current['layout'].key(): continue
                    seen.add(key); proxy = self.model.proxy(layout)
                    row['screened_candidates'] += 1
                    loss_progress = proxy['loss'] < base['loss'] - max(1e-30, base['loss']*1e-8)
                    if loss_progress or 'gradient' not in kind:
                        ranked.append((proxy['loss'], kind, layout, detail, proxy))
                ranked.sort(key=lambda r: r[0])
                row['groups'].append(dict(kind=label, screened=len(group), finalist_count=min(2,len(ranked))))
                for _, kind, layout, detail, proxy in ranked[:2]:
                    record = dict(operation=kind, detail=detail, proxy=proxy, accepted=False)
                    try:
                        trial = self.model.evaluate(layout); count = failed_count(trial)
                        loss_progress = proxy['loss'] < base['loss'] - max(1e-30, base['loss']*1e-8)
                        eligible = loss_progress or (proxy['loss'] <= 1e-28 and count < before_failed)
                        record.update(counts=trial['counts'], serial=trial['serial'],
                                      loss_progress=loss_progress, lost_protected_loads=lost_protected(anchor, trial))
                        if eligible and lost_protected(anchor, trial) == 0 and self.score(trial,anchor) < self.score(current,anchor):
                            best = trial; record['accepted'] = True
                    except (RuntimeError, ValueError, AssertionError) as error:
                        record['error'] = str(error)
                    row['trials'].append(record)
                    if best is not current: break
                if best is not current: break
            accepted = best is not current
            if accepted:
                current = best; self.checkpoint(current, phase)
            row.update(accepted=accepted, after_counts=current['counts'],
                       skipped_candidates=max(0,len(proposals)-len(seen)))
            self.record(row)
            print('CONTINUOUS LOSS GRADIENT', phase, iteration+1, 'accepted', accepted,
                  'failed', failed_count(current), 'screened', row['screened_candidates'],
                  'skipped', row['skipped_candidates'], flush=True)
            if not accepted: break
        self.model.commit(current)
        return current
