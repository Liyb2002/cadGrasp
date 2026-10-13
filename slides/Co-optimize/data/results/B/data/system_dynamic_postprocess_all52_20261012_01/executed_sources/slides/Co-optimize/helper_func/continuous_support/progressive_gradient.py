"""Lazy gradient line search; sample contact events only after descent stalls.

Every candidate loss includes all poses. Each committed step retains the
original all-load check. Generating a candidate does not evaluate its loss:
later line scales and sample pools are skipped after the first accepted step.
"""
import numpy as np
from whole_search.search import failed_count, lost_protected
from .active_fast_gradient import ActiveGradientModel, ActiveGradientSearch


class ProgressiveGradientSearch(ActiveGradientSearch):
    @staticmethod
    def proposal_groups(rows, fine=False):
        gradients = [r for r in rows if 'gradient' in r[0]]
        samples = [r for r in rows if 'gradient' not in r[0]]
        # Try a moderate step before backtracking or crossing a larger cell.
        # Translation uses its own dimensionless body fraction.
        direction_scales = [.125, .03125, .25] if fine else [2., .5, 8.]
        translation_scales = [1/512, 1/1024, 1/256] if fine else [1/32, 1/128, 1/8]
        groups = []
        for degrees, fraction in zip(direction_scales, translation_scales):
            group = [r for r in gradients if
                     (r[0].startswith('direction') and r[2].get('step_degrees') == degrees) or
                     (r[0].startswith('translation') and
                      np.isclose(r[2].get('step_m', 0.), r[2].get('body_extent_m', 1.)*fraction,
                                 rtol=1e-10, atol=0.))]
            if group: groups.append(('gradient_line_search', group))
        # Coherent samples release overlapping blockers together. Individual
        # samples then inspect the contact events missed by the secant.
        coherent = [r for r in samples if 'coherent' in r[0]]
        individual = [r for r in samples if 'coherent' not in r[0]]
        for label, group in [('coherent_contact_samples', coherent),
                             ('individual_contact_samples', individual)]:
            for start in range(0, len(group), 16):
                groups.append((label, group[start:start+16]))
        return groups

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
                        eligible = count < before_failed or (count == before_failed and loss_progress)
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
            print('LAZY GRADIENT', phase, iteration+1, 'accepted', accepted,
                  'failed', failed_count(current), 'screened', row['screened_candidates'],
                  'skipped', row['skipped_candidates'], flush=True)
            if not accepted: break
        self.model.commit(current)
        return current
