"""Bounded expansion after fixed-area selection, only above 98% in every task."""
import numpy as np

THRESHOLD_PERCENT = 98
AREA_FACTORS = (1.01, 1.02, 1.05, 1.10)


def policy():
    return dict(threshold_per_pose=THRESHOLD_PERCENT/100,
                threshold_comparison='strictly greater in every pose',
                target_area_factors=list(AREA_FACTORS), max_area_factor=max(AREA_FACTORS),
                fixed_centers_and_head_count=True, shrinking_allowed=False,
                objective='complete the original fixed samples; no coverage/area objective',
                scope='terminal step only, after fixed 1% sampling has stopped')


def eligible(score):
    # Compare integer counts, not rounded percentages or the two-task mean.
    return all(100*c > THRESHOLD_PERCENT*n
               for c, n in zip(score['covered_counts'], score['sample_counts']))


def summary(score):
    return {k: v for k, v in score.items() if k not in ('masks', 'classifiers', 'gravity')}


def run(search, entries):
    entries = list(entries)
    score = search.evaluate(entries)
    report = dict(policy=policy(), attempted=False, before=summary(score),
                  trials=[], updates=[], area_changed=False)

    def finish(status):
        report.update(status=status, after=summary(score))
        return entries, report

    if score['both_sampled_complete']:
        return finish('already_complete')
    if not eligible(score):
        return finish('sample_coverage_below_expansion_threshold')
    report['attempted'] = True
    for factor in AREA_FACTORS:
        pending = list(range(len(entries)))
        while pending:
            options = []
            for index in pending:
                old = entries[index]
                trial = search.geometry.expand(old, factor)
                proposal = entries.copy()
                proposal[index] = trial
                check = search.geometry.group_check(proposal)
                record = dict(index=index, candidate_id=old['contact']['candidate_id'],
                              target_area_factor=factor, accepted=False, geometry=check)
                report['trials'].append(record)
                if not trial['valid'] or not check['passed']:
                    record['reason'] = trial['reason'] if not trial['valid'] else check['reason']
                    continue
                if trial['contact']['radius_m'] < old['contact']['radius_m'] or trial['area'] < old['area']:
                    record['reason'] = 'not_an_expansion'
                    continue
                tested = search.evaluate(proposal)
                record['score'] = summary(tested)
                if any(np.any(before & ~after) for before, after in zip(score['masks'], tested['masks'])):
                    record['reason'] = 'sample_coverage_regressed'
                    continue
                record['reason'] = 'valid_expansion'
                options.append((index, trial, tested, record))
            if not options:
                break
            # Choose the largest actual coverage gain at this small area level.
            # Zero-gain growth can participate in a later complementary update.
            index, trial, score, record = min(options, key=lambda x: (-x[2]['mean_coverage'], x[0]))
            entries[index] = trial
            pending.remove(index)
            record['accepted'] = True
            report['area_changed'] = True
            report['updates'].append(dict(candidate_id=trial['contact']['candidate_id'],
                radius_m=trial['contact']['radius_m'], area_m2=trial['area'],
                area_fraction=trial['area_fraction'], target_area_factor=factor,
                score=summary(score)))
            if score['both_sampled_complete']:
                return finish('completed_by_expansion')
    return finish('terminal_expansion_exhausted')
