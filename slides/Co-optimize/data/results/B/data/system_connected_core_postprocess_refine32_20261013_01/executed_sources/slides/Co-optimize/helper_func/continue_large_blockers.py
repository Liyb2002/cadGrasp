"""Own-state whole continuation with hardest-demand-weighted blocker selection."""
import _bootstrap
import argparse
from co_common import *
import continue_large_pose_sets as continuation
from run_large_pose_sets import groups
from whole_pipeline import WholeSearch


class DemandWeightedSearch(WholeSearch):
    def blocking_poses(self,current):
        m=self.model;layout=current['layout'];contact=m.contact_delta.state(layout)
        worst,_=m.hardest(current)
        if worst is None:return []
        owner=worst['pose_index'];residual=np.asarray(worst['residual'])
        helpful=np.maximum(0.,-m.point_rays[owner]@residual)
        passing=[k for k in layout.active if current['masks'][k].all()]
        ranked=[]
        for blocker in passing:
            locked=(contact.locks[owner,blocker]&(contact.coverage_counts[owner]>0)
                    &m.contact_allowed(layout,owner))
            # A unique blocker can release a patch immediately; shared locks
            # retain a smaller score for coordinated later operations.
            divisor=np.maximum(contact.lock_counts[owner],1)
            value=float(np.sum(m.point_areas[locked]*helpful[locked]/divisor[locked]))
            points=int(np.count_nonzero(locked&(helpful>1e-12)))
            if value>0:ranked.append((value,blocker,points))
        ranked.sort(reverse=True)
        if not ranked:return super().blocking_poses(current)
        self.record(dict(phase='hardest_demand_blocker_priority',hardest=worst,
            ranking=[dict(pose=m.poses[k],weighted_release_score=score,helpful_locked_points=points)
                     for score,k,points in ranked],
            formula='sum(area_weight * max(0, -wrench_column dot hardest_cone_residual) / lock_count)',
            changes_candidate_priority_only=True))
        return ranked

    def save(self,*args,**kwargs):
        kwargs['blocker_selection']='hardest-demand reduced-cost weighted contact availability'
        report=super().save(*args,**kwargs)
        if 'provenance' in report:report['provenance']['code'].update(I.hashes([Path(__file__)]))
        return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('case')
    parser.add_argument('--iterations',type=int,default=8);args=parser.parse_args()
    group=next(g for g in groups() if g['id']==args.case or g['source_case']==args.case)
    # Dependency injection is local to this standalone continuation process;
    # the frozen production solver and other running cases are untouched.
    continuation.WholeSearch=DemandWeightedSearch
    row=continuation.run_case((group,dict(iterations=args.iterations,finalists=6,
        branch_rounds=4,volume_rounds=3,screen_budget=128,real_budget=3,seed=44)))
    print('WEIGHTED_CONTINUATION',row,flush=True)


if __name__=='__main__':main()
