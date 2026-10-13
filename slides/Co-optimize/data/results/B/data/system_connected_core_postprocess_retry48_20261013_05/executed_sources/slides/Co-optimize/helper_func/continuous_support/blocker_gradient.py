"""Jointly consider failed guests and passing geometric blockers for jumps.

A native passing pose cannot translate before explicit Juxtapose. Its work
cone/body can nevertheless obstruct another pose. Do not defer its discrete
relocation indefinitely while failed-guest branches make small loss progress.
"""
from .loss_first_gradient import ExpandedGradientModel,LossFirstGradientSearch
from whole_search.search import failed_count
from whole_search.reuse_first import registered


class BlockerGradientSearch(LossFirstGradientSearch):
    def jump_guests(self,current):
        self.model.set_working_measure(current)
        base=self.model.proxy(current['layout']);active=list(current['layout'].active)
        failing=sorted([k for k in active if not current['masks'][k].all()],
                       key=lambda k:-base['residual_loss'][active.index(k)])[:2]
        ordered,scores=self.model.useful_coordinates(current)
        passing=[k for k in ordered if current['masks'][k].all() and scores[k]>0.]
        other=[k for k in ordered if k not in failing+passing and scores[k]>0.]
        blockers=(passing+other)[:2]
        guests=list(dict.fromkeys(failing+blockers))
        return guests,blockers,dict(failing=[self.model.poses[k] for k in failing],
            blockers=[self.model.poses[k] for k in blockers],
            release_scores={self.model.poses[k]:scores[k] for k in active},
            all_original_demand_measure_in_score=True)

    def solve_frontier(self,current,anchor=None,insertion_guest=None):
        movable=lambda q:[k for k in q['layout'].active if not registered(q['layout'],k)]
        current=self.refine(current,3,translation_guests=movable(current),phase='rotate_reuse_direction')
        for iteration in range(self.iterations):
            if failed_count(current)==0:break
            if failed_count(current)<=512:
                current=self.fine_refine(current,rounds=2)
                if failed_count(current)==0:break
            before=current;guests,blockers,detail=self.jump_guests(current)
            self.record(dict(phase='whole_jump_pool',iteration=iteration+1,
                             before_counts=current['counts'],**detail))
            # One shared 96-slot screen and three branches, rather than two
            # separate pools. Passing blockers are explicitly authorized guests.
            current=self.rescue(current,guests,anchor=None,blocking_guests=blockers)
            if current is before:break
            current=self.refine(current,2,translation_guests=movable(current),phase='all_movable_gradient_repair')
        if failed_count(current):
            current=self.refine(current,3,translation_guests=movable(current),phase='final_joint_gradient_repair')
        if 0<failed_count(current)<=512:current=self.fine_refine(current,rounds=3)
        return current
