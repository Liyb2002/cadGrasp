"""A bounded structural escape keeps continuous descent and jump ranking apart.

Direction/Translation always descend the whole integral. Discrete Juxtapose
branches may choose broader demand coverage, even with a larger integral;
this is an explicit structural escape from a local basin, never a gradient.
"""
from .balanced_gradient import StableGradientModel, BalancedGradientSearch
from .descent_fast_gradient import DescentGradientSearch
from whole_search.search import failed_count, lost_protected
from whole_search.reuse_first import juxtaposed_count


class JumpRankedGradientSearch(BalancedGradientSearch):
    def score(self, result, anchor=None):
        if not getattr(self, 'ranking_discrete_jump', False):
            return super().score(result, anchor)
        missing = failed_count(result)
        loss = self.model.proxy(result['layout'])['loss'] if missing else 0.
        if result.get('evaluation') and self.model.volume_delta is not None:
            self.model.refresh_volume(result)
        return (lost_protected(anchor, result), missing, loss,
                result['volume_cm3'], result['maximum_projected_footprint_m2'],
                juxtaposed_count(result['layout']))

    def refine(self, current, *args, **kwargs):
        old = getattr(self, 'ranking_discrete_jump', False)
        self.ranking_discrete_jump = False
        try:
            return super().refine(current, *args, **kwargs)
        finally:
            self.ranking_discrete_jump = old

    def rescue(self, current, *args, **kwargs):
        old = getattr(self, 'ranking_discrete_jump', False)
        self.ranking_discrete_jump = True
        try:
            return super().rescue(current, *args, **kwargs)
        finally:
            self.ranking_discrete_jump = old

    def solve_frontier(self, current, anchor=None, insertion_guest=None):
        # Failed guests get the existing bounded pool first. Passing blockers
        # enter only after this structural pool stalls, as in the successful
        # original search; all branch refinements still include every pose.
        return DescentGradientSearch.solve_frontier(self, current, anchor, insertion_guest)
