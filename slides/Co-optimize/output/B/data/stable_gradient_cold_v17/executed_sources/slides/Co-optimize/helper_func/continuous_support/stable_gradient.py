"""Compare every local step and Juxtapose branch on ONE demand measure.

Adaptive nodes can be refreshed between outer decisions. Within a descent
block, including all competing jump refinements, both covered and uncovered
strata stay fixed. A branch cannot select the quadrature that judges itself.
"""
from contextlib import contextmanager
from .paired_blocker_gradient import PairedBlockerGradientSearch
from .releasable_blockers import ReleasableBlockerModel


class StableGradientModel(ReleasableBlockerModel):
    def set_working_measure(self, current):
        if not getattr(self, 'measure_hold_depth', 0):
            super().set_working_measure(current)

    @contextmanager
    def hold_measure(self, current):
        self.set_working_measure(current)
        self.measure_hold_depth = getattr(self, 'measure_hold_depth', 0) + 1
        try:
            yield
        finally:
            self.measure_hold_depth -= 1


class StableGradientSearch(PairedBlockerGradientSearch):
    def refine(self, current, *args, **kwargs):
        with self.model.hold_measure(current):
            return super().refine(current, *args, **kwargs)

    def rescue(self, current, *args, **kwargs):
        with self.model.hold_measure(current):
            return super().rescue(current, *args, **kwargs)
