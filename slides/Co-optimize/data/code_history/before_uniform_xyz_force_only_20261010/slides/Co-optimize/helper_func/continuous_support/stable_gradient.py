"""Compare every local step and Juxtapose branch on ONE demand measure.

Adaptive nodes can be refreshed between outer decisions. Within a descent
block, including all competing jump refinements, both covered and uncovered
strata stay fixed. A branch cannot select the quadrature that judges itself.
"""
from contextlib import contextmanager
from .paired_blocker_gradient import PairedBlockerGradientSearch
from .releasable_blockers import ReleasableBlockerModel
from translation.volume_descent import polish_xyz


class StableGradientModel(ReleasableBlockerModel):
    def __init__(self,*args,allow_z_translation=True,**kwargs):
        super().__init__(*args,allow_z_translation=allow_z_translation,**kwargs)

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
    def polish_volume(self,current,rounds=2):
        current=super().polish_volume(current,rounds)
        return polish_xyz(self,current,rounds=2)

    def refine(self, current, *args, **kwargs):
        with self.model.hold_measure(current):
            return super().refine(current, *args, **kwargs)

    def rescue(self, current, *args, **kwargs):
        with self.model.hold_measure(current):
            return super().rescue(current, *args, **kwargs)
