"""Bound the fallback Boolean sequence; retain every original physical check."""
from serialized_guard_pruner import SerializedGuardPruner


class BoundedGuardPruner(SerializedGuardPruner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.budget=min(self.budget,12)
