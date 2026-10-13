"""Keep geometry queries bounded without changing any geometric tolerance."""
from bounded_guard_pruner import BoundedGuardPruner


class MemoryGuardPruner(BoundedGuardPruner):
    def __init__(self,*args,**kwargs):
        import surface_flow
        from surface_flow_bounded import flow
        surface_flow.flow=flow
        super().__init__(*args,**kwargs)
