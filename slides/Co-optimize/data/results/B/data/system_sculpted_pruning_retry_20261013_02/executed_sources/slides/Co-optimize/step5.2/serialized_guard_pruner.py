"""Reject serialized changes to original stock, mandatory core or locked BBX."""
import numpy as np
from sculpted_pruning import SculptedPruner
from deterministic_base import solid,volume


class SerializedGuardPruner(SculptedPruner):
    def accept(self,candidate,operation):
        before=(self.body,getattr(self,'best_mesh',None),self.proof)
        if not super().accept(candidate,operation):return False
        value=solid(self.best_mesh);expected=self.bounds(self.fixture);actual=self.bounds(self.best_mesh)
        checks=dict(added_to_initial_stock_m3=volume(value-self.original),mandatory_loss_m3=volume(self.mandatory-value),
            max_bbx_bound_difference_m=float(np.max(abs(actual-expected))),
            bbx_volume_difference_l=float(abs(np.prod(actual[1]-actual[0])-np.prod(expected[1]-expected[0]))*1000))
        if (checks['added_to_initial_stock_m3']>1e-10 or checks['mandatory_loss_m3']>1e-10 or
                checks['max_bbx_bound_difference_m']>1e-8 or checks['bbx_volume_difference_l']>1e-8):
            self.body,self.best_mesh,self.proof=before
            self.history[-1].update(accepted=False,serialized_guard_rejected=True,serialized_guard_checks=checks,
                error='Serialized candidate changed original stock, mandatory core or locked BBX; prior certified body retained')
            return False
        self.history[-1]['serialized_guard_checks']=checks
        return True
