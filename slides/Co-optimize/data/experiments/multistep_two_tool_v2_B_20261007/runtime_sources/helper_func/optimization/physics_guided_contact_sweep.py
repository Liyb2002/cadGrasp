"""Nominal contact-core shadowing; clearance outside cores is checked on the true solid."""
import numpy as np
from scipy.special import logsumexp
from physics_guided_geometry import SweepDistanceModel


class NominalContactSweep(SweepDistanceModel):
    def __init__(self,clearance,points,normals,length,extent):
        super().__init__(clearance,points,normals,length,extent,depth=.0005)
        self.margin=0.
        self.temperature=extent*.001
        self.reference=self.field.sample(np.asarray(points))

    def distances(self,direction):
        self.evaluations+=1
        positions=self.probes[:,None,:]-self.times[None,:,None]*direction
        values=self.field.sample(positions.reshape(-1,3)).reshape(len(self.probes),-1)-self.reference[:,None]
        penetration=self.temperature*np.logaddexp(0.,values/self.temperature)
        # Smooth upper approximation to maximum positive penetration, without
        # the large additive tau*log(time_count) bias of signed smooth-max.
        return np.exp(logsumexp(8*np.log(np.maximum(penetration,1e-30)),axis=1)/8)/self.extent
