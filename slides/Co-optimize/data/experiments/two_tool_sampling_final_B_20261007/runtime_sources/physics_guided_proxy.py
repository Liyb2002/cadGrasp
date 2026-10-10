"""Reuse verified primal/dual witnesses for the optimistic necessary cone."""
import numpy as np
from scipy.optimize import linprog


class CachedNecessaryCone:
    def __init__(self,rays,floors,targets,solve):
        self.rays,self.floors,self.targets,self.solve=rays,floors,targets,solve
        self.primal=[{} for _ in rays];self.dual=[[] for _ in rays]
        self.stats=dict(primal_reused=0,dual_reused=0,primal_solves=0,dual_solves=0)

    def check(self,keep,indices):
        for k in range(len(self.rays)):
            # Every stored plane separates an unchanged original target.
            # A plane remains valid precisely when none of its violating
            # potential generators are available; fixed floors were verified.
            for positive in self.dual[k]:
                if not keep[positive].any():self.stats['dual_reused']+=1;return False
            floor_count=len(self.floors[k]);available=np.r_[np.arange(floor_count),floor_count+np.flatnonzero(keep)]
            original=np.vstack([self.floors[k],self.rays[k]]);full=original[available]
            for index in sorted(indices[k]):
                support=self.primal[k].get(index)
                if support is not None and all(i<floor_count or keep[i-floor_count] for i in support):
                    self.stats['primal_reused']+=1;continue
                target=self.targets[k][index];self.stats['primal_solves']+=1
                try:witness=self.solve(full,target)
                except RuntimeError:witness=None
                if witness is not None:
                    self.primal[k][index]=available[np.asarray(witness['indices'],int)]
                    if len(self.primal[k])>256:self.primal[k].pop(next(iter(self.primal[k])))
                    continue
                # Seven variables, regardless of the number of contact rays.
                # Numerical failure merely allows exact construction; only a
                # verified separating plane can discard the proposal.
                self.stats['dual_solves']+=1
                lp=linprog(-target,A_ub=full,b_ub=np.zeros(len(full)),bounds=[(-1,1)]*7,
                    method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
                if lp.success and np.linalg.norm(lp.x)>1e-12:
                    plane=lp.x/np.linalg.norm(lp.x)
                    if (np.max(full@plane) if len(full) else 0.)<=1e-12 and target@plane>1e-8:
                        positive=np.flatnonzero(self.rays[k]@plane>1e-12)
                        self.dual[k].append(positive)
                        self.dual[k]=self.dual[k][-64:]
                        return False
        return True
