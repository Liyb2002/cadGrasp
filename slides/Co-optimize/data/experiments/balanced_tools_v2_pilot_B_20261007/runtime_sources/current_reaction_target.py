"""Joint force-balanced contact guidance for small direction/translation steps.

LP reactions weight contact shadow boundaries, without force capacities. The
actual all-load/solid checks remain the only success criterion.
"""
import numpy as np
import igl
from scipy.special import logsumexp
from scipy.optimize import minimize
from contact_recovery import cooperating_reactions


from joint_placement_target import JointPlacementTarget

class CurrentReactionTarget(JointPlacementTarget):
    def __init__(self,search,current,directions,offsets):
        self.search=search;self.selected=[];self.records=[];self.serial=current['serial'] if current else None
        targets=search.refresh_targets(current)
        by_pose={}
        for k,i,target in targets:by_pose.setdefault(k,[]).append((i,target))
        for owner,entries in by_pose.items():
            # Failed worst load plus a small preserved-load bank per pose.
            ranked=sorted(entries,key=lambda row:search.projection(current,owner,row[0])['loss'] if current else 0.,reverse=True)
            picked=ranked[:3]
            costs=self.costs(owner,search.points,search.outward,directions,offsets)
            avoided=getattr(search,'guidance_avoided',{}).get(owner,[])
            if len(avoided):costs[np.asarray(avoided,int)]+=1.
            weight=np.zeros(len(search.points))
            for index,target in picked:
                failed=current is not None and not current['masks'][owner][index]
                free=current['supplies'][owner] if failed else search.floors[owner]
                reactions,error=cooperating_reactions(free,search.rays[owner],target,costs)
                if reactions.sum()>1e-12:weight+=(1. if failed else .05)*reactions/reactions.sum()
                self.records.append(dict(pose_index=owner,load_index=index,equilibrium_error=error))
            ids=np.flatnonzero(weight>1e-10)
            if len(ids):self.selected.append((owner,ids,weight[ids]))
        total=sum(weight.sum() for owner,ids,weight in self.selected)
        if total<=0:raise ValueError('no reaction guidance contacts')
        self.selected=[(owner,ids,weight/total) for owner,ids,weight in self.selected]
        self.key=tuple((owner,tuple(map(int,ids))) for owner,ids,weight in self.selected)

