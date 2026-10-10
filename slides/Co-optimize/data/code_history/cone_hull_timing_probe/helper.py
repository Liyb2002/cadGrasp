"""Conservative original-ray subsets for inexpensive proposal projection.

For a fixed contact normal, wrench torque is affine in contact position;
interior positions are redundant in exact arithmetic. Retain actual sampled
extreme rays, never synthesize a reaction. This cache is guidance only; the
original full sampled classifier and final actual worker still use all rays.
"""
from collections import OrderedDict
import hashlib
import time
import numpy as np
from scipy.spatial import ConvexHull,QhullError


class ConeHullCache:
    def __init__(self,model):
        self.model=model;self.groups=[];self.cache=OrderedDict();self.seconds=0.
        self.original_rays=0;self.kept_rays=0
        for rays in model.point_rays:
            # Exact equality of force and original seventh coordinate.
            _,inverse=np.unique(rays[:,[0,1,2,6]],axis=0,return_inverse=True)
            groups=[]
            for group in range(int(inverse.max())+1):
                ids=np.flatnonzero(inverse==group);torques=rays[ids,3:6]
                center=torques.mean(0);_,singular,axes=np.linalg.svd(torques-center,full_matrices=False)
                rank=int(np.count_nonzero(singular>max(1e-14,singular.max(initial=0.)*1e-12)))
                groups.append((ids,(torques-center)@axes[:rank].T,rank))
            self.groups.append(groups)

    def supply(self,owner,flags):
        key=owner,hashlib.sha256(flags.tobytes()).digest()
        if key in self.cache:
            self.cache.move_to_end(key);return self.cache[key]
        start=time.monotonic();selected=[]
        for ids,coordinates,rank in self.groups[owner]:
            local=np.flatnonzero(flags[ids])
            if len(local)>rank+1 and rank in [1,2]:
                points=coordinates[local]
                if rank==1:
                    keep=np.unique([points[:,0].argmin(),points[:,0].argmax()])
                else:
                    try:keep=ConvexHull(points).vertices
                    except QhullError:keep=np.arange(len(local))
                local=local[keep]
            selected.extend(ids[local])
        selected=np.asarray(sorted(selected),int)
        floors=self.model.floors[owner]
        rays=np.vstack([floors,self.model.point_rays[owner][selected]])
        columns=np.r_[np.arange(len(floors)),len(floors)+selected]
        self.cache[key]=rays,columns
        while len(self.cache)>512:self.cache.popitem(last=False)
        self.seconds+=time.monotonic()-start
        self.original_rays+=len(floors)+int(flags.sum());self.kept_rays+=len(rays)
        return rays,columns
