"""Finite routing-guide simplification; actual clipped solids remain decisive."""
import heapq
import hashlib
import numpy as np
from continuous_legs import RouteGuide as BaseGuide


def simplify_path(path,valid):
    selected=[path[0]];k=0;fallbacks=[]
    while k<len(path)-1:
        stop=min(len(path)-1,k+24)
        for j in range(stop,k,-1):
            n=max(2,int(np.linalg.norm(path[j]-path[k])*2)+1)
            samples=np.rint(np.linspace(path[k],path[j],n)).astype(int)
            if valid[tuple(samples.T)].all():
                selected.append(path[j]);k=j;break
        else:
            # Adjacent diagonal A* endpoints can be valid while round-to-even
            # interpolation picks a forbidden orthogonal cell. The old loop
            # never changed k in this situation. Keep that original discrete
            # edge as a proposal and advance; native clipping/connectivity
            # must still certify the actual connector, or reject it.
            fallbacks.append(dict(start=path[k].tolist(),end=path[k+1].tolist()))
            selected.append(path[k+1]);k+=1
    return np.asarray(selected),fallbacks


class FiniteRouteGuide(BaseGuide):
    def route(self,target,limit=250000):
        center=np.rint((target-self.origin)/self.pitch).astype(int)
        offsets=np.array([(x,y,z) for x in range(-3,4) for y in range(-3,4) for z in range(-3,4)])
        ids=center+offsets;ids=ids[np.all((ids>=0)&(ids<self.shape),axis=1)]
        ids=ids[self.valid[tuple(ids.T)]]
        if not len(ids):raise ValueError('Target has no legal routing seed')
        distances=np.linalg.norm(self.origin+ids*self.pitch-target,axis=1)
        start=tuple(ids[np.argmin(distances)])
        heap=[(0.,0.,start)];best={start:0.};previous={};end=None;explored=0
        while heap and explored<limit:
            _,cost,key=heapq.heappop(heap)
            if cost!=best.get(key):continue
            explored+=1
            if self.roots[key]:end=key;break
            neighbors=np.asarray(key)+self.directions
            valid=np.all((neighbors>=0)&(neighbors<self.shape),axis=1)
            indices=np.flatnonzero(valid);neighbors=neighbors[indices]
            valid=self.valid[tuple(neighbors.T)];neighbors,indices=neighbors[valid],indices[valid]
            clear=self.clearance[tuple(neighbors.T)]
            costs=cost+self.lengths[indices]*(1.+(.0025/np.maximum(clear,.0008))**2)
            xyz=self.origin+neighbors*self.pitch;hints=self.root_tree.query(xyz)[0]
            for point,next_cost,hint in zip(neighbors,costs,hints):
                neighbor=tuple(point)
                if next_cost<best.get(neighbor,float('inf')):
                    best[neighbor]=float(next_cost);previous[neighbor]=key
                    heapq.heappush(heap,(float(next_cost+1.05*hint),float(next_cost),neighbor))
        if end is None:raise ValueError('No bounded route to the certified core')
        path=[end]
        while path[-1]!=start:path.append(previous[path[-1]])
        path=np.asarray(path[::-1]);selected,fallbacks=simplify_path(path,self.valid)
        xyz=self.origin+selected*self.pitch
        if fallbacks:print('GUIDE_CORNER_PROGRESS',fallbacks,flush=True)
        return np.vstack([target,xyz]),dict(explored_voxels=explored,guide_pitch_m=self.pitch,
            path_length_m=float(np.linalg.norm(np.diff(xyz,axis=0),axis=1).sum()),
            discrete_path_sha256=hashlib.sha256(path.tobytes()).hexdigest(),
            corner_progress_edges=fallbacks,actual_clipped_connection_must_be_checked=True)
