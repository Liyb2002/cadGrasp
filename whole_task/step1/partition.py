"""Shortest-boundary labelling of faces among patches (the cores of areas.py).

Each face gets a label (patch) that contains it.  The energy is the total length
of edges between faces of different labels, plus kappa * perimeter * tilt /
max_tilt per face, so a face trades at most kappa of its perimeter of boundary
against a full max tilt of needle angle.  Alpha-beta swaps solve each two-label
subproblem exactly by a minimum cut (Boykov, Veksler and Zabih 2001).
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order,maximum_flow

UNIT=1e-5;BIG=10**9


class Potts:
    def __init__(self,mesh,ids,beta,kappa):
        local=np.full(len(mesh.faces),-1);local[ids]=np.arange(len(ids))
        pairs=local[mesh.face_adjacency];keep=(pairs>=0).all(1);self.pairs=pairs[keep]
        w=np.linalg.norm(np.diff(mesh.vertices[mesh.face_adjacency_edges[keep]],axis=1)[:,0],axis=1)
        self.w=np.maximum(np.round(w/UNIT),1).astype(np.int64);self.n=len(ids)
        self.perimeter=np.zeros(self.n);np.add.at(self.perimeter,self.pairs.ravel(),np.repeat(w,2))
        self.beta=np.radians(beta);self.kappa=kappa

    def unary(self,tilt):
        """Integer face costs per pose (BIG where the pose cannot reach the face)."""
        ok=np.isfinite(tilt)
        cost=np.round(self.kappa*self.perimeter[:,None]*np.where(ok,tilt,0.)/self.beta/UNIT)
        return np.where(ok,cost,BIG).astype(np.int64)

    def energy(self,D,label):
        return int(D[np.arange(self.n),label].sum()+self.w[label[self.pairs[:,0]]!=label[self.pairs[:,1]]].sum())

    def boundary_m(self,label):
        return float(self.w[label[self.pairs[:,0]]!=label[self.pairs[:,1]]].sum()*UNIT)

    def swap(self,D,label,a,b):
        """Best relabelling of the faces now in a or b between a and b: one minimum cut."""
        P=np.flatnonzero((label==a)|(label==b));m=len(P)
        if m==0:return label
        index=np.full(self.n,-1);index[P]=np.arange(m);s,t=m,m+1
        cost_a=D[P,a].copy();cost_b=D[P,b].copy()
        u,v=self.pairs.T;iu,iv=index[u],index[v]
        for x,y,ix,iy in ((u,v,iu,iv),(v,u,iv,iu)):
            # An edge to a face outside a/b is a fixed cost of choosing the other pose.
            out=(ix>=0)&(iy<0)
            np.add.at(cost_a,ix[out],self.w[out]*(label[y[out]]!=a));np.add.at(cost_b,ix[out],self.w[out]*(label[y[out]]!=b))
        both=(iu>=0)&(iv>=0)
        rows=np.r_[np.full(m,s),np.arange(m),iu[both],iv[both]];cols=np.r_[np.arange(m),np.full(m,t),iv[both],iu[both]]
        cap=np.r_[np.minimum(cost_b,BIG),np.minimum(cost_a,BIG),self.w[both],self.w[both]].astype(np.int32)
        C=csr_matrix((cap,(rows,cols)),shape=(m+2,m+2));C.sum_duplicates()
        residual=C-maximum_flow(C,s,t,method='dinic').flow
        residual.data=(residual.data>0).astype(np.int8);residual.eliminate_zeros()
        source=np.zeros(m+2,bool);source[breadth_first_order(residual,s,directed=True,return_predecessors=False)]=True
        new=label.copy();new[P]=np.where(source[:m],a,b);return new

    def solve(self,D,label,sweeps=30):
        e=self.energy(D,label);K=D.shape[1]
        for _ in range(sweeps):
            improved=False
            for a in range(K):
                for b in range(a+1,K):
                    new=self.swap(D,label,a,b);f=self.energy(D,new)
                    if f<e:label,e,improved=new,f,True
            if not improved:break
        return label,e
