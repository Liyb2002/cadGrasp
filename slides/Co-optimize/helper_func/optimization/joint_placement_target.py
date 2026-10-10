"""Joint force-balanced contact guidance for small direction/translation steps.

LP reactions weight contact shadow boundaries, without force capacities. The
actual all-load/solid checks remain the only success criterion.
"""
import numpy as np
import igl
from scipy.special import logsumexp
from scipy.optimize import minimize
from contact_recovery import cooperating_reactions


class JointPlacementTarget:
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
                reactions,error=cooperating_reactions(search.floors[owner],search.rays[owner],target,costs)
                if reactions.sum()>1e-12:weight+=reactions/reactions.sum()
                self.records.append(dict(pose_index=owner,load_index=index,equilibrium_error=error))
            ids=np.flatnonzero(weight>1e-10)
            if len(ids):self.selected.append((owner,ids,weight[ids]))
        total=sum(weight.sum() for owner,ids,weight in self.selected)
        if total<=0:raise ValueError('no reaction guidance contacts')
        self.selected=[(owner,ids,weight/total) for owner,ids,weight in self.selected]
        self.key=tuple((owner,tuple(map(int,ids))) for owner,ids,weight in self.selected)

    def costs(self,owner,points,normals,directions,offsets):
        search=self.search;levels=[];scale=.001;width=.05
        for blocker in range(search.n):
            q=points+offsets[owner]-offsets[blocker]
            tree,planes=search.boundary.shadows(directions[blocker])
            if not planes:continue
            N=np.array([n for n,b in planes]);B=np.array([b for n,b in planes])
            maximum=np.full(len(points),-np.inf)
            # Bounded batches avoid point x face x plane temporary blowups.
            for first in range(0,len(points),128):
                p=q[first:first+128];n=normals[first:first+128]
                margins=B[None]-np.einsum('pk,mjk->pmj',p,N)
                # A 0.5 mm normal seating probe discourages tiny ghost contacts.
                margins+=np.maximum(0.,-.0005*np.einsum('pk,mjk->pmj',n,N))
                maximum[first:first+len(p)]=np.max(np.min(margins,axis=2),axis=1)
            if blocker!=owner and np.linalg.norm(offsets[owner]-offsets[blocker])>1e-10:
                distance,_,_,_=igl.signed_distance(q,np.asarray(search.mesh.vertices,dtype=np.float64),
                    np.asarray(search.mesh.faces,dtype=np.int64),igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)
                maximum=np.maximum(maximum,-distance)
            levels.append(maximum/scale)
            relative=offsets[owner]-offsets[blocker]
            overlap_weight=1./(1.+float(relative@relative)/(.001**2))
            levels.append(overlap_weight*(normals@directions[blocker])/.05)
        levels.append((normals@directions[owner])/.05)
        # Positive smooth maximum of all blocking poses includes both release
        # and newly created interference as the placement changes.
        values=np.column_stack(levels)/width
        return width*np.logaddexp(0.,logsumexp(values,axis=1))

    def value(self,directions,offsets):
        value=0.
        for owner,ids,weights in self.selected:
            value+=weights@self.costs(owner,self.search.points[ids],self.search.outward[ids],directions,offsets)
        return float(value)

    def propose(self,directions,offsets,kind,indices,targets,base):
        search=self.search;n=3 if kind.startswith('coherent') else 2*len(indices)
        zero=np.zeros(n);h=.25;value=self.value(directions,offsets);g=np.zeros(n);H=np.zeros(n)
        for j in range(n):
            x=np.zeros(n);x[j]=h;costs=[]
            for sign in [-1,1]:
                d,o=search.increment(directions,offsets,kind,indices,sign*x);costs.append(self.value(d,o))
            g[j]=(costs[1]-costs[0])/(2*h);H[j]=abs(costs[1]+costs[0]-2*value)/h**2
        if np.linalg.norm(g)<1e-10:return [],dict(kind=kind,indices=indices,gradient=g.tolist(),stalled=True)
        H=np.maximum(H,np.linalg.norm(g)*.1)
        opt=minimize(lambda x:(float(g@x+.5*np.sum(H*x*x)),g+H*x),zero,jac=True,method='SLSQP',
            constraints=[dict(type='ineq',fun=lambda x:1.-x@x,jac=lambda x:-2*x)],options={'maxiter':30,'ftol':1e-12})
        step=opt.x
        if not np.isfinite(step).all():return [],dict(error='nonfinite guidance step')
        if np.linalg.norm(step)>1:step/=np.linalg.norm(step)
        options=[]
        for fraction in getattr(search,'step_fractions',(1.,.5,.25)):
            d,o=search.increment(directions,offsets,kind,indices,fraction*step);after=self.value(d,o)
            if after>=value-max(1e-10,value*1e-5):continue
            proxy=search.boundary.evaluate(d,o,targets)
            score=(after,proxy['loss'],proxy['sum_loss'])
            options.append((score,kind,d,o,proxy,dict(step_fraction=fraction,reaction_guidance=True,indices=indices,
                step=(fraction*step).tolist(),target_key=str(self.key),guidance_before=value,guidance_after=after)))
        return options,dict(kind=kind,indices=indices,gradient=g.tolist(),guidance_before=value,contacts=sum(len(ids) for k,ids,w in self.selected))
