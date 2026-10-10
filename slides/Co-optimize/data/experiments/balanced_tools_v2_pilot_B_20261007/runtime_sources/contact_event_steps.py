"""Minimum local increments towards a useful contact boundary event.

Only used when ordinary polygon-force descent hits a flat contact-onset region.
No contact force caps; targets are ranked by the real wrench residual.
"""
import numpy as np
from scipy.optimize import minimize
import igl
from physics_guided_geometry import tangent_frames,retract


def minimum_step(A,b,size):
    if not len(A):return None
    A=np.asarray(A);b=np.asarray(b)
    fit=minimize(lambda x:(.5*x@x,x),np.zeros(size),jac=True,method='SLSQP',
        constraints=[dict(type='ineq',fun=lambda x:A@x-b,jac=lambda x:A)],
        options={'maxiter':100,'ftol':1e-12})
    if not fit.success or np.min(A@fit.x-b)<-1e-8:return None
    result=fit.x.copy()
    if np.linalg.norm(result)>1.:result/=np.linalg.norm(result)
    return result,dict(A=A.tolist(),b=b.tolist(),required_norm=float(np.linalg.norm(fit.x)),
        violation_before=float(np.sum(np.maximum(b,0.)**2)),
        violation_after=float(np.sum(np.maximum(b-A@result,0.)**2)))


def event_candidates(search,current,directions,offsets,limit=12):
    if current is None:return []
    targets=search.refresh_targets(current)
    worst=max(targets,key=lambda row:search.projection(current,row[0],row[1])['loss'])
    owner,load,target=worst;projection=search.projection(current,owner,load)
    rays=search.rays[owner];benefit=-(rays@projection['dual'])/np.linalg.norm(rays,axis=1)
    chosen=[];seen=set()
    for index in np.argsort(benefit)[::-1]:
        if benefit[index]<=1e-9:break
        if (owner,int(index)) in getattr(search,'event_avoid',set()):continue
        source=int(search.sources[index])
        if source in seen:continue
        seen.add(source);chosen.append(int(index))
        if len(chosen)>=limit:break
    result=[];frames=tangent_frames(directions)
    for index in chosen:
        point=search.points[index];normal=search.outward[index]
        # Direction event: minimum angular adjustment to admit this contact
        # normal in every coincident pose, always respecting native floor.
        coincident=np.linalg.norm(offsets-offsets[owner],axis=1)<1e-10
        ids=np.flatnonzero(coincident & (directions@normal>1e-9))
        if len(ids):
            A=[];b=[];size=2*len(ids);scale=np.deg2rad(1.)
            for j,k in enumerate(ids):
                a=np.zeros(size);a[2*j:2*j+2]=-normal@frames[k]*scale
                A.append(a);b.append(normal@directions[k]-1e-10)
                a=np.zeros(size);a[2*j:2*j+2]=search.normals[k]@frames[k]*scale
                A.append(a);b.append(-search.normals[k]@directions[k])
            solved=minimum_step(A,b,size)
            if solved is not None:
                x,stats=solved;z=np.zeros((search.n,2))
                for j,k in enumerate(ids):z[k]=scale*x[2*j:2*j+2]
                d=retract(directions,frames,z)
                if np.min(np.sum(d*search.normals,axis=1))>=-1e-12:
                    result.append(dict(kind='direction-contact-event',directions=d,offsets=offsets.copy(),
                        owner=owner,load=load,contact_index=index,benefit=float(benefit[index]),step=stats))
            continue
        # Translation event: leave every currently containing shadow prism.
        # Select its closest reachable outward plane (a local disjunction).
        size=2*(search.n-1);A=[];b=[]
        for blocker in range(search.n):
            query=point+offsets[owner]-offsets[blocker]
            tree,planes=search.boundary.shadows(directions[blocker])
            for prism in tree.intersection(np.r_[query-1e-10,query+1e-10]):
                normals,bounds=planes[prism];margins=bounds-normals@query
                if margins.min()<-1e-9:continue
                if np.any((np.abs(margins)<1e-9)&(normals@normal>0)):continue
                options=[]
                for axis,n in enumerate(normals):
                    a=np.zeros(size)
                    if owner:a[2*(owner-1):2*owner]+=n@search.frames[owner]*.001
                    if blocker:a[2*(blocker-1):2*blocker]-=n@search.frames[blocker]*.001
                    length=np.linalg.norm(a)
                    if length>1e-12:options.append(((margins[axis]+1e-5)/length,a,margins[axis]+1e-5))
                if options:
                    distance,a,value=min(options,key=lambda row:row[0]);A.append(a);b.append(value)
            if blocker!=owner and np.linalg.norm(offsets[owner]-offsets[blocker])>1e-10:
                distance,faces,closest,_=igl.signed_distance(query[None],np.asarray(search.mesh.vertices,dtype=np.float64),
                    np.asarray(search.mesh.faces,dtype=np.int64),igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)
                if distance[0]<-1e-9:
                    n=search.mesh.face_normals[faces[0]];a=np.zeros(size)
                    if owner:a[2*(owner-1):2*owner]+=n@search.frames[owner]*.001
                    if blocker:a[2*(blocker-1):2*blocker]-=n@search.frames[blocker]*.001
                    A.append(a);b.append(-distance[0]+1e-5)
        solved=minimum_step(A,b,size)
        if solved is not None:
            x,stats=solved;o=offsets.copy()
            for k in range(1,search.n):o[k]+=.001*search.frames[k]@x[2*(k-1):2*k]
            result.append(dict(kind='translation-contact-event',directions=directions.copy(),offsets=o,
                owner=owner,load=load,contact_index=index,benefit=float(benefit[index]),step=stats))
    result.sort(key=lambda row:(row['step']['required_norm']/max(row['benefit'],1e-12)))
    return result
