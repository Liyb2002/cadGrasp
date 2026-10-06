"""Use the REAL constructed cone's residual to value missing contact rays."""
import numpy as np
from physics_guided_cone import missing_values


def certificate_candidates(targets, failed, project, limit=6):
    """Scan all saved loads with projection-dual certificates, then project finalists.

    These are lower bounds on distance, not a global worst-load certificate.
    No load is resampled; the callback projects onto the actual constructed cone.
    """
    targets=np.asarray(targets)
    bad=np.flatnonzero(failed)
    if not len(bad):return [],dict(scanned=len(targets),projected=0)
    selected={int(bad[0]),int(bad[np.argmax(np.linalg.norm(targets[bad],axis=1))])}
    for axis in range(targets.shape[1]):
        selected.add(int(bad[np.argmax(targets[bad,axis])]))
        selected.add(int(bad[np.argmin(targets[bad,axis])]))
    projections={i:project(i) for i in selected}
    for _ in range(2):
        normals=[]
        for value in projections.values():
            r=value['dual'];length=np.linalg.norm(r)
            if length>1e-14:normals.append(-r/length)
        if not normals:break
        scores=targets[bad]@np.asarray(normals).T
        # Worst target in every discovered separating direction, plus the
        # largest combined certificate scores; do not sample the bad list.
        extra=set(map(int,bad[np.argmax(scores,axis=0)]))
        extra.update(map(int,bad[np.argsort(scores.max(axis=1))[-limit:]]))
        extra-=projections.keys()
        if not extra:break
        projections.update({i:project(i) for i in extra})
    ranked=sorted(projections,key=lambda i:projections[i]['loss'],reverse=True)
    return ranked[:limit],dict(scanned=len(targets),projected=len(projections),selection='real cone dual scan')


def deficit_contact_values(rays, projections, costs=None, temperature=.03):
    """Benefit of adding a ray to the current actual cone, with frozen geometry weights.

    v=max(0,-a_hat dot residual); v^2/2 is an achievable loss decrease while
    holding the old reaction allocation and optimizing the added ray force.
    Reoptimizing the old allocation can improve further. No force cap or
    acquisition-price LP participates in this missing-force signal.
    """
    if not projections:return np.zeros(len(rays[0])),dict(helpful_rays=0)
    losses=np.asarray([p['loss'] for k,p in projections])
    thermal=max(float(losses.max())*.02,1e-8)
    factors=np.exp((losses-losses.max())/thermal);factors/=factors.sum()
    answer=np.zeros(len(rays[0]));helpful=0;maximum=0.
    for factor,(k,projection) in zip(factors,projections):
        gain=.5*missing_values(rays[k],projection,normalize=True)**2
        peak=float(gain.max());maximum=max(maximum,peak)
        if peak<1e-24:continue
        valid=gain>=.05*peak
        helpful+=int(valid.sum())
        logits=np.full(len(gain),-np.inf)
        logits[valid]=np.log(gain[valid]/peak)
        if costs is not None:logits[valid]-=np.asarray(costs)[valid]/temperature
        weights=np.exp(logits-np.max(logits));weights/=weights.sum()
        answer+=factor*weights
    if answer.sum()>0:answer/=answer.sum()
    return answer,dict(helpful_rays=helpful,maximum_fixed_allocation_gain=maximum,
        source='actual force cone projection dual; no hypothetical contact reaction pricing')
