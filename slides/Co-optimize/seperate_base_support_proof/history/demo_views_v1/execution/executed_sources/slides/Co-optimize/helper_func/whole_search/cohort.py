"""Atomic coordinated Juxtapose candidates, using horizontal missing reactions."""
from .common import *


def candidates(model,layout,hosts=None):
    hosts=list(layout.active[:2]) if hosts is None else hosts
    rows=[]
    # Outward from the needed reaction: translating in this direction can
    # expose the wall whose inward contact normal supplies that reaction.
    votes=[]
    for k in range(len(model.poses)):
        desired=model.tasks[k].targets[:,:3].mean(axis=0)
        outward=-desired;outward[2]=0.
        if np.linalg.norm(outward)<1e-10:
            i=int(np.argmax(np.linalg.norm(model.tasks[k].targets[:,:2],axis=1)))
            outward=-model.tasks[k].targets[i,:3].copy();outward[2]=0.
        if np.linalg.norm(outward)<1e-10:
            outward=np.array([1.,0.,0.])
        votes.append(outward/np.linalg.norm(outward))
    for host in hosts:
        host_native=int(layout.hosts[host]);frame=model.native[host_native]
        for fraction in [0.,1/8,1/4,1/3,1/2,2/3]:
            for phase in [0.,np.pi/2]:
                for tilt in [0.,.2]:
                    result=layout.copy()
                    for k in layout.active:
                        result.hosts[k]=host_native
                        result.placements[k]=np.linalg.inv(frame) @ model.native[k]
                        angle=2*np.pi*k/max(len(layout.active),1)+phase
                        jitter=.06*model.extent*fraction*np.array([np.cos(angle),np.sin(angle),0.])
                        delta=np.zeros(3) if k==host else model.extent*fraction*votes[k]+jitter
                        result.placements[k,:3,3]+=frame[:3,:3].T @ delta
                        world_exit=np.array([0.,0.,1.])-tilt*votes[k]
                        result.directions[k]=legal_direction(frame[:3,:3].T @ world_exit,model.floor_normal(result,k))
                    rows.append(('juxtapose-cohort',result,dict(host=model.poses[host],host_index=host,
                                 radius_fraction=fraction,phase_radians=phase,exit_inward_tilt=tilt,
                                 operation='multiple juxtaposes with force-guided overlapping final positions',
                                 direction_choice='common-world-up' if tilt==0 else 'near-common-world-up')))
    return rows
