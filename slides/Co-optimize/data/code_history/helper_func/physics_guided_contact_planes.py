"""Physics-selected contact tangent proposals; trajectory costs rank, never accept."""
from co_common import np,U,transform_points
from physics_guided_contact_sweep import NominalContactSweep
from physics_guided_geometry import acquisition,distance_cost
from itertools import combinations


def contact_plane_directions(normals,floors,phases=8):
    normals=np.asarray(normals,float);normals=normals/np.linalg.norm(normals,axis=1)[:,None]
    _,singular,vt=np.linalg.svd(normals,full_matrices=True)
    rank=int(np.sum(singular>1e-10));basis=vt[rank:]
    if len(basis)==1:
        for axis in [basis[0],-basis[0]]:
            directions=np.tile(axis,(len(floors),1))
            directions[np.sum(directions*floors,axis=1)<-1e-12]*=-1
            yield directions
    elif len(basis)==2:
        for phase in np.linspace(0,2*np.pi,phases,endpoint=False):
            axis=np.cos(phase)*basis[0]+np.sin(phase)*basis[1];directions=[]
            for floor in floors:
                coordinates=basis@floor;projected=basis.T@coordinates
                dot=float(axis@projected);norm2=float(coordinates@coordinates)
                d=axis.copy()
                if dot<0 and norm2>1e-28:
                    d-=dot/norm2*projected
                    if np.linalg.norm(d)<1e-10:
                        edge=np.cross(normals[0],projected);edge/=np.linalg.norm(edge)
                        d=edge if axis@edge>=0 else -edge
                directions.append(d/np.linalg.norm(d))
            d=np.asarray(directions)
            if np.min(np.sum(d*floors,axis=1))>=-1e-12:yield d


def wanted_normals(search,weights,limit=8):
    values=np.zeros(len(search.mesh.faces));np.maximum.at(values,search.point_sources,weights)
    normals=[]
    for face in np.argsort(values)[::-1]:
        if values[face]<=0:break
        n=search.mesh.face_normals[face]
        if any(n@old>np.cos(np.deg2rad(8)) for old in normals):continue
        normals.append(n)
        if len(normals)>=limit:break
    return normals


def allocated_contacts(search,result):
    """Recover existing head vertices from sparse real-cone force allocations."""
    points=result['triangles'].reshape(-1,3)
    if not len(points):return [],np.empty((0,3)),np.empty((0,3)),np.empty(0)
    sources=np.repeat(result['sources'],3);normals=search.mesh.face_normals[sources]
    selected={}
    for k,(task,T) in enumerate(search.states):
        indices=sorted(search.proxy_indices[k])[:2] or [0]
        world=transform_points(points,T);force=-task.domain.mesh.face_normals[sources]
        raw=np.c_[force,np.cross(world-task.domain.com,force)]
        heads=U.heads(raw,task.scale);unit=heads/np.linalg.norm(heads,axis=1)[:,None]
        for index in indices:
            projection=search.projection(result,k,index)
            for j in np.flatnonzero(projection['coefficients']>1e-10):
                ray=result['supplies'][k][j];f=ray[:3]/task.scale[:3]
                if np.linalg.norm(f)<1e-12 or abs(ray[6]-f[2])>1e-10:continue
                key=ray/np.linalg.norm(ray);match=int(np.argmax(unit@key))
                if np.max(abs(unit[match]-key))>1e-8:continue
                value=float(projection['coefficients'][j]*np.linalg.norm(f))
                selected[match]=selected.get(match,0.)+value
    ids=sorted(selected,key=lambda j:-selected[j])[:64]
    old=[]
    for j in ids:
        n=normals[j]
        if any(n@other>np.cos(np.deg2rad(8)) for other in old):continue
        old.append(n)
        if len(old)>=8:break
    return old,points[ids],normals[ids],np.array([selected[j] for j in ids])


def ranked_contact_planes(search,result,weights,limit=32):
    wanted=wanted_normals(search,weights)
    old,old_points,old_normals,old_weights=allocated_contacts(search,result)
    positive=np.flatnonzero(weights>0)
    if not len(positive) or not wanted:return [],dict(candidate_count=0)
    # Geometry probes are original potential vertices plus existing allocated
    # contact vertices. They do not replace or resample the original load set.
    cumulative=np.cumsum(weights[positive]);quantiles=(np.arange(64)+.5)/64*cumulative[-1]
    quantile_ids=positive[np.searchsorted(cumulative,quantiles)]
    ids=np.unique(np.r_[np.argsort(weights)[-64:],quantile_ids])
    ids=ids[weights[ids]>0]
    points=search.points[ids];normals=search.ray_normals[ids]
    values=weights[ids]/weights[ids].sum()
    if len(old_weights) and old_weights.sum()>0:
        points=np.vstack([points,old_points]);normals=np.vstack([normals,old_normals])
        values=np.r_[.75*values,.25*old_weights/old_weights.sum()]
    model=NominalContactSweep(search.clearance,points,normals,search.length,float(search.mesh.extents.max()))
    sets=[[n] for n in wanted]+[[a,b] for a,b in combinations(wanted,2)]+[[a,b] for a in wanted for b in old]
    unique={}
    for normal_set in sets:
        for d in contact_plane_directions(normal_set,search.normals):
            key=tuple(np.round(d.ravel(),10))
            if key in unique or key in search.seen:continue
            field=np.column_stack([model.distances(direction) for direction in d])
            zeros=np.zeros(field.shape+(2,))
            cost=acquisition(d,normals)[0]+distance_cost(field,zeros,np.zeros((len(d),2)))[0]
            unique[key]=(float(values@cost),d)
    ranked=sorted(unique.values(),key=lambda row:row[0])[:limit]
    search.distance_model.evaluations+=model.evaluations
    return ranked,dict(candidate_count=len(unique),ranked_count=len(ranked),
        wanted_normals=[n.tolist() for n in wanted],allocated_normals=[n.tolist() for n in old],
        geometry_probe_count=len(points),trajectory_scores=[v for v,d in ranked])
