"""Reproducible physical work-area forces, separate smoke and IID suites."""
import numpy as np
import trimesh
from scene import SHAPE, read


def evaluate(case, shape=SHAPE):
    domain = read(shape/'load_domain.json')
    g = domain['geometry']
    i = int(case['work_face_index'])
    if i != case['work_face_index'] or not 0 <= i < len(g['work_face_ids']):
        raise ValueError('Invalid working face')
    u, v, theta, phi, magnitude = np.asarray(case['parameters'], float)
    if not np.isfinite([u,v,theta,phi,magnitude]).all():
        raise ValueError('Nonfinite load')
    if not (u >= 0 and v >= 0 and u+v <= 1+1e-12 and 0 <= theta <= np.deg2rad(domain['load']['cone_half_deg'])+1e-12 and 0 <= magnitude <= domain['load']['K']):
        raise ValueError('Outside baseline physical force domain')
    vertices = np.asarray(g['vertices_m'])
    face = g['faces'][g['work_face_ids'][i]]
    a,b,c = vertices[face]
    q = a+u*(b-a)+v*(c-a)
    n,t1,t2 = (np.asarray(g[key])[i] for key in ('inward_normals','tangent1','tangent2'))
    direction = np.cos(theta)*n + np.sin(theta)*(np.cos(phi)*t1+np.sin(phi)*t2)
    mesh = trimesh.Trimesh(vertices,g['faces'],process=False)
    reachable = not mesh.ray.intersects_any([q-domain['reachability']['ray_offset_m']*n],[-direction])[0]
    if not reachable:
        raise ValueError('Tool ray is occluded')
    return dict(case, point_body_m=q.tolist(), force_body_mg=(magnitude*direction).tolist(),
                mesh_face_id=int(g['work_face_ids'][i]), reachable=True)


def smoke(shape=SHAPE):
    data = read(shape/'load_samples.json')
    points = np.asarray(data['pt_m'])
    forces = np.asarray(data['force_push_mg'])
    forces = .5*forces/np.linalg.norm(forces,axis=1)[:,None]
    com = np.asarray(read(shape/'manifest.json')['com_world_m'])
    moments = np.cross(points-com, forces)
    chosen=[]
    for axis in range(3):
        for sign in (-1,1):
            k=int(np.argmax(sign*moments[:,axis]))
            if k not in [row[0] for row in chosen]:
                chosen.append((k, f'moment_{"xyz"[axis]}_{"plus" if sign>0 else "minus"}',.5))
    # Additional spatially separated sites, including lower magnitudes.
    distance=np.min(np.linalg.norm(points[:,None,:]-points[[i for i,_,_ in chosen]][None,:,:],axis=2),axis=1)
    while len(chosen)<10:
        k=int(np.argmax(distance)); chosen.append((k,'spatial_coverage',[.125,.25,.5,.5][len(chosen)-6]))
        distance=np.minimum(distance,np.linalg.norm(points-points[k],axis=1))
    result=[]
    for j,(k,purpose,magnitude) in enumerate(chosen):
        params=list(data['parameters'][k]); params[-1]=magnitude
        result.append(evaluate(dict(id=f'{j+1:02d}_{purpose}',suite='smoke_adversarial',
            source_sample_index=k,work_face_index=data['work_face_index'][k],parameters=params),shape))
    return result


def random_cases(count, seed, shape=SHAPE):
    """Fresh IID draws, area × solid angle × uniform magnitude, reject blocked rays."""
    rng=np.random.default_rng(seed)
    d=read(shape/'load_domain.json'); g=d['geometry']
    areas=np.asarray(g['work_face_areas_m2']); p=areas/areas.sum()
    result=[]; attempts=0
    while len(result)<count:
        attempts+=1
        face=int(rng.choice(len(p),p=p)); u,v=rng.random(2)
        if u+v>1: u,v=1-u,1-v
        theta=float(np.arccos(rng.uniform(np.cos(np.deg2rad(d['load']['cone_half_deg'])),1)))
        phi=float(rng.uniform(0,2*np.pi)); magnitude=float(rng.uniform(0,d['load']['K']))
        case=dict(id=f'iid_{len(result):06d}',suite='iid',seed=seed,proposal_index=attempts,
                  work_face_index=face,parameters=[u,v,theta,phi,magnitude])
        try: result.append(evaluate(case,shape))
        except ValueError as exc:
            if str(exc) != 'Tool ray is occluded': raise
    return result
