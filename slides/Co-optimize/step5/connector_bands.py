"""Replace pre-existing bottlenecks by short round bands inside legal stock.

Every new rod has a measured physical radius, stays in the original legal
domain and BBX, and must improve the resolved connector test. Grid routes
only propose paths; the actual round solids undergo the original validation.
"""
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
import trimesh
from continuous_legs import capsule,bounded_domain,RouteGuide
from deterministic_base import solid,unpack,union,volume
from periphery import structural_profile
from rounded_front import connector_limits


def lobe_centres(labels,distances,key,origin,pitch,radius):
    ids=np.argwhere((labels==key)&(distances>=radius+.0007))
    if not len(ids):
        ids=np.argwhere(labels==key)
        order=np.argsort(-distances[tuple(ids.T)],kind='stable')
        ids=ids[order[:min(1000,len(ids))]]
    ids=ids[::max(1,len(ids)//2500)]
    return ids*pitch+origin


def reinforce(mesh,witness,body,rows,bounds,calibration,certify,checkpoint=None):
    pitch=calibration['pitch_m'];radius=calibration['guard_radius_m']
    _,terminal_limit=connector_limits(calibration)
    target=calibration['maximum_separated_core_fraction'];history=[]
    current=mesh;domain=None
    for iteration in range(24):
        profile,voxels,distances=structural_profile(current,pitch,(radius,))
        fraction=profile['profiles'][0]['separated_core_fraction']
        sizes=profile['profiles'][0]['component_volumes_cm3']
        worst=sizes[1]if len(sizes)>1 else 0.
        print('CONNECTOR_PROFILE',iteration,'separated_core_cm3',round(worst,6),'reference_terminal_cm3',round(terminal_limit,6),flush=True)
        if worst<=terminal_limit+1e-10:return current,history,profile
        labels,n=ndimage.label(distances>radius)
        sizes=np.bincount(labels.ravel());sizes[0]=0
        keys=np.argsort(-sizes,kind='stable');keys=keys[sizes[keys]>0]
        main,guest=int(keys[0]),int(keys[1])
        if domain is None:domain=bounded_domain(witness,body,rows,bounds)
        origin=voxels.transform[:3,3]
        success=False
        for rod_radius in [radius+.001,radius+.0005,radius+.0002]:
            a=lobe_centres(labels,distances,main,origin,pitch,rod_radius)
            b=lobe_centres(labels,distances,guest,origin,pitch,rod_radius)
            nearest,ids=cKDTree(a).query(b)
            selected=np.argsort(nearest,kind='stable')[:16]
            proposals=[]
            for index in selected:
                rod=capsule(a[ids[index]],b[index],rod_radius)
                if volume(rod-domain)>1e-10:continue
                proposals.append((rod,dict(method='short_round_band',
                    radius_m=rod_radius,a_m=a[ids[index]].tolist(),b_m=b[index].tolist())))
                break
            if not proposals:
                # The whole swept round tube must fit, including bends. The
                # guide's extra clearance is later checked with exact solids.
                seed=labels==main
                core=trimesh.voxel.ops.matrix_to_marching_cubes(seed,pitch=pitch)
                core.apply_translation(origin)
                try:
                    guide=RouteGuide(domain,solid(core),pitch=.001)
                    guide.valid &= guide.clearance>=rod_radius+1.75*guide.pitch
                    guide.roots &= guide.valid
                    if not guide.roots.any():raise ValueError('No full-width root inside legal space')
                    point=b[selected[0]]
                    path,route=guide.route(point)
                    rod=union([capsule(first,last,rod_radius)for first,last in zip(path,path[1:])])
                    if volume(rod-domain)>1e-10:raise ValueError('Swept round band leaves legal stock')
                    proposals.append((rod,dict(method='routed_round_band',radius_m=rod_radius,path_m=path.tolist(),route=route)))
                except (ValueError,RuntimeError)as error:
                    history.append(dict(operation='connector_band',accepted=False,radius_m=rod_radius,error=str(error)))
            for rod,details in proposals:
                candidate=unpack(solid(current)+rod)
                try:
                    candidate,proof=certify(candidate)
                    after,_,_=structural_profile(candidate,pitch,(radius,))
                    next_fraction=after['profiles'][0]['separated_core_fraction']
                    sizes=after['profiles'][0]['component_volumes_cm3']
                    next_worst=sizes[1]if len(sizes)>1 else 0.
                    if next_worst>=worst-1e-5:raise ValueError('Band did not improve the connector bottleneck')
                    history.append(dict(operation='connector_band',accepted=True,**details,
                        added_material_cm3=(abs(float(candidate.volume))-abs(float(current.volume)))*1e6,
                        separated_core_fraction_before=fraction,separated_core_fraction_after=next_fraction))
                    current=candidate;success=True
                    print('CONNECTOR_BAND',details['method'],'radius',rod_radius,
                        'added_cm3',round(history[-1]['added_material_cm3'],4),'fraction',round(next_fraction,6),flush=True)
                    if checkpoint:checkpoint(current,proof,history)
                    break
                except (ValueError,RuntimeError)as error:
                    history.append(dict(operation='connector_band',accepted=False,**details,error=str(error)))
            if success:break
        if not success:
            history.append(dict(operation='connector_band',accepted=False,
                error='No certified full-width band for the remaining bottleneck',iteration=iteration))
            return current,history,profile
    profile,_,_=structural_profile(current,pitch,(radius,))
    return current,history,profile
