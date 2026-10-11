"""Necessary movement certificates and explicitly separated BBX measures."""
import numpy as np
import stage5_common as C


def object_z_certificates(data, failed_indices):
    certificates=[]
    for i in sorted(failed_indices):
        row=data['rows'][i];inverse=np.linalg.inv(row['fixture_transform'])
        lower=0.;upper=np.inf;lower_witness=None;upper_witness=None;constant_failure=None
        for j,other in enumerate(data['rows']):
            trans=other['fixture_transform']@inverse;a=trans[2,:2]
            b=row['cloud'][:,:2]@a+trans[2,3]
            k=row['horizontal_force_ratio']@a
            for sign in [-1,1]:
                indices=np.flatnonzero(k*sign>1e-12)
                if not len(indices):continue
                ratios=b[indices]/k[indices]
                index=int(indices[np.argmin(ratios) if sign==1 else np.argmax(ratios)])
                bound=float(b[index]/k[index])
                witness=dict(other_pose=other['pose'],demand_index=index,b_m=float(b[index]),k=float(k[index]),
                             demand_world_wrench=row['old_wrench_world_origin'][index].tolist())
                if sign==1 and bound<upper:upper=bound;upper_witness=witness
                if sign==-1 and bound>lower:lower=bound;lower_witness=witness
            indices=np.flatnonzero((np.abs(k)<=1e-12)&(b<-1e-9))
            if len(indices):constant_failure=dict(other_pose=other['pose'],demand_index=int(indices[0]),b_m=float(b[indices[0]]))
        impossible=constant_failure is not None or upper<lower-1e-9
        if impossible:
            certificates.append(dict(pose=row['pose'],object_z_only_impossible=True,
                required_minimum_lift_mm=lower*1000,allowed_maximum_lift_mm=upper*1000 if np.isfinite(upper) else None,
                lower_witness=lower_witness,upper_witness=upper_witness,constant_failure=constant_failure,
                inequality='b - k * object_lift >= 0',
                original_grounded_state_already_failed=True,fixture_floors_frozen=True))
    return certificates


def bounds_report(bounds):
    bounds=np.asarray(bounds)
    return dict(min_m=bounds[0].tolist(),max_m=bounds[1].tolist(),
                dimensions_mm=((bounds[1]-bounds[0])*1000).tolist(),volume_cm3=C.volume(bounds)*1e6)


def object_bounds(data, *, native=False):
    points=[]
    for row in data['rows']:
        transform=row['native_object_transform'] if native else row['object_transform']
        points.append(C.F.transform_points(data['body'].vertices,transform))
    points=np.vstack(points)
    return np.array([points.min(0),points.max(0)])


def required_enclosure_bound(data):
    # A COP inside the contact hull is necessarily inside its axis-aligned
    # box. Including every projected original wrench is a necessary bound;
    # it does not construct feet or prove any 3D base exists.
    points=np.vstack([r['cloud'] for r in data['rows']])
    body_box=object_bounds(data)
    bounds=np.array([np.minimum(body_box[0],points.min(0)),np.maximum(body_box[1],points.max(0))])
    return dict(bounds_report(bounds),kind='necessary_bbx_lower_bound_for_this_moved_layout',actual_base_constructed=False)
