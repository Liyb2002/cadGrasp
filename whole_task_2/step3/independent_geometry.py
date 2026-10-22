"""Fitted casting seats on a common flat plate and finite work exclusions."""
import numpy as np
import trimesh
import manifold3d as md
from whole_task_2.common.force_cone import tangent_frame

SCALE=.2


def solid(mesh):
    value=md.Manifold(md.Mesh64(np.ascontiguousarray(mesh.vertices/SCALE,dtype=np.float64),
        np.ascontiguousarray(mesh.faces,dtype=np.uint64)))
    if value.status()!=md.Error.NoError:raise RuntimeError('Unresolved solid input: '+str(value.status()))
    return value


def mesh(value):
    if value.status()!=md.Error.NoError:raise RuntimeError('Unresolved solid result: '+str(value.status()))
    data=value.to_mesh64()
    return trimesh.Trimesh(np.asarray(data.vert_properties[:,:3])*SCALE,np.asarray(data.tri_verts),process=False)


def volume(value):
    if value.status()!=md.Error.NoError:raise RuntimeError('Unresolved Boolean: '+str(value.status()))
    return abs(float(value.volume()))*SCALE**3


def union(values):
    value=md.Manifold.batch_boolean(values,md.OpType.Add)
    volume(value)
    return value


def box(low,high):
    low=np.asarray(low);high=np.asarray(high)
    b=trimesh.creation.box(high-low);b.apply_translation((low+high)/2)
    return solid(b)


def work_sector(workspace,policy):
    O=np.asarray(workspace['reference_head_front_world_m']);a=np.asarray(workspace['nominal_axis_world'])
    length=workspace['needle_length_m'];angle=np.deg2rad(workspace['half_angle_deg']);n=policy['cone_polygon_sides']
    u,v=tangent_frame(a);phi=2*np.pi*np.arange(n)/n
    ring=O+length*a+(length*np.tan(angle)/np.cos(np.pi/n))*(np.cos(phi)[:,None]*u+np.sin(phi)[:,None]*v)
    cone=md.Manifold.hull_points(np.vstack([O,ring])/SCALE)
    b=trimesh.creation.icosphere(subdivisions=policy['ball_subdivisions'],radius=1.)
    # Each tangent plane of the scaled polyhedral sphere lies outside the
    # radius-r ball. Cone sides similarly circumscribe the circular cone.
    apothem=np.einsum('ij,ij->i',b.face_normals,b.triangles_center).min()
    b.vertices*=length/apothem;b.apply_translation(O)
    sector=cone^solid(b)
    # Reserve the finite needle radius and a conservative sphere covering the
    # configured head proxy through the whole direction range. These cut the
    # fixture only; Step1 task masks are never enlarged or filtered.
    radius=workspace['needle_outer_radius_m']
    offsets=np.array([[x,y,z]for x in (-radius,radius)for y in (-radius,radius)for z in (-radius,radius)])
    expanded=md.Manifold.hull_points((mesh(sector).vertices[:,None,:]+offsets[None]).reshape(-1,3)/SCALE)
    offset=workspace['head_front_to_flange_m'];head_radius=workspace['head_proxy_radius_m']+2*offset*np.sin(angle/2)
    head=trimesh.creation.icosphere(subdivisions=3,radius=1.)
    head_apothem=np.einsum('ij,ij->i',head.face_normals,head.triangles_center).min()
    head.vertices*=head_radius/head_apothem;head.apply_translation(O-offset*a)
    result=union([expanded,solid(head)])
    return result,dict(half_angle_deg=workspace['half_angle_deg'],radius_m=length,
        sphere_radial_overestimate_m=float(length/apothem-length),cone_sides=n,
        circular_cone_circumscribed=True,sphere_circumscribed=True,
        needle_radius_reserved_m=radius,head_orbit_sphere_radius_m=float(head_radius),
        head_and_needle_sizes_are_project_proxies=True,full_arm_certified=False)


def fitted_outer(body,thickness):
    """Existing Co-optimize face cells, with bounded outward displacement."""
    from .legacy import kernel as K
    offsets=K.wrap_offsets(body,thickness)
    cells=[K.S.solid(K.G.hull_mesh(K.G.head_cell(body,triangle,face,offsets)))
           for face,triangle in enumerate(body.triangles)]
    return union([solid(body),*cells])


def casting_stock(outer,plate_top):
    """Back a fitted outer skin down to the plate, following its XY outline.

    These vertical triangular cells fill only below the fitted shape. They
    connect overhanging mold parts without any elevated rectangular stock.
    The original body and its own working sector are subtracted afterwards.
    """
    cells=[solid(outer)]
    for triangle in outer.triangles[outer.face_normals[:,2]<-1e-10]:
        if triangle[:,2].max()<=plate_top:continue
        bottom=triangle.copy();bottom[:,2]=plate_top
        cell=md.Manifold.hull_points(np.vstack([triangle,bottom])/SCALE)
        if float(cell.volume())>0:cells.append(cell)
    return union(cells)


def pressure_points(targets,com):
    forces=targets[:,:3];tau=targets[:,3:]+np.cross(com,forces)
    if forces[:,2].min()<=0:raise ValueError('Expected strictly positive vertical reaction')
    return np.c_[-tau[:,1]/forces[:,2],tau[:,0]/forces[:,2]]


def grounded_component(value):
    pieces=value.decompose();kept=[];floating=[]
    for part in pieces:
        # Negative components are enclosed cavities, not disposable material.
        # Keep the original solid and subtract only positive floating bodies.
        if float(part.volume())*SCALE**3<=1e-15:continue
        b=mesh(part).bounds
        if b[0,2]<=1e-9:kept.append(part)
        else:floating.append(part)
    if len(kept)!=1:raise RuntimeError('Expected one connected component attached to the base')
    for part in floating:value=value-part
    return value


def condition(value,forbidden,policy,subset=None):
    tolerance=policy['boolean_volume_tolerance_m3']
    for turn in range(policy.get('maximum_conditioning_rounds',5)+1):
        actual=mesh(value);current=solid(actual)
        residual=[volume(current^f)for f in forbidden]
        extra=0. if subset is None else volume(current-subset)
        if max([extra,*residual])<=tolerance:
            return actual,dict(rounds=turn,forbidden_overlap_m3=residual,added_outside_parent_m3=extra)
        if turn==policy.get('maximum_conditioning_rounds',5):break
        # Recut the same forbidden sets in serialized Mesh64 coordinates.
        # No clearance, contact point or physical tolerance is relaxed.
        if subset is not None:current=current^subset
        for f in forbidden:current=current-f
        value=grounded_component(current)
    raise RuntimeError('Serialized fixture/forbidden intersections remain unresolved')


def exit_length(body,support,direction,margin):
    a=body.vertices@direction;b=support.vertices@direction
    return max(.01,float(b.max()-a.min()+margin))


def whole_object_sweep(body,direction,length):
    # A simple initialization deliberately uses a conservative convex channel.
    # Every original triangle and every t in [0,length] is contained in this
    # hull of both endpoint vertex sets. No sampled-placement approximation.
    displacement=length*np.asarray(direction)
    value=md.Manifold.hull_points(np.vstack([body.vertices,body.vertices+displacement])/SCALE)
    return value,dict(method='convex envelope of whole-object continuous translation',
        displacement=displacement.tolist(),whole_object=True,continuous_parameter_range=[0.,1.],
        conservative_envelope=True,exact_nonconvex_sweep_claimed=False,
        vertex_endpoint_containment_implies_all_triangle_positions=True)


def exit_directions(axis):
    back=-np.asarray(axis);u,v=tangent_frame(back);rows=[back,np.array([0.,0.,1.])]
    for theta in (15.,30.,45.):
        for phi in np.arange(0,2*np.pi,np.pi/2):
            d=np.cos(np.deg2rad(theta))*back+np.sin(np.deg2rad(theta))*(np.cos(phi)*u+np.sin(phi)*v)
            if d[2]>0 and all(np.linalg.norm(d-old)>1e-9 for old in rows):rows.append(d)
    return rows
