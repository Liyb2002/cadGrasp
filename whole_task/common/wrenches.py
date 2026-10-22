"""Current-pose paired COM reactions for declared world or local force cones.

P=m*d; b=(-g-P, -(q-COM) cross P). All selected faces remain in the family,
including provisional fills; geometric admissibility is recorded separately.
"""
import numpy as np
import trimesh

from .transforms import points, directions, rigid_transform
from .force_cone import tangent_frame, outer_cap


def demand(point, process_force, com, gravity):
    point, process_force = np.broadcast_arrays(np.asarray(point, float), np.asarray(process_force, float))
    return np.concatenate((-(process_force+gravity), -np.cross(point-com, process_force)), axis=-1)


def cross_matrix(vector):
    x, y, z = np.moveaxis(np.asarray(vector), -1, 0)
    result = np.zeros(np.asarray(vector).shape[:-1]+(3, 3))
    result[..., 0, 1], result[..., 0, 2] = -z, y
    result[..., 1, 0], result[..., 1, 2] = z, -x
    result[..., 2, 0], result[..., 2, 1] = -y, x
    return result


def tool_visibility(mesh, origins, outward_axis):
    """Confirm fast positive ray hits in float64, avoiding grazing self hits."""
    rays = np.tile(outward_axis, (len(origins), 1))
    blocked = mesh.ray.intersects_any(origins, rays)
    if blocked.any():
        from trimesh.ray.ray_triangle import RayMeshIntersector
        suspect = np.flatnonzero(blocked)
        blocked[suspect] = RayMeshIntersector(mesh).intersects_any(origins[suspect], rays[suspect])
    return ~blocked


def compile_forward(triangles, com, gravity, axis):
    """Exact b=B_f*(1,P,uP,vP), plus its fixed-axis restriction A_f."""
    arms = triangles[:, 0]-com
    edge_u, edge_v = triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]
    matrix = np.zeros((len(triangles), 6, 10))
    matrix[:, :3, 0] = -gravity
    matrix[:, :3, 1:4] = -np.eye(3)
    for start, vectors in ((1, arms), (4, edge_u), (7, edge_v)):
        matrix[:, 3:, start:start+3] = -cross_matrix(vectors)
    restriction = np.zeros((10, 4))
    restriction[0, 0] = 1
    for start, column in ((1, 1), (4, 2), (7, 3)):
        restriction[start:start+3, column] = axis
    return matrix, matrix @ restriction


class WrenchDomain:
    def __init__(self, data):
        self.data = data
        geometry = data["geometry"]
        self.mesh = trimesh.Trimesh(geometry["vertices_m"], geometry["faces"], process=False)
        self.work_ids = np.asarray(geometry["work_face_ids"], dtype=int)
        self.triangles = self.mesh.triangles[self.work_ids]
        self.inward = np.asarray(geometry["inward_normals"])
        self.com = np.asarray(data["frame"]["moment_origin_m"])
        self.gravity = np.asarray(data["load"]["gravity_force_mg"])
        self.axis = np.asarray(data["load"]["fixed_direction_world"])
        # axis is the nominal needle display/workspace direction, not the
        # sampled processing-force direction in the local-cone model.
        self.direction_model=data['load'].get('direction_model','fixed_world_needle_axis')
        self.half_angle=np.radians(data['load'].get('cone_half_deg',0.))
        self.force_axes=(self.inward if self.direction_model=='local_inward_normal_cone'
                         else np.broadcast_to(self.axis,self.inward.shape))
        self.tangent1,self.tangent2=tangent_frame(self.force_axes)
        self.k = float(data["load"]["K"])
        self.weight_N = float(data["physical_properties"]["weight_N"])
        self.face_screened = np.asarray(geometry["face_passes_pose_geometry"], dtype=bool)
        self.B, self.A = compile_forward(self.triangles, self.com, self.gravity, self.axis)

    def evaluate(self, face_index, u, v, magnitude_mg, check_visibility=True, *, theta=0., phi=0.):
        face_index, u, v, magnitude,theta,phi = np.broadcast_arrays(face_index, u, v, magnitude_mg,theta,phi)
        if not all(np.isfinite(value).all() for value in (face_index, u, v, magnitude,theta,phi)):
            raise ValueError("Load parameters must be finite")
        if np.any((face_index != face_index.astype(int)) | (face_index < 0) | (face_index >= len(self.work_ids))):
            raise ValueError("Invalid working-face index")
        if np.any((u < 0) | (v < 0) | (u+v > 1+1e-14) | (magnitude < 0) | (magnitude > self.k)):
            raise ValueError("Load outside its surface triangle or magnitude interval")
        ids = face_index.astype(int)
        if np.any((theta<0)|(theta>self.half_angle+1e-14)):
            raise ValueError('Force direction outside declared cone')
        triangles = self.triangles[ids]
        q = triangles[..., 0, :]+u[..., None]*(triangles[..., 1, :]-triangles[..., 0, :])
        q += v[..., None]*(triangles[..., 2, :]-triangles[..., 0, :])
        if self.direction_model in ('local_inward_normal_cone','world_needle_direction_cone'):
            direction=np.cos(theta)[...,None]*self.force_axes[ids]+np.sin(theta)[...,None]*(
                np.cos(phi)[...,None]*self.tangent1[ids]+np.sin(phi)[...,None]*self.tangent2[ids])
        else:
            if np.any(theta!=0) or np.any(phi!=0):raise ValueError('Fixed-axis domain has no angular parameters')
            direction=np.broadcast_to(self.axis,q.shape)
        force = magnitude[..., None]*direction
        need = demand(q, force, self.com, self.gravity)
        visible = None
        if check_visibility:
            epsilon = self.data["reachability"]["ray_offset_m"]
            origins = (q-epsilon*self.axis).reshape(-1, 3)
            visible_flat = np.empty(len(origins), dtype=bool)
            batch = self.data["reachability"]["ray_batch_size"]
            for start in range(0, len(origins), batch):
                rays = origins[start:start+batch]
                visible_flat[start:start+len(rays)] = tool_visibility(self.mesh, rays, -self.axis)
            visible = visible_flat.reshape(ids.shape)
        return dict(pt_m=q, force_push_mg=force, force_direction_world=direction, need_wrench=need,
                    source_face_id=self.work_ids[ids], tool_visible=visible,
                    face_passes_pose_geometry=self.face_screened[ids],
                    geometrically_screened=(visible & self.face_screened[ids]) if visible is not None else None)

    def forward(self, face_index, parameters):
        """Evaluate all five saved parameters using the general B_f mapping."""
        values = np.asarray(parameters, dtype=float)
        if values.shape[-1] != 5:raise ValueError('Expected u,v,theta,phi,m')
        u,v,theta,phi,m = np.moveaxis(values, -1, 0)
        force=self.evaluate(face_index,u,v,m,check_visibility=False,theta=theta,phi=phi)['force_push_mg']
        features=np.concatenate((np.ones(force.shape[:-1]+(1,)),force,u[...,None]*force,v[...,None]*force),axis=-1)
        return np.einsum("...ij,...j->...i", self.B[np.asarray(face_index,dtype=int)],features)

    def jacobian(self, face_index, u, v, magnitude_mg, *, theta=0., phi=0.):
        """Exact derivative db/d(u,v,m) within one selected face."""
        direction=self.evaluate(face_index,u,v,magnitude_mg,check_visibility=False,theta=theta,phi=phi)['force_direction_world']
        ids, u, v, m = np.broadcast_arrays(face_index, u, v, magnitude_mg)
        B=self.B[ids.astype(int)]
        matrix=np.stack((B[...,0],np.einsum('...ij,...j->...i',B[...,1:4],direction),
            np.einsum('...ij,...j->...i',B[...,4:7],direction),np.einsum('...ij,...j->...i',B[...,7:10],direction)),axis=-1)
        return np.stack((m[..., None]*matrix[..., 2], m[..., None]*matrix[..., 3],
                         matrix[..., 1]+u[..., None]*matrix[..., 2]+v[..., None]*matrix[..., 3]), axis=-1)


def resolve_policy(policy,robot_config):
    policy=dict(policy)
    if policy['force_direction_model'] in ('local_inward_normal_cone','world_needle_direction_cone'):
        policy['direction_half_angle_deg']=float(robot_config['task_constraint']['cone_half_angle_deg'])
        if not 0<policy['direction_half_angle_deg']<90:raise ValueError('Expected inward force cone half angle between 0 and 90')
    return policy


def build_domain(mesh, pose, station, physical, policy):
    if policy['force_direction_model'] not in ('fixed_world_needle_axis','local_inward_normal_cone','world_needle_direction_cone'):
        raise ValueError('Unsupported force direction model')
    transform = np.asarray(pose["world_from_object"])
    np.testing.assert_allclose(transform, rigid_transform(transform[:3, :3], transform[:3, 3]), atol=1e-12)
    local_com = np.asarray(physical["com_mesh_frame"], dtype=float)
    if local_com.shape != (3,) or not np.isfinite(local_com).all():
        raise ValueError("A finite object COM is required")
    np.testing.assert_allclose(local_com, mesh.center_mass, atol=1e-9, rtol=0)
    mass = float(physical["mass_kg"])
    acceleration = np.asarray(station["world_gravity_m_s2"], dtype=float)
    gravity_norm = np.linalg.norm(acceleration)
    if not np.isfinite(mass) or mass <= 0 or not np.isfinite(gravity_norm) or gravity_norm <= 0:
        raise ValueError("Positive object mass and gravity are required")
    axis = np.asarray(station["tool_to_surface_axis_world"], dtype=float)
    np.testing.assert_allclose(np.linalg.norm(axis), 1, atol=1e-12)
    maximum = float(policy["maximum_processing_force_mg"])
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError("Processing-force maximum must be positive")
    ids = np.asarray(pose["working_face_ids"], dtype=int)
    if not len(ids) or len(np.unique(ids)) != len(ids) or ids.min() < 0 or ids.max() >= len(mesh.faces):
        raise ValueError("Unique source working-face IDs required")
    vertices = points(transform, mesh.vertices)
    com = points(transform, local_com)
    inward = -directions(transform, mesh.face_normals[ids])
    passed = pose["geometry_audit"]["geometrically_passed_face_ids"]
    data = dict(schema="cadgrasp_current_pose_direction_domain_v2", pose_id=pose["id"],
        frame=dict(name="world", T_world_mesh=transform.tolist(), moment_origin_m=com.tolist()),
        physical_properties=dict(mass_kg=mass, com_object_m=local_com.tolist(),
            gravity_acceleration_m_s2=acceleration.tolist(), weight_N=mass*gravity_norm),
        load=dict(K=maximum, magnitude_range_mg=[0., maximum], gravity_force_mg=(acceleration/gravity_norm).tolist(),
            direction_model=policy['force_direction_model'], fixed_direction_world=axis.tolist(),
            cone_half_deg=policy['direction_half_angle_deg'],
            direction_half_angle_source=policy.get('direction_half_angle_source'),
            continuous_cover=policy.get('continuous_cover'),
            free_processing_couple_included=False),
        pair_order=policy["reaction_target_order"],
        units=dict(position="m", force="mg", moment="mg*m", SI_conversion="multiply both wrench blocks by weight_N"),
        mapping=dict(process_force="P=m*d; d lies in the declared cone about its axis",
            cone_axis="current face inward normal" if policy['force_direction_model']=='local_inward_normal_cone' else "robot nominal direction in world coordinates",
            external_wrench="[P+g, cross(q-COM,P)]",
            need_wrench="[-g-P, -cross(q-COM,P)]", moment_origin="current workpiece COM",
            pairing="same application point q and complete P produce both blocks"),
        parameter_domain=dict(u_v="u>=0, v>=0, u+v<=1", magnitude_mg=[0., maximum],
            theta_rad=[0.,float(np.radians(policy['direction_half_angle_deg']))],phi_rad=[0.,2*np.pi],
            intrinsic_dimension_upper_bound=5, ambient_dimension=6,
            interpretation="mechanical demand on entire requested surface; access exceptions flagged separately"),
        reachability=dict(ray_offset_m=policy["visibility_ray_offset_m"], ray_batch_size=policy["ray_batch_size"],
            positive_ray_hits_confirmed_with_float64_triangles=True,
            escape_direction="negative nominal needle axis; geometric screening independent of force uncertainty", force_applied_at="q, never the offset ray origin",
            rejection_sampling=False, unresolved_faces_retained=True),
        geometry=dict(vertices_m=vertices.tolist(), faces=mesh.faces.tolist(), work_face_ids=ids.tolist(),
            inward_normals=inward.tolist(), work_face_areas_m2=mesh.area_faces[ids].tolist(),
            face_passes_pose_geometry=np.isin(ids, passed).tolist(), filled_face_ids=pose["filled_face_ids"],
            selected_area_m2=float(mesh.area_faces[ids].sum()), total_area_m2=float(mesh.area)),
        ground_state_independent=True, ground_contact_assumed=False,
        fixture_weight_included=False, source_pose_status=pose["status"])
    return WrenchDomain(data)


def sample_domain(domain, count=32768, seed=20260907, *, check_visibility=True):
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("Positive integer sample count required")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("Nonnegative integer sample seed required")
    random = np.random.Generator(np.random.PCG64(seed)).random((count, 6))
    areas = np.asarray(domain.data["geometry"]["work_face_areas_m2"])
    cdf = np.cumsum(areas/areas.sum())
    cdf[-1] = 1.
    ids = np.searchsorted(cdf, random[:, 0], side="right")
    u, v = random[:, 1].copy(), random[:, 2].copy()
    reflected = u+v > 1
    u[reflected], v[reflected] = 1-u[reflected], 1-v[reflected]
    theta=np.arccos(1-random[:,3]*(1-np.cos(domain.half_angle)))
    phi=2*np.pi*random[:,4] if domain.half_angle else np.zeros(count)
    magnitude = domain.k*random[:, 5]
    result = domain.evaluate(ids, u, v, magnitude,theta=theta,phi=phi,check_visibility=check_visibility)
    result['visibility_checked']=np.full(count,check_visibility,dtype=bool)
    if not check_visibility:
        result['geometrically_screened']=domain.face_screened[ids]
    world_from_object = np.asarray(domain.data["frame"]["T_world_mesh"])
    rotation, translation = world_from_object[:3, :3], world_from_object[:3, 3]
    world_origin = result["need_wrench"].copy()
    world_origin[:, 3:] += np.cross(domain.com, world_origin[:, :3])
    scale = np.r_[np.ones(3), np.ones(3)/domain.mesh.extents.max()]
    scaled = result["need_wrench"]*scale
    result.update(work_face_index=ids, parameters=np.c_[u,v,theta,phi,magnitude],
        pt_object_m=(result["pt_m"]-translation) @ rotation,
        need_wrench_object=np.c_[result["need_wrench"][:, :3] @ rotation, result["need_wrench"][:, 3:] @ rotation],
        need_wrench_SI=result["need_wrench"]*domain.weight_N,
        need_wrench_world_origin=world_origin, need_wrench_world_origin_SI=world_origin*domain.weight_N,
        wrench_scale=scale, targets_scaled=scaled, targets_7d=np.c_[scaled, np.zeros(count)],
        moment_origin_m=domain.com, gravity_force_mg=domain.gravity,
        weight_per_sample=np.asarray(1./count), sample_seed=np.asarray(seed))
    return result


def endpoint_demands(domain):
    """Finite demands covering every triangle position and allowed force.

    Local cones use a conservative outer polytope, paired with each face's
    three vertices. Passing every target certifies the continuous domain for
    one convex reaction set. Outer-target failure alone is inconclusive.
    The legacy fixed-axis branch remains exact. Neither certifies tool access.
    """
    if domain.direction_model=='world_needle_direction_cone':
        from .surface_hull import hull_vertex_ids
        source=np.unique(domain.mesh.faces[domain.work_ids])
        keep,proof=hull_vertex_ids(domain.mesh.vertices[source])
        source=source[keep]
        cfg=domain.data['load']['continuous_cover']
        vertices,_,_=outer_cap(np.degrees(domain.half_angle),cfg['azimuth_sides'],cfg['polar_intervals'])
        vertices=vertices[np.linalg.norm(vertices,axis=1)>1e-12]
        u,v=tangent_frame(domain.axis)
        forces=domain.k*(vertices[:,0,None]*u+vertices[:,1,None]*v+vertices[:,2,None]*domain.axis)
        rows=demand(domain.mesh.vertices[source,None],forces[None],domain.com,domain.gravity).reshape(-1,6)
        values=np.vstack([np.r_[-domain.gravity,np.zeros(3)],rows])
        return dict(source_vertex_ids=np.r_[-1,np.repeat(source,len(forces))],
            force_cover_vertex_ids=np.r_[-1,np.tile(np.arange(len(forces)),len(source))],
            endpoint_need_wrench=values,endpoint_need_wrench_SI=values*domain.weight_N,
            application_hull_vertex_ids=source,application_hull_maximum_residual_m=np.asarray(proof['maximum_residual_m']),
            unreduced_source_vertex_count=np.asarray(proof['source_point_count']))
    if domain.direction_model=='local_inward_normal_cone':
        cfg=domain.data['load']['continuous_cover']
        vertices,_,_=outer_cap(np.degrees(domain.half_angle),cfg['azimuth_sides'],cfg['polar_intervals'])
        vertices=vertices[np.linalg.norm(vertices,axis=1)>1e-12]
        force=domain.k*(vertices[None,:,0,None]*domain.tangent1[:,None,:]+
            vertices[None,:,1,None]*domain.tangent2[:,None,:]+vertices[None,:,2,None]*domain.inward[:,None,:])
        q=domain.triangles[:,:,None,:]
        rows=demand(q,force[:,None,:,:],domain.com,domain.gravity).reshape(-1,6)
        ids=np.repeat(domain.mesh.faces[domain.work_ids].reshape(-1),len(vertices))
        values=np.vstack((np.r_[-domain.gravity,np.zeros(3)],rows))
        return dict(source_vertex_ids=np.r_[-1,ids],source_face_ids=np.r_[-1,np.repeat(domain.work_ids,3*len(vertices))],
            force_cover_vertex_ids=np.r_[-1,np.tile(np.arange(len(vertices)),3*len(domain.work_ids))],
            endpoint_need_wrench=values,endpoint_need_wrench_SI=values*domain.weight_N)
    vertex_ids = np.unique(domain.mesh.faces[domain.work_ids])
    vertices = domain.mesh.vertices[vertex_ids]
    force = np.broadcast_to(domain.k*domain.axis, vertices.shape)
    wrenches = demand(vertices, force, domain.com, domain.gravity)
    gravity = np.r_[-domain.gravity, np.zeros(3)]
    values = np.vstack((gravity, wrenches))
    return dict(source_vertex_ids=np.r_[-1, vertex_ids], endpoint_need_wrench=values,
                endpoint_need_wrench_SI=values*domain.weight_N)
