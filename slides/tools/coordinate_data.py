"""Schema-aware, one-time conversion of stored scientific coordinates to Z-up.

Indices, colors and contact-plane coordinates are not world vectors. These
explicit fields keep them out of the axis permutation. Barycentric coordinates
follow the corresponding reversed triangle winding.
"""
import re
import numpy as np
import coordinate_transport as C

POLAR = set('''arrow_head_m arrow_tail_m arrow_surface_end_points_m center_m center
    centers_m centerlines_m centre_mm com_m com_mesh_frame com_world com_world_m
    connector_centerline_m detail_focus_m direction candidate_directions
    displayed_direction_vectors displayed_insertion_directions displayed_withdrawal_directions
    floor_contact_m focus_m force_balance_residual_mg force_mg force_on_support_direction
    force_push_mg force_w gravity_application_point_m gravity_force_mg gravity_in_mesh_frame
    inward_direction inward_normals local_constraint_normals lookat_m moment_origin_m
    normals normal_rows origin_m original_pivot_m outward_normal p point points_m
    position_m pos preferred_withdrawal_direction pt_m push pushes q_m skin_centres_m
    skin_outward_normals tangent1 tangent2 terminals_m translation_m up vector vectors
    vertices_m part_vertices_m union_vertices_m view_toward_camera camera_vector
    camera_view_vector waypoints_m weakest_direction weakest_outward_normal withdrawal_direction
    sample_pt sample_force support_union_vertices_m support_part_vertices_m contact_centers_m contact_normals contact_points_m ground_forces ground_points_m bounds_m bounds_mm
    extents_m arrow_points_m arrow_directions sampled_forces_mg floor_point_m ground_corners_m floor_contact_corners_m d anchor translation v points supports patch_min_rho_at upper focus camera_position camera_target direction_vector'''.split())
AXIAL = set('''axis moment_balance_residual_mgm moment_from_component_expansion_mgm
    moment_mg_m moment_wmm gravity_torque_about_original_floor_point_mgm'''.split())
WRENCH = set('''support_wrenches sample_wrench external_wrench head_resultant_on_support head_resultant_on_workpiece
    motion need_wrench physical_dual push_wrench target_wrench load_wrenches
    continuous_outer_load_wrenches full target targets floor6 dual'''.split())
TRIANGLES = {'triangles', 'triangles_m', 'floor_triangles_m', 'support_floor_triangles_m', 'contact_triangles_m'}
FACES = {'faces', 'union_faces', 'part_faces', 'f', 'support_union_faces', 'support_part_faces'}
CAMERA = {'basis', 'camera_basis'}
MATRICES = {'inertia_com'}
POSES = {'T_world_mesh', 'poses', 'transform', 'rotation_matrix', 'rotation'}


def world_key(key):
    """Rename world planar fields; leave local parameter-plane names unchanged."""
    return key.replace('ground_min_y_m', 'ground_min_z_m').replace(
        'minimum_work_vertex_y_m', 'minimum_work_vertex_z_m').replace('_xz_', '_xy_').replace('_normal_y', '_normal_z').replace(
        'withdrawal_y', 'withdrawal_z').replace('inward_y_', 'inward_z_').replace(
        'workpiece_y', 'workpiece_z').replace('common_top_y_', 'common_top_z_').replace(
        'normal_y_guard', 'normal_z_guard').replace('cut_normal_xz', 'cut_normal_xy')



def field(key, value):
    a = np.asarray(value)
    if a.size == 0 or a.dtype.kind not in 'fiu':
        return value
    if key == 'parameters' and a.shape[-1:] == (5,):
        return a[..., [1, 0, 2, 3, 4]].copy()
    if key in TRIANGLES or re.fullmatch(r'C\d+_\d+', key):
        return C.triangles(a) if a.shape[-2:] == (3, 3) else value
    if key in FACES and a.ndim >= 2 and a.shape[-1] == 3:
        return C.faces(a)
    if key in CAMERA and a.shape[-2:] == (3, 3):
        return C.polar(a)
    if key in MATRICES and a.shape[-2:] == (3, 3):
        return C.rotation(a)
    if key in POSES and a.shape[-2:] == (4, 4):
        return C.transform(a)
    if key in POSES and a.shape[-2:] == (3, 3):
        return C.rotation(a)
    if key == 'quat_wxyz' and a.shape[-1:] == (4,):
        return C.quaternion_wxyz(a)
    if key in AXIAL and a.shape[-1:] == (3,):
        return C.axial(a)
    if key in POLAR and a.shape[-1:] == (3,):
        return C.polar(a)
    if (key in WRENCH or re.fullmatch(r'(columns|duals?|halfspaces|cone_axis)_\d+', key)
            or re.fullmatch(r'without_\d+_dual', key)) and a.shape[-1:] in ((6,), (7,)):
        return C.wrench(a)
    if key == 'scale' and a.shape == (6,):
        return a[C.WRENCH_ORDER]
    if key == 'equilibrium_matrix' and a.ndim == 2 and a.shape[0] % 6 == 0:
        result = a.copy()
        for first in range(0, len(a), 6):
            result[first:first+6] = C.wrench(a[first:first+6].T).T
        return result
    return value


def record(value):
    if isinstance(value, list):
        return [record(v) for v in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if isinstance(item, dict):
            converted = record(item)
        elif isinstance(item, list):
            try:
                converted = field(key, item)
            except (ValueError, TypeError):
                converted = item
            if isinstance(converted, np.ndarray):
                converted = converted.tolist()
            else:
                converted = record(converted)
        else:
            converted = item
        result[world_key(key)] = converted
    if result.get('world_up') == '+Y':
        result['world_up'] = '+Z'
    if all(k in value for k in ('u', 'v', 'work_face_index')):
        result['u'], result['v'] = value['v'], value['u']
    if 'parameters' in value and isinstance(value['parameters'], dict):
        params = value['parameters']
        if 'u' in params and 'v' in params:
            result['parameters'].update(u=params['v'], v=params['u'])
    if 'lower_wrench' in value and 'upper_wrench' in value:
        first = C.wrench(value['lower_wrench']); second = C.wrench(value['upper_wrench'])
        result['lower_wrench'] = np.minimum(first, second).tolist()
        result['upper_wrench'] = np.maximum(first, second).tolist()
    return result


def arrays(value):
    return {world_key(k): np.asarray(field(k, v)) for k, v in value.items()}
