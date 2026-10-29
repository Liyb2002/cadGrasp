"""Legal needle-axis alignment and advisory contact signatures for repair."""
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from subdivide_task.operations.state import transform


def closest_axial_angle(owner, host, axis):
    """Analytic SO(3) distance minimizer over the legal needle-axis spin."""
    axis = np.asarray(axis, float); axis = axis/np.linalg.norm(axis)
    x,y,z = axis
    cross = np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    relative = owner[:3,:3]@host[:3,:3].T
    angle = np.arctan2(np.trace(cross@relative),
        np.trace(relative)-axis@relative@axis)
    adjusted = Rotation.from_rotvec(axis*angle).as_matrix()@relative
    return float(np.degrees(angle)), float(Rotation.from_matrix(adjusted).magnitude())


def remember_contacts(model, tasks):
    """Weight actual available contact faces by representative paired demands.

    This is a soft relevance signature, not a claim that one LP basis is the
    required contact set. All original source faces remain free to change.
    """
    model.current_contacts = getattr(model,'current_contacts',{}).copy()
    model.contact_profiles = getattr(model,'contact_profiles',{}).copy()
    model.retention_witnesses=getattr(model,'retention_witnesses',{}).copy()
    for task in tasks:
        data = task['contact_data']
        from subdivide_task.operations.force_witness import required_generators
        witness,witness_report=required_generators(task,data['supply_7d'])
        model.retention_witnesses[task['id']]=witness
        tri = data['contact_triangles_world_m']
        sources = data['contact_source_face_ids']
        model.current_contacts[task['id']] = tri.reshape(-1,3)
        centers = tri.mean(1)
        normals = -task['body_world'].face_normals[sources]
        rays = np.c_[normals, np.cross(centers-task['domain'].com, normals)]
        scale = data['wrench_scale']
        rays *= scale
        rays /= np.maximum(np.linalg.norm(rays,axis=1,keepdims=True),1e-15)
        needs = task['samples']['need_wrench']
        needs = needs[np.linspace(0,len(needs)-1,32).astype(int)]*scale
        needs /= np.maximum(np.linalg.norm(needs,axis=1,keepdims=True),1e-15)
        relevance = .25 + .75*np.maximum(rays@needs.T,0.).mean(1)
        weights = trimesh.triangles.area(tri)*relevance
        profile = np.bincount(sources,weights=weights,minlength=len(model.source.faces))
        model.contact_profiles[task['id']] = profile/max(profile.sum(),1e-15)


