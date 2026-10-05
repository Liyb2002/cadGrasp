"""Register all saved poses into one original-object coordinate system."""
from co_common import *

def run(name,group,out):
    out.mkdir(parents=True,exist_ok=True);states={};reference=None;rows=[];inputs=[]
    for pose in group['poses']:
        task,T,mesh=state(name,pose)
        if reference is None:reference=mesh
        np.testing.assert_array_equal(mesh.faces,reference.faces)
        np.testing.assert_allclose(mesh.vertices,reference.vertices,atol=1e-10,rtol=0)
        states[pose]=(task,T,mesh);inputs+=task.inputs
        rows.append(dict(pose=pose,T_fixture_to_world=T.tolist(),T_world_to_fixture=np.linalg.inv(T).tolist(),alignment_max_error_m=float(np.max(np.abs(mesh.vertices-reference.vertices))),work_faces=task.domain.work_ids.tolist()))
    D.export_exact_obj(reference,out/'registered_object.obj')
    report=dict(complete=True,passed=True,object=name,pose_set=group['id'],poses=group['poses'],coordinate_frame='original object mesh, meters',object_native_poses_fixed=True,registered_objects_coincide=True,support_can_reorient_between_states=True,states=rows,provenance=provenance(inputs+[ROOT/'objects'/name/'pose_sets.json'],code_sources()),artifacts={'../registered_object.obj':I.sha256(out/'registered_object.obj')})
    save(out/'data/report.json',report)
    return states,reference,report
