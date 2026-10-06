"""Keep native inputs; fall back to independent outward facet prisms for wrapping."""
from pathlib import Path
import prepare_five_pose_group_fast as preparation
from co_common import np,G,I,save

_original_run=preparation.step32.run


def facet_cell(mesh,triangle,index,thickness):
    normal=mesh.face_normals[index]
    if not np.isfinite(normal).all() or np.linalg.norm(normal)<.99:raise RuntimeError('Invalid original facet normal')
    return np.vstack([triangle,triangle+thickness*normal])


def surface_run(name,group,states,mesh,out,thickness=.005):
    try:return _original_run(name,group,states,mesh,out,thickness)
    except RuntimeError as error:
        if str(error) not in ['Full surface offset unresolved; cannot label this as force failure','Non-outward wrap offset']:
            raise
    stage=preparation.step32;offset_function=stage.wrap_offsets;head_function=G.head_cell
    try:
        stage.wrap_offsets=lambda mesh,depth:None
        G.head_cell=lambda mesh,triangle,index,offsets:facet_cell(mesh,triangle,index,thickness)
        wrapped,triangles,sources,report=_original_run(name,group,states,mesh,out,thickness)
    finally:
        stage.wrap_offsets=offset_function;G.head_cell=head_function
    report.update(surface_offset_backend='independent outward original-face prisms',
        surface_offset_fallback_reason='shared vertex displacement cannot be outward for every incident original facet',
        geometry_kind='Union of outward original non-work-facet prisms; subtract original body and all original work-facet prisms',
        nominal_facet_prism_depth_m=thickness)
    report['provenance']['code'].update(I.hashes([Path(__file__)]))
    save(out/'data/report.json',report)
    I.check_report(out/'data/report.json')
    return wrapped,triangles,sources,report


def prepare_group_ready(name,group):
    original=preparation.step32.run
    preparation.step32.run=surface_run
    try:return preparation.prepare_group_fast(name,group)
    finally:preparation.step32.run=original

CommonSurfaceInfeasible=preparation.CommonSurfaceInfeasible
