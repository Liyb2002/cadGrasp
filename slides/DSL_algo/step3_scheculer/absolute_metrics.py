"""Use current baseline Step5, selecting the actual accepted witness explicitly."""
from pathlib import Path
import numpy as np
import trimesh
from step3_scheculer import contacts as I
from step5_current import evaluate as M,render_bbox as V


def sources():return [Path(__file__),Path(M.__file__)]+V.SOURCES


def measure(tasks,state,mesh):
    return M.measure([t.domain.mesh.vertices for t in tasks],mesh.vertices,state.bases,state.offsets)


def volume(tasks,state,mesh):return measure(tasks,state,mesh)[0]['object_and_support_poses']['box_volume_cm3']


def evaluate(group,tasks,state,shape,record,out,render=True):
    shape,record,out=Path(shape),Path(record),Path(out);out.mkdir(parents=True,exist_ok=True)
    I.check_report(record)
    mesh=trimesh.load(shape,force='mesh',process=False)
    aggregate,per_pose,supports=measure(tasks,state,mesh)
    artifacts={}
    if render:
        V.draw(out/'bbox.png',[t.domain.mesh for t in tasks],supports,mesh.faces,aggregate)
        artifacts['bbox.png']=I.sha256(out/'bbox.png')
    report=dict(complete=True,passed=True,schema='absolute_dsl_step5_v7',poses=[t.pose for t in tasks],pose_count=len(tasks),
        metrics=aggregate,per_pose=[dict(pose=t.pose,**row) for t,row in zip(tasks,per_pose)],
        definition=dict(primary='XYZ occupied box volume of all object and installed support poses in saved workstation axes',
            secondary='XY occupied box area',material_volume_is_primary=False,recentered=False,motion_sweeps_included=False,
            source='latest copied baseline Step5 measure; actual saved task geometries and accepted witness selected explicitly'),
        material_volume_cm3=float(mesh.volume*1e6),geometry_changed=False,force_or_geometry_acceptance_rerun=False,
        provenance=dict(inputs=I.hashes([shape,record]+[p for t in tasks for p in t.inputs]),code=I.hashes(sources())),artifacts=artifacts)
    I.save(out/'report.json',report)
    return report
