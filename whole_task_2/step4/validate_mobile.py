"""Independent replay of saved complete solids and centered execution poses."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np

from .mobile import MobileTaskModel
from .export_geometry import exclusions
from whole_task_2.step1.common import DEFAULT_OUTPUT, sha256, write_json
from whole_task_2.step1.pose_coverage import spatial_workspace_mask
from whole_task_2.step3.legacy import kernel as K
from whole_task_2.common.wrenches import endpoint_demands
from whole_task_2.common.force_cone import certificate_description
from whole_search.model import Layout
from whole_search.common import transform_mesh
from material_force import certify_material_contacts
from prune_support import ForceCertificate
from feasible_geometry import validate as validate_complete


def validate_artifacts(model, folder, layout):
    folder=Path(folder)
    second = folder/'step4.2'
    model.check_station(layout)
    core = K.trimesh.load(second/'support.obj', force='mesh', process=False)
    full = K.trimesh.load(second/'complete_support.obj', force='mesh', process=False)
    witness = K.trimesh.load(second/'legal_component.obj', force='mesh', process=False)
    oracle_rows = model.base_oracle.rows(layout)
    from system_floor import system_floor_demands
    from base_oracle import cover
    sample_pressure_gaps=[]
    for k,row in zip(layout.active,oracle_rows):
        U=row['U'];com=U[:3,:3]@model.mesh.center_mass+U[:3,3]
        xy=system_floor_demands(model.tasks[k].targets/model.tasks[k].scale,com)['floor_demands_xy_m']
        passed,gap,_=cover(xy,row['hull'])
        if not passed:raise ValueError('Continuous pressure hull omits original samples')
        sample_pressure_gaps.append(gap)
    rows = [dict(pose=r['pose'], fixture_transform=r['T'], object_ground=r['own'],
                 hull=np.c_[r['hull'],np.zeros(len(r['hull']))], object_transform=r['U']) for r in oracle_rows]
    checker = ForceCertificate(model.poses, layout, model.mesh, rows, model.name, states=model.states)
    proof, geometry = validate_complete(full, witness, model.mesh,
        layout.placements[list(layout.active)], rows, checker, mandatory=core)
    # Witness containment alone is insufficient to test an incorrectly built
    # witness. Rebuild actual finite work and full exit solids independently.
    work, bodies, sweeps = exclusions(model, layout, full)
    solid = K.S.solid(full)
    overlaps = {name:[K.material_volume(solid ^ x) for x in pieces]
                for name,pieces in [('work',work),('object',bodies),('exit',sweeps)]}
    if max(v for values in overlaps.values() for v in values) > 1e-10:
        raise ValueError('Complete solid intersects a physical exclusion: '+str(overlaps))
    endpoint_counts = {}
    for k in layout.active:
        needs = endpoint_demands(model.tasks[k].domain)['endpoint_need_wrench']*model.tasks[k].scale
        mask,_ = K.J.classify(proof['supplies'][k], needs)
        endpoint_counts[model.poses[k]] = dict(passed=int(mask.sum()), total=len(mask))
        if not mask.all():
            raise ValueError('Conservative force-cone cover failed; true-cone infeasibility is not established')
    recorded = json.loads((folder/'execution.json').read_text())['poses']
    execution_checks = []
    for k in layout.active:
        actual = model.execution(layout, k)
        for key in ['world_from_object','world_from_support','head_world_m','work_window_center_m']:
            np.testing.assert_allclose(recorded[k][key], actual[key], atol=1e-12, rtol=0)
        T, U = np.asarray(actual['world_from_support']), np.asarray(actual['world_from_object'])
        body_world = transform_mesh(model.mesh, U)
        support_world = transform_mesh(full, T)
        # Check the actual centered execution against the first-ring work
        # volume as well: translating a fixture cannot enlarge that volume.
        workspace = deepcopy(model.station['workspace'])
        shift = np.asarray(actual['work_window_translation_m'])
        workspace['center_world_m'] = (np.asarray(workspace['center_world_m'])+shift).tolist()
        cone = workspace['spatial_cone']
        cone['apex_world_m'] = (np.asarray(cone['apex_world_m'])+shift).tolist()
        work = body_world.triangles[model.tasks[k].domain.work_ids]
        if not spatial_workspace_mask(work,workspace).all():
            raise ValueError('Centered execution puts working faces outside the first solid ring')
        work_depth = (work-np.asarray(cone['apex_world_m']))@np.asarray(cone['axis_world'])
        if body_world.bounds[0,2]<-1e-9 or support_world.bounds[0,2]<-1e-8:
            raise ValueError('Centered execution crosses ground')
        np.testing.assert_allclose(body_world.vertices[model.floor_vertices[k]], actual['lowest_vertex_world_m'], atol=1e-12)
        # Recompute needs in execution coordinates, rather than only asserting
        # transform identities. COM-relative wrenches must be unchanged.
        task=model.tasks[k]
        u,v,theta,phi,magnitude=task.parameters.T
        samples=task.domain.evaluate(task.work_face_index,u,v,magnitude,
            check_visibility=False,theta=theta,phi=phi)
        points=samples['pt_m']+U[:3,3]-model.native[k,:3,3]
        com=task.domain.com+U[:3,3]-model.native[k,:3,3]
        from whole_task_2.common.wrenches import demand
        translated=demand(points, samples['force_push_mg'], com, task.domain.gravity)
        np.testing.assert_allclose(translated*task.scale,task.targets,atol=1e-13,rtol=0)
        execution_checks.append(dict(pose=model.poses[k], grounded=model.floor_signature(layout,k),
            first_solid_ring_work_limit_passed=True,
            minimum_work_depth_m=float(work_depth.min()), maximum_work_depth_m=float(work_depth.max()),
            first_solid_ring_distance_m=cone['reference_axial_distance_m'],
            maximum_usable_work_depth_m=cone['maximum_axial_distance_m'],
            minimum_object_height_m=float(body_world.bounds[0,2]), minimum_support_height_m=float(support_world.bounds[0,2]),
            recentering_and_wrenches_verified=True, geometric_reach_passed=True))
    result=dict(status='passed', full_fixture_accepted=True,
        complete_geometry=geometry, complete_path_overlap_m3=overlaps,
        continuous_endpoint_counts=endpoint_counts, execution=execution_checks,
        continuous_load_cover=certificate_description(model.tasks[0].domain),
        continuous_floor_envelope_verified=True, original_sample_pressure_hull_gaps_m=sample_pressure_gaps,
        artifact_sha256={str(p.relative_to(folder)):sha256(p) for p in [second/'layout.npz',second/'support.obj',
            second/'complete_support.obj',second/'legal_component.obj',folder/'execution.json']})
    return result


def validate(folder):
    folder = Path(folder)
    report = json.loads((folder/'report.json').read_text())
    for path, digest in report['input_sha256'].items():
        if sha256(path) != digest:
            raise ValueError('Changed run input: '+path)
    model = MobileTaskModel(report['poses'], report['object'], report['output_root'])
    with np.load(folder/'step4.2/layout.npz') as data:
        layout = Layout.from_arrays(data)
    result=validate_artifacts(model,folder,layout)
    result['report_sha256']=sha256(folder/'report.json')
    write_json(folder/'validation.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',nargs='?',type=Path,default=DEFAULT_OUTPUT/'B/step4/movable_station')
    a=p.parse_args();print(json.dumps(validate(a.folder),indent=2))


if __name__=='__main__':main()
