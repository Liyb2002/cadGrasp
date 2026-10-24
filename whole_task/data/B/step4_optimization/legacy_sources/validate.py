"""Replay actual saved Step4 geometry, station transforms and paired loads."""
import argparse
import json
from pathlib import Path
import numpy as np
from whole_task.step1.common import DEFAULT_OUTPUT, sha256, write_json
from whole_task.step3.validate import check_binding
from .adapter import TaskModel
from whole_search.model import Layout
from whole_search.common import transform_solid
from whole_task.step3.legacy import kernel as K
from material_force import certify_material_contacts
from prune_support import material_components, ForceCertificate
from feasible_geometry import validate as validate_complete
from fixture_state import installed_state


def validate_directory(folder,recheck_exits=True):
    folder=Path(folder);report=json.loads((folder/'report.json').read_text())
    check_binding(folder,report)
    model=TaskModel(report['poses'],report['object'],report['output_root'])
    rows=[]
    for stage in ['step4.1','step4.2']:
        out=folder/stage
        with np.load(out/'layout.npz')as arrays:layout=Layout.from_arrays(arrays)
        model.check_station(layout)
        mesh=K.trimesh.load(out/'support.obj',force='mesh',process=False)
        if not mesh.is_watertight or not mesh.is_winding_consistent:raise ValueError('Invalid saved support boundary')
        solid=K.S.solid(mesh)
        work=[];body=[];exit_volumes=[];floor=[];endpoint_gaps=[]
        for k in layout.active:
            Q=layout.placements[k];T=installed_state(model.native,layout,layout.hosts[k])
            work.append(K.material_volume(solid ^ transform_solid(model.work_rays[k].solid(),Q)))
            body.append(K.material_volume(solid ^ transform_solid(model.body,Q)))
            floor.append(float((mesh.vertices@T[2,:3]+T[2,3]).min()))
            d=Q[:3,:3].T@layout.directions[k]
            if layout.directions[k]@T[2,:3]<-1e-10:raise ValueError('Exit direction crosses floor')
            local=(mesh.vertices-Q[:3,3])@Q[:3,:3]
            length=max(model.length,float((local@d).max()-(model.mesh.vertices@d).min())+.01)
            endpoint_gaps.append(float((model.mesh.vertices@d).min()+length-(local@d).max()))
            if recheck_exits:
                swept=transform_solid(model.clearance.sweep(length*d,padded=False),Q)
                exit_volumes.append(K.material_volume(solid ^ swept))
        if max(work+body+exit_volumes)>1e-10:raise ValueError('Saved material intersects an exclusion')
        if min(floor)<-1e-8:raise ValueError('Saved material crosses an installed floor')
        if min(endpoint_gaps)<=0:raise ValueError('Exit does not reach full separation')
        proof=certify_material_contacts(model,layout,mesh)
        for k in layout.active:
            with np.load(out/f'{model.poses[k]}_force.npz')as saved:
                np.testing.assert_array_equal(saved['mask'],proof['masks'][k])
                np.testing.assert_allclose(saved['supply_7d'],proof['supplies'][k],atol=1e-12,rtol=0)
        row=dict(stage=stage,fixed_robot_world_poses_verified=True,
            maximum_work_overlap_m3=max(work),maximum_object_overlap_m3=max(body),
            maximum_exit_overlap_m3=max(exit_volumes,default=0),full_exits_rebuilt=recheck_exits,
            minimum_exit_end_gap_m=min(endpoint_gaps),minimum_installed_floor_height_m=min(floor),
            material_components=len(material_components(solid)),actual_contact_force_counts=proof['counts'],
            full_original_demands_replayed=True)
        if stage=='step4.2':
            assert row['material_components']==1
            assert report['final_core_counts']=={model.poses[k]:int(v.sum())for k,v in proof['masks'].items()}
            if report['full_fixture_accepted']:
                full=K.trimesh.load(out/'complete_support.obj',force='mesh',process=False)
                witness=K.trimesh.load(out/'legal_component.obj',force='mesh',process=False)
                ground=[dict(pose=r['pose'],fixture_transform=r['T'],object_ground=r['own'],
                             hull=np.c_[r['hull'],np.zeros(len(r['hull']))],object_transform=r['U'])
                        for r in model.base_oracle.rows(layout)]
                checker=ForceCertificate(model.poses,layout,model.mesh,ground,model.name,states=model.states)
                fullproof,checks=validate_complete(full,witness,model.mesh,layout.placements[list(layout.active)],ground,checker,mandatory=mesh)
                with np.load(out/'complete_force_certificate.npz')as saved:
                    for k in layout.active:np.testing.assert_array_equal(saved[f'mask_{k}'],fullproof['masks'][k])
                row['complete_solid_checks']=checks
        rows.append(row)
    result=dict(status='passed',report_sha256=sha256(folder/'report.json'),stages=rows,
                full_fixture_accepted=report['full_fixture_accepted'],search_status=report['status'])
    write_json(folder/'validation.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--object',default='B');p.add_argument('--skip-exit-rebuild',action='store_true')
    a=p.parse_args();print(json.dumps(validate_directory(a.output_root/a.object/'step4',not a.skip_exit_rebuild),indent=2))


if __name__=='__main__':main()
