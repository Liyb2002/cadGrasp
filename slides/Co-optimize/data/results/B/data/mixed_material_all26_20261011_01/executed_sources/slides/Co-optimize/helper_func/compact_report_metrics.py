"""Physical operation deltas and exact projected footprint; no mesh edits."""
import _bootstrap
import argparse
from scipy.spatial import ConvexHull
from co_common import *
from whole_pipeline import load_layout
from run_all import saved_groups,output_root


def metrics(model_mesh,native,before,after,poses):
    center=model_mesh.center_mass[None];rows=[]
    for k in after.active:
        a=native[before.hosts[k]]@before.placements[k]
        b=native[after.hosts[k]]@after.placements[k]
        shift=transform_points(center,b)[0]-transform_points(center,a)[0]
        da=before.placements[k,:3,:3].T@before.directions[k]
        db=after.placements[k,:3,:3].T@after.directions[k]
        angle=float(np.degrees(np.arccos(np.clip(da@db,-1.,1.))))
        if np.linalg.norm(shift)>1e-10 or angle>1e-5 or before.hosts[k]!=after.hosts[k]:
            rows.append(dict(pose=poses[k],world_object_center_displacement_mm=(1000*shift).tolist(),
                world_object_center_displacement_norm_mm=float(np.linalg.norm(shift)*1000),
                object_relative_exit_angle_degrees=angle,
                previous_host=poses[before.hosts[k]],host=poses[after.hosts[k]]))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('object',nargs='?',default='B')
    args=parser.parse_args();count=0
    for group in saved_groups(args.object):
        source=output_root(args.object)/group['id']/'step4/step4.2'
        for mode in ['greedy','structural-greedy','beam']:
            out=source/'compact'/mode
            if not (out/'data/report.json').exists():continue
            report=I.check_report(out/'data/report.json');before=load_layout(source/'layout.npz')
            after=load_layout(out/'layout.npz');states=[state(args.object,p) for p in group['poses']]
            mesh=states[0][2];native=np.array([s[1] for s in states])
            support=trimesh.load(out/'support.obj',force='mesh',process=False)
            protected=I.hashes([out/'support.obj',out/'layout.npz',out/'process.json']+
                list(out.glob('*_force.npz'))+list((out/'process_states').glob('*.npz')))
            path=[];old=before
            for row in json.loads((out/'process.json').read_text()):
                new=load_layout(out/row['layout'])
                path.append(dict(index=row['index'],phase=row['phase'],
                    changes=metrics(mesh,native,old,new,group['poses'])))
                old=new
            details=dict(complete=True,presentation_only=True,geometry_or_mechanics_changed=False,
                pose_set=group['id'],method=mode,
                physical_endpoint_changes=metrics(mesh,native,before,after,group['poses']),
                selected_path=path,
                note='Changing host changes fixture coordinates; physical motion uses WORLD body centers and BODY-relative directions',
                provenance=provenance([source/'layout.npz',out/'layout.npz',out/'process.json'],[Path(__file__)]))
            save(out/'data/operation_metrics.json',details)
            footprints=[float(ConvexHull(transform_points(support.vertices,native[after.hosts[k]])[:,:2]).volume)
                for k in after.active]
            if abs(max(footprints)-report['maximum_projected_footprint_m2'])>1e-12:
                report['previous_projected_footprint_m2']=report['maximum_projected_footprint_m2']
                report['maximum_projected_footprint_m2']=max(footprints)
                report['footprint_metadata_corrected_without_geometry_or_ranking_change']=True
            report['final_physical_operation_metrics']='data/operation_metrics.json'
            report['provenance']['code'].update(I.hashes([Path(__file__)]))
            current=dict(report);current['artifacts']={
                k[3:] if k.startswith('../') else k:v for k,v in report['artifacts'].items()}
            current['artifacts']['data/operation_metrics.json']=I.sha256(out/'data/operation_metrics.json')
            save(out/'report.json',current);data=current.copy()
            data['artifacts']={'../'+k:v for k,v in current['artifacts'].items()}
            save(out/'data/report.json',data);I.check_report(out/'data/report.json')
            for relative,digest in protected.items():assert I.sha256(ROOT/relative)==digest
            count+=1
    print('COMPACT PHYSICAL METRICS',count,flush=True)


if __name__=='__main__':main()
