"""One isolated exact Boolean/force evaluation, using the frozen run sources."""
import argparse
import faulthandler
import json
from .common import *
from .model import Model,Layout


def write_result(result,path):
    layout=result['layout'];mesh=result_mesh(result)
    arrays=dict(vertices=mesh.vertices,faces=mesh.faces,placements=layout.placements,
                directions=layout.directions,hosts=layout.hosts,active=np.array(layout.active))
    for k in layout.active:
        arrays[f'mask_{k}']=result['masks'][k]
        arrays[f'supply_{k}']=result['supplies'][k]
        arrays[f'triangles_{k}']=result['contacts'][k]['triangles']
        arrays[f'sources_{k}']=result['contacts'][k]['sources']
    np.savez_compressed(path.with_suffix('.npz'),**arrays)
    keys=['counts','diagnostics','volume_cm3','maximum_projected_footprint_m2','seconds','classifiers']
    metadata={key:result[key] for key in keys}
    if 'timings' in result:
        metadata['timings']=result['timings']
    metadata['component_volumes_cm3'] = sorted([C.material_volume(p)*1e6 for p in result['remaining'].decompose()], reverse=True)
    if abs(abs(mesh.volume)*1e6-result['volume_cm3']) > TOL*1e6:
        raise RuntimeError('Accepted solid volume differs from its exact exported boundary')
    if 'actual_work_surface_checks' in result:
        metadata['actual_work_surface_checks']=result['actual_work_surface_checks']
    C.save(path.with_suffix('.json'),metadata)


def read_result(path,serial):
    arrays=np.load(path.with_suffix('.npz'))
    metadata=json.loads(path.with_suffix('.json').read_text())
    layout=Layout(arrays['placements'],arrays['directions'],arrays['hosts'],tuple(map(int,arrays['active'])))
    contacts={}
    for k in layout.active:
        tri=arrays[f'triangles_{k}']
        contacts[k]=dict(triangles=tri,sources=arrays[f'sources_{k}'],triangle_count=len(tri),
            area_m2=float(np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1).sum()/2))
    mesh=C.trimesh.Trimesh(arrays['vertices'],arrays['faces'],process=False)
    classifiers={int(k):v for k,v in metadata.pop('classifiers').items()}
    if abs(abs(mesh.volume)*1e6-metadata['volume_cm3']) > TOL*1e6:
        raise RuntimeError('Exact worker material volume changed in transport')
    metadata['lossless_worker_geometry_transport'] = True
    return dict(layout=layout,serial=serial,remaining=None,remaining_mesh=mesh,contacts=contacts,
        masks={k:arrays[f'mask_{k}'] for k in layout.active},
        supplies={k:arrays[f'supply_{k}'] for k in layout.active},classifiers=classifiers,**metadata)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--poses',required=True)
    parser.add_argument('--object',default='B')
    parser.add_argument('--initialization-report',type=Path)
    parser.add_argument('--layout',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    faulthandler.enable()
    faulthandler.dump_traceback_later(60,repeat=True)
    arrays=np.load(args.layout)
    model=Model(args.poses.split(','),args.object,initialization_report=args.initialization_report)
    layout=Layout(arrays['placements'],arrays['directions'],arrays['hosts'],tuple(map(int,arrays['active'])))
    write_result(model.exact(layout),args.out)


if __name__=='__main__':
    main()
