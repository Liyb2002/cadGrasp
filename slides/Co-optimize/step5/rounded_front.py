"""Coherent inward skin removal with certified contact ports and round bands.

The whole exterior advances inward by a physical distance. Rounded retention
bands preserve only the selected contact patches. No independent grid blocks
or internal pocket cutters determine the published surface.
"""
import numpy as np
import manifold3d as md
import trimesh
from scipy import ndimage
from skimage.measure import marching_cubes
from feasible_geometry import C,validate
from periphery import structural_profile
from continuous_legs import capsule
from deterministic_base import solid,unpack,union,volume,SCALE
from prune_support import material_components


def connector_limits(calibration):
    radius=calibration['guard_radius_m']
    terminal=0.
    for reference in calibration['references']:
        profile=next(p for p in reference['profiles']if abs(p['radius_m']-radius)<1e-12)
        sizes=profile['component_volumes_cm3']
        terminal=max(terminal,sizes[1]if len(sizes)>1 else 0.)
    return radius,terminal


def connector_quality(mesh,calibration):
    radius,terminal=connector_limits(calibration)
    profile,_,_=structural_profile(mesh,calibration['pitch_m'],(radius,))
    sizes=profile['profiles'][0]['component_volumes_cm3']
    largest_separated=sizes[1]if len(sizes)>1 else 0.
    return largest_separated<=terminal+1e-10,dict(profile=profile,
        guard_radius_m=radius,reference_maximum_terminal_core_cm3=terminal,
        measured_largest_separated_core_cm3=largest_separated)


def accepted_bands(history):
    rods=[]
    for row in history:
        if not row.get('accepted')or row.get('operation')!='connector_band':continue
        if row['method']=='short_round_band':points=[row['a_m'],row['b_m']]
        else:points=row['path_m']
        # This is a retention envelope, so cover the exact certified rod with
        # positive slack instead of intersecting nearly coincident surfaces.
        rods.extend(capsule(a,b,row['radius_m']+.00015)for a,b in zip(points,points[1:]))
    return union(rods)if rods else None


def ground_patches(stock,rows):
    patches=[];seen=set()
    for row in rows:
        T=row['fixture_transform'];key=np.round(T,11).tobytes()
        if key in seen:continue
        seen.add(key)
        z=stock.vertices@T[2,:3]+T[2,3]
        faces=stock.faces[np.all(np.abs(z[stock.faces])<1e-8,axis=1)]
        for face in faces:
            points=stock.vertices[face];center=(points.min(0)+points.max(0))/2
            extent=float(np.linalg.norm(points-center,axis=1).max())
            patches.append(dict(center_m=center.tolist(),extent_m=extent))
    return patches


def round_ports(patches,margin):
    balls=[];seen=set()
    for patch in patches:
        center=np.asarray(patch['center_m'])
        # Quantization only builds a generous retention guide. All real
        # contact/floor polygons stay in the original exact solid.
        quantized=np.round(center/.00025)*.00025
        radius=patch['extent_m']+margin+float(np.linalg.norm(center-quantized))
        radius=float(np.ceil(radius/.00025)*.00025)
        key=(*quantized.tolist(),radius)
        if key in seen:continue
        seen.add(key)
        balls.append(md.Manifold.sphere(radius/SCALE,20).translate(tuple(quantized/SCALE)))
    return union(balls)


def inner_envelope(distance,origin,pitch,erosion,inflation):
    centres=distance>erosion
    if not centres.any():return None
    field=inflation-ndimage.distance_transform_edt(~centres).astype(np.float32)*pitch
    field=ndimage.gaussian_filter(field,.45)
    if float(field.max())<=0:return None
    vertices,faces,_,_=marching_cubes(field,level=0.,spacing=(pitch,)*3)
    mesh=trimesh.Trimesh(vertices+origin,faces,process=True)
    if mesh.volume<0:mesh.invert()
    return solid(mesh)


def prune(stock,original_core,contact_records,band_history,calibration,context,
          certify,checkpoint=None,step=.0005):
    radius,terminal=connector_limits(calibration);pitch=calibration['pitch_m']
    bands=accepted_bands(band_history)
    # Do not split and reassemble coincident core/foot boundaries: preserve
    # exact floor ports in the same retention guide and intersect only once.
    floor_ports=round_ports(ground_patches(stock,context['rows']),.003)
    fixed=floor_ports if bands is None else floor_ports+bands
    profile,voxels,distance=structural_profile(stock,pitch,(radius,))
    origin=voxels.transform[:3,3]
    inflation=radius+.0005
    max_depth=max(step,float(distance.max())-inflation)
    depths=np.arange(step,max_depth+step/2,step)
    current=stock;history=[];proof=None
    patches=[p for row in contact_records for p in row['patches']]
    for margin in [.003,.002,.0015]:
        ports=round_ports(patches,margin)
        for depth in depths:
            print('FRONT_PROPOSAL','depth_mm',round(float(depth)*1000,2),'margin_mm',margin*1000,flush=True)
            guide=inner_envelope(distance,origin,pitch,inflation+depth,inflation)
            retained=ports+fixed if guide is None else ports+fixed+guide
            retained=retained.simplify(.00004/SCALE)
            proposed=solid(current)^retained
            if volume(solid(current)-proposed)<=1e-12:continue
            parts=material_components(proposed)
            variants=[proposed]if len(parts)==1 else [parts[0]]
            accepted=False;error=None
            for value in variants:
                try:
                    mesh,certificate=certify(unpack(value))
                    passed,quality=connector_quality(mesh,calibration)
                    if not passed:raise ValueError('Inward front creates a connector below the reference')
                    removed=(abs(float(current.volume))-abs(float(mesh.volume)))*1e6
                    history.append(dict(operation='coherent_inward_front',accepted=True,
                        skin_depth_m=float(depth),port_margin_m=margin,
                        retained_ball_radius_m=inflation,removed_material_cm3=removed,
                        connector_quality=quality,grid_pocket_cutting_used=False,
                        actual_full_original_demands_checked=True))
                    current=mesh;proof=certificate;accepted=True
                    print('ROUNDED_FRONT',round(float(depth)*1000,2),'mm ports',margin,
                        'removed cm3',round(removed,3),flush=True)
                    if checkpoint:checkpoint(current,proof,history)
                    break
                except (ValueError,RuntimeError)as failure:error=str(failure)
            if not accepted:
                print('FRONT_REJECT',error,flush=True)
                history.append(dict(operation='coherent_inward_front',accepted=False,
                    skin_depth_m=float(depth),port_margin_m=margin,error=error))
    return current,history,proof


if __name__=='__main__':
    from pathlib import Path
    import argparse,json,time,importlib.util
    from feasible_geometry import Layout,ForceCertificate
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('set');parser.add_argument('--source-experiment',default='outside_in_pilot_20261015_03')
    parser.add_argument('--name',default='rounded_front_probe_20261015_01')
    parser.add_argument('--step',type=float,default=.0005)
    args=parser.parse_args()
    selection=C.COOPT/'data/results/B/data/step5_show_all52_20261014_02_inputs/selection.json'
    inputs=C.Inputs('B',selection,'pso_all52_20261013_01_floor')
    case=next(c for c in inputs.cases if c['id']==args.set);data=inputs.load(case)
    with np.load(data['selected']/'layout.npz')as a:layout=Layout.from_arrays(a)
    checker=ForceCertificate(data['record']['poses'],layout,data['body'],data['rows'],'B')
    root=C.COOPT/'data/results/B'/args.set/'step5/feasible'/args.source_experiment/'outside_in'
    cache=C.COOPT/'data/results/B/data/contact_subset_cache_20261015'/args.set
    contact=json.loads((cache/'report.json').read_text());assert contact['complete']
    if (root/'retained_stock.obj').exists():
        source=root/'retained_stock.obj'
        bands=json.loads((root/'reinforcement.json').read_text())['history']
    else:
        checkpoints=sorted(p for p in(root/'checkpoints').iterdir()if(p/'report.json').exists())
        latest=checkpoints[-1];r=json.loads((latest/'report.json').read_text())
        source=latest/'support.obj';assert C.digest(source)==r['support_sha256'];bands=r['history']
    stock=C.trimesh.load(source,force='mesh',process=False)
    core=data['fixture']
    calibration=json.loads((C.COOPT/'data/results/B/data/step5_feasible_outside_in_calibration_20261015_03/thickness_calibration.json').read_text())
    witness=C.trimesh.load(data['selected']/'oracle_connected_witness.obj',force='mesh',process=False)
    context=dict(witness=witness,body=data['body'],placements=layout.placements[list(layout.active)],
        rows=data['rows'],checker=checker,bounds=C.occupied_bounds(data['body'],stock,data['rows']))
    output=C.COOPT/'data/results/B/data'/args.name/args.set
    output.mkdir(parents=True,exist_ok=False);began=time.monotonic()
    source_archive=C.freeze(output)
    proof,_=validate(stock,**context)
    def certify(mesh):
        mesh.export(output/'candidate.obj',digits=17)
        mesh=C.trimesh.load(output/'candidate.obj',force='mesh',process=False)
        certificate,_=validate(mesh,**context,stock=stock)
        return mesh,certificate
    def checkpoint(mesh,certificate,history):
        mesh.export(output/'support.obj',digits=17)
        np.savez_compressed(output/'force_certificate.npz',**{'mask_'+str(k):v for k,v in certificate['masks'].items()})
        C.save(output/'progress.json',dict(history=history,material_cm3=abs(float(mesh.volume))*1e6,
            support_sha256=C.digest(output/'support.obj')))
    result,history,last=prune(stock,core,contact['records'],bands,calibration,context,certify,checkpoint,step=args.step)
    result,proof=certify(result)
    checkpoint(result,proof,history)
    spec=importlib.util.spec_from_file_location('rounded_front_renderer',C.COOPT/'step5.3/render.py')
    renderer_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer_module)
    renderer_module.reference(C.V.Renderer(),data,output/'overview.png',fixture=result)
    C.save(output/'report.json',dict(complete=True,passed=True,history=history,
        material_before_cm3=abs(float(stock.volume))*1e6,material_after_cm3=abs(float(result.volume))*1e6,
        physical_counts=proof['counts'],source=C.relative(source),source_sha256=C.digest(source),
        seconds=time.monotonic()-began,executed_sources=source_archive,
        connector_quality=connector_quality(result,calibration)[1]))
    inputs.verify()
