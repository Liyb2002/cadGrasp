"""Exercise tube construction with resolved terminal routing on one set."""
from pathlib import Path
import argparse,json,time,importlib.util
from feasible_geometry import C,np,Layout,ForceCertificate,validate
import tube_support as T
from tube_routing import TerminalRouteGuide,full_width_middle
T.RouteGuide=TerminalRouteGuide;T.full_width_middle=full_width_middle


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('set')
    parser.add_argument('--name',default='rounded_tube_probe_20261015_02')
    parser.add_argument('--minimum-radius',type=float,default=.0015)
    parser.add_argument('--resume-plan',type=Path)
    args=parser.parse_args()
    inputs=C.Inputs('B',C.COOPT/'data/results/B/data/step5_show_all52_20261014_02_inputs/selection.json',
        'pso_all52_20261013_01_floor')
    case=next(c for c in inputs.cases if c['id']==args.set);data=inputs.load(case)
    with np.load(data['selected']/'layout.npz')as arrays:layout=Layout.from_arrays(arrays)
    checker=ForceCertificate(data['record']['poses'],layout,data['body'],data['rows'],'B')
    source=C.COOPT/'data/results/B'/args.set/'step5/step5.3/step5_show_all52_20261014_02/support_with_base.obj'
    stock=C.trimesh.load(source,force='mesh',process=False)
    witness=C.trimesh.load(data['selected']/'oracle_connected_witness.obj',force='mesh',process=False)
    contact=json.loads((C.COOPT/'data/results/B/data/contact_subset_cache_20261015'/args.set/'report.json').read_text())
    calibration=json.loads((C.COOPT/'data/results/B/data/step5_feasible_outside_in_calibration_20261015_03/thickness_calibration.json').read_text())
    context=dict(witness=witness,body=data['body'],placements=layout.placements[list(layout.active)],rows=data['rows'],
        checker=checker,bounds=C.occupied_bounds(data['body'],stock,data['rows']))
    output=C.COOPT/'data/results/B/data'/args.name/args.set;output.mkdir(parents=True,exist_ok=False)
    began=time.monotonic();code=C.freeze(output)
    def certify(mesh):
        from exact_mesh import save_mesh
        mesh=save_mesh(mesh,output/'candidate.obj')
        proof,_=validate(mesh,**context)
        return mesh,proof
    resume=json.loads(args.resume_plan.read_text())if args.resume_plan else {}
    if resume and resume['minimum_radius_m']!=args.minimum_radius:raise ValueError('Resume radius differs')
    def proposed(paths,history,radius):
        C.save(output/'connection_proposals.json',dict(physical_complete=False,
            paths=[p.tolist()for p in paths],history=history,minimum_radius_m=radius,
            source_sha256=C.digest(source),set=args.set))
    try:
        full,proof,record=T.build(stock,witness,data['body'],data['rows'],context['bounds'],
            contact['records'],calibration,certify,minimum_radius=args.minimum_radius,
            proposal_checkpoint=proposed,resume_paths=resume.get('paths'),resume_history=resume.get('history'))
    except Exception as error:
        import traceback
        C.save(output/'failure.json',dict(passed=False,error=str(error),traceback=traceback.format_exc()))
        raise
    from exact_mesh import save_mesh
    save_mesh(full,output/'support.obj')
    np.savez_compressed(output/'force_certificate.npz',**{'mask_'+str(k):v for k,v in proof['masks'].items()})
    spec=importlib.util.spec_from_file_location('tube_renderer',C.COOPT/'step5.3/render.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.reference(C.V.Renderer(),data,output/'overview.png',fixture=full)
    C.save(output/'report.json',dict(complete=True,passed=True,construction=record,counts=proof['counts'],
        material_before_cm3=abs(float(stock.volume))*1e6,material_after_cm3=abs(float(full.volume))*1e6,
        seconds=time.monotonic()-began,executed_sources=code))


if __name__=='__main__':main()
