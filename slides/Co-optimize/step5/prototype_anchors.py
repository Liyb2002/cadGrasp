"""Validate an anchored round support, including actual full-demand masks."""
from pathlib import Path
import argparse,json,time,importlib.util
from feasible_geometry import C,np,Layout,ForceCertificate,validate
from reaction_ports import select_ports
from anchor_network import build
from exact_mesh import save_mesh


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('set')
    parser.add_argument('--name',default='anchored_round_probe_20261015_01')
    args=parser.parse_args()
    inputs=C.Inputs('B',C.COOPT/'data/results/B/data/step5_show_all52_20261014_02_inputs/selection.json','pso_all52_20261013_01_floor')
    data=inputs.load(next(c for c in inputs.cases if c['id']==args.set))
    with np.load(data['selected']/'layout.npz')as a:layout=Layout.from_arrays(a)
    checker=ForceCertificate(data['record']['poses'],layout,data['body'],data['rows'],'B')
    source=C.COOPT/'data/results/B'/args.set/'step5/step5.3/step5_show_all52_20261014_02/support_with_base.obj'
    stock=C.trimesh.load(source,force='mesh',process=False)
    witness=C.trimesh.load(data['selected']/'oracle_connected_witness.obj',force='mesh',process=False)
    face_cache=C.COOPT/'data/results/B/data/contact_subset_cache_20261015'/args.set
    faces=json.loads((face_cache/'report.json').read_text())
    calibration=json.loads((C.COOPT/'data/results/B/data/step5_feasible_outside_in_calibration_20261015_03/thickness_calibration.json').read_text())
    output=C.COOPT/'data/results/B/data'/args.name/args.set;output.mkdir(parents=True,exist_ok=False)
    code=C.freeze(output);began=time.monotonic()
    points,point_records=select_ports(checker,stock,faces['records'],face_cache/'reaction_ports')
    context=dict(witness=witness,body=data['body'],placements=layout.placements[list(layout.active)],rows=data['rows'],
        checker=checker,bounds=C.occupied_bounds(data['body'],stock,data['rows']))
    last={'mesh':None}
    def certify(mesh):
        mesh=save_mesh(mesh,output/'candidate.obj');proof,_=validate(mesh,**context,stock=last['mesh'])
        return mesh,proof
    def checkpoint(mesh,proof,history):
        save_mesh(mesh,output/'support.obj');last['mesh']=mesh.copy()
        np.savez_compressed(output/'force_certificate.npz',**{'mask_'+str(k):v for k,v in proof['masks'].items()})
        C.save(output/'progress.json',dict(physical_complete=True,counts=proof['counts'],history=history))
    try:full,proof,record,minimum=build(stock,witness,data['body'],data['rows'],context['bounds'],points,calibration,certify,checkpoint)
    except Exception as error:
        import traceback
        C.save(output/'failure.json',dict(error=str(error),traceback=traceback.format_exc()));raise
    full=save_mesh(full,output/'support.obj');proof,checks=validate(full,**context,mandatory=minimum)
    spec=importlib.util.spec_from_file_location('anchor_renderer',C.COOPT/'step5.3/render.py')
    renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
    renderer.reference(C.V.Renderer(),data,output/'overview.png',fixture=full)
    C.save(output/'report.json',dict(complete=True,passed=True,checks=checks,counts=proof['counts'],construction=record,
        material_before_cm3=abs(float(stock.volume))*1e6,material_after_cm3=abs(float(full.volume))*1e6,
        selected_reaction_locations=point_records,seconds=time.monotonic()-began,executed_sources=code))


if __name__=='__main__':main()
