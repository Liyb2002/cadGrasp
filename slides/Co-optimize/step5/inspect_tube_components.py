"""Test whether a disconnected proposal contains a complete useful component."""
from pathlib import Path
import json,argparse,importlib.util
from feasible_geometry import C,np,Layout,ForceCertificate,validate
from tube_support import ground_ports,path_tube
from rounded_front import round_ports
from continuous_legs import bounded_domain
from deterministic_base import solid,unpack,union,volume
from prune_support import material_components


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('set');p.add_argument('proposal',type=Path)
    args=p.parse_args()
    inputs=C.Inputs('B',C.COOPT/'data/results/B/data/step5_show_all52_20261014_02_inputs/selection.json',
        'pso_all52_20261013_01_floor')
    data=inputs.load(next(c for c in inputs.cases if c['id']==args.set))
    with np.load(data['selected']/'layout.npz')as a:layout=Layout.from_arrays(a)
    checker=ForceCertificate(data['record']['poses'],layout,data['body'],data['rows'],'B')
    source=C.COOPT/'data/results/B'/args.set/'step5/step5.3/step5_show_all52_20261014_02/support_with_base.obj'
    stock=C.trimesh.load(source,force='mesh',process=False)
    witness=C.trimesh.load(data['selected']/'oracle_connected_witness.obj',force='mesh',process=False)
    contacts=json.loads((C.COOPT/'data/results/B/data/contact_subset_cache_20261015'/args.set/'report.json').read_text())
    proposal=json.loads(args.proposal.read_text());paths=[np.asarray(p)for p in proposal['paths']]
    ports=[p for r in contacts['records']for p in r['patches']]+ground_ports(stock,data['rows'])[0]
    bounds=C.occupied_bounds(data['body'],stock,data['rows'])
    domain=bounded_domain(witness,data['body'],data['rows'],bounds)
    guide=round_ports(ports,.0035)+union([path_tube(p,.0025)for p in paths])
    parts=material_components(domain^guide)
    output=C.COOPT/'data/results/B/data/rounded_tube_component_probe_20261015_01'/args.set
    output.mkdir(parents=True,exist_ok=False);C.freeze(output)
    for i,part in enumerate(parts):unpack(part).export(output/f'component_{i}.obj',digits=17)
    mesh=C.trimesh.load(output/'component_0.obj',force='mesh',process=False)
    try:
        proof,checks=validate(mesh,witness,data['body'],layout.placements[list(layout.active)],data['rows'],checker,bounds=bounds)
        result=dict(passed=True,checks=checks,counts=proof['counts'])
        spec=importlib.util.spec_from_file_location('component_renderer',C.COOPT/'step5.3/render.py')
        renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
        renderer.reference(C.V.Renderer(),data,output/'overview.png',fixture=mesh)
    except Exception as error:result=dict(passed=False,error=str(error))
    result.update(component_volumes_cm3=[volume(p)*1e6 for p in parts])
    C.save(output/'report.json',result);print(result,flush=True)


if __name__=='__main__':main()
