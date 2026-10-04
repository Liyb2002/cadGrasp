"""Try alternative constructor-owned collar depths; preserve all hard checks."""
import argparse
import numpy as np
import experiment as E
import recover as R
import refined_recovery as P
from step4_connect_support.fixture_view import cells_for
from step5_evaluate.evaluate import evaluate

# Use the ordinary exact-cell retention rather than repeated union retries.
E.AbsoluteGrow.connect=P.NORMAL_CONNECT
FACTOR=1.5

def root_cells(problem,contact):
    offsets,valid=E.G.vertex_offsets(problem.domain.mesh,1.)
    ids=np.unique(problem.domain.mesh.faces[np.unique(contact['source_faces'])])
    if not valid[ids].all():return None
    clearance=float(contact['triangles_m'][:,:,2].min())
    if clearance<.002-1e-9:return None
    depth=min(FACTOR*E.S.RELIEF,.5*clearance/np.linalg.norm(offsets[ids],axis=1).max())
    return cells_for(contact,problem.domain,depth*offsets)
E.root_cells=root_cells

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--groups',nargs='+',required=True);args=p.parse_args();E.install_recorded_recovery(E.OUT);lookup=dict(E.specs())
    for name in args.groups:
        passed=False;errors=[]
        for factor in [1.5,3.,4.,1.25]:
            FACTOR=factor
            try:
                E.run(name,lookup[name]);group=E.OUT/name;path=group/'step4/data/growing_support/report.json';report=E.json.loads(path.read_text())
                report['provenance']['code'].update(E.I.hashes([E.Path(__file__),E.Path(R.__file__),E.Path(P.__file__)]));report['constructor_root_depth_search']=dict(chosen_relief_multiplier=factor,root_is_constructor_owned=True,original_contact_surface_preserved=True,full_rod_diameter_unchanged=True,hard_acceptance_thresholds_unchanged=True);E.save(path,report);evaluate(group);passed=True;break
            except Exception as error:
                import traceback;traceback.print_exc();errors.append(dict(factor=factor,error=str(error)))
        if not passed:E.save(E.OUT/name/'comparison.json',dict(complete=True,passed=False,error=str(errors)))
