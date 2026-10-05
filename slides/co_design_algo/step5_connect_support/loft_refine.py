"""Delete redundant whole lofts only when the full original acceptance survives."""
import contextlib
import io

import numpy as np

from step5_connect_support import loft_growth as G, loft_acceptance as A, material_network as N
from step5_connect_support.loft_candidates import volume


def prune(case, context, graph, result):
    selected = result['selected'].copy(); rejected = set(); history = []
    while True:
        trials = []
        for node in np.flatnonzero(selected):
            if node in graph['heads'] or node in rejected: continue
            proposal = selected.copy(); proposal[node] = False
            if not N.connected_selection(proposal,graph['edges'],len(proposal)): continue
            coverage = [N.containment(N.floor_points(proposal,graph['floors'],k),case.demands[k]) for k in (0,1)]
            if not all(c['passed'] for c in coverage): continue
            try:
                candidate_volume=volume(G.selection_solid(graph,proposal))
            except RuntimeError as error:
                rejected.add(node)
                history.append(dict(removed_node=int(node),passed=False,reason=str(error)))
                continue
            trials.append((candidate_volume,int(node),proposal))
        accepted = False
        for candidate_volume, node, proposal in sorted(trials,key=lambda r:(r[0],r[1])):
            try:
                with contextlib.redirect_stdout(io.StringIO()): other=A.inspect(case,context,graph,proposal)
                passed=other['passed']
                reason=None if passed else 'original_loads'
            except RuntimeError as error:
                passed=False; reason=str(error)
            history.append(dict(removed_node=node, passed=passed, proposed_volume_cm3=candidate_volume,reason=reason))
            if passed:
                result=other;selected=proposal;accepted=True
                print('Deleted redundant loft',node,'volume',other['mesh'].volume*1e6,flush=True)
                break
            rejected.add(node)
        if not accepted: break
    return result,history
