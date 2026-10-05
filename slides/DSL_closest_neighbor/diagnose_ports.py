"""Read a completed head trial and diagnose support ports without changing it."""
import argparse
from dataclasses import replace
import json
import numpy as np
from pathlib import Path
import merge_release as M

class DiagnosticGrow(M.PhysicalGrow):
    def verify(self, mesh):
        try:
            return super().verify(mesh)
        except Exception as e:
            if hasattr(e, 'checks'):
                M.I.save(self.out/'geometry_rejection.json', dict(error=str(e), checks=e.checks))
            raise

    def roots(self):
        rows=[]
        for h in self.physical_heads:
            w=self.port_witnesses.get(h.ident)
            if not w:
                continue
            seed=h.solid+w['extra']+w['ball']
            rows.append(dict(head=h.ident,ball_legal=self.legal(w['ball']),
                ball_cut_overlap=M.volume(w['ball']^self.removed),
                extra_outside_mask=M.volume(w['extra']-self.mask),
                extra_cut_overlap=M.volume(w['extra']^self.removed),
                seed_cut_overlap=M.volume(seed^self.removed),
                positive_components=len(M.nonempty_parts(seed))))
        M.I.save(self.out/'port_diagnostics.json',rows)
        print('PORT DIAGNOSTICS',json.dumps(rows),flush=True)
        return super().roots()

def main():
    p=argparse.ArgumentParser();p.add_argument('group');p.add_argument('--source-stage',default='global_release_v6');a=p.parse_args()
    base=M.HERE/'output/B'/a.group/'step3_scheculer'/a.source_stage
    r=json.loads((base/'report.json').read_text());cfg=r['config'].copy();cfg['stage']='port_diagnostic'
    solver=M.Solver(a.group,cfg);solver.initialize();solver.initial_build_indices=solver.initial_sets['selected_indices']
    initial,_=solver.construct(solver.heads,solver.state,solver.sets,'initial')
    if initial is None:raise RuntimeError('Initial full construction unavailable')
    grow=initial[3];solver.port_witnesses={}
    for terminal,h in zip(grow.terminals[:len(solver.heads)],solver.heads):
        solver.port_witnesses[h.ident]=dict(extra=terminal['solid']-h.solid,
            ball=next(c for k,c in grow.core_solids if k==h.ident+':start_ball'),
            start=np.array(next(x['start_m'] for x in grow.thickness if x['name']==h.ident)))
    meta=json.loads((base/'final/state.json').read_text());original={h.ident:h for h in solver.heads}
    heads=[]
    for x in meta['physical_heads']:
        cuts=tuple((np.array(c['center_fixture_m']),c['halfwidth_m']) for c in x['cutters'])
        heads.append(M.cut_head(original[x['id']],cuts))
    groups=[]
    for task,row in zip(solver.tasks,solver.state.groups):
        depths={c['candidate_id']:c['head_depth_m'] for c in row}
        contacts=M.I.read_contacts(base/'final'/f'contacts_{task.pose}.npz')
        for c in contacts:c['head_depth_m']=depths[c['candidate_id']]
        groups.append(tuple(contacts))
    solver.heads=heads;solver.state=replace(solver.state,groups=tuple(groups))
    solver.directions=np.array(r['final_head_exits']['directions_world_xyz']);solver.sweeps.clear()
    solver.sets=solver.readout(heads)
    old=M.PhysicalGrow;M.PhysicalGrow=DiagnosticGrow
    try:
        _,errors=solver.construct(heads,solver.state,solver.sets,'final')
        print('ERRORS',[e['error'] for e in errors])
    finally:M.PhysicalGrow=old

if __name__=='__main__':main()
