"""Additional legal contact-start positions and exact-cell Boolean conditioning."""
import argparse
import numpy as np
import experiment as E
import recover as R
from step5_evaluate.evaluate import evaluate

NORMAL_ATTACH=E.AbsoluteGrow.attach

def attach(self,original,points,name):
    try:return NORMAL_ATTACH(self,original,points,name)
    except RuntimeError:
        if ':foot' in name:raise
        pose,ident=name.split(':',1);k=self.case.poses.index(pose)
        contact=next(c for c in self.case.groups[k] if c['candidate_id']==ident)
        center=contact['center_m'].copy();face=contact['center_face']
        triangles=contact['triangles_m'];heights=triangles.mean(1)[:,2]
        try:
            # Reposition the transition start on actual selected contact facets.
            for i in np.argsort(-heights)[:16]:
                contact['center_m']=triangles[i].mean(0);contact['center_face']=int(contact['source_faces'][i])
                try:return NORMAL_ATTACH(self,original,points,name)
                except RuntimeError:continue
        finally:contact['center_m']=center;contact['center_face']=face
        raise RuntimeError('No legal original contact growth start: '+name)

NORMAL_CONNECT=E.AbsoluteGrow.connect

def connect(self):
    try:return NORMAL_CONNECT(self)
    except RuntimeError as error:
        if str(error)!='Contact-cell repair disconnected':raise
        expected=[E.S.solid(E.G.hull_mesh(v@b+o)) for row,b,o in zip(self.case.support_seeds,self.bases,self.offsets) for cells in row for v in cells]
        parts=[t['solid'] for t in self.terminals]+list(self.beams)
        attempts=[]
        orders=[parts,list(reversed(parts)),sorted(parts,key=lambda part:float(part.volume()))]
        for order in orders:
            full=order[0]
            for part in order[1:]:full=full+part
            for repair in range(3):
                missing=[(abs(float((cell-full).volume()))*E.S.SCALE**3,i) for i,cell in enumerate(expected)]
                bad=[(v,i) for v,i in missing if v>8e-14]
                if not bad and len(full.decompose())==1:
                    self.final_solid=full;self.save_stage('shared_tree',full)
                    return full,dict(terminal_count=len(self.terminals),grown_segment_count=len(self.beams),
                        direct_connection_count=sum(len(p)==2 for p in self.paths),fallback_route_count=sum(len(p)>2 for p in self.paths),
                        growth_journal=self.journal,partial_stage_step_count=self.partial_steps,initial_head_name='All original contact starts',
                        exact_boolean_union_conditioning=True,mandatory_cells_preserved=True,hard_tolerances_unchanged=True)
                for _,i in sorted(bad,reverse=True):full=full+expected[i]
                # Reinsert complete existing legal transitions to retain bridges.
                for terminal in self.terminals:full=full+terminal['solid']
            attempts.append(dict(components=len(full.decompose()),maximum_missing_cell_m3=max(v for v,_ in missing)))
        raise RuntimeError('Strict exact-terminal union retries failed: '+str(attempts))
E.AbsoluteGrow.connect=connect

E.AbsoluteGrow.attach=attach

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--groups',nargs='+',required=True);args=p.parse_args()
    E.install_recorded_recovery(E.OUT);lookup=dict(E.specs())
    for name in args.groups:
        try:
            E.run(name,lookup[name]);group=E.OUT/name;path=group/'step4/data/growing_support/report.json';report=E.json.loads(path.read_text())
            report['provenance']['code'].update(E.I.hashes([E.Path(__file__),E.Path(R.__file__)]));report['direction_objective']['contact_start_search']='selected-facet centroids; same complete 5 mm cores and all strict constraints';E.save(path,report);evaluate(group)
        except Exception as error:
            import traceback;traceback.print_exc();E.save(E.OUT/name/'comparison.json',dict(complete=True,passed=False,error=str(error)))
