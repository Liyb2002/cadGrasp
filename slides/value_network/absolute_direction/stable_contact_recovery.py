"""Retry finite construction failures with alternate contacts, keeping all tests."""
import argparse
import re
import numpy as np
import experiment as E
import recover as R
import refined_recovery as P
from step5_evaluate.evaluate import evaluate

NORMAL_CONNECT=P.NORMAL_CONNECT
NORMAL_CANDIDATES=E.candidates
BANNED=set()

def connect(self):
    try:return NORMAL_CONNECT(self)
    except RuntimeError as error:
        if str(error)!='Contact-cell repair disconnected':raise
        bad=[]
        for row,contacts,b,o in zip(self.case.support_seeds,self.case.groups,self.bases,self.offsets):
            for cells,contact in zip(row,contacts):
                missing=max(abs(float((E.S.solid(E.G.hull_mesh(v@b+o))-self.final_solid).volume()))*E.S.SCALE**3 for v in cells)
                if missing>8e-14:bad.append(contact['candidate_id'])
        if not bad:raise
        raise RuntimeError('Unstable constructor contact collars: '+','.join(bad))

def candidates(problem,exit_vector):
    entries,path=NORMAL_CANDIDATES(problem,exit_vector)
    return [e for e in entries if e['contact']['candidate_id'] not in BANNED],path
E.AbsoluteGrow.connect=connect;E.candidates=candidates

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--groups',nargs='+',required=True);args=p.parse_args();E.install_recorded_recovery(E.OUT);lookup=dict(E.specs())
    for name in args.groups:
        BANNED.clear();passed=False;errors=[]
        for attempt in range(8):
            try:
                E.run(name,lookup[name]);group=E.OUT/name;path=group/'step4/data/growing_support/report.json';report=E.json.loads(path.read_text())
                report['provenance']['code'].update(E.I.hashes([E.Path(__file__),E.Path(R.__file__),E.Path(P.__file__)]));report['construction_contact_search']=dict(rejected_constructor_contacts=sorted(BANNED),attempts=attempt+1,step2_physical_impossibility_claim=False,hard_tolerances_unchanged=True);E.save(path,report);evaluate(group);passed=True;break
            except Exception as error:
                import traceback;traceback.print_exc();errors.append(dict(attempt=attempt,error=str(error)))
                ids=re.findall(r'pose_\d+_C\d+',str(error)) if 'Unstable constructor contact collars:' in str(error) or 'No legal original contact growth start:' in str(error) else []
                if not ids:break
                BANNED.update(ids);print('RESELECT CONSTRUCTIBLE CONTACTS',name,sorted(BANNED),flush=True)
        if not passed:E.save(E.OUT/name/'comparison.json',dict(complete=True,passed=False,error=str(errors)))
