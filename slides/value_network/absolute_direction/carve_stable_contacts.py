"""Combine alternate constructible contacts and strict original-sweep carving."""
import argparse
import re
import experiment as E
import stable_contact_recovery as S
import raw_sweep_carving as C
import recover as R
import refined_recovery as P
from step5_evaluate.evaluate import evaluate

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--groups',nargs='+',required=True);args=p.parse_args();E.install_recorded_recovery(E.OUT);lookup=dict(E.specs())
    for name in args.groups:
        group=E.OUT/name;comparison=group/'comparison.json';previous=E.json.loads(comparison.read_text()) if comparison.exists() else {}
        S.BANNED.clear();S.BANNED.update(re.findall(r'No legal original contact growth start: pose_\d+:(pose_\d+_C\d+)',previous.get('error','')))
        passed=False;errors=[]
        for attempt in range(8):
            try:
                E.run(name,lookup[name]);path=group/'step4/data/growing_support/report.json';report=E.json.loads(path.read_text())
                report['provenance']['code'].update(E.I.hashes([E.Path(__file__),E.Path(S.__file__),E.Path(C.__file__),E.Path(R.__file__),E.Path(P.__file__)]))
                report['construction_contact_search']=dict(rejected_constructor_contacts=sorted(S.BANNED),hard_tolerances_unchanged=True);E.save(path,report);evaluate(group);passed=True;break
            except Exception as error:
                import traceback;traceback.print_exc();errors.append(dict(attempt=attempt,error=str(error)))
                ids=re.findall(r'No legal original contact growth start: pose_\d+:(pose_\d+_C\d+)',str(error))
                if not ids:break
                S.BANNED.update(ids)
        if not passed:E.save(comparison,dict(complete=True,passed=False,error=str(errors)))
