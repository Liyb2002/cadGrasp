"""Neural batch with original constructor alternatives and full acceptance."""
import argparse
from pathlib import Path
import run as N
from step5_evaluate.evaluate import evaluate


def passed(name):
    group=N.OUT/name
    try:
        comparison=N.json.loads((group/'comparison.json').read_text())
        return bool(comparison.get('passed') and N.E.I.check_report(group/'step4/data/growing_support/report.json')['passed'])
    except (RuntimeError,FileNotFoundError,AssertionError):return False


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--groups',nargs='+');parser.add_argument('--plain',action='store_true');args=parser.parse_args()
    N.E.OUT=N.OUT;N.E.install_recorded_recovery(N.OUT);policy=N.Policy();specs=dict(N.E.specs())
    for name in args.groups or list(specs):
        if passed(name):continue
        failures=[]
        methods=[('plain',N.P.NORMAL_CONNECT)] if args.plain else [('carved',N.C.connect),('plain',N.P.NORMAL_CONNECT)]
        for mode,connect in methods:
            N.E.AbsoluteGrow.connect=connect
            try:
                N.run(policy,name,specs[name])
                group=N.OUT/name;path=group/'step4/data/growing_support/report.json'
                report=N.json.loads(path.read_text())
                report['neural_constructor_mode']=dict(mode=mode,neural_selection_unchanged=True,hard_tolerances_unchanged=True)
                report['provenance']['code'].update(N.E.I.hashes([Path(__file__)]))
                N.E.save(path,report);result=evaluate(group);N.V.render(group)
                break
            except Exception as error:
                import traceback;traceback.print_exc();failures.append(dict(mode=mode,error=str(error)))
        else:
            N.E.save(N.OUT/name/'comparison.json',dict(complete=True,passed=False,new_neural_network_used=True,constructor_failures=failures))
    rows=[]
    for name in specs:
        group=N.OUT/name;path=group/'comparison.json'
        if passed(name):
            comparison=N.json.loads(path.read_text());rows.append(dict(group=name,passed=True,volume_cm3=comparison['step5']['object_and_support_poses']['box_volume_cm3']))
        else:rows.append(dict(group=name,passed=False))
    N.E.save(N.OUT/'progress.json',dict(complete=True,results=rows))


if __name__=='__main__':main()
