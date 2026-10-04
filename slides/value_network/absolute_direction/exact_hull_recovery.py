"""Preserve tiny collar facets with the existing normalized float64 hull builder."""
import argparse
import shutil
import experiment as E
import recover as R
import refined_recovery as P
from step5_evaluate.evaluate import evaluate

# The Step2 trajectory hull normalizes coordinates and keeps input vertices.
# Generic trimesh process=True merges vertices at metre-scale rounding. Use the
# existing exact builder consistently for construction AND mandatory-cell checks.
E.G.hull_mesh=E.insertion.engine.hull_mesh
E.AbsoluteGrow.connect=P.NORMAL_CONNECT

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--groups',nargs='+',required=True);args=p.parse_args();E.install_recorded_recovery(E.OUT);lookup=dict(E.specs())
    for name in args.groups:
        group=E.OUT/name;comparison=group/'comparison.json';previous=E.json.loads(comparison.read_text()) if comparison.exists() else dict(passed=False)
        if previous.get('passed'):
            try:E.I.check_report(group/'step4/data/growing_support/report.json')
            except RuntimeError:previous=dict(passed=False)
        archive=group/'history/before_exact_hulls';saved=['step3','step4/data/growing_support','step4/data/source_inputs','step5_evaluate','comparison.json']
        if previous.get('passed'):
            if archive.exists():shutil.rmtree(archive)
            for rel in saved:
                src=group/rel;dst=archive/rel
                if src.is_dir():shutil.copytree(src,dst)
                elif src.exists():dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        try:
            E.run(name,lookup[name]);path=group/'step4/data/growing_support/report.json';report=E.json.loads(path.read_text())
            report['provenance']['code'].update(E.I.hashes([E.Path(__file__),E.Path(R.__file__),E.Path(P.__file__),E.Path(E.insertion.engine.__file__)]))
            report['exact_hull_policy']=dict(builder='existing Step2 normalized float64 hull',vertex_rounding_or_merging=False,hard_checks_and_volume_tolerances_unchanged=True);E.save(path,report);evaluate(group)
        except Exception as error:
            import traceback;traceback.print_exc();E.save(comparison,dict(complete=True,passed=False,error=str(error)))
        current=E.json.loads(comparison.read_text())
        if previous.get('passed') and (not current.get('passed') or current['step5']['object_and_support_poses']['box_volume_cm3']>=previous['step5']['object_and_support_poses']['box_volume_cm3']):
            for rel in saved:
                src=archive/rel;dst=group/rel
                if src.is_dir():shutil.copytree(src,dst,dirs_exist_ok=True)
                elif src.exists():shutil.copy2(src,dst)
            print('KEPT BETTER INCUMBENT',name,flush=True)
