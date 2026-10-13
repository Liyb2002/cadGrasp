"""Verify actual run inputs/artifacts; report historical code hashes separately."""
import _bootstrap
from co_common import *
from codes.precompute_objects.dataset import read_selected_pose_groups


def audit(names=('B','A1-f')):
    reports=[];errors=[];historical=[];checks=0
    for name in names:
        root=HERE/'output'/name if name=='B' else HERE/'data/object_inputs'/name
        for group in read_selected_pose_groups(name):
            base=root/group['id']
            candidates=[base/'step3/step3.1/data/report.json',base/'step3/step3.2/data/report.json',base/'step3/data/report.json',base/'step3/step3.3/data/report.json',base/'step4/step4.1/data/report.json']
            for path in candidates:
                if not path.exists():continue
                data=json.loads(path.read_text())
                if 'step4/step4.1' in str(path) and group['id'] not in data.get('initialization',{}).get('groups',{}):continue
                if not data.get('complete'):errors.append(dict(report=str(path.relative_to(ROOT)),error='report incomplete'))
                mismatches=[]
                for relative,expected in data.get('provenance',{}).get('inputs',{}).items():
                    source=ROOT/relative;checks+=1
                    if not source.exists() or I.sha256(source)!=expected:mismatches.append(dict(kind='input',path=relative))
                for relative,expected in data.get('artifacts',{}).items():
                    source=path.parent/relative;checks+=1
                    if not source.exists() or I.sha256(source)!=expected:mismatches.append(dict(kind='artifact',path=str(source.relative_to(ROOT))))
                for relative,expected in data.get('provenance',{}).get('code',{}).items():
                    source=ROOT/relative
                    if not source.exists() or I.sha256(source)!=expected:historical.append(dict(report=str(path.relative_to(ROOT)),source=relative,saved_sha256=expected,current_sha256=I.sha256(source) if source.exists() else None))
                if mismatches:errors.append(dict(report=str(path.relative_to(ROOT)),mismatches=mismatches))
                reports.append(str(path.relative_to(ROOT)))
    result=dict(complete=True,objects=list(names),reports_checked=len(reports),input_and_artifact_hash_checks=checks,input_and_artifact_hashes_passed=not errors,errors=errors,historical_code_hash_mismatches=historical,historical_code_policy='Saved code hashes remain unchanged; current source changed during numerical and render improvements. This audit verifies saved input and artifact bytes, not rerun equivalence.',reports=reports)
    save(HERE/'data/selected_to41_artifact_audit.json',result)
    print('ARTIFACT AUDIT',len(reports),'reports',checks,'input/artifact hashes','errors',len(errors),'historical code entries',len(historical),flush=True)
    return result


if __name__=='__main__':
    result=audit()
    if result['errors']:raise SystemExit(2)
