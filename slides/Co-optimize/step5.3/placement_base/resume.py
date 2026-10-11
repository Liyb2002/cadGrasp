"""Run remaining groups in separate checkpointed batches; retain earlier data."""
import os
import run as R

original_construction=R.construction_attempts


def continue_partial(data,tasks,body,trials,out):
    previous=R.C.COOPT/'output/B'/data['case']['id']/'step5/step5.3/placement_base_resume_v1/construction_attempts.json'
    if not previous.exists():return original_construction(data,tasks,body,trials,out)
    saved=R.C.json.loads(previous.read_text())
    remaining=[t for t in trials if t['label']==saved[-1]['proposal']]
    fresh,accepted,best=original_construction(data,tasks,body,remaining,out)
    R.C.save(out/'resumed_attempt_provenance.json',dict(
        previous_artifact=R.C.relative(previous),previous_sha256=R.C.digest(previous),
        reused_attempt_count=len(saved),fresh_attempt_count=len(fresh)))
    return saved+fresh,accepted,best

if __name__=='__main__':
    R.STAGE=os.environ.get('CADGRASP_PLACEMENT_STAGE','step5.3/placement_base_resume_v1')
    if R.STAGE.endswith('resume_v2'):R.construction_attempts=continue_partial
    R.main()
