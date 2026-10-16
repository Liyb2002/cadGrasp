"""Update canonical Evaluation and the existing three Franka movies after Step5."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import os
import shutil
import subprocess
import sys
import time
from wrap_pipeline import ROOT,COOPT,read,save,sha,DEFAULT_EXPERIMENT


def main():
    experiment=sys.argv[1] if len(sys.argv)>1 else DEFAULT_EXPERIMENT
    numeric=COOPT/'data/results/B/data'/('step5_'+experiment)
    while not (numeric/'completion.json').exists():time.sleep(2)
    complete=read(numeric/'completion.json')
    if not complete['complete']:raise ValueError('New numerical results are not complete')
    idea=COOPT/'idea';assets=idea/'assets'
    work=assets/'data'/('wrap_support_movies_'+experiment)
    stage=work/'staging';stage_assets=stage/'assets';stage_data=stage_assets/'data'
    stage_data.mkdir(parents=True,exist_ok=True)
    old_selection=read(assets/'selection.json');labels=[r['id'] for r in old_selection['results']]
    if len(labels)!=3:raise ValueError('Expected the existing three Franka movies')
    allowed=set()
    original_selection=read(ROOT/read(numeric/'execution.json')['selection'])['results']
    for case in original_selection:
        allowed.add(str((COOPT/'output/B'/case['id']/'evaluation.png').relative_to(ROOT)))
        allowed.add(str((COOPT/'data/results/B'/case['id']/'evaluation.json').relative_to(ROOT)))
    allowed.update(str((idea/(label+'_franka.mp4')).relative_to(ROOT)) for label in labels)
    before={p:h for p,h in read(numeric/'protected_originals.json').items() if p in allowed}
    save(work/'authorization.json',dict(user_requested_existing_evaluation_and_three_franka_updates=True,
        only_after_new_results_complete=True,protected_originals_authorized_to_change=before))
    env=dict(os.environ,MUJOCO_GL='egl',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    def invoke(command,log):
        with Path(log).open('w') as handle:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
        if result.returncode:raise ValueError('Visual command failed: '+str(log))
    invoke([sys.executable,str(COOPT/'evaluation/run.py'),'B'],work/'evaluation.log')
    invoke([sys.executable,str(COOPT/'evaluation/verify_outputs.py'),'B'],work/'evaluation_verify.log')
    all_rows={r['id']:r for r in original_selection};selected=[]
    for previous in old_selection['results']:
        label=previous['id'];case=dict(all_rows[label]);case.update(previous)
        evaluation=read(COOPT/'data/results/B'/label/'evaluation.json')
        if not evaluation['passed'] or not evaluation['step53_source'].endswith('/'+experiment):
            raise ValueError('Canonical Evaluation did not select the new result: '+label)
        case.update(output=evaluation['step53_source'],
            unsupported_bbx_l=evaluation['object_static']['volume_litres'],
            supported_bbx_l=evaluation['with_full_support']['volume_litres'],
            bbx_increase_percent=evaluation['bbx_increase_percent']['with_full_support'])
        if case['bbx_increase_percent']>50:raise ValueError('Existing movie selection exceeds its BBX limit')
        selected.append(case)
    selection=dict(complete=True,results=selected,source=str((numeric/'summary.json').relative_to(ROOT)),
        support_experiment=experiment,playback_policy=old_selection['playback_policy'])
    selection_path=stage_assets/'selection.json';save(selection_path,selection)
    archive_name='executed_sources_wrap_'+experiment
    archive=stage_data/archive_name;archive.mkdir()
    sources={}
    for name in ['run.py','inputs.py','workflow.py','franka.py','grasps.py','presentation.py','force_arrows.py','review.py','finalize_movies.py']:
        target=archive/name;shutil.copy2(assets/name,target);sources[str(target.relative_to(ROOT))]=sha(target)
    save(stage_data/'source_manifest.json',dict(executed_sources=sources,support_experiment=experiment,
        authorized_prior_videos_and_previews_replacement=before,prior_videos_and_previews_removed_by_user=before,
        publication_pending=True))
    cpus=sorted(os.sched_getaffinity(0))
    def render(index,label):
        group=cpus[6+2*index:8+2*index] or cpus[-2:]
        prefix=['taskset','-c',','.join(map(str,group)),sys.executable]
        invoke(prefix+[str(assets/'run.py'),'--sets',label,'--result-selection',str(selection_path),
            '--output-dir',str(stage),'--expanded-grasps','--joint-space-transfer'],work/(label+'.render.log'))
        invoke(prefix+[str(assets/'review.py'),'--sets',label,'--result-selection',str(selection_path),
            '--output-dir',str(stage)],work/(label+'.review.log'))
        print('FRANKA_RENDERED_REVIEWED',label,flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        for result in as_completed([pool.submit(render,k,label) for k,label in enumerate(labels)]):result.result()
    reviews=[read(stage_data/label/'motion_review.json') for label in labels]
    save(stage_data/'motion_review.json',dict(cases=reviews,robot_FK_and_joint_limits_passed=True))
    invoke([sys.executable,str(assets/'finalize_movies.py'),'--output-dir',str(stage)],work/'staging_verify.log')
    replacements={}
    for label in labels:
        video=idea/(label+'_franka.mp4');preview=assets/(label+'_storyboard.png')
        for target in [video,preview]:replacements[str(target.relative_to(ROOT))]=dict(old_sha256=sha(target))
        shutil.copytree(stage_data/label,assets/'data'/label,dirs_exist_ok=True)
        (stage/(label+'_franka.mp4')).replace(video)
        shutil.copy2(stage_assets/(label+'_storyboard.png'),preview)
        for target in [video,preview]:replacements[str(target.relative_to(ROOT))]['new_sha256']=sha(target)
    shutil.copy2(selection_path,assets/'selection.json')
    shutil.copy2(stage_data/'motion_review.json',assets/'data/motion_review.json')
    final_archive=assets/'data'/archive_name;shutil.copytree(archive,final_archive)
    final_sources={str((final_archive/p.name).relative_to(ROOT)):sha(final_archive/p.name) for p in final_archive.glob('*.py')}
    save(work/'replacement.json',dict(complete=True,authorized_by_latest_user=True,replacements=replacements,
        support_experiment=experiment,no_old_mp4_backup_created=True))
    save(assets/'data/source_manifest.json',dict(executed_sources=final_sources,support_experiment=experiment,
        authorized_prior_videos_and_previews_replacement=before,prior_videos_and_previews_removed_by_user=before,
        current_replacement=str((work/'replacement.json').relative_to(ROOT)),publication_pending=False))
    invoke([sys.executable,str(assets/'finalize_movies.py')],work/'delivered_verify.log')
    originals=read(numeric/'protected_originals.json')
    changed={p:dict(before=h,after=sha(ROOT/p)) for p,h in originals.items() if sha(ROOT/p)!=h}
    if set(changed)-allowed:raise ValueError('Unexpected original changes: '+str(set(changed)-allowed))
    save(numeric/'authorized_visual_updates.json',dict(complete=True,
        user_requested_existing_evaluation_and_three_franka_updates=True,changes=changed))
    save(work/'completion.json',dict(complete=True,passed=True,existing_evaluation_images_updated=52,
        existing_franka_movies_updated=3,new_support_experiment=experiment,authorized_changes=changed,
        numerical_models_layouts_and_certificates_unchanged=True,
        video_verification=str((assets/'data/verification.json').relative_to(ROOT))))
    print('VISUALS COMPLETE',work,flush=True)

if __name__=='__main__':main()
