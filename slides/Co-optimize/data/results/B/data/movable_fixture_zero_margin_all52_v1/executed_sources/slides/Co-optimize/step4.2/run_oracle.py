"""Cold all-set Step4 search with base enclosure / common-connection gate."""
from pathlib import Path
import sys
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
COOPT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(COOPT/'helper_func'))
import _bootstrap
import argparse
import json
import time
import traceback
import shutil
import multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
from oracle_search import OracleModel,OracleSearch
from whole_pipeline import load_layout
from co_common import ROOT,I,S,save,trimesh
from whole_step4_render import render_search
from render_pose_oracle import export_pose_constraints
import artifact_store as A


def read_upstream(path):
    """Frozen upstream artifacts/physical inputs, not current search source."""
    report=json.loads(path.read_text());assert report['complete']
    I.check_hashes(report['provenance']['inputs'])
    for filename,digest in report.get('artifacts',{}).items():
        assert I.sha256(path.parent/filename)==digest,filename
    return report


def worker(job):
    label,options=job;base=A.result_root('B')/label
    out=base/'step4/step4.2'/options['output_name'];out.mkdir(parents=True,exist_ok=False);(out/'data').mkdir()
    began=time.monotonic();result=dict(id=label,passed=False,force_passed=False,base_oracle_passed=False)
    try:
        source=base/'step4/step4.1';original=read_upstream(source/'data/report.json')
        initialized=base/'step3/step3.1';read_upstream(initialized/'data/report.json')
        model=OracleModel(original['poses'],'B',demand_nodes=32)
        seed=initialized/'wrapped_support.obj'
        model.registered_seed=S.solid(trimesh.load(seed,force='mesh',process=False))
        model.inputs += [seed,initialized/'data/report.json',source/'layout.npz']
        search=OracleSearch(model,out,iterations=options['iterations'],finalists=3,branch_rounds=2,seed=42)
        search.screen_budget=96;search.capture_process=True;search.volume_rounds=2
        layout=load_layout(source/'layout.npz')
        current=search.initialize_exact(layout,'saved_step4.1_registered_initial');search.baseline(current)
        initial=current['counts']
        if options.get('resume'):
            current=model.evaluate(load_layout(base/'step4/step4.2'/options['resume']/'layout.npz'))
            model.commit(current)
        current=search.solve_frontier(current)
        if current['base_oracle']['passed']:
            current=search.restore_reuse(current);current=search.polish_volume(current,1)
        report=search.save(current,'whole_gradient_base_oracle',began,dict(source='saved Step4.1',
            old_answers_used_as_start=False,resume_own_experiment=options.get('resume')),initial_counts=initial)
        data=dict(report);data['artifacts']={'../'+k:v for k,v in report['artifacts'].items()}
        save(out/'data/report.json',data)
        if report['mesh_exported']:
            try:render_search(model,out,report)
            except Exception as error:report['render_error']=str(error)
        result.update(passed=report['passed'],force_passed=report['force_passed'],
            base_oracle_passed=report['base_oracle_passed'],pose_count=len(original['poses']),
            volume_cm3=report['volume_cm3'],counts=report['counts'],
            oracle_status=report['base_oracle']['status'],oracle_seconds=model.base_oracle.seconds,
            oracle_queries=model.base_oracle.queries,component_queries=model.base_oracle.component_queries,
            maximum_object_height_m=max(model.workpiece_height(current['layout'],k) for k in current['layout'].active),
            state_count=len(set(current['layout'].hosts.tolist())),experiment=options['output_name'])
        save(out/'report.json',report)
        if report['base_oracle_passed']:
            # Keep the final task-owned cone equations and display excerpts.
            # Videos are generated on request; no search/force replay here.
            oracle_out=A.VISUALS/'B'/label/'step4/step4.2'/options['output_name']/'oracle'
            export_pose_constraints(model,current['layout'],oracle_out,model.base_oracle)
    except Exception as error:
        traceback.print_exc();result.update(status='unresolved',error=str(error))
    A.publish_visuals(out)
    A.cleanup_stage(out)
    result['seconds']=time.monotonic()-began;save(out/'run_result.json',result)
    print('ORACLE RESULT',json.dumps(result),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sets',nargs='+',default=['all']);parser.add_argument('--jobs',type=int,default=3)
    parser.add_argument('--iterations',type=int,default=6);parser.add_argument('--resume')
    parser.add_argument('--output-name',default='base_oracle_v1');args=parser.parse_args()
    base=A.result_root('B')
    sets=args.sets
    if sets==['all']:
        sets=[str(p.relative_to(base)).split('/step4/')[0] for p in base.glob('*/step4/step4.1/data/report.json')]
        sets += [str(p.relative_to(base)).split('/step4/')[0] for p in (base/'illegal').glob('*/step4/step4.1/data/report.json')]
        sets.sort(key=lambda label:(-len(json.loads((base/label/'step4/step4.1/data/report.json').read_text())['poses']),label))
    directory=base/'data'/args.output_name;directory.mkdir(parents=True,exist_ok=False)
    sources=list((COOPT/'helper_func').rglob('*.py'))+list((COOPT/'vis_func').glob('*.py'))+list((COOPT/'step4.2').glob('*.py'))
    hashes=I.hashes(sources)
    for relative in hashes:
        target=directory/'executed_sources'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,target)
    save(directory/'sources.json',hashes)
    protected={}
    for label in sets:
        for stage in ['step3','step4/step4.1']:
            protected.update(I.hashes([p for p in (base/label/stage).rglob('*') if p.is_file()]))
    save(directory/'protected.json',protected)
    batch=dict(complete=False,sets=sets,group_count=len(sets),options=vars(args),results=[],
        old_incumbents_accepted_without_oracle=False,intermediate_rotations_constrained=False)
    save(directory/'batch.json',batch);start=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.jobs,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(worker,(label,vars(args))) for label in sets]
        for future in as_completed(futures):
            batch['results'].append(future.result());batch['seconds']=time.monotonic()-start
            batch['passed']=sum(r['passed'] for r in batch['results']);save(directory/'batch.json',batch)
    I.check_hashes(protected);I.check_hashes(hashes)
    batch.update(complete=True,protected_unchanged=True,executed_sources_unchanged=True,seconds=time.monotonic()-start)
    save(directory/'batch.json',batch);A.finish_batch(directory)
    for label in sets:
        (base/label/'step4/step4.2'/args.output_name/'run_result.json').unlink(missing_ok=True)
    print('ALL ORACLE',batch['passed'],'/',len(sets),flush=True)


if __name__=='__main__':main()
