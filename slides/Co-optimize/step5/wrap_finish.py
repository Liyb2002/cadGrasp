"""Audit actual outputs and publish this Step5 run without replacing old images."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from wrap_pipeline import ROOT,COOPT,read,save,sha,DEFAULT_EXPERIMENT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',default=DEFAULT_EXPERIMENT)
    parser.add_argument('--jobs',type=int,default=4)
    args=parser.parse_args()
    base=COOPT/'data/results/B';directory=base/'data'/('step5_'+args.experiment)
    summary=read(directory/'summary.json')
    if not summary['complete']:raise ValueError('Numerical batch still running')
    if (directory/'audit_scheduler.json').exists():
        for _ in range(30):
            if (directory/'audit_scheduler_complete.json').exists():break
            time.sleep(1)
    passed=[r for r in summary['results'] if r['passed']]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    def check(row):
        from wrap_audit_watch import valid_audit
        if valid_audit(ROOT/row['output']):
            return read(ROOT/row['output']/'data/independent_audit.json')
        p=directory/'payloads'/(row['id'].replace('/','__')+'.json')
        payload=read(p);payload.update(action='audit',cpus=sorted(os.sched_getaffinity(0))[-2:])
        target=p.with_name(p.stem+'.audit.json');save(target,payload)
        log=directory/'logs'/(row['id'].replace('/','__')+'.audit.log')
        with log.open('w') as handle:
            result=subprocess.run([sys.executable,str(COOPT/'step5/wrap_frozen_worker.py'),str(target)],
                cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
        if result.returncode:raise ValueError('Independent audit failed: '+row['id']+'; '+str(log))
        return read(ROOT/row['output']/'data/independent_audit.json')
    audits=[]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures=[pool.submit(check,row) for row in passed]
        for future in as_completed(futures):
            result=future.result();audits.append(result);print('AUDITED',result['id'],flush=True)
    save(directory/'actual_solid_audit.json',dict(complete=True,passed=True,results=audits,
        groups=len(audits),final_full_demands=sum(r['final_full_demands'] for r in audits),
        checkpoint_masks_checked=sum(r['checkpoint_masks_checked'] for r in audits)))
    from wrap_geometry import C
    import numpy as np
    import trimesh
    from PIL import Image,ImageDraw,ImageFont
    spec=importlib.util.spec_from_file_location('wrap_evaluation',COOPT/'evaluation/run.py')
    E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)
    from metrics import static_bounds
    body=trimesh.load(ROOT/'objects/B/mesh.stl',force='mesh',process=False)
    renderer=E.V.Renderer()
    tiles=[];published=[]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    selection=read(directory/'execution.json')['selection'];cases=read(ROOT/selection)['results']
    byid={c['id']:c for c in cases}
    for row in summary['results']:
        case=byid[row['id']]
        if row['passed']:
            out=ROOT/row['output'];report=read(out/'data/report.json');s4=ROOT/report['selected_step4_source']
            with np.load(s4/'layout.npz') as z:layout={k:z[k].copy() for k in z.files}
            poses=read(s4/'data/report.json')['poses']
            nominal=trimesh.load(s4/'support.obj',force='mesh',process=False)
            full=trimesh.load(out/'support.obj',force='mesh',process=False)
            metrics=static_bounds(body.vertices,layout,nominal.vertices,full.vertices,poses)
            metrics.update(id=case['id'],passed=True,status='PASS',support_sha256=sha(out/'support.obj'))
        else:
            poses=E.poses_for(case['id']);native=[]
            for pose in poses:
                with np.load(ROOT/'objects/B/poses'/pose/'setup.npz') as z:native.append(z['T_world_mesh'].copy())
            n=len(poses);layout=dict(native_world=np.asarray(native),hosts=np.arange(n),
                placements=np.repeat(np.eye(4)[None],n,axis=0),active=np.arange(n))
            nominal=full=None;metrics=static_bounds(body.vertices,layout,None,None,poses)
            metrics.update(id=case['id'],passed=False,status=row['status'])
            out=base/case['id']/'step5'/args.experiment
        image=COOPT/'output/B'/case['id']/'step5'/args.experiment/'overview.png'
        E.render(renderer,body,nominal,full,layout,metrics,image)
        save(out/'data/static_metrics.json',metrics)
        published.append(dict(id=case['id'],image=str(image.relative_to(ROOT)),sha256=sha(image),passed=row['passed']))
        with Image.open(image) as im:tile=im.convert('RGB').resize((390,242),Image.Resampling.LANCZOS)
        card=Image.new('RGB',(390,266),'white');card.paste(tile,(0,24));ink=ImageDraw.Draw(card)
        ink.text((8,5),case['id']+'  '+row['status'],font=font,fill='#296B81' if row['passed'] else '#B05C54')
        tiles.append(card);print('PREVIEW',case['id'],flush=True)
    gallery=Image.new('RGB',(390*4,266*((len(tiles)+3)//4)),'white')
    for i,tile in enumerate(tiles):gallery.paste(tile,((i%4)*390,(i//4)*266))
    gallery_path=COOPT/'output/B'/('step5_'+args.experiment+'.png');gallery.save(gallery_path)
    save(directory/'publication.json',dict(complete=True,images=published,gallery=str(gallery_path.relative_to(ROOT)),
        gallery_sha256=sha(gallery_path),old_evaluation_images_preserved=True))
    protected={p:h for row in passed for p,h in row['report']['provenance']['inputs'].items()}
    changed=[p for p,h in protected.items() if C.digest(ROOT/p)!=h]
    originals=read(directory/'protected_originals.json')
    authorized_path=directory/'authorized_visual_updates.json'
    authorized=read(authorized_path).get('changes',{}) if authorized_path.exists() else {}
    original_changed=[p for p,h in originals.items() if sha(ROOT/p)!=authorized.get(p,{}).get('after',h)]
    archives={Path(read(directory/'payloads'/(r['id'].replace('/','__')+'.json'))['source_snapshot']) for r in passed}
    indices={str(archive.relative_to(ROOT)):read(archive/'index.json') for archive in archives}
    frozen_changed=[str(archive/p) for archive in archives for p,h in read(archive/'index.json').items()
                    if sha(archive/p)!=h]
    for row in passed:
        payload=read(directory/'payloads'/(row['id'].replace('/','__')+'.json'))
        expected=read(Path(payload['source_snapshot'])/'index.json')
        if row['report'].get('execution_launcher_sha256'):
            if row['report']['execution_launcher_sha256']!=expected['slides/Co-optimize/step5/wrap_frozen_worker.py']:
                raise ValueError('Numerical launcher source mismatch: '+row['id'])
        for key,value in row['report']['executed_frozen_sources'].items():
            if expected.get(key)!=value:raise ValueError('Executed source binding mismatch: '+row['id']+' '+key)
    if changed or frozen_changed or original_changed:raise ValueError('Input/source verification failed: '+str((changed,frozen_changed,original_changed)))
    publication_code=C.freeze(directory/'publication_execution')
    verification=dict(complete=True,passed=True,protected_inputs=protected,frozen_sources=indices,
        publication_sources=publication_code,
        saved_models_audited=True,original_inputs_unchanged=True,frozen_sources_unchanged=True,
        protected_originals_count=len(originals),protected_originals_unchanged=not authorized,
        protected_originals_except_authorized_visual_updates_unchanged=True,
        authorized_visual_updates=authorized)
    save(directory/'verification.json',verification)
    totals=dict(initial_material_cm3=sum(r['report']['initial_material_cm3'] for r in passed),
        final_material_cm3=sum(r['report']['final_material_cm3'] for r in passed),
        initial_bbx_l=sum(r['report']['initial_bbx_l'] for r in passed),
        final_bbx_l=sum(r['report']['final_bbx_l'] for r in passed),
        accepted_deletions=sum(r['report']['accepted_deletions'] for r in passed))
    previous=read(directory/'completion.json') if (directory/'completion.json').exists() else {}
    media=previous.get('delivered_media',{});models=previous.get('delivered_model_sha256',{})
    media_valid=(bool(media) and all(sha(ROOT/p)==h for p,h in media.items())
        and all(models.get(row['id'])==row['report']['support_sha256'] for row in passed))
    save(directory/'completion.json',dict(previous,complete=True,all52_attempts_finished=summary['group_count']==52,
        group_count=summary['group_count'],certified=summary['passed'],step5_failed=summary['step5_failed'],
        upstream_step4_fail=summary['upstream_step4_fail'],totals=totals,
        final_full_demands=sum(r['final_full_demands'] for r in audits),
        video_complete=media_valid,protected_inputs=len(protected),frozen_files=sum(len(index) for index in indices.values())))
    if summary['group_count']==len(cases):
        save(base/'data/step5_current.json',dict(complete=True,experiment=args.experiment,
            summary=str((directory/'summary.json').relative_to(ROOT)),completion=str((directory/'completion.json').relative_to(ROOT))))
    print(json.dumps(totals),flush=True)

if __name__=='__main__':main()
