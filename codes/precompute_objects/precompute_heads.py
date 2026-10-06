"""Build Step2 once for each native pose, including reusable geometry and exits."""
import argparse,gzip,json,os,sys,time,tempfile,shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'slides/baseline_algo'))
import numpy as np
from codes.precompute_objects import head_cache as C
from codes.precompute_objects.head_directions import catalogue
from codes.precompute_objects.head_geometry import PairGeometry
from step2_local_support import withdrawal as W,render as R
from step3_scheculer.pair_tasks import read_task
from PIL import Image,ImageDraw

def picture(task,geometry,pools,path):
    size=460;sheet=Image.new('RGB',(size*3,1040),'white');d=ImageDraw.Draw(sheet)
    for column,fraction in enumerate(C.AREAS):
        rows=pools[fraction];valid=[e for e in rows if e['valid']]
        d.text((column*size+16,12),f'{task.domain.data["object"]} / {task.pose} | area {fraction*100:g}%',font=R.font(18),fill=R.INK)
        d.text((column*size+16,40),f'{len(valid)} / {len(rows)} candidates with exit witnesses',font=R.font(15),fill=R.INK)
        for j,view in enumerate(([1,-1,.8],[-1,1,.8])):
            basis=R.axes(view);focus,width=R.overall_camera(task.domain,basis)
            triangles=task.domain.mesh.triangles.copy();colors=np.tile(R.GREY,(len(triangles),1));colors[task.domain.work_ids]=R.ORANGE
            chosen=valid[:8]
            if chosen:
                patches=np.concatenate([e['contact']['triangles_m'] for e in chosen]);start=len(triangles)
                triangles=np.concatenate([triangles,patches]);colors=np.concatenate([colors,np.tile([74,173,136],(len(patches),1))])
                overlay=np.arange(start,len(triangles))
            else:overlay=None
            im,depth=R.raster(triangles,colors,focus,basis,width,size,overlay=overlay)
            draw=ImageDraw.Draw(im)
            p=R.project(geometry.centers,focus,basis,width,size)
            origins=geometry.centers+basis[2]*geometry.scale*1e-7
            visible=~task.domain.mesh.ray.intersects_any(origins,np.broadcast_to(basis[2],origins.shape))
            for e,point,seen in zip(rows,p,visible):
                x,y=point[:2];color=(24,119,89) if e['valid'] else (199,61,61)
                if 0<=x<size and 0<=y<size and point[2]>=((task.domain.mesh.vertices-focus)@basis.T)[:,2].min():
                    # Only mark centers facing this camera; actual patches remain depth tested.
                    if seen and task.domain.mesh.face_normals[e['contact']['center_face']]@basis[2]>0:
                        draw.ellipse((x-2,y-2,x+2,y+2),fill=color)
            sheet.paste(im,(column*size,72+j*size))
    d.text((16,1000),'Orange: work region | Green / red: accepted / rejected centers | Patches: first 8 accepted heads',font=R.font(15),fill=R.INK)
    sheet.save(path)

def build(item,count=200,resume=True):
    name,pose=item;folder=ROOT/'objects'/name/'poses'/pose
    if resume:
        cache=C.load(folder,count,strict=False)
        if cache:return dict(object=name,pose=pose,reused=True,**cache['report']['summary'])
    source_hashes=C.sources()
    began=time.monotonic();task=read_task(name,pose)
    g=PairGeometry([task],count=count,initialize_candidates=False,use_precomputed=False)
    cat=catalogue(task.domain.mesh);g.catalogues=[cat];g.analyzers=[W.Analyzer(g.mesh,g.depth,cat)]
    pools={};arrays=dict(centers=g.centers,center_faces=g.faces,com=task.domain.com,
        roadmap_points=g.paths.points,roadmap_labels=g.paths.labels)
    for fraction in C.AREAS:
        rows=[];triangles=[];faces=[];areas=[];forces=[];offsets=[0];fo=[0]
        for index,(center,face) in enumerate(zip(g.centers,g.faces)):
            patch,fit=g.surface.fit_area(center,int(face),fraction*g.mesh.area)
            e=g.make(index,fit['radius_m'],patch,target_area=fraction*g.mesh.area);e['area_fit']=fit
            c=e['contact'];c['candidate_id']=f'{pose}_A{fraction*100:g}_C{index+1:03d}'
            normals=np.repeat(-g.mesh.face_normals[c['source_faces']],3,axis=0)
            raw=np.c_[normals,np.cross(c['triangles_m'].reshape(-1,3)-task.domain.com,normals)]
            triangles.append(c['triangles_m']);faces.append(c['source_faces']);areas.append(c['triangle_areas_m2']);forces.append(raw)
            offsets.append(offsets[-1]+len(c['triangles_m']));fo.append(fo[-1]+len(raw))
            rows.append(e)
        pools[fraction]=rows;prefix='a'+f'{fraction:g}'
        arrays.update({prefix+'_triangles':np.concatenate(triangles),prefix+'_faces':np.concatenate(faces),
            prefix+'_areas':np.concatenate(areas),prefix+'_forces':np.concatenate(forces),
            prefix+'_offsets':np.array(offsets),prefix+'_force_offsets':np.array(fo)})
    stage=Path(tempfile.mkdtemp(prefix='.step2-',dir=folder))
    try:
        np.savez_compressed(stage/'geometry.npz',**arrays)
        metadata={}
        for fraction,rows in pools.items():
            entries=[]
            for e in rows:
                out=dict(e);out['contact']={k:v for k,v in e['contact'].items() if k not in ('center_m','triangles_m','source_faces','triangle_areas_m2')}
                # Keep full certified / locked / unresolved direction checks for audit.
                entries.append(out)
            metadata[f'{fraction:g}']=entries
        with gzip.open(stage/'entries.json.gz','wt') as f:json.dump(dict(pools=metadata,catalogue=cat,roadmap=g.paths.record,sampling=g.sampling),f,default=C.json_value,separators=(',',':'))
        picture(task,g,pools,stage/'candidates.png')
        summary=dict(candidates=count*len(C.AREAS),accepted={f'{f:g}':sum(e['valid'] for e in rows) for f,rows in pools.items()},seconds=round(time.monotonic()-began,3))
        report=dict(schema=C.SCHEMA,object=name,pose=pose,count=count,area_fractions=list(C.AREAS),depth_m=g.depth,
            inputs=C.inputs(folder),sources=source_hashes,summary=summary,
            scope='Native pose heads and continuous finite-menu exits. Joint force coverage, common directions, all-pose exclusions and full support construction remain runtime checks.',
            artifacts={p.name:C.digest(p) for p in stage.iterdir()})
        if C.sources()!=source_hashes:raise RuntimeError('Head computation sources changed during build; rerun required')
        (stage/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        old=folder/'.step2-old';target=folder/'step2'
        if old.exists():shutil.rmtree(old)
        if target.exists():target.rename(old)
        try:stage.rename(target)
        except BaseException:
            if old.exists():old.rename(target)
            raise
        if old.exists():shutil.rmtree(old)
    finally:
        if stage.exists():shutil.rmtree(stage)
    return dict(object=name,pose=pose,reused=False,**summary)

def main():
    p=argparse.ArgumentParser();p.add_argument('--object');p.add_argument('--pose');p.add_argument('--workers',type=int,default=4);p.add_argument('--count',type=int,default=200);p.add_argument('--no-resume',action='store_true');a=p.parse_args()
    # Compile the optional scalar geometry kernel before forking workers.
    import trimesh
    from step2_local_support.geometry import Clearance
    box=trimesh.creation.box()
    Clearance(box,1e-12).obstruction(box.vertices)
    from step2_local_support.circles import edge_interval
    edge_interval(box.triangles[0],box.face_normals[0],box.vertices[:2],1e-9)
    names=[a.object] if a.object else json.loads((ROOT/'objects/cases.json').read_text())['active_objects']
    jobs=[(n,a.pose or f'pose_{i}') for n in names for i in ([1] if a.pose else range(1,31))];rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(build,j,a.count,not a.no_resume):j for j in jobs}
        for future in as_completed(futures):
            row=future.result();rows.append(row);print(json.dumps(row),flush=True)
    report=dict(complete=True,objects=len(names),poses=len(rows),candidate_count=sum(r['candidates'] for r in rows),cases=rows)
    (ROOT/'codes/precompute_objects/heads_report.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
