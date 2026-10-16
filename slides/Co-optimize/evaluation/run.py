"""Overwrite one normalized three-column BBX image per pose set."""
from pathlib import Path
import argparse, json, os, shutil, sys
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

COOPT=Path(__file__).resolve().parents[1];ROOT=COOPT.parents[1]
sys.path[:0]=[str(ROOT),str(COOPT/'evaluation'),str(COOPT/'helper_func'),str(COOPT/'vis_func'),str(COOPT/'step5.1')]
from metrics import static_bounds,BOX_KEYS
import artifact_store as A
import step51_render as V
import hull_render as H

VERSION='representative_pose_strip_box_v4'
DEFAULT_EXPERIMENT='outside_in_all52_20261015_03'
FIGURE_SIZE=(1950,1210)


def sha(path):return A.digest(path)
def relative(path):return str(Path(path).resolve().relative_to(ROOT))
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def directory(name):return A.result_root(name)/'data/evaluation'
def image_path(name,label):return A.VISUALS/name/label/'evaluation.png'
def report_path(name,label):return A.result_root(name)/label/'evaluation.json'


def normalize(result):
    reference=result['object_static']['volume_litres']
    if not np.isfinite(reference)or reference<=0:raise ValueError('Invalid object-only BBX')
    result['relative_bbx_volume']={k:None if result[k]is None else result[k]['volume_litres']/reference for k in BOX_KEYS}
    result['relative_bbx_volume']['object_static']=1.
    result['bbx_increase_percent']={k:None if v is None else 100*(v-1)for k,v in result['relative_bbx_volume'].items()}
    result['bbx_increase_litres']={k:None if result[k]is None else result[k]['volume_litres']-reference for k in BOX_KEYS}
    result['evaluation_output_version']=VERSION
    return result


def corners(record):
    lo,hi=record['min_m'],record['max_m']
    return np.array([[x,y,z]for x in [lo[0],hi[0]]for y in [lo[1],hi[1]]for z in [lo[2],hi[2]]])


def presentation(layout,result):
    """Group actual installation transforms without superimposing their meshes."""
    poses={p['layout_index']:p for p in result['per_pose']};groups={}
    for value in layout['active']:
        k=int(value);T=layout['native_world'][layout['hosts'][k]]
        group=groups.setdefault(np.round(T,11).tobytes(),dict(indices=[],fixture_transform=T.tolist()))
        group['indices'].append(k)
    groups=list(groups.values())
    if not result['passed']:groups=[dict(indices=[int(k)for k in layout['active']],fixture_transform=None)]
    largest=max(groups,key=lambda group:len(group['indices']))
    representative=max(largest['indices'],key=lambda k:poses[k]['objects_in_final_layout']['dimensions_m'][2])
    for index,group in enumerate(groups):
        group.update(placement=index+1 if result['passed']else None,poses=[poses[k]['pose']for k in group['indices']])
    return dict(representative_layout_index=representative,representative_pose=poses[representative]['pose'],
        representative_selection='largest placement group; greatest object vertical extent',
        groups=groups,main_instances=[],thumbnail_poses=[],bbox_includes_all_task_poses=True,
        column_percent_labels=['']+[''if result[k]is None else f'{result["bbx_increase_percent"][k]:+.0f}%'for k in BOX_KEYS[1:]],
        visible_text=[],annotation_policy='percent_and_placement_only',
        bbox_style='complete_wireframe_box',bbox_edges_per_main_column=[],
        volume_ratios_printed=False,litres_printed=False,figure_size=list(FIGURE_SIZE))


def box_piece(record,camera,size):
    """Render all twelve exact BBX edges with ordinary scene depth occlusion."""
    _,basis,span=camera;box=corners(record)
    pairs=[(index,index^bit)for index in range(8)for bit in [1,2,4]if index<(index^bit)]
    starts=box[[a for a,b in pairs]];ends=box[[b for a,b in pairs]]
    projected=(ends-starts)@basis[:2].T
    lengths=np.linalg.norm(projected,axis=1)
    perpendicular=np.c_[-projected[:,1],projected[:,0]]/lengths[:,None]
    offsets=perpendicular@basis[:2]*(.9*span/size)
    vertices=np.stack([starts+offsets,ends+offsets,ends-offsets,starts-offsets],axis=1)
    triangles=vertices[:,[[0,1,2],[0,2,3]],:].reshape(-1,3,3)
    return dict(triangles=triangles,normals=np.broadcast_to(basis[2],triangles.shape),
        colors=np.broadcast_to(V.CPU.rgb('#9FB1BD'),triangles.shape),bias=4e-6,unlit=True),len(pairs)


def posed_piece(mesh,transform,color,opacity=1.):
    posed=mesh.copy();posed.apply_transform(transform)
    return V.piece(posed,color,opacity)


def render(renderer,body,nominal,full,layout,result,out):
    """One representative scene, full-task BBX, and one shared grouped pose strip."""
    normalize(result);out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    size=650;header=145;display=presentation(layout,result);result['presentation']=display
    cameras={k:V.camera_for(corners(result[k]))for k in BOX_KEYS if result[k]is not None}
    span=max(c[2]for c in cameras.values())
    baseline={p['layout_index']:np.array(p['unsupported_world_transform'])for p in result['per_pose']}
    k=display['representative_layout_index'];T=layout['native_world'][layout['hosts'][k]]
    board=Image.new('RGB',FIGURE_SIZE,'white');ink=ImageDraw.Draw(board)
    font=lambda n:ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',n)
    for column,(key,fixture)in enumerate(zip(BOX_KEYS,[None,nominal,full])):
        center=size*(column+.5)
        label=display['column_percent_labels'][column]
        if label:
            ink.text((center,106),label,font=font(36),fill='#267C9F',anchor='mt')
            display['visible_text'].append(label)
        record=result[key]
        if record is None:
            display['main_instances'].append(dict(objects=0,supports=0))
            display['bbox_edges_per_main_column'].append(0)
            continue
        focus,basis,_=cameras[key];camera=(focus,basis,span)
        U=baseline[k]if column==0 else T@layout['placements'][k]
        pieces=[posed_piece(body,U,V.GRAY)]
        if fixture is not None:pieces.append(posed_piece(fixture,T,V.BLUE))
        display['main_instances'].append(dict(objects=1,supports=int(fixture is not None)))
        box,edge_count=box_piece(record,camera,size);pieces.append(box)
        display['bbox_edges_per_main_column'].append(edge_count)
        tile=renderer.render([V.CPU.placed(p)for p in pieces],camera,size)
        board.paste(tile,(column*size,header))
    ink.line([(60,853),(1890,853)],fill='#E3E9ED',width=2)
    groups=display['groups'];gap=34;count=sum(len(g['indices'])for g in groups)
    cell=min(210.,(1830-gap*(len(groups)-1))/count);thumb=min(176,int(cell)-8)
    group_cameras=[]
    for group in groups:
        box=result['fixture_states'][group['placement']-1]['with_full_support']if result['passed']else result['object_static']
        group_cameras.append(V.camera_for(corners(box)))
    thumb_span=max(c[2]for c in group_cameras)
    x=(1950-(cell*count+gap*(len(groups)-1)))/2
    for group,(focus,basis,_)in zip(groups,group_cameras):
        width=cell*len(group['indices']);center=x+width/2;camera=(focus,basis,thumb_span)
        group['thumbnail_camera']=dict(focus_m=focus.tolist(),basis=basis.tolist(),span_m=thumb_span)
        if result['passed']:
            label='Placement '+str(group['placement'])
            ink.text((center,926),label,font=font(21),fill='#5E7D8E',anchor='mt')
            display['visible_text'].append(label)
            ink.line([(x+10,951),(x+10,958),(x+width-10,958),(x+width-10,951)],fill='#B3C7D2',width=2)
        for position,index in enumerate(group['indices']):
            cx=x+cell*(position+.5);S=layout['native_world'][layout['hosts'][index]]
            U=S@layout['placements'][index]if result['passed']else baseline[index]
            pieces=[posed_piece(body,U,V.GRAY)]
            if full is not None:pieces.append(posed_piece(full,S,V.BLUE,.8))
            tile=renderer.render([V.CPU.placed(p)for p in pieces],camera,thumb)
            board.paste(tile,(round(cx-thumb/2),round(975+(176-thumb)/2)))
            pose=next(p['pose']for p in result['per_pose']if p['layout_index']==index)
            display['thumbnail_poses'].append(pose)
        x+=width+gap
    info=PngImagePlugin.PngInfo()
    info.add_text('evaluation_output_version',VERSION)
    info.add_text('relative_bbx_volume',json.dumps(result['relative_bbx_volume']))
    info.add_text('bbx_increase_percent',json.dumps(result['bbx_increase_percent']))
    info.add_text('presentation',json.dumps(display))
    pending=out.with_name('.'+out.name+'.pending')
    board.save(pending,format='PNG',pnginfo=info);pending.replace(out)


def publish_case(name,result,body,nominal,full,layout,renderer,protected=None):
    result=normalize(dict(result));result.setdefault('passed',True);result.setdefault('status','PASS')
    result.update(object=name,force_solver_run=False,geometry_boolean_run=False,
        columns=['object_only','object_and_support_body','object_and_complete_support_with_feet'],
        protected_inputs=dict(protected or result.get('protected_inputs',{})))
    image=image_path(name,result['id']);render(renderer,body,nominal,full,layout,result,image)
    result['output']=relative(image);result['image_sha256']=sha(image)
    save(report_path(name,result['id']),result)
    return result


def poses_for(label):return ['pose_'+n for n in label.split('/')[-1].removeprefix('pose').split('+')]


def unavailable(name,case,body,renderer):
    poses=poses_for(case['id']);native=[];protected={}
    for pose in poses:
        path=ROOT/'objects'/name/'poses'/pose/'setup.npz';protected[relative(path)]=sha(path)
        with np.load(path)as arrays:native.append(arrays['T_world_mesh'].copy())
    count=len(poses);layout=dict(native_world=np.asarray(native),hosts=np.arange(count),
        placements=np.repeat(np.eye(4)[None],count,axis=0),active=np.arange(count))
    result=static_bounds(body.vertices,layout,pose_names=poses)
    result.update(id=case['id'],passed=False,status=case.get('status','STEP4_FAIL'),step53_source=None)
    return publish_case(name,result,body,None,None,layout,renderer,protected)


def final_stage(root,experiment):
    unified=root/'step5'/experiment
    path=unified/'data/report.json'
    if path.exists():
        record=read(path)
        if record.get('passed') and record.get('complete'):return unified
    for stage in ['step5.3','step5.2']:
        folder=root/'step5'/stage/experiment;path=folder/'data/report.json'
        if path.exists():
            record=read(path)
            if record.get('stage_type')=='support_pruning':continue
            if record.get('passed')and record.get('complete'):return folder
    raise FileNotFoundError('No certified complete support: '+str(root))


def evaluate(name,case,body,renderer,experiment=None):
    if not case['passed']:return unavailable(name,case,body,renderer)
    base=A.result_root(name);folder=final_stage(base/case['id'],experiment)if experiment else ROOT/case['output']
    path=folder/'data/report.json';final=read(path)
    if not final['passed']or not final['complete']or final['selected_step4_experiment']!=case['experiment']:
        raise ValueError('Unfinished or mismatched selected support: '+case['id'])
    s4=ROOT/final['selected_step4_source'];protected={}
    def protect(p):protected[relative(p)]=sha(p);return p
    protect(path)
    for filename,expected in final['artifacts'].items():
        if sha(folder/filename)!=expected:raise ValueError('Changed saved artifact: '+filename)
        protect(folder/filename)
    nominal_path=protect(s4/'support.obj');layout_path=protect(s4/'layout.npz')
    layout_report=s4/'data/report.json'
    if not layout_report.exists():layout_report=s4/'report.json'
    saved=read(protect(layout_report))
    nominal=trimesh.load(nominal_path,force='mesh',process=False)
    full=trimesh.load(folder/final['support'],force='mesh',process=False)
    with np.load(layout_path)as arrays:layout={k:arrays[k].copy()for k in arrays.files}
    result=static_bounds(body.vertices,layout,nominal.vertices,full.vertices,saved['poses'])
    result.update(id=case['id'],step4_source=relative(s4),step53_source=relative(folder),
        material_step4_cm3=abs(float(nominal.volume))*1e6,material_full_cm3=abs(float(full.volume))*1e6)
    return publish_case(name,result,body,nominal,full,layout,renderer,protected)


def write_summary(name,selection,selection_path=None):
    folder=directory(name);cases=selection.get('results',selection.get('selected',selection.get('cases',[])))
    rows=[];missing=[]
    for case in cases:
        path=report_path(name,case['id'])
        if not path.exists():missing.append(case['id']);continue
        row=read(path)
        if row.get('evaluation_output_version')!=VERSION or bool(row['passed'])!=bool(case['passed']):
            missing.append(case['id']);continue
        rows.append(row)
    valid=[r for r in rows if r['passed']]
    protected={p:h for row in rows for p,h in row['protected_inputs'].items()}
    protected[relative(ROOT/'objects'/name/'mesh.stl')]=sha(ROOT/'objects'/name/'mesh.stl')
    changed=[p for p,h in protected.items()if sha(ROOT/p)!=h]
    if changed:raise ValueError('Saved numerical input changed: '+str(changed[:3]))
    code={}
    for source in [Path(__file__),Path(__file__).with_name('metrics.py'),Path(__file__).with_name('verify_outputs.py'),
            Path(V.__file__),Path(H.__file__),V.CPU_PATH,V.BASELINE/'step4_connect_support/clean_render.py',
            V.BASELINE/'step4_connect_support/translucent_raster.cpp']:
        target=folder/'executed_sources'/source.resolve().relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);code[relative(source)]=sha(source)
    report=dict(complete=not missing,passed=not missing,object=name,group_count=len(cases),rendered_groups=len(rows),
        certified_groups=len(valid),unavailable_groups=len(rows)-len(valid),missing=missing,results=rows,
        evaluation_output_version=VERSION,source_selection=None if selection_path is None else relative(selection_path),
        unsupported_baseline=1.,fixed_output_pattern=f'output/{name}/<pose-set>/evaluation.png',
        mean_bbx_increase_percent={k:float(np.mean([r['bbx_increase_percent'][k]for r in valid]))for k in BOX_KEYS}if valid else{},
        force_solver_run=False,geometry_boolean_run=False)
    save(folder/'selection.json',selection);save(folder/'summary.json',report)
    save(folder/'verification.json',dict(passed=not missing,protected_inputs=protected,executed_code=code,
        published_images={r['output']:r['image_sha256']for r in rows},independent_metrics_recomputed=False))
    return report


def clean_legacy(name):
    """Delete derived evaluation outputs, preserving models and physical audits."""
    base=A.result_root(name);visual=A.VISUALS/name;fixed=directory(name);targets=set();folders=set()
    allowed={ROOT/r['output']for r in read(fixed/'summary.json')['results']}
    prune=lambda d:any(tag in d for tag in ['sources','snapshot','dependencies'])
    for tree in [base,visual]:
        for parent,dirs,files in os.walk(tree):
            parent=Path(parent);dirs[:]=[d for d in dirs if not prune(d)and parent/d!=fixed]
            for old_name in ['evaluation','evaluations']:
                if old_name in dirs:folders.add(parent/old_name);dirs.remove(old_name)
            for filename in files:
                path=parent/filename
                if tree==visual and filename in ['static_occupancy.png','evaluation.png']and path not in allowed:
                    targets.add(path)
                elif tree==base and 'evaluation'in filename.lower()and path.suffix=='.json':
                    value=read(path)
                    if value.get('evaluation_output_version')==VERSION:continue
                    physical_audit=any(k in value for k in ['geometry_audit','saved_full_demand_masks_checked',
                        'original_demand_masks_checked','full_certificate_binding'])
                    derived_report=filename in ['previous_evaluation.json','evaluation_results.json',
                        'control_evaluation.json','progress_evaluation.json']and not physical_audit
                    if 'object_static'in value or 'independent_bbx_sum_l'in value or derived_report:targets.add(path)
                elif tree==base and filename=='static_occupancy.png':targets.add(path)
    for path in (base/'data').glob('evaluation_*'):
        if path.is_dir():folders.add(path)
        elif path.is_file():targets.add(path)
    targets.update(p for p in base.glob('evaluation_*')if p.is_file())
    for folder in folders:
        if any(p.suffix.lower()in ['.obj','.npz','.stl','.py','.cpp']for p in folder.rglob('*')if p.is_file()):
            raise ValueError('Refuse to delete physical artifacts or code from '+str(folder))
        targets.update(p for p in folder.rglob('*')if p.is_file())
    protected=read(fixed/'verification.json')['protected_inputs']
    collision=set(map(relative,targets))&set(protected)
    if collision:raise ValueError('Evaluation cleanup intersects saved physical inputs: '+str(collision))
    old=read(fixed/'cleanup.json')if(fixed/'cleanup.json').exists()else{}
    deleted=old.get('deleted_files',{})
    for path in sorted(targets):
        if path.exists():deleted[relative(path)]=sha(path);path.unlink()
    removed=set(old.get('removed_directories',[]))
    for folder in sorted(folders,key=lambda p:-len(p.parts)):
        if folder.exists():shutil.rmtree(folder);removed.add(relative(folder))
    save(fixed/'cleanup.json',dict(old,complete=True,deleted_files=deleted,removed_directories=sorted(removed),
        derived_file_count=len(deleted),removed_directory_root_count=len(removed),
        physical_models_and_certificates_preserved=True,no_backup_copies_created=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('object',nargs='?',default='B')
    parser.add_argument('--experiment',help='Select input geometry; never names an output folder')
    parser.add_argument('--selection-summary',type=Path);parser.add_argument('--sets',nargs='+')
    parser.add_argument('--batch-name',help=argparse.SUPPRESS);parser.add_argument('--replace',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args();base=A.result_root(args.object);selection=args.selection_summary
    if selection is None:
        latest=base/'data/step5_current.json'
        current=directory(args.object)/'selection.json'
        selection=(ROOT/read(latest)['summary'] if latest.exists() and read(latest).get('complete')
            else current if current.exists() else base/'data'/('step5_feasible_'+DEFAULT_EXPERIMENT)/'summary.json')
    selection=A.resolve_artifact(selection);record=read(selection)
    cases=record.get('results',record.get('selected',record.get('cases',[])))
    if args.sets and set(args.sets)-{c['id']for c in cases}:parser.error('Unknown pose set')
    chosen=[c for c in cases if not args.sets or c['id']in args.sets]
    body=trimesh.load(ROOT/'objects'/args.object/'mesh.stl',force='mesh',process=False);renderer=V.Renderer()
    for i,case in enumerate(chosen):
        evaluate(args.object,case,body,renderer,args.experiment)
        print('EVALUATION',i+1,'/',len(chosen),case['id'],flush=True)
    summary=write_summary(args.object,record,selection)
    if summary['complete']:clean_legacy(args.object)
    print('EVALUATION_FIXED',summary['rendered_groups'],'/',summary['group_count'],flush=True)


if __name__=='__main__':main()
