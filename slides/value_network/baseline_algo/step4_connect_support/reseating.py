"""Finite K-pose placement search for rigid, independently selected head groups.

The same solid is used in every task. Object copies may overlap each other;
all physical roots and all eventual connecting material must clear each task.
"""
import hashlib
import itertools
import json
from pathlib import Path
import tempfile

import manifold3d as md
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import trimesh

from step3_scheculer import contacts as I
from step4_connect_support.run_independent import read_case as read_independent
from step4_connect_support import build_coupled_saddle as S, zero_thickness_heads as Z
from step4_connect_support import compact_layout as C
from step4_connect_support.fixture_view import cells_for
from step4_connect_support.codesign_port import build_local_bodies as L
from step2_local_support import geometry as G

TOL=C.TOL
transform_solid=C.transform_solid
root_records=C.root_records


def span_metrics(points,bases,offsets):
    metrics=C.span_metrics(points,bases,offsets)
    spans=[np.ptp((points-o)@b.T,axis=0) for b,o in zip(bases,offsets)]
    metrics['maximum_spatial_span_m']=float(max(s.max() for s in spans))
    metrics['maximum_box_volume_m3']=float(max(np.prod(s) for s in spans))
    return metrics


def load_case(output):
    output=Path(output).resolve()
    group=output.parent if output.name=='step4' else output.parent.parent
    poses=['pose_'+p for p in group.name.removeprefix('pose').split('+')]
    case=read_independent(group.parent.name,poses,output/'data/input',
        independent_root=group/'step3_scheculer/independent_poses_floor2mm')
    case.pair=group;case.output=output
    case.heads=[[list(c['triangles_m']) for c in row] for row in case.groups]
    case.head_model=Z.MODEL
    # Current designs retain their exact root depths as numeric inputs. They
    # must remain runnable after the user deletes historical Step4 outputs.
    current_input=output/'data/design_inputs.json'
    old_path=current_input if current_input.exists() else group/'step4/data/report.json'
    old=I.check_report(old_path)
    seeds={r['candidate_id']:r for r in old['generated_support_roots']}
    case.paths.append(old_path)
    case.support_seeds=[];case.support_seed_records=[]
    case.root_solids=[];case.root_points=[];case.patch_points=[];case.mandatory=[]
    for task,group in zip(case.tasks,case.groups):
        row=[]
        for c in group:
            saved=seeds[c['candidate_id']]
            offsets=G.vertex_offsets(task.domain.mesh,saved['constructor_root_normal_depth_m'])[0]
            cells=cells_for(c,task.domain,offsets);row.append(cells)
            case.support_seed_records.append(dict(saved))
        case.support_seeds.append(row)
        vertices=np.vstack([v for cells in row for v in cells])
        case.root_points.append(vertices[ConvexHull(vertices).vertices])
        case.patch_points.append(np.vstack([c['triangles_m'].reshape(-1,3) for c in group]))
        case.root_solids.append(L.union([S.solid(G.hull_mesh(v)) for cells in row for v in cells]))
    for k in range(len(poses)):
        xy=case.demands[k];xy=xy[ConvexHull(xy).vertices]
        case.mandatory.append(np.vstack([case.root_points[k],np.c_[xy,np.zeros(len(xy))],case.tasks[k].floor]))
    case.scale=max(float(t.domain.mesh.extents.max()) for t in case.tasks)
    case.schedule.update(head_model=Z.MODEL,mandatory_head_thickness_m=0.,
        model='Independent groups; K-pose compact rigid seating search',
        shared_head_count=0,constructor_generates_support_roots=True)
    I.save(case.source/'schedule.json',case.schedule)
    # Rebuild source hashes after the explicit staged schedule update.
    return case


def precompute(case):
    """Continuous sweeps, with cache keys based on the actual mesh and code."""
    folder=case.output/'data/search';folder.mkdir(parents=True,exist_ok=True)
    shared=Path(tempfile.gettempdir())/'cadgrasp-reseating-sweeps';shared.mkdir(exist_ok=True)
    cube=md.Manifold.cube([2*S.RELIEF/S.SCALE]*3).translate([-S.RELIEF/S.SCALE]*3)
    case.sweeps=[];case.padded_sweeps=[];case.sweep_planes=[];case.sweep_bounds=[]
    for k,task in enumerate(case.tasks):
        raw={};padded={};planes={};bounds={};legal=[];rejected=[]
        for ident in case.menus[k]:
            direction=case.catalogues[k][ident]
            # Saved vectors describe FIXTURE withdrawal. We animate the
            # opposite object motion; downward object motion crosses its floor.
            if direction[2]>1e-12:
                rejected.append(dict(direction_id=ident,reason='object_exit_crosses_floor'));continue
            key=hashlib.sha256(task.domain.mesh.vertices.tobytes()+task.domain.mesh.faces.tobytes()
                +direction.tobytes()+str((S.SWEEP_LENGTH,I.sha256(Path(S.swept_solid.__code__.co_filename)))).encode()).hexdigest()
            cache=shared/(key+'.npz')
            if cache.exists():
                with np.load(cache) as z:mesh=trimesh.Trimesh(z['v'],z['f'],process=False)
            else:
                mesh=S.swept_solid(task.domain.mesh,-S.SWEEP_LENGTH*direction)
                with tempfile.NamedTemporaryFile(dir=shared,suffix='.npz',delete=False) as tmp:
                    np.savez_compressed(tmp,v=mesh.vertices,f=mesh.faces)
                Path(tmp.name).replace(cache)
            value=S.solid(mesh)
            own=abs(float((value^case.root_solids[k]).volume()))*S.SCALE**3
            if own>TOL:
                rejected.append(dict(direction_id=ident,reason='generated_owner_roots_block_exit',overlap_m3=own));continue
            raw[ident]=value;padded[ident]=value.minkowski_sum(cube)
            hull=ConvexHull(mesh.vertices)
            equation=hull.equations.copy()
            equation[:,3]-=S.RELIEF*np.abs(equation[:,:3]).sum(axis=1)
            planes[ident]=equation
            bounds[ident]=mesh.bounds+np.array([[-S.RELIEF]*3,[S.RELIEF]*3])
            np.savez_compressed(folder/f'sweep_{k}_{ident}.npz',v=mesh.vertices,f=mesh.faces)
            legal.append(ident)
        I.save(folder/f'directions_{k}.json',dict(pose=task.pose,retained=legal,rejected=rejected))
        case.menus[k]=legal
        case.sweeps.append(raw);case.padded_sweeps.append(padded)
        case.sweep_planes.append(planes);case.sweep_bounds.append(bounds)
        print('SEATING SWEEPS',case.pair.name,task.pose,len(legal),flush=True)


def normal_families(count):
    """Faces of compact prisms/corners/polyhedra, not separated stations."""
    angles=np.arange(count)*2*np.pi/count
    yield 'prism',np.c_[np.zeros(count),np.sin(angles),np.cos(angles)]
    if count==2:
        for angle in (110,140,160):
            yield f'wedge_{angle}',np.array([[0,0,1],[0,np.sin(np.deg2rad(angle)),np.cos(np.deg2rad(angle))]])
    elif count==3:
        yield 'corner',np.eye(3)
        yield 'three_sides',np.array([[0,0,1],[0,0,-1],[0,1,0]])
    elif count==4:
        yield 'tetrahedron',np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])/np.sqrt(3)
        a=np.arange(3)*2*np.pi/3
        yield 'prism_and_end',np.vstack([np.c_[np.zeros(3),np.sin(a),np.cos(a)],[1,0,0]])
    elif count==5:
        a=np.arange(3)*2*np.pi/3
        yield 'bipyramid',np.vstack([np.c_[np.zeros(3),np.sin(a),np.cos(a)],[1,0,0],[-1,0,0]])
        yield 'five_sides',np.array([[0,0,1],[0,0,-1],[0,1,0],[0,-1,0],[1,0,0]])


def local_basis(normal,exit_direction,twist):
    normal=np.asarray(normal,dtype=float)
    reference=np.array([1.,0,0]) if abs(normal[0])<.95 else np.array([0.,1,0])
    x=reference-normal*(reference@normal);x/=np.linalg.norm(x)
    frame=np.array([x,np.cross(normal,x),normal])
    angle=np.arctan2(-exit_direction[1],-exit_direction[0])+twist
    return Rotation.from_euler('z',angle).as_matrix()@frame


def radial_placement(case,normals,twists,jitter,direction_ids,extra):
    """Minimal shared radius satisfying every ordered pair of floor planes.

    Keeping original demand endpoints and pivots above all floors is an
    explicitly restricted placement family, not a universal reseating theorem.
    """
    count=len(case.poses)
    bases=np.array([local_basis(n,cat[i],t) for n,cat,i,t in zip(normals,case.catalogues,direction_ids,twists)])
    centers=np.array([p.mean(0) for p in case.root_points])
    shifts=np.array([np.r_[-c[:2]+j,0.]@b for c,j,b in zip(centers,jitter,bases)])
    radius=0.
    for i in range(count):
        for j in range(count):
            if i==j:continue
            n=normals[i];denominator=1-float(normals[j]@n)
            if denominator<1e-6:return None
            low=float(((case.mandatory[j]@bases[j]+shifts[j]-shifts[i])@n).min())
            radius=max(radius,(.002-low)/denominator)
    radius+=extra
    offsets=shifts-radius*normals
    # Gauge: pose0 world is the common fixture frame. This changes neither
    # relative seating nor any task's actual floor/exit geometry.
    first=bases[0].copy();origin=offsets[0].copy()
    bases=bases@first.T;offsets=(offsets-origin)@first.T
    points=np.vstack([p@b+o for p,b,o in zip(case.mandatory,bases,offsets)])
    return dict(bases=bases.tolist(),offsets=offsets.tolist(),radius_m=radius,
        preferred_direction_ids=direction_ids,metrics=span_metrics(points,bases,offsets))


def candidate_pool(case,limit_ratio=2.2,count=18000,seed=20260930):
    rng=np.random.default_rng(seed);size=len(case.poses)
    families=list(normal_families(size));candidates=[]
    # Re-evaluate previous two-pose placements with the new all-pair screen.
    previous=case.pair/'step4/compact/data/search/layouts.json'
    if size==2 and previous.exists():
        old=I.check_report(previous);case.paths.append(previous)
        for row in old['selected']:
            candidates.append(dict(row,family='previous_two_pose_seed',seed_candidate=True))
    # Systematic radius sweeps keep larger, genuinely separated face layouts
    # in the tested budget; sorting only the tightest random placements can
    # otherwise spend the entire budget on intersecting object/head groups.
    permutations=list(itertools.permutations(range(size)))
    if len(permutations)>24:
        chosen=rng.choice(len(permutations),24,replace=False)
        permutations=[permutations[i] for i in sorted(chosen)]
    for family,normals in families:
        for permutation in permutations:
            for extra in (0.,.2,.4,.5,.6,.7,.8,1.):
                row=radial_placement(case,normals[list(permutation)],np.zeros(size),np.zeros((size,2)),
                    [m[0] for m in case.menus],extra*case.scale)
                if row and row['metrics']['maximum_spatial_span_m']<=limit_ratio*case.scale:
                    row.update(family=family,candidate_index=-len(candidates)-1,seed_candidate=True)
                    candidates.append(row)
    rejected=0
    for index in range(count):
        family,normals=families[index%len(families)]
        normals=normals[rng.permutation(size)]
        # Aligned tangent exits are seeds; independent twists permit each task
        # to choose a different direction in the final common fixture frame.
        amplitude=(0.,15.,35.,70.,180.)[index%5]
        twists=np.deg2rad(rng.uniform(-amplitude,amplitude,size))
        ids=[int(rng.choice(menu)) for menu in case.menus]
        jitter=rng.uniform(-.16,.16,(size,2))*case.scale
        if index%7==0:jitter[:]=0
        extra=(0.,.15,.30,.45,.60,.75,1.)[(index//5)%7]*case.scale
        row=radial_placement(case,normals,twists,jitter,ids,extra)
        if row is None:continue
        metrics=row['metrics']
        if metrics['maximum_spatial_span_m']>limit_ratio*case.scale:
            rejected+=1;continue
        row.update(family=family,candidate_index=index,seed_candidate=False)
        candidates.append(row)
    for row in candidates:
        bases=np.asarray(row['bases']);offsets=np.asarray(row['offsets'])
        cloud=np.vstack([p@b+o for p,b,o in zip(case.mandatory,bases,offsets)])
        row['metrics']=span_metrics(cloud,bases,offsets)
        m=row['metrics']
        row['score']=(m['maximum_horizontal_box_area_m2']/case.scale**2
            +.2*m['maximum_spatial_span_m']/case.scale+.2*m['maximum_box_volume_m3']/case.scale**3)
    candidates.sort(key=lambda r:(0 if r['family']=='previous_two_pose_seed' else 1 if r.get('seed_candidate',False) else 2,r['score']))
    return candidates,dict(generated=count,retained=len(candidates),span_rejected=rejected,seed=seed)


def screen(case,record):
    """Every physical head group versus every other task's complete corridor."""
    bases=np.asarray(record['bases']);offsets=np.asarray(record['offsets'])
    count=len(bases);menus=[];checks=[]
    fixture_solids=[transform_solid(s,b,o) for s,b,o in zip(case.root_solids,bases,offsets)]
    for k in range(count):
        legal=[]
        preferred=record.get('preferred_direction_ids',[m[0] for m in case.menus])[k]
        menu=list(dict.fromkeys([preferred]+case.menus[k]))
        foreign=[]
        for j in range(count):
            if j==k:continue
            points=(case.root_points[j]@bases[j]+offsets[j]-offsets[k])@bases[k].T
            foreign.append((j,points,None))
        for ident in menu:
            if ident not in case.padded_sweeps[k]:continue
            valid=True
            for j,points,_ in foreign:
                bounds=case.sweep_bounds[k][ident]
                disjoint=bool(np.any(points.max(0)<bounds[0]-1e-9) or np.any(points.min(0)>bounds[1]+1e-9))
                if not disjoint:
                    planes=case.sweep_planes[k][ident]
                    disjoint=bool(np.any((points@planes[:,:3].T+planes[:,3]).min(axis=0)>1e-9))
                if disjoint:
                    overlap=0.;method='conservative_convex_separation'
                else:
                    solid=transform_solid(fixture_solids[j],bases[k].T,-offsets[k]@bases[k].T)
                    overlap=abs(float((solid^case.padded_sweeps[k][ident]).volume()))*S.SCALE**3
                    method='continuous_mesh_boolean'
                checks.append(dict(pose=case.poses[k],foreign_pose=case.poses[j],direction_id=ident,
                    overlap_m3=overlap,method=method))
                if overlap>TOL:valid=False;break
            if valid:legal.append(ident)
        if not legal:return None,checks
        menus.append(legal)
    return menus,checks


def search(case,limit_ratio=2.2,count=18000,max_checks=6000,keep=6,seed=20260930):
    precompute(case)
    folder=case.output/'data/search'
    if not all(case.menus):
        rows=[];summary=dict(reason='at_least_one_owner_has_no_floor_safe_exit');pool=[]
    else:pool,summary=candidate_pool(case,limit_ratio,count,seed)
    selected=[];tests=[];fallback=pool[0] if pool else None
    for index,row in enumerate(pool[:max_checks]):
        b=np.asarray(row['bases']);o=np.asarray(row['offsets'])
        if any(np.linalg.norm(o-np.asarray(r['offsets']))<.018 and np.linalg.norm(b-np.asarray(r['bases']))<.4 for r in selected):continue
        menu,checks=screen(case,row)
        tests.append(dict(rank=index,passed=menu is not None,checks=checks))
        if menu is not None:
            # Contact height is evaluated on the exact complete patches.
            C.root_records(case,b,o)
            row.update(direction_menus=menu,screen=checks)
            selected.append(row)
            print('SEATING LAYOUT',case.pair.name,len(selected),'rank',index,'span_mm',row['metrics']['maximum_horizontal_span_m']*1000,flush=True)
        if index%100==0 or menu is not None:
            I.save(folder/'progress.json',dict(complete=False,tested=len(tests),selected=len(selected),rank=index))
        if len(selected)>=keep:break
    result=dict(complete=True,selected=selected,fallback=fallback,tests=tests,summary=summary,
        exact_layout_checks=len(tests),limit_ratio=limit_ratio,seed=seed,count=count,max_checks=max_checks,keep=keep,
        minimum_contact_height_m=.002,global_infeasibility_claim=False,
        scope='Finite polyhedral rigid-group layouts; original demand/pivot cross-floor restriction retained',
        provenance=dict(inputs=I.hashes(case.paths),code=I.hashes([Path(__file__),Path(C.__file__),Path(S.swept_solid.__code__.co_filename)])))
    I.save(folder/'layouts.json',result)
    print('SEATING SEARCH COMPLETE',case.pair.name,len(selected),'layouts',len(tests),'checked',flush=True)
    return result
