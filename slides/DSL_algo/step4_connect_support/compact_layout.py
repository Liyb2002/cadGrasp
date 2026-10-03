"""Finite compact reseating search with independent, rigidly grouped heads.

All contact patches and loads stay fixed in their owner task coordinates. The
entire group receives one task-to-fixture transform. Virtual workpieces in two
different tasks may overlap; real material must clear both workpieces/sweeps.
"""
import itertools
import json
from pathlib import Path

import manifold3d as md
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import trimesh

from step3_scheculer import contacts as I
from step4_connect_support.run_independent import read_case as read_independent
from step4_connect_support import build_coupled_saddle as S, zero_thickness_heads as Z
from step4_connect_support import head_registration as H
from step4_connect_support.fixture_view import cells_for
from step4_connect_support.codesign_port import build_local_bodies as L
from step2_local_support import geometry as G

TOL=8e-14


def load_case(output):
    output=Path(output).resolve()
    group=output.parent.parent
    poses=['pose_'+p for p in group.name.removeprefix('pose').split('+')]
    if len(poses)!=2:
        raise ValueError('The first compact-search implementation requires two independent poses')
    case=read_independent(group.parent.name,poses,output/'data/input',
        independent_root=group/'step3_scheculer/independent_poses_floor2mm')
    case.pair=group;case.output=output
    case.heads=[[list(c['triangles_m']) for c in row] for row in case.groups]
    case.head_model=Z.MODEL
    old_path=group/'step4/data/report.json'
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
    for k in range(2):
        xy=case.demands[k];xy=xy[ConvexHull(xy).vertices]
        case.mandatory.append(np.vstack([case.root_points[k],np.c_[xy,np.zeros(len(xy))],case.tasks[k].floor]))
    case.scale=max(float(t.domain.mesh.extents.max()) for t in case.tasks)
    case.schedule.update(head_model=Z.MODEL,mandatory_head_thickness_m=0.,
        model='Independent contact groups; compact task-to-fixture placements searched',
        shared_head_count=0,constructor_generates_support_roots=True)
    I.save(case.source/'schedule.json',case.schedule)
    # Rebuild source hashes after the explicit staged schedule update.
    return case


def transform_solid(solid,basis,offset):
    """Row-vector world @ basis + offset, applied to normalized Manifold data."""
    return solid.transform(np.c_[np.asarray(basis).T,np.asarray(offset)/S.SCALE])


def span_metrics(points,bases,offsets):
    spans=[np.ptp((points-o)@b.T,axis=0) for b,o in zip(bases,offsets)]
    return dict(maximum_horizontal_span_m=float(max(s[:2].max() for s in spans)),
        horizontal_spans_m=[s[:2].tolist() for s in spans],
        maximum_horizontal_box_area_m2=float(max(np.prod(s[:2]) for s in spans)))


def precompute(case):
    work=case.output/'data/search';work.mkdir(parents=True,exist_ok=True)
    case.sweeps=[];case.padded_sweeps=[]
    cube=md.Manifold.cube([2*S.RELIEF/S.SCALE]*3).translate([-S.RELIEF/S.SCALE]*3)
    for k,task in enumerate(case.tasks):
        raw={};padded={}
        for ident in case.menus[k]:
            if abs(case.catalogues[k][ident][2])>1e-12:
                raise ValueError('Current compact seating replay requires horizontal object withdrawal')
            key=dict(inputs=I.hashes(task.inputs),direction=case.catalogues[k][ident].tolist(),length_m=S.SWEEP_LENGTH)
            meta=work/f'sweep_{k}_{ident}.json';npz=meta.with_suffix('.npz')
            if meta.exists() and json.loads(meta.read_text())==key and npz.exists():
                with np.load(npz) as z:mesh=trimesh.Trimesh(z['v'],z['f'],process=False)
            else:
                mesh=S.swept_solid(task.domain.mesh,-S.SWEEP_LENGTH*case.catalogues[k][ident])
                np.savez_compressed(npz,v=mesh.vertices,f=mesh.faces);I.save(meta,key)
            raw[ident]=S.solid(mesh);padded[ident]=raw[ident].minkowski_sum(cube)
            own=abs(float((raw[ident]^case.root_solids[k]).volume()))*S.SCALE**3
            if own>TOL:raise ValueError(f'Owner roots block the saved local direction: {task.pose}/{ident}: {own}')
        case.sweeps.append(raw);case.padded_sweeps.append(padded)


def orientation_menu():
    # A compact slab with opposite usable faces is the first family. Tilted
    # families permit wedges; twist and floor-normal azimuth vary separately.
    for yaw in range(0,360,15):
        yield Rotation.from_euler('x',180,degrees=True).as_matrix()@Rotation.from_euler('z',yaw,degrees=True).as_matrix(),dict(tilt_deg=180,twist_deg=yaw,azimuth_deg=0)
    for tilt,azimuth,twist in itertools.product((100,130,155),(0,90,180,270),range(0,360,45)):
        b=(Rotation.from_euler('z',twist,degrees=True).as_matrix()
            @Rotation.from_euler('x',tilt,degrees=True).as_matrix()
            @Rotation.from_euler('z',azimuth,degrees=True).as_matrix())
        yield b,dict(tilt_deg=tilt,twist_deg=twist,azimuth_deg=azimuth)


def candidates(case,max_span_ratio=1.65):
    """Cheap placement bounds before exact all-material sweep screening."""
    cloud0=case.mandatory[0]
    # Use head groups plus demand endpoints, not coincident object copies, as
    # the compactness lower bound. Actual bodies are checked again on export.
    center0=case.root_points[0].mean(0)
    records=[];rejected=dict(floors=0,span=0);limit=max_span_ratio*case.scale
    for b,angles in orientation_menu():
        cloud1=case.mandatory[1]@b
        heads1=case.root_points[1]@b
        center1=heads1.mean(0)
        # This finite family varies translations in task0 XY, then chooses the
        # lowest height satisfying BOTH halfspaces for required roots/demands.
        for dx,dy in itertools.product(np.linspace(-.55,.55,9)*case.scale,repeat=2):
            xy=center0[:2]-center1[:2]+[dx,dy]
            low=-float(cloud1[:,2].min())+.002
            n=b[2]
            if n[2]>=-.05:continue
            upper_height=(float((cloud0@n).min())-xy@n[:2]-.002)/n[2]
            # n_z<0 makes this a lower bound on the top-plane translation.
            z0=max(low,upper_height)
            for extra in (0.,.04*case.scale,.10*case.scale,.20*case.scale):
                o=np.r_[xy,z0+extra]
                points=np.vstack([cloud0,cloud1+o])
                bases=np.array([np.eye(3),b]);offsets=np.array([np.zeros(3),o])
                # Ground-demand and pivot halfspace compatibility here is an
                # explicit restricted family, not a general reseating theorem.
                if min(points[:,2].min(),float(((points-o)@b.T)[:,2].min())) < -1e-9:
                    rejected['floors']+=1;continue
                patches=[case.patch_points[0],case.patch_points[1]@b+o]
                if min(float(((v-off)@base.T)[:,2].min()) for v in patches for base,off in zip(bases,offsets)) < .002-1e-9:
                    rejected['floors']+=1;continue
                metrics=span_metrics(points,bases,offsets)
                if metrics['maximum_horizontal_span_m']>limit or np.ptp(points,axis=0).max()>1.8*case.scale:
                    rejected['span']+=1;continue
                score=(metrics['maximum_horizontal_box_area_m2']/case.scale**2
                    +.2*metrics['maximum_horizontal_span_m']/case.scale
                    +.1*np.linalg.norm(center1+o-center0)/case.scale)
                records.append(dict(bases=bases.tolist(),offsets=offsets.tolist(),
                    score=float(score),angles=angles,metrics=metrics))
    records.sort(key=lambda r:r['score'])
    return records,rejected


def screen(case,record):
    bases=np.asarray(record['bases']);offsets=np.asarray(record['offsets'])
    b,o=bases[1],offsets[1]
    foreign1=transform_solid(case.root_solids[1],b,o)
    foreign0=transform_solid(case.root_solids[0],b.T,-o@b.T)
    menus=[];tests=[]
    for k,foreign in enumerate((foreign1,foreign0)):
        legal=[]
        for ident in case.menus[k]:
            overlap=abs(float((foreign^case.padded_sweeps[k][ident]).volume()))*S.SCALE**3
            tests.append(dict(pose=case.poses[k],direction_id=ident,foreign_root_sweep_overlap_m3=overlap))
            if overlap<=TOL:legal.append(ident)
        if not legal:return None,tests
        menus.append(legal)
    return menus,tests


def search(case,max_span_ratio=1.65,max_checks=2500,keep=12):
    precompute(case)
    proposals,rejected=candidates(case,max_span_ratio)
    folder=case.output/'data/search'
    I.save(folder/'search_config.json',dict(max_span_ratio=max_span_ratio,max_checks=max_checks,keep=keep,
        object_scale_m=case.scale,maximum_horizontal_span_m=max_span_ratio*case.scale,
        candidate_count=len(proposals),cheap_rejections=rejected,
        floor_domain='Restricted to demand endpoints and original object pivots above every floor',
        global_optimality_or_infeasibility_claim=False))
    selected=[];tested=[]
    for i,record in enumerate(proposals[:max_checks]):
        # Avoid spending all retained slots on negligible variations of the
        # same seating, while still evaluating them in the recorded search.
        if any(np.linalg.norm(np.asarray(record['offsets'][1])-np.asarray(r['offsets'][1]))<.018
            and np.linalg.norm(np.asarray(record['bases'][1])-np.asarray(r['bases'][1]))<.4 for r in selected):continue
        menus,checks=screen(case,record)
        tested.append(dict(candidate_index=i,passed=menus is not None,checks=checks))
        if menus is not None:
            record.update(direction_menus=menus,candidate_index=i)
            selected.append(record)
            print('COMPACT LAYOUT',case.pair.name,len(selected),'rank',i,'span mm',record['metrics']['maximum_horizontal_span_m']*1000,flush=True)
        if i%100==0 or menus is not None:
            I.save(folder/'progress.json',dict(complete=False,tested=len(tested),selected=len(selected),candidate_index=i))
        if len(selected)>=keep:break
    result=dict(complete=True,proposals=len(proposals),exact_checks=len(tested),selected=selected,
        tests=tested,search_exhausted=len(tested)==len(proposals),global_infeasibility_claim=False,
        provenance=dict(inputs=I.hashes(case.paths),code=I.hashes([Path(__file__),Path(S.__file__),Path(S.swept_solid.__code__.co_filename)])))
    I.save(folder/'layouts.json',result)
    print('COMPACT SEARCH COMPLETE',case.pair.name,len(selected),'layouts',len(tested),'tested',flush=True)
    return selected


def root_records(case,bases,offsets):
    records=[];index=0
    for k,group in enumerate(case.groups):
        for contact,cells in zip(group,case.support_seeds[k]):
            row=dict(case.support_seed_records[index]);index+=1
            patch=contact['triangles_m'].reshape(-1,3)@bases[k]+offsets[k]
            root=np.vstack(cells)@bases[k]+offsets[k]
            row['minimum_contact_height_by_pose_m']=[float(((patch-o)@b.T)[:,2].min()) for b,o in zip(bases,offsets)]
            row['minimum_root_height_by_pose_m']=[float(((root-o)@b.T)[:,2].min()) for b,o in zip(bases,offsets)]
            row['placement_model']='compact_independent_seating'
            assert min(row['minimum_contact_height_by_pose_m'])>=.002-1e-9
            assert min(row['minimum_root_height_by_pose_m'])>=-1e-9
            records.append(row)
    return records
