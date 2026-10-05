"""Construct local bodies only after the shared physical head is registered.

Historical six-patch recipes remain readable, but the mandatory identity check
rejects them. Disabling final load audits does not disable this model constraint.
"""
import argparse
from contextlib import redirect_stdout
import itertools
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import MultiPoint
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step1.needs import ROOT, OUTPUTS
from step2_local_support import withdrawal as W, circles as P
from step3_scheculer import contacts as I
from step5_connect_support import build_local_bodies as L, build_coupled_saddle as S
from step5_connect_support.run_greedy import read_case, plain, clear_previous_outputs
from step5_connect_support.fixture_view import frame
from step5_connect_support.refresh_shared_geometry_view import write_viewer

HERE = Path(__file__).resolve().parent


def original_case():
    return read_case(OUTPUTS/'B/pose1+3/step3_scheculer/sequential_3plus2/from_pose_1/terminal_expansion/particle_009')


def original_placement(case):
    return dict(kind='original_pose1_3_fixed_placement', bases=np.array([np.eye(3), S.ROTATION]),
        offsets=np.array([np.zeros(3), -S.TRANSLATION@S.ROTATION]),
        directions=np.array([case.catalogues[k][i] for k,i in enumerate(S.DIRECTION_IDS)]),
        direction_ids=S.DIRECTION_IDS)


def cheap_floor_test(case, bases, offsets):
    # The object pivot is NOT fixture material. It may lie below the other
    # fixture floor in the independent-placement model; retain it in the bound.
    if not hasattr(case, 'demand_hulls'):
        case.demand_hulls = [xy[ConvexHull(xy).vertices] for xy in case.demands]
    for k in (0,1):
        j=1-k; n=bases[j][2]
        xy=case.demand_hulls[k]
        h=(np.c_[xy,np.zeros(len(xy))]@bases[k]+offsets[k]-offsets[j])@n
        pivot=(case.tasks[k].floor@bases[k]+offsets[k]-offsets[j])@n
        if h.min() < min(0.,pivot)-1e-9:
            return False
    return True


def tilted_placements(case, reference, limit=12):
    if case.poses == reference.poses and case.schedule['particle'] == 9:
        yield original_placement(case)
        return
    old=original_placement(reference)
    oldcenters=[np.unique(np.concatenate([v for row in hs for v in row]),axis=0).mean(0) for hs in reference.heads]
    delta=oldcenters[1]@S.ROTATION+old['offsets'][1]-oldcenters[0]
    oldframes=[frame(d) for d in old['directions']]
    menus=[]
    for k,row in enumerate(case.schedule['geometry']['per_pose']):
        ids=row['common_direction_ids']
        ids=sorted(ids,key=lambda i: np.linalg.norm(case.catalogues[k][i]-old['directions'][k]))
        menus.append(ids)
    vertices=[np.unique(np.concatenate([v for row in hs for v in row]),axis=0) for hs in case.heads]
    centers=[v.mean(0) for v in vertices]
    # Exact cell tests use all original heads, not a point-only collision verdict.
    analyzers=[W.Analyzer(t.domain.mesh,P.DEPTH_FRACTION*t.domain.mesh.extents.max(),dict(vectors=c.tolist()))
               for t,c in zip(case.tasks,case.catalogues)]
    cloud=[v[ConvexHull(v).vertices] for v in vertices]
    offsets_menu=[np.zeros(3)]
    for radius in (.012,.025,.045,.07):
        offsets_menu += [radius*np.asarray(q) for q in itertools.product((-1,0,1),repeat=3) if any(q)]
    candidates=[]
    # Transfer the original arrangement in the two withdrawal-aligned frames.
    # Small twists add an orientation search without forcing parallel floors.
    from scipy.spatial.transform import Rotation
    twists=[np.eye(3)]+[Rotation.from_rotvec(np.array(axis)*np.deg2rad(angle)).as_matrix()
                         for angle in (-20,20,-40,40) for axis in ((1,0,0),(0,1,0),(0,0,1))]
    seen=set(); screened=0
    for twist in twists:
        for ids in itertools.product(*menus):
            directions=np.array([case.catalogues[k][i] for k,i in enumerate(ids)])
            if np.abs(directions[:,2]).max()>1e-9:
                continue
            newframes=[frame(d) for d in directions]
            gauge=oldframes[0]@newframes[0].T
            basis=newframes[1]@oldframes[1].T@S.ROTATION@twist@gauge
            bases=np.array([np.eye(3),basis])
            center_offset=centers[0]+delta@gauge-centers[1]@basis
            for shift in offsets_menu:
                offset=center_offset+shift@gauge
                key=tuple(np.round(np.r_[basis.ravel(),offset],7))
                if key in seen: continue
                seen.add(key); screened+=1
                offsets=np.array([np.zeros(3),offset])
                other0=vertices[1]@basis+offset
                other1=(vertices[0]-offset)@basis.T
                if min(other0[:,2].min(),other1[:,2].min()) < -1e-9:
                    continue
                if not cheap_floor_test(case,bases,offsets): continue
                bad=False
                for k,points in enumerate((cloud[1]@basis+offset,(cloud[0]-offset)@basis.T)):
                    if case.tasks[k].domain.mesh.ray.intersects_any(points+directions[k]*1e-7,np.tile(directions[k],(len(points),1))).any():
                        bad=True;break
                if bad:continue
                local=[v@b+o for hs,b,o in zip(case.heads,bases,offsets) for row in hs for v in row]
                for k in (0,1):
                    if not analyzers[k].test([SimpleNamespace(vertices=(v-offsets[k])@bases[k].T) for v in local],directions[k])['clear']:
                        bad=True;break
                if bad:continue
                extent=np.ptp(np.vstack([vertices[0],other0]),axis=0)
                score=float(np.linalg.norm(extent)+.3*(other0[:,2].mean()+other1[:,2].mean()))
                candidate=dict(kind='original_arrangement_transfer',bases=bases,offsets=offsets,directions=directions,
                    direction_ids=list(ids),head_checks_passed=True,compactness_score_m=score,
                    translation_adjustment_m=shift.tolist(),screened_candidates=screened)
                if any(np.linalg.norm(offset-c['offsets'][1]) < .009 and np.linalg.norm(basis-c['bases'][1]) < .05 for c in candidates): continue
                candidates.append(candidate)
                if len(candidates)>=limit:
                    yield from sorted(candidates,key=lambda c:c['compactness_score_m'])
                    return
        if candidates:
            print('Placement search:',case.pair.name,case.source.name,len(candidates),'legal from',screened,flush=True)
    yield from sorted(candidates,key=lambda c:c['compactness_score_m'])


def opposite_placements(case, limit):
    menus=[r['common_direction_ids'] for r in case.schedule['geometry']['per_pose']]
    vertices=[np.unique(np.concatenate([v for row in hs for v in row]),axis=0) for hs in case.heads]
    analyzers=[W.Analyzer(t.domain.mesh,P.DEPTH_FRACTION*t.domain.mesh.extents.max(),dict(vectors=c.tolist()))
               for t,c in zip(case.tasks,case.catalogues)]
    count=0
    for ids in itertools.product(*menus):
        directions=np.array([c[i] for c,i in zip(case.catalogues,ids)])
        if np.abs(directions[:,2]).max()>1e-9:continue
        bases=np.array([frame(directions[0]),frame(directions[1],True)])
        offsets=np.array([np.r_[-(t.domain.mesh.vertices@b).mean(0)[:2],0.] for t,b in zip(case.tasks,bases)])
        points=[v@b+o for v,b,o in zip(vertices,bases,offsets)]
        low=max(points[0][:,2].max(),-points[1][:,2].min())+.015
        for extra in (0.,.025,.05,.10):
            for sy in (0.,.04,-.04,.08,-.08,.12,-.12,.16,-.16,.20,-.20):
                offset=offsets.copy();offset[1]+=np.array([0.,sy,low+extra])
                local=[v@b+o for hs,b,o in zip(case.heads,bases,offset) for row in hs for v in row]
                good=True
                for k in (0,1):
                    other=(points[1-k]+(offset[1-k]-offsets[1-k])-offset[k])@bases[k].T
                    if case.tasks[k].domain.mesh.ray.intersects_any(other+directions[k]*1e-7,np.tile(directions[k],(len(other),1))).any():
                        good=False;break
                    if not analyzers[k].test([SimpleNamespace(vertices=(v-offset[k])@bases[k].T) for v in local],directions[k])['clear']:
                        good=False;break
                if not good:continue
                yield dict(kind='opposite_floor_local_body_fallback',bases=bases,offsets=offset,directions=directions,
                    direction_ids=list(ids),head_checks_passed=True)
                count+=1
                if count>=limit:return


def placements(case, reference, limit=12):
    yield from tilted_placements(case,reference,limit)
    if not(case.poses==reference.poses and case.schedule['particle']==9):
        yield from opposite_placements(case,limit)


def clip_floor(polygon, coefficients):
    out=[];a=np.asarray(coefficients[:2]);c=coefficients[2]
    for start,end in zip(polygon,np.roll(polygon,-1,axis=0)):
        x,y=start@a+c,end@a+c
        if x >= 0:out.append(start)
        if (x >= 0)!=(y >= 0):out.append(start+(end-start)*x/(x-y))
    return np.asarray(out).reshape(-1,2)


def foot_menu(case, placement, reference):
    if reference is not None and case.poses==reference.poses and case.schedule['particle']==9 and placement['kind']=='original_pose1_3_fixed_placement':
        parameters=json.loads((HERE/'local_body_case.json').read_text())
        yield dict(kind='original_eight_fixed_terminals',feet=parameters['feet_xy_m'],assignments=parameters['assignments'])
        return
    # Fit the old terminal layout to each new demand's bounding box. Assignment
    # is recomputed by the original distance-plus-added-volume score.
    original=json.loads((HERE/'local_body_case.json').read_text())['feet_xy_m']
    if reference is not None:
        fitted=[]
        for old,new,polygons in zip(reference.demands,case.demands,original):
            factor=np.ptp(new,axis=0)/np.ptp(old,axis=0)
            fitted.append([((np.asarray(poly)-np.mean(old,axis=0))*factor+np.mean(new,axis=0)).tolist() for poly in polygons])
        yield dict(kind='transferred_original_terminal_layout',feet=fitted,assignments=None)
    for margin in (.025,.045,.065,.085,.11,.15,.20):
        for family in ('rectangle','hull'):
            feet=[]
            for k,xy in enumerate(case.demands):
                polygon=MultiPoint(xy).convex_hull
                if family=='rectangle':polygon=polygon.minimum_rotated_rectangle
                polygon=polygon.buffer(margin,join_style=2)
                rim=np.asarray(polygon.exterior.coords)[:-1]
                b,o=placement['bases'],placement['offsets'];normal=b[1-k][2]
                coefficients=np.r_[(b[k]@normal)[:2],(o[k]-o[1-k])@normal]
                rim=clip_floor(rim,coefficients)
                pads=[]
                square=np.array([[-1.,-1],[1,-1],[1,1],[-1,1]])*.004
                for q in rim:pads.append((q+square).tolist())
                # A few elongated local terminals follow the original 8 x 40 mm
                # patches. They are not joined along the polygon perimeter.
                if family == 'hull':
                    for q,r in zip(rim,np.roll(rim,-1,axis=0)):
                        length=np.linalg.norm(r-q)
                        if length < .025: continue
                        tangent=(r-q)/length;normal=np.array([-tangent[1],tangent[0]])
                        half=min(.02,length/3)
                        center=(q+r)/2
                        pads.append(np.array([center+s*half*tangent+t*.004*normal for s,t in ((-1,-1),(1,-1),(1,1),(-1,1))]).tolist())
                feet.append(pads)
            if all(feet):yield dict(kind=f'{family}_margin_{margin:.3f}',feet=feet,assignments=None)


def publish(work,out,case,placement,parameters,attempts):
    report=json.loads((work/'report.json').read_text())
    clear_previous_outputs(out)
    for name in ('registration_checks.json','head_layout.npz','head_layout.obj','floor_conflict.png','shared_head_audit.json','equilibrium_diagnostic.npz'):
        (out/name).unlink(missing_ok=True)
    policy = report['body_design']['initial_floor_policy']
    report.update(schema=f'{policy}_floor_local_bodies_v1',physical_head_definition='five_unique_heads_one_shared_patch',
        original_method_reimplemented=policy=='opposite', reference_construction_retained=True,
        five_physical_shared_heads_claimed=True,
        placement=plain(placement),parameter_generation=parameters['kind'],attempts=attempts,
        successful_step3_particles_tried=len({r['particle'] for r in attempts}),
        body_construction_attempted=True,complete_fixture_verified=report['passed'])
    report['provenance']['code'].update(I.hashes([Path(__file__),HERE/'run_greedy.py',HERE/'fixture_view.py',HERE/'refresh_shared_geometry_view.py']))
    report['provenance']['inputs'].update(I.hashes([HERE/'local_body_case.json']))
    for name in report['artifacts']:
        shutil.copy2(work/name,out/name)
    I.save(out/'attempts.json',attempts)
    I.save(out/'report.json',report)
    data,_=json.JSONDecoder().raw_decode((work/'index.html').read_text().split('const DATA=',1)[1])
    data['report']=report
    if not report['passed']:
        data['report']['presentation_description']='已连接的一件实体，但未通过完整验收。'+report['presentation_description']
    write_viewer(out,data)
    return report


def run_pair(pair,limit=12,only_particle=None,floor_policy='nearest'):
    out=pair/'step5';out.mkdir(parents=True,exist_ok=True)
    reference=original_case()
    sources=[p.parent for p in sorted(pair.glob('step3_scheculer/sequential_3plus2/from_*/terminal_expansion/particle_*/schedule.json')) if json.loads(p.read_text())['passed']]
    if only_particle is not None:sources=[s for s in sources if int(s.name.split('_')[-1])==only_particle]
    sources.sort(key=lambda s:(not(pair.name in ('pose1+3','pose1+6') and s.name=='particle_009'),s.name))
    attempts=[];best=None;best_score=None;began=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='cadgrasp_local_batch_') as temporary:
        stage=Path(temporary);bestdir=stage/'best'
        for source in sources:
            case=read_case(source)
            print('CASE',pair.name,source.name,flush=True)
            found=False
            for placement_id,placement in enumerate(placements(case,reference,limit)):
                found=True;work=stage/f'{source.name}_{placement_id}';work.mkdir()
                print('PLACEMENT',placement_id,placement['kind'],flush=True)
                for parameters in foot_menu(case,placement,reference):
                    row=dict(particle=case.schedule['particle'],placement_index=placement_id,placement_kind=placement['kind'],parameters=parameters['kind'],passed=False)
                    attempts.append(row)
                    try:
                        artifacts=L.build(work,parameters['feet'],parameters['assignments'],case=case,placement=placement,allow_failed=True,
                            skip_unreachable=parameters['kind']!='original_eight_fixed_terminals', floor_policy=floor_policy,
                            verify=True)
                        report=json.loads((work/'report.json').read_text())
                        row.update(passed=report['passed'],connected=report['solid']['one_solid'],volume_cm3=report['volume_cm3'],
                            checks=report['checks'])
                        print('CANDIDATE',pair.name,row['parameters'],'PASS' if row['passed'] else 'CONNECTED / FAIL',report['volume_cm3'],flush=True)
                        if report['passed']:
                            result=publish(work,out,case,placement,parameters,attempts)
                            print('RESULT',pair.name,'PASS',result['volume_cm3'],'cm3',round(time.monotonic()-began,1),'s',flush=True)
                            return result
                        score=(sum(c['coupled_equilibrium_passed'] for c in report['checks']),min(c['verified_sample_count'] for c in report['checks']),-report['volume_cm3'])
                        if best_score is None or score>best_score:
                            best_score=score
                            if bestdir.exists():shutil.rmtree(bestdir)
                            bestdir.mkdir()
                            for name in artifacts:shutil.copy2(work/name,bestdir/name)
                            best=(case,placement,parameters)
                    except (RuntimeError,ValueError) as error:
                        row['reason']=str(error)
                        print('CANDIDATE',pair.name,row['parameters'],'REJECT',str(error),flush=True)
                    I.save(out/'local_body_attempts.json',plain(attempts))
                    if 'initial local head body' in row.get('reason',''):
                        break
            if not found:attempts.append(dict(particle=case.schedule['particle'],passed=False,reason='No legal transferred compact placement in bounded search'))
        if best is not None:
            result=publish(bestdir,out,*best,attempts)
            print('RESULT',pair.name,'CONNECTED BUT FAILED',result['volume_cm3'],flush=True)
            return result
    report=dict(schema=f'{floor_policy}_floor_local_bodies_v1',object=pair.parent.name,poses=[f'pose_{x}' for x in pair.name[4:].split('+')],passed=False,complete=True,
        status='no_connected_body_in_bounded_search',physical_head_definition='five_unique_heads_one_shared_patch',
        attempts=attempts,successful_step3_particles_tried=len(sources),body_construction_attempted=any('parameters' in a for a in attempts),
        finite_search_failure_not_impossibility=True)
    clear_previous_outputs(out);I.save(out/'report.json',report);I.save(out/'attempts.json',attempts)
    print('RESULT',pair.name,'NO CONNECTED BODY',flush=True)
    return report


class Tee:
    def __init__(self,*streams):self.streams=streams
    def write(self,value):
        for stream in self.streams:stream.write(value)
        return len(value)
    def flush(self):
        for stream in self.streams:stream.flush()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object',default='B');parser.add_argument('--pair',nargs='*')
    parser.add_argument('--placements',type=int,default=12);parser.add_argument('--particle',type=int)
    parser.add_argument('--floor-policy',choices=('nearest','opposite'),default='nearest',
                        help='Initial body target; opposite exactly replays the reference rule')
    parser.add_argument('--search', action='store_true', help='Use the historical placement/terminal search with full verification')
    parser.add_argument('--verify', action='store_true', help='Opt in to full verification in fixed-layout mode')
    parser.add_argument('--repeat', type=int, default=1, help='Repeat fixed-layout generation to measure cache reuse')
    args=parser.parse_args()
    pairs=[OUTPUTS/args.object/p for p in args.pair] if args.pair else sorted((OUTPUTS/args.object).glob('pose*+*'))
    if not args.search:
        if args.floor_policy != 'nearest' or args.particle is not None:
            parser.error('Use --search to change the saved particle or replay opposite-floor search')
        if args.repeat < 1:
            parser.error('--repeat must be positive')
        from step5_connect_support.run_fast_local_bodies import run_pair as fixed_run, publish_batch, publish_failure, failure_record
        results=[]
        for pair in pairs:
            try:
                result=fixed_run(pair,args.repeat,args.verify)
            except Exception as error:
                result=failure_record(pair,error)
                publish_failure(pair,result)
            results.append(result)
            print(json.dumps(result),flush=True)
        publish_batch(results,pairs[0]/'step5/data')
        return 0 if all(r['constructed'] for r in results) else 1
    results=[]
    for pair in pairs:
        out=pair/'step5';out.mkdir(parents=True,exist_ok=True)
        with (out/'run.log').open('w') as log,redirect_stdout(Tee(sys.stdout,log)):
            result=run_pair(pair,args.placements,args.particle,args.floor_policy);results.append(dict(pair=pair.name,passed=result['passed']))
    print(json.dumps(results,indent=2))
    return 0 if all(r['passed'] for r in results) else 1


if __name__=='__main__':raise SystemExit(main())
