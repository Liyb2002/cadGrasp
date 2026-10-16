"""Connected hull wrapping followed by deletion of the SAME actual solid.

The hull perimeter is construction stock, never mandatory retained material.
No contact selection or second support reconstruction precedes deletion.
"""
from pathlib import Path
import importlib.util
import itertools
import json
import sys
import time
from types import SimpleNamespace

from feasible_geometry import C, ForceCertificate, Layout, material_components
from deterministic_base import solid, unpack, volume, union, box, moved, enclosure, SCALE, transform, Builder as GroundBuilder
from continuous_legs import capsule
from wrap_routes import FiniteRouteGuide as RouteGuide
from exact_mesh import save_mesh
from wrap_dimensions import (INITIAL_FOOT_WIDTH_M, INITIAL_FOOT_HEIGHT_M,
                             CONNECTOR_RADIUS_M, MINIMUM_WIDTH_M, PREVIOUS_FOOT_WIDTH_M, physical_component_count)
import manifold3d as md
import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

GROUND_MARGIN_SCALE=1.
TERMINAL_FOOT_WIDTH_M=MINIMUM_WIDTH_M


class WrapInputs(C.Inputs):
    """Read Step4 directly: no pre-existing Step5.1 result is required."""
    def load(self,case):
        if not case['passed']:raise ValueError('Step4 did not pass')
        selected=self.base/case['id']/'step4/step4.2'/case['experiment']
        report_path=selected/'data/report.json';record=json.loads(report_path.read_text())
        if not record['complete']:raise ValueError('Incomplete selected Step4')
        for path in [report_path,selected/'layout.npz',selected/'support.obj']:
            self.protected[C.relative(path)]=C.digest(path)
        fixture=C.trimesh.load(selected/'support.obj',force='mesh',process=False)
        if abs(abs(float(fixture.volume))*1e6-case['volume_cm3'])>1e-6:
            raise ValueError('Selected Step4 core volume mismatch')
        with np.load(selected/'layout.npz') as z:
            native=z['native_world'].copy();placements=z['placements'].copy()
            hosts=z['hosts'].copy();directions=z['directions'].copy();active=z['active'].copy()
        poses=record['poses']
        if len(poses)!=case['pose_count'] or not np.array_equal(np.sort(active),np.arange(len(poses))):
            raise ValueError('Every original pose must remain active')
        rows=[]
        for k,pose in enumerate(poses):
            T,U=C.F.final_task_frame(native,placements,hosts,k)
            world=C.F.transform_points(self.mesh.vertices,U)
            rows.append(dict(pose=pose,color=C.V.COLORS[k%len(C.V.COLORS)],
                fixture_transform=T,object_transform=U,
                object_ground=world[np.abs(world[:,2])<=1e-9,:2],direction=directions[k],
                world_exit_direction=T[:3,:3]@directions[k],common_exit_direction=directions[k]))
        return dict(case=case,record=record,fixture=fixture,body=self.mesh,rows=rows,
            states=[r['fixture_transform'] for r in rows],selected=selected,
            bounds=C.occupied_bounds(self.mesh,fixture,rows))


def original_demands(inputs, data, out):
    """Recompute every pressure point from unchanged native paired demands."""
    path = C.COOPT/'step5.1/run.py'
    spec = importlib.util.spec_from_file_location('wrap_native_inputs', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    reader = module.NativeInputs(inputs.name, data['body'])
    with np.load(data['selected']/'layout.npz') as z:
        native = z['native_world'].copy()
        original = z['native_world_original'].copy()
        placements = z['placements'].copy(); hosts = z['hosts'].copy()
    records = []
    for k, row in enumerate(data['rows']):
        task = reader.get(row['pose'])
        np.testing.assert_allclose(original[k], task['setup']['T_world_mesh'], atol=1e-12, rtol=0)
        T, U = C.F.final_task_frame(native, placements, hosts, k)
        com = task['setup']['com_m'] + U[:3, 3] - original[k, :3, 3]
        result = C.F.system_floor_demands(task['loads'], com)
        cloud = result['floor_demands_world_m']
        if 'cloud' in row:np.testing.assert_allclose(cloud,row['cloud'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(T, row['fixture_transform'], atol=1e-12, rtol=0)
        hull = ConvexHull(cloud[:, :2]).vertices
        row.update(cloud=cloud, hull=cloud[hull], fixture_transform=T, object_transform=U)
        target=None
        if out is not None:
            target=out/'demands'/(row['pose']+'.npz');target.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(target,**result,hull_indices=hull,
                T_world_fixture=T,T_world_object=U,object_com_world_m=com)
        for source in task['sources']:
            inputs.protected[source['path']] = source['sha256']
        records.append(dict(pose=row['pose'],count=len(cloud),arrays=C.relative(target) if target else None,
                            sha256=C.digest(target) if target else None,native_sources=task['sources']))
    return records


class Validator:
    """Original tolerances, cached immutable geometry, actual serialized mesh."""
    def __init__(self, data, layout, witness, name):
        self.data = data; self.rows = data['rows']; self.witness = solid(witness)
        body = data['body'].copy(); body.merge_vertices(digits_vertex=15)
        self.objects = union([moved(solid(body), Q) for Q in layout.placements[list(layout.active)]])
        self.checker = ForceCertificate(data['record']['poses'], layout, data['body'], self.rows, name)
        self.width_guard = None

    def check(self, mesh, previous=None, bounds=None, *, forces=True, ground=True, width=True, connected=True):
        if not np.isfinite(mesh.vertices).all() or not mesh.is_watertight or not mesh.is_winding_consistent:
            raise ValueError('Nonfinite, open or inconsistent actual boundary')
        value = solid(mesh)
        cover = enclosure(mesh, self.rows)
        actual_bounds = C.occupied_bounds(self.data['body'], mesh, self.rows)
        checks = dict(components=physical_component_count(value),
            material_cm3=abs(float(mesh.volume))*1e6,
            outside_m3=volume(value-self.witness), object_overlap_m3=volume(value ^ self.objects),
            added_m3=volume(value-previous) if previous is not None else 0.,
            bbx_l=C.volume(actual_bounds)*1000, bounds_world_m=actual_bounds.tolist(),
            bbx_expansion_m=0. if bounds is None else float(max(0.,
                np.max(bounds[0]-actual_bounds[0]), np.max(actual_bounds[1]-bounds[1]))),
            floor_passed=all(r['passed'] for r in cover), pressure_enclosure=cover)
        if ((connected and checks['components'] != 1) or checks['outside_m3'] > 1e-10
                or checks['object_overlap_m3'] > 1e-10 or checks['added_m3'] > 1e-10
                or checks['bbx_expansion_m'] > 1e-8 or (ground and not checks['floor_passed'])
                or (bounds is not None and checks['bbx_l'] > C.volume(bounds)*1000+1e-8)):
            raise ValueError('Original solid guards: '+str({k:v for k,v in checks.items()
                if k not in ['pressure_enclosure','bounds_world_m']}))
        if not forces:
            checks.update(passed=True, watertight=True, winding_consistent=True,
                          serialized_geometry_checked=True, full_force_check=False)
            return value, None, checks
        proof = self.checker.evaluate_candidate(mesh)
        if not proof['passed']:
            raise ValueError('Original load guard: '+str(proof['counts']))
        if len(proof['masks']) != len(self.rows) or any(len(v)!=32768 or not v.all() for v in proof['masks'].values()):
            raise ValueError('Incomplete original demand certificate')
        checks.update(passed=True, watertight=True, winding_consistent=True,
                      serialized_geometry_checked=True, counts=proof['counts'])
        return value, proof, checks


def rim(row, width=INITIAL_FOOT_WIDTH_M, height=INITIAL_FOOT_HEIGHT_M):
    hull=Polygon(row['hull'][:,:2])
    outward=min(PREVIOUS_FOOT_WIDTH_M,width)
    outer=orient(hull.buffer(outward,join_style=2),sign=1.)
    inner=orient(hull.buffer(-(width-outward),join_style=2),sign=1.)
    contours=[np.asarray(outer.exterior.coords)[:-1]/SCALE]
    if not inner.is_empty:contours.append(np.asarray(inner.exterior.coords)[:-1][::-1]/SCALE)
    value = md.CrossSection(contours, md.FillRule.NonZero).extrude(height/SCALE)
    return moved(value, np.linalg.inv(row['fixture_transform']))


def construct(data, baseline, validator, out, completion=None, reference=None):
    """Start from the whole Step4 core; add legal bands and connecting stock."""
    # Use Manifold's own current tolerance to remove redundant subdivisions.
    # The saved stock is fully recertified below; no acceptance tolerance changes.
    start = solid(baseline).simplify(0.)
    bands = [rim(row) for row in data['rows']]
    points = np.vstack([baseline.vertices, *[unpack(b).vertices for b in bands]])
    domain = validator.witness ^ box(points.min(0)-.012, points.max(0)+.012)
    domain = domain-validator.objects
    stock = start
    history = []
    def anchors(value):
        mesh=unpack(value)
        return np.vstack([mesh.vertices,mesh.triangles.mean(axis=1)])
    def connecting_parts(link,left,right):
        return union([piece for piece in material_components(link)
            if physical_component_count(left+piece)==1 and physical_component_count(right+piece)==1])
    # Every saved construction stage is the actual union, before any pruning.
    save_mesh(baseline, out/'construction/000.obj')
    # Some historical Step4 inputs contain tiny separate positive pieces that
    # its old component cutoff ignored. Preserve and actually join them; that
    # cutoff is never used for a saved complete stock or deletion checkpoint.
    for iteration in range(64):
        parts=[p for p in stock.decompose() if float(p.volume())>0.]
        if len(parts)==1:break
        main=max(parts,key=lambda p:float(p.volume()))
        av=unpack(main).vertices;tree=cKDTree(av);proposals=[]
        for part in parts:
            if part is main:continue
            bv=unpack(part).vertices;distances,ids=tree.query(bv)
            j=int(np.argmin(distances));proposals.append((float(distances[j]),bv[j],av[ids[j]]))
        joined=False
        for distance,a,b in sorted(proposals,key=lambda q:q[0]):
            link=capsule(a,b,CONNECTOR_RADIUS_M)^domain
            proposed=stock+link
            if physical_component_count(proposed)>=len(parts):continue
            m=save_mesh(unpack(proposed.simplify(0.)),out/'candidate.obj')
            actual,_,accepted=validator.check(m,forces=False,ground=False,width=False,connected=False)
            if accepted['components']>=len(parts):continue
            if volume(stock-actual)>1e-10:continue
            stock=actual
            history.append(dict(operation='join_original_core_parts',distance_m=distance,
                components_before=len(parts),components_after=accepted['components'],material_cm3=accepted['material_cm3']))
            joined=True;break
        if not joined:raise ValueError('Original separate core pieces cannot be joined by the finite legal connector proposals')
    if physical_component_count(stock)!=1:raise ValueError('Original core connection did not finish')
    guide = None
    for k, (row, band) in enumerate(zip(data['rows'], bands)):
        legal = band ^ domain
        pieces = material_components(legal)
        details = []
        for part in pieces:
            print('WRAP_PART',row['pose'],round(volume(part)*1e6,4),flush=True)
            if volume(part)<2e-8:
                details.append(dict(accepted=False,reason='Below 0.02 cm3 construction fragment resolution',
                                    part_cm3=volume(part)*1e6))
                continue
            if volume(part-stock) <= 1e-13:
                continue
            candidate = stock+part
            route_record = None
            accepted_links=[]
            if physical_component_count(candidate) != 1:
                print('CONNECT',row['pose'],flush=True)
                av,bv=anchors(part),anchors(stock)
                distances, ids = cKDTree(bv).query(av)
                connected = False
                for j in np.argsort(distances)[:12]:
                    rod = capsule(av[j], bv[ids[j]], CONNECTOR_RADIUS_M) ^ domain
                    rod=connecting_parts(rod,stock,part)
                    proposed = stock+part+rod
                    if physical_component_count(proposed) == 1:
                        candidate = proposed; connected = True
                        route_record = dict(kind='short_round_link', length_m=float(distances[j]), radius_m=CONNECTOR_RADIUS_M,
                            path_m=[av[j].tolist(),bv[ids[j]].tolist()])
                        accepted_links=[(rod,av[j],bv[ids[j]])]
                        break
                if not connected:
                    if guide is None:
                        guide = RouteGuide(domain, stock, pitch=.002)
                    try:
                        print('ROUTE',row['pose'],flush=True)
                        target=unpack(part).triangles[np.argmax(unpack(part).area_faces)].mean(axis=0)
                        route, stats = guide.route(target)
                        rods = union([capsule(a,b,CONNECTOR_RADIUS_M) for a,b in zip(route[:-1], route[1:])]) ^ domain
                        proposed = stock+part+rods
                        if physical_component_count(proposed) != 1:
                            raise ValueError('Clipped routed link does not connect material')
                        candidate = proposed; route_record = dict(kind='legal_round_route',path_m=route.tolist(), **stats)
                        accepted_links=[(capsule(a,b,CONNECTOR_RADIUS_M)^domain,a,b) for a,b in zip(route[:-1],route[1:])]
                    except (ValueError, RuntimeError) as error:
                        details.append(dict(accepted=False, reason=str(error), part_cm3=volume(part)*1e6))
                        continue
            # Union can expose tiny source boundary defects; accept only after
            # exactly the same physical guards as a final saved support.
            path = out/'candidate.obj'
            mesh = save_mesh(unpack(candidate.simplify(0.)), path)
            print('WRAP_CHECK',row['pose'],len(mesh.faces),flush=True)
            try:
                if volume(stock-solid(mesh))>1e-10:
                    raise ValueError('Construction union lost previously grown material')
                actual, proof, checks = validator.check(mesh, forces=False, ground=False, width=False)
            except (ValueError, RuntimeError) as error:
                details.append(dict(accepted=False, reason=str(error), part_cm3=volume(part)*1e6))
                continue
            stock = actual
            details.append(dict(accepted=True, link=route_record, material_cm3=checks['material_cm3']))
        mesh = save_mesh(unpack(stock), out/f'construction/{k+1:03d}.obj')
        # These may repeat an earlier solid when a rim part is illegal or unreachable.
        history.append(dict(pose=row['pose'], proposed_rim_cm3=volume(band)*1e6,
            legal_rim_cm3=volume(legal)*1e6, pieces=details,
            mesh=C.relative(out/f'construction/{k+1:03d}.obj')))
        print('WRAP',row['pose'],len(details),round(volume(stock)*1e6,3),flush=True)
    # Another pose's forbidden space can remove a hull-edge segment. Extend
    # the legal ground contacts just enough to restore convex-hull coverage,
    # then connect those finite patches to the already grown support.
    if not all(r['passed'] for r in enclosure(unpack(stock),data['rows'])):
        ground_domain=(validator.witness ^ box(points.min(0)-.05,points.max(0)+.05))-validator.objects
        from wrap_ground_stock import prepare
        builder,patches=prepare(stock,ground_domain,data['rows'],GROUND_MARGIN_SCALE,TERMINAL_FOOT_WIDTH_M)
        ground_guide=None
        for part in patches:
            if volume(part-stock)<=1e-13:continue
            candidate=stock+part
            accepted_links=[]
            if physical_component_count(candidate)!=1:
                av,bv=anchors(part),anchors(stock)
                distances,ids=cKDTree(bv).query(av)
                for j in np.argsort(distances)[:24]:
                    link=connecting_parts(capsule(av[j],bv[ids[j]],CONNECTOR_RADIUS_M)^ground_domain,stock,part)
                    candidate=stock+part+link
                    if physical_component_count(candidate)==1:
                        accepted_links=[(link,av[j],bv[ids[j]])]
                        break
                else:
                    if ground_guide is None:ground_guide=RouteGuide(ground_domain,stock,pitch=.002)
                    target=unpack(part).center_mass
                    try:
                        route,stats=ground_guide.route(target)
                        rods=union([capsule(a,b,CONNECTOR_RADIUS_M) for a,b in zip(route[:-1],route[1:])])^ground_domain
                        candidate=stock+part+rods
                        accepted_links=[(capsule(a,b,CONNECTOR_RADIUS_M)^ground_domain,a,b) for a,b in zip(route[:-1],route[1:])]
                    except (ValueError,RuntimeError) as error:
                        candidate=stock+part
                        history.append(dict(operation='ground_route_proposal_rejected',reason=str(error)))
            if physical_component_count(candidate)!=1:
                # A guide path is only a proposal: clipping can break its rod.
                # Grow an actual local legal bridge before declaring connection.
                connector=GroundBuilder(unpack(candidate),unpack(ground_domain),data['rows'],width=MINIMUM_WIDTH_M)
                connector.body=candidate
                connector.connect()
                candidate=connector.body
                history.append(dict(operation='ground_patch_local_bridges',history=connector.history))
            try:
                m=save_mesh(unpack(candidate.simplify(0.)),out/'candidate.obj')
                actual,_,checks=validator.check(m,forces=False,ground=False,width=False)
                if volume(stock-actual)>1e-10:raise ValueError('Ground extension lost existing material')
            except (ValueError,RuntimeError) as error:
                history.append(dict(operation='ground_extension_rejected',reason=str(error)))
                continue
            stock=actual
        history.append(dict(operation='legal_ground_extensions',history=builder.history,
                            material_cm3=volume(stock)*1e6))
        save_mesh(unpack(stock),out/f'construction/{len(data["rows"])+1:03d}.obj')
        history[-1]['mesh']=C.relative(out/f'construction/{len(data["rows"])+1:03d}.obj')
    mesh = save_mesh(unpack(stock), out/'initial_support.obj')
    try:
        value, proof, checks = validator.check(mesh,width=False)
    except (ValueError,RuntimeError) as error:
        history.append(dict(operation='complete_legal_ground_coverage',initial_wrap_rejection=str(error)))
        # A pressure-hull edge can pass through another pose's forbidden
        # space. In that case grow the saved Step4 ground scaffold as well.
        # The ENTIRE core and every accepted band remain in this union.
        last_error=error
        for label,extra in [('step4_ground_scaffold',completion),('same_layout_complete_reference',reference)]:
            if extra is None:continue
            extra_value=solid(extra)
            if volume(extra_value-validator.witness)>1e-10:extra_value=extra_value ^ validator.witness
            if volume(extra_value ^ validator.objects)>1e-10:extra_value=extra_value-validator.objects
            candidate=(stock+extra_value).simplify(0.)
            candidate_mesh=save_mesh(unpack(candidate),out/'candidate.obj')
            try:
                value,proof,checks=validator.check(candidate_mesh,width=False)
                if volume(stock-value)>1e-10:raise ValueError('Construction lost previously grown stock')
            except (ValueError,RuntimeError) as failed:
                last_error=failed;history.append(dict(operation=label,accepted=False,reason=str(failed)))
                continue
            mesh=candidate_mesh;stock=value
            save_mesh(mesh,out/f'construction/{len(data["rows"])+2:03d}.obj')
            history.append(dict(operation=label,accepted=True,material_cm3=checks['material_cm3'],
                mesh=C.relative(out/f'construction/{len(data["rows"])+2:03d}.obj')))
            break
        else:raise ValueError('Connected hull wrapping could not be certified: '+str(last_error))
        mesh=save_mesh(mesh,out/'initial_support.obj')
    return mesh, value, proof, checks, history


class Deleter:
    def __init__(self, mesh, value, proof, checks, validator, out):
        self.mesh=mesh; self.value=value; self.proof=proof; self.checks=checks
        self.validator=validator; self.out=out; self.history=[]; self.states=[]
        self.initial=value; self.mask=md.Manifold()
        from wrap_dimensions import RETAINED_VOLUME_FRACTION
        self.minimum_retained_volume=volume(value)*RETAINED_VOLUME_FRACTION
        self.save('connected_wrapped_stock', {})

    def save(self, operation, detail):
        index=len(self.states); path=self.out/f'checkpoints/{index:04d}.obj'
        mesh=save_mesh(self.mesh,path)
        # save_mesh is bit-exact; record and check bytes rather than claiming
        # that a different visualization mesh has the certificate.
        if not np.array_equal(mesh.vertices,self.mesh.vertices) or not np.array_equal(mesh.faces,self.mesh.faces):
            raise ValueError('Exact serialization changed checkpoint geometry')
        maskpath=path.with_suffix('.npz')
        np.savez_compressed(maskpath, **{f'pose_{k}':v for k,v in self.proof['masks'].items()})
        row=dict(index=index, operation=operation, detail=detail, mesh=C.relative(path),
                 mesh_sha256=C.digest(path), force_masks=C.relative(maskpath),
                 force_masks_sha256=C.digest(maskpath), **self.checks)
        self.states.append(row)
        C.save(self.out/'states.json',dict(complete=False,states=self.states))
        print('ACCEPT',index,operation,round(row['material_cm3'],3),round(row['bbx_l'],6),flush=True)

    def attempt(self, candidate, operation, detail, mask=None):
        before=volume(self.value)
        rebase_mask=False
        if not candidate.is_empty():
            parts=[part for part in candidate.decompose() if float(part.volume())>0.]
            if len(parts)>1:
                main=max(range(len(parts)),key=lambda i:float(parts[i].volume()))
                discarded=union([part for i,part in enumerate(parts) if i!=main])
                # Propose one larger deletion of the cut plus dependent
                # branches. Keep cavities by subtracting the discarded parts,
                # never by rebuilding a filled outer shell. The final one-piece
                # proposal still needs every original physical/load guard.
                candidate=candidate-discarded
                if mask is not None:rebase_mask=True
                operation+='_and_dependent_branches'
                detail=dict(detail,proposed_components=len(parts),
                    dependent_material_removed_cm3=volume(discarded)*1e6)
        if candidate.is_empty() or before-volume(candidate)<1e-10:
            return False
        record=dict(operation=operation,detail=detail,accepted=False)
        began=time.monotonic()
        try:
            if volume(candidate)<self.minimum_retained_volume:
                raise ValueError('Conservative pruning material budget reached')
            if physical_component_count(candidate) != 1:
                raise ValueError('Deletion disconnects actual material')
            mesh=save_mesh(unpack(candidate.simplify(0.)),self.out/'candidate.obj')
            value,proof,checks=self.validator.check(mesh,self.value,np.asarray(self.checks['bounds_world_m']))
            self.mesh,self.value,self.proof,self.checks=mesh,value,proof,checks
            if rebase_mask:
                # Continue from the accepted actual solid. Unioning discarded
                # mesh components into a box mask creates nearly coincident
                # surfaces and can make later Boolean operations explode.
                self.initial=value;self.mask=md.Manifold()
            elif mask is not None:self.mask=mask
            record.update(accepted=True,removed_cm3=(before-volume(value))*1e6,state=len(self.states))
            self.save(operation,detail)
        except (ValueError,RuntimeError) as error:
            record['reason']=str(error)
        record['seconds']=time.monotonic()-began
        self.history.append(record)
        if record['accepted'] or len(self.history)%25==0:
            C.save(self.out/'deletions_progress.json',dict(complete=False,attempts=self.history))
        if not record['accepted'] and operation.startswith('remove_narrow'):
            print('REJECT',operation,detail.get('feature',detail.get('features')),record.get('reason','No change'),flush=True)
        return record['accepted']

    def run(self, grids=(3,5)):
        rows=self.validator.rows
        frames=list({np.round(r['fixture_transform'],11).tobytes():r['fixture_transform'] for r in rows}.values())
        objects=np.vstack([transform(self.validator.data['body'].vertices,r['object_transform']) for r in rows])
        target=np.array([objects.min(0),objects.max(0)])
        # Exterior global BBX faces first, then all cells at each declared
        # resolution. The finite proposal family is exhausted; no time cutoff.
        for fraction in (1.,.5,.25,.125):
            bounds=np.asarray(self.checks['bounds_world_m'])
            proposals=[]
            for axis in range(3):
                for side in range(2):
                    gap=(bounds[side,axis]-target[side,axis])*(1 if side else -1)
                    if gap>1e-8:proposals.append((-gap,axis,side))
            for _,axis,side in sorted(proposals):
                bounds=np.asarray(self.checks['bounds_world_m'])
                edge=bounds[side,axis];limit=edge+fraction*(target[side,axis]-edge)
                candidate=self.value
                for T in frames:
                    normal=-T[axis,:3] if side else T[axis,:3]
                    offset=(T[axis,3]-limit if side else limit-T[axis,3])/SCALE
                    candidate=candidate.trim_by_plane(tuple(normal),float(offset))
                self.attempt(candidate,'outside_bbx_face',dict(axis=axis,side=side,fraction=fraction))
        # Start a single cumulative subtraction mask from the BBX-trimmed
        # solid, preventing successive coplanar subdivisions of the stock.
        self.initial=self.value;self.mask=md.Manifold()
        low,high=self.mesh.bounds;center=(low+high)/2;extent=np.maximum((high-low)/2,1e-9)
        for grid in grids:
            cells=[];width=(high-low)/grid
            for index in itertools.product(range(grid),repeat=3):
                a=low+np.array(index)*width;b=a+width
                distance=float(np.linalg.norm(((a+b)/2-center)/extent))
                cells.append((-distance,index,a,b))
            for _,index,a,b in sorted(cells,key=lambda x:(x[0],x[1])):
                cut=box(a-1e-12,b+1e-12)
                if volume(self.value ^ cut)<1e-10:continue
                mask=self.mask+cut
                self.attempt(self.initial-mask,'outside_cell',dict(grid=grid,cell=list(index)),mask)
        final=save_mesh(self.mesh,self.out/'support.obj')
        self.value,self.proof,self.checks=self.validator.check(final,self.value,np.asarray(self.checks['bounds_world_m']))
        self.mesh=final
        C.save(self.out/'states.json',dict(complete=True,states=self.states,
            all_states_original_full_demands=True,ring_perimeter_protected=False,
            proposal_grids=list(grids),wall_clock_cutoff_used=False))
        C.save(self.out/'deletions.json',dict(complete=True,attempts=self.history))
