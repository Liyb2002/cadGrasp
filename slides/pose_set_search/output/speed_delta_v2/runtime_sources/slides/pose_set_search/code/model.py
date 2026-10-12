"""Full SE(3) configuration sharing; contact-only guidance, exact solids last."""
from dataclasses import dataclass
import hashlib
import json
import time
import subprocess
import tempfile
from collections import OrderedDict

from common import *
from common import _PRISM_FACES
from classify import classify
from projection import cone_projection


@dataclass
class Layout:
    placements: np.ndarray
    directions: np.ndarray
    hosts: np.ndarray
    active: tuple

    def copy(self):
        return Layout(self.placements.copy(), self.directions.copy(), self.hosts.copy(), self.active)

    def key(self):
        return (self.active, tuple(self.hosts[list(self.active)]),
                np.round(self.placements[list(self.active)], 11).tobytes(),
                np.round(self.directions[list(self.active)], 11).tobytes())


class Model:
    def __init__(self, poses, name='B', thickness=.005):
        began = time.monotonic()
        self.name, self.poses = name, poses
        self.states = [C.state(name, p) for p in poses]
        self.tasks = [s[0] for s in self.states]
        self.native = np.array([s[1] for s in self.states])
        self.mesh = self.states[0][2]
        for _, _, mesh in self.states:
            assert np.array_equal(mesh.faces, self.mesh.faces)
            assert np.allclose(mesh.vertices, self.mesh.vertices, atol=1e-12, rtol=0)
        self.extent = float(self.mesh.extents.max())
        self.length = max(.5, 4*self.extent)
        self.thickness = thickness
        self.body = C.S.solid(self.mesh)
        self.clearance = ConservativeExitClearance(self.mesh)
        offsets = C.wrap_offsets(self.mesh, thickness)
        cells = [C.S.solid(C.G.hull_mesh(C.G.head_cell(self.mesh, tri, i, offsets)))
                 for i, tri in enumerate(self.mesh.triangles)]
        self.wraps, self.works, self.allowed, self.work_rays, self.wrap_rays = [], [], [], [], []
        for task in self.tasks:
            ids = task.domain.work_ids
            allowed = np.setdiff1d(np.arange(len(self.mesh.faces)), ids)
            self.allowed.append(allowed)
            self.wraps.append(C.union([cells[i] for i in allowed])-self.body-C.union([cells[i] for i in ids]))
            exclusion = C.union([C.S.solid(C.G.hull_mesh(np.vstack([
                self.mesh.triangles[i]-1e-6*self.mesh.face_normals[i],
                self.mesh.triangles[i]+thickness*self.mesh.face_normals[i]]))) for i in ids])
            self.works.append(exclusion)
            self.work_rays.append(RayMeshIntersector(unpack_solid(exclusion)))
            self.wrap_rays.append(RayMeshIntersector(unpack_solid(self.wraps[-1])))
        self.ray = RayMeshIntersector(self.mesh)
        # Quadrature is proposal guidance, never a force-capacity model or
        # final contact certificate. Exact contact vertices are used finally.
        bary = np.array([[2/3,1/6,1/6], [1/6,2/3,1/6], [1/6,1/6,2/3], [1/3,1/3,1/3]])
        self.points = np.einsum('av,tvc->tac', bary, self.mesh.triangles).reshape(-1,3)
        self.sources = np.repeat(np.arange(len(self.mesh.faces)), 4)
        self.point_normals = self.mesh.face_normals[self.sources]
        self.point_areas = np.repeat(self.mesh.area_faces/4, 4)
        self.epsilon = self.extent*1e-7
        self.point_rays, self.floors = [], []
        for task, transform, _ in self.states:
            points = C.transform_points(self.points, transform)
            normal = -task.domain.mesh.face_normals[self.sources]
            self.point_rays.append(C.U.heads(np.c_[normal, np.cross(points-task.domain.com, normal)], task.scale))
            self.floors.append(C.U.floor(C.FLOOR.columns(task.floor, task.domain.com), task.scale))
        self.inputs = sorted(set(p for task in self.tasks for p in task.inputs))
        self.exact_cache, self.lock_cache, self.seed_cache = OrderedDict(), OrderedDict(), OrderedDict()
        self.fixed_lock_cache = OrderedDict()
        self.exact_calls = self.proxy_calls = 0
        self.unresolved_layouts = {}
        self.startup_sources = C.provenance([], [p for p in Path(__file__).parent.glob('*.py')])['code']
        print('MODEL', len(poses), 'poses; prepared in', round(time.monotonic()-began, 2), 'seconds', flush=True)

    def floor_normal(self, layout, k):
        return self.native[layout.hosts[k], :3, :3].T @ np.array([0.,0.,1.])

    def initial(self, active=None):
        active = tuple(range(len(self.poses))) if active is None else tuple(active)
        normals = np.array([self.native[k,:3,:3].T @ [0.,0.,1.] for k in active])
        directions, info = initialize_close_directions(normals)
        layout = Layout(np.repeat(np.eye(4)[None],len(self.poses),axis=0),
                        np.zeros((len(self.poses),3)), np.arange(len(self.poses)), active)
        for k, d, n in zip(active, directions, normals):
            layout.directions[k] = legal_direction(d, n)
        return layout, info

    def juxtapose(self, layout, guest, host, translation=None):
        result = layout.copy()
        host_native = layout.hosts[host]
        result.hosts[guest] = host_native
        result.placements[guest] = np.linalg.inv(self.native[host_native]) @ self.native[guest]
        if translation is not None:
            result.placements[guest,:3,3] += translation
        result.directions[guest] = legal_direction(layout.directions[host], self.floor_normal(result,guest))
        return result

    def locks(self, layout, owner, blocker):
        relative = np.linalg.inv(layout.placements[blocker]) @ layout.placements[owner]
        d = layout.placements[blocker,:3,:3].T @ layout.directions[blocker]
        key = (blocker, np.round(relative,11).tobytes(),np.round(d,11).tobytes())
        if key in self.lock_cache:
            self.lock_cache.move_to_end(key)
            return self.lock_cache[key]
        # Work points use unshifted surface positions; body/exit rays start
        # just outside the owner's surface to avoid boundary self-hits.
        surface = C.transform_points(self.points, relative)
        outward = self.point_normals @ relative[:3,:3].T
        origins = surface+self.epsilon*outward
        fixed_key=(blocker,np.round(relative,11).tobytes())
        if fixed_key not in self.fixed_lock_cache:
            fixed=self.ray.contains_points(origins)
            fixed |= self.work_rays[blocker].contains_points(surface)
            self.fixed_lock_cache[fixed_key]=fixed
            if len(self.fixed_lock_cache)>512:
                self.fixed_lock_cache.popitem(last=False)
        locked=self.fixed_lock_cache[fixed_key].copy()
        ids = np.flatnonzero(~locked)
        if len(ids):
            locations, rays, _ = self.ray.intersects_location(origins[ids], np.tile(-d,(len(ids),1)), multiple_hits=False)
            distance = (origins[ids[rays]]-locations) @ d
            locked[ids[rays[(distance >= -self.epsilon) & (distance <= self.length)]]] = True
        self.lock_cache[key] = locked
        if len(self.lock_cache)>512:
            self.lock_cache.popitem(last=False)
        return locked

    def material_at_points(self, layout, owner, provider):
        relative = np.linalg.inv(layout.placements[provider]) @ layout.placements[owner]
        key = (provider, np.round(relative, 11).tobytes())
        if key not in self.seed_cache:
            outward = self.point_normals @ relative[:3, :3].T
            outside = C.transform_points(self.points, relative)+self.epsilon*outward
            self.seed_cache[key] = self.wrap_rays[provider].contains_points(outside)
            if len(self.seed_cache) > 256:
                self.seed_cache.popitem(last=False)
        return self.seed_cache[key]

    def contact_flags(self, layout):
        active = {}
        for k in layout.active:
            # A free surface point supplies a reaction only if a CURRENT
            # fitted wrap actually provides material immediately outside it.
            # This accounts for lost as well as newly grown support.
            has_material = np.zeros(len(self.points),bool)
            for j in layout.active:
                has_material |= self.material_at_points(layout, k, j)
            available = np.isin(self.sources, self.allowed[k]) & has_material
            for j in layout.active:
                available &= ~self.locks(layout,k,j)
            active[k] = available
        return active

    def proxy(self, layout, targets):
        self.proxy_calls += 1
        active = self.contact_flags(layout)
        losses=[]
        by_owner={}
        for k,i in targets:
            by_owner.setdefault(k,[]).append(i)
        for k,indices in by_owner.items():
            rays = np.vstack([self.floors[k], self.point_rays[k][active[k]]])
            # Reuse primal/dual certificates for the small guidance batch.
            # Solve projections only for its genuinely uncovered demands.
            masks,_=classify(rays,self.tasks[k].targets[indices])
            losses.extend([0.] * int(masks.sum()))
            for p in np.flatnonzero(~masks):
                result = cone_projection(rays,C.U.target(self.tasks[k].targets[indices[p]]))
                if result['kkt_max_violation']>1e-7:
                    raise RuntimeError('Proxy projection KKT unresolved')
                losses.append(float(result['loss']))
        span = self.layout_span(layout)
        return dict(loss=max(losses,default=0.), sum_loss=sum(losses), span_m=span,
                    available_areas_m2={str(k):float(self.point_areas[active[k]].sum()) for k in layout.active})

    def layout_span(self,layout):
        # Translations of differently rotated, uncentered coordinate systems
        # are not physical object separation. Compare actual object centers.
        centers=np.array([C.transform_points(self.mesh.center_mass[None],layout.placements[k])[0]
                          for k in layout.active])
        return float(np.linalg.norm(np.ptp(centers,axis=0)))

    def exact(self, layout):
        key = layout.key()
        if key in self.unresolved_layouts:
            raise RuntimeError('Previously unresolved candidate: '+self.unresolved_layouts[key])
        if key in self.exact_cache:
            self.exact_cache.move_to_end(key)
            return self.exact_cache[key]
        if hasattr(self,'worker_program'):
            return self.isolated_exact(layout)
        began=time.monotonic(); self.exact_calls += 1
        active=layout.active
        for k in active:
            if layout.directions[k] @ self.floor_normal(layout,k)<-1e-12:
                raise ValueError('Direction crosses its world ground plane')
            relative = self.native[layout.hosts[k]] @ layout.placements[k]
            assert np.allclose(relative[:3,:3],self.native[k,:3,:3],atol=1e-10)
            assert abs(relative[2,3]-self.native[k,2,3])<1e-9
        stage=time.monotonic();timings={}
        wrap_keys=set();wrap_parts=[]
        for k in active:
            wrap_key=(layout.placements[k].tobytes(),tuple(self.allowed[k]))
            if wrap_key not in wrap_keys:
                wrap_keys.add(wrap_key)
                wrap_parts.append(transform_solid(self.wraps[k],layout.placements[k]))
        seed=C.union(wrap_parts)
        work=C.union([transform_solid(self.works[k],layout.placements[k]) for k in active])
        bodies=[];body_keys=set()
        for k in active:
            body_key=layout.placements[k].tobytes()
            if body_key not in body_keys:
                body_keys.add(body_key);bodies.append(transform_solid(self.body,layout.placements[k]))
        nominal_sweeps=[];padded_sweeps=[];objects={};allowed={}
        sweep_keys=set()
        for k in active:
            q=layout.placements[k];d=q[:3,:3].T @ layout.directions[k]
            sweep_key=(q.tobytes(),d.tobytes())
            if sweep_key not in sweep_keys:
                sweep_keys.add(sweep_key)
                nominal_sweeps.append(transform_solid(self.clearance.sweep(self.length*d,padded=False),q))
                padded_sweeps.append(transform_solid(self.clearance.sweep(self.length*d,padded=True),q))
            objects[k]=transform_mesh(self.mesh,q)
            allowed[k]=self.allowed[k][self.mesh.face_normals[self.allowed[k]] @ d <= 1e-9]
        timings['wrap_and_sweeps_s']=time.monotonic()-stage;stage=time.monotonic()
        seed=seed-work
        # Every exact sweep contains its original body; do not union the same
        # coplanar bodies into the cut again. Body overlaps are still checked.
        nominal_cut=C.union(nominal_sweeps)
        padded_cut=C.union(padded_sweeps)
        nominal=seed-nominal_cut
        nominal_mesh=unpack_solid(nominal)
        core_parts=[]
        core_groups={}
        for k in active:
            core_groups.setdefault(layout.placements[k].tobytes(),[]).append(k)
        for members in core_groups.values():
            k=members[0]
            shared_allowed=np.unique(np.concatenate([allowed[j] for j in members]))
            triangles,sources=C.contact_boundary(objects[k],nominal_mesh,shared_allowed)
            for tri,source in zip(triangles,sources):
                prism=C.trimesh.Trimesh(np.vstack([tri,tri+CONTACT_DEPTH_M*objects[k].face_normals[source]]),_PRISM_FACES,process=False)
                if prism.volume<0:
                    prism.invert()
                part=C.S.solid(prism)
                # Distribute the intersection over the union only once below.
                # Testing each prism separately repeats the same large Boolean
                # hundreds of times and does not change the resulting core.
                core_parts.append(part)
        core=C.union(core_parts) ^ nominal
        remaining=((seed-padded_cut)+core)-nominal_cut
        remaining=remaining-(padded_cut-core)
        # Only actual exclusions may be recut; never relax the guard.
        volumes=None
        for _ in range(3):
            violations=[remaining ^ v for v in nominal_sweeps+bodies+[work]]
            violations.append((remaining-core)^padded_cut)
            volumes=[C.material_volume(v) for v in violations]
            if max(volumes,default=0.)<=TOL:
                break
            remaining=remaining-C.union([v for v,volume in zip(violations,volumes) if volume>TOL])
            volumes=None
        # Reuse the very intersections just validated. Re-evaluate only if
        # the final repair changed the solid after their computation.
        if volumes is None:
            volumes=[C.material_volume(remaining ^ v) for v in nominal_sweeps+bodies+[work]]
            volumes.append(C.material_volume((remaining-core)^padded_cut))
        ns,nb=len(nominal_sweeps),len(bodies)
        diagnostics=dict(nominal_sweep_overlap_m3=max(volumes[:ns],default=0.),
                         body_overlap_m3=max(volumes[ns:ns+nb],default=0.),
                         padded_overlap_outside_contact_cores_m3=volumes[-1],
                         work_band_overlap_m3=volumes[-2])
        if max(diagnostics.values())>TOL:
            raise RuntimeError('Exact exclusions unresolved: '+str(diagnostics))
        remaining=remaining.as_original()
        final=unpack_solid(remaining)
        timings['solid_contacts_and_exclusions_s']=time.monotonic()-stage;stage=time.monotonic()
        if not len(final.faces):
            raise RuntimeError('No remaining material')
        masks={};supplies={};contacts={};classifiers={};footprints=[]
        for k in active:
            tri,src=C.contact_boundary(objects[k],final,allowed[k])
            # Map contacts back to immutable native task coordinates. A lateral
            # layout shift translates its physical task and floor together;
            # this inverse shift preserves COM-relative original wrench data.
            native_map=self.native[k] @ np.linalg.inv(layout.placements[k])
            full=C.supply(self.tasks[k],native_map,tri,src)
            mask,info=classify(full,self.tasks[k].targets)
            masks[k]=mask;supplies[k]=full;classifiers[k]=info
            contacts[k]=dict(triangle_count=len(tri),area_m2=float(np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1).sum()/2),triangles=tri,sources=src)
            xy=C.transform_points(final.vertices,self.native[layout.hosts[k]])[:,:2]
            footprints.append(float(ConvexHull(xy).volume))
        timings['final_contacts_and_loads_s']=time.monotonic()-stage;stage=time.monotonic()
        end_overlaps=[];gaps=[]
        for k in active:
            q=layout.placements[k].copy();q[:3,3]+=self.length*layout.directions[k]
            end_overlaps.append(C.material_volume(transform_solid(self.body,q)^seed))
            d=layout.directions[k]
            gaps.append(float((objects[k].vertices @ d).min()+self.length-(final.vertices @ d).max()))
        if max(end_overlaps,default=0.)>TOL or min(gaps,default=1.)<=0:
            raise RuntimeError('Full exit endpoint not detached')
        diagnostics.update(endpoint_overlap_m3=max(end_overlaps,default=0.),minimum_endpoint_projection_gap_m=min(gaps,default=0.))
        result=dict(layout=layout.copy(),serial=self.exact_calls,remaining=remaining,
                    masks=masks,supplies=supplies,contacts=contacts,classifiers=classifiers,
                    counts={str(k):int(m.sum()) for k,m in masks.items()},diagnostics=diagnostics,
                    volume_cm3=C.material_volume(remaining)*1e6,maximum_projected_footprint_m2=max(footprints),
                    seconds=time.monotonic()-began,timings=timings)
        if all(mask.all() for mask in masks.values()):
            result['actual_work_surface_checks']=self.verify_work(result)
            if not all(row['passed'] for row in result['actual_work_surface_checks']):
                raise RuntimeError('Full force coverage but working surface verification unresolved: '+
                                   str(result['actual_work_surface_checks']))
        timings['exit_endpoint_and_work_s']=time.monotonic()-stage
        result['seconds']=time.monotonic()-began
        self.exact_cache[key]=result
        self.store_checkpoint(result)
        if len(self.exact_cache)>2:
            self.exact_cache.popitem(last=False)
        print('EXACT',self.exact_calls, 'counts', list(result['counts'].values()),'seconds',round(result['seconds'],2),flush=True)
        return result

    def evaluate(self, layout):
        """Search evaluation; the legacy model retains exact evaluation."""
        return self.exact(layout)

    def store_checkpoint(self,result):
        if hasattr(self,'checkpoint_dir'):
            folder=self.checkpoint_dir;folder.mkdir(parents=True,exist_ok=True)
            layout=result['layout'];active=layout.active
            final=result_mesh(result)
            masks=result['masks'];supplies=result['supplies'];diagnostics=result['diagnostics']
            arrays=dict(vertices=final.vertices,faces=final.faces,placements=layout.placements,
                        directions=layout.directions,hosts=layout.hosts,active=np.array(active))
            for k in active:
                arrays[f'mask_{k}']=masks[k];arrays[f'supply_{k}']=supplies[k]
            np.savez_compressed(folder/'latest.npz',**arrays)
            C.save(folder/'latest.json',dict(serial=result['serial'],counts=result['counts'],
                diagnostics=diagnostics,volume_cm3=result['volume_cm3'],
                maximum_projected_footprint_m2=result['maximum_projected_footprint_m2']))
            if all(mask.all() for mask in masks.values()):
                np.savez_compressed(folder/f'feasible_{self.exact_calls}.npz',**arrays)
                C.save(folder/f'feasible_{self.exact_calls}.json',dict(counts=result['counts'],
                    volume_cm3=result['volume_cm3'],
                    component_volumes_cm3=result.get('component_volumes_cm3'),
                    lossless_worker_geometry_transport=result.get('lossless_worker_geometry_transport', False),
                    actual_work_surface_checks=result['actual_work_surface_checks'],diagnostics=diagnostics))

    def isolated_exact(self,layout):
        from exact_worker import read_result
        began=time.monotonic();self.exact_calls+=1
        serial=self.exact_calls
        logs=self.checkpoint_dir.parent/'worker_logs';logs.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='pose_set_exact_') as folder:
            folder=Path(folder);path=folder/'layout.npz'
            np.savez_compressed(path,placements=layout.placements,directions=layout.directions,
                                hosts=layout.hosts,active=np.array(layout.active))
            env=os.environ.copy();env['POSE_SET_REPO_ROOT']=str(ROOT)
            with (logs/f'evaluation_{serial}.log').open('w') as log:
                try:
                    process=subprocess.run([sys.executable,'-u','-X','faulthandler',str(self.worker_program),
                        '--poses',','.join(self.poses),'--layout',str(path),'--out',str(folder/'result')],
                        env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
                except subprocess.TimeoutExpired as error:
                    self.unresolved_layouts[layout.key()]=f'evaluation {serial} exceeded 180s'
                    raise RuntimeError(f'Exact evaluation {serial} exceeded 180s; candidate unresolved') from error
            if process.returncode:
                self.unresolved_layouts[layout.key()]=f'evaluation {serial} exited {process.returncode}'
                raise RuntimeError(f'Exact evaluation {serial} failed; see worker_logs/evaluation_{serial}.log')
            result=read_result(folder/'result',serial)
        result['seconds']=time.monotonic()-began
        self.exact_cache[layout.key()]=result
        self.store_checkpoint(result)
        if len(self.exact_cache)>2:
            self.exact_cache.popitem(last=False)
        print('EXACT',serial,'counts',list(result['counts'].values()),'seconds',round(result['seconds'],2),'isolated',flush=True)
        return result

    def hardest(self, result):
        best=None;scans=[]
        for k,mask in result['masks'].items():
            ids=np.flatnonzero(~mask)
            if not len(ids):
                continue
            targets=C.U.target(self.tasks[k].targets[ids])
            upper=.5*np.einsum('ij,ij->i',targets,targets)
            projections=0
            while len(ids):
                p=int(np.argmax(upper))
                if best is not None and upper[p]<best['loss']-1e-10:
                    break
                value=cone_projection(result['supplies'][k],targets[p])
                if value['kkt_max_violation']>1e-7:
                    raise RuntimeError('Exact worst projection KKT unresolved')
                projections+=1
                if best is None or value['loss']>best['loss']:
                    best=dict(pose_index=int(k),load_index=int(ids[p]),loss=float(value['loss']),residual=value['residual'].tolist())
                point=targets[p]+value['residual']
                delta=targets-point
                upper=np.minimum(upper,.5*np.einsum('ij,ij->i',delta,delta)+1e-10)
                upper[p]=-np.inf
            scans.append(dict(pose=self.poses[k],failed_loads=len(ids),exact_projections=projections))
        return best,scans

    def verify_work(self, result):
        mesh=result_mesh(result);rows=[]
        for k in result['layout'].active:
            world=transform_mesh(mesh,self.native[k] @ np.linalg.inv(result['layout'].placements[k]))
            check=work_check(world,self.tasks[k])
            rows.append(dict(pose=self.poses[k],**check))
        return rows

    def save(self, result, out, extra):
        if result.get('pending_geometry'):
            raise RuntimeError('Cannot export an unevaluated insertion frontier as a final support')
        out.mkdir(parents=True,exist_ok=True)
        layout=result['layout'];active=layout.active
        C.D.export_exact_obj(result_mesh(result),out/'support.obj')
        np.savez_compressed(out/'layout.npz',placements=layout.placements,directions=layout.directions,
                            hosts=layout.hosts,active=np.array(active),native_world=self.native)
        for k in active:
            np.savez_compressed(out/f'{self.poses[k]}_force.npz',mask=result['masks'][k],supply_7d=result['supplies'][k],
                                triangles_fixture_m=result['contacts'][k]['triangles'],source_faces=result['contacts'][k]['sources'])
        work=result.get('actual_work_surface_checks')
        if work is None:
            work=self.verify_work(result)
        bodies={k:transform_solid(self.body,layout.placements[k]) for k in active}
        overlaps={}
        for i,k in enumerate(active):
            for j in active[i+1:]:
                overlaps[f'{self.poses[k]}+{self.poses[j]}']=C.material_volume(bodies[k] ^ bodies[j])
        shell_reference=sum(C.material_volume(self.wraps[k]) for k in active)*1e6
        force=all(result['masks'][k].all() for k in active)
        component_volumes = (result['component_volumes_cm3'] if 'component_volumes_cm3' in result else
                            sorted([C.material_volume(p)*1e6 for p in result['remaining'].decompose()], reverse=True))
        report=dict(complete=True,object=self.name,poses=[self.poses[k] for k in active],
                    pose_count=len(active),counts={self.poses[k]:int(result['masks'][k].sum()) for k in active},
                    load_count_per_pose=32768,original_loads_reused=True,load_subsampling_for_final_acceptance=False,
                    lossless_worker_geometry_transport=result.get('lossless_worker_geometry_transport', False),
                    force_passed=force,force_exit_work_passed=bool(force and all(r['passed'] for r in work)),
                    full_fixture_accepted=False,connectivity_acceptance_run=False,installed_floor_acceptance_run=False,
                    strength_acceptance_run=False,force_model=C.U.description(),diagnostics=result['diagnostics'],
                    original_floor_model=C.FLOOR.description(),
                    actual_work_surface_checks=work,volume_cm3=result['volume_cm3'],
                    classifiers={self.poses[k]:result['classifiers'][k] for k in active},
                    maximum_projected_footprint_m2=result['maximum_projected_footprint_m2'],
                    component_volumes_cm3=component_volumes,
                    actual_body_overlap_m3=overlaps,uncut_individual_wrap_volume_sum_cm3=shell_reference,
                    hosts={self.poses[k]:self.poses[int(layout.hosts[k])] for k in active},
                    placements=layout.placements.tolist(),directions=layout.directions.tolist(),
                    exact_evaluations=self.exact_calls,proxy_evaluations=self.proxy_calls,
                    clearance=self.clearance.metadata,full_exit_length_m=self.length,relocation_sweep_carved=False,
                    numerical_tolerance_m3=TOL,provenance=C.provenance(self.inputs,[Path(__file__),Path(__file__).with_name('common.py')]+C.code_sources()),
                    executed_search_sources_sha256=self.startup_sources,
                    **extra)
        report['artifacts']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!='report.json'}
        C.save(out/'report.json',report)
        return report
