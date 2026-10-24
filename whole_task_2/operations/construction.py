"""Local CSG with independent 3D installations and actual supporting feet."""
from copy import copy
from itertools import product
import time
import numpy as np
import manifold3d as md
from scipy.spatial import ConvexHull
from scipy.sparse.csgraph import minimum_spanning_tree

from whole_task.step3 import independent_geometry as G
from .geometry import OperationModel
from .state import Layout, transform
from .floors import planes, clip, connected_material, condition


def compact(solid):
    return G.solid(G.mesh(solid))


def overlaps(a, b):
    return bool(np.all(np.maximum(a[0], b[0]) < np.minimum(a[1], b[1]) - 1e-12))


class CompactConstructionModel(OperationModel):
    @staticmethod
    def reference_material(matrices):
        normals=np.asarray(matrices)[:,:3,2]
        index=np.argmax((normals@normals.T>1.-1e-9).sum(1))
        return matrices[index]

    def patch_points(self, data):
        # The ground needs the actual pressure envelope. Covering the entire
        # casting projection adds unused material, especially on a side face.
        points = G.pressure_points(data['endpoints']['endpoint_need_wrench'], data['domain'].com)
        hull = points[ConvexHull(points).vertices]
        margin = self.policy['plate_edge_margin_m']
        offsets = np.array(list(product((-margin, margin), repeat=2)))
        hull = (hull[:, None]+offsets).reshape(-1, 2)
        return np.vstack([np.c_[hull, np.zeros(len(hull))],
                          np.c_[hull, np.full(len(hull), self.policy['plate_thickness_m'])]])

    def prepare(self, layout):
        # A finite bound BEFORE extraction: casting stocks, floor patches, and
        # their connecting ribs. All swept-path endpoints are measured in each
        # installed WORLD, including 90 degree side installations.
        raw = [self.world_data(r,p) for r,p in zip(self.records,layout.poses)]
        patches = [transform(self.patch_points(d), M) for d,M in zip(raw,layout.material)]
        points = np.vstack([*patches,*[transform(d['stock_world'].vertices,M)
                           for d,M in zip(raw,layout.material)]])
        if getattr(self,'stock_grid',None) is not None:
            from .ribbed_stock import contact_skin
            skin=contact_skin(self)
            points=np.vstack([points,*[transform(skin.vertices,M@P)for M,P in zip(layout.material,layout.poses)]])
        if hasattr(self,'family_before_support') and not hasattr(self,'guide_envelope_bounds'):
            points=np.vstack([points,G.mesh(self.family_before_support).vertices])
        reference=self.reference_material(layout.material)
        ref_n = reference[:3,2]
        ref_offset = ref_n@reference[:3,3]
        projected = points-(points@ref_n-ref_offset)[:,None]*ref_n
        points = np.vstack([points,projected,projected+self.policy['plate_thickness_m']*ref_n])
        lo,hi = points.min(0)-.002,points.max(0)+.002
        fixed = getattr(self,'guide_envelope_bounds',None)
        if fixed is not None:
            # A local gradient trust region may keep a fixed finite envelope
            # to reuse unedited task sweeps. Production builds stay adaptive.
            fixed_lo,fixed_hi = fixed
            if np.any(lo<fixed_lo) or np.any(hi>fixed_hi):
                raise ValueError('Gradient probe leaves its finite guidance envelope')
            lo,hi = fixed_lo,fixed_hi
        corners = np.array(list(product(*zip(lo,hi))))
        rows = []
        for i in range(len(self.ids)):
            sub = copy(self)
            sub.ids, sub.records = [self.ids[i]], [self.records[i]]
            sub.safe_top = float(transform(corners,np.linalg.inv(layout.material[i]))[:,2].max())
            q = Layout(layout.poses[i:i+1],layout.material[i:i+1],layout.exits[i:i+1])
            row = OperationModel.prepare(sub,q)[0]
            row['index'] = i
            key = sub.geometry_key(q.poses[0],q.material[0],q.exits[0])
            cached = self.geometry_cache[key]
            if not cached.get('compacted'):
                for name in ('stock','work','sweep'):
                    cached[name] = compact(cached[name])
                cached['compacted'] = True
            row.update(cached)
            row['dense_stock']=row['stock']
            if getattr(self,'stock_grid',None) is not None:
                from .ribbed_stock import world_stock
                sparse=world_stock(self,raw[i],layout.poses[i],*self.stock_grid)
                from .geometry import moved_mesh
                # Do not replace the dense geometry cache: stock profiles
                # are independent proposals, each followed by a full audit.
                row['stock']=G.solid(moved_mesh(sparse,layout.material[i]))
            row['patch_points'] = patches[i]
            rows.append(row)
        floor_planes = planes(layout.material)
        for task in rows:
            for normal,offset in floor_planes:
                if (task['pressure_3d']@normal-offset).min() < -1e-9:
                    raise ValueError(task['id']+': pressure envelope is below another installation floor')
        return rows

    def plates(self, tasks):
        margin = self.policy['plate_edge_margin_m']
        top = self.policy['plate_thickness_m']
        reference=self.reference_material(np.array([t['material']for t in tasks]))
        ref_n = reference[:3,2]
        ref_offset = ref_n@reference[:3,3]
        patches,centers = [],[]
        for task in tasks:
            points = task['patch_points']
            patches.append(md.Manifold.hull_points(points/G.SCALE))
            center = points.mean(0)
            normal=task['material'][:3,2]
            tangent=ref_n-(ref_n@normal)*normal
            if np.linalg.norm(tangent)>1e-8:
                # A narrow real post joins a side pressure plate to the base.
                # Filling the whole plate-to-base hull adds an unnecessary
                # wall. Stay tangent to the side floor so both post ends are
                # inside its 8 mm slab, then clip by every installed floor.
                foot=center-tangent*((center@ref_n-ref_offset-top/2)/(tangent@ref_n))
                across=np.cross(ref_n,normal);across/=np.linalg.norm(across)
                corners=np.array([p+u*margin/2*across+v*top/2*normal
                                  for p in(center,foot)for u,v in product((-1.,1.),repeat=2)])
                patches.append(md.Manifold.hull_points(corners/G.SCALE))
            else:
                foot=center-(center@ref_n-ref_offset-top/2)*ref_n
            centers.append(foot)
        centers = np.asarray(centers)
        if len(centers)>1:
            distances = np.linalg.norm(centers[:,None]-centers[None],axis=2)
            distances += (1-np.eye(len(centers)))*1e-12
            graph = minimum_spanning_tree(distances).tocoo()
            for i,j in zip(graph.row,graph.col):
                delta = centers[j]-centers[i]
                length = np.linalg.norm(delta)
                if length<1e-10:
                    continue
                u = delta/length
                v = np.cross(ref_n,u)
                v /= np.linalg.norm(v)
                corners = np.array([centers[i]-margin*u-margin*v/2,
                                    centers[i]-margin*u+margin*v/2,
                                    centers[j]+margin*u-margin*v/2,
                                    centers[j]+margin*u+margin*v/2])
                vertices = np.vstack([corners-top/2*ref_n,corners+top/2*ref_n])
                patches.append(md.Manifold.hull_points(vertices/G.SCALE))
        return patches

    def build(self, layout, *, facewise=False):
        began = time.monotonic()
        tasks = self.prepare(layout)
        patches = self.plates(tasks)
        parents = [*patches,*[t['stock'] for t in tasks]]
        floor_planes = planes(layout.material)
        all_bounds = np.asarray([G.mesh(p).bounds for p in parents])
        if hasattr(self,'family_before_support'):
            # A new connection tree can have a smaller box than the actual
            # parent. Do not clip away an old valid detour outside that box.
            all_bounds=np.concatenate([all_bounds,G.mesh(self.family_before_support).bounds[None]])
        lo = all_bounds[:,0].min(0)-.001
        hi = all_bounds[:,1].max(0)+.001
        # Clip one simple convex envelope, then intersect stocks with it.
        # Repeated plane cuts through dense fitted boundaries can create
        # nonfinite vertices in the native plane-trimming implementation.
        parent = clip(G.box(lo,hi),floor_planes)
        obstacles = [t[k] for t in tasks for k in ('sweep','work')]
        bounds = [G.mesh(o).bounds for o in obstacles]
        pieces = []
        for stock in parents:
            stock = stock^parent
            if G.volume(stock)<=1e-15:
                continue
            stock_bounds = G.mesh(stock).bounds
            hits = [o for o,b in zip(obstacles,bounds) if overlaps(stock_bounds,b)]
            cut = stock-G.union(hits) if hits else stock
            if G.volume(cut)>1e-15:
                pieces.append(compact(cut))
        if not pieces:
            raise ValueError('All material removed by installation floor halfspaces')
        # Temporary new pieces can connect through the retained parent.
        # Discarding them before that union erases useful new contact layers.
        retaining_parent=hasattr(self,'family_before_support')
        keep_parts=retaining_parent and getattr(self,'keep_intermediate_material_components',True)
        value = G.union(pieces)
        if not keep_parts:value=connected_material(value)
        forbidden = [p for t in tasks for p in (G.solid(t['body']),t['work'])]
        if keep_parts:
            # These are intermediate parts, not an export or an accepted
            # solid. Condition the completed parent union once below.
            stock_value=value
            conditioning=dict(intermediate_conditioning_deferred_to_final_union=True)
        else:
            support,conditioning = condition(value,forbidden,parent,self.tolerance)
            stock_value=G.solid(support)
        conditioning['intermediate_components_retained_until_parent_union']=keep_parts
        if hasattr(self,'family_before_support'):
            # Keep the actual parent outside the two selected families. Cut
            # out their old slots and insert their new local construction.
            # Pressure patches/posts/rails supply physical connections across
            # removed slots; no other contact layer is regenerated globally.
            selected=[t for t in tasks if t['id'] in self.family_edit_ids]
            regions=[*self.family_previous_regions,
                *[G.box(G.mesh(t['stock']).bounds[0]-1e-5,
                        G.mesh(t['stock']).bounds[1]+1e-5) for t in selected]]
            if getattr(self,'family_refit_mode','local') in ('free_material','free_material_box'):
                editable=G.union([*regions,*patches])-self.family_protected_contact_regions
                # Away from the selected family, remove redundant fill only.
                # Retain every prior contact layer and actual pressure foot.
                # New material is restricted to this family's placement and
                # the necessary connections, as in the local constructor.
                fixed=(stock_value^self.family_before_support)+self.family_retained_material
                fixed=fixed+(stock_value^editable)
                conditioning['free_material_refit']=True
            else:
                # Existing thin families already have legal, connected feet
                # and ribs. Replace selected seats only. New connection
                # strips may be added, but must not erase unrelated old ribs
                # merely because a freshly computed MST chooses other edges.
                editable=G.union(regions)-self.family_protected_contact_regions
                fixed=(self.family_before_support-editable)+(stock_value^editable)
                fixed=fixed+(stock_value^G.union(patches))
                conditioning['existing_connection_network_retained']=True
            if getattr(self,'simplify_construction_boundaries',False):
                fixed=fixed.simplify(0.)
                conditioning['native_boundary_simplification']=dict(
                    proposal_only=True,native_tolerance_increased=False,
                    original_force_exit_and_floor_gates_required=True)
            support,frozen_conditioning=condition(fixed,forbidden,parent,self.tolerance)
            conditioning['independent_family_material_preserved']=True
            conditioning['frozen_material_conditioning']=frozen_conditioning
        candidate = dict(layout=layout.copy(),support=support,tasks=tasks,volume_cm3=float(support.volume*1e6),
            construction=conditioning,build_seconds=time.monotonic()-began,parent=parent,
            floor_halfspaces=[dict(normal=n.tolist(),offset_m=o)for n,o in floor_planes],
            base_construction='actual pressure patches, narrow side posts and base strips, clipped by every installed floor')
        if getattr(self,'stock_grid',None) is not None:
            candidate['construction']['ribbed_stock']=dict(pitch_m=self.stock_grid[0],
                post_width_m=self.stock_grid[1],original_fitted_contact_skin_retained=True)
        if getattr(self,'optimize_connecting_stock',False):
            from .connection_stock import connecting_stock
            reduced,connections=connecting_stock(self,candidate)
            if connections['applied']:
                reduced,cleanup=condition(G.solid(reduced),forbidden,parent,self.tolerance)
                candidate.update(support=reduced,volume_cm3=float(reduced.volume*1e6))
                connections.update(final_conditioning=cleanup,conditioned_volume_cm3=candidate['volume_cm3'])
            candidate['construction']['connecting_stock']=connections
        if facewise:
            from .clearance import refine_candidate
            support,refinement = refine_candidate(self,candidate)
            candidate.update(support=support,volume_cm3=float(support.volume*1e6))
            conditioning['source_face_prisms'] = refinement
        return candidate
