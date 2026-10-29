"""Fixed-sample guidance; exact CSG and complete certificates accept designs.

No swept solid is rebuilt for finite differences. Moving-body occupancy is
queried by rays on the original nonconvex mesh. Common Sobol volume points
and source-face contact probes are diagnostic estimates, never certificates.
"""
from collections import OrderedDict
import numpy as np
from scipy.spatial import ConvexHull
from scipy.stats import qmc

from subdivide_task.operations.state import transform
from subdivide_task.operations.geometry import moved_mesh
from subdivide_task.operations.ribbed_stock import contact_skin
from subdivide_task.step3 import independent_geometry as G
from subdivide_task.step3.legacy import kernel as K
from subdivide_task.common.force_cone import tangent_frame


def swept_contains(body,direction,length,points):
    """Membership in body + [0,length*direction], without a CSG sweep mesh."""
    points=np.asarray(points);direction=np.asarray(direction)
    u,v=tangent_frame(direction);frame=np.column_stack([u,v,direction])
    local_body=body.vertices@frame
    low=local_body.min(0)-1e-9;high=local_body.max(0)+1e-9;high[2]+=length
    projected=points@frame
    active=np.flatnonzero(((projected>=low)&(projected<=high)).all(1))
    result=np.zeros(len(points),bool)
    if not len(active):return result
    origins=points[active]
    hits,ids,_=body.ray.intersects_location(origins,np.tile(-direction,(len(active),1)),multiple_hits=True)
    counts=np.bincount(ids,minlength=len(active))
    inside=counts%2==1
    if len(ids):
        distances=np.einsum('ij,j->i',origins[ids]-hits,direction)
        inside[ids[distances<=length+1e-9]]=True
    result[active]=inside
    return result


def mesh_contains(mesh,points):
    result=np.zeros(len(points),bool)
    active=np.flatnonzero(((points>=mesh.bounds[0])&(points<=mesh.bounds[1])).all(1))
    if len(active):result[active]=mesh.contains(points[active])
    return result


def convex_contains(mesh,points):
    result=np.zeros(len(points),bool)
    active=np.flatnonzero(((points>=mesh.bounds[0]-1e-10)&(points<=mesh.bounds[1]+1e-10)).all(1))
    if len(active):
        eq=ConvexHull(mesh.vertices).equations
        result[active]=(points[active]@eq[:,:3].T+eq[:,3]<=1e-10).all(1)
    return result


def probe_indices(count,samples,required=()):
    required=np.asarray(required,dtype=int)
    required=required[(required>=0)&(required<count)]
    return np.unique(np.r_[np.linspace(0,count-1,samples).astype(int),required])


def surface_probes(triangles):
    """Probe face interiors; offset vertices can escape a blocking prism.

    At an edge an outward-offset point may be outside its leading-face
    sweep while the entire face has no remaining contact area. Use inset
    points for guidance, never fictitious reactions on those bare edges.
    """
    center=np.asarray(triangles).mean(1)[:,None]
    return np.concatenate([.95*triangles+.05*center,center],axis=1).reshape(-1,3)


def compose_material(raw,parent,editable,protected,patches,mode='local',retained=None):
    """Point membership in the same local Boolean rules as construction."""
    region=editable&~protected
    if mode in ('free_material','free_material_box'):
        region=(editable|patches)&~protected
        return (raw&parent)|retained|(raw&region)
    return (parent&~region)|(raw&region)|(raw&patches)


class SampledObjective:
    def __init__(self,model,layout,owners,samples=24,obstruction_context=None,volume_power=16):
        self.model=model;self.owners=list(owners);self.samples=samples
        self.extent=float(model.source.extents.max());self.calls=0;self.cache=OrderedDict()
        self.context=model.prepare(layout) if obstruction_context is None else obstruction_context
        selected={model.ids[k] for k in owners}
        local=[t for t in self.context if t['id'] in selected]
        points=np.vstack([*[G.mesh(t['stock']).bounds for t in local],*[t['patch_points'] for t in local]])
        reference=model.reference_material(layout.material);normal=reference[:3,2];offset=normal@reference[:3,3]
        points=np.vstack([points,points-(points@normal-offset)[:,None]*normal])
        self.low=points.min(0)-.08;self.high=points.max(0)+.08
        self.box_volume=float(np.prod(self.high-self.low)*1e6)
        self.volume_points=self.low+qmc.Sobol(3,scramble=True,seed=42).random_base2(volume_power)*(self.high-self.low)
        self.reference_volume=None
        self.skin=contact_skin(model)
        self.foreign_rows=[dict(t,direction=layout.exits[k],
            length=max(t['length'],4*np.linalg.norm(self.high-self.low)))
            for k,t in enumerate(self.context) if t['id'] not in selected]
        # All world work sectors have the same fixed head and axis.
        self.world_work=moved_mesh(G.mesh(self.context[owners[0]]['work']),np.linalg.inv(layout.material[owners[0]]))
        self.outside_points=self.parent_probes(layout,selected)
        self.material_masks=None
        frozen=getattr(model,'family_before_support',None)
        if frozen is not None and getattr(model,'guidance_material','raw_stock')=='family_composition':
            pts=self.volume_points
            previous=np.zeros(len(pts),bool);foreign=previous.copy()
            for region in model.family_previous_regions:previous|=convex_contains(G.mesh(region),pts)
            for task in self.context:
                if task['id'] not in selected:foreign|=mesh_contains(G.mesh(task['stock']),pts)
            retained=getattr(model,'family_retained_material',None)
            self.material_masks=dict(parent=mesh_contains(G.mesh(frozen),pts),
                protected=mesh_contains(G.mesh(model.family_protected_contact_regions),pts),
                previous=previous,foreign=foreign,
                retained=mesh_contains(G.mesh(retained),pts) if retained is not None else None)

    def parent_probes(self,layout,selected):
        """Cheap interference guidance on parent witnesses and pressure feet.

        No unedited task load is resolved. Only the moving family's cutouts
        and floors can change these fixed probes. Full retention still uses
        actual contact cones and continuous primitives at acceptance.
        """
        points=[]
        witnesses=getattr(self.model,'retention_witnesses',{})
        signatures=getattr(self.model,'current_contacts',{})
        for k,task in enumerate(self.context):
            if task['id'] in selected:continue
            old_points=signatures.get(task['id'])
            basis=witnesses.get(task['id'])
            if old_points is not None and basis is not None:
                extent=task['body_world'].extents.max();com=task['domain'].com
                for ray in basis:
                    normal=ray[:3]
                    if np.linalg.norm(normal)<1e-10:continue
                    error=np.max(np.abs(np.cross(old_points-com,normal)-ray[3:6]*extent),axis=1)
                    match=np.flatnonzero(error<5e-11)
                    if len(match):points.append(transform(old_points[match[0]]-1e-5*normal,task['material'])[None])
            pressure=task['pressure_world']
            ids=np.unique(np.r_[ConvexHull(pressure).vertices,probe_indices(len(pressure),64,
                getattr(self.model,'repair_pressure_indices',{}).get(task['id'],()))])
            points.append(transform(np.c_[pressure[ids],np.full(len(ids),1e-5)],task['material']))
        frozen=getattr(self.model,'family_before_support',None)
        if frozen is not None and len(selected)<len(self.model.ids):
            # Contacts alone do not detect a severed old post. Also sample
            # material around the retained connection network, weighted by
            # actual boundary area so slender but long ribs receive probes.
            mesh=G.mesh(frozen);areas=mesh.area_faces
            cdf=np.cumsum(areas)
            ids=np.searchsorted(cdf,(np.arange(2048)+.5)*cdf[-1]/2048)
            uv=qmc.Sobol(2,scramble=True,seed=13).random_base2(11);r=np.sqrt(uv[:,0])
            bary=np.c_[1-r,r*(1-uv[:,1]),r*uv[:,1]]
            samples=(mesh.triangles[ids]*bary[:,:,None]).sum(1)-1e-5*mesh.face_normals[ids]
            kept=np.ones(len(samples),bool)
            for region in self.model.family_previous_regions:
                active=np.flatnonzero(kept)
                if not len(active):break
                kept[active[convex_contains(G.mesh(region),samples[active])]]=False
            points.append(samples[kept])
        if not points:return np.empty((0,3))
        points=np.vstack(points)
        before=getattr(self.model,'family_before_layout',layout)
        rows=[dict(self.model.world_data(self.model.records[k],before.poses[k]),
            material=before.material[k],direction=before.exits[k],
            length=max(self.context[k]['length'],4*np.linalg.norm(self.high-self.low))) for k in self.owners]
        # Discard probes already on ambiguous old cutout boundaries.
        return points[self.admissible(points,before,rows)]

    def data(self,layout):
        rows=[];patches=list(self.context)
        for k in self.owners:
            data=self.model.world_data(self.model.records[k],layout.poses[k])
            row=dict(data,id=self.model.ids[k],index=k,pose=layout.poses[k],material=layout.material[k],
                direction=layout.exits[k],length=max(self.context[k]['length'],4*np.linalg.norm(self.high-self.low)))
            row['patch_points']=transform(self.model.patch_points(data),layout.material[k])
            patches[k]=dict(self.context[k],material=layout.material[k],patch_points=row['patch_points'])
            rows.append(row)
        return rows,[G.mesh(p) for p in self.model.plates(patches)]

    def admissible(self,points,layout,rows):
        normals=layout.material[:,:3,2];offsets=np.einsum('ij,ij->i',normals,layout.material[:,:3,3])
        keep=(points@normals.T-offsets>=-1e-10).all(1)
        for row in [*self.foreign_rows,*rows]:
            active=np.flatnonzero(keep)
            if not len(active):return keep
            world=transform(points[active],np.linalg.inv(row['material']))
            cut=mesh_contains(self.world_work,world)
            remaining=np.flatnonzero(~cut)
            if len(remaining):
                cut[remaining]=swept_contains(row['body_world'],row['direction'],row['length'],world[remaining])
            keep[active[cut]]=False
        return keep

    def volume(self,layout,rows,plates):
        points=self.volume_points;raw=np.zeros(len(points),bool)
        edit=np.zeros(len(points),bool)
        if self.material_masks is not None:
            raw|=self.material_masks['foreign'];edit|=self.material_masks['previous']
        for row in rows:
            world=transform(points,np.linalg.inv(row['material']))
            objects=transform(world,np.linalg.inv(row['pose']))
            raw|=mesh_contains(self.skin,objects)
            lo,hi=row['stock_world'].bounds
            pitch,width=getattr(self.model,'stock_grid',None) or (.035,.008)
            axes=[np.linspace(lo[k]+width/2,hi[k]-width/2,max(2,int(np.ceil((hi[k]-lo[k])/pitch))+1)) for k in (0,1)]
            lattice=(np.abs(world[:,0,None]-axes[0]).min(1)<=width/2)&(np.abs(world[:,1,None]-axes[1]).min(1)<=width/2)
            base=(world[:,2]>=0.)&(world[:,2]<=self.model.policy['plate_thickness_m'])
            active=np.flatnonzero((lattice|base)&~raw)
            if len(active):raw[active]|=mesh_contains(row['stock_world'],world[active])
            if self.material_masks is not None:
                vertices=np.vstack([transform(self.skin.vertices,row['material']@row['pose']),
                    transform(row['stock_world'].vertices,row['material'])])
                edit|=((points>=vertices.min(0)-1e-5)&(points<=vertices.max(0)+1e-5)).all(1)
        patch=np.zeros(len(points),bool)
        for plate in plates:patch|=convex_contains(plate,points)
        raw|=patch
        if self.material_masks is not None:
            masks=self.material_masks
            raw=compose_material(raw,masks['parent'],edit,masks['protected'],patch,
                getattr(self.model,'family_refit_mode','local'),masks['retained'])
        active=np.flatnonzero(raw)
        if len(active):raw[active]=self.admissible(points[active],layout,rows)
        return float(raw.mean()*self.box_volume)

    def evaluate(self,layout):
        key=tuple(a.tobytes() for a in layout.arrays().values())
        if key in self.cache:return self.cache[key]
        self.calls+=1;result=None
        try:
            self.model.check_deadline()
            if ((layout.exits[:,2]<.05).any() or
                    (np.abs(np.linalg.norm(layout.exits,axis=1)-1.)>1e-12).any()):
                raise ValueError('Invalid upper-hemisphere extraction direction')
            from .gradient_repair import global_pressure_feasible,project_pressure
            if getattr(self.model,'use_floor_constraints',True) and not global_pressure_feasible(self.model,layout):
                layout=project_pressure(self.model,layout,self.owners)
                if layout is None:raise ValueError('An installed floor is violated')
            rows,plates=self.data(layout)
            losses=[];maximum=[];details=[]
            from continuous_support.projection import project_demands
            for row in rows:
                body=row['body_world'];allowed=np.setdiff1d(np.arange(len(body.faces)),row['domain'].work_ids)
                if len(allowed)>512:
                    selected=allowed[np.unique(np.linspace(0,len(allowed)-1,512).astype(int))]
                    profile=getattr(self.model,'contact_profiles',{}).get(row['id'])
                    if profile is not None:selected=np.unique(np.r_[selected,allowed[np.argsort(profile[allowed])[-64:]]])
                else:selected=allowed
                triangles=body.triangles[selected]
                contacts=surface_probes(triangles)
                normal=np.repeat(body.face_normals[selected],4,axis=0)
                probes=transform(contacts+1e-5*normal,row['material'])
                available=self.admissible(probes,layout,rows)
                contacts=contacts[available];reaction=-normal[available]
                scale=np.r_[np.ones(3),np.ones(3)/body.extents.max()]
                rays=K.I.merge_columns(K.U.floor(np.empty((0,6)),scale),
                    K.U.heads(np.c_[reaction,np.cross(contacts-row['domain'].com,reaction)],scale))
                targets=[]
                for kind,values in (('sample',row['samples']['need_wrench']),('endpoint',row['endpoints']['endpoint_need_wrench'])):
                    required=getattr(self.model,'repair_'+kind+'_indices',{}).get(row['id'],())
                    ids=probe_indices(len(values),self.samples,required);targets.append(values[ids]*scale)
                targets=np.vstack(targets);passed,_=K.J.classify(rays,targets)
                loss=residual=0.
                if not passed.all():
                    projected=project_demands(rays,K.J.C.U.target(targets,rays.shape[1]))
                    loss=float(np.mean(projected['normalized_losses']));residual=float(np.max(np.abs(projected['residuals'])))
                floor_loss=0.
                if getattr(self.model,'use_floor_constraints',True):
                    pressure=G.pressure_points(row['endpoints']['endpoint_need_wrench'],row['domain'].com)
                    ids=np.unique(np.r_[ConvexHull(pressure).vertices,probe_indices(len(pressure),64,
                        getattr(self.model,'repair_pressure_indices',{}).get(row['id'],()))])
                    pressure=transform(np.c_[pressure[ids],np.full(len(ids),1e-5)],row['material'])
                    floor_loss=float((~self.admissible(pressure,layout,rows)).mean())
                losses.append(loss+floor_loss);maximum.append(max(residual,np.sqrt(floor_loss)))
                details.append(dict(task=row['id'],residual_loss=loss,maximum_residual=residual,
                    floor_loss=floor_loss,sampled_paired_demands=len(targets),sampled_contact_points=len(contacts),
                    sampled_contact_source_faces=len(selected),diagnostic_only=True,
                    contact_method='interior source surface probes with original nonconvex ray sweeps'))
            outside_loss=(float((~self.admissible(self.outside_points,layout,rows)).mean())
                if len(self.outside_points) else 0.)
            covered=bool(max(maximum)<=2e-8 and outside_loss==0.)
            if covered or self.reference_volume is None:
                volume=self.volume(layout,rows,plates)
                if self.reference_volume is None:self.reference_volume=volume
            else:volume=self.reference_volume
            result=dict(layout=layout.copy(),loss=float(np.mean(losses))+outside_loss,covered=covered,
                volume_cm3=volume,details=details,volume_guidance_active=covered,
                volume_material_rule=('family_composition' if self.material_masks is not None else 'raw_stock'),
                volume_frozen_during_force_repair=not covered,
                unchanged_family_interference_loss=outside_loss,
                unchanged_family_probe_count=len(self.outside_points))
        except (ValueError,RuntimeError,AssertionError,np.linalg.LinAlgError):pass
        self.cache[key]=result
        if len(self.cache)>8:self.cache.popitem(last=False)
        return result
