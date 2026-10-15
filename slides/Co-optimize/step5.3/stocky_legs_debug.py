"""Wider leg shafts and localized smoothly flared roots inside a fixed BBX."""
import time
import numpy as np
import manifold3d as md
from scipy.spatial import cKDTree,ConvexHull
from continuous_legs import capsule,no_tunnels,bounded_domain
from continuous_legs_safe_recovery import RouteGuide
from deterministic_base import solid,unpack,volume,union,SCALE,enclosure
from prune_support import material_components


def smoothstep(t):
    return t*t*(3.-2.*t)


from flared_loft_envelope import flared_path as raw_flared_path

_active_envelope=None
def flared_path(path,shaft_radius,root_radius,flare_length=.020):
    return raw_flared_path(path,shaft_radius,root_radius,flare_length,minimum_envelope=_active_envelope)


def recipe_path(record):
    if 'path_m'in record:return np.asarray(record['path_m'])[::-1]
    return np.array([record['root_m'],record['tip_m']],dtype=float)


def cross_section_widths(value,path):
    """Actual widths in sampled interior sections, excluding terminal caps.

    These measured samples are not a global minimum-thickness certificate.
    The narrow caliper of each section detects thin clipped sheets even when
    their nominal template is round and their area is relatively large.
    """
    path=np.asarray(path);lengths=np.linalg.norm(np.diff(path,axis=0),axis=1)
    cumulative=np.r_[0.,np.cumsum(lengths)];total=float(cumulative[-1]);widths=[]
    if total<.010:return widths
    for fraction in [.15,.30,.50,.70,.85]:
        distance=fraction*total;index=min(len(lengths)-1,int(np.searchsorted(cumulative,distance,side='right')-1))
        if lengths[index]<1e-9:continue
        t=(distance-cumulative[index])/lengths[index]
        point=path[index]*(1-t)+path[index+1]*t
        direction=(path[index+1]-path[index])/lengths[index]
        axis=np.eye(3)[int(np.argmin(abs(direction)))];x=np.cross(direction,axis);x/=np.linalg.norm(x)
        y=np.cross(direction,x);rotation=np.array([x,y,direction])
        matrix=np.c_[rotation,-rotation@point/SCALE]
        contours=value.transform(matrix).slice(0.).to_polygons()
        measurements=[]
        for contour in contours:
            xy=np.asarray(contour)*SCALE
            if len(xy)<3:continue
            try:hull=xy[ConvexHull(xy).vertices]
            except Exception:continue
            edges=np.roll(hull,-1,axis=0)-hull
            normals=np.c_[-edges[:,1],edges[:,0]]
            norms=np.linalg.norm(normals,axis=1);normals=normals[norms>1e-10]/norms[norms>1e-10,None]
            projections=hull@normals.T
            width=float(np.ptp(projections,axis=0).min())
            area=float(abs((hull[:,0]*np.roll(hull[:,1],-1)-hull[:,1]*np.roll(hull[:,0],-1)).sum())*.5)
            measurements.append((area,width))
        if measurements:widths.append(max(measurements)[1])
    return widths


class Builder:
    def __init__(self,core,previous,witness,body,rows,bounds,recipes):
        self.core=solid(core);self.previous=solid(previous);self.rows=rows;self.object=body
        self.bounds=bounds;self.domain=bounded_domain(witness,body,rows,bounds)
        self.recipes=recipes;self.guide=None;self.began=time.monotonic()
        self.limit=volume(self.previous)*1.20
        self.old_reference=[]
        for r in recipes:
            path=recipe_path(r);radius=r['radius_m']
            old=(capsule(path[-1],path[0],radius,float(r.get('root_radius_m',radius*1.35)))^self.domain if 'root_m'in r and r['operation']=='solid_rounded_leg' else union([capsule(a,b,radius)for a,b in zip(path[:-1],path[1:])])^self.domain)
            self.old_reference.append(cross_section_widths(old,path))

    def part(self,proposal,point,current):
        probe=md.Manifold.sphere(.0002/SCALE,12).translate(tuple(point/SCALE))
        candidates=[]
        for piece in material_components(proposal^self.domain):
            if volume(piece^probe)<=1e-14 or volume(piece^current)<=1e-13 or not no_tunnels(piece):continue
            merged=current+piece
            if len(material_components(merged))!=1:continue
            candidates.append((volume(merged)-volume(current),piece,merged))
        return min(candidates,key=lambda x:x[0])if candidates else None

    def run(self):
        global _active_envelope
        attempts=[]
        # The larger profile is preferred; bounded material determines whether
        # a more modest shaft increase is needed, instead of adding huge roots.
        profiles=[(.0045,.0065,.016)]
        for shaft,root,reach in profiles:
            current=self.core;history=[];pieces=[];failed=None
            for index,record in enumerate(self.recipes):
                path=recipe_path(record);point=path[-1]
                _active_envelope=(float(record['radius_m']),float(record.get('root_radius_m',record['radius_m']*1.35))) if record['operation']=='solid_rounded_leg' else (float(record['radius_m']),float(record['radius_m']))
                actual_shaft=max(shaft,float(record['radius_m']))
                actual_root=max(root,float(record.get('root_radius_m',record['radius_m']*1.35)))
                proposal=flared_path(path,actual_shaft,actual_root,reach)
                chosen=self.part(proposal,point,current);selected_path=path
                if chosen is None:
                    mesh=unpack(current);_,ids=cKDTree(mesh.vertices).query(point,k=min(1000,len(mesh.vertices)))
                    roots=[]
                    for k in np.atleast_1d(ids):
                        a=mesh.vertices[k]
                        if not roots or min(np.linalg.norm(a-b)for b in roots)>.003:roots.append(a)
                        if len(roots)>=16:break
                    for a in roots:
                        alt=np.array([a,point]);candidate=self.part(flared_path(alt,actual_shaft,actual_root,reach),point,current)
                        if candidate is not None:chosen=candidate;selected_path=alt;break
                if chosen is None:
                    if self.guide is None:self.guide=RouteGuide(self.domain,self.core,pitch=.001)
                    try:routed,description=self.guide.route(point,limit=350000)
                    except ValueError:routed=None
                    if routed is not None:
                        alt=routed[::-1];chosen=self.part(flared_path(alt,actual_shaft,actual_root,reach),point,current)
                        if chosen is not None:selected_path=alt
                if chosen is None:failed='No unperforated broad leg '+str(index);break
                added,piece,candidate=chosen
                if volume(candidate)>self.limit*1.45:failed='Provisional material budget';break
                measured=cross_section_widths(piece,selected_path)
                previous=self.old_reference[index]
                old_min=min(previous)if previous else None
                requested=min(.005,max(.004,old_min*1.20))if old_min else .004
                thin_clipping_limited=False
                if measured and min(measured)<requested:
                    # A wider template is insufficient if clipping leaves a
                    # sheet. Compare actual sections for alternative roots and
                    # a clearance-preferring legal route before retaining it.
                    best=(min(measured),chosen,selected_path,measured)
                    vertices=unpack(current).vertices
                    _,ids=cKDTree(vertices).query(point,k=min(2000,len(vertices)))
                    roots=[]
                    for k in np.atleast_1d(ids):
                        a=vertices[k]
                        if not roots or min(np.linalg.norm(a-b)for b in roots)>.005:roots.append(a)
                        if len(roots)>=12:break
                    for a in roots:
                        alternative=np.array([a,point])
                        option=self.part(flared_path(alternative,actual_shaft,actual_root,reach),point,current)
                        if option is None or volume(option[2])>self.limit*1.45:continue
                        widths=cross_section_widths(option[1],alternative)
                        if widths and min(widths)>best[0]:best=(min(widths),option,alternative,widths)
                        if best[0]>=requested:break
                    if best[0]<requested:
                        if self.guide is None:self.guide=RouteGuide(self.domain,self.core,pitch=.001)
                        try:routed,_=self.guide.route(point,limit=180000)
                        except ValueError:routed=None
                        if routed is not None:
                            alternative=routed[::-1]
                            option=self.part(flared_path(alternative,actual_shaft,actual_root,reach),point,current)
                            if option is not None and volume(option[2])<=self.limit*1.45:
                                widths=cross_section_widths(option[1],alternative)
                                if widths and min(widths)>best[0]:best=(min(widths),option,alternative,widths)
                    _,chosen,selected_path,measured=best
                    added,piece,candidate=chosen
                    thin_clipping_limited=best[0]<requested
                    if old_min is not None and best[0]<old_min-1e-7:
                        print('THICKNESS DEBUG',index,'old',old_min,'new',best[0],'selected_path',selected_path.tolist(),'original_path',path.tolist(),flush=True); failed='Actual sampled shaft thinner than its prior leg '+str(index);break
                entry=dict(operation='flared_solid_leg',kind=record['kind'],tip_m=point.tolist(),
                    root_m=selected_path[0].tolist(),path_m=selected_path[::-1].tolist(),
                    radius_m=actual_shaft,root_radius_m=actual_root,flare_length_m=reach,
                    added_cm3=added*1e6,leg_tunnels=0,
                    sampled_section_widths_m=measured,
                    previous_sampled_section_widths_m=previous,
                    sampled_min_width_m=min(measured)if measured else None,
                    previous_sampled_min_width_m=min(previous)if previous else None,
                    nominal_thickness_is_not_actual_minimum=True,requested_sampled_width_m=requested,clipping_limited=thin_clipping_limited)
                current=candidate;history.append(entry);pieces.append(piece)
                print('FLARED LEG',index+1,'shaft/root mm',actual_shaft*2e3,actual_root*2e3,flush=True)
            attempts.append(dict(shaft_radius_m=shaft,root_radius_m=root,flare_length_m=reach,error=failed))
            if failed:print('PROFILE REJECT',failed,flush=True);continue
            from stage5_shared import occupied_bounds
            active=set(range(len(pieces)));removed=[]
            # Wider feet can make old terminals redundant. Remove only whole
            # legs, with actual connectivity, all floors and exact BBX retained.
            for k in sorted(active,key=lambda j:volume(pieces[j]),reverse=True):
                candidate=self.core+union([pieces[j]for j in sorted(active-{k})])
                if len(material_components(candidate))!=1:continue
                candidate_mesh=unpack(candidate)
                if not np.allclose(occupied_bounds(self.object,candidate_mesh,self.rows),self.bounds,atol=1e-8,rtol=0):continue
                if not all(r['passed']for r in enclosure(candidate_mesh,self.rows)):continue
                active.remove(k);current=candidate;removed.append(k)
            history=[entry for k,entry in enumerate(history)if k in active]
            if volume(current)>self.limit+1e-10:
                attempts[-1]['error']='20% final material budget after whole-leg deletion';continue
            mesh=unpack(current)
            cover=enclosure(mesh,self.rows);actual=occupied_bounds(self.object,mesh,self.rows)
            if not all(r['passed']for r in cover)or not np.allclose(actual,self.bounds,atol=1e-8,rtol=0):
                attempts[-1]['error']='Floor coverage or BBX';continue
            return mesh,dict(passed=True,connected=True,watertight=bool(mesh.is_watertight),
                material_cm3=volume(current)*1e6,added_material_cm3=volume(current-self.core)*1e6,
                retained_legal_step4_material_loss_cm3=volume(self.core-current)*1e6,
                per_pose=cover,history=history,leg_profile='stocky_flared_solid',contact_core_preserved=True,
                large_box_bridge_used=False,shaft_radius_m=shaft,root_radius_m=root,flare_length_m=reach,
                profile_attempts=attempts,removed_redundant_leg_indices=removed,material_growth_limit_fraction=.20,seconds=time.monotonic()-self.began)
        raise ValueError('No wider flared profile passed: '+str(attempts))
