"""Rounded support around full-width interior anchors and necessary ports.

Interior anchors contain an actual minimum-radius ball. Connector endpoints
share these anchors, giving positive-width junctions. Functional contact and
ground caps reach the original required locations from that interior backbone.
"""
import numpy as np
import manifold3d as md
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.graph import MCP_Geometric
from feasible_geometry import C
from continuous_legs import bounded_domain,raster
from deterministic_base import solid,unpack,union,volume,SCALE
from prune_support import material_components
from tube_support import ground_ports,capsule


def ball(point,radius):
    return md.Manifold.sphere(radius/SCALE,48).translate(tuple(np.asarray(point)/SCALE))


class InteriorField:
    def __init__(self,domain,radius,pitch=.00075):
        self.domain=domain;self.radius=radius
        bounds=unpack(domain).bounds
        self.pitch=max(pitch,float(np.prod(bounds[1]-bounds[0])/24_000_000)**(1/3))
        self.origin=np.floor(bounds[0]/self.pitch)*self.pitch-self.pitch
        self.shape=tuple(np.ceil((bounds[1]-self.origin)/self.pitch).astype(int)+2)
        inside=raster(domain,self.origin,self.shape,self.pitch)
        self.clearance=ndimage.distance_transform_edt(inside,sampling=self.pitch).astype(np.float32)
        valid=inside&(self.clearance>=radius+.75*self.pitch)
        labels,_=ndimage.label(valid,structure=np.ones((3,3,3),bool))
        sizes=np.bincount(labels.ravel());sizes[0]=0
        if sizes.max()==0:raise ValueError('No resolved interior region for reference-width connectors')
        self.valid=labels==int(sizes.argmax())
        self.cells=np.argwhere(self.valid)
        self.points=self.origin+self.cells*self.pitch;self.tree=cKDTree(self.points)
        print('INTERIOR_FIELD',self.shape,'usable',len(self.cells),'pitch',self.pitch,flush=True)

    def anchor(self,point,normal=None):
        point=np.asarray(point)
        for count in [64,512,4096]:
            _,ids=self.tree.query(point,k=min(count,len(self.points)));ids=np.atleast_1d(ids)
            for index in ids:
                center=self.points[index]
                if normal is not None and np.dot(center-point,normal)<-.0001:continue
                if volume(ball(center,self.radius)-self.domain)>1e-12:continue
                return center
        raise ValueError('No native-verified interior anchor near required contact')

    def fits(self,a,b):return volume(capsule(a,b,self.radius)-self.domain)<=1e-12

    def route(self,a,b):
        if self.fits(a,b):return np.array([a,b])
        start=tuple(np.rint((a-self.origin)/self.pitch).astype(int))
        end=tuple(np.rint((b-self.origin)/self.pitch).astype(int))
        valid=self.valid.copy()
        for attempt in range(24):
            costs=np.where(valid,1.+(.0025/np.maximum(self.clearance,.0008))**2,np.inf)
            solver=MCP_Geometric(costs,fully_connected=True)
            cumulative,_=solver.find_costs([start],ends=[end],find_all_ends=False)
            if not np.isfinite(cumulative[end]):raise ValueError('Interior grid has no connector route')
            cells=np.asarray(solver.traceback(end))
            if tuple(cells[0])!=start:cells=cells[::-1]
            points=self.origin+cells*self.pitch;selected=[points[0]];k=0;failed=None
            while k<len(points)-1:
                found=False
                for j in range(min(len(points)-1,k+32),k,-1):
                    if self.fits(points[k],points[j]):
                        selected.append(points[j]);k=j;found=True;break
                if not found:failed=k+1;break
            if failed is None:return np.asarray(selected)
            index=tuple(cells[failed])
            if index==end:index=tuple(cells[max(0,failed-1)])
            if index==start:raise ValueError('No exact-width departure from anchor')
            valid[index]=False
        raise ValueError('No native-certified path after conservative grid refinements')


def path_body(path,radius):return union([capsule(a,b,radius)for a,b in zip(path,path[1:])])


def build(stock,witness,body,rows,bounds,object_ports,calibration,certify,checkpoint=None,proposal_checkpoint=None):
    radius=max(.0015,calibration['guard_radius_m']+.0005)
    domain=bounded_domain(witness,body,rows,bounds)
    # Preserve the already certified reference's exact material at functional
    # boundaries; redundant recutting must not erase accepted floor vertices.
    allowed=solid(witness)
    domain=(domain+solid(stock))^allowed
    print('CONSTRUCTION_DOMAIN_TRIANGLES',domain.num_tri(),flush=True)
    field=InteriorField(domain,radius)
    floors,floor_records=ground_ports(stock,rows)
    descriptions=[dict(p,kind='object')for p in object_ports]+[dict(p,kind='ground')for p in floors]
    anchors=[];port_radii=[];port_data=[]
    for port in descriptions:
        point=np.asarray(port['center_m']);normal=port.get('normal_m')
        if port['kind']=='ground':
            normals=[r['fixture_transform'][2,:3]for r in rows
                if abs(r['fixture_transform'][2,:3]@point+r['fixture_transform'][2,3])<=1e-8]
            normal=np.sum(normals,axis=0)if normals else None
        center=field.anchor(point,None if normal is None else np.asarray(normal))
        distance=float(np.linalg.norm(center-point))+port.get('extent_m',0.)
        anchors.append(center);port_radii.append(max(radius+.001,distance+.0015))
        port_data.append(dict(port,anchor_m=center.tolist(),reach_m=distance))
    anchors=np.asarray(anchors);port_radii=np.asarray(port_radii)
    parent=list(range(len(anchors)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def join(a,b):parent[find(a)]=find(b)
    for i in range(len(anchors)):
        for j in range(i):
            if np.linalg.norm(anchors[i]-anchors[j])<1e-12:join(i,j)
    paths=[];tree_paths=[];history=[];core=union([ball(a,radius)for a in anchors])
    ports=union([ball(a,r)for a,r in zip(anchors,port_radii)])
    guide=ports;value=domain^guide
    edges=sorted((float(np.linalg.norm(anchors[i]-anchors[j])),i,j)
        for i in range(len(anchors))for j in range(i))
    for iteration in range(len(anchors)):
        # Overlapping large contact caps do not prove a full-width neck.
        # Join centers only through explicitly verified round connectors.
        groups=len({find(i)for i in range(len(anchors))})
        print('ANCHORED_GROUPS',iteration,groups,flush=True)
        if groups==1:break
        last_error=None;connected=False
        for _,i,j in edges:
            if find(i)==find(j):continue
            try:path=field.route(anchors[i],anchors[j])
            except (ValueError,RuntimeError)as error:last_error=str(error);continue
            paths.append(path);tree_paths.append((i,j,path));join(i,j)
            core=core+path_body(path,radius)
            guide=guide+path_body(path,radius+.001)
            value=domain^guide;connected=True
            history.append(dict(operation='full_width_anchored_connector',accepted=True,
                path_m=path.tolist(),minimum_radius_m=radius,exact_native_corridor_checked=True))
            if proposal_checkpoint:proposal_checkpoint(paths,history,radius)
            break
        if not connected:raise ValueError('Cannot connect interior anchors: '+str(last_error))
    from backbone_regularization import regularize
    core_mesh,certified_radius,regularization=regularize(unpack(core),anchors,tree_paths,radius,calibration['guard_radius_m'])
    core=solid(core_mesh)
    if regularization:history.append(dict(operation='native_inner_tree_shell_regularization',accepted=True,evidence=regularization))
    if len(material_components(core))!=1:raise ValueError('Minimum-width connector core is disconnected')
    def useful(value):
        # Normalize the actual Boolean result against the original witness.
        # Nearly coincident input faces can otherwise leave finite slivers.
        value=(value^allowed).simplify(0.)
        parts=material_components(value)
        scored=[]
        for part in parts:
            mesh=unpack(part);score=int(mesh.contains(anchors).sum());scored.append((score,volume(part),mesh))
        score,_,mesh=max(scored,key=lambda row:row[:2])
        if volume(core-solid(mesh))>1e-10:
            raise ValueError('Rounded support loses its full-width interior connector core')
        return mesh
    full=None;proof=None
    for expansion in [1.,1.25,1.5,2.]:
        expanded_ports=union([ball(a,r*expansion)for a,r in zip(anchors,port_radii)])
        proposal=domain^(expanded_ports+union([path_body(p,radius+.001)for p in paths]))
        try:
            full,proof=certify(useful(proposal));ports=expanded_ports;port_radii*=expansion
            history.append(dict(operation='certified_rounded_ports',accepted=True,expansion=expansion));break
        except (ValueError,RuntimeError)as error:
            history.append(dict(operation='certified_rounded_ports',accepted=False,expansion=expansion,error=str(error)))
    if full is None:raise ValueError('No rounded functional-cap size passed complete physical certification: '+str(history[-1]))
    if checkpoint:checkpoint(full,proof,history)
    for tube_radius in [radius+.00075,radius+.0005,radius+.00025,radius+.0001]:
        proposal=domain^(ports+union([path_body(p,tube_radius)for p in paths]))
        try:
            candidate,certificate=certify(useful(proposal))
            removed=(abs(float(full.volume))-abs(float(candidate.volume)))*1e6
            if removed<=1e-7:continue
            full,proof=candidate,certificate
            history.append(dict(operation='outside_in_round_bands',accepted=True,
                radius_m=tube_radius,removed_material_cm3=removed))
            if checkpoint:checkpoint(full,proof,history)
            print('ANCHORED_SHRINK',tube_radius,removed,flush=True)
        except (ValueError,RuntimeError)as error:history.append(dict(operation='outside_in_round_bands',accepted=False,error=str(error)))
    tube_radius=next((h['radius_m']for h in history[::-1]if h['operation']=='outside_in_round_bands'and h['accepted']),radius+.001)
    tubes=union([path_body(p,tube_radius)for p in paths])
    reaches=np.array([p['reach_m']for p in port_data])
    for margin in [.001,.0005,.00025]:
        radii=np.minimum(port_radii,np.maximum(radius+.0005,reaches+margin))
        candidate_ports=union([ball(a,r)for a,r in zip(anchors,radii)])
        try:
            candidate,certificate=certify(useful(domain^(candidate_ports+tubes)))
            removed=(abs(float(full.volume))-abs(float(candidate.volume)))*1e6
            if removed<=1e-7:continue
            full,proof=candidate,certificate;ports=candidate_ports;port_radii=radii
            history.append(dict(operation='outside_in_contact_cap_shrink',accepted=True,margin_m=margin,removed_material_cm3=removed))
            if checkpoint:checkpoint(full,proof,history)
            print('CONTACT_CAP_SHRINK',margin,removed,flush=True)
        except (ValueError,RuntimeError)as error:history.append(dict(operation='outside_in_contact_cap_shrink',accepted=False,error=str(error)))
    # Replace oversized spherical end regions by round, full-width stems.
    # Intersection with the accepted solid makes each operation a genuine
    # exterior deletion. Split a rejected batch so one necessary wide pad
    # cannot freeze all other object/ground contacts.
    cap_shapes=[ball(a,r)for a,r in zip(anchors,port_radii)]
    def slim_caps(indices):
        nonlocal full,proof,cap_shapes
        proposed=list(cap_shapes)
        for i in indices:
            proposed[i]=capsule(anchors[i],descriptions[i]['center_m'],radius+.0005)
        try:
            keep=union(proposed)+tubes
            candidate=useful(solid(full)^keep)
            removed=(abs(float(full.volume))-abs(float(candidate.volume)))*1e6
            if removed<=1e-6:return
            candidate,certificate=certify(candidate)
            full,proof=candidate,certificate;cap_shapes=proposed
            history.append(dict(operation='outside_in_round_contact_stems',accepted=True,
                port_indices=indices,radius_m=radius+.0005,removed_material_cm3=removed))
            if checkpoint:checkpoint(full,proof,history)
            print('CONTACT_STEMS',len(indices),removed,flush=True)
        except (ValueError,RuntimeError)as error:
            history.append(dict(operation='outside_in_round_contact_stems',accepted=False,
                port_indices=indices,error=str(error)))
            if len(indices)>1:
                half=len(indices)//2;slim_caps(indices[:half]);slim_caps(indices[half:])
    order=sorted(range(len(anchors)),key=lambda i:-port_radii[i])
    # An uncertain or invalid proposal is rejected without affecting the last
    # complete checkpoint. The full-demand certificate remains authoritative.
    try:slim_caps(order)
    except (ValueError,RuntimeError)as error:
        history.append(dict(operation='outside_in_round_contact_stems',accepted=False,error=str(error)))
    # All anchor balls and minimum-radius tubes remain mandatory during
    # subsequent physical surface rounding.
    record=dict(history=history,minimum_connector_radius_m=certified_radius,outer_connector_design_radius_m=radius,
        object_ports=len(object_ports),ground_ports=len(floors),ground_port_selection=floor_records,
        port_data=port_data,actual_round_geometry=True,exact_corridor_checks=True,
        shared_interior_junctions=True,local_pocket_cutting_used=False)
    record['minimum_connector_core_connected']=True
    return full,proof,record,unpack(core)
