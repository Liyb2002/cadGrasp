"""Step5.2 profile trial: planar flattened pyramids with integral ground ends."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial import ConvexHull
from shapely.geometry import MultiPoint,Point
import trimesh

from tapered_base import Builder as RoundedBuilder
from deterministic_base import (solid, unpack, volume, components, box,
                                moved, transform, VOLUME_TOL)


def flat_leg(root, tip, up, root_width=.034, root_thickness=.009,
             tip_width=.010, tip_thickness=.006, rotation=0.):
    """A flat rectangular shoulder tapers directly to a rectangular floor end.

    The final section lies on the actual floor. It is part of the leg itself;
    no cylinder, extra pad, raised collar or separate terminal block is added.
    """
    root, tip, up = np.asarray(root), np.asarray(tip), np.asarray(up)
    direction = tip-root
    length = float(np.linalg.norm(direction))
    if length <= 1e-9:
        raise ValueError('A flat tapered leg needs distinct endpoints')
    direction /= length
    across = np.cross(direction, up)
    if np.linalg.norm(across) < 1e-8:
        across = np.cross(direction, np.eye(3)[np.argmin(np.abs(direction))])
    across /= np.linalg.norm(across)
    thin = np.cross(direction, across)
    root_across = np.cos(rotation)*across+np.sin(rotation)*thin
    root_thin = -np.sin(rotation)*across+np.cos(rotation)*thin
    floor_thin = np.cross(up, across)
    floor_thin /= np.linalg.norm(floor_thin)
    corners = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    root_center = root-direction*min(.003, length*.15)
    shoulder = root+direction*length*.06
    vertices = []
    for center in [root_center, shoulder]:
        vertices.extend(center+corners[:, :1]*root_across*root_width/2+
                        corners[:, 1:]*root_thin*root_thickness/2)
    vertices.extend(tip+corners[:, :1]*across*tip_width/2+
                    corners[:, 1:]*floor_thin*tip_thickness/2)
    # The convex hull retains planar sides and makes the shoulder/floor
    # orientation independent of the selected oblique connection direction.
    mesh = trimesh.Trimesh(vertices=vertices, process=False).convex_hull
    return solid(mesh)


class Builder(RoundedBuilder):
    def __init__(self, fixture, witness, rows, reference, construction,
                 root_width=.034, root_thickness=.009, tip_width=.010,
                 tip_thickness=.006, envelope_growth=.0015,
                 object_mesh=None,object_placements=None):
        super().__init__(fixture, witness, rows, reference, construction,
            root_width=root_width, root_thickness=root_thickness,
            foot_diameter=tip_width, envelope_growth=envelope_growth)
        self.tip_width = tip_width
        self.tip_thickness = tip_thickness
        self.actual_object_exclusions=False
        self.object_solids=[]
        if object_mesh is not None:
            workpiece=object_mesh.copy()
            workpiece.merge_vertices(digits_vertex=15)
            original_object=solid(workpiece)
            self.object_solids=[moved(original_object,np.asarray(Q))for Q in object_placements]
            self.actual_object_exclusions=True

    def clip_proposal(self,proposal):
        clipped=proposal^self.allowed
        # Apply exact workpiece exclusions to each small new connection.
        # Recutting the entire saved witness along coincident body boundaries
        # can create spurious large inverse shells.
        for body in self.object_solids:
            clipped=clipped-body
        return clipped

    def ground(self):
        # Pruning can remove floor terminals originally supplied by the
        # nominal shell, absent from the old added-feet list. Select required
        # sites from the accepted solid's actual floor boundary instead.
        locations=[]
        distinct={np.round(r['fixture_transform'],11).tobytes():r['fixture_transform'] for r in self.rows}
        original=unpack(self.legal_nominal)
        for T in distinct.values():
            world=transform(self.reference.vertices,T)
            terminals=world[np.abs(world[:,2])<=1e-8]
            xy,indices=np.unique(terminals[:,:2],axis=0,return_index=True)
            if len(xy)<3:
                continue
            hull_ids=ConvexHull(xy).vertices
            selected=xy[hull_ids]
            selected_z=terminals[indices[hull_ids],2]
            current=transform(original.vertices,T)
            existing=current[np.abs(current[:,2])<=1e-8,:2]
            relevant=[r for r in self.rows if np.allclose(T,r['fixture_transform'],atol=1e-11,rtol=0)]
            order=sorted(range(len(selected)),key=lambda k:(-float(cKDTree(current[:,:2]).query(selected[k])[0]),k))
            active=set(range(len(selected)))
            for k in order:
                remaining=np.array([selected[j] for j in sorted(active-{k})]).reshape(-1,2)
                ok=True
                for row in relevant:
                    combined=np.vstack([remaining,existing,row['object_ground']])
                    region=MultiPoint(combined).convex_hull if len(combined) else MultiPoint([])
                    if region.is_empty or any(region.distance(Point(p[:2]))>1e-8 for p in row['hull']):
                        ok=False
                        break
                if ok:
                    active.remove(k)
            for k in sorted(active):
                locations.append(dict(world_xy_m=selected[k].tolist(),world_z_m=float(selected_z[k]),
                                      fixture_transform=T.tolist()))
        self.locations=locations
        # Only store the desired floor ends. No standalone foot material is
        # added; the tapered legs themselves must reach these original sites.
        self.feet = []
        for location in self.locations:
            T = np.asarray(location['fixture_transform'])
            p = np.asarray(location['world_xy_m'])
            tip = transform(np.array([[p[0], p[1], location.get('world_z_m',0.)]]), np.linalg.inv(T))[0]
            # This shallow region is a connection predicate, never material.
            target = moved(box(np.array([p[0]-.007, p[1]-.007, 0.]),
                               np.array([p[0]+.007, p[1]+.007, .001])),
                           np.linalg.inv(T))
            self.feet.append(dict(location=location, center=tip, up=T[2, :3],
                                  T=T, target=target))
        self.body = self.legal_nominal
        self.history.append(dict(operation='ground_patches',
            foot_locations=self.locations, foot_count=len(self.feet),
            foot_profile='integral rectangular tapered-leg end',
            separate_foot_pads_added=False, saved_foot_positions_reused=False,
            accepted_reference_floor_boundary_used=True,
            tip_width_m=self.tip_width, tip_thickness_m=self.tip_thickness,
            component_count=len(components(self.body))))

    def connect(self):
        # Required contact islands are joined before choosing leg roots, so
        # these legal internal connections can also carry the broad shoulders.
        from deterministic_base import Builder as OriginalBuilder
        if len(components(self.body)) > 1:
            OriginalBuilder.connect(self)
        pending = sorted(self.feet, key=lambda foot: float(
            cKDTree(self.nominal_mesh.vertices).query(foot['center'])[0]))
        for number, foot in enumerate(pending):
            world = transform(unpack(self.body).vertices, foot['T'])
            ground = world[np.abs(world[:, 2]) <= 1e-8, :2]
            center = np.asarray(foot['location']['world_xy_m'])
            region=MultiPoint(ground).convex_hull if len(ground) else MultiPoint([])
            if not region.is_empty and region.distance(Point(center))<=1e-8:
                self.history.append(dict(operation='reuse_existing_ground_terminal',foot_index=number))
                continue
            connected = False
            alternatives = []
            for source, mesh in [('nominal', self.nominal_mesh),
                                 ('connected_material', unpack(self.body)),
                                 ('accepted_reference',self.reference)]:
                for root in self.attachment_points(mesh, foot['center']):
                    if np.linalg.norm(root-foot['center'])<=1e-9:
                        continue
                    for angle in [0., np.pi/4, np.pi/2]:
                        proposal = self.clip_proposal(flat_leg(root, foot['center'], foot['up'],
                            self.root_width, self.root_thickness,
                            self.tip_width, self.tip_thickness, angle))
                        for piece in components(proposal):
                            if volume(piece ^ foot['target']) <= VOLUME_TOL:
                                continue
                            world = transform(unpack(piece).vertices, foot['T'])
                            contact = world[np.abs(world[:, 2]) <= 1e-8, :2]
                            center = np.asarray(foot['location']['world_xy_m'])
                            contact = contact[np.linalg.norm(contact-center, axis=1) <= .012]
                            if len(contact) < 3:
                                continue
                            if volume(piece ^ self.body) <= VOLUME_TOL:
                                distance=float(cKDTree(unpack(self.body).vertices).query(unpack(piece).vertices)[0].min())
                                alternatives.append((distance,volume(piece),piece,root,source))
                                continue
                            candidate = self.body+piece
                            if len(components(candidate)) > len(components(self.body)):
                                continue
                            added = volume(candidate)-volume(self.body)
                            self.body = candidate
                            self.history.append(dict(operation='flat_tapered_support_leg',
                                foot_index=number, attachment_source=source,
                                root_m=root.tolist(), tip_m=foot['center'].tolist(),
                                root_width_m=self.root_width,
                                root_thickness_m=self.root_thickness,
                                tip_width_m=self.tip_width, tip_thickness_m=self.tip_thickness,
                                separate_foot_pad_added=False, planar_sides=True,
                                added_cm3=added*1e6, component_count=len(components(candidate))))
                            connected = True
                            break
                        if connected:
                            break
                    if connected:
                        break
                if connected:
                    break
            if not connected:
                # A straight loft can be cut into pieces by the shared work
                # exclusions. Keep its actual flat floor end, then join it
                # through the same bounded legal bridge family as the core.
                previous,history=self.body,len(self.history)
                for _,_,piece,root,source in sorted(alternatives,key=lambda row:row[:2])[:8]:
                    self.body=previous+piece
                    try:
                        OriginalBuilder.connect(self)
                    except ValueError:
                        self.body=previous
                        del self.history[history:]
                        continue
                    self.history.append(dict(operation='flat_tapered_support_leg',
                        foot_index=number,attachment_source=source,legal_connection_routed=True,
                        root_m=root.tolist(),tip_m=foot['center'].tolist(),
                        root_width_m=self.root_width,root_thickness_m=self.root_thickness,
                        tip_width_m=self.tip_width,tip_thickness_m=self.tip_thickness,
                        separate_foot_pad_added=False,planar_sides=True,
                        added_cm3=(volume(self.body)-volume(previous))*1e6,component_count=1))
                    connected=True
                    break
                if not connected:
                    raise ValueError('No legal flat tapered leg reaching ground site '+str(number))
        if len(components(self.body)) > 1:
            OriginalBuilder.connect(self)

    def run(self):
        mesh, record = super().run()
        record.pop('foot_diameter_m', None)
        record.update(leg_profile='planar flattened rectangular pyramid with integral floor end',
            root_width_m=self.root_width, root_thickness_m=self.root_thickness,
            tip_width_m=self.tip_width, tip_thickness_m=self.tip_thickness,
            separate_foot_pads_added=False, circular_sections=False,
            planar_sides=True,saved_foot_positions_reused=False,
            accepted_reference_floor_boundary_used=True,
            exact_original_object_interiors_excluded=self.actual_object_exclusions)
        return mesh, record
