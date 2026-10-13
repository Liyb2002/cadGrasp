"""Optional Step5.2 trial: flattened shoulders taper into small ground feet.

Reuse saved foot locations, nominal material and the legal-space witness.
Only the added feet and connections change; all original acceptance checks
remain those of the deterministic construction backend.
"""
import numpy as np
import trimesh
from scipy.spatial import cKDTree

from deterministic_base import (Builder as OriginalBuilder, WIDTH, VOLUME_TOL,
    SCALE, solid, unpack, volume, union, moved, transform, components, enclosure)


def tapered_leg(root, tip, up, root_width=.022, root_thickness=.008,
                tip_width=.007, rotation=0.):
    """Closed elliptical loft with a short broad shoulder and a tapered end."""
    root, tip = np.asarray(root), np.asarray(tip)
    direction = tip-root
    length = float(np.linalg.norm(direction))
    if length <= 1e-9:
        raise ValueError('A tapered connection needs distinct endpoints')
    direction /= length
    across = np.cross(direction, up)
    if np.linalg.norm(across) < 1e-8:
        across = np.cross(direction, np.eye(3)[np.argmin(np.abs(direction))])
    across /= np.linalg.norm(across)
    thin = np.cross(direction, across)
    across, thin = (np.cos(rotation)*across+np.sin(rotation)*thin,
                    -np.sin(rotation)*across+np.cos(rotation)*thin)
    angles = np.arange(16)*2*np.pi/16
    ring = np.c_[np.cos(angles), np.sin(angles)]
    # The shoulder starts slightly inside the attachment surface. Keep its
    # broad/flat section briefly, then narrow continuously towards the foot.
    stations = [(-min(.003, length*.15), root_width, root_thickness),
                (length*.10, root_width, root_thickness),
                (length*.38, root_width*.72+tip_width*.28,
                 root_thickness*.72+tip_width*.28),
                (length*.72, root_width*.30+tip_width*.70,
                 root_thickness*.30+tip_width*.70),
                (length+min(.0015, length*.10), tip_width, tip_width)]
    vertices = []
    for distance, width, thickness in stations:
        vertices.extend(root+distance*direction+
                        ring[:, :1]*across*(width/2)+
                        ring[:, 1:]*thin*(thickness/2))
    faces = []
    for k in range(len(stations)-1):
        for j in range(16):
            a, b = 16*k+j, 16*k+(j+1)%16
            faces.extend([[a, b, b+16], [a, b+16, a+16]])
    for base, reverse in [(0, True), (16*(len(stations)-1), False)]:
        center = len(vertices)
        vertices.append(np.asarray(vertices[base:base+16]).mean(0))
        for j in range(16):
            a, b = base+j, base+(j+1)%16
            faces.append([center, b, a] if reverse else [center, a, b])
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    mesh.fix_normals()
    return solid(mesh)


class Builder(OriginalBuilder):
    def __init__(self, fixture, witness, rows, reference, construction,
                 root_width=.022, root_thickness=.008, foot_diameter=.008,
                 envelope_growth=.0015):
        super().__init__(fixture, witness, rows)
        # A certified pruned core within the original 1e-10 m3 construction
        # tolerance keeps its exact contact boundary. Redundant coincident
        # clipping can remove a numerically tiny but necessary reaction port.
        if self.removed <= 1e-10:
            self.legal_nominal = self.original
            self.body = self.original
            self.removed = 0.
        self.reference = reference
        self.source_construction = construction
        self.root_width = root_width
        self.root_thickness = root_thickness
        self.foot_diameter = foot_diameter
        self.envelope_growth = envelope_growth
        self.locations = next(row['foot_locations'] for row in construction['history']
                              if row['operation'] == 'ground_patches')
        # Bound the added material by the old complete fixture in EVERY final
        # installation; this limits overall dimensions, rather than just a
        # common-coordinate box that could enlarge a rotated installation.
        seen = set()
        for row in rows:
            T = row['fixture_transform']
            key = np.round(T, 11).tobytes()
            if key in seen:
                continue
            seen.add(key)
            points = transform(reference.vertices, T)
            low = points.min(0)-envelope_growth
            high = points.max(0)+envelope_growth
            for axis in range(3):
                self.allowed = self.allowed.trim_by_plane(
                    tuple(T[axis, :3]), float((low[axis]-T[axis, 3])/SCALE))
                self.allowed = self.allowed.trim_by_plane(
                    tuple(-T[axis, :3]), float((T[axis, 3]-high[axis])/SCALE))
        if volume(self.legal_nominal-self.allowed) > 1e-10:
            raise ValueError('The size limit would remove protected nominal material')
        self.nominal_mesh = unpack(self.legal_nominal)
        self.feet = []

    def ground(self):
        import manifold3d as md
        for location in self.locations:
            T = np.asarray(location['fixture_transform'])
            p = np.asarray(location['world_xy_m'])
            pad = md.Manifold.cylinder(.004/SCALE, self.foot_diameter/2/SCALE,
                                      circular_segments=20)
            pad = pad.translate((p[0]/SCALE, p[1]/SCALE, 0.))
            pad = moved(pad, np.linalg.inv(T)) ^ self.allowed
            if volume(pad) <= VOLUME_TOL:
                raise ValueError('Saved foot location has no legal pad material')
            self.feet.append(dict(solid=pad, location=location,
                center=transform(np.array([[p[0], p[1], .002]]), np.linalg.inv(T))[0],
                up=T[2, :3].copy()))
        self.body = union([self.legal_nominal, *[row['solid'] for row in self.feet]])
        if not all(row['passed'] for row in enclosure(unpack(self.body), self.rows)):
            raise ValueError('Rounded feet do not enclose every saved pressure hull')
        self.history.append(dict(operation='ground_patches',
            foot_locations=self.locations, foot_count=len(self.feet),
            foot_profile='rounded pad', foot_diameter_m=self.foot_diameter,
            foot_height_m=.004, saved_foot_positions_reused=True,
            component_count=len(components(self.body))))

    def attachment_points(self, mesh, tip):
        nearest, _, _ = trimesh.proximity.closest_point_naive(mesh, tip[None])
        centers = mesh.triangles_center
        _, ids = cKDTree(centers).query(tip, k=min(16, len(centers)))
        candidates = np.vstack([nearest, centers[np.atleast_1d(ids)]])
        result = []
        for point in candidates:
            if not any(np.linalg.norm(point-old) < .001 for old in result):
                result.append(point)
        return result

    def connect(self):
        # Attach each foot to the original body where possible. This avoids
        # thin foot-to-foot chains and gives each leg a broad body shoulder.
        pending = sorted(self.feet, key=lambda foot: float(
            cKDTree(self.nominal_mesh.vertices).query(foot['center'])[0]))
        for number, foot in enumerate(pending):
            before = len(components(self.body))
            if before <= 1:
                break
            connected = False
            for source, mesh in [('nominal', self.nominal_mesh),
                                 ('connected_material', unpack(self.body-foot['solid']))]:
                candidates = self.attachment_points(mesh, foot['center'])
                for root in candidates:
                    for angle in [0., np.pi/2, np.pi/4]:
                        proposal = tapered_leg(root, foot['center'], foot['up'],
                            self.root_width, self.root_thickness,
                            self.foot_diameter*.875, angle) ^ self.allowed
                        for piece in components(proposal):
                            if volume(piece ^ foot['solid']) <= VOLUME_TOL:
                                continue
                            candidate = self.body+piece
                            if len(components(candidate)) >= before:
                                continue
                            added = volume(candidate)-volume(self.body)
                            self.body = candidate
                            self.history.append(dict(operation='tapered_support_leg',
                                foot_index=number, attachment_source=source,
                                root_m=root.tolist(), tip_m=foot['center'].tolist(),
                                root_width_m=self.root_width,
                                root_thickness_m=self.root_thickness,
                                tip_width_m=self.foot_diameter*.875,
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
                raise ValueError('No legal tapered connection for saved foot '+str(number))
        if len(components(self.body)) != 1:
            raise ValueError('Tapered construction has disconnected material')

    def run(self):
        mesh, record = super().run()
        record.update(leg_profile='flattened broad shoulder tapering to rounded foot',
            root_width_m=self.root_width, root_thickness_m=self.root_thickness,
            foot_diameter_m=self.foot_diameter,
            maximum_installed_envelope_growth_per_side_m=self.envelope_growth,
            saved_foot_positions_reused=True)
        if not record['passed'] or not record['watertight']:
            raise ValueError('Tapered solid failed original construction acceptance')
        return mesh, record
