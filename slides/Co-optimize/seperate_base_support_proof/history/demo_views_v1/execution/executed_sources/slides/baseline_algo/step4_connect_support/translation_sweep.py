"""Finite translation sweep of a closed oriented triangular solid.

This constructs a continuous sweep, not a union of sampled placements.  In
exact arithmetic S + [0,d] equals S union the prisms F + [0,d] over boundary
triangles whose outward normal has positive dot product with d.  Intersect a
line parallel to d with S: each occupied interval grows from its forward end,
and those forward ends lie on the selected boundary faces.  This argument also
applies to nonconvex solids, multiple components, and correctly oriented cavity
boundaries. Tangential faces add no volume.

All Boolean inputs use Mesh64 in one normalized frame. No transverse band,
offset, enlargement, contact-face movement, or Boolean simplification is added.
"Exact" describes the continuous geometric construction, not exact arithmetic:
manifold3d still uses floating point, and nearly coplanar/sliver features may be
unresolved. Positive-volume input prisms that collapse are rejected explicitly.
Do not treat tiny Boolean residuals as a manufacturing clearance certificate.
"""
from __future__ import annotations

import numpy as np
import trimesh
import manifold3d as md


_PRISM_FACES = np.array([
    [2, 1, 0], [3, 4, 5],
    [0, 1, 4], [0, 4, 3],
    [1, 2, 5], [1, 5, 4],
    [2, 0, 3], [2, 3, 5],
], dtype=np.uint64)


def _solid(vertices, faces, label):
    value = md.Manifold(md.Mesh64(
        np.ascontiguousarray(vertices, dtype=np.float64),
        np.ascontiguousarray(faces, dtype=np.uint64),
    ))
    if value.status() != md.Error.NoError:
        raise ValueError(f'{label}: manifold input rejected: {value.status()}')
    return value


def _union(parts, fan_in=8):
    # A bounded fan-in limits the temporary Boolean expression size.
    while len(parts) > 1:
        parts = [md.Manifold.batch_boolean(parts[i:i+fan_in], md.OpType.Add)
                 for i in range(0, len(parts), fan_in)]
        for part in parts:
            if part.status() != md.Error.NoError:
                raise RuntimeError(f'Sweep Boolean unresolved: {part.status()}')
    return parts[0]


def swept_solid(mesh: trimesh.Trimesh, displacement, *, fan_in=8) -> trimesh.Trimesh:
    """Return the whole closed solid swept along ``t*displacement``, 0<=t<=1.

    ``mesh`` must have outward-facing, consistently wound closed boundaries;
    cavities must use their inward-facing cavity boundary orientation.  The
    returned mesh is in the original coordinates and units. Input geometry is
    never repaired, merged, offset, or modified in-place.
    """
    displacement = np.asarray(displacement, dtype=np.float64)
    if not isinstance(fan_in, int) or fan_in < 2:
        raise ValueError('Boolean fan-in must be an integer >= 2')
    if displacement.shape != (3,) or not np.isfinite(displacement).all():
        raise ValueError('displacement must be a finite 3-vector')
    if not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError('An empty mesh has no solid sweep')
    if not np.isfinite(mesh.vertices).all():
        raise ValueError('Mesh contains non-finite vertices')
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
        raise ValueError('Expected a closed consistently wound positive-volume solid')
    if np.all(displacement == 0):
        return mesh.copy()

    origin = np.asarray(mesh.bounds).mean(axis=0)
    scale = max(float(np.max(mesh.extents)), float(np.linalg.norm(displacement)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid geometric scale')
    vertices = (np.asarray(mesh.vertices) - origin) / scale
    delta = displacement / scale
    initial = _solid(vertices, mesh.faces, 'Original solid')
    triangles = vertices[np.asarray(mesh.faces)]

    # Classify using unnormalized area normals in extended precision. There is
    # deliberately no angular threshold that could omit a genuinely leading
    # sliver. Zero means zero under the input-coordinate arithmetic, not a band.
    extended = triangles.astype(np.longdouble)
    area_normals = np.cross(extended[:, 1]-extended[:, 0],
                            extended[:, 2]-extended[:, 0])
    signed = area_normals @ delta.astype(np.longdouble)
    leading = np.flatnonzero(signed > 0)
    pieces = [initial]
    for index in leading:
        triangle = triangles[index]
        prism = _solid(np.vstack([triangle, triangle + delta]),
                       _PRISM_FACES, f'Face prism {index}')
        if prism.is_empty() or prism.volume() <= 0:
            raise RuntimeError(f'Positive leading prism {index} collapsed numerically')
        pieces.append(prism)
    joined = _union(pieces, fan_in)
    result = joined.to_mesh64()
    swept = trimesh.Trimesh(
        np.asarray(result.vert_properties[:, :3]) * scale + origin,
        np.asarray(result.tri_verts), process=False,
    )
    if not swept.is_watertight or not swept.is_winding_consistent or swept.volume <= 0:
        raise RuntimeError('Sweep output is not a closed positive-volume solid')
    swept.metadata['translation_sweep'] = dict(
        method='initial solid plus all positive-normal face prisms',
        displacement=displacement.tolist(), leading_face_count=len(leading),
        scale=scale, transverse_padding=0.0, arithmetic='Mesh64; floating point',
        boolean_fan_in=fan_in,
    )
    return swept


def _mesh(value):
    data = value.to_mesh64()
    return trimesh.Trimesh(np.asarray(data.vert_properties[:, :3]),
                           np.asarray(data.tri_verts), process=False)


def _test():
    """Independent reference: exact convex sweeps of a known box decomposition."""
    import json

    def block(low, high):
        low, high = np.asarray(low, float), np.asarray(high, float)
        value = trimesh.creation.box(high-low)
        value.apply_translation((low+high)/2)
        return value

    examples = [
        ('nonconvex_L_axis', [block([0, 0, 0], [2, 1, 1]),
                             block([0, 0, 0], [1, 2, 1])], [0.75, 0, 0], 4.5),
        ('nonconvex_L_diagonal', [block([0, 0, 0], [2, 1, 1]),
                                 block([0, 0, 0], [1, 2, 1])], [.7, .6, .4], None),
        ('ring_with_through_hole', [block([0, 0, 0], [3, 1, 1]),
                                    block([0, 2, 0], [3, 3, 1]),
                                    block([0, 0, 0], [1, 3, 1]),
                                    block([2, 0, 0], [3, 3, 1])], [.4, -.3, .2], None),
    ]
    rows = []
    for name, boxes, delta, expected_volume in examples:
        source = _mesh(_union([_solid(b.vertices, b.faces, 'Test box') for b in boxes]))
        generated = swept_solid(source, delta)
        swept_boxes = [trimesh.convex.convex_hull(
            np.vstack([b.vertices, b.vertices + np.asarray(delta)])) for b in boxes]
        reference = _union([_solid(b.vertices, b.faces, 'Reference swept box')
                            for b in swept_boxes])
        result = _solid(generated.vertices, generated.faces, 'Result')
        extra = abs(float((result-reference).volume()))
        missing = abs(float((reference-result).volume()))
        tolerance = max(source.extents.max(), np.linalg.norm(delta))**3 * 1e-10
        assert extra < tolerance and missing < tolerance, (name, extra, missing)
        if expected_volume is not None:
            assert abs(generated.volume-expected_volume) < tolerance
        rows.append(dict(case=name, source_volume=float(source.volume),
                         sweep_volume=float(generated.volume),
                         reference_volume=float(reference.volume()),
                         extra_volume=extra, missing_volume=missing,
                         watertight=bool(generated.is_watertight)))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    _test()
