"""Record contact orientation without a vertical-hemisphere eligibility filter.

All selected heads will belong to one rigid support. Local upward, downward and
horizontal pressures are admitted; this is not a bearing-equilibrium certificate.
"""
import numpy as np

NORMAL_Z_GUARD = 1e-10  # dimensionless numerical exclusion band at horizontal


def definition():
    return dict(force_convention='Object on support: lambda * outward object normal; lambda >= 0',
        world_up='+Z', mathematical_rule='No vertical hemisphere restriction',
        enforced=False, normal_z_guard=NORMAL_Z_GUARD,
        horizontal_allowed=True, upward_oblique_allowed=True,
        average_normal_used=False, opposing_directions_may_cancel=True,
        on_failure='Reject empty or nonfinite contact geometry only',
        scope='Orientation diagnostic; all heads must form one rigid body before sharing reactions')


def check(mesh, faces):
    ids = np.asarray(sorted(set(map(int, faces))), int)
    if not len(ids):
        return dict(passed=False, classification='empty_contact', checked_source_faces=0)
    normals = mesh.face_normals[ids]
    z = normals[:, 2]
    bad = ~np.isfinite(normals).all(axis=1)
    index = int(np.argmax(z))
    return dict(passed=not bool(np.any(bad)),
        classification='unrestricted_contact_orientation' if not np.any(bad) else 'nonfinite_normal',
        hemisphere_filter_enforced=False,
        non_downward_face_count=int(np.sum(z >= -NORMAL_Z_GUARD)),
        checked_source_faces=len(ids), violating_face_count=int(np.sum(bad)),
        minimum_outward_normal_z=float(z.min()), maximum_outward_normal_z=float(z.max()),
        normal_z_guard=NORMAL_Z_GUARD, weakest_source_face=int(ids[index]),
        weakest_outward_normal=normals[index].tolist())
