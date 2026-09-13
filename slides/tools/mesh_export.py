"""Export display meshes accepted by MuJoCo; shared by area and trajectory."""
import numpy as np


def export(piece, path):
    """MuJoCo refuses a coplanar mesh (METHOD s7); jitter exactly as
    `shrink_support.paint` does before writing.

    It also refuses FEWER THAN FOUR VERTICES -- it hulls every mesh geom, and
    a hull needs a fourth point.  A ONE-TRIANGLE piece is exactly that, and B
    makes them (2026-08-22): its floor band is a 1.5 mm strip cut out of an
    11 mm tessellation, so on poses 0, 1 and 5 the band is a single face.  No
    earlier workpiece could produce one -- the cuboid's and A1-f's bands are
    either empty (the `sel.any()` guard catches those) or nine faces and up.
    Answered the same way and at the same scale as the coplanarity jitter: a
    fourth point 20 um off the face along its own normal, invisible at render
    scale, enough for the hull.  Never fires for a piece that already has four
    vertices, so it cannot move an existing figure."""
    if len(piece.vertices) < 4:
        import trimesh as _tm
        piece = _tm.Trimesh(
            vertices=np.vstack([piece.vertices, piece.triangles_center[0]
                                + 2e-5 * piece.face_normals[0]]),
            faces=np.vstack([piece.faces, [[0, 1, len(piece.vertices)]]]),
            process=False)
    v = piece.vertices - piece.vertices.mean(axis=0)
    if np.linalg.matrix_rank(v, tol=1e-9 * max(piece.scale, 1e-9)) < 3:
        piece.vertices[::2] += 2e-5 * piece.face_normals[0]
    piece.export(path)
