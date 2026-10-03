"""Regression for export rounding that broke accepted continuous exits."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import trimesh

from step4_connect_support.deterministic_space import export_exact_obj


class ExactExportTests(unittest.TestCase):
    def test_preserve_near_zero_and_full_precision_boolean_vertices(self):
        mesh = trimesh.creation.box(extents=[.02, .03, .04])
        mesh.vertices[0] = [np.nextafter(.001, 0.), -2.493373007934512e-17,
                            np.nextafter(0., 1.)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'shape.obj'
            export_exact_obj(mesh, path)
            loaded = trimesh.load(path, force='mesh', process=False)
        np.testing.assert_array_equal(loaded.vertices, mesh.vertices)
        np.testing.assert_array_equal(loaded.faces, mesh.faces)


if __name__ == '__main__':
    unittest.main()
