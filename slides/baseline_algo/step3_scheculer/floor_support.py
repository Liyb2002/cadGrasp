"""Sufficient floor friction at the original workpiece contact point.

The four-ray pyramid is the same finite inner approximation used by Step5
at its largest tested coefficient. It permits compression and bounded shear,
but neither tensile floor reactions nor an independent contact moment.
"""
import numpy as np

COEFFICIENT = 64.0


def rays():
    mu = COEFFICIENT
    return np.array([[mu, 1., 0.], [-mu, 1., 0.],
                     [0., 1., mu], [0., 1., -mu]])


def columns(point, com):
    forces = rays()
    return np.c_[forces, np.cross(np.asarray(point)-com, forces)]


def description():
    return dict(model='four_ray_coulomb_inner_pyramid', coefficient=COEFFICIENT,
                location='original_workpiece_floor_contact', unilateral=True,
                inequality='abs(Fx) + abs(Fz) <= coefficient * Fy; Fy >= 0',
                independent_contact_moment=False,
                scope='Finite sufficient-friction assumption, matching the largest Step5 coefficient; not a measured material coefficient.')
