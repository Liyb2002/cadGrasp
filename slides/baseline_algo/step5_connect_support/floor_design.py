"""Step5 owns demand hull construction and all choices of floor material."""
from step1.needs import COORD
import numpy as np
from scipy.spatial import ConvexHull

def boundary(points):
    points = np.unique(np.asarray(points, float).reshape(-1, 2), axis=0)
    if len(points) < 3 or np.linalg.matrix_rank(points-points.mean(axis=0)) < 2: return points
    return points[ConvexHull(points).vertices]

def support_polygon(required, pivot):
    """Closed convex envelope in z=0; no arbitrary margin or solid material."""
    vertices = boundary(np.vstack([required, COORD.floor(pivot)]))
    if len(vertices) < 3 or np.linalg.matrix_rank(vertices-vertices.mean(axis=0)) < 2:
        raise ValueError('A closed floor region requires a nondegenerate 2D demand')
    return vertices, np.vstack([vertices, vertices[0]])



def prepare(points):
    result = dict(points)
    result['required_hull_xy_m'] = boundary(np.vstack([points['floor_demands_xy_m'], points['continuous_floor_enclosure_xy_m']]))
    polygon, loop = support_polygon(result['required_hull_xy_m'], points['original_pivot_m'])
    result['support_polygon_xy_m'] = polygon
    return result
