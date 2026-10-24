"""Admission of actual contact heads to the saved Step 2 access domain."""
from itertools import product
import numpy as np

from step2_local_support import geometry as G, insertion as D, surface as S
from step2_local_support.support_policy import skipped_access_check


class ContactClearance:
    def __init__(self, mesh, depth, work, offsets=None):
        self.mesh, self.depth, self.work = mesh, depth, work
        self.offsets = G.vertex_offsets(mesh, depth)[0] if offsets is None else offsets

    def check_parts(self, heads):
        if self.work is None:
            return skipped_access_check()
        if not heads:
            return dict(passed=False, classification='empty_contact', component='contact_head')
        # A separated enclosing box proves separation of every enclosed head.
        # A box collision says nothing about the actual nonconvex union.
        points = np.vstack([h.vertices for h in heads])
        bounds = np.array(list(product(*zip(points.min(axis=0), points.max(axis=0)))))
        bound = self.work.check_piece(G.hull_mesh(bounds))
        if bound['passed']:
            result = dict(bound, method='separated_enclosing_box', part_count=len(heads))
        else:
            result = dict(self.work.check_parts(heads), method='actual_joined_head_cells')
        return dict(result, component='contact_and_joined_head',
            contact_surface_included=True, normal_depth_m=self.depth,
            scope='Installed head including its complete contact boundary; does not certify a connector to ground')

    def check(self, polygons):
        if self.work is None:
            return skipped_access_check()
        heads = [D.engine.hull_mesh(G.head_cell(self.mesh, p, int(f), self.offsets))
                 for f, p in polygons.items()]
        result = self.check_parts(heads)
        if not result['passed'] and 'part_index' in result:
            face = list(polygons)[result['part_index']]
            result['source_face'] = int(face)
            # Explain whether this failing cell already conflicts at its interface.
            triangles = S.fan(polygons[face])
            result['interface_check'] = self.work.check_surface(triangles, [face]*len(triangles))
        return result


def apply_constraint(local, check):
    """Keep the initial fitted geometry; a failure changes eligibility only."""
    if check['passed']:
        return dict(local, work_volume_check=check)
    kind = 'collision' if check['classification'] == 'circular_cone_intersection' else 'unresolved'
    return dict(local, valid=False, status=f'work_volume_{kind}', work_volume_check=check)
