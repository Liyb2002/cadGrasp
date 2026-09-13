"""Maintain a nonempty common 3-D withdrawal set through greedy selection and sizing."""
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from step3_scheculer import contacts as I
from step2_local_support import withdrawal as D
from step2_local_support import insertion_directions as catalogue
from step3_scheculer.stage_imports import load_stage
area_limit = load_stage('optimize', 'area_limit')


def code_hashes():
    return dict(D.code_hashes(), **I.hashes([Path(__file__), Path(catalogue.__file__), Path(area_limit.__file__)]))


def candidate_filter(rows, selected, geometry_valid, areas=None, minimum_area=None, common_allowed=None):
    if (areas is None) != (minimum_area is None):
        raise ValueError('Area filtering requires both candidate areas and the minimum')
    selected = set(selected)
    eligible = np.zeros(len(geometry_valid), bool)
    records = []
    for index, row in enumerate(rows):
        assert row['candidate_index'] == index
        allowed = row['certified_directions']
        remaining = D.intersect(common_allowed, allowed) if common_allowed is not None else allowed
        if index in selected:
            reason = 'already_selected'
        elif not geometry_valid[index]:
            reason = 'rejected_step2_geometry'
        elif areas is not None and not areas[index] > minimum_area:
            reason = 'initial_area_not_above_minimum'
        elif not D.nonempty(allowed):
            reason = 'no_certified_individual_direction'
        elif not D.nonempty(remaining):
            reason = 'would_lock_last_common_direction'
        else:
            reason = 'eligible'
            eligible[index] = True
        records.append(dict(candidate_index=index, candidate_id=row['candidate_id'],
                            eligible=bool(eligible[index]), reason=reason,
                            certified_directions=allowed, remaining_if_selected=remaining))
        if areas is not None:
            records[-1].update(initial_area_m2=float(areas[index]), minimum_area_m2=float(minimum_area))
    return eligible, records


def candidate_areas(problem):
    return np.array([problem.data.triangle_areas[a:b].sum()
                     for a,b in zip(problem.data.offsets[:-1],problem.data.offsets[1:])])


@contextmanager
def scoring_mask(problem, eligible, selected):
    """Only change the input eligibility during 3.1; fixed contacts stay valid."""
    original = problem.data.valid
    active = np.asarray(eligible, bool).copy()
    active[list(selected)] = True
    assert np.all(~active | original)
    problem.data.valid = active
    try:
        yield
    finally:
        problem.data.valid = original


def inputs_match(report, paths):
    recorded = report['provenance']['inputs']
    return all(recorded.get(key) == digest for key, digest in I.hashes(paths).items())


class Tracker:
    def __init__(self, problem, out):
        self.problem, self.out = problem, Path(out)
        self.catalogue = catalogue.ensure(problem.name)
        self.catalogue_path = catalogue.path(problem.name)
        self.direction_catalogue = self.catalogue['direction_catalogue']
        self.current = dict(contacts=[], common_directions=self.direction_catalogue['global_allowed_directions'])
        self.current_path = None
        self.filter_paths = []
        self.analyzer = None
        self.areas = candidate_areas(problem)
        self.minimum_area = area_limit.MIN_AREA_FRACTION*float(problem.domain.mesh.area)

    def prepare(self, number):
        selected = [p['candidate_index'] for p in self.current['contacts']]
        eligible, rows = candidate_filter(self.catalogue['candidates'], selected, self.problem.data.valid,
                                         self.areas, self.minimum_area, self.current['common_directions'])
        inputs = [self.catalogue_path] + ([self.current_path] if self.current_path else [])
        path = self.out/f'round_{number:03d}'/'candidate_filter.json'
        result = dict(object=self.problem.name, complete=True, round=number,
                      selected_indices=selected, eligible=eligible.tolist(),
                      eligible_count=int(eligible.sum()), candidates=rows,
                      mode=D.MODE,
                      common_before_selection=self.current['common_directions'],
                      direction_catalogue=self.direction_catalogue,
                      minimum_contact_area_fraction=area_limit.MIN_AREA_FRACTION,
                      minimum_contact_area_m2=self.minimum_area,
                      previous_direction_record=str(self.current_path.relative_to(I.ROOT)) if self.current_path else None,
                      rule='Keep only candidates above the area minimum whose complete-head directions intersect the surviving common directions',
                      size_mismatch_policy='Optimize while preserving a common direction, then recompute the actual exported geometry.',
                      provenance=dict(inputs=I.hashes(inputs), code=code_hashes()))
        I.save(path, result)
        self.filter_paths.append(path)
        print(self.problem.name, 'round', number, 'insertion-eligible candidates', int(eligible.sum()), flush=True)
        return path, eligible, selected

    def update(self, contacts_path, state_path, number):
        """Re-evaluate changed geometry only; never substitute its initial area."""
        path = contacts_path.parent/'insertion_directions.json'
        inputs = [self.catalogue_path, contacts_path, state_path]
        if self.current_path:
            inputs.append(self.current_path)
        provenance = dict(inputs=I.hashes(inputs), code=code_hashes())
        actual = I.read_contacts(contacts_path)
        depth = self.catalogue['normal_depth_m']
        signatures = [D.signature(p, depth) for p in actual]
        try:
            cached = I.check_report(path)
            assert cached['provenance'] == provenance
            assert [p['geometry_signature'] for p in cached['contacts']] == signatures
        except (OSError, RuntimeError, AssertionError):
            cached = None
        if cached is None:
            previous = {p['geometry_signature']: p for p in self.current['contacts']}
            records = []
            for contact, signature in zip(actual, signatures):
                initial = self.catalogue['candidates'][contact['candidate_index']]
                record = previous.get(signature)
                if record is None and signature == initial.get('geometry_signature'):
                    record = initial
                if record is None:
                    if self.analyzer is None:
                        self.analyzer = D.Analyzer(self.problem.domain.mesh, depth, self.direction_catalogue)
                    record = self.analyzer.analyze(contact)
                record = dict(record, initial_step2_radius_m=initial['radius_m'],
                              differs_from_step2_geometry=signature != initial['geometry_signature'])
                record = dict(record, representative=D.representative(record['certified_directions'], self.direction_catalogue))
                records.append(record)
            common = D.common(records, self.direction_catalogue['global_allowed_directions'])
            valid = D.nonempty(common)
            cached = dict(object=self.problem.name, complete=True, round=number,
                          mode=D.MODE,
                          status='common_head_withdrawal_verified' if valid else 'no_common_head_direction',
                          common_directions=common, direction_catalogue=self.direction_catalogue,
                          common_representative=D.representative(common,self.direction_catalogue),
                          motion=D.MOTION, normal_depth_m=depth, contacts=records,
                          selected_ids=[p['candidate_id'] for p in actual],
                          all_contacts_have_certified_direction=valid,
                          length_m=max((p['analysis']['length_m'] for p in records), default=0.),
                          actual_contacts_file=str(contacts_path.relative_to(I.ROOT)),
                          size_mismatch_policy='Actual geometry must retain a direction shared with every fixed head and the work/floor policy',
                          provenance=provenance)
            I.save(path, cached)
        assert cached['mode'] == D.MODE
        self.current, self.current_path = cached, path
        print(self.problem.name, 'round', number, 'surviving common 3-D directions',
              len(cached['common_directions']['ids']), flush=True)
        return cached, path

    def finish(self, contacts_path):
        records = self.current['contacts']
        common = D.common(records, self.direction_catalogue['global_allowed_directions'])
        valid = D.nonempty(common)
        result = dict(self.current, object=self.problem.name, complete=True, motion=D.MOTION,
                      mode=D.MODE, contacts=records,
                      selected_ids=[p['candidate_id'] for p in records],
                      all_contacts_have_certified_direction=valid,
                      actual_contacts_file=str(contacts_path.relative_to(I.ROOT)),
                      status='common_head_withdrawal_verified' if valid else 'no_common_head_direction',
                      common_directions=common, direction_catalogue=self.direction_catalogue,
                      common_representative=D.representative(common,self.direction_catalogue),
                      provenance=dict(inputs=I.hashes([self.catalogue_path, contacts_path]
                          + ([self.current_path] if self.current_path else [])), code=code_hashes()))
        path = contacts_path.parent/'insertion_directions.json'
        I.save(path, result)
        return result, path
