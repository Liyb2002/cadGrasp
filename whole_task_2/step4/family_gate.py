"""Connect contact patches in the final solid rather than deleting them early.

The final full fixture must still be one closed connected material component,
avoid every actual exclusion, enclose ground pressure and pass every load.
"""
from pathlib import Path
import tempfile
import numpy as np

from whole_task_2.step3.legacy import kernel as K
from .export_geometry import exclusions
from .validate_mobile import validate_artifacts
from material_evaluator import MaterialEvaluator
from material_force import certify_material_contacts
from prune_support import material_components
from whole_step4_render import NominalBuilder


class LegalCoreBuilder(NominalBuilder):
    def solid(self, layout):
        record = self.model.family_legal_cores.get(layout.key())
        return K.S.solid(record['mesh']) if record is not None else super().solid(layout)


def condition(model, layout, mesh, path, require_connected, allowed_mesh=None):
    original = K.S.solid(mesh)
    allowed = K.S.solid(allowed_mesh) if allowed_mesh is not None else None
    work, bodies, sweeps = exclusions(model, layout, mesh)
    forbidden = work+bodies+sweeps
    additions = []
    for cycle in range(8):
        K.D.export_exact_obj(mesh, path)
        mesh = K.trimesh.load(path, force='mesh', process=False)
        solid = K.S.solid(mesh)
        overlap = [K.material_volume(solid ^ x) for x in forbidden]
        outside = K.material_volume(solid-allowed) if allowed is not None else 0.
        if max(overlap) <= 1e-10 and outside <= 1e-10:
            count = len(material_components(solid))
            if require_connected and count != 1:
                raise ValueError('Complete fixture has disconnected material')
            if not mesh.is_watertight or not mesh.is_winding_consistent:
                raise ValueError('Invalid exported material boundary')
            added = K.material_volume(solid-original)
            additions.append(added)
            if added > 1e-10:
                # Near-coplanar recuts can move a Boolean boundary. Project
                # it back into the original material, then serialize and
                # replay the same exclusion and load gates again.
                from whole_search.common import unpack_solid
                mesh = unpack_solid(solid ^ original)
                continue
            return mesh, dict(conditioning_cycles=cycle, component_count=count,
                maximum_exclusion_overlap_m3=max(overlap),
                removed_cm3=K.material_volume(original-solid)*1e6,
                added_material_m3=added, intermediate_added_material_m3=additions,
                outside_allowed_component_m3=outside,
                connected_required=require_connected, tolerance_m3=1e-10)
        remaining = solid-K.union([x for x, volume in zip(forbidden, overlap) if volume > 1e-10])
        if outside > 1e-10:
            remaining = remaining ^ allowed
        mesh = K.unpack_solid(remaining) if hasattr(K, 'unpack_solid') else None
        if mesh is None:
            from whole_search.common import unpack_solid
            mesh = unpack_solid(remaining)
    raise ValueError(f'Exported exclusions did not resolve at the original tolerance: {path.name}; added={additions}; overlaps={overlap}; outside={outside}')


class FamilyEvaluator(MaterialEvaluator):
    def space_candidate(self, item):
        return item.result is not None and sum(int((~mask).sum()) for mask in item.result['masks'].values()) <= 4096

    def preview(self, layout, *args, **kwargs):
        self.model.activate(layout)
        return super().preview(layout, *args, **kwargs)

    def classify(self, item):
        self.model.activate(item.layout)
        return super().classify(item)

    def connect(self, item):
        self.model.activate(item.layout)
        item = super().connect(item)
        record = self.model.family_legal_cores.get(item.layout.key())
        if record is not None:
            item.nominal = record['mesh']
            item.result = dict(item.result, **record['proof'], remaining_mesh=item.nominal,
                material_contact_certification_run=True, force_material_geometry_certified=True)
        return item

    def construct(self, item):
        self.model.activate(item.layout)
        if self.builders is None:
            import material_builder_bounded as backend
            self.builders = LegalCoreBuilder(self.model), backend
        item = super().construct(item)
        if not item.completed or item.construction.get('family_finite_access_gate_passed'):
            return item
        try:
            with tempfile.TemporaryDirectory(prefix='cadgrasp-family13-gate-') as directory:
                folder = Path(directory); stage = folder/'step4.2'; stage.mkdir()
                np.savez_compressed(stage/'layout.npz', **item.layout.arrays(self.model.native))
                K.D.export_exact_obj(item.witness, stage/'legal_component.obj')
                core, core_record = condition(self.model, item.layout, item.nominal, stage/'support.obj', False, item.witness)
                full, full_record = condition(self.model, item.layout, item.solid, stage/'complete_support.obj', True, item.witness)
                from whole_task_2.step1.common import write_json
                write_json(folder/'execution.json', dict(poses=[self.model.execution(item.layout, k) for k in item.layout.active]))
                validation = validate_artifacts(self.model, folder, item.layout)
                proof = certify_material_contacts(self.model, item.layout, full)
                if not all(mask.all() for mask in proof['masks'].values()):
                    raise ValueError('Actual complete material lost an original load')
                item.nominal, item.solid = core, full
                item.result = dict(item.result, **proof, remaining_mesh=core,
                    material_contact_certification_run=True, force_material_geometry_certified=True)
                item.construction.update(family_finite_access_gate_passed=True,
                    contact_components_connected_by_complete_fixture=True,
                    export_conditioning=dict(core=core_record, complete=full_record),
                    independent_final_checks=validation)
                item.evaluation['material_full_cm3'] = abs(float(full.volume))*1e6
        except (ValueError, RuntimeError, AssertionError) as error:
            item.construction = dict(item.construction, passed=False,
                family_finite_access_gate_passed=False, physical_certificate_error=str(error))
        return item
