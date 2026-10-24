"""Reject incomplete physical certificates before the population can stop."""
from pathlib import Path
import tempfile
import numpy as np

from .export_geometry import condition_core
from .validate_mobile import validate_artifacts
from whole_task.step1.common import write_json
from whole_task.step3.legacy import kernel as K
from material_evaluator import MaterialEvaluator
from material_force import certify_material_contacts


class MobileEvaluator(MaterialEvaluator):
    def construct(self, item):
        item=super().construct(item)
        if not item.completed or item.construction.get('finite_access_gate_passed'):
            return item
        try:
            with tempfile.TemporaryDirectory(prefix='cadgrasp-mobile-complete-') as directory:
                folder=Path(directory);second=folder/'step4.2';second.mkdir()
                np.savez_compressed(second/'layout.npz',**item.layout.arrays(self.model.native))
                K.D.export_exact_obj(item.witness,second/'legal_component.obj')
                core,core_record=condition_core(self.model,item.layout,item.nominal,second/'support.obj')
                full,full_record=condition_core(self.model,item.layout,item.solid,second/'complete_support.obj')
                write_json(folder/'execution.json',dict(poses=[self.model.execution(item.layout,k)for k in item.layout.active]))
                validation=validate_artifacts(self.model,folder,item.layout)
                proof=certify_material_contacts(self.model,item.layout,core)
                if not all(mask.all()for mask in proof['masks'].values()):
                    raise ValueError('Conservative export correction lost mandatory core force support')
                item.nominal=core;item.solid=full
                full_proof=certify_material_contacts(self.model,item.layout,full)
                if not all(mask.all()for mask in full_proof['masks'].values()):
                    raise ValueError('Corrected complete solid lost force support')
                item.full_certificate=dict(full_proof,passed=True)
                item.result=dict(item.result,**proof,remaining_mesh=core,
                    volume_cm3=abs(float(core.volume))*1e6,force_material_geometry_certified=True)
                item.construction.update(finite_access_gate_passed=True,
                    conservative_export_correction=dict(core=core_record,complete=full_record),
                    actual_full_solid_checks=validation['complete_geometry'],
                    complete_path_overlap_m3=validation['complete_path_overlap_m3'],
                    continuous_endpoint_counts=validation['continuous_endpoint_counts'],
                    material_cm3=abs(float(full.volume))*1e6)
                item.evaluation['material_full_cm3']=abs(float(full.volume))*1e6
                self.stats['material_contact_certifications']+=2
        except (ValueError,RuntimeError,AssertionError) as error:
            item.construction=dict(item.construction,passed=False,finite_access_gate_passed=False,
                physical_certificate_error=str(error))
            # PopulationSoftSearch keeps exploring; an original nominal PASS
            # cannot become an incumbent until this stronger gate succeeds.
        return item
