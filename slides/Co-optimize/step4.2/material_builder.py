"""Existing base construction followed by bounded exact legality recuts."""
from pathlib import Path
import importlib.util
import hashlib

path=Path(__file__).resolve().parents[1]/'step5.3/deterministic_base.py'
spec=importlib.util.spec_from_file_location('material_base_backend',path)
backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend)
BACKEND_SHA256=hashlib.sha256(path.read_bytes()).hexdigest()


class Builder(backend.Builder):
    def run(self):
        mesh,record=super().run()
        initial=record['outside_saved_legal_component_cm3']
        rounds=0
        # Unions of already clipped parts can emit small coplanar artifacts.
        # Remove their actual illegal volume; keep the original acceptance
        # tolerances and verify all legal nominal material still survives.
        while backend.volume(self.body-self.allowed)>1e-10 and rounds<3:
            self.body=backend.union(backend.components(self.body^self.allowed));rounds+=1
        if rounds:
            mesh=backend.unpack(self.body);cover=backend.enclosure(mesh,self.rows)
            outside=backend.volume(self.body-self.allowed)
            retained=backend.volume(self.legal_nominal-self.body)
            connected=len(backend.components(self.body))==1
            record.update(passed=all(r['passed'] for r in cover) and connected
                and outside<=1e-10 and retained<=1e-10,
                connected=connected,per_pose=cover,watertight=bool(mesh.is_watertight),
                material_cm3=backend.volume(self.body)*1e6,
                **self.added_material_record(outside,retained),
                outside_saved_legal_component_cm3=outside*1e6,
                retained_legal_step4_material_loss_cm3=retained*1e6,
                floor_pressure_enclosure_passed=all(r['passed'] for r in cover))
        record.update(legality_recut_rounds=rounds,
            initial_outside_saved_legal_component_cm3=initial,
            deterministic_backend_sha256=BACKEND_SHA256)
        return mesh,record
