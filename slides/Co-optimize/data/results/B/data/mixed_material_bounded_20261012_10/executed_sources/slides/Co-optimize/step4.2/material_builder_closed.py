"""Close construction after legality recuts without moving any physical pose."""
import time
from material_builder import Builder as ClippedBuilder,backend


class Builder(ClippedBuilder):
    def run(self):
        mesh,record=super().run();cycles=0
        # Cutting an illegal numerical bridge can separate otherwise legal
        # pieces. Reconnect those pieces inside the SAME legal witness, then
        # apply and check legality again. Every iteration is geometrical.
        while not record['passed'] and cycles<4:
            if backend.volume(self.legal_nominal-self.body)>1e-10:break
            if len(backend.components(self.body))>1:self.connect()
            if backend.volume(self.body-self.allowed)>1e-10:
                self.body=backend.union(backend.components(self.body^self.allowed))
            mesh=backend.unpack(self.body);cover=backend.enclosure(mesh,self.rows)
            outside=backend.volume(self.body-self.allowed)
            retained=backend.volume(self.legal_nominal-self.body)
            connected=len(backend.components(self.body))==1;cycles+=1
            record.update(passed=all(r['passed']for r in cover) and connected
                and mesh.is_watertight and outside<=1e-10 and retained<=1e-10,
                connected=connected,per_pose=cover,watertight=bool(mesh.is_watertight),
                material_cm3=backend.volume(self.body)*1e6,
                added_material_cm3=backend.volume(self.body-self.original)*1e6,
                outside_saved_legal_component_cm3=outside*1e6,
                retained_legal_step4_material_loss_cm3=retained*1e6,
                floor_pressure_enclosure_passed=all(r['passed']for r in cover))
        record.update(legality_reconnect_cycles=cycles,seconds=time.monotonic()-self.began)
        return mesh,record
