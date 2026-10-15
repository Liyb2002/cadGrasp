"""Close construction after legality recuts without moving any physical pose."""
import time
from material_builder import Builder as ClippedBuilder,backend


def regularize_external_boundaries(value):
    """Remove numerical shells and certify external inverse shells as empty.

    Negative inner boundaries can represent real cavities. Preserve those;
    discard a larger inverse shell only when its positively oriented region
    has no above-tolerance intersection with the positive material components.
    """
    parts=list(value.decompose())
    signed=[float(p.volume())*backend.SCALE**3 for p in parts]
    if all(v>backend.VOLUME_TOL for v in signed):return value,[]
    positive=backend.union([p for p,v in zip(parts,signed)if v>backend.VOLUME_TOL])
    kept=[];removed=[]
    for part,volume in zip(parts,signed):
        overlap=None
        if volume < -backend.VOLUME_TOL:
            filled=backend.unpack(part);filled.invert()
            overlap=backend.volume(backend.solid(filled)^positive)
        if abs(volume)<=backend.VOLUME_TOL or (overlap is not None and overlap<=backend.VOLUME_TOL):
            removed.append(dict(signed_boundary_volume_m3=volume,
                positive_material_overlap_m3=overlap))
        else:kept.append(part)
    return (backend.md.Manifold.compose(kept),removed) if removed else (value,[])


class Builder(ClippedBuilder):
    def final_boundary_check(self,record,regularizations):
        # Reconnection may emit an external inverse shell after the original
        # regularization. Genuine inner cavities must remain unchanged.
        self.body,removed=regularize_external_boundaries(self.body)
        regularizations.extend(removed)
        mesh=backend.unpack(self.body);cover=backend.enclosure(mesh,self.rows)
        outside=backend.volume(self.body-self.allowed)
        retained=backend.volume(self.legal_nominal-self.body)
        connected=len(backend.components(self.body))==1
        record.update(passed=all(r['passed']for r in cover) and connected
            and mesh.is_watertight and outside<=1e-10 and retained<=1e-10,
            connected=connected,per_pose=cover,watertight=bool(mesh.is_watertight),
            material_cm3=backend.volume(self.body)*1e6,
            **self.added_material_record(outside,retained),
            outside_saved_legal_component_cm3=outside*1e6,
            retained_legal_step4_material_loss_cm3=retained*1e6,
            floor_pressure_enclosure_passed=all(r['passed']for r in cover))
        return mesh

    def run(self):
        mesh,record=super().run();cycles=0
        regularizations=[]
        mesh=self.final_boundary_check(record,regularizations)
        # Cutting an illegal numerical bridge can separate otherwise legal
        # pieces. Reconnect those pieces inside the SAME legal witness, then
        # apply and check legality again. Every iteration is geometrical.
        while not record['passed'] and cycles<4:
            if backend.volume(self.legal_nominal-self.body)>1e-10:break
            if len(backend.components(self.body))>1:self.connect()
            if backend.volume(self.body-self.allowed)>1e-10:
                self.body=backend.union(backend.components(self.body^self.allowed))
            mesh=self.final_boundary_check(record,regularizations);cycles+=1
        record.update(legality_reconnect_cycles=cycles,final_boundary_regularizations=regularizations,
                      seconds=time.monotonic()-self.began)
        return mesh,record
