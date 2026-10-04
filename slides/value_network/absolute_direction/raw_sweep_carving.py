"""Carve original continuous sweeps before acceptance; retain every full core."""
import argparse
import time
import numpy as np
import trimesh
import experiment as E
import recover as R
import refined_recovery as P
from step5_evaluate.evaluate import evaluate

NORMAL_CONNECT=P.NORMAL_CONNECT

def connect(self):
    started=time.monotonic();full,construction=NORMAL_CONNECT(self)
    sweeps=[E.S.solid(trimesh.Trimesh(m.vertices@b+o,m.faces,process=False)) for m,b,o in zip(self.search.sweep_meshes,self.bases,self.offsets)]
    overlap=[abs(float((full^s).volume()))*E.S.SCALE**3 for s in sweeps]
    if max(overlap)>8e-14:
        # Existing constructor-owned contact transitions may be carved. Full
        # rods, start balls and foot cores must remain whole under strict tests.
        full=full-E.F.union(sweeps)
    cores=self.core_solids+[(f'beam_{i}',b) for i,b in enumerate(self.beams)]+[(f'sole_{i}',s) for i,s in enumerate(self.original_soles)]
    missing=[dict(name=name,missing_volume_m3=abs(float((solid-full).volume()))*E.S.SCALE**3) for name,solid in cores]
    if any(row['missing_volume_m3']>8e-14 for row in missing):raise RuntimeError('Original sweep carving would clip a complete 5 mm core')
    if len(E.material_components(full))!=1:raise RuntimeError('Original sweep carving disconnected material')
    self.final_solid=full;self.save_stage('shared_tree',full)
    construction['raw_sweep_carving']=dict(overlaps_before_m3=overlap,performed=max(overlap)>8e-14,
        original_continuous_sweeps=True,hard_acceptance_tolerances_unchanged=True,complete_core_checks=missing,
        guaranteed_inscribed_rod_diameter_mm=self.guaranteed_radius*2000,scope='constructor contact transitions; no full rod or sole clipped',seconds=time.monotonic()-started)
    return full,construction
E.AbsoluteGrow.connect=connect

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--groups',nargs='+',required=True);args=p.parse_args();E.install_recorded_recovery(E.OUT);lookup=dict(E.specs())
    for name in args.groups:
        try:
            E.run(name,lookup[name]);group=E.OUT/name;path=group/'step4/data/growing_support/report.json';report=E.json.loads(path.read_text())
            report['provenance']['code'].update(E.I.hashes([E.Path(__file__),E.Path(R.__file__),E.Path(P.__file__)]));E.save(path,report);evaluate(group)
        except Exception as error:
            import traceback;traceback.print_exc();E.save(E.OUT/name/'comparison.json',dict(complete=True,passed=False,error=str(error)))
