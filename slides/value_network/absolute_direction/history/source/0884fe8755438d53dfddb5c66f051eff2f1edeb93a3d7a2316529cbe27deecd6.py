"""Repair small original-contact collar islands before strict construction acceptance.

Only sub-millimetre contact transitions may taper. Full rods and soles are
unchanged. Bridges must stay in the original legal construction domain.
"""
import argparse
import numpy as np
import trimesh
import experiment as E
import recover as R
import refined_recovery as P
from step5_evaluate.evaluate import evaluate

BASE_CONNECT=E.EnvelopeGrow.connect

def connect(self):
    full,construction=BASE_CONNECT(self)
    expected=[E.S.solid(E.G.hull_mesh(v@b+o)) for row,b,o in zip(self.case.support_seeds,self.bases,self.offsets) for cells in row for v in cells]
    for retry in range(3):
        bad=[cell for cell in expected if abs(float((cell-full).volume()))*E.S.SCALE**3>8e-14]
        if not bad:break
        for cell in bad:full=full+cell
    else:raise RuntimeError('Exact contact-cell retention unresolved')
    contact_points=np.vstack([c['triangles_m'].reshape(-1,3)@b+o for row,b,o in zip(self.case.groups,self.bases,self.offsets) for c in row])
    contact_mesh=trimesh.Trimesh(contact_points,np.arange(len(contact_points)).reshape(-1,3),process=False)
    transitions=[]
    for iteration in range(12):
        components=full.decompose()
        if len(components)==1:break
        main=max(components,key=lambda p:abs(float(p.volume())))
        island=min(components,key=lambda p:abs(float(p.volume())))
        mi=E.S.unpack(island);mm=E.S.unpack(main)
        _,near,_=trimesh.proximity.closest_point(contact_mesh,mi.vertices)
        if abs(float(island.volume()))*E.S.SCALE**3>1e-10 or near.max()>.0015:raise RuntimeError('Disconnected component exceeds original-contact transition exception')
        center=mi.vertices.mean(0);point,distance,face=trimesh.proximity.closest_point(mm,center[None])
        if distance[0]>.001:raise RuntimeError('Contact collar gap exceeds 1 mm transition bound')
        normal=mm.face_normals[face[0]];joined=None
        for depth in [.00008,.00016,.0003]:
            end=point[0]-depth*normal
            octa=end+np.vstack([np.eye(3),-np.eye(3)])*.00004
            transition=E.S.solid(E.G.hull_mesh(np.vstack([mi.vertices,octa])))^self.mask
            if transition.is_empty() or not self.legal(transition):continue
            candidate=full+transition
            if len(candidate.decompose())>=len(components):continue
            joined=candidate;transitions.append(dict(original_contact_exception=True,maximum_endpoint_gap_mm=float(distance[0]*1000),
                contact_surface_distance_mm=float(near.max()*1000),transition_may_taper=True,full_rod_or_sole_modified=False,
                added_volume_cm3=float((candidate-full).volume())*E.S.SCALE**3*1e6));break
        if joined is None:raise RuntimeError('No legal original-contact collar bridge')
        full=joined
    else:raise RuntimeError('Contact collar bridge budget exhausted')
    self.final_solid=full;self.save_stage('shared_tree',full)
    construction['contact_collar_transition_repairs']=transitions
    construction['mandatory_cells_retained_without_tolerance_relaxation']=True
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
