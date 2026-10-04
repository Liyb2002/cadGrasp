"""Repair small original-contact collar islands before strict construction acceptance.

Only sub-millimetre contact transitions may taper. Full rods and soles are
unchanged. Bridges must stay in the original legal construction domain.
"""
import argparse
import shutil
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
        group=E.OUT/name;comparison=group/'comparison.json'
        previous=E.json.loads(comparison.read_text()) if comparison.exists() else dict(passed=False)
        if previous.get('passed'):
            try:E.I.check_report(group/'step4/data/growing_support/report.json')
            except RuntimeError:previous=dict(passed=False)
        archive=group/'history/before_contact_bridge';saved=['step3','step4/data/growing_support','step4/data/source_inputs','step5_evaluate','comparison.json']
        if previous.get('passed'):
            if archive.exists():shutil.rmtree(archive)
            for rel in saved:
                src=group/rel;dst=archive/rel
                if src.is_dir():shutil.copytree(src,dst)
                elif src.exists():dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        try:
            E.run(name,lookup[name]);path=group/'step4/data/growing_support/report.json';report=E.json.loads(path.read_text())
            report['provenance']['code'].update(E.I.hashes([E.Path(__file__),E.Path(R.__file__),E.Path(P.__file__)]));E.save(path,report);evaluate(group)
        except Exception as error:
            import traceback;traceback.print_exc();E.save(comparison,dict(complete=True,passed=False,error=str(error)))
        current=E.json.loads(comparison.read_text())
        if previous.get('passed') and (not current.get('passed') or current['step5']['object_and_support_poses']['box_volume_cm3']>=previous['step5']['object_and_support_poses']['box_volume_cm3']):
            for rel in saved:
                src=archive/rel;dst=group/rel
                if src.is_dir():shutil.copytree(src,dst,dirs_exist_ok=True)
                elif src.exists():shutil.copy2(src,dst)
            print('KEPT BETTER INCUMBENT',name,flush=True)
