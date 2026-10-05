"""Keep a conservative full-face access cone when GEOS cannot overlay a face.

This opt-in recovery extends visibility_retry without invalidating results made
by that earlier entry point. A failed face is unresolved, never certified clear.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathlib import Path
import json
import runpy
import sys
from shapely.errors import GEOSException

HERE=Path(__file__).resolve().parent.parent


def recover_cells(original,mesh,triangles,normals,e1,e2,scale,offset,slopes,max_depth):
    cells=[];records=[];failures=[]
    # Reuse the original report schema and bounds; no geometry is processed here.
    _,report=original(mesh,triangles[:0],normals[:0],e1[:0],e2[:0],scale,offset,slopes,max_depth)
    for face in range(len(triangles)):
        try:
            local,checked=original(mesh,triangles[face:face+1],normals[face:face+1],
                e1[face:face+1],e2[face:face+1],scale,offset,slopes,max_depth)
            cells.extend((tri,face,directions,clear,depth) for tri,_,directions,clear,depth in local)
            record=dict(checked['work_faces'][0],work_face_index=face)
        except GEOSException as error:
            # The whole original triangle and complete outer cap contain every
            # reachable family from this face, including its boundary sources.
            cells.append((triangles[face].copy(),face,slopes.copy(),False,0))
            record=dict(work_face_index=face,possible_occluder_faces=None,visible_cells=0,
                transition_cells=1,fully_occluded_cells=0,visited_cells=0,
                occluded_parameter_measure=0.,transition_cover_added_parameter_measure=0.)
            failures.append(dict(work_face_index=face,error=str(error),
                recovery='Retain original triangle and full outer cap as unresolved'))
            print('  visibility overlay unresolved: face',face,'retained full access cone',flush=True)
        records.append(record)
        if (face+1)%100==0:
            print('  work visibility',face+1,'/',len(triangles),'families',len(cells),flush=True)
    report.update(work_faces=records,cell_count=len(cells),
        certified_visible_cells=sum(r['visible_cells'] for r in records),
        transition_cells=sum(r['transition_cells'] for r in records),
        fully_occluded_cells_removed=sum(r['fully_occluded_cells'] for r in records),
        occluded_parameter_measure=sum(r['occluded_parameter_measure'] for r in records),
        transition_cover_added_parameter_measure=sum(r['transition_cover_added_parameter_measure'] for r in records),
        overlay_recovery=dict(failed_faces=failures,failed_face_count=len(failures),
            uncertainty='Full-face upper bound; no visibility or collision claim on recovered faces',
            failed_face_extra_cover_measure='Unquantified; failed faces excluded from the incremental cover statistic'))
    return cells,report


def install():
    from step2_local_support import visibility_retry
    from step2_local_support import visibility as V,work_volume as W
    visibility_retry.install()
    original=V.build_cells
    def build_cells(mesh,triangles,normals,e1,e2,scale,offset,slopes,max_depth=V.MAX_DEPTH):
        return recover_cells(original,mesh,triangles,normals,e1,e2,scale,offset,slopes,max_depth)
    V.build_cells=build_cells
    previous_export=W.WorkVolume.export
    def export(self,out,needs_path):
        path=previous_export(self,out,needs_path)
        report=json.loads(path.read_text())
        report['provenance']['code'].update(W.I.hashes([Path(__file__)]))
        W.I.save(path,report)
        return path
    W.WorkVolume.export=export


if __name__=='__main__':
    stage=Path(sys.argv[1]).resolve()
    if HERE not in stage.parents:raise ValueError('Stage must be inside baseline_algo')
    install()
    sys.argv=[str(stage)]+sys.argv[2:]
    runpy.run_path(str(stage),run_name='__main__')
