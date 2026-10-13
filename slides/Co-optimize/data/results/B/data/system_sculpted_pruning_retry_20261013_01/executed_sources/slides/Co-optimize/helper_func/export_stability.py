"""Certify the serialized contact geometry, not a Boolean's input IDs."""
from pathlib import Path
import tempfile
import numpy as np
import trimesh
from deterministic_base import solid,unpack,volume,moved,union,components,enclosure,Builder,SCALE
from material_builder_closed import regularize_external_boundaries


def stabilize_export(mesh,witness,body,placements,rows,certificate,*,final_support=True,max_cycles=4):
    allowed=solid(witness)
    workpiece=body.copy();workpiece.merge_vertices(digits_vertex=15)
    original_object=solid(workpiece)
    objects=[moved(original_object,np.asarray(Q))for Q in placements]
    object_union=union(objects)
    history=[]
    with tempfile.TemporaryDirectory(prefix='cadgrasp-export-stability-')as folder:
        path=Path(folder)/'actual.obj'
        for cycle in range(max_cycles+1):
            mesh.export(path,digits=16)
            actual=trimesh.load(path,force='mesh',process=False)
            value=solid(actual)
            outside=volume(value-allowed)
            intrusion=volume(value^object_union)
            cover=enclosure(actual,rows)if final_support else []
            geometry=bool(actual.is_watertight and actual.is_winding_consistent and
                outside<=1e-10 and intrusion<=1e-10 and
                (not final_support or (len(components(value))==1 and all(r['passed']for r in cover))))
            proof=certificate.evaluate_candidate(actual)if geometry else None
            record=dict(cycle=cycle,geometry_passed=geometry,force_passed=bool(proof and proof['passed']),
                outside_saved_witness_cm3=outside*1e6,actual_object_overlap_cm3=intrusion*1e6,
                material_cm3=volume(value)*1e6,positive_component_count=len(components(value)),
                watertight=bool(actual.is_watertight),winding_consistent=bool(actual.is_winding_consistent),
                maximum_pressure_enclosure_gap_m=max([r['gap_m']for r in cover],default=0.))
            if proof:record['force_counts']=proof['counts']
            history.append(record)
            if geometry and proof['passed']:
                return actual,proof,dict(passed=True,serialized_geometry_checked=True,history=history,
                    pressure_enclosure=cover,cycles=cycle)
            if cycle==max_cycles:
                break
            value=value^allowed
            for obj in objects:value=value-obj
            value,_=regularize_external_boundaries(value)
            if final_support and len(components(value))>1:
                connector=Builder(unpack(value),witness,rows)
                connector.body=value
                connector.allowed=allowed
                for row in rows:
                    T=row['fixture_transform']
                    world=actual.vertices@T[:3,:3].T+T[:3,3]
                    lo,hi=world.min(0)-1e-8,world.max(0)+1e-8
                    for axis in range(3):
                        connector.allowed=connector.allowed.trim_by_plane(tuple(T[axis,:3]),float((lo[axis]-T[axis,3])/SCALE))
                        connector.allowed=connector.allowed.trim_by_plane(tuple(-T[axis,:3]),float((T[axis,3]-hi[axis])/SCALE))
                def clipped(proposal):
                    result=proposal^connector.allowed
                    for obj in objects:result=result-obj
                    return result
                connector.clip_proposal=clipped
                try:
                    connector.connect()
                    value,_=regularize_external_boundaries(connector.body)
                except ValueError:
                    pass
            mesh=unpack(value)
    raise ValueError('Serialized boundary did not stabilize under original checks: '+str(history))
