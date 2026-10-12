"""Paired working-exclusion volumes on unchanged registered geometry."""
from pathlib import Path
import sys, json
import numpy as np
ROOT=Path.cwd();COOPT=ROOT/'slides/Co-optimize'
sys.path[:0]=[str(COOPT/'helper_func'),str(COOPT/'step4.2')]
import _bootstrap
from oracle_search import OracleModel
from base_oracle import BaseOracle
from whole_search.common import C,unpack_solid

base=COOPT/'data/results/B'
records=[]
for label in ['pose1+2+4+7','pose1+2+3+4+5+6','pose1+2+3+4+5+6+7+8']:
    report=json.loads((base/label/'step4/step4.1/data/report.json').read_text())
    model=OracleModel(report['poses'],'B',demand_nodes=32)
    normal=model.base_oracle;angular=BaseOracle(model,work_mode='angular')
    center=model.mesh.bounds.mean(0);radius=.9*model.extent
    ball=C.trimesh.creation.icosphere(subdivisions=3,radius=radius);ball.apply_translation(center)
    stock=C.S.solid(ball);volumes={}
    outside=model.points+model.thickness*model.point_normals
    masks={}
    for mode,oracle in [('normal',normal),('angular',angular)]:
        parts=[];blocked=np.zeros(len(outside),bool)
        for work in oracle.work_access:
            length=work.required_length(ball.vertices,minimum=.001)
            parts.append(work.solid(length))
            blocked |= work.contains_points(outside)
        region=C.union(parts)^stock
        volumes[mode]=float(C.material_volume(region))*1e6;masks[mode]=blocked
    records.append(dict(id=label,pose_count=len(report['poses']),registered_Q_identity=True,
        spherical_crop_center_object_m=center.tolist(),spherical_crop_radius_m=radius,
        angular_work_exclusion_cm3=volumes['angular'],normal_work_exclusion_cm3=volumes['normal'],
        exclusion_volume_reduction_percent=100*(1-volumes['normal']/volumes['angular']),
        oracle_freed_outer_wrap_area_cm2=float(model.point_areas[masks['angular'] & ~masks['normal']].sum()*1e4),
        area_measure='4 fixed barycentric probes per source triangle, on a5mm outward shell; geometric diagnostic only',
        force_or_contact_model_changed=False,force_solver_run=False,
        crop_applied_to_both_modes=True,crop_is_not_a_search_limit=True))
    print(json.dumps(records[-1],ensure_ascii=False),flush=True)
out=base/'data/normal_ray_oracle_4to8_v1/working_volume_comparison.json'
out.write_text(json.dumps(dict(mode_scope='Oracle only',cases=records),ensure_ascii=False,indent=2)+'\n')
