"""Whole-solid legality for new base material; no prior stage is rerun."""
from __future__ import annotations

import json
from itertools import product

import numpy as np
import trimesh

import stage5_common as C
import growth as G
from exit_clearance import ExitClearance
from work_access import WorkAccess, unpack_solid


class AddedMaterialAccess:
    def __init__(self, data, inputs, width=.005):
        # Every foot, endpoint and orthogonal detour lies in this common-frame
        # window. Unbounded exclusions are capped strictly beyond the window,
        # rather than using a physical tool length or a sampled direction set.
        points=[data['fixture'].vertices]
        for row in data['rows']:
            points.append(C.F.transform_points(row['hull'],np.linalg.inv(row['fixture_transform'])))
        joined=np.vstack(points)
        low,high=joined.min(0)-4*width,joined.max(0)+4*width
        window=np.array(list(product(*zip(low,high))))
        previous=json.loads((data['selected']/'data/report.json').read_text())
        radius=previous.get('clearance',{}).get('minimum_radial_clearance_m',None)
        self.obstacles=[]
        self.window=[low.tolist(),high.tolist()]
        for row in data['rows']:
            folder=C.ROOT/'objects'/inputs.name/'poses'/row['pose']
            with np.load(folder/'setup.npz',allow_pickle=False) as setup:
                angle=int(float(setup['cone_half_deg']))
            path=folder/f'angle_{angle}'/'needs.json'
            inputs.protected[C.relative(path)]=C.digest(path)
            needs=json.loads(path.read_text())
            native=np.asarray(needs['frame']['T_world_mesh'])
            translation=row['object_transform'][:3,3]-native[:3,3]
            vertices=C.F.transform_points(np.asarray(needs['geometry']['vertices_m'])+translation,
                                          np.linalg.inv(row['fixture_transform']))
            mesh=trimesh.Trimesh(vertices,needs['geometry']['faces'],process=False)
            direction=np.asarray(row['common_exit_direction'])
            direction=direction/np.linalg.norm(direction)
            distance=max(.001,float((window@direction).max()-(vertices@direction).min())+.01)
            physical_radius=float(radius) if radius is not None else .01*float(data['body'].extents.max())
            clearance=ExitClearance(mesh,fraction=physical_radius/float(mesh.extents.max()))
            swept=clearance.sweep(distance*direction,padded=True)
            # ExitClearance's Manifolds use its own scale. Convert their meshes
            # back to metres before applying this stage's millimetre kernel.
            from co_common import S
            self.obstacles.append(G.solid(S.unpack(swept)))
            work=WorkAccess(mesh,needs['geometry']['work_face_ids'],needs['load']['cone_half_deg'])
            length=work.required_length(window,minimum=.001)
            self.obstacles.append(G.solid(unpack_solid(work.solid(length))))

    def __call__(self, added):
        return all(G.material_volume(added^obstacle)<=G.SOLID_TOL_M3 for obstacle in self.obstacles)
