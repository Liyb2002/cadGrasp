"""Add uniformly spread surface-centred candidates to the directional pool."""
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .step1_3 import _seed_faces
from .needle_geometry import _init_worker,_face_candidates


def augment(mesh,config,workspace,rows,count=512,workers=8):
    seeds=_seed_faces(mesh,np.ones(len(mesh.faces),bool),limit=count)
    seen=set()
    def key(row):
        mask=np.zeros(len(mesh.faces),bool);mask[row['face_ids']]=True
        return np.packbits(mask).tobytes()
    for row in rows:seen.add(key(row))
    with ProcessPoolExecutor(max_workers=workers,initializer=_init_worker,
            initargs=(mesh.vertices,mesh.faces,config,workspace)) as pool:
        for index,items in enumerate(pool.map(_face_candidates,seeds,chunksize=1)):
            for row in items:
                signature=key(row)
                if signature not in seen:seen.add(signature);rows.append(row)
            if index%80==0:print('SURFACE ANCHORS',index,'/',len(seeds),'regions',len(rows),flush=True)
    return sorted(rows,key=lambda row:-row['selection_score_m2'])
