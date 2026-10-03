"""Choose an above-ground view from the visible support side of each pose."""
import numpy as np

from step4_connect_support import clean_render as V
from step4_connect_support import publish_compact as P


def mask_piece(mesh,color,bias=0):
    part=P.piece(mesh,color,bias);part['unlit']=True
    return P.CPU.placed(part)


def choose(renderer,body,heads,object_mesh=None):
    """Rank opaque diagnostic masks; final object transparency is unchanged.

    Blue is physical support, red the active head collars, grey the object.
    Prefer views where material is in front of the object and head collars
    remain visible, then favor a reasonably large readable silhouette.
    """
    points=body.vertices if object_mesh is None else np.vstack([body.vertices,object_mesh.vertices])
    support=[mask_piece(body,'#2080e0')]
    support.extend(mask_piece(h,'#f02020',.000003) for h in heads if len(h.faces))
    obj=[] if object_mesh is None else [mask_piece(object_mesh,'#606060')]
    directions=[[-.8,-1.,.65]]
    for elevation in (28.,42.):
        for azimuth in np.arange(0,360,30):
            a,e=np.deg2rad([azimuth,elevation])
            directions.append([np.cos(a)*np.cos(e),np.sin(a)*np.cos(e),np.sin(e)])
    trials=[];cameras=[]
    for direction in directions:
        camera=P.CPU.fit(points,direction,1.,padding=1.13)
        alone=np.asarray(renderer.render(support,camera,160),dtype=float)
        together=np.asarray(renderer.render(support+obj,camera,160),dtype=float) if obj else alone
        def count(pixels):
            red=(pixels[:,:,0]>pixels[:,:,1]+65)&(pixels[:,:,0]>pixels[:,:,2]+65)
            blue=(pixels[:,:,2]>pixels[:,:,0]+65)&(pixels[:,:,2]>pixels[:,:,1]+35)
            return int((red|blue).sum()),int(red.sum())
        full,full_heads=count(alone);visible,visible_heads=count(together)
        support_ratio=min(1.,visible/max(full,1));head_ratio=min(1.,visible_heads/max(full_heads,1))
        trials.append(dict(direction_world=camera[1][2].tolist(),
            support_front_fraction=support_ratio,active_head_front_fraction=head_ratio,
            visible_support_pixels=visible,visible_head_pixels=visible_heads))
        cameras.append(camera)
    largest_head_area=max(max(t['visible_head_pixels'] for t in trials),1)
    for trial in trials:
        # A head seen edge-on must not win merely because its tiny projection
        # is unobstructed. Reward readable projected head area across views.
        trial['score']=float(1.5*trial['support_front_fraction']+
            2*np.sqrt(trial['visible_head_pixels']/largest_head_area)+
            .2*np.sqrt(trial['visible_support_pixels']/160**2))
    selected=int(np.argmax([t['score'] for t in trials]))
    camera=cameras[selected]
    return camera,dict(method='support_and_active_head_visibility_from_opaque_masks',
        selected_index=selected,selected=trials[selected],previous_fixed_view=trials[0],
        candidate_count=len(trials),mask_size=[160,160],
        focus_world=camera[0].tolist(),basis_world=camera[1].tolist(),span_m=float(camera[2]),
        candidates=trials)
