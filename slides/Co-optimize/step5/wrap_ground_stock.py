"""Actual supplemental foot proposals: 8 mm wide, original 5 mm height."""
import numpy as np
from deterministic_base import Builder,unpack,box,moved,SCALE
from prune_support import material_components
from wrap_dimensions import MINIMUM_WIDTH_M,INITIAL_FOOT_HEIGHT_M


def prepare(stock,domain,rows,margin_scale=1.,terminal_width=MINIMUM_WIDTH_M):
    builder=Builder(unpack(stock),unpack(domain),rows,width=MINIMUM_WIDTH_M*margin_scale)
    builder.ground()
    record=builder.history[-1];patches=[]
    if record['compact_feet_used']:
        for location in record['foot_locations']:
            p=np.asarray(location['world_xy_m']);T=np.asarray(location['fixture_transform'])
            foot=moved(box(np.r_[p-terminal_width/2,0.],
                np.r_[p+terminal_width/2,INITIAL_FOOT_HEIGHT_M]),np.linalg.inv(T))^domain
            patches.extend(material_components(foot))
    else:
        margin=record['margin_m']
        limited=domain^box(builder.low-margin,builder.high+margin)
        distinct={np.round(r['fixture_transform'],11).tobytes():r['fixture_transform'] for r in rows}
        for T in distinct.values():
            slab=limited.trim_by_plane(tuple(-T[2,:3]),float((T[2,3]-INITIAL_FOOT_HEIGHT_M)/SCALE))
            required=np.vstack([r['hull'][:,:2] for r in rows if np.allclose(T,r['fixture_transform'],atol=1e-11,rtol=0)])
            low,high=required.min(0)-margin,required.max(0)+margin
            for axis in range(2):
                slab=slab.trim_by_plane(tuple(T[axis,:3]),float((low[axis]-T[axis,3])/SCALE))
                slab=slab.trim_by_plane(tuple(-T[axis,:3]),float((T[axis,3]-high[axis])/SCALE))
            patches.extend(material_components(slab))
    record.update(actual_terminal_width_m=terminal_width,actual_terminal_height_m=INITIAL_FOOT_HEIGHT_M,
                  ground_margin_scale=margin_scale,
                  raw_proposals_recorded=True,original_contact_body_thickened=False)
    return builder,patches
