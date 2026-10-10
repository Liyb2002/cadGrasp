"""Whole-coordinate repair when an active-coordinate search remains stuck.

The successful cached all-pose repair supplies every independent direction
and every explicit seat translation. Lazy line search still avoids scoring
the later sample pools after a successful step. This is an extra repair budget.
"""
from .active_fast_gradient import ActiveGradientModel
from .strict_nominal_cache import StrictNominalCache
from .geometry_cache import GeometryCache
from .cached_fast_gradient import CachedGradientSearch
from .progressive_gradient import ProgressiveGradientSearch
from whole_search.reuse_first import registered


class LargeSetGeometryCache(StrictNominalCache):
    def bound(self,cache,size):
        # Nine distinct seats already have 73 body-relative transforms. A
        # 64-entry LRU can cycle through them without keeping the static rows.
        # Hold a few full layouts, with finite memory bounds on every table.
        GeometryCache.bound(cache,256 if size==64 else max(size,2048))


class ExpandedGradientModel(ActiveGradientModel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.geometry_cache=LargeSetGeometryCache(self)


class ExpandedGradientSearch(ProgressiveGradientSearch):
    def local_proposals(self,current,targets,worst,translation_guests=()):
        rows,base=CachedGradientSearch.local_proposals(self,current,targets,worst,translation_guests)
        self.last_coordinate_selection=dict(
            direction_poses=[self.model.poses[k] for k in current['layout'].active],
            translation_poses=[self.model.poses[k] for k in translation_guests
                               if not registered(current['layout'],k)],
            all_owner_loss=True,expanded_after_active_coordinate_stall=True)
        return rows,base
