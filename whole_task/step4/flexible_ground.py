"""Repair all affected task translations when rigid-family floor repair fails.

Old sharing contacts are not frozen. The proposal may reorganize member seats,
but one actual connected thin-layer body must still support the whole union,
save volume and pass the identical full ground/force/work/exit oracle.
"""
from . import membrane_search as search
from . import ground_oracle


def proposal(args):
    original=ground_oracle.project_families;alias=search.project_families
    def individual(model,layout,groups):
        return original(model,layout,[[i] for i in range(len(layout.poses))])
    try:
        ground_oracle.project_families=individual;search.project_families=individual
        result=search.proposal(args)
        result['floor_repair']='individual_task_translations_with_full_union_recertification'
        return result
    finally:
        ground_oracle.project_families=original;search.project_families=alias
