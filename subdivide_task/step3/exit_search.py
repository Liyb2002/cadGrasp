"""Independent task candidates, with a mandatory final-assembly acceptance gate."""


class AssemblyRejected(RuntimeError):
    def __init__(self,task_ids,reason):
        super().__init__(reason)
        self.task_ids=list(task_ids)


def independent_search(task_ids,catalogs,try_local,assemble_and_audit,event):
    """Keep accepted task directions; advance a failing task and rebuild everything.

    Local acceptance is only a proposal. Only assemble_and_audit can accept the
    initialization. Every changed path triggers a fresh audit of every task.
    Catalogues are finite; unresolved/exhausted searches never publish a result.
    """
    selected={};indices={id:-1 for id in task_ids}
    def advance(id):
        for index in range(indices[id]+1,len(catalogs[id])):
            indices[id]=index
            try:
                proposal=try_local(id,index,catalogs[id][index])
            except (RuntimeError,ValueError,AssertionError)as ex:
                event(dict(stage='local',id=id,candidate_index=index,status='rejected_or_unresolved',reason=str(ex)))
                continue
            selected[id]=proposal
            event(dict(stage='local',id=id,candidate_index=index,status='passed'))
            return
        raise RuntimeError(id+' exhausted its independent exit catalogue; no feasible initialization certified')

    for id in task_ids:advance(id)
    while True:
        try:
            result=assemble_and_audit(selected)
        except AssemblyRejected as ex:
            failed=[id for id in task_ids if id in ex.task_ids]
            if not failed:raise
            # Change one failing task at a time. Its neighbours keep their
            # directions, but all neighbours must pass the new assembly audit.
            repair=next((id for id in failed if indices[id]+1<len(catalogs[id])),None)
            event(dict(stage='assembly',status='rejected_or_unresolved',failed_task_ids=failed,
                repair_task_id=repair,reason=str(ex)))
            if repair is None:raise RuntimeError('No remaining exit candidate for failing tasks: '+str(failed))from ex
            advance(repair)
            continue
        event(dict(stage='assembly',status='accepted',selected_candidate_indices=dict(indices)))
        return selected,result,indices
