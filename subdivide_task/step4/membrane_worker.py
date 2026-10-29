"""Bound native CSG/LP work without blocking all other population branches."""
import multiprocessing
import time
import traceback


def child(send,args):
    try:
        from .membrane_search import proposal
        if len(args)>7 and args[7].get('world_axis_spin',False):
            from .world_spin import proposal
        if len(args)>7 and args[7].get('flexible_ground_repair',False):
            from .flexible_ground import proposal
        send.send(proposal(args[:7]))
    except BaseException as error:
        send.send(dict(status='worker_failed',reason=str(error),traceback=traceback.format_exc()))
    finally:send.close()


def isolated_proposal(args,timeout=None):
    """A timed-out proposal is rejected; committed parent files stay intact."""
    if timeout is None:
        from .membrane_search import TRIAL_TIMEOUT_S as timeout
    ctx=multiprocessing.get_context('spawn');receive,send=ctx.Pipe(duplex=False)
    worker=ctx.Process(target=child,args=(send,args));began=time.monotonic()
    worker.start();send.close()
    try:
        if receive.poll(timeout):
            try:result=receive.recv()
            except EOFError:result=dict(status='worker_failed',reason='Candidate process exited without a complete certificate')
        else:result=dict(status='timed_out',reason='Native candidate evaluation exceeded its bounded wall time')
    finally:
        if worker.is_alive():worker.terminate()
        worker.join(3.)
        if worker.is_alive():worker.kill();worker.join()
        receive.close()
    result.setdefault('operation',args[3]);result.setdefault('seconds',time.monotonic()-began)
    result['isolated_wall_time_s']=time.monotonic()-began
    return result
