"""One isolated geometric trial, supervised by the population's time budget."""
import json
import sys
import faulthandler
import resource
from pathlib import Path
from .population_search import worker


def main():
    config=json.load(sys.stdin)
    policy=json.loads(Path(__file__).with_name('policy.json').read_text())
    budget=int(policy.get('worker_address_space_gib',16))*1024**3
    _,hard=resource.getrlimit(resource.RLIMIT_AS)
    if hard!=resource.RLIM_INFINITY:budget=min(budget,hard)
    resource.setrlimit(resource.RLIMIT_AS,(budget,budget))
    faulthandler.enable()
    faulthandler.dump_traceback_later(120,repeat=True)
    try:result=worker(config)
    finally:faulthandler.cancel_dump_traceback_later()
    print('POPULATION_RESULT',json.dumps(result),flush=True)


if __name__=='__main__': main()
