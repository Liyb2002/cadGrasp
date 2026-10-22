"""Isolated Step4.1 Boolean evaluation with the computed-task adapter."""
import argparse
import faulthandler
from pathlib import Path
import numpy as np
from .adapter import TaskModel
from whole_search.model import Layout
from .initialization import initialize_exact
from whole_search.exact_worker import write_result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--object',required=True);p.add_argument('--layout',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    faulthandler.enable();faulthandler.dump_traceback_later(60,repeat=True)
    model=TaskModel(name=a.object,output_root=a.output_root)
    with np.load(a.layout)as arrays:layout=Layout.from_arrays(arrays)
    write_result(initialize_exact(model,layout),a.out)


if __name__=='__main__':main()
