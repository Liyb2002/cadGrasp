"""Step3: one independent membrane per Step1 task, cut by the tool Step1 certified."""
import argparse
from pathlib import Path
from whole_task_2.step1.common import DEFAULT_OUTPUT
from .tool_membranes import run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--object',default='B')
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    run(args.output_root,args.object,args.workers)


if __name__=='__main__':main()
