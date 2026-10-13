"""Run 32 registered slots, each taking one random move and one adjustment."""
import sys
from run_oracle import main as run_oracle_main


def main():
    if len(sys.argv)>1 and sys.argv[1]=='B':
        del sys.argv[1]
    for name,value in [('--search-policy','population32'),('--frontier-width','32'),
                       ('--iterations','8'),('--min-poses','2'),('--max-poses','10')]:
        if name not in sys.argv:
            sys.argv.extend([name,value])
    run_oracle_main()


if __name__=='__main__':
    main()
