"""Run 32 registered slots; choose Juxtapose identities during optimization."""
import sys
from run_oracle import main as run_oracle_main


def main():
    if len(sys.argv)>1 and sys.argv[1]=='B':
        del sys.argv[1]
    for name,value in [('--search-policy','population_soft'),('--frontier-width','32'),
                       ('--iterations','8'),('--min-poses','2'),('--max-poses','10')]:
        if name not in sys.argv:
            sys.argv.extend([name,value])
    for name in ['--allow-oracle-fallback','--allow-coupled-adjustment','--allow-group-seating',
                 '--allow-joint-reseating','--allow-placement-guidance','--allow-height-repair','--stop-when-feasible']:
        if name not in sys.argv:sys.argv.append(name)
    run_oracle_main()


if __name__=='__main__':
    main()
