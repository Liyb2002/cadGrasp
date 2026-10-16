"""Opt-in equivalent population search; preserve the original search budget."""
import sys
from run_population_soft import main


if __name__=='__main__':
    for name in ('equivalent-search',):
        if '--'+name not in sys.argv and '--no-'+name not in sys.argv:
            sys.argv.append('--'+name)
    main()
