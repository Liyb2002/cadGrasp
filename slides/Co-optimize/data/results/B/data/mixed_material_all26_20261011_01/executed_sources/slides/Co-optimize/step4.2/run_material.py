"""Run shared-material mixed search; old frontier entry remains available."""
import sys
from run_oracle import main

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='B':
        del sys.argv[1]
    for name,value in [('--search-policy','material'),('--frontier-width','8'),
                       ('--iterations','8'),('--min-poses','5'),('--max-poses','8')]:
        if name not in sys.argv:
            sys.argv.extend([name,value])
    main()
