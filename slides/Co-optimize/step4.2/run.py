"""Whole-gradient search with layout-dependent base feasibility."""
import sys
from run_oracle import main

if __name__ == '__main__':
    # Retain the established positional B spelling while keeping the new
    # explicit --sets safeguard against restarting the stopped full batch.
    if len(sys.argv) > 1 and sys.argv[1] == 'B':
        del sys.argv[1]
    main()
