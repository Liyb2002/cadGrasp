"""Current cached whole-gradient entry; select B sets explicitly."""
import sys
from stable_pipeline import main

if __name__ == '__main__':
    # Retain the established positional B spelling while keeping the new
    # explicit --sets safeguard against restarting the stopped full batch.
    if len(sys.argv) > 1 and sys.argv[1] == 'B':
        del sys.argv[1]
    main()
