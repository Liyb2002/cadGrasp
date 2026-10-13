"""Regenerate the independent single-contact insertion experiment and all figures."""
import argparse
import study
import audit
import draw

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('objects',nargs='*')
    for name in parser.parse_args().objects or study.P.OBJECTS:
        study.run(name)
        audit.run(name)
        draw.run(name)
