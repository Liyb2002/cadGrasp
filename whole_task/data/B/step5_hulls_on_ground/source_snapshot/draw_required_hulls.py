"""Render the reference-style support and surrounding hulls without axes.

This keeps the former command name as an entry point. Its output is now
hulls_and_support.png; the obsolete unfolded required_hulls.png is removed.
"""
import argparse
import json
from pathlib import Path
import tempfile

from whole_task.step1.common import ROOT,DEFAULT_OUTPUT,write_json,sha256
from whole_task.step3.run_membranes import load_model
from .height_reference import model_for_report
from .hulls_presentation import render_movie


def render(folder,output=None):
    folder=Path(folder)
    report=json.loads((folder/'report.json').read_text())
    model,_,_=load_model(ROOT/'whole_task/data'/report['object']/'step3_membranes')
    model=model_for_report(model,report)
    output=Path(output or DEFAULT_OUTPUT/report['object']/'step5/hulls_and_support.png')
    # Two orbit samples suffice for a still; both are the same 0/360-degree view.
    with tempfile.TemporaryDirectory(prefix='hull_still_') as temporary:
        temporary=Path(temporary)
        info=render_movie(model,report['groups'],temporary/'still.mp4',temporary/'motion.npz',
                          seconds_per_set=2/24,poster_path=output)
    metadata=dict(status='passed',figure=str(output.resolve()),figure_sha256=sha256(output),
                  source_sha256=sha256(__file__),renderer_sha256=info['source_sha256'],
                  input_report_sha256=sha256(folder/'report.json'),groups=info['groups'],
                  all_original_pressure_samples_drawn=True,no_text=True,no_axes=True,
                  no_unfolded_panels=True,geometry_and_working_poses_modified=False)
    write_json(folder/'hulls_figure.json',metadata)
    (output.parent/'required_hulls.png').unlink(missing_ok=True)
    (folder/'required_hulls.png').unlink(missing_ok=True)
    return metadata


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    folder=args.folder or Path(json.loads((ROOT/'whole_task/data/B/step5_active.json').read_text())['workdir'])
    render(folder,args.output)
