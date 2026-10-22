"""Publishing keeps the requested output flat and does not erase other files."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import publish as P
from whole_task.step1.common import sha256


class PublishTests(unittest.TestCase):
    def source(self,root):
        work=root/'work';work.mkdir()
        for name in P.DATA_FILES: (work/name).write_bytes(name.encode())
        (work/'construction').mkdir();(work/'construction/stock.obj').write_bytes(b'stock')
        (work/'evaluation.png').write_bytes(b'picture');(work/'movie.mp4').write_bytes(b'movie')
        (work/'debug').mkdir();(work/'debug/unused.txt').write_text('not retained')
        report=dict(object='B',input_sha256={},video=dict(video=str(work/'movie.mp4'),
            sha256=sha256(work/'movie.mp4'),inputs={},motion_validation=dict(status='passed')))
        (work/'report.json').write_text(json.dumps(report))
        return work

    def test_flat_results_and_minimum_computational_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);work=self.source(root);output=root/'output'
            with patch.object(P,'ROOT',root):report,data=P.publish(work,output)
            self.assertEqual({p.name for p in output.iterdir()},set(P.PUBLIC_FILES))
            self.assertFalse(any(p.is_dir() for p in data.iterdir()))
            self.assertFalse((data/'debug').exists())
            self.assertEqual(sha256(output/'video.mp4'),report['video']['sha256'])
            self.assertEqual((data/'support.obj').read_bytes(),(work/'support.obj').read_bytes())
            self.assertFalse(report['video']['intermediates_retained'])

    def test_unrelated_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);work=self.source(root);output=root/'output';output.mkdir()
            (output/'notes.txt').write_text('keep')
            with patch.object(P,'ROOT',root),self.assertRaises(ValueError):P.publish(work,output)
            self.assertEqual((output/'notes.txt').read_text(),'keep')


if __name__=='__main__':unittest.main()
