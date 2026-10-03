"""Check actual encoded files, not just the writer's requested arguments."""
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step4_connect_support.video import mp4_writer


@unittest.skipUnless(shutil.which('ffprobe') and shutil.which('ffmpeg'), 'FFmpeg tools required')
class VideoCompatibilityTests(unittest.TestCase):
    def test_native_profile_preserves_size_timing_and_decodes_every_frame(self):
        # Failure videos are 760 pixels high: preserve them without macroblock resizing.
        for width, height in ((640, 640), (1280, 760)):
            with self.subTest(size=(width, height)), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / 'video.mp4'
                with mp4_writer(path) as writer:
                    for k in range(6):
                        frame = np.full((height, width, 3), 240, dtype=np.uint8)
                        frame[100:200, 40 + k * 20:140 + k * 20] = [30, 100, 170]
                        writer.append_data(frame)
                probe = json.loads(subprocess.check_output([
                    'ffprobe', '-v', 'error', '-count_frames', '-show_streams',
                    '-show_format', '-of', 'json', str(path)], text=True))
                stream = probe['streams'][0]
                self.assertEqual(stream['codec_name'], 'h264')
                self.assertEqual(stream['profile'], 'Main')
                self.assertEqual(stream['pix_fmt'], 'yuv420p')
                self.assertEqual(stream['codec_tag_string'], 'avc1')
                self.assertEqual(stream['color_range'], 'tv')
                self.assertEqual((stream['width'], stream['height']), (width, height))
                self.assertEqual(stream['r_frame_rate'], '12/1')
                self.assertEqual(int(stream['nb_read_frames']), 6)
                self.assertAlmostEqual(float(probe['format']['duration']), .5)
                subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(path),
                                '-f', 'null', '-'], check=True, capture_output=True)
                atoms = []
                with path.open('rb') as file:
                    while header := file.read(8):
                        size, kind = struct.unpack('>I4s', header)
                        header_size = 8
                        if size == 1:
                            size = struct.unpack('>Q', file.read(8))[0]
                            header_size = 16
                        atoms.append(kind)
                        if size == 0:
                            break
                        file.seek(size - header_size, 1)
                self.assertLess(atoms.index(b'moov'), atoms.index(b'mdat'))


if __name__ == '__main__':
    unittest.main()
