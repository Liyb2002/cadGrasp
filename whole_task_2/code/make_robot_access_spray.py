"""Render and encode the silent, text-free eight-second Panda spraying video."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'whole_task_2'
IDEA=HERE/'idea'
DATA=HERE/'assets/robot_access_spray'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    IDEA.mkdir(parents=True,exist_ok=True)
    motion=json.loads((DATA/'motion.json').read_text())
    with tempfile.TemporaryDirectory(prefix='cadgrasp_spray_frames_') as temp:
        frames=Path(temp)
        subprocess.run(['blender','-b','-t','24','--factory-startup','--python-exit-code','1',
            '--python',str(HERE/'code/render_robot_access_spray.py'),'--','--output',str(frames)],check=True)
        images=sorted(frames.glob('*.png'))
        if len(images)!=motion['frames']: raise RuntimeError('Incomplete render')
        temporary=IDEA/'robot_access_spray.tmp.mp4'
        subprocess.run(['ffmpeg','-y','-v','error','-framerate',str(motion['fps']),'-i',str(frames/'%04d.png'),
            '-vf','hqdn3d=1.2:1.2:2.5:2.5','-c:v','libx264','-preset','slow','-crf','18',
            '-pix_fmt','yuv420p','-movflags','+faststart','-an',str(temporary)],check=True)
        video=IDEA/'robot_access_spray.mp4';os.replace(temporary,video)
        # Inspect frames from the ACTUAL encoded video, not the raw render.
        sheet=DATA/'encoded_frames.png'
        subprocess.run(['ffmpeg','-y','-v','error','-i',str(video),'-vf',
            'fps=1,scale=480:-1,tile=4x2','-frames:v','1',str(sheet)],check=True)
        subprocess.run(['ffmpeg','-y','-v','error','-ss','5','-i',str(video),'-frames:v','1',
            str(IDEA/'robot_access_spray.png')],check=True)
        subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],check=True)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams',
            '-show_format','-of','json',str(video)],text=True))
        stream=probe['streams'][0]
        assert len(probe['streams'])==1 and stream['codec_type']=='video'
        assert int(stream['nb_read_frames'])==192 and abs(float(probe['format']['duration'])-8)<.001
        report=dict(video=str(video.relative_to(ROOT)),sha256=sha(video),
            width=stream['width'],height=stream['height'],frames=192,duration_s=8,fps=24,
            no_audio=True,no_text_or_number_overlay=True,manufacturer_lettering_mesh_omitted=True,
            all_frames_decode=True,scene_source=motion['source_scene'],
            config='whole_task_2/assets/robot_access_spray/config.json',
            motion_verification=motion['verification'],
            inputs={str(path.relative_to(ROOT)):sha(path) for path in [
                DATA/'config.json',DATA/'motion.json',DATA/'trajectory.npz',DATA/'robot_meshes.npz',
                HERE/'code/render_robot_access_spray.py',HERE/'code/prepare_robot_access_spray.py',
                HERE/'code/make_robot_access_spray.py',ROOT/'robots_config/arm_visuals.py']},
            actual_encoded_contact_sheet=str(sheet.relative_to(ROOT)),
            scope=motion['scope'])
        (DATA/'video.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:report[k] for k in ['video','width','height','frames','duration_s','all_frames_decode']},indent=2),flush=True)


if __name__=='__main__': main()
