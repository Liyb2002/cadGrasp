"""Encode, inspect and publish the two idea videos and their seated frames."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

from PIL import Image, ImageDraw, ImageFont

IDEA = Path(__file__).resolve().parents[1]
PUBLIC = {
    "pose_following_wrap": ("reoriented_reuse", "变姿复用", "Reoriented Reuse"),
    "fixed_seat_reuse": ("fixed_pose_reuse", "定姿复用", "Fixed-pose Reuse"),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    vis = IDEA / "vis"
    vis.mkdir(exist_ok=True)
    report = {}
    gallery = Image.new("RGB", (1600, 984), "white")
    draw = ImageDraw.Draw(gallery)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    for row, (mode, (public, chinese, english)) in enumerate(PUBLIC.items()):
        motion = json.loads((IDEA / f"data/{mode}_motion.json").read_text())
        frames = IDEA / "data/frames" / mode
        actual = sorted(frames.glob("*.png"))
        expected = [frames / f"{n:05d}.png" for n in range(motion["frame_count"])]
        if actual != expected:
            raise RuntimeError(f"Incomplete rendered frames for {mode}: {len(actual)}")
        video = vis / f"{public}.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(motion["fps"]),
            "-i", str(frames / "%05d.png"), "-an", "-c:v", "libx264",
            "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(video),
        ], check=True)
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration,pix_fmt",
            "-of", "json", str(video),
        ]))["streams"][0]
        if int(probe["nb_read_frames"]) != motion["frame_count"]:
            raise RuntimeError("Encoded frame count differs from saved motion")
        if abs(float(probe["duration"]) - motion["duration_seconds"]) > 1e-6:
            raise RuntimeError("Encoded duration differs from saved motion")
        # Decode the finished file, rather than checking only the PNG inputs.
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"], check=True)
        for pose, n in [(1, 84), (2, 294)]:
            destination = vis / f"{public}_pose{pose}.png"
            shutil.copy2(frames / f"{n:05d}.png", destination)
            im = Image.open(destination).resize((800, 450), Image.Resampling.LANCZOS)
            x, y = (pose - 1) * 800, row * 492
            draw.text((x + 24, y + 12), f"{english}  |  Pose {pose}", fill="#28394B", font=font)
            gallery.paste(im, (x, y + 42))
        report[public] = dict(
            name_chinese=chinese, name_english=english, video=str(video.relative_to(IDEA)),
            video_sha256=sha(video), size_bytes=video.stat().st_size, media=probe,
            geometry_sha256=sha(IDEA / f"data/{mode}.obj"),
            motion_sha256=sha(IDEA / f"data/{mode}_motion.json"),
        )
    gallery.save(vis / "comparison.png")
    report["source_sha256"] = {p.name: sha(p) for p in (IDEA / "code").glob("*.py")}
    (IDEA / "data/video_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("VIDEOS COMPLETE", {k: v["media"] for k, v in report.items() if k != "source_sha256"}, flush=True)


if __name__ == "__main__":
    main()
