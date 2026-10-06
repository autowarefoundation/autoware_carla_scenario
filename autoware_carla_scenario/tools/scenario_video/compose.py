"""Lay CARLA and SUMO frames side by side and encode an mp4 at 10 fps.

    python compose.py <carla_dir> <sumo_dir> <out.mp4> <title> <result>

Needs ffmpeg.  A frame either side is missing (a screenshot still being written
when the run ended, say) repeats the previous one.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 960, 540


def _stamp(path: Path) -> float:
    return float(path.stem.split("_")[1])


def _load(path: Path | None) -> Image.Image | None:
    if path is None:
        return None
    try:
        return Image.open(path).convert("RGB")
    except (OSError, ValueError):
        return None


def _fit(image: Image.Image) -> Image.Image:
    image = image.copy()
    image.thumbnail((W, H))
    canvas = Image.new("RGB", (W, H), (20, 24, 22))
    canvas.paste(image, ((W - image.width) // 2, (H - image.height) // 2))
    return canvas


def main() -> None:
    carla_dir, sumo_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    title, result = sys.argv[4], sys.argv[5]
    frames = sorted(carla_dir.glob("carla_*.png"), key=_stamp)
    sumo_frames = {round(_stamp(p), 3): p for p in sumo_dir.glob("sumo_*.png")}
    font = ImageFont.load_default(size=24)
    small = ImageFont.load_default(size=20)
    tmp = Path(tempfile.mkdtemp())
    t0 = _stamp(frames[0])
    last_c = last_s = None
    n = 0
    for path in frames:
        t = _stamp(path)
        c = _load(path) or last_c
        s = _load(sumo_frames.get(round(t, 3))) or last_s
        if c is None:
            continue
        last_c, last_s = c, s
        sheet = Image.new("RGB", (2 * W + 30, H + 70), (14, 17, 16))
        draw = ImageDraw.Draw(sheet)
        sheet.paste(_fit(c), (10, 60))
        if s is not None:
            sheet.paste(_fit(s), (W + 20, 60))
        draw.text((10, 8), title, fill=(240, 244, 240), font=font)
        draw.text((10, 36), "CARLA (chase camera)", fill=(170, 190, 180), font=small)
        draw.text(
            (W + 20, 36),
            "SUMO (sumo-gui, replayed from the run's FCD and signal states)",
            fill=(170, 190, 180),
            font=small,
        )
        draw.text(
            (2 * W - 330, 8),
            f"t = {t - t0:5.1f} s   {result}",
            fill=(255, 210, 90),
            font=font,
        )
        sheet.save(tmp / f"f_{n:05d}.png")
        n += 1
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            "10",
            "-i",
            str(tmp / "f_%05d.png"),
            "-vf",
            "scale=1440:-2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "28",
            str(out),
        ],
        check=True,
    )
    print(out, n, "frames")


if __name__ == "__main__":
    main()
