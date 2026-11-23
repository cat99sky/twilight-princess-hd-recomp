"""Calibrate the Vulkan frame dumps and TV swap image without a game dump."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

from PIL import Image


def check(path, expected, size=None, solid=False):
    with Image.open(path) as image:
        if size and image.size != size:
            raise ValueError(f"{path.name}: expected {size}, got {image.size}")
        rgb = image.convert("RGB")
        if solid:
            if rgb.getextrema() != tuple((channel, channel) for channel in expected):
                raise ValueError(f"{path.name}: expected solid {expected}, got {rgb.getextrema()}")
        else:
            for x in (0.35, 0.5, 0.65):
                pixel = rgb.getpixel((int(x * rgb.width), rgb.height // 2))
                if pixel != expected:
                    raise ValueError(f"{path.name}: expected {expected}, got {pixel}")
        print(f"{path.name}: {image.width}x{image.height} RGB {expected} OK")


def run(exe, root, trace):
    env = dict(os.environ, WWHD_SMOKE_SCAN_DUMPS="1", WWHD_NO_RENDER_THREAD="1",
               WWHD_DUMP_FRAMES="1,2", WWHD_DUMP_PRESENT="1", WWHD_DRC_MODE="off",
               WWHD_TRACE_SCAN_COPY=str(int(trace)), WWHD_SIM_SCREEN="128x72", WWHD_FULLSCREEN="0", WWHD_HIDDEN_WINDOWS="0",
               WWHD_PRESENT_CAPTURE_PATH=str(root / "swap.png"), WWHD_PRESENT_CAPTURE_FRAME="1")
    subprocess.run([str(exe), "--renderer-smoke"], cwd=root, env=env, check=True, timeout=120)
    blue, green = (0, 0, 255), (0, 255, 0)
    for frame, tv, drc in ((1, blue, green), (2, green, blue)):
        check(root / f"frame_{frame}.png", tv, (64, 36), solid=True)
        check(root / f"frame_{frame}_drc.png", drc, (32, 24), solid=True)
        check(root / f"frame_{frame}_present.png", tv, (128, 72))
        if trace:
            for stage in ("source", "scan"):
                check(root / f"frame_{frame}_copy_1_tv_{stage}.png", tv, (64, 36), solid=True)
                check(root / f"frame_{frame}_copy_2_drc_{stage}.png", drc, (32, 24), solid=True)
    # Actual swap-image capture uses the zero-based frame, numbered dumps use one-based.
    check(root / "swap.png", green)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    args = parser.parse_args()
    exe = args.executable.resolve(strict=True)
    for trace in (False, True):
        with tempfile.TemporaryDirectory(prefix="gx2-scan-") as temp:
            run(exe, Path(temp), trace)


if __name__ == "__main__":
    main()
