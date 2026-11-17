"""Export the plain colour video of a recording (.db3 from record_rgbd.py) as an mp4, for real-vs-sim comparison.

The bag already stores raw colour + depth for every frame; this just writes the colour stream out at the
recorded frame rate, with nothing drawn on it.

usage:
  python scripts/export_rgb.py press_01                      # outputs/data/recordings/press_01.db3 -> press_01_rgb.mp4
  python scripts/export_rgb.py press_01 press_02 press_03    # several clips
  python scripts/export_rgb.py path/to/clip.db3 --out-dir somewhere
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from softfit.capture import BagSource  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("clips", nargs="+", help="clip names (in outputs/data/recordings) or .db3 paths")
ap.add_argument("--out-dir", default=None, help="default: next to the recording")
a = ap.parse_args()

for clip in a.clips:
    rec = Path(clip) if clip.endswith((".db3", ".bag")) else ROOT / "outputs" / "data" / "recordings" / f"{clip}.db3"
    if not rec.exists():
        sys.exit(f"no recording {rec}")
    out_dir = Path(a.out_dir) if a.out_dir else rec.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{rec.stem}_rgb.mp4"
    src = BagSource(rec, align=False)
    vw, n, t_last, nums = None, 0, 0.0, []
    for fr in src:
        if vw is None:
            h, w = fr.color.shape[:2]
            vw = cv2.VideoWriter(str(out) + ".tmp.mp4", cv2.VideoWriter_fourcc(*"mp4v"), float(src.fps), (w, h))
        vw.write(fr.color)
        n, t_last = n + 1, fr.t
        nums.append(fr.number)
    if vw is None:
        sys.exit(f"{rec}: no frames")
    vw.release()
    os.system(f"ffmpeg -y -loglevel error -i '{out}.tmp.mp4' -c:v libx264 -pix_fmt yuv420p -crf 18 '{out}' "
              f"&& rm '{out}.tmp.mp4'")
    gaps = np.diff(np.array(nums))
    dropped = int(np.clip(gaps - 1, 0, None).sum()) if len(gaps) and min(nums) >= 0 else -1
    lost = "" if dropped < 0 else f", {dropped} dropped ({dropped / max(1, n + dropped):.1%}), longest gap {gaps.max()} frames"
    print(f"{rec.name}: {n} frames, {t_last:.1f} s, {src.fps:g} fps (achieved {(n - 1) / max(t_last, 1e-9):.1f}){lost} -> {out}")
