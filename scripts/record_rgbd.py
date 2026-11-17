"""Record a RealSense (D405) RGB-D clip to a ROS2 bag (.db3) for scripts/rgbd_to_pcd.py.

Two phases:
  1. AIM: live view (colour | depth). Auto-exposure settles. Shows the distance to the centre and how
     much of the centre box has depth. Press SPACE to lock exposure + white balance and start recording.
  2. RECORD: raw depth + colour go straight to the bag (no filtering, no frames dropped by Python).
     Press SPACE / q / Esc to stop (or use --seconds).

Protocol: camera level, facing the sponge's front face. Hands off for the first ~2 s. Then press the sponge
from the top (at most ~25-30 % of its height), hold, let go fast, and keep recording while it springs back.

usage:
  python scripts/record_rgbd.py press_07 --fps 90 --seconds 15   # -> outputs/data/recordings/press_07.db3
"""
import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pyrealsense2 as rs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def pick_mode(dev, want_w, want_h, want_fps):
    """Highest fps <= want_fps where depth z16 AND colour bgr8/rgb8 both exist at want_w x want_h."""
    modes = {}
    for s in dev.query_sensors():
        for p in s.get_stream_profiles():
            v = p.as_video_stream_profile()
            if (v.width(), v.height()) != (want_w, want_h):
                continue
            key = (p.stream_type(), p.format())
            modes.setdefault(key, set()).add(p.fps())
    dep = modes.get((rs.stream.depth, rs.format.z16), set())
    cfmt = rs.format.bgr8 if (rs.stream.color, rs.format.bgr8) in modes else rs.format.rgb8
    col = modes.get((rs.stream.color, cfmt), set())
    both = sorted(f for f in dep & col if f <= want_fps)
    if not both:
        raise SystemExit(f"no common depth+colour mode at {want_w}x{want_h}; depth fps {sorted(dep)}, colour {sorted(col)}")
    return both[-1], cfmt


def config(w, h, fps, cfmt, bag=None):
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
    cfg.enable_stream(rs.stream.color, w, h, cfmt, fps)
    if bag:
        cfg.enable_record_to_file(str(bag))
    return cfg


def view(fs, cfmt, scale, box=0.2):
    d = np.asanyarray(fs.get_depth_frame().get_data())
    c = np.asanyarray(fs.get_color_frame().get_data())
    if cfmt == rs.format.rgb8:
        c = cv2.cvtColor(c, cv2.COLOR_RGB2BGR)
    h, w = d.shape
    y0, y1, x0, x1 = int(h * (0.5 - box)), int(h * (0.5 + box)), int(w * (0.5 - box)), int(w * (0.5 + box))
    cen = d[y0:y1, x0:x1]
    fill = (cen > 0).mean() * 100
    dist = np.median(cen[cen > 0]) * scale if (cen > 0).any() else 0.0   # metres
    v = d[d > 0]
    lo, hi = (np.percentile(v, 2), np.percentile(v, 98)) if len(v) else (0, 1)
    dimg = cv2.applyColorMap(255 - np.clip((d.astype(np.float32) - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    dimg[d == 0] = 0
    for im in (c, dimg):
        cv2.rectangle(im, (x0, y0), (x1, y1), (255, 255, 255), 1)
    return np.hstack([c, dimg]), fill, dist


def lock_exposure(sensor, fs, fps):
    """Freeze auto exposure / gain / white balance at their settled values (keeps colours + blur constant)."""
    md = rs.frame_metadata_value
    out = {}
    for frame, tag in ((fs.get_depth_frame(), "depth"), (fs.get_color_frame(), "color")):
        for key, opt in ((md.actual_exposure, rs.option.exposure), (md.gain_level, rs.option.gain)):
            if frame.supports_frame_metadata(key):
                out[opt] = frame.get_frame_metadata(key)
    try:
        if sensor.supports(rs.option.enable_auto_exposure):
            sensor.set_option(rs.option.enable_auto_exposure, 0)
        cap = 0.9e6 / fps                                   # exposure must fit in one frame [us]
        for opt, val in out.items():
            if sensor.supports(opt):
                r = sensor.get_option_range(opt)
                val = min(val, cap) if opt == rs.option.exposure else val
                sensor.set_option(opt, float(np.clip(val, r.min, r.max)))
        if sensor.supports(rs.option.enable_auto_white_balance):
            wb = sensor.get_option(rs.option.white_balance)
            sensor.set_option(rs.option.enable_auto_white_balance, 0)
            sensor.set_option(rs.option.white_balance, wb)
    except RuntimeError as e:
        print(f"  (could not lock exposure: {e})")
    return {str(k).split(".")[-1]: v for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="clip name, e.g. press_07")
    ap.add_argument("--width", type=int, default=848)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=60, help="max fps (picks the highest the camera offers <= this)")
    ap.add_argument("--seconds", type=float, default=None, help="auto-stop after this long")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "data" / "recordings"))
    ap.add_argument("--headless", type=float, default=None, metavar="AIM_S",
                    help="no windows: let exposure settle AIM_S seconds, then record --seconds (for tests)")
    a = ap.parse_args()

    ctx = rs.context()
    if not len(ctx.query_devices()):
        raise SystemExit("no RealSense camera found")
    dev = ctx.query_devices()[0]
    fps, cfmt = pick_mode(dev, a.width, a.height, a.fps)
    bag = Path(a.out) / f"{a.name}.db3"   # librealsense >= 2.57 records ROS2 bags (.db3)
    bag.parent.mkdir(parents=True, exist_ok=True)
    if bag.exists():
        raise SystemExit(f"{bag} exists - pick another name or delete it")
    print(f"{dev.get_info(rs.camera_info.name)}: {a.width}x{a.height} @ {fps} fps")

    # ---- 1. aim ---------------------------------------------------------------------------
    pipe = rs.pipeline()
    prof = pipe.start(config(a.width, a.height, fps, cfmt))
    sensor = prof.get_device().first_depth_sensor()
    scale = sensor.get_depth_scale()
    if a.headless is not None and not a.seconds:
        raise SystemExit("--headless needs --seconds")
    for opt, val in ((rs.option.enable_auto_exposure, 1), (rs.option.enable_auto_white_balance, 1)):
        if sensor.supports(opt):
            sensor.set_option(opt, val)
    print("AIM: put only the sponge in view, 20-35 cm away. SPACE = lock exposure + start recording, q = quit")
    win = "record_rgbd"
    fs = None
    t_aim = time.time()
    while True:
        fs = pipe.wait_for_frames()
        if a.headless is not None:
            if time.time() - t_aim > a.headless:
                break
            continue
        img, fill, dist = view(fs, cfmt, scale)
        ok = 0.15 < dist < 0.45 and fill > 90
        cv2.putText(img, f"centre {dist * 1e3:.0f} mm  depth fill {fill:.0f}%  {'OK' if ok else 'adjust'}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if ok else (0, 165, 255), 2)
        cv2.putText(img, "SPACE: start recording   q: quit", (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
        cv2.imshow(win, img)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            pipe.stop()
            return
        if k == ord(" "):
            break
    locked = lock_exposure(sensor, fs, fps)
    pipe.stop()
    print(f"  locked {locked}")

    # ---- 2. record -----------------------------------------------------------------------
    pipe = rs.pipeline()
    pipe.start(config(a.width, a.height, fps, cfmt, bag))
    t0, n, last_num, dropped = time.time(), 0, None, 0
    print(f"RECORDING -> {bag}   (SPACE / q to stop)")
    try:
        while True:
            fs = pipe.wait_for_frames()
            num = fs.get_depth_frame().get_frame_number()
            if last_num is not None and num > last_num + 1:
                dropped += num - last_num - 1
            last_num = num
            n += 1
            el = time.time() - t0
            if a.headless is None and n % 2 == 0:   # preview at half rate; the bag gets every frame
                img, _, _ = view(fs, cfmt, scale)
                cv2.circle(img, (20, 20), 8, (0, 0, 255), -1)
                cv2.putText(img, f"REC {el:5.1f}s", (35, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow(win, img)
            k = cv2.waitKey(1) & 0xFF if a.headless is None else -1
            if k in (ord("q"), 27, ord(" ")) or (a.seconds and el >= a.seconds):
                break
    finally:
        pipe.stop()
        if a.headless is None:
            cv2.destroyAllWindows()
    el = time.time() - t0
    mb = bag.stat().st_size / 1e6 if bag.exists() else 0
    print(f"saved {bag}  {el:.1f} s, {n} frames seen ({n / el:.0f} fps), {dropped} dropped, {mb:.0f} MB")
    print(f"next: uv run python scripts/export_rgb.py {bag}\n      uv run python scripts/rgbd_to_pcd.py {bag} --size X Y H --select")


if __name__ == "__main__":
    main()
