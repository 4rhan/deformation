"""Recording -> sponge point clouds + press detection -> trial .npz for scripts/fit_press.py.

The recording must START with the sponge untouched for ~1 s (the first 15 frames set the coordinate frame).
See src/softfit/capture.py for every step.

usage:
  python scripts/rgbd_to_pcd.py outputs/data/recordings/press_01.db3 --size 135 80 35 --select
  python scripts/rgbd_to_pcd.py outputs/data/recordings/press_01.db3 --size 135 80 35 --roi 260 192 340 104

  --size     sponge length x depth x height in mm (ruler); length = the side facing the camera
  --select   draw a box round the sponge on the first frame (prints the --roi to reuse)

outputs:
  outputs/data/real/<name>.npz            trial: point clouds, press position/depth/releases, size
  outputs/rgbd/<name>/plot.png            height of the pressed part / rest height over time, releases marked
  outputs/rgbd/<name>/cloud_check.png     point cloud from 3 sides with the sponge box drawn on
  outputs/rgbd/<name>/preview.mp4         colour video, green = sponge points
  outputs/rgbd/<name>/rest.ply, squeezed.ply   clouds (open in MeshLab / CloudCompare)
  outputs/rgbd/<name>/report.json         calibration + checks
  outputs/rgbd/<name>/calib.npz           camera pose (used by replay_trial.py to place the MuJoCo camera)
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import os  # noqa: E402

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")
import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from softfit.capture import BagSource, Options, calibrate, press_region, process_frame  # noqa: E402
from softfit.fit import observed_top as _ot  # noqa: E402
from softfit.trial import Trial  # noqa: E402


def observed_top(c):
    """Sponge top height from a front-face cloud: its 99th percentile height."""
    return _ot(c, q=99)


def write_ply(path, pts):
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\n"
                "property float z\nend_header\n")
        np.savetxt(f, pts, fmt="%.5f")


def overlay(color, res, crop, t, top_mm):
    img = color.copy()
    y0, y1, x0, x1 = crop
    sub = img[y0:y1, x0:x1]
    sub[res["mask"]] = (0.5 * sub[res["mask"]] + (0, 127, 0)).astype(np.uint8)
    cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 255), 1)
    txt = f"t {t:5.2f}s  top {top_mm:5.1f}mm"
    cv2.putText(img, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


CHUNK = 128          # frames handed to the worker pool at a time
_W = {}


def _init_worker(rays, cal, opt):
    cv2.setNumThreads(1)       # one process per core already; avoid oversubscription in the workers
    _W.update(rays=rays, cal=cal, opt=opt)


def _work(fr):
    return process_frame(fr, _W["rays"], _W["cal"], _W["opt"])


def run(src, opt: Options, name, trial_dir, diag_dir, video=True, quiet=False, workers=None):
    """Process one clip. Returns (trial, report)."""
    log = (lambda *a: None) if quiet else print
    workers = os.cpu_count() if workers is None else workers
    diag_dir.mkdir(parents=True, exist_ok=True)
    it = iter(src)
    ref = []
    for fr in it:
        ref.append(fr)
        if len(ref) >= opt.ref_frames:
            break
    if len(ref) < opt.ref_frames:
        raise RuntimeError(f"clip too short: {len(ref)} frames")
    cal = calibrate([f.depth for f in ref], src.rays, src.K, opt)
    log(f"  front face plane: {cal.plane['rms_mm']:.2f} mm RMS, camera {cal.info['camera_to_plane_mm']:.0f} mm from it")
    log(f"  sponge measured: {' x '.join(f'{v:.1f}' for v in cal.info['size_measured_mm'])} mm"
        f"   (ruler {' x '.join(f'{v * 1e3:.1f}' for v in opt.size_hint)})")

    vw = None
    t, num, clouds = [], [], []

    def frames():
        yield from ref
        yield from it

    def results():
        """process_frame over the clip, CHUNK frames at a time on `workers` processes (frames are independent
        once calibrated); chunks keep memory bounded (each raw frame is ~3 MB)."""
        if workers <= 1:
            for fr in frames():
                yield fr, process_frame(fr, src.rays, cal, opt)
            return
        from multiprocessing import get_context
        with get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(src.rays, cal, opt)) as pool:
            chunk = []
            for fr in frames():
                chunk.append(fr)
                if len(chunk) == CHUNK:
                    yield from zip(chunk, pool.map(_work, chunk, chunksize=max(1, CHUNK // workers)))
                    chunk = []
            if chunk:
                yield from zip(chunk, pool.map(_work, chunk, chunksize=max(1, len(chunk) // workers)))

    for fr, r in results():
        t.append(fr.t)
        num.append(fr.number)
        clouds.append(r["pts"].astype(np.float32))
        if video:
            top = observed_top(r["pts"]) if len(r["pts"]) else np.nan
            img = overlay(fr.color, r, cal.crop, fr.t, top * 1e3)
            if vw is None:
                vw = cv2.VideoWriter(str(diag_dir / "preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                     min(src.fps, 30), (img.shape[1], img.shape[0]))
            vw.write(img)
        if not quiet and fr.i % 100 == 0:
            print(f"  frame {fr.i}: {len(r['pts'])} points", end="\r")
    if vw is not None:
        vw.release()
    t, num = np.array(t), np.array(num)

    # ---- trial fields ------------------------------------------------------------------------
    size = np.array(opt.size_hint)
    if abs(cal.frame.size[0] - size[1]) < abs(cal.frame.size[0] - size[0]):
        size = size[[1, 0, 2]]                     # sponge turned: the short side faces the camera
    rest_top = float(np.nanmedian([observed_top(c) for c in clouds[:opt.ref_frames]]))
    ruler = size.copy()
    size = np.array([size[0], size[1], rest_top])  # sim rest height = MEASURED rest height (ruler only checks it)
    # where it was pressed, how deep, and when it was let go -- all from the sponge's own top edge
    pr = press_region(clouds, size, opt.ref_frames, rest_top, t)
    plate_size, plate_x, plate_h, releases, depth = pr["plate_size"], pr["plate_x"], pr["plate_h"], pr["releases"], pr["depth"]
    present = pr["pressed"]
    plate = np.where(present, rest_top - depth, np.nan)
    where = "flat plate over the whole top" if plate_size is None else \
        f"{plate_size[0] * 1e3:.0f} mm wide indenter centred at x = {plate_x * 1e3:+.0f} mm (- = left)"
    log(f"  pressed: {where}; deepest {depth.max() / rest_top * 100:.1f}% of the height; "
        f"{len(releases)} release(s) at t = {', '.join(f'{t[r]:.2f}' for r in releases)} s")
    trial = Trial(name=name, t=t, plate_h=plate_h, clouds=clouds, size=size,
                  plate_size=None if plate_size is None else np.array(plate_size), plate_x=plate_x)
    trial_dir.mkdir(parents=True, exist_ok=True)
    trial.save(trial_dir / f"{name}.npz")

    # ---- checks ------------------------------------------------------------------------------
    tops = np.array([observed_top(c) if len(c) else np.nan for c in clouds])
    ref_tops = tops[:opt.ref_frames]
    gaps = np.diff(num)[num[1:] > 0] if (num >= 0).all() else np.array([1])
    dropped = int(np.clip(gaps - 1, 0, None).sum())
    meas = cal.frame.size * 1e3
    report = {"name": name, "frames": len(t), "duration_s": float(t[-1]), "fps": float(1 / np.median(np.diff(t))),
              "dropped_frames": dropped, "source": src.meta, "calib": cal.info,
              "T_cam_table": cal.frame.T_cam_table.tolist(), "crop_y0y1x0x1": list(cal.crop),
              "points_per_frame_median": int(np.median([len(c) for c in clouds])),
              "rest_top_mm": rest_top * 1e3, "rest_top_std_mm": float(np.nanstd(ref_tops) * 1e3),
              "pressed_frames": int(present.sum()),
              "min_plate_mm": float(np.nanmin(plate) * 1e3) if present.any() else None,
              "max_compression_pct": float((1 - np.nanmin(plate) / rest_top) * 100) if present.any() else 0.0,
              "size_used_mm": (size * 1e3).tolist(),
              "indenter_mm": None if plate_size is None else [v * 1e3 for v in plate_size],
              "indenter_x_mm": plate_x * 1e3, "release_t": [float(t[r]) for r in releases], "options": {k: v for k, v in vars(opt).items()}}
    checks = []

    def chk(nm, ok, warn, msg):
        checks.append((nm, "PASS" if ok else "WARN" if warn else "FAIL", msg))

    chk("plane", cal.plane["rms_mm"] < 1.0, cal.plane["rms_mm"] < 2.5,
        f"front face plane fit {cal.plane['rms_mm']:.2f} mm RMS")
    dif = meas - ruler * 1e3
    # the camera sees the front face only: its width (x) and height (z) are measured, the depth (y) is the ruler's
    chk("size", abs(dif[0]) < 3.0 and abs(dif[2]) < 3.0, abs(dif[0]) < 6.0 and abs(dif[2]) < 6.0,
        f"front face measured - ruler: x {dif[0]:+.1f}, z {dif[2]:+.1f} mm")
    chk("noise", report["rest_top_std_mm"] < 0.2, report["rest_top_std_mm"] < 0.5,
        f"top height std while still {report['rest_top_std_mm']:.2f} mm")
    chk("drops", dropped == 0, dropped < 0.01 * len(t), f"{dropped} dropped frames, {report['fps']:.0f} fps")
    chk("points", report["points_per_frame_median"] > 300, report["points_per_frame_median"] > 100,
        f"{report['points_per_frame_median']} points / frame (2 mm voxels)")
    chk("depth", report["max_compression_pct"] <= 30, report["max_compression_pct"] <= 35,
        f"deepest press {report['max_compression_pct']:.1f}% of the height (the sim handles <= ~30%)")
    report["checks"] = checks
    json.dump(report, open(diag_dir / "report.json", "w"), indent=1, default=float)

    # ---- figures -----------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(t, np.where(present, np.nan, tops) / rest_top, color="#2a7", lw=1.5, label="sponge top / rest top (not pressed)")
    pp = np.where(present, plate, np.nan) / rest_top
    ax.plot(t, pp, color="#36c", lw=1.5, label="pressed part top / rest top")
    lo = np.nanmin(np.r_[tops[~present] / rest_top, pp[np.isfinite(pp)], 1.0])
    ax.set_ylim(min(0.6, lo - 0.05), 1.1)
    for r in releases:
        ax.axvline(t[r], color="#c53", lw=1, ls="--")
    if releases:
        ax.plot([], [], color="#c53", lw=1, ls="--", label="release (sim indenter lifted away here)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("height / rest height")
    ax.set_title(f"{name}: max compression {report['max_compression_pct']:.1f}%")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(diag_dir / "plot.png", dpi=120)
    plt.close(fig)

    rest = np.concatenate(clouds[:opt.ref_frames])
    squeezed = clouds[int(np.argmax(depth))]       # the most pressed frame
    write_ply(diag_dir / "rest.ply", clouds[opt.ref_frames - 1])
    write_ply(diag_dir / "squeezed.ply", squeezed)
    sx, sy, sz = size
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for a, (i, j, lab, (w, h)) in zip(axs, [(0, 1, "top view (x, y)", (sx, sy)), (0, 2, "front (x, z)", (sx, sz)),
                                            (1, 2, "side (y, z)", (sy, sz))]):
        a.scatter(rest[:, i] * 1e3, rest[:, j] * 1e3, s=0.3, c="#2a7", label="rest")
        a.scatter(squeezed[:, i] * 1e3, squeezed[:, j] * 1e3, s=0.3, c="#c53", label="most squeezed")
        y0 = -h / 2 if j == 1 else 0
        a.add_patch(plt.Rectangle((-w / 2 * 1e3, y0 * 1e3), w * 1e3, h * 1e3, fill=False, ec="k", lw=1, ls="--"))
        a.set_title(lab)
        a.set_aspect("equal")
        a.grid(alpha=0.3)
    axs[0].legend(markerscale=15, fontsize=8)
    fig.suptitle("point cloud in the sponge frame [mm]; dashed = sponge size used for the fit")
    fig.tight_layout()
    fig.savefig(diag_dir / "cloud_check.png", dpi=110)
    plt.close(fig)

    np.savez_compressed(diag_dir / "calib.npz", T_cam_table=cal.frame.T_cam_table, K=src.K,
                        seed_mask=cal.seed_mask, crop=np.array(cal.crop))
    return trial, report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", help=".db3 from record_rgbd.py")
    ap.add_argument("--name", default=None, help="trial name (default: file name)")
    ap.add_argument("--size", type=float, nargs=3, metavar=("X", "Y", "H"), required=True,
                    help="sponge size from a ruler [mm]")
    ap.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"), help="pixel box round the sponge")
    ap.add_argument("--select", action="store_true", help="draw the box round the sponge on the first frame")
    ap.add_argument("--ref-frames", type=int, default=15, help="first frames with the sponge untouched")
    ap.add_argument("--face-depth", type=float, default=12.0,
                    help="keep only points within this many mm behind the front face [mm] (anything further back, "
                         "e.g. the weight pressed into the top, is not the face)")
    ap.add_argument("--trial-dir", default=str(ROOT / "outputs" / "data" / "real"))
    ap.add_argument("--diag-dir", default=str(ROOT / "outputs" / "rgbd"))
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count(),
                    help="processes for the per-frame work (default: all CPU threads; 1 = no multiprocessing)")
    a = ap.parse_args()

    path = Path(a.recording)
    name = a.name or path.stem
    src = BagSource(path)
    print(f"{name}: {src.meta['device']} {src.width}x{src.height} @ {src.fps:.0f} fps")
    roi = tuple(a.roi) if a.roi else None
    if a.select:
        fr = next(iter(src))
        img = fr.color.copy()
        img[fr.depth <= 0] //= 3                      # dim the pixels without depth
        r = cv2.selectROI("box ROUND THE SPONGE (a little margin), then ENTER", img, showCrosshair=False)
        cv2.destroyAllWindows()
        roi = tuple(int(v) for v in r) if r[2] > 0 and r[3] > 0 else None
        if roi:
            print(f"  reuse with: --roi {' '.join(map(str, roi))}")
    if roi is None:
        sys.exit("need a box round the sponge: --select or --roi X Y W H")
    opt = Options(roi=roi, size_hint=tuple(v / 1e3 for v in a.size), ref_frames=a.ref_frames,
                  face_depth=a.face_depth / 1e3)
    diag = Path(a.diag_dir) / name
    try:
        trial, rep = run(src, opt, name, Path(a.trial_dir), diag, video=not a.no_video, workers=a.workers)
    except RuntimeError as e:
        sys.exit(f"\n[FAIL] {e}")
    print(" " * 40)
    for nm, st, msg in rep["checks"]:
        print(f"  [{st:4}] {nm:7} {msg}")
    worst = "FAIL" if any(c[1] == "FAIL" for c in rep["checks"]) else "WARN" if any(c[1] == "WARN" for c in rep["checks"]) else "PASS"
    print(f"\n{worst}: {len(trial.t)} frames, ~{rep['points_per_frame_median']} pts/frame, "
          f"max compression {rep['max_compression_pct']:.1f}%")
    print(f"  trial  -> {Path(a.trial_dir) / (name + '.npz')}")
    print(f"  check  -> {diag}/plot.png, cloud_check.png, preview.mp4")


if __name__ == "__main__":
    main()
