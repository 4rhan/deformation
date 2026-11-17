"""RealSense recording -> per-frame point clouds of the sponge's front face, and press detection.

The camera is level, facing the sponge's front face. Per clip:
  1. reference: per-pixel median depth of the first `ref_frames` frames (sponge untouched, nothing moving)
  2. front face: RANSAC plane inside the user's pixel box (--roi), largest flat blob = the face
  3. coordinate frame (the same as the simulator's): x along the face, y into the sponge, z up,
     z = 0 at the face's bottom edge, x = 0 at its middle, y = 0 half the sponge depth behind the face
  4. every frame: keep depth pixels in a thin slab at the front face -> drop flying pixels at depth edges ->
     back-project -> outlier removal -> 2 mm voxel grid
  5. press_region: from the top edge of those clouds, where the sponge was pressed (indenter width and x),
     how deep over time, and when the hand let go (releases)

Depth is back-projected with the depth stream's own rays (colour is mapped onto depth, never the other way),
so depth values are never resampled.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.spatial import cKDTree


# ----------------------------------------------------------------------------------------- recording
@dataclass
class Frame:
    i: int            # frame index in the clip
    t: float          # seconds from the first frame (sensor timestamps)
    depth: np.ndarray  # (H,W) float32 metres, 0 = invalid
    color: np.ndarray  # (H,W,3) uint8 BGR, registered to the depth pixels
    number: int = -1  # sensor frame counter (gaps = dropped frames)


def pinhole_rays(K, width, height):
    """(H,W,3) unit-depth rays for an undistorted pinhole camera."""
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    return np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1).astype(np.float32)


class BagSource:
    """A RealSense recording (.db3 from scripts/record_rgbd.py), played back frame by frame, every frame in order."""

    def __init__(self, path, align=True):
        import pyrealsense2 as rs
        self.rs, self.path, self.align = rs, str(path), align   # align=False: raw colour image (for videos)
        pipe, prof = self._start()
        dev = prof.get_device()
        dp = prof.get_stream(rs.stream.depth).as_video_stream_profile()
        intr = dp.get_intrinsics()
        self.depth_scale = dev.first_depth_sensor().get_depth_scale()
        self.width, self.height, self.fps = intr.width, intr.height, dp.fps()
        self.K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1.0]])
        if all(abs(c) < 1e-9 for c in intr.coeffs):
            self.rays = pinhole_rays(self.K, intr.width, intr.height)
        else:   # exact librealsense deprojection, once per pixel
            self.rays = np.array([[rs.rs2_deproject_pixel_to_point(intr, [u, v], 1.0) for u in range(intr.width)]
                                  for v in range(intr.height)], np.float32)
        g = lambda k: dev.get_info(k) if dev.supports(k) else "?"
        self.meta = {"source": self.path, "device": g(rs.camera_info.name), "serial": g(rs.camera_info.serial_number),
                     "width": self.width, "height": self.height, "fps": self.fps, "depth_scale": self.depth_scale,
                     "K": self.K.tolist(), "distortion": str(intr.model), "coeffs": [float(c) for c in intr.coeffs]}
        pipe.stop()

    def _start(self):
        rs = self.rs
        cfg = rs.config()
        cfg.enable_device_from_file(self.path, repeat_playback=False)
        pipe = rs.pipeline()
        prof = pipe.start(cfg)
        prof.get_device().as_playback().set_real_time(False)
        return pipe, prof

    def __iter__(self):
        rs = self.rs
        pipe, _ = self._start()
        align = rs.align(rs.stream.depth)          # colour -> depth pixels; depth untouched
        t0, i, last = None, 0, None
        try:
            while True:
                ok, fs = pipe.try_wait_for_frames(2000)
                if not ok:
                    break
                fs = align.process(fs) if self.align else fs
                d, c = fs.get_depth_frame(), fs.get_color_frame()
                if not d or not c or d.get_frame_number() == last:   # playback repeats a frameset now and then
                    continue
                last = d.get_frame_number()
                ts = d.get_timestamp() / 1000.0
                t0 = ts if t0 is None else t0
                depth = np.asanyarray(d.get_data()).astype(np.float32) * self.depth_scale
                color = np.asanyarray(c.get_data())
                if c.get_profile().format() == rs.format.rgb8:
                    color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                yield Frame(i, ts - t0, depth, color.copy(), d.get_frame_number())
                i += 1
        finally:
            pipe.stop()


# ----------------------------------------------------------------------------------------- geometry
def plane_ransac(P, thr=0.003, iters=400, seed=0, max_pts=30000):
    """Dominant plane among camera-frame points P (N,3). Returns unit n, d with n.p + d = 0 (n towards the
    camera) and fit info."""
    rng = np.random.default_rng(seed)
    if len(P) > max_pts:
        P = P[rng.choice(len(P), max_pts, replace=False)]
    best, best_n = None, -1
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        d = -n @ a
        if d < 0:                       # orient towards the camera (origin): n.0 + d > 0
            n, d = -n, -d
        cnt = int((np.abs(P @ n + d) < thr).sum())
        if cnt > best_n:
            best, best_n = (n, d), cnt
    if best is None:
        raise RuntimeError("no plane found in the sponge box")
    n, d = best
    for _ in range(3):                  # least-squares refit on the inliers
        inl = P[np.abs(P @ n + d) < thr]
        c = inl.mean(0)
        n2 = np.linalg.svd(inl - c, full_matrices=False)[2][-1]
        n = n2 if n2 @ n > 0 else -n2
        d = -n @ c
    res = inl @ n + d
    return n, float(d), {"inliers": int(len(inl)), "inlier_frac": float(len(inl) / len(P)),
                         "rms_mm": float(np.sqrt(np.mean(res ** 2)) * 1e3)}


@dataclass
class TableFrame:
    R: np.ndarray        # rows = frame x, y, z axes in camera coordinates
    o: np.ndarray        # frame origin in camera coordinates
    size: np.ndarray     # measured sponge extents x, y, height [m]

    def to_table(self, Pc):
        return (Pc - self.o) @ self.R.T

    @property
    def T_cam_table(self):  # 4x4: frame coords -> camera coords
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self.R.T, self.o
        return T


def depth_edges(dep, thr):
    """Flying pixels: depth mixtures of foreground and background at silhouettes, plus pixels next to holes.

    A pixel is flagged when, along image rows or columns, it is
      - off the straight line through its two neighbours by > thr   (step edge next to it), or
      - on a ramp steeper than 1.5 * thr per pixel on BOTH sides     (a mixture between the two levels).
    Plain slopes (a face seen at a grazing angle, ~1-2 mm/px) pass, so faces keep their full extent."""
    d = np.pad(dep.astype(np.float32), 1, mode="edge")
    c = d[1:-1, 1:-1]
    bad = np.zeros(dep.shape, bool)
    for a, b in ((d[1:-1, :-2], d[1:-1, 2:]), (d[:-2, 1:-1], d[2:, 1:-1])):
        bad |= np.abs(c - (a + b) / 2) > thr
        da, db = c - a, b - c
        bad |= (np.abs(da) > 1.5 * thr) & (np.abs(db) > 1.5 * thr) & (np.sign(da) == np.sign(db))
        bad |= (a <= 0) | (b <= 0)
    return bad


def voxel_down(pts: np.ndarray, voxel: float = 0.002) -> np.ndarray:
    """Average points per voxel (keeps density uniform, ~1 point per `voxel` metres)."""
    if len(pts) == 0:
        return pts
    key = np.floor(pts / voxel).astype(np.int64)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    out = np.zeros((len(cnt), 3))
    np.add.at(out, inv.ravel(), pts)
    return out / cnt[:, None]


def remove_outliers(pts: np.ndarray, k: int = 8, std_ratio: float = 2.0) -> np.ndarray:
    """Statistical outlier removal (mean k-NN distance > mean + std_ratio * std)."""
    if len(pts) <= k:
        return pts
    d, _ = cKDTree(pts).query(pts, k + 1)
    md = d[:, 1:].mean(1)
    return pts[md < md.mean() + std_ratio * md.std()]


# ----------------------------------------------------------------------------------------- per clip
@dataclass
class Options:
    roi: tuple                        # x, y, w, h pixel box round the sponge
    size_hint: tuple                  # sponge x, y, height from a ruler [m]
    ref_frames: int = 15              # first frames of the clip: sponge untouched
    front_band: float = 0.010         # sponge points may bulge this far towards the camera [m]
    face_depth: float = 0.012         # keep points at most this far BEHIND the front face [m]; the weight
                                      # sitting in the dent further back is then never taken as sponge
    margin: float = 0.015             # extra width allowed left/right of the face (bulge) [m]
    edge_thr: float = 0.003           # flying-pixel filter (see depth_edges) [m]
    voxel: float = 0.002
    zmax: float = 0.8                 # ignore depth beyond this [m]


@dataclass
class Calib:
    n: np.ndarray
    d: float
    plane: dict
    frame: TableFrame
    seed_mask: np.ndarray
    crop: tuple                       # y0, y1, x0, x1 image region that can contain the sponge
    info: dict = field(default_factory=dict)


def _largest_blob(m, roi=None, min_area=400):
    m = m.astype(np.uint8)
    if roi is not None:
        x, y, w, h = roi
        r = np.zeros_like(m)
        r[y:y + h, x:x + w] = 1
        m &= r
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1 or st[1:, cv2.CC_STAT_AREA].max() < min_area:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return lab == k


def front_frame(Pc, valid, dep, opt: Options):
    """The coordinate frame from the sponge's front face:
      y = into the sponge (away from the camera, = -face normal), z = up along the face's vertical edges
      (the edge direction closest to image-up), x = y cross z; z = 0 at the face's bottom edge,
      x = 0 at the middle of the face, y = 0 half the ruler depth behind the face."""
    x0, y0, w, h = opt.roi
    box = np.zeros_like(valid)
    box[y0:y0 + h, x0:x0 + w] = valid[y0:y0 + h, x0:x0 + w]
    if box.sum() < 500:
        raise RuntimeError(f"only {box.sum()} pixels with depth inside the sponge box")
    n, d, pinfo = plane_ransac(Pc[box])
    dist = Pc @ n + d
    seed = _largest_blob(valid & (np.abs(dist) < 0.008), opt.roi)
    if seed is None:
        raise RuntimeError("no flat front face found inside the sponge box")
    F = Pc[seed & ~depth_edges(dep, opt.edge_thr)]
    e1 = np.array([1.0, 0, 0]) - n[0] * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    p0 = -d * n
    q = np.stack([(F - p0) @ e1, (F - p0) @ e2], 1)
    (_, _), (_, _), ang0 = cv2.minAreaRect((q * 1e4).astype(np.float32))

    def axes(deg):
        a = np.deg2rad(deg)
        return np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])

    def area(deg):
        s2 = q @ axes(deg).T
        return np.prod(np.percentile(s2, 99, axis=0) - np.percentile(s2, 1, axis=0))

    cands = ang0 + np.arange(-5, 5.001, 0.1)
    dirs = axes(cands[int(np.argmin([area(c) for c in cands]))]) @ np.stack([e1, e2])   # (2,3) in camera coords
    img_up = np.array([0.0, -1.0, 0.0])
    k = int(np.argmax(np.abs(dirs @ img_up)))
    up = dirs[k] * np.sign(dirs[k] @ img_up)
    yax = -n
    xax = np.cross(yax, up)
    u, v = (F - p0) @ up, (F - p0) @ xax
    bottom, top = np.percentile(u, 0.5), np.percentile(u, 99.5)
    vlo, vhi = np.percentile(v, 0.5), np.percentile(v, 99.5)
    W, H = vhi - vlo, top - bottom
    sx, sy = opt.size_hint[0], opt.size_hint[1]
    if abs(W - sy) < abs(W - sx):          # the sponge is turned: its short side faces the camera
        sx, sy = sy, sx
    o = p0 + (vlo + vhi) / 2 * xax + bottom * up + sy / 2 * yax
    tf = TableFrame(R=np.stack([xax, yax, up]), o=o, size=np.array([W, sy, H]))
    return tf, seed, n, d, pinfo


def calibrate(ref_depths, rays, K, opt: Options) -> Calib:
    """Steps 1-3 from the reference frames (sponge untouched, nothing moving)."""
    D = np.stack(ref_depths)
    D[D <= 0] = np.nan
    with np.errstate(all="ignore"):
        dep = np.nanmedian(D, 0)
        flicker = np.nanstd(D, 0)
    valid = np.isfinite(dep) & (dep < opt.zmax)
    dep = np.where(valid, dep, 0).astype(np.float32)
    Pc = rays * dep[..., None]
    tf, seed, n, d, pinfo = front_frame(Pc, valid, dep, opt)
    seed_e = cv2.erode(seed.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    # image region that can contain the sponge: project a generous box around it
    sx, sy, sz = tf.size
    bx, by = sx / 2 + 0.06, sy / 2 + 0.06
    corners = np.array([[x, y, z] for x in (-bx, bx) for y in (-by, by) for z in (0, sz + 0.15)])
    pc = corners @ tf.R + tf.o
    us = pc[:, 0] / pc[:, 2] * K[0, 0] + K[0, 2]
    vs = pc[:, 1] / pc[:, 2] * K[1, 1] + K[1, 2]
    H, W = dep.shape
    crop = (max(int(vs.min()) - 10, 0), min(int(vs.max()) + 11, H), max(int(us.min()) - 10, 0), min(int(us.max()) + 11, W))
    info = {"plane": pinfo, "size_measured_mm": (tf.size * 1e3).round(2).tolist(), "seed_px": int(seed.sum()),
            "ref_flicker_mm": float(np.nanmedian(flicker[seed_e]) * 1e3), "camera_to_plane_mm": float(d * 1e3),
            "camera_tilt_deg": float(np.degrees(np.arcsin(abs(tf.R[2] @ np.array([0, 0, 1.0])))))}
    return Calib(n=n, d=d, plane=pinfo, frame=tf, seed_mask=seed, crop=crop, info=info)


def process_frame(fr: Frame, rays, cal: Calib, opt: Options):
    """One frame -> front-face sponge cloud (sponge frame) and its image mask. Only the crop region is touched."""
    y0, y1, x0, x1 = cal.crop
    dep = fr.depth[y0:y1, x0:x1]
    valid = (dep > 0) & (dep < opt.zmax)
    Pt = cal.frame.to_table(rays[y0:y1, x0:x1] * dep[..., None])
    x, y, z = Pt[..., 0], Pt[..., 1], Pt[..., 2]
    sx, sy, sz = cal.frame.size
    # a thin slab at the front face: may bulge towards the camera by front_band, at most face_depth behind it
    foot = (np.abs(x) < sx / 2 + opt.margin) & (y > -sy / 2 - opt.front_band) & (y < -sy / 2 + opt.face_depth)
    gate = valid & foot & (z > 0.001) & (z < sz + 0.004)
    m = gate.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) & gate
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:   # drop specks: keep blobs >= 5 % of the biggest
        big = st[1:, cv2.CC_STAT_AREA]
        keep = np.zeros(n, bool)
        keep[1:] = big >= max(30, 0.05 * big.max())
        m = keep[lab].astype(np.uint8)
    full_mask = m.astype(bool)
    m = m & ~depth_edges(dep, opt.edge_thr)
    pts = Pt[m.astype(bool)]
    if len(pts) > 20:
        pts = remove_outliers(pts.astype(np.float64), k=8, std_ratio=2.5)
        pts = voxel_down(pts, opt.voxel)
    return {"pts": np.asarray(pts, np.float64), "mask": full_mask, "n_px": int(full_mask.sum())}


# ----------------------------------------------------------------------------------------- press detection
def top_profile(c, xs, half=0.005):
    """Top edge height of a front-face cloud in x-strips of width 2*half."""
    out = np.full(len(xs), np.nan)
    for i, x in enumerate(xs):
        s_ = c[np.abs(c[:, 0] - x) < half]
        if len(s_) > 8:
            out[i] = np.percentile(s_[:, 2], 99)
    return out


def press_region(clouds, size, ref_frames, rest_top, t, step=0.005, frac=0.5, full_frac=0.8, k_noise=4.0,
                 min_depth=0.001, rel_depth=0.15, min_gap_s=0.4, min_press_s=0.3, release_frac=0.75, gap=0.002,
                 park=0.03):
    """Where, how deep and when was the sponge pressed? Everything from the front face's top-edge profile.

    * where: at the most squeezed moment, the contiguous run of x-strips (around the deepest one) that dropped
      >= frac of the largest drop -> indenter width; centre x = drop-weighted centroid of that run. Wider than
      full_frac of the sponge -> flat plate over the whole top.
    * how deep: d(t) = median drop of that run vs the reference frames; the sim indenter is driven to rest_top - d.
      Pressed = d > max(k_noise x reference noise, min_depth, rel_depth x deepest d); pressed stretches closer
      than min_gap_s are merged, ones shorter than min_press_s dropped (hand hovering must not count).
    * release: in every pressed stretch, the last frame still at >= release_frac of that stretch's deepest d.
      From there the indenter is PARKED `park` above the rest top until the dent has recovered (d back below
      the threshold), so the sim's spring-back
      is a prediction, not an input. Elsewhere it hovers `gap` above the top.
    Returns dict(plate_size, plate_x, plate_h, depth, pressed, releases, threshold)."""
    xs = np.arange(-size[0] / 2 + step, size[0] / 2 - step + 1e-9, step)
    prof = np.array([top_profile(c, xs) if len(c) > 50 else np.full(len(xs), np.nan) for c in clouds])
    with np.errstate(all="ignore"):
        drop = np.nanmedian(prof[:ref_frames], 0) - prof
        per_frame = np.nanpercentile(drop, 90, axis=1)
    F = len(clouds)
    out = dict(plate_size=None, plate_x=0.0, plate_h=np.full(F, rest_top + gap), depth=np.zeros(F),
               pressed=np.zeros(F, bool), releases=[], threshold=np.nan)
    if not np.isfinite(per_frame).any() or np.nanmax(per_frame) < min_depth:
        return out
    f = int(np.nanargmax(per_frame))
    with np.errstate(all="ignore"):
        dm = np.nanmean(drop[max(0, f - 10):f + 10], 0)
    j = int(np.nanargmax(dm))
    hit = np.nan_to_num(dm) >= frac * dm[j]
    lo, hi = j, j
    while lo > 0 and hit[lo - 1]:
        lo -= 1
    while hi < len(xs) - 1 and hit[hi + 1]:
        hi += 1
    width = float(xs[hi] - xs[lo] + 2 * step)
    if width < full_frac * size[0]:
        out["plate_size"] = (width, float(size[1]))
        # centre = drop-weighted centroid of the run (above the half-max): the bending sheet spreads the dent
        # unevenly to the sides, so the run's midpoint can sit 10+ mm off the indenter
        w_ = np.clip(np.nan_to_num(dm[lo:hi + 1]) - frac * dm[j], 0, None)
        out["plate_x"] = float(np.sum(xs[lo:hi + 1] * w_) / np.sum(w_)) if w_.sum() > 0 else float((xs[lo] + xs[hi]) / 2)
        part = np.abs(xs - out["plate_x"]) < width / 2 - step                # inner strips (edges are half-pressed)
        if not part.any():
            part[j] = True
    else:
        part = np.ones(len(xs), bool)
    with np.errstate(all="ignore"):
        d = np.nanmedian(drop[:, part], 1)
    d = np.where(np.isfinite(d), d, 0.0)
    from .fit import smooth_plate
    noise = float(np.std(d[:ref_frames]))
    d = smooth_plate(d, t)
    thr = max(k_noise * noise, min_depth, rel_depth * float(d.max()))
    pressed = d > thr
    dt = float(np.median(np.diff(t)))
    runs = np.flatnonzero(np.diff(np.r_[0, pressed.astype(int), 0]))
    runs = runs.reshape(-1, 2)                     # [start, end) of each pressed stretch
    merged = []
    for a_, b_ in runs:
        if merged and (a_ - merged[-1][1]) * dt < min_gap_s:
            merged[-1][1] = b_
        else:
            merged.append([a_, b_])
    pressed[:] = False
    for a_, b_ in merged:
        if (b_ - a_) * dt >= min_press_s:
            pressed[a_:b_] = True
    h = np.where(pressed, rest_top - d, rest_top + gap)
    releases = []
    k = 0
    while k < F:                                   # every pressed stretch
        if not pressed[k]:
            k += 1
            continue
        e = k
        while e < F and pressed[e]:
            e += 1
        seg = d[k:e]
        r = k + int(np.where(seg >= release_frac * seg.max())[0][-1])
        h[r + 1:e] = rest_top + park               # spring-back: indenter gone
        releases.append(r)
        k = e
    out.update(plate_h=h, depth=d, pressed=pressed, releases=releases, threshold=thr)
    return out
