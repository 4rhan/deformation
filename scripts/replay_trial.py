"""Run the MuJoCo sponge through a recorded press and compare it with the camera.

The indenter follows the measured dent while the hand presses and is lifted away at each release, exactly as
in the fit. The material is taken from --fit (fit.json) or from --young/--poisson/--damping. Outputs:
  * outputs/rgbd/<name>/replay.png : height of the pressed part / rest and front bulge, real vs sim
  * --video: real camera | MuJoCo seen from the same camera pose, with the height graph underneath
  * otherwise: interactive MuJoCo viewer (yellow = sim sponge, green dots = measured points, blue = indenter)

usage:
  python scripts/replay_trial.py outputs/data/real/press_05.npz --fit results/fit.json --video outputs/press_05.mp4
  python scripts/replay_trial.py outputs/data/real/press_05.npz --fit results/fit.json        # viewer
  python scripts/replay_trial.py outputs/data/real/press_05.npz --poisson 0.3 --damping 0.2   # try a material
viewer keys: SPACE play/pause   RIGHT/LEFT step (paused)   UP/DOWN speed   HOME restart
"""
import argparse
import os
import signal
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("trial")
ap.add_argument("--young", type=float, default=2e4, help="Pa (held fixed in the fit)")
ap.add_argument("--poisson", type=float, default=0.25)
ap.add_argument("--damping", type=float, default=0.02, help="s")
ap.add_argument("--count", type=int, nargs=3, default=[7, 5, 4], help="flex grid")
ap.add_argument("--timestep", type=float, default=1e-3)
ap.add_argument("--video", default=None, help="write a side-by-side mp4 here instead of opening the viewer")
ap.add_argument("--recording", default=None,
                help="the .db3 (default outputs/data/recordings/<name>.db3): real camera video for --video")
ap.add_argument("--fps", type=float, default=30.0)
ap.add_argument("--brighten", type=float, default=1.6,
                help="display only: gamma for the real camera panel of --video (1 = as recorded; the D405 colour at "
                     "90 fps is dark). Gain is set once from the first frame so brightness never flickers")
ap.add_argument("--fit", default=None, help="fit.json from fit_press.py: use its E/nu/tau and grid "
                                            "(this is how a held-out clip is predicted)")
a = ap.parse_args()
if a.fit:
    import json
    _fit = json.load(open(a.fit))
    a.young, a.poisson, a.damping = (float(_fit["best"][k]) for k in ("young", "poisson", "damping"))
    a.count = list(_fit.get("grid", a.count))
    print(f"using fit {a.fit}: nu={a.poisson:.3f} tau={a.damping:.3f} s  trained on {_fit.get('trials')}")
if a.video:
    os.environ.setdefault("MUJOCO_GL", "egl")
import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from softfit.capture import BagSource, top_profile  # noqa: E402
from softfit.fit import observed_top, smooth_plate  # noqa: E402
from softfit.press_scene import PressParams, PressSim  # noqa: E402
from softfit.trial import Trial  # noqa: E402

tr = Trial.load(a.trial)
F = len(tr.t)
tops = np.array([observed_top(c, q=99) if len(c) else np.nan for c in tr.clouds])
rest = float(np.nanmedian(tops[:15]))

plate_size = None if tr.plate_size is None else tuple(float(v) for v in tr.plate_size)   # found by rgbd_to_pcd
plate_x = float(tr.plate_x)
xs_prof = np.arange(-tr.size[0] / 2 + 0.005, tr.size[0] / 2 - 0.004, 0.005)
if plate_size is not None:
    contact = np.abs(xs_prof - plate_x) < plate_size[0] / 2 - 0.005
    centre = np.array([np.nanmedian(top_profile(c, xs_prof[contact])) if len(c) > 50 else np.nan for c in tr.clouds])
    print(f"  indenter: {plate_size[0] * 1e3:.0f} x {plate_size[1] * 1e3:.0f} mm at x = {plate_x * 1e3:+.0f} mm")
else:
    centre = tops.copy()
real_c = centre / np.nanmedian(centre[:15])              # real top of the pressed part / its rest height
size = (float(tr.size[0]), float(tr.size[1]), rest)      # sim height = measured rest height (no pre-squeeze)
cam_kw = {}
calib = ROOT / "outputs" / "rgbd" / tr.name / "calib.npz"
if calib.exists():          # put MuJoCo's camera where the real camera was (table frame), same field of view
    cz = np.load(calib)
    Tct, Kc = cz["T_cam_table"], cz["K"]
    Rt = Tct[:3, :3].T                                    # camera -> table rotation
    cpos = -Rt @ Tct[:3, 3]
    cam_kw = dict(cam_pos=tuple(cpos), cam_target=tuple(cpos + 0.2 * Rt[:, 2]),
                  cam_fovy=float(np.degrees(2 * np.arctan(cz["seed_mask"].shape[0] / 2 / Kc[1, 1]))))
p = PressParams(size=size, count=tuple(a.count), timestep=a.timestep, young=a.young, poisson=a.poisson,
                damping=a.damping, plate_size=plate_size, plate_x=plate_x, **cam_kw)
sim = PressSim(p)
m, d = sim.model, sim.data
t = tr.t - tr.t[0]
h = smooth_plate(tr.plate_h, t)
real_cs = smooth_plate(real_c, t)          # same filter on the measured curve (depth noise), raw drawn faint
print(f"{tr.name}: {F} frames, {t[-1]:.1f} s | sim {size[0] * 1e3:.0f}x{size[1] * 1e3:.0f}x{size[2] * 1e3:.1f} mm, "
      f"nu={a.poisson:.3f} tau={a.damping:.3f} s | simulating ...")


# ---- one run through the whole clip, keep the state at every camera frame --------------------------------
qs, sim_top, sim_bulge = [], [], []
front_mid = None


def grab(s):
    global front_mid
    v = s.vertices()
    if front_mid is None:   # front-face vertices around mid height (the camera's view of the bulge)
        z, y = v[:, 2], v[:, 1]
        front_mid = np.where((y < y.min() + 1e-6) & (np.abs(z - np.median(np.unique(z))) < 0.6 * size[2] / (a.count[2] - 1)))[0]
    qs.append(s.data.qpos.copy())
    vt = v[s.top_ids]
    sel = np.abs(vt[:, 0] - plate_x) < (plate_size[0] / 2 if plate_size else 1.0)
    sim_top.append(float(vt[sel, 2].mean()) + s.p.radius)
    sim_bulge.append(-(v[front_mid, 1].mean()) - size[1] / 2)


t0 = time.time()
res = sim.run(t, h, t, keep_verts=False, on_sample=grab)
n = len(qs)
print(f"  done in {time.time() - t0:.0f} s{'' if res['ok'] else '  -- SIM WENT UNSTABLE at t=%.2f s' % t[n - 1]}")
sim_top, sim_bulge = np.array(sim_top), np.array(sim_bulge)
sim_bulge -= sim_bulge[:15].mean()

# real front bulge: front-face points at 35-65 % of the current height, middle half of the width
real_bulge = np.full(F, np.nan)
for f, c in enumerate(tr.clouds):
    if len(c) < 50 or not np.isfinite(tops[f]):
        continue
    sel = (np.abs(c[:, 0]) < size[0] / 4) & (c[:, 2] > 0.35 * tops[f]) & (c[:, 2] < 0.65 * tops[f])
    if sel.sum() > 10:
        real_bulge[f] = -np.median(c[sel, 1]) - size[1] / 2
real_bulge -= np.nanmedian(real_bulge[:15])

# ---- the one graph -------------------------------------------------------------------------------------
diag = ROOT / "outputs" / "rgbd" / tr.name
diag.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
ax[0].plot(t, real_c, color="#2a7", lw=0.6, alpha=0.35)
ax[0].plot(t, real_cs, color="#2a7", lw=1.8, label="real (camera)" + (", dented part" if plate_size else ""))
ax[0].plot(t[:n], sim_top / rest, color="#c53", lw=1.5, ls="--", label="MuJoCo")
ax[0].set_ylabel("top height / rest" + ("\n(pressed part)" if plate_size else ""))
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3)
ax[1].plot(t, real_bulge * 1e3, color="#2a7", lw=1.2)
ax[1].plot(t[:n], sim_bulge * 1e3, color="#c53", lw=1.5, ls="--")
ax[1].set_ylabel("front bulge [mm]")
ax[1].set_xlabel("time [s]")
ax[1].grid(alpha=0.3)
fig.suptitle(f"{tr.name}: nu={a.poisson:.2f}, tau={a.damping:.2f} s  "
             + (f"({plate_size[0] * 1e3:.0f} mm indenter at x={plate_x * 1e3:+.0f} mm)" if plate_size else "(flat plate)"))
fig.tight_layout()
fig.savefig(diag / "replay.png", dpi=120)
plt.close(fig)
ok = np.isfinite(real_bulge[:n])
rmse_b = float(np.sqrt(np.mean((real_bulge[:n][ok] - sim_bulge[ok]) ** 2)) * 1e3)
print(f"  graph -> {diag / 'replay.png'}   front bulge real vs sim: RMS diff {rmse_b:.2f} mm, "
      f"max real {np.nanmax(real_bulge) * 1e3:.2f} mm, max sim {sim_bulge.max() * 1e3:.2f} mm")

m.geom_rgba[sim.plate_geom] = (0.2, 0.45, 0.9, 0.5)
m.flex_rgba[:] = (0.95, 0.8, 0.2, 0.35)     # see-through sponge: the measured points sit ON its front face
eye = np.eye(3).ravel()


def draw_points(scn, f):
    P = tr.clouds[f]
    if len(P) > 3000:
        P = P[np.random.default_rng(f).choice(len(P), 3000, replace=False)]
    for q in P:
        if scn.ngeom >= scn.maxgeom:
            break
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.0013, 0, 0]), q, eye,
                            np.array([0.05, 0.85, 0.25, 1], np.float32))
        g.emission = 0.6
        scn.ngeom += 1


def set_frame(f):
    f = min(f, n - 1)
    d.qpos[:] = qs[f]
    sim.set_plate(float(h[f]))
    mujoco.mj_forward(m, d)
    return f


def label(f):
    return (f"t = {t[f]:.2f} s   real top {real_c[f] * 100:.0f}%   sim top {sim_top[min(f, n - 1)] / rest * 100:.0f}%   "
            f"nu={a.poisson:.2f} tau={a.damping:.2f} s")


if a.video:
    if not calib.exists():
        sys.exit(f"--video needs {calib} (written by rgbd_to_pcd.py) to place the MuJoCo camera")
    rec = Path(a.recording) if a.recording else ROOT / "outputs" / "data" / "recordings" / f"{tr.name}.db3"
    src = BagSource(rec, align=False)
    Hc, Wc = cz["seed_mask"].shape
    r = mujoco.Renderer(m, Hc, Wc)
    m.flex_rgba[:] = (0.85, 0.75, 0.2, 1.0)              # solid foam, like a real sponge
    m.geom_rgba[sim.plate_geom] = (0.2, 0.45, 0.9, 0.8)
    GH = 170                                              # graph strip height
    real_n, sim_n = real_cs, np.r_[sim_top / rest, np.full(F - n, np.nan)]
    lo = np.nanmin(np.r_[real_n, sim_n]) - 0.01
    hi = np.nanmax(np.r_[real_n, sim_n]) + 0.01
    gx = lambda tt: int(50 + (tt / t[-1]) * (2 * Wc - 70))
    gy = lambda v: int(GH - 25 - (v - lo) / (hi - lo) * (GH - 45))
    base = np.full((GH, 2 * Wc, 3), 255, np.uint8)
    for curve, col in ((real_n, (90, 160, 40)), (sim_n, (40, 80, 210))):
        ok_ = np.isfinite(curve)
        pts_ = np.array([[gx(tt), gy(v)] for tt, v in zip(t[ok_], curve[ok_])], np.int32)
        cv2.polylines(base, [pts_], False, col, 2, cv2.LINE_AA)
    cv2.putText(base, "top height / rest" + (" (pressed part)" if plate_size else "") + ":   real (green)   MuJoCo (red)", (50, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (30, 30, 30), 1, cv2.LINE_AA)
    for v in (np.round(lo, 2), np.round(hi, 2)):
        cv2.putText(base, f"{v:.2f}", (5, gy(v) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)
    step = max(1, int(round((1 / np.median(np.diff(t))) / a.fps)))
    out = None
    lut = None
    for fr in src:
        f = fr.i
        if lut is None:   # one fixed tone curve for the whole clip: scale the 99th pct to ~white, then gamma
            g99 = max(float(np.percentile(fr.color, 99)), 1.0)
            x = np.clip(np.arange(256) / g99 * 0.95, 0, 1)
            lut = (255 * x ** (1 / max(a.brighten, 1e-3))).astype(np.uint8)
        if f >= n:
            break
        if f % step:
            continue
        set_frame(f)
        r.update_scene(d, camera="depthcam")
        simg = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)
        real = cv2.LUT(fr.color, lut)
        cv2.putText(real, f"REAL  t = {t[f]:.2f} s", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(simg, f"MuJoCo (fitted)  nu={a.poisson:.2f}  tau={a.damping:.2f} s",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        g = base.copy()
        cv2.line(g, (gx(t[f]), 25), (gx(t[f]), GH - 20), (0, 0, 0), 1)
        img = np.vstack([np.hstack([real, simg]), g])
        if out is None:
            out = cv2.VideoWriter(a.video + ".tmp.mp4", cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (img.shape[1], img.shape[0]))
        out.write(img)
    out.release()
    os.system(f"ffmpeg -y -loglevel error -i '{a.video}.tmp.mp4' -c:v libx264 -pix_fmt yuv420p -crf 20 "
              f"-vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' '{a.video}' && rm '{a.video}.tmp.mp4'")
    print(f"  video -> {a.video}")
    sys.exit(0)

import mujoco.viewer  # noqa: E402

lock = threading.Lock()
st = {"i": 0, "paused": False, "speed": 1.0, "step": 0, "home": False}


def on_key(k):
    with lock:
        if k == 32: st["paused"] = not st["paused"]
        elif k == 262: st["step"] += 1
        elif k == 263: st["step"] -= 1
        elif k == 265: st["speed"] = min(8, st["speed"] * 2)
        elif k == 264: st["speed"] = max(1 / 8, st["speed"] / 2)
        elif k == 268: st["home"] = True


signal.signal(signal.SIGINT, lambda *_: os._exit(0))
dt = float(np.median(np.diff(t)))
with mujoco.viewer.launch_passive(m, d, key_callback=on_key) as v:
    v.cam.lookat[:] = (0, 0, size[2] / 2)
    v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.30, 70, -15
    t_last = time.time()
    while v.is_running():
        with lock:
            if st["home"]:
                st["i"], st["home"] = 0, False
            if st["paused"]:
                st["i"] = int(np.clip(st["i"] + st["step"], 0, n - 1))
            st["step"] = 0
            paused, speed = st["paused"], st["speed"]
        now = time.time()
        if not paused and now - t_last >= dt / speed:
            with lock:
                st["i"] = (st["i"] + max(1, int((now - t_last) * speed / dt))) % n
            t_last = now
        f = st["i"]
        with v.lock():
            set_frame(f)
            v.user_scn.ngeom = 0
            draw_points(v.user_scn, f)
        v.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, "", label(f)
                      + ("  [paused]" if paused else "") + f"  speed {speed:g}x"),
                     (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, "",
                      "yellow = MuJoCo sponge   green = measured points   blue = indenter   "
                      "SPACE pause  LEFT/RIGHT step  UP/DOWN speed  HOME restart")])
        v.sync()
        time.sleep(0.005)
