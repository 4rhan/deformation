"""Loss between a simulated press and a recorded one, and the parameter vector the optimiser sees.

Loss for one trial (both terms dimensionless, ~1 at the noise floor):
  pcd    one-sided: every MEASURED point -> nearest point on the simulated outer surface, Huber in mm,
         averaged over points and frames (spring-back frames weighted up). One-sided on purpose: the camera
         only sees the front face.
  top    spring-back curve: height of the pressed part after each release, real vs sim. This is the term
         that carries the damping signal.
The indenter follows the MEASURED dent (smoothed) while the hand presses and is lifted away at each release.
E is not learnable this way (with a prescribed dent the shape barely depends on it), so only nu and tau are
fitted and E is held fixed (scripts/fit_press.py).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.signal import medfilt, savgol_filter
from scipy.spatial import cKDTree

from .press_scene import PressParams, PressSim
from .trial import Trial

# optimiser coordinates: u in [0,1]^3  <->  (log10 E, nu, log10 tau)
BOUNDS = np.array([[np.log10(3e3), np.log10(3e5)],   # log10 young [Pa]   3 kPa .. 300 kPa (PU foams)
                   [0.05, 0.45],                     # poisson
                   [np.log10(3e-3), np.log10(3.0)]])  # log10 damping [s]  3 ms .. 3 s (slow foam spring-back)
PENALTY = 1e6   # score of a failed / crashed simulation (finite, so CMA-ES can rank it last)
NAMES = ("young", "poisson", "damping")


def u_to_params(u):
    x = BOUNDS[:, 0] + np.clip(u, 0, 1) * (BOUNDS[:, 1] - BOUNDS[:, 0])
    return {"young": 10 ** x[0], "poisson": x[1], "damping": 10 ** x[2]}


def params_to_u(d):
    x = np.array([np.log10(d["young"]), d["poisson"], np.log10(d["damping"])])
    return (x - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])


def huber(r, delta):
    a = np.abs(r)
    return np.where(a <= delta, 0.5 * r ** 2, delta * (a - 0.5 * delta))


PLATE_MEDIAN_S = 0.15   # median window [s]: kills single-frame depth spikes, keeps the release step sharp
PLATE_SAVGOL_S = 0.15   # Savitzky-Golay window [s]: removes the remaining ~0.5 mm frame-to-frame depth noise
PLATE_JUMP_M = 0.004    # an indenter jump larger than this (~8x the depth noise) = release (lifted away)
RECOVERY_S = 1.0        # spring-back window after each release [s]


def _odd_frames(seconds, dt, n):
    w = int(round(seconds / dt)) | 1
    return max(1, min(w, n if n % 2 else n - 1))


def smooth_plate(h, t=None, median_s=PLATE_MEDIAN_S, savgol_s=PLATE_SAVGOL_S, jump=PLATE_JUMP_M):
    """Measured indenter height -> the trajectory that drives the sim. The sim follows it exactly, so raw
    depth noise (~0.5 mm/frame) would shake the foam; windows are in SECONDS so any frame rate works.
    Each contiguous run is filtered on its own: a frame-to-frame jump > `jump` (release -> indenter parked
    above) is never smoothed across, so a release stays a one-frame step."""
    h = np.asarray(h, float)
    bad = ~np.isfinite(h)
    if bad.all():
        return h
    if bad.any():   # fill gaps for filtering, put them back afterwards
        i = np.arange(len(h))
        h = np.where(bad, np.interp(i, i[~bad], h[~bad]), h)
    dt = float(np.median(np.diff(t))) if t is not None and len(t) > 1 else 1 / 60
    cuts = np.concatenate([[0], np.where(np.abs(np.diff(h)) > jump)[0] + 1, [len(h)]])
    out = h.copy()
    for a, b in zip(cuts[:-1], cuts[1:]):
        seg = h[a:b]
        wm, ws = _odd_frames(median_s, dt, len(seg)), _odd_frames(savgol_s, dt, len(seg))
        if wm > 1:
            seg = medfilt(np.pad(seg, wm // 2, mode="edge"), wm)[wm // 2: wm // 2 + len(seg)]
        out[a:b] = savgol_filter(seg, ws, 2, mode="nearest") if ws > 3 else seg
    return np.where(bad, np.nan, out)


def simulate(trial: Trial, params: dict, base: PressParams, stride: int = 2):
    """Run the model through the trial's measured press; return per-sampled-frame outputs.
    Every `stride`-th frame is compared, plus every frame of the recovery window."""
    p = replace(base, size=tuple(float(x) for x in trial.size), **params,
                plate_size=None if trial.plate_size is None else tuple(float(x) for x in trial.plate_size),
                plate_x=float(trial.plate_x))
    sim = PressSim(p)
    idx = np.union1d(np.arange(0, len(trial.t), stride), np.where(recovery_window(trial))[0])
    t = trial.t - trial.t[0]
    res = sim.run(t, smooth_plate(trial.plate_h, t), t[idx], keep_verts=True)
    res["idx"] = idx[:len(res["t"])]
    res["surface"] = [sim.surface_points(v) for v in res["verts"]]
    return res


def recovery_window(trial: Trial, duration=RECOVERY_S):
    """Frames right after each release (the sim indenter jumps up and parks), up to `duration` later: the part
    of a trial that carries the damping signal. Without a detected release: after the last deepest frame."""
    h = smooth_plate(trial.plate_h, trial.t)
    rel = np.where(np.diff(trial.plate_h) > PLATE_JUMP_M)[0]
    if not len(rel):
        rel = [np.where(h <= h.min() + 0.0005)[0][-1]]
    win = np.zeros(len(trial.t), bool)
    for r in rel:
        win |= (trial.t > trial.t[r]) & (trial.t <= trial.t[r] + duration)
    return win


def observed_top(cloud, q=90):
    """Top height of the sponge from one cloud: a high percentile of z (robust to depth noise)."""
    return np.percentile(cloud[:, 2], q) if len(cloud) else np.nan


def pressed_top(trial: Trial, cloud, step=0.005):
    """Measured top edge of the PRESSED part (indenter footprint in x) -- the whole-cloud percentile would be
    dominated by the unpressed rest of the sponge for a local / off-centre dent."""
    from .capture import top_profile
    if len(cloud) < 50:
        return np.nan
    if trial.plate_size is None:
        return observed_top(cloud, q=99)
    hw = trial.plate_size[0] / 2
    xs = np.arange(trial.plate_x - hw + step, trial.plate_x + hw - step + 1e-9, step)
    with np.errstate(all="ignore"):
        return float(np.nanmedian(top_profile(cloud, xs))) if len(xs) else observed_top(cloud, q=99)


def top_term(trial: Trial, res: dict, sigma_top_mm=0.5, margin=0.010, n_ref=8):
    """Spring-back term: top height of the pressed part, real vs sim, on recovery-window frames where the
    indenter is >= `margin` clear of the top (parked). Both curves are referenced to their mean over the last
    `n_ref` sampled frames (sponge at rest), which cancels the measurement bias.
    This is the damping signal: the point-cloud term buries it under ~1 mm per-point noise."""
    rec = recovery_window(trial)
    idx = res["idx"]
    obs = np.array([pressed_top(trial, trial.clouds[f]) for f in idx])
    use = rec[idx] & np.isfinite(obs) & (trial.plate_h[idx] > obs + margin)
    ref = slice(len(idx) - n_ref, len(idx))
    if not use.any() or not np.isfinite(obs[ref]).all():
        return 0.0, 0
    sim = res["press_top"]
    d_obs = (obs - obs[ref].mean()) * 1e3
    d_sim = (sim - sim[ref].mean()) * 1e3
    return float(np.mean(((d_obs[use] - d_sim[use]) / sigma_top_mm) ** 2)), int(use.sum())


def trial_loss(trial: Trial, res: dict, sigma_pcd_mm=1.0, delta_mm=3.0, w_recovery=10.0):
    if not res["ok"] or len(res["idx"]) == 0:
        return {"total": PENALTY, "pcd": PENALTY, "top": PENALTY, "ok": False}
    rec = recovery_window(trial)
    per_frame, weights = [], []
    for k, f in enumerate(res["idx"]):
        obs = trial.clouds[f]
        if len(obs) == 0:
            continue
        d, _ = cKDTree(res["surface"][k]).query(obs)
        per_frame.append(huber(d * 1e3, delta_mm).mean())
        weights.append(w_recovery if rec[f] else 1.0)   # up-weight the few frames that see the rebound
    pcd = float(np.average(per_frame, weights=weights)) / (0.5 * sigma_pcd_mm ** 2)
    top, n_top = top_term(trial, res)
    return {"total": pcd + top, "pcd": pcd, "top": top, "n_top": n_top, "ok": True}


def evaluate(job):
    """Multiprocessing entry: job = (trial_path, params_dict, base_dict, stride)."""
    path, params, base, stride = job
    try:
        trial = Trial.load(path)
        res = simulate(trial, params, PressParams(**base), stride)
        out = trial_loss(trial, res)
    except Exception as e:  # a crashed sim must not kill the whole pool.map -> penalty, keep going
        return {"total": PENALTY, "pcd": PENALTY, "top": PENALTY, "ok": False, "error": repr(e)}
    if not np.isfinite(out["total"]):
        out = {**out, "total": PENALTY, "ok": False}
    return out
