"""Held-out test: run the FITTED model on clips it never saw and compare with the camera -- one figure.

For every clip the sim plate follows the measured press only while the hand presses; at each release it is
lifted away, so everything after a release (the spring-back) is the model's PREDICTION. The figure shows the
pressed part's top height / rest height, real vs MuJoCo, with the releases marked; the table gives the
spring-back error (after releases only) in % of the sponge height, also for the training clips.

usage:
  python scripts/eval_heldout.py --fit outputs/fit/fit.json \
      --test outputs/data/real/press_02.npz outputs/data/real/press_05.npz outputs/data/real/press_06.npz
  (training clips are taken from fit.json and scored too, for comparison)
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from softfit.fit import PLATE_JUMP_M, pressed_top, recovery_window, simulate, smooth_plate  # noqa: E402
from softfit.press_scene import PressParams  # noqa: E402
from softfit.trial import Trial  # noqa: E402


def run(job):
    path, params, base = job
    tr = Trial.load(path)
    res = simulate(tr, params, PressParams(**base), stride=1)
    idx = res["idx"]
    real = np.array([pressed_top(tr, tr.clouds[f]) for f in idx])
    rest = float(np.nanmedian(real[:15]))
    t = tr.t[idx] - tr.t[0]
    sim = res["press_top"]
    sim_n = sim / sim[0]
    real_n = smooth_plate(real / rest, t)
    rec = recovery_window(tr)[idx] & np.isfinite(real_n)
    rel = np.where(np.diff(tr.plate_h) > PLATE_JUMP_M)[0]
    err = float(np.sqrt(np.mean((real_n[rec] - sim_n[rec]) ** 2)) * 100) if rec.any() else np.nan
    return {"name": tr.name, "t": t, "real": real_n, "real_raw": real / rest, "sim": sim_n, "ok": res["ok"],
            "rel_t": tr.t[rel] - tr.t[0], "err": err, "x_mm": tr.plate_x * 1e3, "depth": 1 - np.nanmin(real_n)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--test", nargs="+", required=True, help="held-out trial .npz files")
    ap.add_argument("--data", default=str(ROOT / "outputs" / "data" / "real"), help="where the training trials are")
    ap.add_argument("--out", default=None, help="default: next to fit.json")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    fit = json.load(open(a.fit))
    params = {k: float(v) for k, v in fit["best"].items()}
    base = PressParams(count=tuple(fit["grid"]), timestep=fit["timestep"]).to_dict()
    train = [str(Path(a.data) / f"{n}.npz") for n in fit["trials"]]
    jobs = [(p, params, base) for p in list(a.test) + train]
    with Pool(min(a.workers, len(jobs))) as pool:
        out = pool.map(run, jobs)
    test, tr_out = out[:len(a.test)], out[len(a.test):]

    od = Path(a.out) if a.out else Path(a.fit).parent
    fig, axs = plt.subplots(len(test), 1, figsize=(10, 2.6 * len(test)), sharex=True, squeeze=False)
    for i, (ax, r) in enumerate(zip(axs[:, 0], test), 1):
        ax.plot(r["t"], r["real_raw"], color="#2a7", lw=0.5, alpha=0.3)
        ax.plot(r["t"], r["real"], color="#2a7", lw=1.8, label="real (camera)")
        ax.plot(r["t"], r["sim"], color="#c53", lw=1.6, ls="--", label="MuJoCo, fitted model (prediction after release)")
        for k, tt in enumerate(r["rel_t"]):
            ax.axvline(tt, color="k", lw=0.8, ls=":", label="release" if k == 0 else None)
        side = "centre" if abs(r["x_mm"]) < 15 else ("left" if r["x_mm"] < 0 else "right")
        ax.set_title(f"Unseen press {i} ({side}): spring-back error {r['err']:.1f} % of the sponge height", fontsize=10)
        ax.set_ylabel("height / rest height")
        ax.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=8, loc="lower right")
    axs[-1, 0].set_xlabel("time [s]")
    fig.suptitle(f"Fitted sponge model (ν = {params['poisson']:.2f}, τ = {params['damping']:.2f} s) on presses it "
                 f"never saw: height of the pressed part, real vs simulated", fontsize=11)
    fig.tight_layout()
    fig.savefig(od / "heldout.png", dpi=120)
    def row(r):
        return {"springback_rmse_pct": r["err"], "x_mm": r["x_mm"], "ok": r["ok"]}
    summary = {"params": params, "fixed": fit.get("fixed"),
               "test": {r["name"]: row(r) for r in test}, "train": {r["name"]: row(r) for r in tr_out}}
    json.dump(summary, open(od / "heldout.json", "w"), indent=1)
    print(f"nu={params['poisson']:.3f} tau={params['damping']:.3f} s")
    for grp, rs in (("train", tr_out), ("TEST ", test)):
        for r in rs:
            print(f"  {grp} {r['name']}: spring-back error {r['err']:.1f}% of height  (press x={r['x_mm']:+.0f} mm)")
    print(f"figure -> {od / 'heldout.png'}")


if __name__ == "__main__":
    main()
