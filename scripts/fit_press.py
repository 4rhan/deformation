"""Fit the sponge material to recorded presses (trial .npz files from rgbd_to_pcd.py) with CMA-ES.

Every candidate material is simulated through each training trial (indenter following the measured dent while
pressed, lifted away at each release) and scored with softfit.fit.trial_loss: point-cloud distance + spring-back
curve of the pressed part. E cannot be learned from video (no force is measured), so by default only poisson
and damping are fitted and E stays at its --init value.

usage:
  python scripts/fit_press.py --trials outputs/data/real/press_01.npz outputs/data/real/press_03.npz \
         outputs/data/real/press_04.npz --count 7 5 4 --maxiter 6 --popsize 8 --init 2e4 0.15 0.9 --out outputs/fit
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cma  # noqa: E402

from softfit.fit import NAMES, evaluate, params_to_u, u_to_params  # noqa: E402
from softfit.press_scene import PressParams  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--trials", nargs="+", required=True, help="training trial .npz files")
ap.add_argument("--out", default=str(ROOT / "outputs" / "fit"), help="output folder for fit.json")
ap.add_argument("--count", type=int, nargs=3, default=[7, 5, 4], help="FEM grid vertices x y z (finer = slower)")
ap.add_argument("--timestep", type=float, default=1e-3)
ap.add_argument("--init", type=float, nargs=3, default=[2e4, 0.2, 0.02], help="initial E nu tau")
ap.add_argument("--sigma0", type=float, default=0.2, help="initial CMA step in normalised units")
ap.add_argument("--popsize", type=int, default=8)
ap.add_argument("--maxiter", type=int, default=20, help="generations; ALL are run (no early stop)")
ap.add_argument("--workers", type=int, default=os.cpu_count())
ap.add_argument("--stride", type=int, default=2)
ap.add_argument("--free", nargs="+", choices=NAMES, default=["poisson", "damping"],
                help="parameters to fit; the rest stay at --init (E is not learnable from video, so it is fixed)")
a = ap.parse_args()

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
trials = sorted(str(Path(t)) for t in a.trials)
base = PressParams(count=tuple(a.count), timestep=a.timestep).to_dict()
free = a.free
fixed = {n: v for n, v in zip(NAMES, a.init) if n not in free}
free_idx = [NAMES.index(n) for n in free]
u_init = params_to_u(dict(zip(NAMES, a.init)))


def full_u(uf):
    u = u_init.copy()
    u[free_idx] = uf
    return u


print(f"fitting on {len(trials)} trials: {[Path(t).stem for t in trials]}  grid={a.count}  "
      f"free={free}  fixed={fixed or '-'}")

es = cma.CMAEvolutionStrategy(u_init[free_idx], a.sigma0,
                              {"bounds": [0, 1], "popsize": a.popsize, "seed": 1, "verbose": -9})
history = []   # EVERY candidate: generation, params, total + per-trial losses
gens = []      # per-generation summary
t0 = time.time()


def dump():
    best = u_to_params(full_u(es.result.xbest))
    result = {"best": best, "free": free, "fixed": fixed, "loss": es.result.fbest, "grid": a.count, "timestep": a.timestep,
              "trials": [Path(t).stem for t in trials], "cma_std_normalised": list(map(float, es.result.stds)),
              "generations": gens, "history": history}
    json.dump(result, open(out / "fit.json", "w"), indent=1)
    return best


with Pool(a.workers) as pool:
    for gen in range(1, a.maxiter + 1):   # fixed budget: run every generation
        U = [full_u(uf) for uf in es.ask()]
        jobs = [(tp, u_to_params(u), base, a.stride) for u in U for tp in trials]
        res = pool.map(evaluate, jobs, chunksize=1)
        f = []
        for i, u in enumerate(U):
            per = res[i * len(trials):(i + 1) * len(trials)]
            fi = float(sum(r["total"] for r in per))
            f.append(fi)
            history.append({"gen": gen, **u_to_params(u), "loss": fi, "ok": all(r["ok"] for r in per),
                            "per_trial": {Path(tp).stem: r for tp, r in zip(trials, per)}})
        es.tell([u[free_idx] for u in U], f)
        b = dump()
        n_fail = sum(not h["ok"] for h in history if h["gen"] == gen)
        gens.append({"gen": gen, "best_loss": float(es.result.fbest), "gen_min": float(min(f)), "n_failed": n_fail,
                     **b, "elapsed_s": time.time() - t0})
        dump()
        print(f"gen {gen:2d}  best {es.result.fbest:9.3f}  gen-min {min(f):9.3f}  fails {n_fail}  "
              + "  ".join(f"{k}={b[k]:.4g}" for k in free) + f"  ({time.time() - t0:.0f}s)", flush=True)

best = dump()
print("best:", {k: float(f"{best[k]:.4g}") for k in free})
print("wrote", out / "fit.json")
