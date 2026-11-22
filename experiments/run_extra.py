"""Additional experiments promised in the project proposal.

E6  Number of users N  in {6, 9, 12, 15, 18}
E7  Server heterogeneity at equal total capacity (20):
        homogeneous (6.67, 6.67, 6.67), moderate (4, 6, 10), strong (2, 4, 14)

For every setting: Q-learning (selfish users, 3 seeds) and all baselines under
selfish users, plus the no-allocation delay under cooperative users so that an
empirical price of anarchy can be computed.

Usage:  python experiments/run_extra.py            (full run)
        python experiments/run_extra.py --quick    (smoke test)
Writes results/results_extra.json; plot with experiments/make_figures_extra.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from env import EnvConfig  # noqa: E402
from run_all import train_q, run_baselines, summary, SEEDS, RES  # noqa: E402

NS = [6, 9, 12, 15, 18]
CAPS = {"homogeneous": (20 / 3, 20 / 3, 20 / 3),
        "moderate": (4.0, 6.0, 10.0),
        "strong": (2.0, 4.0, 14.0)}


def agg(runs):
    raw = {k: np.concatenate([r["eval_raw"][k] for r in runs]) for k in ("return", "delay", "units")}
    return summary(raw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()
    N_EP = 60 if a.quick else 3000       # same budget as the other sweeps
    N_EVAL = 20 if a.quick else 200
    base = EnvConfig()
    t0 = time.time()

    settings = [("N", str(n), replace(base, N=n)) for n in NS] + \
               [("C", name, replace(base, C=c)) for name, c in CAPS.items()]

    with Pool(a.workers) as pool:
        q_jobs = [(cfg, N_EP, s, "visit", N_EVAL) for _, _, cfg in settings for s in SEEDS]
        bl_jobs = [(cfg, N_EVAL) for _, _, cfg in settings]
        coop_jobs = [(replace(cfg, user_policy="cooperative"), N_EVAL) for _, _, cfg in settings]
        q_res = pool.map_async(train_q, q_jobs)
        bl_res = pool.map_async(run_baselines, bl_jobs)
        co_res = pool.map_async(run_baselines, coop_jobs)
        q_res, bl_res, co_res = q_res.get(), bl_res.get(), co_res.get()

    out = {"settings": dict(N_EP=N_EP, N_EVAL=N_EVAL, seeds=SEEDS, Ns=NS,
                            capacities={k: list(v) for k, v in CAPS.items()}),
           "N_sweep": {}, "cap_sweep": {}}
    n = len(SEEDS)
    for i, (kind, key, cfg) in enumerate(settings):
        entry = {"Q-learning": agg(q_res[i * n:(i + 1) * n]), **bl_res[i],
                 "coop_no_alloc": co_res[i]["No allocation"],
                 "coop_threshold": co_res[i]["Threshold heuristic"],
                 "N": cfg.N}
        out["N_sweep" if kind == "N" else "cap_sweep"][key] = entry
    out["runtime_sec"] = time.time() - t0

    with open(os.path.join(RES, "results_extra.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"done in {out['runtime_sec']:.0f}s")


if __name__ == "__main__":
    main()
