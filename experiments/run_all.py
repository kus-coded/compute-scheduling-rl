from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from env import CongestionEnv, EnvConfig, index_state
from agents import (QLearningAgent, DPAgent, NoAlloc, Uniform, Threshold,
                    MyopicOracle, evaluate)

RES = os.path.join(ROOT, "results")
POLICIES = ["random", "selfish", "cooperative"]
SEEDS = [0, 1, 2]


def ci95(x):
    x = np.asarray(x)
    return float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


def summary(m):
    return {k: [float(v.mean()), ci95(v)] for k, v in m.items()}


def pooled(runs):
    raw = {k: np.mean([r["eval_raw"][k] for r in runs], axis=0) for k in ("return", "delay", "units")}
    return summary(raw)


def train_q(args):
    cfg, n_ep, seed, alpha, n_eval = args[:5]
    omega = args[5] if len(args) > 5 else 0.6
    env = CongestionEnv(cfg)
    ag = QLearningAgent(env.nS, env.nA, alpha=alpha, gamma=cfg.gamma, seed=seed, omega=omega)
    curve = ag.train(env, n_ep, seed_offset=seed * 1_000_000)
    ev = evaluate(env, ag, n_eval)
    return {"curve": curve.tolist(), "eval": summary(ev), "eval_raw": ev,
            "Q": ag.Q, "visits": ag.visits}


def run_dp(args):
    cfg, n_ep, seed, n_eval = args
    env = CongestionEnv(cfg)
    dp = DPAgent(env.nS, env.nA, cfg.gamma).fit_model(env, n_ep, seed_offset=500_000 + seed * 1_000_000,
                                                     seed=seed).solve()
    ev = evaluate(env, dp, n_eval)
    return {"eval": summary(ev), "eval_raw": ev, "pi": dp.pi.tolist(), "iters": dp.iters,
            "coverage": float((dp.n_sa > 0).mean())}


def run_baselines(args):
    cfg, n_eval = args
    env = CongestionEnv(cfg)
    out = {}
    for ag in (NoAlloc(env), Uniform(env), Threshold(env), MyopicOracle()):
        out[ag.name] = summary(evaluate(env, ag, n_eval))
    return out


def run_main(a):
    N_EP = 100 if a.quick else 10000
    N_EP_SWEEP = 60 if a.quick else 3000
    N_DP = 50 if a.quick else 1500
    N_EVAL = 20 if a.quick else 200
    os.makedirs(os.path.join(RES, "figures"), exist_ok=True)
    base = EnvConfig()
    t0 = time.time()
    results = {"config": base.__dict__ | {"K": base.K, "nS": 3 ** base.K,
                                          "nA": CongestionEnv(base).nA},
               "settings": dict(N_EP=N_EP, N_EP_SWEEP=N_EP_SWEEP, N_DP=N_DP, N_EVAL=N_EVAL,
                                seeds=SEEDS)}

    with Pool(a.workers) as pool:
        q_jobs = [(replace(base, user_policy=pol), N_EP, s, "visit", N_EVAL) for pol in POLICIES for s in SEEDS]
        dp_jobs = [(replace(base, user_policy=pol), N_DP, s, N_EVAL) for pol in POLICIES for s in SEEDS]
        bl_jobs = [(replace(base, user_policy=pol), N_EVAL) for pol in POLICIES]
        q_res = pool.map_async(train_q, q_jobs)
        dp_res = pool.map_async(run_dp, dp_jobs)
        bl_res = pool.map_async(run_baselines, bl_jobs)

        lams = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
        lam_jobs = [(replace(base, lam=l), N_EP_SWEEP, s, "visit", N_EVAL) for l in lams for s in SEEDS]
        lam_res = pool.map_async(train_q, lam_jobs)
        Bs = [0, 1, 2, 3, 4, 6]
        B_jobs = [(replace(base, B=b), N_EP_SWEEP, s, "visit", N_EVAL) for b in Bs for s in SEEDS]
        B_res = pool.map_async(train_q, B_jobs)
        B_bl = pool.map_async(run_baselines, [(replace(base, B=b), N_EVAL) for b in Bs])
        alpha_cfgs = [(0.05, 0.6), (0.1, 0.6), (0.3, 0.6), ("visit", 0.5), ("visit", 0.6), ("visit", 0.8)]
        alphas = [f"const {al}" if al != "visit" else f"1/n^{om}" for al, om in alpha_cfgs]
        a_jobs = [(base, N_EP_SWEEP, s, al, N_EVAL, om) for al, om in alpha_cfgs for s in SEEDS]
        a_res = pool.map_async(train_q, a_jobs)

        q_res, dp_res, bl_res = q_res.get(), dp_res.get(), bl_res.get()
        lam_res, B_res, B_bl, a_res = lam_res.get(), B_res.get(), B_bl.get(), a_res.get()

    def by(lst, keys, n=len(SEEDS)):
        return {k: lst[i * n:(i + 1) * n] for i, k in enumerate(keys)}

    qp, dpp = by(q_res, POLICIES), by(dp_res, POLICIES)
    main_tab = {}
    for i, pol in enumerate(POLICIES):
        tab = dict(bl_res[i])
        for name, runs in (("Q-learning", qp[pol]), ("DP (value iteration)", dpp[pol])):
            tab[name] = pooled(runs)
            tab[name]["per_seed_return"] = [r["eval"]["return"][0] for r in runs]
        main_tab[pol] = tab
    results["main"] = main_tab
    results["dp_info"] = {pol: [{"iters": r["iters"], "coverage": r["coverage"]} for r in dpp[pol]]
                          for pol in POLICIES}
    results["curves"] = {pol: [r["curve"] for r in qp[pol]] for pol in POLICIES}

    results["lambda_sweep"] = {str(l): pooled(r) for l, r in by(lam_res, lams).items()}
    results["budget_sweep"] = {str(b): {"Q-learning": pooled(r), **B_bl[i]}
                               for i, (b, r) in enumerate(by(B_res, Bs).items())}
    results["alpha_sweep"] = {str(al): pooled(r) | {"final_train_return":
                              float(np.mean([np.mean(x["curve"][-200:]) for x in r]))}
                              for al, r in by(a_res, alphas).items()}

    env = CongestionEnv(base)
    Q, visits = qp["selfish"][0]["Q"], qp["selfish"][0]["visits"]
    sv = visits.sum(axis=1)
    top = np.argsort(-sv)[:10]
    results["policy_table"] = [{"state": index_state(s, env.K).tolist(),
                                "visits": int(sv[s]),
                                "q_action": env.actions[int(Q[s].argmax())].tolist(),
                                "dp_action": env.actions[dpp["selfish"][0]["pi"][s]].tolist()}
                               for s in top]
    results["runtime_sec"] = time.time() - t0

    with open(os.path.join(RES, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(f"done in {results['runtime_sec']:.0f}s")


NS = [6, 9, 12, 15, 18]
CAPS = {"homogeneous": (20 / 3, 20 / 3, 20 / 3),
        "moderate": (4.0, 6.0, 10.0),
        "strong": (2.0, 4.0, 14.0)}


def run_extra(a):
    N_EP = 60 if a.quick else 3000
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
        entry = {"Q-learning": pooled(q_res[i * n:(i + 1) * n]), **bl_res[i],
                 "coop_no_alloc": co_res[i]["No allocation"],
                 "coop_threshold": co_res[i]["Threshold heuristic"],
                 "N": cfg.N}
        out["N_sweep" if kind == "N" else "cap_sweep"][key] = entry
    out["runtime_sec"] = time.time() - t0
    with open(os.path.join(RES, "results_extra.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"done in {out['runtime_sec']:.0f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--extra", action="store_true")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()
    if a.quick:
        global RES
        RES = tempfile.mkdtemp(prefix="quick_")
        print("quick run: results written to", RES)
    if a.extra:
        run_extra(a)
    else:
        run_main(a)


if __name__ == "__main__":
    main()
