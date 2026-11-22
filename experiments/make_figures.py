from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)
R = json.load(open(os.path.join(RES, "results.json")))

PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3de"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "pdf.fonttype": 42, "lines.linewidth": 2, "savefig.bbox": "tight", "savefig.dpi": 200,
})
POL = ["random", "selfish", "cooperative"]
POL_LBL = {"random": "Random", "selfish": "Selfish (Nash)", "cooperative": "Cooperative"}
MGR = ["No allocation", "Uniform (full budget)", "Threshold heuristic",
       "DP (value iteration)", "Q-learning", "Myopic oracle (exact loads)"]
MGR_SHORT = {"No allocation": "No alloc.", "Uniform (full budget)": "Uniform",
             "Threshold heuristic": "Threshold", "DP (value iteration)": "DP",
             "Q-learning": "Q-learning", "Myopic oracle (exact loads)": "Myopic oracle"}


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"))
    plt.close(fig)


def smooth(x, w=200):
    x = np.asarray(x)
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[w:] - c[:-w]) / w


fig, ax = plt.subplots(figsize=(5.2, 2.9))
for i, pol in enumerate(POL):
    W_ = min(200, len(R["curves"][pol][0]) // 5)
    curves = np.array([smooth(c, W_) for c in R["curves"][pol]])
    m, s = curves.mean(0), curves.std(0)
    x = np.arange(len(m)) + W_
    ax.plot(x, m, color=PAL[i], label=POL_LBL[pol])
    ax.fill_between(x, m - s, m + s, color=PAL[i], alpha=0.15, linewidth=0)
ax.set_xlabel("Training episode")
ax.set_ylabel("Discounted return (200-ep. avg)")
ax.legend(loc="lower right")
ax.set_xlim(0, len(m) + W_)
save(fig, "learning_curves")

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
for k, pol in enumerate(POL):
    ax = axes[k]
    tab = R["main"][pol]
    vals = [-tab[m]["return"][0] for m in MGR]
    errs = [tab[m]["return"][1] for m in MGR]
    bars = ax.bar(range(len(MGR)), vals, yerr=errs, color=PAL, width=0.8,
                  edgecolor="white", linewidth=1, error_kw=dict(ecolor=MUTED, lw=1, capsize=2))
    for b, v, er in zip(bars, vals, errs):
        ax.text(b.get_x() + b.get_width() / 2, v + er + max(vals) * 0.02, f"{v:.0f}",
                ha="center", fontsize=6.5, color=INK)
    ax.set_xticks(range(len(MGR)))
    ax.set_xticklabels([MGR_SHORT[m] for m in MGR], rotation=55, ha="right", fontsize=7)
    ax.set_title(POL_LBL[pol] + " users", fontsize=9, color=INK)
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Discounted cost  (−return, lower is better)")
save(fig, "comparison")

lams = sorted(R["lambda_sweep"], key=float)
lv = [float(l) for l in lams]
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
for ax, key, lbl in ((axes[0], "delay", "Avg. total delay / step"),
                     (axes[1], "units", "Avg. resource units / step")):
    m = [R["lambda_sweep"][l][key][0] for l in lams]
    e = [R["lambda_sweep"][l][key][1] for l in lams]
    ax.errorbar(range(len(lv)), m, yerr=e, color=PAL[0], marker="o", ms=5, capsize=2)
    ax.set_xticks(range(len(lv)))
    ax.set_xticklabels([f"{v:g}" for v in lv])
    ax.set_xlabel(r"Resource cost $\lambda$")
    ax.set_ylabel(lbl)
save(fig, "lambda_sweep")

Bs = sorted(R["budget_sweep"], key=int)
fig, ax = plt.subplots(figsize=(5.2, 2.8))
order = ["No allocation", "Uniform (full budget)", "Threshold heuristic", "Q-learning",
         "Myopic oracle (exact loads)"]
cidx = {m: MGR.index(m) for m in order}
for m in order:
    y = [-R["budget_sweep"][b][m]["return"][0] for b in Bs]
    ax.plot([int(b) for b in Bs], y, color=PAL[cidx[m]], marker="o", ms=4, label=MGR_SHORT[m])
ax.set_yscale("log")
ax.set_xlabel("Budget $B$ (extra units per step)")
ax.set_ylabel("Discounted cost (log scale)")
ax.legend(fontsize=7, ncol=2)
save(fig, "budget_sweep")

fig, ax = plt.subplots(figsize=(5.2, 2.6))
mgrs = ["No allocation", "Threshold heuristic", "Q-learning"]
w = 0.26
for i, pol in enumerate(POL):
    y = [R["main"][pol][m]["delay"][0] for m in mgrs]
    e = [R["main"][pol][m]["delay"][1] for m in mgrs]
    xs = np.arange(len(mgrs)) + (i - 1) * w
    bars = ax.bar(xs, y, w * 0.92, yerr=e, color=PAL[i], label=POL_LBL[pol],
                  error_kw=dict(ecolor=MUTED, lw=1, capsize=2))
    for b, v, er in zip(bars, y, e):
        ax.text(b.get_x() + b.get_width() / 2, v + er + 0.8, f"{v:.1f}", ha="center", fontsize=6.5, color=INK)
ax.set_xticks(range(len(mgrs)))
ax.set_xticklabels([MGR_SHORT[m] for m in mgrs])
ax.set_ylabel("Avg. total user delay / step")
ax.legend(fontsize=7)
ax.grid(axis="x", visible=False)
save(fig, "user_policies")

al = list(R["alpha_sweep"])
fig, ax = plt.subplots(figsize=(3.4, 2.4))
y = [-R["alpha_sweep"][a]["return"][0] for a in al]
e = [R["alpha_sweep"][a]["return"][1] for a in al]
ax.errorbar(range(len(al)), y, yerr=e, color=PAL[0], marker="o", ms=5, capsize=2, linestyle="none")
ax.axvline(2.5, color=GRID, lw=1)
ax.set_xticks(range(len(al)))
ax.set_xticklabels([a.replace("const ", "") if a.startswith("const") else "$n^{-%s}$" % a.split("^")[1] for a in al], fontsize=7)
ax.set_xlabel(r"Learning rate $\alpha$ (constant, or visit-decayed $n^{-\omega}$)")
ax.set_ylabel("Discounted cost")
save(fig, "alpha_sweep")

lines = []
for pol in POL:
    tab = R["main"][pol]
    lines.append(r"\textit{" + POL_LBL[pol] + r" users} & & &\\")
    best = max((m for m in MGR if m != "Myopic oracle (exact loads)"),
               key=lambda m: tab[m]["return"][0])
    for m in MGR:
        r_, d_, u_ = tab[m]["return"], tab[m]["delay"], tab[m]["units"]
        ret = f"{r_[0]:.1f} $\\pm$ {r_[1]:.1f}"
        if m == best:
            ret = r"\textbf{" + ret + "}"
        lines.append(f"\\quad {MGR_SHORT[m]} & {ret} & {d_[0]:.2f} & {u_[0]:.2f}\\\\")
    lines.append(r"\midrule")
lines[-1] = r"\bottomrule"
open(os.path.join(RES, "table_main.tex"), "w").write("\n".join(lines) + "\n")

pt = []
for row in R["policy_table"][:8]:
    pt.append(f"({','.join(map(str, row['state']))}) & {row['visits']} & "
              f"({','.join(map(str, row['q_action']))}) & ({','.join(map(str, row['dp_action']))})\\\\")
pt.append(r"\bottomrule")
open(os.path.join(RES, "table_policy.tex"), "w").write("\n".join(pt) + "\n")

txt = ["Mean discounted return +/- 95% CI over 200 evaluation episodes",
       "(Q-learning and DP: three trained policies averaged per episode)", ""]
for pol in POL:
    txt.append(f"[{POL_LBL[pol]} users]")
    txt.append(f"  {'manager':<16}{'return':>18}{'delay/step':>12}{'units/step':>12}")
    for m in MGR:
        t = R["main"][pol][m]
        txt.append(f"  {MGR_SHORT[m]:<16}{t['return'][0]:>10.1f} +/- {t['return'][1]:<5.1f}"
                   f"{t['delay'][0]:>11.2f}{t['units'][0]:>12.2f}")
    txt.append("")
open(os.path.join(RES, "summary.txt"), "w").write("\n".join(txt))
print("\n".join(txt))

fig = plt.figure(figsize=(7.2, 0.17 * len(txt) + 0.3))
fig.patch.set_facecolor("#1f1f1e")
fig.text(0.02, 0.98, "$ python experiments/make_figures.py\n" + "\n".join(txt), family="monospace",
         fontsize=8, color="#f0efec", va="top")
fig.savefig(os.path.join(ROOT, "screenshots", "results_summary.png"), dpi=170, facecolor=fig.get_facecolor())
plt.close(fig)


EXTRA = os.path.join(RES, "results_extra.json")
if os.path.exists(EXTRA):
    X = json.load(open(EXTRA))
    XMGR = {"No allocation": ("No alloc.", PAL[0]), "Uniform (full budget)": ("Uniform", PAL[1]),
            "Threshold heuristic": ("Threshold", PAL[2]), "Q-learning": ("Q-learning", PAL[4]),
            "Myopic oracle (exact loads)": ("Myopic oracle", PAL[5])}
    CAP_LBL = {"homogeneous": "Homogeneous\n(6.7, 6.7, 6.7)", "moderate": "Moderate\n(4, 6, 10)",
               "strong": "Strong\n(2, 4, 14)"}

    Ns = sorted(X["N_sweep"], key=int)
    caps = list(X["cap_sweep"])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    for m, (lbl, col) in XMGR.items():
        y = [-X["N_sweep"][n][m]["return"][0] for n in Ns]
        ax.plot([int(n) for n in Ns], y, color=col, marker="o", ms=4, label=lbl)
    ax.set_yscale("log")
    ax.set_xticks([int(n) for n in Ns])
    ax.set_xlabel("Number of users $N$")
    ax.set_ylabel("Discounted cost (log scale)")
    ax.set_title("(a) Number of users", fontsize=9, color=INK)
    ax.legend(fontsize=7)

    ax = axes[1]
    w = 0.16
    for k, (m, (lbl, col)) in enumerate(XMGR.items()):
        y = [-X["cap_sweep"][c][m]["return"][0] for c in caps]
        e = [X["cap_sweep"][c][m]["return"][1] for c in caps]
        ax.bar(np.arange(len(caps)) + (k - 2) * w, y, w * 0.92, yerr=e, color=col,
               error_kw=dict(ecolor=MUTED, lw=1, capsize=1.5))
    ax.set_xticks(range(len(caps)))
    ax.set_xticklabels([CAP_LBL[c] for c in caps], fontsize=7)
    ax.set_ylabel("Discounted cost")
    ax.set_title("(b) Server heterogeneity (total capacity 20)", fontsize=9, color=INK)
    ax.set_ylim(bottom=0)
    ax.grid(axis="x", visible=False)
    save(fig, "scale_heterogeneity")

    rows = []
    for kind, keys, lab in (("N_sweep", Ns, lambda k: f"$N={k}$"),
                            ("cap_sweep", caps, lambda k: k.capitalize())):
        for k in keys:
            e = X[kind][k]
            n = e["N"]
            poa = e["No allocation"]["delay"][0] / e["coop_no_alloc"]["delay"][0]
            q, th = e["Q-learning"]["return"][0], e["Threshold heuristic"]["return"][0]
            rows.append(f"{lab(k)} & {e['No allocation']['delay'][0] / n:.2f} & "
                        f"{e['Q-learning']['delay'][0] / n:.3f} & {poa:.2f} & "
                        f"{q:.1f} & {th:.1f} & {100 * (q - th) / abs(th):+.1f}\\%\\\\")
        rows.append(r"\midrule")
    rows[-1] = r"\bottomrule"
    open(os.path.join(RES, "table_extra.tex"), "w").write("\n".join(rows) + "\n")
