"""Figures and table for the extra experiments (results/results_extra.json)."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)
R = json.load(open(os.path.join(RES, "results_extra.json")))

PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3de"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "pdf.fonttype": 42, "lines.linewidth": 2, "savefig.bbox": "tight", "savefig.dpi": 200,
})
# same colours as make_figures.py
MGR = {"No allocation": ("No alloc.", PAL[0]), "Uniform (full budget)": ("Uniform", PAL[1]),
       "Threshold heuristic": ("Threshold", PAL[2]), "Q-learning": ("Q-learning", PAL[4]),
       "Myopic oracle (exact loads)": ("Myopic oracle", PAL[5])}
CAP_LBL = {"homogeneous": "Homogeneous\n(6.7, 6.7, 6.7)", "moderate": "Moderate\n(4, 6, 10)",
           "strong": "Strong\n(2, 4, 14)"}


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"))
    plt.close(fig)


Ns = sorted(R["N_sweep"], key=int)
caps = list(R["cap_sweep"])

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1, 1.15]})
ax = axes[0]
for m, (lbl, col) in MGR.items():
    y = [-R["N_sweep"][n][m]["return"][0] for n in Ns]
    ax.plot([int(n) for n in Ns], y, color=col, marker="o", ms=4, label=lbl)
ax.set_yscale("log")
ax.set_xticks([int(n) for n in Ns])
ax.set_xlabel("Number of users $N$")
ax.set_ylabel("Discounted cost (log scale)")
ax.set_title("(a) Number of users", fontsize=9, color=INK)
ax.legend(fontsize=7)

ax = axes[1]
w = 0.16
for k, (m, (lbl, col)) in enumerate(MGR.items()):
    y = [-R["cap_sweep"][c][m]["return"][0] for c in caps]
    e = [R["cap_sweep"][c][m]["return"][1] for c in caps]
    ax.bar(np.arange(len(caps)) + (k - 2) * w, y, w * 0.92, yerr=e, color=col,
           error_kw=dict(ecolor=MUTED, lw=1, capsize=1.5))
ax.set_xticks(range(len(caps)))
ax.set_xticklabels([CAP_LBL[c] for c in caps], fontsize=7)
ax.set_ylabel("Discounted cost")
ax.set_title("(b) Server heterogeneity (total capacity 20)", fontsize=9, color=INK)
ax.set_ylim(bottom=0)
ax.grid(axis="x", visible=False)
save(fig, "scale_heterogeneity")

# ---------------------------------------------------------------- LaTeX table
rows = []
for kind, keys, lab in (("N_sweep", Ns, lambda k: f"$N={k}$"),
                        ("cap_sweep", caps, lambda k: k.capitalize())):
    for k in keys:
        e = R[kind][k]
        n = e["N"]
        poa = e["No allocation"]["delay"][0] / e["coop_no_alloc"]["delay"][0]
        q, th = e["Q-learning"]["return"][0], e["Threshold heuristic"]["return"][0]
        rows.append(f"{lab(k)} & {e['No allocation']['delay'][0] / n:.2f} & "
                    f"{e['Q-learning']['delay'][0] / n:.3f} & {poa:.2f} & "
                    f"{q:.1f} & {th:.1f} & {100 * (q - th) / abs(th):+.1f}\\%\\\\")
    rows.append(r"\midrule")
rows[-1] = r"\bottomrule"
open(os.path.join(RES, "table_extra.tex"), "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
