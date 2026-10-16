"""Robustness for the enforcement design and its figure.
Run from the package root after code/12_enforcement.py:  python3 code/13_enforcement_robust.py
Outputs: results/enforcement_robust.csv, figures/fig_enforcement.pdf"""
from paper_style import save_figure
import numpy as np
import pandas as pd
import pyfixest as pf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

x = pd.read_parquet("data/derived/enforcement_panel.parquet")
TREAT = sorted(x[x.rel != -1000].p.unique())
rows = []
def fit(name, d, y):
    m = pf.feols(f"{y} ~ treat_post | p + week", data=d.dropna(subset=[y]), vcov={"CRV1": "p"})
    r = m.tidy().loc["treat_post"]
    rows.append({"spec": name, "outcome": y, "est": r["Estimate"], "se": r["Std. Error"], "p": r["Pr(>|t|)"]})
for y in ["y_catchall", "y_human"]:
    fit("Baseline", x, y)
    for t in TREAT:
        fit(f"Without {t}", x[x.p != t], y)
    fit("Without Google Shopping", x[x.p != "Google Shopping"], y)
    # Placebo: treatment dates moved 26 weeks earlier, sample restricted to true pre periods
    d = x[(x.rel < 0)].copy()
    d["treat_post"] = ((d.rel != -1000) & (d.rel >= -26)).astype(int)
    fit("Placebo, dates 26 weeks earlier, pre period only", d, y)
    # Record weighted version
    m = pf.feols(f"{y} ~ treat_post | p + week", data=x.dropna(subset=[y]), weights="n", vcov={"CRV1": "p"})
    r = m.tidy().loc["treat_post"]
    rows.append({"spec": "Weighted by records", "outcome": y, "est": r["Estimate"], "se": r["Std. Error"], "p": r["Pr(>|t|)"]})
rb = pd.DataFrame(rows)
rb.to_csv("results/enforcement_robust.csv", index=False)
print(rb.round(4).to_string())

plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Liberation Serif", "DejaVu Serif"], "font.size": 8,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
e = pd.read_csv("results/enforcement_event.csv")
fig, ax = plt.subplots(1, 2, figsize=(6.66, 2.0))
for a, y, l in [(ax[0], "y_catchall", "Catch all category share"), (ax[1], "y_human", "Share decided without automation")]:
    c = e[e.outcome == y].sort_values("k")
    c = pd.concat([c, pd.DataFrame({"outcome": [y], "k": [-1], "est": [0.0], "se": [0.0]})]).sort_values("k")
    a.errorbar(c.k, c.est, yerr=1.96 * c.se, fmt="o", ms=2.5, color="black", elinewidth=0.5, capsize=1)
    a.axhline(0, color="black", lw=0.5); a.axvline(-0.5, color="black", lw=0.5, ls=":")
    a.set_xlabel("Weeks relative to opening of formal proceedings"); a.set_ylabel(l)
fig.tight_layout(); save_figure(fig, "figures/fig_enforcement.pdf", metadata={"Creator": None, "Producer": None, "CreationDate": None, "ModDate": None}, bbox_inches="tight", pad_inches=0.06); plt.close(fig)
