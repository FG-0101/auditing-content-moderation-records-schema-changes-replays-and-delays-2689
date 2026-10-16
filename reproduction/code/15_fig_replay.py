"""Figure for the replay detection section (vector PDF).
Run from the package root after code/14_duplicates.py:  python3 code/15_fig_replay.py
Output: figures/fig_replay.pdf"""
from paper_style import save_figure
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
s = json.load(open("results/dup.json"))
days = pd.read_csv("results/dup_days.csv", parse_dates=["d"]).sort_values("d")
null = pd.read_csv("results/dup_null.csv", parse_dates=["d"])
nmax = null.groupby("d").rho.max().reset_index().sort_values("d")

fig, ax = plt.subplots(1, 2, figsize=(7.0, 1.9))
a = ax[0]
a.plot(days.d, days.ratio, color="black", lw=0.7)
k = days[days.known]
a.scatter(k.d, k.ratio, s=9, facecolor="white", edgecolor="black", lw=0.7, zorder=3)
a.set_ylim(0.9, 2.1); a.set_ylabel("Simple over complete records")
a.set_title("Test D, reference product", fontsize=8)
a.xaxis.set_major_locator(mdates.YearLocator()); a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

b = ax[1]
b.plot(nmax.d, nmax.rho, color="black", lw=0.5, ls="--", label="Thinning null, maximum of %d draws" % (s["null_draws"] // s["days"]))
b.plot(days.d, days.rho, color="black", lw=0.7, label="Observed")
b.scatter(k.d, k.rho, s=9, facecolor="white", edgecolor="black", lw=0.7, zorder=3, label="Doubled days")
b.axhline(s["thresholds"]["tau_C"], color="black", lw=0.5, ls=":")
b.set_ylim(-0.03, 1.05); b.set_ylabel("Keys repeated with equal counts")
b.set_title("Test C, no reference and no provenance", fontsize=8)
b.xaxis.set_major_locator(mdates.YearLocator()); b.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
b.legend(frameon=False, fontsize=6.5, loc="upper left")
fig.tight_layout(); save_figure(fig, "figures/fig_replay.pdf", metadata={"Creator": None, "Producer": None, "CreationDate": None, "ModDate": None}); plt.close(fig)
print("figures/fig_replay.pdf written")
