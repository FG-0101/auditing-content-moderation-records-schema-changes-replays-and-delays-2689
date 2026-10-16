"""Figures for the paper (vector PDF). Run from the package root:  python3 code/06_figures.py"""
from paper_style import save_figure
import json
import numpy as np
import pandas as pd
import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pyfixest as pf
import sys
sys.path.insert(0, 'code')
from common import sql_excl

plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
W = 3.33
con = duckdb.connect()

# Figure 2: monthly volume and number of reporting platforms
pd_ = con.sql("SELECT * FROM 'data/derived/panel_platform_day.parquet' WHERE d <= DATE '2026-09-30'" + sql_excl()).df()
pd_["m"] = pd.to_datetime(pd_.d).dt.to_period("M").dt.to_timestamp()
mm = pd_.groupby("m").agg(n=("n", "sum"), plats=("p", "nunique")).reset_index()
gs = pd_[pd_.p == "Google Shopping"].groupby("m").n.sum()
mm["other"] = mm.n - mm.m.map(gs).fillna(0)
fig, ax = plt.subplots(1, 2, figsize=(W * 2, 1.9))
ax[0].plot(mm.m, mm.n / 1e9, color="black", lw=1.1, label="All platforms")
ax[0].plot(mm.m, mm.other / 1e9, color="black", lw=1.1, ls="--", label="Excluding the largest platform")
ax[0].set_ylabel("Records per month (billions)"); ax[0].legend(frameon=False, loc="upper right")
ax[0].axvline(pd.Timestamp("2025-07-01"), color="black", lw=0.5, ls=":")
ax[1].plot(mm.m, mm.plats, color="black", lw=1.1)
ax[1].set_ylabel("Platforms submitting in month")
ax[1].axvline(pd.Timestamp("2025-07-01"), color="black", lw=0.5, ls=":")
for a in ax:
    a.xaxis.set_major_locator(mdates.YearLocator())
    a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.tight_layout(); save_figure(fig, "figures/fig_volume.pdf", metadata={"Creator": None, "Producer": None, "CreationDate": None, "ModDate": None}, bbox_inches="tight", pad_inches=0.06); plt.close(fig)

# Figure 3: schema change, sexualized content and the catch all category
ser = pd.read_csv("results/schema_daily_series.csv", parse_dates=["d"])
lab = {"STATEMENT_CATEGORY_PORNOGRAPHY_OR_SEXUALIZED_CONTENT": ("Pornography or sexualized content (retired)", "-"),
       "STATEMENT_CATEGORY_SCOPE_OF_PLATFORM_SERVICE": ("Scope of platform service (retired)", ":"),
       "STATEMENT_CATEGORY_OTHER_VIOLATION_TC": ("Other violation of terms (new)", "--"),
       "STATEMENT_CATEGORY_PROTECTION_OF_MINORS": ("Protection of minors (kept)", "-.")}
dst = pd.read_csv("results/schema_sexualized_destinations.csv").head(8)
fig, ax = plt.subplots(1, 2, figsize=(W * 2, 2.3), gridspec_kw={"width_ratios": [1.25, 1]})
for c, (l, ls) in lab.items():
    x = ser[ser.c == c].sort_values("d")
    ax[0].plot(x.d, x.n, color="black", lw=0.9, ls=ls, label=l)
ax[0].set_yscale("log"); ax[0].set_ylabel("Records per day, log scale")
ax[0].axvline(pd.Timestamp("2025-07-01"), color="black", lw=0.5)
ax[0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[4, 6, 8, 10]))
ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax[0].legend(frameon=False, fontsize=6.0, loc="upper center", bbox_to_anchor=(0.5, 1.32), ncol=2)
y = np.arange(len(dst))
ax[1].barh(y + 0.2, dst.sex_share_pre, height=0.38, color="black", label="Sexualized share, 60 days before")
ax[1].barh(y - 0.2, dst.gain_to_catch_all, height=0.38, color="white", edgecolor="black", lw=0.6,
           label="Gain of new catch all, 60 days after")
ax[1].set_yticks(y); ax[1].set_yticklabels(dst.p); ax[1].invert_yaxis()
ax[1].set_xlabel("Share of the platform's records")
ax[1].legend(frameon=False, fontsize=6.0, loc="upper center", bbox_to_anchor=(0.45, 1.32), ncol=1)
fig.tight_layout(); save_figure(fig, "figures/fig_schema.pdf", metadata={"Creator": None, "Producer": None, "CreationDate": None, "ModDate": None}, bbox_inches="tight", pad_inches=0.06); plt.close(fig)

# Figure 4: capacity signature, residualized binned scatter and event profile
s = pd.read_parquet("data/derived/capacity_sample.parquet")
def resid(var):
    m = pf.feols(f"{var} ~ 1 | pdow + month", data=s)
    return m.resid()
rx, rh, ra = resid("log_nn"), resid("log_h"), resid("log_auto")
b = pd.DataFrame({"x": rx, "h": rh, "a": ra})
b["bin"] = pd.qcut(b.x, 20, labels=False)
bb = b.groupby("bin").mean()
reg = pd.read_csv("results/capacity_regressions.csv").set_index("model")
bh = reg.loc["M3 log non automated on log inflow", "estimate"]
ba = reg.loc["M4 log fully automated on log inflow", "estimate"]
ev = pd.read_csv("results/capacity_event.csv")
fig, ax = plt.subplots(1, 2, figsize=(W * 2, 2.0))
ax[0].scatter(bb.x, bb.h, s=10, color="black", label=f"Not automated, slope {bh:.2f}")
ax[0].scatter(bb.x, bb.a, s=10, facecolor="white", edgecolor="black", label=f"Fully automated, slope {ba:.2f}")
xs = np.linspace(bb.x.min(), bb.x.max(), 10)
ax[0].plot(xs, xs, color="black", lw=0.6, ls=":", label="Proportional scaling, slope 1")
ax[0].set_xlabel("Log reported notice-sourced decisions (residual)"); ax[0].set_ylabel("Log decisions by route (residual)")
ax[0].legend(frameon=False, fontsize=6.3)
ax[1].errorbar(ev.k, ev.dev_mean, yerr=[ev.dev_mean - ev.boot_lo, ev.boot_hi - ev.dev_mean], fmt="o", ms=3,
               color="black", ecolor="black", elinewidth=0.6, capsize=1.5)
ax[1].axhline(0, color="black", lw=0.5); ax[1].axvline(0, color="black", lw=0.5, ls=":")
ax[1].set_xlabel("Days relative to reported-volume surge"); ax[1].set_ylabel("Change in fully automated share")
fig.tight_layout(); save_figure(fig, "figures/fig_capacity.pdf", metadata={"Creator": None, "Producer": None, "CreationDate": None, "ModDate": None}, bbox_inches="tight", pad_inches=0.06); plt.close(fig)

# Figure 5: platform heterogeneity in the surge slope
pl = pd.read_csv("results/capacity_platform.csv").sort_values("slope")
fig, ax = plt.subplots(figsize=(W, 3.0))
y = np.arange(len(pl))
ax.errorbar(pl.slope, y, xerr=1.96 * pl.se, fmt="o", ms=2.5, color="black", elinewidth=0.6)
ax.axvline(0, color="black", lw=0.5)
ax.set_yticks(y); ax.set_yticklabels(pl.p, fontsize=6)
ax.set_xlabel("Slope of fully automated share on surge")
fig.tight_layout(); save_figure(fig, "figures/fig_platforms.pdf", metadata={"Creator": None, "Producer": None, "CreationDate": None, "ModDate": None}, bbox_inches="tight", pad_inches=0.06); plt.close(fig)
print("figures written")

# Figure 6: submission lag by platform (twelve largest platforms by volume)
tp = pd.read_csv("results/timeliness_platforms.csv").sort_values("n", ascending=False).head(12)
segs = [("sub_0", "Same day", "black"), ("sub_1_7", "1 to 7 days", "0.35"), ("sub_8_30", "8 to 30 days", "0.6"),
        ("sub_31_365", "31 to 365 days", "0.85"), ("sub_gt365", "Over 365 days", "white")]
fig, ax = plt.subplots(figsize=(W, 2.8))
left = np.zeros(len(tp)); y = np.arange(len(tp))
for c, l, col in segs:
    v = (tp[c] / tp.n).values
    ax.barh(y, v, left=left, color=col, edgecolor="black", lw=0.4, label=l, height=0.7)
    left += v
ax.set_yticks(y); ax.set_yticklabels(tp.p, fontsize=6.5); ax.invert_yaxis()
ax.set_xlabel("Share of records by submission lag"); ax.set_xlim(0, 1)
ax.legend(frameon=False, fontsize=5.8, ncol=3, loc="upper center", bbox_to_anchor=(0.45, 1.22))
fig.tight_layout(); save_figure(fig, "figures/fig_timeliness.pdf", metadata={"Creator": None, "Producer": None, "CreationDate": None, "ModDate": None}, bbox_inches="tight", pad_inches=0.06); plt.close(fig)

# Figure 7: sexualized content under three definitions, and keyword recovery by platform
ks = pd.read_csv("results/keywords_series.csv", parse_dates=["d"])
kr = pd.read_csv("results/keywords_platform_recovery.csv").head(12)
fig, ax = plt.subplots(1, 2, figsize=(W * 2, 2.2), gridspec_kw={"width_ratios": [1.25, 1]})
ax[0].plot(ks.d, ks.by_category.fillna(0) + 1, color="black", lw=0.9, ls="-", label="Category only")
ax[0].plot(ks.d, ks.by_keyword.fillna(0) + 1, color="black", lw=0.9, ls="--", label="Keyword only")
ax[0].plot(ks.d, ks.by_union.fillna(0) + 1, color="black", lw=1.6, ls=":", label="Category or keyword")
ax[0].set_yscale("log"); ax[0].axvline(pd.Timestamp("2025-07-01"), color="black", lw=0.5)
ax[0].set_ylabel("Records per day plus one, log scale")
ax[0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 5, 7, 9])); ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax[0].legend(frameon=False, fontsize=6.2, loc="lower left")
yy = np.arange(len(kr))
ax[1].barh(yy + 0.2, kr.pre_share, height=0.38, color="black", label="Sexualized category share before")
ax[1].barh(yy - 0.2, kr.post_share_kw, height=0.38, color="white", edgecolor="black", lw=0.6, label="Sexual keyword share after")
ax[1].set_yticks(yy); ax[1].set_yticklabels(kr.p, fontsize=6.5); ax[1].invert_yaxis()
ax[1].set_xlabel("Share of the platform's records")
ax[1].legend(frameon=False, fontsize=6.0, loc="upper center", bbox_to_anchor=(0.45, 1.25))
fig.tight_layout(); save_figure(fig, "figures/fig_keywords.pdf", metadata={"Creator": None, "Producer": None, "CreationDate": None, "ModDate": None}, bbox_inches="tight", pad_inches=0.06); plt.close(fig)

print("extra figures written")
