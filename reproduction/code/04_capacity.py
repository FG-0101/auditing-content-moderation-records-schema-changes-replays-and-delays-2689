"""Capacity analysis on notice sourced decisions.

Question: when the inflow of notice sourced decisions on a platform rises above its recent level,
does the platform resolve a larger share of them by fully automated decision, and does the
number of non automated decisions rise less than proportionally?

Run from the package root:  python3 code/04_capacity.py
Outputs: results/capacity_regressions.csv, results/capacity_event.csv, results/capacity_platform.csv, results/capacity.json
"""
import json
import numpy as np
import pandas as pd
import duckdb
import pyfixest as pf
import sys
sys.path.insert(0, 'code')
from common import sql_excl

con = duckdb.connect()
df = con.sql("""SELECT p, d, nn, nn_full, nn_part, nn_human FROM 'data/derived/panel_platform_day.parquet'
                WHERE d <= DATE '2026-09-30' AND nn > 0""" + sql_excl()).df()
df = df.fillna(0).sort_values(["p", "d"])
df["d"] = pd.to_datetime(df.d)

# Trailing 28 day median of notice inflow (strictly before day d), computed per platform on a full calendar
def trailing(g):
    g = g.set_index("d").asfreq("D")
    g["p"] = g["p"].ffill()
    g[["nn", "nn_full", "nn_part", "nn_human"]] = g[["nn", "nn_full", "nn_part", "nn_human"]].fillna(0)
    g["med28"] = g["nn"].shift(1).rolling(28, min_periods=21).median()
    g["share_prev28"] = (g["nn_full"].shift(1).rolling(28, min_periods=21).sum() /
                         g["nn"].shift(1).rolling(28, min_periods=21).sum())
    return g.reset_index()
df = pd.concat([trailing(g) for _, g in df.groupby("p")], ignore_index=True)

MIN_DAY = 50
plat = df[df.nn >= MIN_DAY].groupby("p").agg(days=("d", "count"), nn=("nn", "sum"), nf=("nn_full", "sum"))
plat["full_share"] = plat.nf / plat.nn
# Platforms with enough history and interior automation shares (both routes in use)
keep = plat[(plat.days >= 180) & (plat.full_share > 0.02) & (plat.full_share < 0.98)].index
s = df[df.p.isin(keep) & (df.nn >= MIN_DAY) & (df.med28 > 0)].copy()
s["share_full"] = s.nn_full / s.nn
s["log_nn"] = np.log(s.nn)
s["surge"] = np.log(s.nn) - np.log(s.med28)          # log deviation from recent level
s["log_h"] = np.log1p(s.nn_human)
s["log_auto"] = np.log1p(s.nn_full)
s["dow"] = s.d.dt.dayofweek.astype(str)
s["pdow"] = s.p + "_" + s.dow
s["month"] = s.d.dt.to_period("M").astype(str)

res = []
def fit(name, fml, data, **kw):
    m = pf.feols(fml, data=data, vcov={"CRV1": "p"}, **kw)
    t = m.tidy().reset_index()
    r = t.iloc[0]
    res.append({"model": name, "term": r["Coefficient"], "estimate": r["Estimate"], "se": r["Std. Error"],
                "p_value": r["Pr(>|t|)"], "ci_low": r["2.5%"], "ci_high": r["97.5%"], "nobs": int(m._N),
                "platforms": int(data.p.nunique())})
    return m

fit("M1 share fully automated on surge", "share_full ~ surge | pdow + month", s)
fit("M2 share fully automated on log inflow", "share_full ~ log_nn | pdow + month", s)
fit("M3 log non automated on log inflow", "log_h ~ log_nn | pdow + month", s)
fit("M4 log fully automated on log inflow", "log_auto ~ log_nn | pdow + month", s)
# Composition control at platform by day by category level
c = con.sql("SELECT * FROM 'data/derived/panel_notice_platform_day_category.parquet' WHERE d <= DATE '2026-09-30'" + sql_excl()).df()
c = c[c.p.isin(keep)].fillna(0)
c["d"] = pd.to_datetime(c.d)
c = c.merge(s[["p", "d", "surge", "month", "pdow"]], on=["p", "d"], how="inner")
c = c[c.nn >= 20].copy()
c["share_full"] = c.nn_full / c.nn
c["pc"] = c.p + "_" + c.c
fit("M5 share fully automated on surge within category", "share_full ~ surge | pc + pdow + month", c, weights="nn")
c["log_nn_c"] = np.log(c.nn)
c["log_h_c"] = np.log1p(c.nn - c.nn_full)
fit("M7 log not fully automated on log inflow within category", "log_h_c ~ log_nn_c | pc + pdow + month", c)
# Placebo: surge on the following day should not predict today's share once today's surge is held
s = s.sort_values(["p", "d"])
s["surge_lead"] = s.groupby("p").surge.shift(-1)
fit("M6 placebo lead surge with own surge", "share_full ~ surge_lead + surge | pdow + month", s.dropna(subset=["surge_lead"]))
reg = pd.DataFrame(res)
reg.to_csv("results/capacity_regressions.csv", index=False)

# Event study around surge days (inflow above 3 times the trailing median, at least 500 decisions)
ev_days = s[(s.surge > np.log(3)) & (s.nn >= 500)][["p", "d"]]
full = df[df.p.isin(keep)].copy()
full["share_full"] = np.where(full.nn > 0, full.nn_full / full.nn.replace(0, np.nan), np.nan)
rows = []
for _, e in ev_days.iterrows():
    g = full[(full.p == e.p) & (full.d >= e.d - pd.Timedelta(days=7)) & (full.d <= e.d + pd.Timedelta(days=7))]
    base = full[(full.p == e.p) & (full.d == e.d)].share_prev28.values
    if len(base) == 0 or np.isnan(base[0]):
        continue
    for _, r in g.iterrows():
        if r.nn >= MIN_DAY:
            rows.append({"p": e.p, "event": str(e.d.date()), "k": int((r.d - e.d).days),
                         "dev": r.share_full - base[0], "inflow_ratio": r.nn / max(r.med28, 1)})
ev = pd.DataFrame(rows)
# Normalise each event by its own pre period (k = -7 to -2) so that the reference is the same daily share measure
pre = ev[(ev.k <= -2) & (ev.k >= -7)].groupby(["p", "event"]).dev.mean().rename("pre")
ev = ev.merge(pre, on=["p", "event"], how="inner")
ev["dev"] = ev.dev - ev.pre
evs = ev.groupby("k").agg(dev_mean=("dev", "mean"), dev_se=("dev", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
                          n=("dev", "size"), inflow_ratio_median=("inflow_ratio", "median")).reset_index()
evs.to_csv("results/capacity_event.csv", index=False)

# Platform level slopes for the heterogeneity figure
pl = []
for p_, g in s.groupby("p"):
    if len(g) < 180:
        continue
    m = pf.feols("share_full ~ surge | dow + month", data=g, vcov="hetero")
    t = m.tidy().iloc[0]
    pl.append({"p": p_, "slope": t["Estimate"], "se": t["Std. Error"], "days": len(g),
               "full_share": float(plat.loc[p_, "full_share"])})
pl = pd.DataFrame(pl).sort_values("slope")
pl.to_csv("results/capacity_platform.csv", index=False)

out = {"platforms_in_sample": int(s.p.nunique()), "platform_days": int(len(s)),
       "notice_decisions_in_sample": int(s.nn.sum()), "surge_events": int(len(ev_days)),
       "surge_platforms": int(ev_days.p.nunique()), "platforms_positive_slope_sig": int(((pl.slope - 1.96 * pl.se) > 0).sum()),
       "platforms_negative_slope_sig": int(((pl.slope + 1.96 * pl.se) < 0).sum()), "platform_slopes": int(len(pl))}
json.dump(out, open("results/capacity.json", "w"), indent=2)
print(reg.to_string())
print(evs.to_string())
print(json.dumps(out, indent=2))

# Platform clustered bootstrap for the event profile (events on the same platform are resampled together)
rng = np.random.default_rng(20261007)
plats = ev.p.unique()
boot = []
for b in range(500):
    pick = rng.choice(plats, size=len(plats), replace=True)
    bb = pd.concat([ev[ev.p == q] for q in pick])
    boot.append(bb.groupby("k").dev.mean())
boot = pd.concat(boot, axis=1)
evs["boot_lo"] = boot.quantile(0.025, axis=1).values
evs["boot_hi"] = boot.quantile(0.975, axis=1).values
evs.to_csv("results/capacity_event.csv", index=False)
print(evs[["k", "dev_mean", "boot_lo", "boot_hi"]].to_string())
s[["p", "d", "nn", "nn_full", "nn_human", "share_full", "log_nn", "surge", "log_h", "log_auto", "pdow", "month"]].to_parquet(
    "data/derived/capacity_sample.parquet", index=False)
