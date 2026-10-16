"""Territorial scope, automation and timeliness, account sanctions, and weekend supply of human review.

Run from the package root after code/08_complete.py and code/11_reconcile.py:  python3 code/16_extensions.py
Inputs : data/derived/complete/scope_*.parquet, data/derived/complete/timeliness_*.parquet,
         data/derived/panel_platform_day.parquet, data/derived/capacity_sample.parquet,
         data/agg/aggregated-simple.parquet (account sanctions)
Outputs: results/ext.json, results/ext_scope_monthly.csv, results/ext_scope_countries.csv, results/ext_scope_platforms.csv,
         results/ext_autotime_platforms.csv, results/ext_accounts_platforms.csv, results/ext_weekend.csv,
         results/ext_weekday_share.csv, figures/fig_scope.pdf, figures/fig_weekday.pdf
The six doubled days found by code/11_reconcile.py are excluded from every query on the simple product.
"""
from paper_style import save_figure
import glob
import json
import re
import sys
import numpy as np
import pandas as pd
import duckdb
import pyfixest as pf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "code")
from common import excluded_days, sql_excl

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 8,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
D = "data/derived"
END = "2026-09-30"
EXCL = excluded_days()
HOLI = ["12-24", "12-25", "12-26", "12-31", "01-01"]
con = duckdb.connect()
out = {"excluded_days": EXCL}


def tidy(m, terms):
    t = m.tidy()
    return {v: {"est": float(t.loc[v, "Estimate"]), "se": float(t.loc[v, "Std. Error"]), "p": float(t.loc[v, "Pr(>|t|)"])} for v in terms}


# 1. Territorial scope (complete product)
rows = []
for f in sorted(glob.glob(f"{D}/complete/scope_*.parquet")):
    m = re.search(r"scope_(\d{4}-\d{2})", f).group(1)
    x = con.sql(f"SELECT p, scope, SUM(n) AS n FROM '{f}' GROUP BY 1, 2").df(); x["m"] = m; rows.append(x)
sc = pd.concat(rows)
sc = sc[sc.m <= END[:7]].copy()
WIDE = {"EU", "EEA", "EEA_no_IS"}
sc["wide"] = sc.scope.isin(WIDE)
sc["single"] = sc.scope.fillna("").str.fullmatch(r"[A-Z]{2}") & ~sc.wide
tot = sc.n.sum()
mon = sc.groupby("m").apply(lambda g: pd.Series({"partial": g[~g.wide].n.sum() / g.n.sum(),
                                                  "single": g[g.single].n.sum() / g.n.sum()})).reset_index()
cty = sc[sc.single].groupby("scope").n.sum().sort_values(ascending=False)
plat = sc.groupby("p").apply(lambda g: pd.Series({"n": g.n.sum(), "partial": g[~g.wide].n.sum() / g.n.sum()}))
big = plat[plat.n >= 1e6].sort_values("partial", ascending=False)
mon.to_csv("results/ext_scope_monthly.csv", index=False)
(cty / cty.sum()).rename("share").to_csv("results/ext_scope_countries.csv")
big.to_csv("results/ext_scope_platforms.csv")
out["scope"] = {"total": int(tot), "partial_share": float(sc[~sc.wide].n.sum() / tot),
                "single_share": float(sc[sc.single].n.sum() / tot),
                "top_countries": [[c, float(v / cty.sum())] for c, v in cty.head(3).items()],
                "big_platforms": int(len(big)), "big_partial_over_half": int((big.partial > 0.5).sum()),
                "big_partial_below_tenth_pct": int((big.partial < 0.001).sum())}
fig, ax = plt.subplots(1, 2, figsize=(7.0, 1.9))
ax[0].plot(pd.to_datetime(mon.m), mon.partial, color="black", lw=0.9, label="Not applied Union wide")
ax[0].plot(pd.to_datetime(mon.m), mon.single, color="black", lw=0.9, ls="--", label="Applied in one country")
ax[0].set_ylabel("Share of records"); ax[0].legend(frameon=False, fontsize=6.3)
c10 = cty.head(10) / cty.sum()
ax[1].barh(range(len(c10)), c10.values, color="black"); ax[1].set_yticks(range(len(c10))); ax[1].set_yticklabels(c10.index)
ax[1].invert_yaxis(); ax[1].set_xlabel("Share of single country records")
fig.tight_layout(); save_figure(fig, "figures/fig_scope.pdf", metadata={"Creator": None, "Producer": None, "CreationDate": None, "ModDate": None}); plt.close(fig)

# 2. Automation and timeliness within platform and day (complete product)
t = con.sql(f"""SELECT p, d, ad, SUM(n) AS n, SUM(COALESCE(sub_0, 0)) AS s0,
  SUM(COALESCE(sub_8_30, 0) + COALESCE(sub_31_365, 0) + COALESCE(sub_gt365, 0)) AS l7, SUM(COALESCE(mod_0, 0)) AS m0
  FROM read_parquet('{D}/complete/timeliness_*.parquet') WHERE d <= DATE '{END}' GROUP BY 1, 2, 3""").df()
t = t[t.n >= 20].copy()
t["same"] = t.s0 / t.n; t["late7"] = t.l7 / t.n; t["modsame"] = t.m0 / t.n
t["pd"] = t.p + "_" + t.d.astype(str)
t["full"] = (t.ad == "AUTOMATED_DECISION_FULLY").astype(int); t["part"] = (t.ad == "AUTOMATED_DECISION_PARTIALLY").astype(int)
t2 = t[t.groupby("pd").ad.transform("nunique") >= 2].copy()
r8 = {y: tidy(pf.feols(f"{y} ~ full + part | pd", data=t2, weights="n", vcov={"CRV1": "p"}), ["full", "part"])
      for y in ["same", "late7", "modsame"]}
pl8 = t2.groupby(["p", "full"]).apply(lambda g: g.s0.sum() / g.n.sum()).unstack().dropna()
pl8.columns = ["not_full", "full"]; pl8["diff"] = pl8.full - pl8.not_full
pl8.to_csv("results/ext_autotime_platforms.csv")
out["autotime"] = {"cells": int(len(t2)), "platforms": int(t2.p.nunique()), "reg": r8,
                   "plat_n": int(len(pl8)), "plat_full_faster": int((pl8["diff"] > 0.05).sum()),
                   "plat_full_slower": int((pl8["diff"] < -0.05).sum())}

# 3. Account sanctions (simple product, doubled days excluded)
SRC = "data/agg/aggregated-simple.parquet/*/*.parquet"
con.sql(f"""CREATE VIEW a AS SELECT * FROM read_parquet('{SRC}', hive_partitioning=true)
            WHERE created_at::DATE <= DATE '{END}' {sql_excl('created_at')}""")
acc = con.sql("""SELECT platform_name AS p, decision_account IS NOT NULL AS acct, automated_decision AS ad, SUM(count) AS n
                 FROM a GROUP BY ALL""").df()
g = acc.groupby(["p", "acct"]).apply(lambda x: pd.Series({
    "n": x.n.sum(), "full": x[x.ad == "AUTOMATED_DECISION_FULLY"].n.sum() / x.n.sum(),
    "human": x[x.ad == "AUTOMATED_DECISION_NOT_AUTOMATED"].n.sum() / x.n.sum()})).reset_index()
w = g.pivot(index="p", columns="acct", values=["n", "full", "human"]).dropna()
w.columns = [f"{a_}_{'acct' if b_ else 'content'}" for a_, b_ in w.columns]
w = w[(w.n_acct >= 10000) & (w.n_content >= 10000)].copy()
w["acct_share"] = w.n_acct / (w.n_acct + w.n_content); w["human_gap"] = w.human_acct - w.human_content
w.to_csv("results/ext_accounts_platforms.csv")
out["accounts"] = {"acct_records": int(acc[acc.acct].n.sum()), "all_records": int(acc.n.sum()), "platforms": int(len(w)),
                   "more_human_for_accounts": int((w.human_gap > 0.05).sum()),
                   "less_human_for_accounts": int((w.human_gap < -0.05).sum()),
                   "median_human_gap": float(w.human_gap.median())}

# 4. Weekends and holidays as shifts in the supply of human review (capacity sample, doubled days excluded)
pdp = con.sql(f"SELECT * FROM '{D}/panel_platform_day.parquet' WHERE d <= DATE '{END}' {sql_excl('d')}").df().fillna(0)
pdp["d"] = pd.to_datetime(pdp.d)
keep = pd.read_parquet(f"{D}/capacity_sample.parquet").p.unique()
q = pdp[pdp.p.isin(keep) & (pdp.nn >= 50)].copy()
q["hshare"] = q.nn_human / q.nn; q["lognn"] = np.log(q.nn)
q["weekend"] = (q.d.dt.dayofweek >= 5).astype(int)
q["holiday"] = q.d.dt.strftime("%m-%d").isin(HOLI).astype(int)
q["pw"] = q.p + "_" + q.d.dt.strftime("%G-%V")
r9 = {}
for name, fml in [("share", "hshare ~ weekend + holiday | pw"), ("share_ctrl", "hshare ~ weekend + holiday + lognn | pw"),
                  ("inflow", "lognn ~ weekend + holiday | pw")]:
    r9[name] = tidy(pf.feols(fml, data=q, vcov={"CRV1": "p"}), ["weekend", "holiday"])
dl = con.sql(f"""SELECT p, d, SUM(n) AS n, SUM(COALESCE(mod_0, 0)) AS m0 FROM read_parquet('{D}/complete/timeliness_*.parquet')
  WHERE notice AND ad = 'AUTOMATED_DECISION_NOT_AUTOMATED' AND d <= DATE '{END}' GROUP BY 1, 2""").df()
dl["d"] = pd.to_datetime(dl.d); dl = dl[dl.p.isin(keep) & (dl.n >= 20)].copy(); dl["msame"] = dl.m0 / dl.n
dl["weekend"] = (dl.d.dt.dayofweek >= 5).astype(int); dl["holiday"] = dl.d.dt.strftime("%m-%d").isin(HOLI).astype(int)
dl["pw"] = dl.p + "_" + dl.d.dt.strftime("%G-%V")
r9["same_day_moderation"] = tidy(pf.feols("msame ~ weekend + holiday | pw", data=dl, vcov={"CRV1": "p"}), ["weekend", "holiday"])
rows = [{"model": k, "term": v, **r9[k][v]} for k in r9 for v in ["weekend", "holiday"]]
pd.DataFrame(rows).to_csv("results/ext_weekend.csv", index=False)
dow = q.groupby(q.d.dt.dayofweek).apply(lambda g_: g_.nn_human.sum() / g_.nn.sum())
dow.rename("share_not_automated").to_csv("results/ext_weekday_share.csv")
out["weekend"] = {"platforms": int(q.p.nunique()), "platform_days": int(len(q)), "delay_platform_days": int(len(dl)),
                  "reg": r9, "dow_share": [float(v) for v in dow.values]}
fig, ax = plt.subplots(figsize=(3.33, 1.5))
ax.bar(range(7), dow.values, color="black", width=0.6)
ax.set_xticks(range(7)); ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
ax.set_ylabel("Share not automated"); ax.set_ylim(0, max(dow.values) * 1.15)
fig.tight_layout(); save_figure(fig, "figures/fig_weekday.pdf", metadata={"Creator": None, "Producer": None, "CreationDate": None, "ModDate": None}); plt.close(fig)

json.dump(out, open("results/ext.json", "w"), indent=1)
print(json.dumps(out, indent=1)[:4000])
