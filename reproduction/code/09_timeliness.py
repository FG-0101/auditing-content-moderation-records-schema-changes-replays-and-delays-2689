"""Timeliness and date validity from the complete aggregates, and the queueing test on delays.

Run from the package root after code/08_complete.py:  python3 code/09_timeliness.py
Outputs: results/timeliness_*.csv, results/timeliness.json, results/delay_capacity.csv
"""
import json
import numpy as np
import pandas as pd
import duckdb
import pyfixest as pf

con = duckdb.connect()
con.sql("CREATE VIEW t AS SELECT * FROM read_parquet('data/derived/complete/timeliness_*.parquet') WHERE d <= DATE '2026-09-30'")
tot = con.sql("SELECT SUM(n) FROM t").fetchone()[0]
cols = ["sub_neg", "sub_0", "sub_1_7", "sub_8_30", "sub_31_365", "sub_gt365", "sub_null",
        "mod_neg", "mod_0", "mod_1_7", "mod_8_30", "mod_31_365", "mod_gt365", "mod_null",
        "app_after_submit", "app_far_future", "content_after_app", "content_2000_01_01", "content_before_2005"]
agg = con.sql("SELECT " + ", ".join(f"SUM(COALESCE({c},0)) AS {c}" for c in cols) + ", SUM(n) AS n FROM t").df().iloc[0]

# Platform level timeliness (platforms with at least 100,000 records)
pl = con.sql("SELECT p, SUM(n) AS n, " + ", ".join(f"SUM(COALESCE({c},0)) AS {c}" for c in cols) +
             ", SUM(COALESCE(sub_sum,0)) AS sub_sum FROM t GROUP BY p HAVING SUM(n) >= 100000").df()
pl["late7"] = (pl.sub_8_30 + pl.sub_31_365 + pl.sub_gt365) / pl.n
pl["late30"] = (pl.sub_31_365 + pl.sub_gt365) / pl.n
pl["sameday"] = pl.sub_0 / pl.n
pl["mean_sub"] = pl.sub_sum / (pl.n - pl.sub_null)
pl["neg_mod"] = pl.mod_neg / pl.n
pl.sort_values("n", ascending=False).to_csv("results/timeliness_platforms.csv", index=False)

# Monthly share of records submitted more than 7 and 30 days after application, all platforms except the dominant one
mon = con.sql("""SELECT date_trunc('month', d)::DATE AS m, p = 'Google Shopping' AS dom, SUM(n) AS n,
    SUM(COALESCE(sub_8_30,0)+COALESCE(sub_31_365,0)+COALESCE(sub_gt365,0)) AS late7,
    SUM(COALESCE(sub_31_365,0)+COALESCE(sub_gt365,0)) AS late30, SUM(COALESCE(sub_0,0)) AS same
    FROM t GROUP BY 1, 2 ORDER BY 1""").df()
mon.to_csv("results/timeliness_monthly.csv", index=False)

# Moderation delay distribution (application date minus content date) by route, notice versus own initiative
route = con.sql("""SELECT notice, ad, SUM(n) AS n, SUM(COALESCE(mod_0,0)) AS m0, SUM(COALESCE(mod_1_7,0)) AS m7,
    SUM(COALESCE(mod_8_30,0)) AS m30, SUM(COALESCE(mod_31_365,0)) AS m365, SUM(COALESCE(mod_gt365,0)) AS mgt,
    SUM(COALESCE(mod_neg,0)) AS mneg, SUM(COALESCE(mod_null,0)) AS mnull FROM t GROUP BY 1, 2 ORDER BY 1, 2""").df()
route.to_csv("results/timeliness_routes.csv", index=False)

# Queueing test: on platforms in the capacity sample, does the delay of notice sourced decisions
# taken without automation rise in the days after an inflow surge?
s = pd.read_parquet("data/derived/capacity_sample.parquet")[["p", "d", "surge", "pdow", "month"]]
s["d"] = pd.to_datetime(s.d)
dl = con.sql("""SELECT p, d, SUM(n) AS n, SUM(COALESCE(sub_sum,0)) AS sub_sum, SUM(COALESCE(sub_null,0)) AS sub_null,
    SUM(COALESCE(mod_sum,0)) AS mod_sum, SUM(COALESCE(mod_null,0)) AS mod_null
    FROM t WHERE notice AND ad = 'AUTOMATED_DECISION_NOT_AUTOMATED' GROUP BY 1, 2""").df()
dl["d"] = pd.to_datetime(dl.d)
dl = dl[dl.n >= 20].copy()
dl["log_sub"] = np.log1p(dl.sub_sum / (dl.n - dl.sub_null).replace(0, np.nan))
dl["log_mod"] = np.log1p(dl.mod_sum / (dl.n - dl.mod_null).replace(0, np.nan))
s = s.sort_values(["p", "d"])
for k in range(1, 8):
    s[f"surge_l{k}"] = s.groupby("p").surge.shift(k)
s["surge_lag_mean"] = s[[f"surge_l{k}" for k in range(1, 8)]].mean(axis=1)
x = dl.merge(s, on=["p", "d"], how="inner").dropna(subset=["surge_lag_mean"])
res = []
for y in ["log_sub", "log_mod"]:
    xx = x.dropna(subset=[y])
    for term in ["surge", "surge_lag_mean"]:
        m = pf.feols(f"{y} ~ surge + surge_lag_mean | pdow + month", data=xx, vcov={"CRV1": "p"})
        r = m.tidy().loc[term]
        res.append({"outcome": y, "term": term, "estimate": r["Estimate"], "se": r["Std. Error"],
                    "p_value": r["Pr(>|t|)"], "ci_low": r["2.5%"], "ci_high": r["97.5%"], "nobs": int(m._N),
                    "platforms": int(xx.p.nunique())})
pd.DataFrame(res).to_csv("results/delay_capacity.csv", index=False)

big = pl[pl.n >= 1_000_000]
out = {"records": int(tot),
       "share_same_day": float(agg.sub_0 / agg.n), "share_late7": float((agg.sub_8_30 + agg.sub_31_365 + agg.sub_gt365) / agg.n),
       "share_late30": float((agg.sub_31_365 + agg.sub_gt365) / agg.n), "records_late365": int(agg.sub_gt365),
       "app_after_submit": int(agg.app_after_submit), "app_far_future": int(agg.app_far_future),
       "content_after_app": int(agg.content_after_app), "content_2000_01_01": int(agg.content_2000_01_01),
       "content_before_2005": int(agg.content_before_2005), "content_null": int(agg.mod_null),
       "platforms_ge_1m": int(len(big)), "platforms_ge_1m_late7_over_10pct": int((big.late7 > 0.10).sum()),
       "platforms_ge_1m_median_sameday": float(big.sameday.median()),
       "platforms_ge_100k_any_content_after_app": int((pl.content_after_app > 0).sum()),
       "platforms_ge_100k": int(len(pl))}
json.dump(out, open("results/timeliness.json", "w"), indent=2)
print(json.dumps(out, indent=2))
print(pd.DataFrame(res).to_string())
print(big.sort_values("late7", ascending=False)[["p", "n", "sameday", "late7", "late30", "mean_sub", "neg_mod"]].head(15).to_string())
