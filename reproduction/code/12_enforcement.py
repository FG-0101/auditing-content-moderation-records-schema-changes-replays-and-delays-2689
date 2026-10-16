"""Causal design: did the opening of DSA formal proceedings change how the targeted platform reports?

Treatment: the first opening of formal proceedings by the European Commission against a platform
(X 18 Dec 2023, TikTok 19 Feb 2024, AliExpress 14 Mar 2024, Facebook and Instagram 30 Apr 2024,
Temu 31 Oct 2024). Controls: very large online platforms that file SoRs throughout and were not
subject to formal proceedings in the window. Window: weeks from 25 Sep 2023 to 29 Jun 2025, before
the schema migration, so that category definitions are constant.
Estimators: two way fixed effects, the two stage estimator of Gardner (did2s), an event study,
and randomization inference over treatment assignment.
Run from the package root:  python3 code/12_enforcement.py
Outputs: results/enforcement_att.csv, results/enforcement_event.csv, results/enforcement.json, data/derived/enforcement_panel.parquet
"""
import json
import numpy as np
import pandas as pd
import duckdb
import pyfixest as pf

TREAT = {"X": "2023-12-18", "TikTok": "2024-02-19", "AliExpress": "2024-03-14", "Facebook": "2024-04-30",
         "Instagram": "2024-04-30", "Temu": "2024-10-31"}
CONTROL_CANDIDATES = ["Amazon", "Pinterest", "YouTube", "Google Maps", "Google Shopping", "Snapchat", "Google Play",
                      "LinkedIn", "Booking.com", "Zalando", "App Store", "Wikipedia"]
START, END = "2023-09-25", "2025-06-29"
con = duckdb.connect()
plats = list(TREAT) + CONTROL_CANDIDATES
plist = ", ".join("'" + p.replace("'", "''") + "'" for p in plats)
t = con.sql(f"""SELECT p, date_trunc('week', d)::DATE AS w, SUM(n) AS n, SUM(COALESCE(sub_0,0)) AS same,
    SUM(COALESCE(sub_8_30,0)+COALESCE(sub_31_365,0)+COALESCE(sub_gt365,0)) AS late7,
    SUM(COALESCE(app_after_submit,0)+COALESCE(content_after_app,0)) AS badd,
    SUM(n) FILTER (WHERE ad = 'AUTOMATED_DECISION_NOT_AUTOMATED') AS human,
    SUM(n) FILTER (WHERE notice) AS notice
    FROM read_parquet('data/derived/complete/timeliness_*.parquet')
    WHERE p IN ({plist}) AND d >= DATE '{START}' AND d <= DATE '{END}' GROUP BY 1, 2""").df()
k = con.sql(f"""SELECT p, date_trunc('week', d)::DATE AS w,
    SUM(n) FILTER (WHERE c = 'STATEMENT_CATEGORY_SCOPE_OF_PLATFORM_SERVICE') AS catchall, SUM(n) AS nk
    FROM read_parquet('data/derived/complete/keywords_*.parquet', union_by_name=true)
    WHERE p IN ({plist}) AND d >= DATE '{START}' AND d <= DATE '{END}' GROUP BY 1, 2""").df()
x = t.merge(k, on=["p", "w"], how="left").fillna(0)
x = x[x.n >= 100].copy()
x["y_same"] = x.same / x.n
x["y_late7"] = x.late7 / x.n
x["y_baddate"] = x.badd / x.n
x["y_human"] = x.human / x.n
x["y_notice"] = x.notice / x.n
x["y_catchall"] = x.catchall / x.nk.replace(0, np.nan)
x["y_logn"] = np.log(x.n)
weeks = x.groupby("p").w.nunique()
x = x[x.p.isin(weeks[weeks >= 30].index)].copy()
x["w"] = pd.to_datetime(x.w)
x["g"] = x.p.map(lambda p: pd.Timestamp(TREAT[p]) if p in TREAT else pd.NaT)
x["gw"] = x.g.dt.to_period("W-SUN").dt.start_time
x["treat_post"] = ((x.p.isin(TREAT)) & (x.w >= x.gw)).astype(int)
x["rel"] = np.where(x.p.isin(TREAT), ((x.w - x.gw).dt.days // 7), -1000).astype(int)
x["rel_b"] = np.where(x.rel == -1000, -1000, np.clip(x.rel, -12, 26))
x["week"] = x.w.dt.strftime("%Y-%m-%d")
x.to_parquet("data/derived/enforcement_panel.parquet", index=False)
controls = sorted(set(x.p) - set(TREAT))
treated = sorted(set(x.p) & set(TREAT))

OUT = ["y_same", "y_late7", "y_baddate", "y_human", "y_notice", "y_catchall", "y_logn"]
LAB = {"y_same": "Share submitted on the day of the decision", "y_late7": "Share submitted more than seven days late",
       "y_baddate": "Share violating a date constraint", "y_human": "Share decided without automation",
       "y_notice": "Share triggered by a notice", "y_catchall": "Share in the catch all category", "y_logn": "Log weekly records"}

def twfe(df, y):
    m = pf.feols(f"{y} ~ treat_post | p + week", data=df, vcov={"CRV1": "p"})
    return m.tidy().loc["treat_post"]

rng = np.random.default_rng(20261008)
pool = sorted(set(x.p))
dates = [TREAT[p] for p in treated]
rows, evrows = [], []
for y in OUT:
    d = x.dropna(subset=[y])
    r = twfe(d, y)
    g2 = pf.did2s(d, yname=y, first_stage="~ 0 | p + week", second_stage="~ treat_post", treatment="treat_post", cluster="p")
    r2 = g2.tidy().loc["treat_post"]
    # randomization inference: reassign the observed treatment dates to random platforms
    null = []
    for b in range(300):
        pick = rng.choice(pool, size=len(treated), replace=False)
        mp = {str(a): str(b) for a, b in zip(pick, rng.permutation(dates))}
        dd = d.copy()
        gg = dd.p.map(lambda p: pd.Timestamp(mp[p]) if p in mp else pd.NaT).dt.to_period("W-SUN").dt.start_time
        dd["treat_post"] = (dd.p.isin(mp) & (dd.w >= gg)).astype(int)
        try:
            null.append(pf.feols(f"{y} ~ treat_post | p + week", data=dd).coef()["treat_post"])
        except Exception:
            pass
    null = np.array(null)
    ri_p = float((np.abs(null) >= abs(r["Estimate"])).mean())
    pre = d[(d.rel_b < 0) & (d.rel_b != -1000)]
    rows.append({"outcome": y, "label": LAB[y], "twfe": r["Estimate"], "twfe_se": r["Std. Error"], "twfe_p": r["Pr(>|t|)"],
                 "did2s": r2["Estimate"], "did2s_se": r2["Std. Error"], "did2s_p": r2["Pr(>|t|)"], "ri_p": ri_p,
                 "ri_draws": int(len(null)), "control_mean": float(d[d.treat_post == 0][y].mean()),
                 "nobs": int(len(d)), "treated": len(treated), "controls": len(controls)})
    ks = [k for k in range(-12, 27) if k != -1]
    names = []
    for k in ks:
        nm = ("em" if k < 0 else "ep") + str(abs(k))
        d[nm] = ((d.rel_b == k) & d.p.isin(TREAT)).astype(int)
        names.append((nm, k))
    es = pf.did2s(d, yname=y, first_stage="~ 0 | p + week", second_stage="~ " + " + ".join(n for n, _ in names),
                  treatment="treat_post", cluster="p")
    te = es.tidy()
    for nm, k in names:
        if nm in te.index:
            evrows.append({"outcome": y, "k": k, "est": te.loc[nm, "Estimate"], "se": te.loc[nm, "Std. Error"]})
att = pd.DataFrame(rows)
att.to_csv("results/enforcement_att.csv", index=False)
pd.DataFrame(evrows).to_csv("results/enforcement_event.csv", index=False)
json.dump({"treated": treated, "controls": controls, "weeks": int(x.w.nunique()), "obs": int(len(x))},
          open("results/enforcement.json", "w"), indent=2)
print(treated, controls)
print(att[["outcome", "twfe", "twfe_se", "twfe_p", "did2s", "did2s_se", "did2s_p", "ri_p", "control_mean", "nobs"]].to_string())
