"""Reconcile the two official aggregate products (simple and complete) month by month and platform by month.
Run from the package root:  python3 code/11_reconcile.py
Outputs: results/reconcile_monthly.csv, results/reconcile_platform_month.csv, results/reconcile.json"""
import json
import duckdb
con = duckdb.connect()
s = con.sql("""SELECT p, date_trunc('month', d)::DATE AS m, SUM(n) AS n_simple FROM 'data/derived/panel_platform_day.parquet'
               WHERE d <= DATE '2026-09-30' GROUP BY 1, 2""").df()
c = con.sql("""SELECT p, date_trunc('month', d)::DATE AS m, SUM(n) AS n_complete FROM read_parquet('data/derived/complete/timeliness_*.parquet')
               WHERE d <= DATE '2026-09-30' GROUP BY 1, 2""").df()
x = s.merge(c, on=["p", "m"], how="outer").fillna(0)
x["diff"] = x.n_simple - x.n_complete
x.to_csv("results/reconcile_platform_month.csv", index=False)
mo = x.groupby("m")[["n_simple", "n_complete", "diff"]].sum().reset_index()
mo["rel"] = mo["diff"] / mo.n_simple
mo.to_csv("results/reconcile_monthly.csv", index=False)
cells = x[x["diff"] != 0]
out = {"simple_total": int(x.n_simple.sum()), "complete_total": int(x.n_complete.sum()), "diff_total": int(x["diff"].sum()),
       "platform_month_cells": int(len(x)), "cells_disagree": int(len(cells)),
       "months_disagree": int((mo["diff"] != 0).sum()), "months": int(len(mo)),
       "max_abs_month_rel": float(mo.rel.abs().max()), "month_max": str(mo.loc[mo.rel.abs().idxmax(), "m"]),
       "abs_diff_total": int(x["diff"].abs().sum())}
json.dump(out, open("results/reconcile.json", "w"), indent=2)
print(json.dumps(out, indent=2)); print(mo[mo["diff"] != 0].to_string())
print(cells.reindex(cells["diff"].abs().sort_values(ascending=False).index).head(10).to_string())
# Day level localisation of the disagreement
ds = con.sql("SELECT d, SUM(n) AS n_simple FROM 'data/derived/panel_platform_day.parquet' WHERE d <= DATE '2026-09-30' GROUP BY d").df()
dc = con.sql("SELECT d, SUM(n) AS n_complete FROM read_parquet('data/derived/complete/timeliness_*.parquet') WHERE d <= DATE '2026-09-30' GROUP BY d").df()
dd = ds.merge(dc, on="d", how="outer").fillna(0)
dd["diff"] = dd.n_simple - dd.n_complete
bad = dd[dd["diff"] != 0].sort_values("d")
bad.to_csv("results/reconcile_days.csv", index=False)
out.update({"days_disagree": int(len(bad)), "days_missing_in_complete": int((bad.n_complete == 0).sum()),
            "days_disagree_list": [str(x)[:10] for x in bad.d]})
json.dump(out, open("results/reconcile.json", "w"), indent=2)
print(bad.to_string())
