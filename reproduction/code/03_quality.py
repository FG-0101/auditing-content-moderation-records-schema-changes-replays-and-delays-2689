"""Consistency rules, coverage and anomaly checks.

Run from the package root:  python3 code/03_quality.py
Outputs: results/quality_rules.csv, results/quality_platforms.csv, results/anomalies.csv, results/quality.json
"""
import json, os
import duckdb
import pandas as pd
import sys
sys.path.insert(0, 'code')
from common import sql_excl

os.makedirs("results", exist_ok=True)
con = duckdb.connect()
con.sql("CREATE VIEW fc AS SELECT * FROM 'data/derived/field_combinations.parquet'")
con.sql("CREATE VIEW pd AS SELECT * FROM 'data/derived/panel_platform_day.parquet' WHERE d <= DATE '2026-09-30'" + sql_excl())
TOTAL = con.sql("SELECT SUM(n) FROM fc").fetchone()[0]

TOS_ONLY = "('STATEMENT_CATEGORY_SCOPE_OF_PLATFORM_SERVICE','STATEMENT_CATEGORY_OTHER_VIOLATION_TC')"
rules = {
 "R1 full automated decision with manual detection, own initiative":
   "ad = 'AUTOMATED_DECISION_FULLY' AND adet = 'No' AND st = 'SOURCE_VOLUNTARY'",
 "R2 notice sourced record with automated detection":
   "st <> 'SOURCE_VOLUNTARY' AND adet = 'Yes'",
 "R3 illegal ground with terms of service catch all category":
   f"dg = 'DECISION_GROUND_ILLEGAL_CONTENT' AND c IN {TOS_ONLY}",
 "R4 no restriction recorded":
   "no_restriction",
 "R5 catch all category":
   f"c IN {TOS_ONLY}",
}
rows = []
for name, cond in rules.items():
    n, npf = con.sql(f"SELECT SUM(n), COUNT(DISTINCT p) FROM fc WHERE {cond}").fetchone()
    n = int(n or 0)
    nbig = con.sql(f"""SELECT COUNT(*) FROM (SELECT p, SUM(n) FILTER (WHERE {cond}) AS k, SUM(n) AS t FROM fc GROUP BY p)
                       WHERE t >= 100000 AND k::DOUBLE / t >= 0.5""").fetchone()[0]
    rows.append({"rule": name, "records": n, "share_of_all": n / TOTAL, "platforms_with_any": int(npf),
                 "platforms_majority_flagged_100k": int(nbig)})
rules_df = pd.DataFrame(rows)
rules_df.to_csv("results/quality_rules.csv", index=False)

# Degenerate automation reporting: one automated_decision value for all records of a platform
deg = con.sql("""SELECT p, SUM(n) AS n, COUNT(DISTINCT ad) AS ad_values, COUNT(DISTINCT adet) AS adet_values,
                 SUM(n) FILTER (WHERE ad = 'AUTOMATED_DECISION_NOT_AUTOMATED')::DOUBLE / SUM(n) AS human_share
                 FROM fc GROUP BY p ORDER BY n DESC""").df()
deg.to_csv("results/quality_platforms.csv", index=False)
big = deg[deg.n >= 100000]

# Volume anomalies: platform days above 10 times the trailing 28 day median, on platforms with history
con.sql("""CREATE TABLE z AS SELECT p, d, n,
  median(n) OVER (PARTITION BY p ORDER BY d ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING) AS med28,
  COUNT(*) OVER (PARTITION BY p ORDER BY d ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING) AS hist
  FROM pd""")
an = con.sql("SELECT * FROM z WHERE hist >= 28 AND n >= 10000 AND n > 10 * med28 ORDER BY n DESC").df()
an.to_csv("results/anomalies.csv", index=False)
an_share = an.n.sum() / con.sql("SELECT SUM(n) FROM pd").fetchone()[0]

# Continuity: gaps inside each platform's active span
gaps = con.sql("""SELECT p, MIN(d) AS first_d, MAX(d) AS last_d, COUNT(*) AS active_days,
                  date_diff('day', MIN(d), MAX(d)) + 1 AS span_days, SUM(n) AS n FROM pd GROUP BY p""").df()
gaps["missing_days"] = gaps.span_days - gaps.active_days
gaps["missing_share"] = gaps.missing_days / gaps.span_days
gaps.to_csv("results/continuity.csv", index=False)
gb = gaps[gaps.n >= 100000]

# Concentration
conc = con.sql("SELECT p, SUM(n) AS n FROM pd GROUP BY p ORDER BY n DESC").df()
conc["share"] = conc.n / conc.n.sum()
top1, top5 = conc.share.iloc[0], conc.share.iloc[:5].sum()
hhi = float((conc.share ** 2).sum())

out = {
 "total_records_to_2026_09_30": int(conc.n.sum()),
 "top1_platform": conc.p.iloc[0], "top1_share": float(top1), "top5_share": float(top5), "hhi": hhi,
 "platforms_ge_100k": int(len(big)),
 "platforms_ge_100k_single_ad_value": int((big.ad_values == 1).sum()),
 "platforms_ge_100k_single_ad_value_names": big[big.ad_values == 1].p.tolist(),
 "platforms_ge_100k_single_adet_value": int((big.adet_values == 1).sum()),
 "anomaly_platform_days": int(len(an)), "anomaly_platforms": int(an.p.nunique()),
 "anomaly_record_share": float(an_share),
 "platforms_ge_100k_with_missing_days": int((gb.missing_days > 0).sum()),
 "median_missing_share_ge_100k": float(gb.missing_share.median()),
 "platforms_ge_100k_missing_share_over_10pct": int((gb.missing_share > 0.10).sum()),
}
with open("results/quality.json", "w") as f:
    json.dump(out, f, indent=2)
print(rules_df.to_string())
print(json.dumps(out, indent=2))
print(an.head(10).to_string())
