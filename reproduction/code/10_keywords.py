"""Can the keyword specification restore the sexualized content series across the 1 July 2025 migration?

Run from the package root after code/08_complete.py:  python3 code/10_keywords.py
Outputs: results/keywords_*.csv, results/keywords.json
"""
import json
import pandas as pd
import duckdb

con = duckdb.connect()
con.sql("CREATE VIEW k AS SELECT * FROM read_parquet('data/derived/complete/keywords_*.parquet', union_by_name=true) WHERE d <= DATE '2026-09-30'")
SEXCAT = "STATEMENT_CATEGORY_PORNOGRAPHY_OR_SEXUALIZED_CONTENT"
CUT = "DATE '2025-07-01'"

# Keyword use before and after the migration, all records
use = con.sql(f"""SELECT d >= {CUT} AS v2, SUM(n) AS n, SUM(COALESCE(any_keyword,0)) AS kw, SUM(COALESCE(sex_keyword,0)) AS sexkw
                  FROM k GROUP BY 1 ORDER BY 1""").df()
use.to_csv("results/keywords_use.csv", index=False)

# Daily series of sexualized content under three definitions, all platforms except the dominant one
ser = con.sql(f"""SELECT d, SUM(n) FILTER (WHERE c = '{SEXCAT}') AS by_category,
    SUM(COALESCE(sex_keyword,0)) AS by_keyword,
    SUM(n) FILTER (WHERE c = '{SEXCAT}' OR COALESCE(sex_keyword,0) > 0) AS by_union_upper
    FROM k WHERE p <> 'Google Shopping' AND d >= DATE '2025-03-01' AND d < DATE '2025-11-01' GROUP BY d ORDER BY d""").df()
# The union cannot be formed exactly from group counts, so it is computed exactly from the cell level instead
ser2 = con.sql(f"""SELECT d, SUM(CASE WHEN c = '{SEXCAT}' THEN n ELSE COALESCE(sex_keyword,0) END) AS by_union
    FROM k WHERE p <> 'Google Shopping' AND d >= DATE '2025-03-01' AND d < DATE '2025-11-01' GROUP BY d ORDER BY d""").df()
ser = ser.drop(columns="by_union_upper").merge(ser2, on="d")
ser["d"] = pd.to_datetime(ser.d)
ser.to_csv("results/keywords_series.csv", index=False)

# Platform level recovery: daily volume of sexualized content 60 days before (category or keyword) versus 60 days after (keyword)
plat = con.sql(f"""SELECT p,
    SUM(n) FILTER (WHERE d < {CUT} AND d >= DATE '2025-05-02' AND c = '{SEXCAT}') AS pre_cat,
    SUM(COALESCE(sex_keyword,0)) FILTER (WHERE d < {CUT} AND d >= DATE '2025-05-02' AND c = '{SEXCAT}') AS pre_cat_with_kw,
    SUM(CASE WHEN c = '{SEXCAT}' THEN n ELSE COALESCE(sex_keyword,0) END) FILTER (WHERE d < {CUT} AND d >= DATE '2025-05-02') AS pre_union,
    SUM(COALESCE(sex_keyword,0)) FILTER (WHERE d > {CUT} AND d <= DATE '2025-08-30') AS post_kw,
    SUM(n) FILTER (WHERE d < {CUT} AND d >= DATE '2025-05-02') AS pre_all,
    SUM(n) FILTER (WHERE d > {CUT} AND d <= DATE '2025-08-30') AS post_all,
    SUM(COALESCE(any_keyword,0)) FILTER (WHERE d > {CUT} AND d <= DATE '2025-08-30') AS post_any_kw
    FROM k GROUP BY p HAVING SUM(n) FILTER (WHERE d < {CUT} AND d >= DATE '2025-05-02' AND c = '{SEXCAT}') >= 10000
    ORDER BY pre_cat DESC""").df().fillna(0)
plat["pre_share"] = plat.pre_cat / plat.pre_all
plat["post_share_kw"] = plat.post_kw / plat.post_all
plat["recovery"] = plat.post_share_kw / plat.pre_share
plat["post_any_kw_share"] = plat.post_any_kw / plat.post_all
plat["pre_kw_within_cat"] = plat.pre_cat_with_kw / plat.pre_cat
plat.to_csv("results/keywords_platform_recovery.csv", index=False)

# Which categories carry the sexual keywords after the migration
where = con.sql(f"""SELECT c, SUM(COALESCE(sex_keyword,0)) AS sexkw FROM k WHERE d > {CUT} GROUP BY c ORDER BY sexkw DESC""").df()
where["share"] = where.sexkw / where.sexkw.sum()
where.to_csv("results/keywords_sex_categories.csv", index=False)

pre_w = ser[(ser.d < pd.Timestamp("2025-07-01")) & (ser.d >= pd.Timestamp("2025-05-02"))]
post_w = ser[(ser.d > pd.Timestamp("2025-07-01")) & (ser.d <= pd.Timestamp("2025-08-30"))]
out = {"kw_share_v1": float(use[~use.v2].kw.iloc[0] / use[~use.v2].n.iloc[0]),
       "kw_share_v2": float(use[use.v2].kw.iloc[0] / use[use.v2].n.iloc[0]),
       "union_daily_pre": float(pre_w.by_union.mean()), "union_daily_post": float(post_w.by_union.mean()),
       "category_daily_pre": float(pre_w.by_category.mean()), "keyword_daily_post": float(post_w.by_keyword.mean()),
       "platforms": int(len(plat)), "platforms_recovery_ge_half": int((plat.recovery >= 0.5).sum()),
       "platforms_recovery_lt_tenth": int((plat.recovery < 0.1).sum()),
       "pre_kw_within_cat_weighted": float(plat.pre_cat_with_kw.sum() / plat.pre_cat.sum()),
       "top_sex_kw_category_post": where.c.iloc[0] if len(where) else "", "top_sex_kw_category_share": float(where.share.iloc[0]) if len(where) else 0.0}
json.dump(out, open("results/keywords.json", "w"), indent=2)
print(json.dumps(out, indent=2))
print(plat[["p", "pre_cat", "pre_share", "post_share_kw", "recovery", "pre_kw_within_cat", "post_any_kw_share"]].to_string())
print(where.head(8).to_string())
