"""Schema change of 1 July 2025: category retirement, adoption timing and lossy migration.

Run from the package root:  python3 code/05_schema.py
Outputs: results/schema_*.csv, results/schema.json
"""
import json
import duckdb
import pandas as pd

con = duckdb.connect()
pdc = con.sql("SELECT * FROM 'data/derived/platform_day_category_2025q2q3.parquet'").df()
pdc["d"] = pd.to_datetime(pdc.d)
CUT = pd.Timestamp("2025-07-01")
OLD_ONLY = ["STATEMENT_CATEGORY_PORNOGRAPHY_OR_SEXUALIZED_CONTENT", "STATEMENT_CATEGORY_SCOPE_OF_PLATFORM_SERVICE",
            "STATEMENT_CATEGORY_NON_CONSENSUAL_BEHAVIOUR", "STATEMENT_CATEGORY_UNSAFE_AND_ILLEGAL_PRODUCTS"]
NEW_ONLY = ["STATEMENT_CATEGORY_CONSUMER_INFORMATION", "STATEMENT_CATEGORY_CYBER_VIOLENCE",
            "STATEMENT_CATEGORY_CYBER_VIOLENCE_AGAINST_WOMEN", "STATEMENT_CATEGORY_OTHER_VIOLATION_TC",
            "STATEMENT_CATEGORY_UNSAFE_AND_PROHIBITED_PRODUCTS", "STATEMENT_CATEGORY_NOT_SPECIFIED_NOTICE"]
# Observed first and last use of every category across the full history
life = con.sql("""SELECT c, MIN(m) AS first_month, MAX(m) AS last_month, SUM(n) AS n
                  FROM 'data/derived/platform_month_category.parquet' GROUP BY c ORDER BY first_month, c""").df()
life.to_csv("results/schema_category_lifetimes.csv", index=False)
old_after = pdc[(pdc.d > CUT) & pdc.c.isin(OLD_ONLY)].n.sum()
old_on_cut = pdc[(pdc.d == CUT) & pdc.c.isin(OLD_ONLY)].n.sum()
new_before = pdc[(pdc.d < CUT) & pdc.c.isin(NEW_ONLY)].n.sum()

# Category mix 60 days before and 60 days after the change, per platform
win = 60
pre = pdc[(pdc.d >= CUT - pd.Timedelta(days=win)) & (pdc.d < CUT)]
post = pdc[(pdc.d > CUT) & (pdc.d <= CUT + pd.Timedelta(days=win))]
def mix(x):
    t = x.groupby(["p", "c"]).n.sum().reset_index()
    t["share"] = t.n / t.groupby("p").n.transform("sum")
    return t
mpre, mpost = mix(pre), mix(post)
m = mpre.merge(mpost, on=["p", "c"], how="outer", suffixes=("_pre", "_post")).fillna(0)
m["delta"] = m.share_post - m.share_pre
m.to_csv("results/schema_platform_mix.csv", index=False)

SEX = "STATEMENT_CATEGORY_PORNOGRAPHY_OR_SEXUALIZED_CONTENT"
# Platforms that used the sexualized content category before the change
sx = m[(m.c == SEX) & (m.n_pre >= 10000)].sort_values("n_pre", ascending=False)
dest = []
for p_ in sx.p:
    g = m[(m.p == p_) & (m.c != SEX)].sort_values("delta", ascending=False)
    top = g.iloc[0]
    dest.append({"p": p_, "sex_share_pre": float(sx[sx.p == p_].share_pre.iloc[0]),
                 "sex_records_pre": int(sx[sx.p == p_].n_pre.iloc[0]),
                 "top_gaining_category": top.c.replace("STATEMENT_CATEGORY_", ""), "top_gain": float(top.delta),
                 "gain_to_catch_all": float(g[g.c.isin(["STATEMENT_CATEGORY_OTHER_VIOLATION_TC"])].delta.sum()),
                 "gain_to_minors": float(g[g.c == "STATEMENT_CATEGORY_PROTECTION_OF_MINORS"].delta.sum())})
dest = pd.DataFrame(dest)
dest.to_csv("results/schema_sexualized_destinations.csv", index=False)

# Catch all reliance before and after: SCOPE_OF_PLATFORM_SERVICE (old) versus OTHER_VIOLATION_TC (new), excluding the dominant platform
def catchall(x, cats):
    y = x[x.p != "Google Shopping"]
    return float(y[y.c.isin(cats)].n.sum() / y.n.sum())
ca_pre = catchall(pre, ["STATEMENT_CATEGORY_SCOPE_OF_PLATFORM_SERVICE"])
ca_post = catchall(post, ["STATEMENT_CATEGORY_OTHER_VIOLATION_TC"])

# Daily series of the sexualized content related categories, all platforms
ser = pdc[pdc.c.isin([SEX, "STATEMENT_CATEGORY_PROTECTION_OF_MINORS", "STATEMENT_CATEGORY_OTHER_VIOLATION_TC",
                     "STATEMENT_CATEGORY_SCOPE_OF_PLATFORM_SERVICE"])]
ser = ser[ser.p != "Google Shopping"].groupby(["d", "c"]).n.sum().reset_index()
ser.to_csv("results/schema_daily_series.csv", index=False)

out = {"old_category_records_after_cut": int(old_after), "old_category_records_on_cut_day": int(old_on_cut),
       "new_category_records_before_cut": int(new_before),
       "sexualized_platforms_ge_10k_pre": int(len(sx)),
       "sexualized_records_pre_window": int(sx.n_pre.sum()),
       "catch_all_share_pre_excl_dominant": ca_pre, "catch_all_share_post_excl_dominant": ca_post,
       "sexualized_top_gain_to_other_violation_tc": int((dest.top_gaining_category == "OTHER_VIOLATION_TC").sum()) if len(dest) else 0}
json.dump(out, open("results/schema.json", "w"), indent=2)
print(life.to_string()); print(dest.to_string()); print(json.dumps(out, indent=2))
