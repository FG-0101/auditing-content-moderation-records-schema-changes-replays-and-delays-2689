"""Build analysis tables from the official monthly aggregates.

Run from the package root:  python3 code/02_build.py
Inputs : data/agg/aggregated-simple.parquet (hive partitioned by created_at_month)
Outputs: data/derived/*.parquet and data/derived/inventory.json
"""
import json
import duckdb

SRC = "data/agg/aggregated-simple.parquet/*/*.parquet"
con = duckdb.connect()
END = "2026-09-30"  # analysis window ends here, because the last days of a month are appended over several days
con.sql(f"CREATE VIEW a_all AS SELECT * FROM read_parquet('{SRC}', hive_partitioning=true)")
con.sql(f"CREATE VIEW a AS SELECT * FROM a_all WHERE created_at::DATE <= DATE '{END}'")

VIS = ["DECISION_VISIBILITY_CONTENT_REMOVED", "DECISION_VISIBILITY_CONTENT_DISABLED",
       "DECISION_VISIBILITY_CONTENT_DEMOTED", "DECISION_VISIBILITY_CONTENT_AGE_RESTRICTED",
       "DECISION_VISIBILITY_CONTENT_INTERACTION_RESTRICTED", "DECISION_VISIBILITY_CONTENT_LABELLED",
       "DECISION_VISIBILITY_OTHER"]
no_vis = " AND ".join(f"COALESCE({c}, false) = false" for c in VIS)

# Inventory of the download (all files) and of the analysis window
dl = con.sql("""SELECT COUNT(*), MAX(created_at)::DATE,
                 SUM(CASE WHEN date_trunc('month', created_at)::DATE <> created_at_month THEN 1 ELSE 0 END),
                 SUM(CASE WHEN count IS NULL OR count <= 0 THEN 1 ELSE 0 END) FROM a_all""").fetchone()
inv = con.sql("""SELECT COUNT(*) AS agg_rows, SUM(count) AS sors, COUNT(DISTINCT platform_name) AS platforms,
                 COUNT(DISTINCT created_at::DATE) AS days, MIN(created_at)::DATE AS first_day, MAX(created_at)::DATE AS last_day
                 FROM a""").fetchone()
keys = ["agg_rows", "sors", "platforms", "days", "first_day", "last_day"]
inventory = {k: (str(v) if k.endswith("day") else int(v)) for k, v in zip(keys, inv)}
inventory.update({"window_end": END, "download_agg_rows": int(dl[0]), "download_last_day": str(dl[1]),
                  "partition_mismatch": int(dl[2]), "nonpositive_counts": int(dl[3])})
inventory["parquet_parts"] = int(con.sql(f"SELECT COUNT(DISTINCT filename) FROM read_parquet('{SRC}', filename=true)").fetchone()[0])

# Platform by day panel with automation and source breakdowns
con.sql("""CREATE TABLE pd AS SELECT platform_name AS p, created_at::DATE AS d,
  SUM(count) AS n,
  SUM(count) FILTER (WHERE automated_decision = 'AUTOMATED_DECISION_FULLY') AS n_full,
  SUM(count) FILTER (WHERE automated_decision = 'AUTOMATED_DECISION_PARTIALLY') AS n_part,
  SUM(count) FILTER (WHERE automated_decision = 'AUTOMATED_DECISION_NOT_AUTOMATED') AS n_human,
  SUM(count) FILTER (WHERE automated_detection = 'Yes') AS n_detect,
  SUM(count) FILTER (WHERE source_type <> 'SOURCE_VOLUNTARY') AS nn,
  SUM(count) FILTER (WHERE source_type <> 'SOURCE_VOLUNTARY' AND automated_decision = 'AUTOMATED_DECISION_FULLY') AS nn_full,
  SUM(count) FILTER (WHERE source_type <> 'SOURCE_VOLUNTARY' AND automated_decision = 'AUTOMATED_DECISION_PARTIALLY') AS nn_part,
  SUM(count) FILTER (WHERE source_type <> 'SOURCE_VOLUNTARY' AND automated_decision = 'AUTOMATED_DECISION_NOT_AUTOMATED') AS nn_human
  FROM a GROUP BY 1, 2""")
con.sql("COPY pd TO 'data/derived/panel_platform_day.parquet' (FORMAT parquet)")

# Notice sourced decisions at platform by day by category level (for composition controls)
con.sql("""COPY (SELECT platform_name AS p, created_at::DATE AS d, category AS c,
  SUM(count) AS nn,
  SUM(count) FILTER (WHERE automated_decision = 'AUTOMATED_DECISION_FULLY') AS nn_full
  FROM a WHERE source_type <> 'SOURCE_VOLUNTARY' GROUP BY 1, 2, 3)
  TO 'data/derived/panel_notice_platform_day_category.parquet' (FORMAT parquet)""")

# Platform by month by category table (schema drift analysis)
con.sql("""COPY (SELECT platform_name AS p, date_trunc('month', created_at)::DATE AS m, category AS c,
  SUM(count) AS n FROM a GROUP BY 1, 2, 3)
  TO 'data/derived/platform_month_category.parquet' (FORMAT parquet)""")
# Platform by day by category around the schema change (2025-04-01 to 2025-09-30)
con.sql("""COPY (SELECT platform_name AS p, created_at::DATE AS d, category AS c, SUM(count) AS n
  FROM a WHERE created_at >= '2025-04-01' AND created_at < '2025-10-01' GROUP BY 1, 2, 3)
  TO 'data/derived/platform_day_category_2025q2q3.parquet' (FORMAT parquet)""")

# Field combination counts used by the consistency rules
con.sql(f"""COPY (SELECT platform_name AS p, created_at::DATE >= DATE '2025-07-01' AS v2,
  automated_decision AS ad, automated_detection AS adet, source_type AS st, decision_ground AS dg, category AS c,
  ({no_vis}) AND decision_account IS NULL AND decision_monetary IS NULL AND decision_provision IS NULL AS no_restriction,
  SUM(count) AS n
  FROM a GROUP BY ALL) TO 'data/derived/field_combinations.parquet' (FORMAT parquet)""")

with open("data/derived/inventory.json", "w") as f:
    json.dump(inventory, f, indent=2)
print(json.dumps(inventory, indent=2))
