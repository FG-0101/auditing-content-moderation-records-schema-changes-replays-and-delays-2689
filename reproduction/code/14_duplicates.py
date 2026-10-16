"""Detect replayed daily files in an append based aggregate product.

Four tests, run on every month of the simple product up to 30 September 2026:
  A  exact replay, provenance based: two write jobs hold identical multisets of rows for the same day
     (64 bit candidate fingerprint followed by canonical multiset comparison).
  B  near replay, provenance based: share of keys on which two jobs carry identical counts, over the union
     of their keys (catches a replay that was recomputed after late records arrived).
  C  provenance free: share of a day's keys that occur in at least two rows with identical counts.
  D  reference based: ratio of the simple to the complete product on each platform day.
A and B and C are evaluated at day and at platform by day granularity.
Calibration: a thinning null that splits every cell of every day binomially into two units.
Evaluation: injection of synthetic replays and of two negative controls into clean days.

Run from the package root:  python3 code/14_duplicates.py
Inputs : data/agg/aggregated-simple.parquet (from code/01_fetch.sh), data/derived/complete/timeliness_*.parquet,
         data/derived/panel_platform_day.parquet
Outputs: results/dup_days.csv, results/dup_flags.csv, results/dup_null.csv, results/dup_injection.csv,
         results/dup_timing.csv, results/dup.json
"""
import json
import time
import numpy as np
import pandas as pd
import duckdb

END = "2026-09-30"
TAU_B, TAU_C, MINKEYS, TOL_D = 0.9, 0.5, 20, 0.01
SEED = 20261008
rng = np.random.default_rng(SEED)
M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def splitmix64(x):
    x = np.asarray(x, dtype=np.uint64)
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def row_hash(h, c):
    return splitmix64(np.asarray(h, dtype=np.uint64) ^ splitmix64(np.asarray(c, dtype=np.uint64)))


# ---------------------------------------------------------------- load
t0 = time.perf_counter()
con = duckdb.connect()
SRC = "data/agg/aggregated-simple.parquet/*/*.parquet"
con.sql(f"""CREATE VIEW a AS SELECT *, regexp_extract(filename, 'part-\\d+-([0-9a-f-]+)\\.c000', 1) AS job
            FROM read_parquet('{SRC}', hive_partitioning=true, filename=true)""")
cols = [r[0] for r in con.sql("DESCRIBE a").fetchall() if r[0] not in ("count", "filename", "job", "created_at_month")]
key = ", ".join(cols)
# Verify key hash injectivity on every distinct observed grouping tuple.
collisions = con.sql(f"SELECT COUNT(*) FROM (SELECT hash({key}) AS h FROM (SELECT DISTINCT {key} FROM a) GROUP BY h HAVING COUNT(*) > 1)").fetchone()[0]
assert collisions == 0, "grouping-key hash collision: replace key representation before testing"
df = con.sql(f"""SELECT job, created_at::DATE AS d, platform_name AS p, hash({key}) AS h, count AS c
                 FROM a WHERE created_at::DATE <= DATE '{END}'""").df()
df["d"] = pd.to_datetime(df.d)
df["h"] = df.h.astype(np.uint64)
df["c"] = df.c.astype(np.int64)
t_load = time.perf_counter() - t0
N = len(df)
assert not df.duplicated(["job", "d", "h"]).any(), "a key occurs twice inside one job and day"

# Consistency of this download with the snapshot used by the other scripts
chk = con.sql(f"""WITH new AS (SELECT platform_name AS p, created_at::DATE AS d, SUM(count) AS n FROM a
                   WHERE created_at::DATE <= DATE '{END}' GROUP BY 1, 2),
               old AS (SELECT p, d, n FROM 'data/derived/panel_platform_day.parquet' WHERE d <= DATE '{END}')
               SELECT COUNT(*) AS cells, SUM(CASE WHEN new.n IS DISTINCT FROM old.n THEN 1 ELSE 0 END) AS differ
               FROM new FULL OUTER JOIN old ON new.p = old.p AND new.d = old.d""").fetchone()

# Complete product totals by platform and day (reference for test D)
comp = con.sql(f"""SELECT p, d, SUM(n) AS nc FROM read_parquet('data/derived/complete/timeliness_*.parquet')
                   WHERE d <= DATE '{END}' GROUP BY 1, 2""").df()
comp["d"] = pd.to_datetime(comp.d)


# ---------------------------------------------------------------- tests
def test_A(x, level):
    """Exact replay. Units are (job, *level); returns flagged groups of units with identical fingerprints."""
    x = x.assign(rh=row_hash(x.h.values, x.c.values))
    g = list(level) + ["job"]
    x = x.sort_values(g, kind="stable")
    codes = x.groupby(g, sort=False, observed=True).ngroup().values
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    fp = np.add.reduceat(x.rh.values.astype(np.uint64), starts)
    u = x.iloc[starts][g].reset_index(drop=True)
    u["fp"] = fp
    u["nrow"] = np.diff(np.r_[starts, len(x)])
    u["n"] = np.add.reduceat(x.c.values, starts)
    u = u[u.nrow >= MINKEYS]
    grp = u.groupby(list(level) + ["fp", "nrow"], observed=True)
    k = grp.job.transform("nunique")
    candidates = u[k >= 2].assign(k=k[k >= 2])
    # Hash equality is only a candidate screen; exactness comes from the row multiset.
    keep = []
    blocks = {t: np.array(sorted(zip(z.h.tolist(), z.c.tolist())), dtype=[("h", "u8"), ("c", "i8")])
              for t, z in x.merge(candidates[g].drop_duplicates(), on=g).groupby(g, observed=True, sort=False)}
    for _, z in candidates.groupby(list(level) + ["fp", "nrow"], observed=True):
        classes = []
        for idx, row in z.iterrows():
            t = tuple(row[q] for q in g)
            for indices, canonical in classes:
                if np.array_equal(blocks[t], canonical):
                    indices.append(idx); break
            else:
                classes.append(([idx], blocks[t]))
        for indices, canonical in classes:
            if len(indices) >= 2: keep.extend(indices)
    return candidates.loc[keep]


def test_B(x, level):
    """Near replay. For every pair of jobs on the same day (and platform), share of keys with identical counts."""
    sizes = x.groupby(list(level) + ["job"], observed=True).agg(keys=("h", "size"), n=("c", "sum")).reset_index()
    on = ["d", "h"] + [l for l in level if l not in ("d",)]
    m = x.merge(x, on=on, suffixes=("_u", "_v"))
    m = m[m.job_u < m.job_v]
    if m.empty:
        return pd.DataFrame(columns=list(level) + ["job_u", "job_v", "mu"])
    pr = m.assign(eq=(m.c_u == m.c_v)).groupby(list(level) + ["job_u", "job_v"], observed=True).agg(
        inter=("eq", "size"), eq=("eq", "sum")).reset_index()
    pr = pr.merge(sizes.rename(columns={"job": "job_u", "keys": "k_u", "n": "n_u"}), on=list(level) + ["job_u"])
    pr = pr.merge(sizes.rename(columns={"job": "job_v", "keys": "k_v", "n": "n_v"}), on=list(level) + ["job_v"])
    pr["union"] = pr.k_u + pr.k_v - pr.inter
    pr["mu"] = pr["eq"] / pr["union"]
    return pr


def test_C(x, level):
    """Provenance free. Share of keys that occur in at least two rows carrying identical counts."""
    g = x.groupby(list(level) + ["h"], observed=True).c.agg(["size", "nunique"]).reset_index()
    g["rep"] = (g["size"] >= 2) & (g["nunique"] < g["size"])
    r = g.groupby(list(level), observed=True).agg(keys=("h", "size"), rep=("rep", "sum")).reset_index()
    r["rho"] = r.rep / r["keys"]
    return r


def test_D(x, ref):
    """Reference based. Ratio of platform day totals to the complete product, rounded to an integer multiple."""
    s = x.groupby(["p", "d"], observed=True).c.sum().rename("ns").reset_index()
    s["p"] = s.p.astype(str)
    z = s.merge(ref, on=["p", "d"], how="left")
    z = z[z.nc >= 100].copy()
    z["r"] = z.ns / z.nc
    z["k"] = np.rint(z.r).astype(int)
    z["replay"] = (z.k >= 2) & ((z.r - z.k).abs() <= TOL_D)
    return z


def day_flags(x, ref):
    """Run all tests on frame x and return per day booleans for each test and granularity."""
    out = {}
    a1 = test_A(x, ["d"]); out["A_day"] = set(a1.d)
    a2 = test_A(x, ["d", "p"]); out["A_plat"] = set(a2.d)
    b1 = test_B(x, ["d"]); out["B_day"] = set(b1.d[(b1.mu >= TAU_B) & (b1["union"] >= MINKEYS)])
    b2 = test_B(x, ["d", "p"]); out["B_plat"] = set(b2.d[(b2.mu >= TAU_B) & (b2["union"] >= MINKEYS)])
    c1 = test_C(x, ["d"]); out["C_day"] = set(c1.d[(c1.rho >= TAU_C) & (c1["keys"] >= MINKEYS)])
    c2 = test_C(x, ["d", "p"]); out["C_plat"] = set(c2.d[(c2.rho >= TAU_C) & (c2["keys"] >= MINKEYS)])
    dd = test_D(x, ref); out["D"] = set(dd.d[dd.replay])
    return out, dict(a1=a1, a2=a2, b1=b1, b2=b2, c1=c1, c2=c2, dd=dd)


# ---------------------------------------------------------------- full run with timing
timing = {"load_and_hash_keys": t_load}
res = {}
for name, f in [("A_day", lambda: test_A(df, ["d"])), ("A_plat", lambda: test_A(df, ["d", "p"])),
                ("B_day", lambda: test_B(df, ["d"])), ("B_plat", lambda: test_B(df, ["d", "p"])),
                ("C_day", lambda: test_C(df, ["d"])), ("C_plat", lambda: test_C(df, ["d", "p"])),
                ("D", lambda: test_D(df, comp))]:
    t = time.perf_counter(); res[name] = f(); timing[name] = time.perf_counter() - t

flags = {
    "A_day": set(res["A_day"].d), "A_plat": set(res["A_plat"].d),
    "B_day": set(res["B_day"].d[(res["B_day"].mu >= TAU_B) & (res["B_day"]["union"] >= MINKEYS)]),
    "B_plat": set(res["B_plat"].d[(res["B_plat"].mu >= TAU_B) & (res["B_plat"]["union"] >= MINKEYS)]),
    "C_day": set(res["C_day"].d[(res["C_day"].rho >= TAU_C) & (res["C_day"]["keys"] >= MINKEYS)]),
    "C_plat": set(res["C_plat"].d[(res["C_plat"].rho >= TAU_C) & (res["C_plat"]["keys"] >= MINKEYS)]),
    "D": set(res["D"].d[res["D"].replay]),
}
known = set(pd.to_datetime(pd.read_csv("results/reconcile_days.csv").d))

# Per day table
days = df.groupby("d").agg(rows=("h", "size"), n=("c", "sum"), jobs=("job", "nunique")).reset_index()
days = days.merge(res["C_day"][["d", "keys", "rho"]], on="d", how="left")
bmax = res["B_day"].groupby("d").mu.max().rename("mu_max").reset_index()
days = days.merge(bmax, on="d", how="left").fillna({"mu_max": 0.0})
dr = res["D"].groupby("d").apply(lambda z: (z.ns.sum() / z.nc.sum())).rename("ratio").reset_index()
days = days.merge(dr, on="d", how="left")
for k, v in flags.items():
    days[k] = days.d.isin(v)
days["known"] = days.d.isin(known)
days.to_csv("results/dup_days.csv", index=False)

# Flagged units on platform granularity: which platforms, how many records
fl = []
for name in ["A_plat", "B_plat", "C_plat"]:
    r = res[name]
    if name == "A_plat":
        z = r.groupby(["d", "p"], observed=True).agg(n=("n", "max")).reset_index()
    elif name == "B_plat":
        z = r[(r.mu >= TAU_B) & (r["union"] >= MINKEYS)].groupby(["d", "p"], observed=True).agg(n=("n_u", "max")).reset_index()
    else:
        z = r[(r.rho >= TAU_C) & (r["keys"] >= MINKEYS)][["d", "p"]].assign(n=np.nan)
    fl.append(z.assign(test=name))
z = res["D"][res["D"].replay][["d", "p", "nc"]].rename(columns={"nc": "n"}).assign(test="D")
fl.append(z)
flagged = pd.concat(fl, ignore_index=True)
flagged["known_day"] = flagged.d.isin(known)
flagged.to_csv("results/dup_flags.csv", index=False)


# ---------------------------------------------------------------- thinning null
def thin_split(x, p=0.5):
    """Merge all units of each day into cells, then split each cell binomially into two disjoint units."""
    cell = x.groupby(["d", "p", "h"], observed=True).c.sum().reset_index()
    c1 = rng.binomial(cell.c.values, p)
    u1 = cell.assign(c=c1, job="null_1")
    u2 = cell.assign(c=cell.c.values - c1, job="null_2")
    out = pd.concat([u1, u2], ignore_index=True)
    return out[out.c > 0]


NULL_REPS = 10
nulls, null_plat = [], []
for rep in range(NULL_REPS):
    z = thin_split(df)
    c1 = test_C(z, ["d"]).assign(rep=rep)
    b1 = test_B(z, ["d"])[["d", "mu"]]
    c1 = c1.merge(b1, on="d", how="left")
    c2 = test_C(z, ["d", "p"])
    c2 = c2[c2["keys"] >= MINKEYS]
    null_plat.append((len(c2), int((c2.rho >= TAU_C).sum())))
    c2 = c2.groupby("d").rho.max().rename("rho_plat_max").reset_index()
    nulls.append(c1.merge(c2, on="d", how="left"))
null = pd.concat(nulls, ignore_index=True)
# Analytic bound on the expected share under the null: E[rho] <= mean over keys of 1[c even] / sqrt(pi c / 2)
cell = df.groupby(["d", "h"]).c.sum().reset_index()
cell["b"] = np.where((cell.c % 2 == 0) & (cell.c >= 2), 1.0 / np.sqrt(np.pi * cell.c / 2.0), 0.0)
bound = cell.groupby("d").b.mean().rename("bound").reset_index()
null = null.merge(bound, on="d", how="left")
null.to_csv("results/dup_null.csv", index=False)

# ---------------------------------------------------------------- injection study on clean days
share_main = df.groupby(["d", "job"]).h.size().groupby(level=0).max() / df.groupby("d").h.size()
clean = sorted(set(share_main[share_main >= 0.99].index) - set().union(*flags.values()) - known)
R = 200
pick = rng.choice(len(clean), size=min(R, len(clean)), replace=False)
pick_days = [clean[i] for i in sorted(pick)]
by_day = {d: g for d, g in df[df.d.isin(pick_days)].groupby("d")}


def scenario(x, s):
    main = x.groupby("job").h.size().idxmax()
    m = x[x.job == main]
    rest = x[x.job != main]
    if s == "replay_k2":
        return pd.concat([x, m.assign(job="inj_1")])
    if s == "replay_k3":
        return pd.concat([x, m.assign(job="inj_1"), m.assign(job="inj_2")])
    if s == "partial_replay_20pct_platforms":
        plats = m.p.unique()
        sel = rng.choice(plats, size=max(1, int(round(0.2 * len(plats)))), replace=False)
        return pd.concat([x, m[m.p.isin(sel)].assign(job="inj_1")])
    if s == "replay_with_late_records":
        idx = rng.random(len(m)) < 0.02
        inc = rng.geometric(0.3, size=len(m))
        rerun = m.assign(job="inj_1", c=m.c.values + np.where(idx, inc, 0))
        late = m[idx].assign(job="inj_2", c=inc[idx])
        return pd.concat([x, rerun, late])
    if s == "control_split_no_replay":
        c1 = rng.binomial(m.c.values, 0.5)
        u1 = m.assign(c=c1); u2 = m.assign(job="inj_1", c=m.c.values - c1)
        return pd.concat([rest, u1[u1.c > 0], u2[u2.c > 0]])
    if s == "control_late_records":
        idx = rng.random(len(m)) < 0.02
        return pd.concat([x, m[idx].assign(job="inj_1", c=rng.geometric(0.3, size=int(idx.sum())))])
    raise ValueError(s)


SCEN = ["replay_k2", "replay_k3", "partial_replay_20pct_platforms", "replay_with_late_records",
        "control_split_no_replay", "control_late_records"]
rows = []
for s in SCEN:
    hits = {k: 0 for k in flags}
    for d in pick_days:
        x = scenario(by_day[d], s)
        x = x.assign(p=x.p.astype(str))
        f, _ = day_flags(x, comp[comp.d == d])
        for k in hits:
            hits[k] += int(d in f[k])
    rows.append({"scenario": s, "days": len(pick_days), **{k: hits[k] / len(pick_days) for k in hits}})
inj = pd.DataFrame(rows)
inj.to_csv("results/dup_injection.csv", index=False)

tim = pd.DataFrame([{"step": k, "seconds": v} for k, v in timing.items()])
tim.to_csv("results/dup_timing.csv", index=False)

# ---------------------------------------------------------------- summary
other = days[~days.known]
out = {
    "key_hash_collisions": int(collisions), "exact_confirmation": "sorted canonical observed-key/count multisets", "rows": int(N), "days": int(days.d.nunique()), "jobs": int(df.job.nunique()),
    "units_job_day": int(df.groupby(["job", "d"]).ngroups),
    "units_job_day_platform": int(df.groupby(["job", "d", "p"]).ngroups),
    "multi_job_days": int((days.jobs >= 2).sum()),
    "redownload_cells": int(chk[0]), "redownload_cells_differ": int(chk[1]),
    "known_days": sorted(str(x)[:10] for x in known),
    "flagged": {k: sorted(str(x)[:10] for x in v) for k, v in flags.items()},
    "recall_on_known": {k: len(v & known) / len(known) for k, v in flags.items()},
    "false_flags_other_days": {k: len(v - known) for k, v in flags.items()},
    "exact_replay_days": sorted(str(x)[:10] for x in flags["A_day"]),
    "near_replay_mu_known_min": float(days[days.known].mu_max.min()),
    "rho_known_min": float(days[days.known].rho.min()),
    "rho_other_max": float(other.rho.max()), "rho_other_p99": float(other.rho.quantile(0.99)),
    "mu_other_max": float(other.mu_max.max()),
    "null_rho_max": float(null.rho.max()), "null_rho_mean": float(null.rho.mean()),
    "null_mu_max": float(null.mu.max()), "null_rho_plat_max": float(null.rho_plat_max.max()),
    "null_bound_max": float(null.bound.max()), "null_bound_mean": float(null.bound.mean()),
    "null_draws": int(len(null)),
    "null_plat_units": int(sum(a for a, b in null_plat)), "null_plat_exceed": int(sum(b for a, b in null_plat)),
    "null_day_exceed_C": int((null.rho >= TAU_C).sum()), "null_day_exceed_B": int((null.mu >= TAU_B).sum()),
    "real_plat_units_C": int((res["C_plat"]["keys"] >= MINKEYS).sum()),
    "d_platform_days": int(len(res["D"])), "d_replay_platform_days": int(res["D"].replay.sum()),
    "d_replay_platform_days_known": int(res["D"][res["D"].replay].d.isin(known).sum()),
    "d_ratio_known_min": float(days[days.known].ratio.min()), "d_ratio_known_max": float(days[days.known].ratio.max()),
    "d_other_abs_dev_max": float((other.ratio - 1).abs().max()),
    "injection_days": len(pick_days), "clean_days": len(clean),
    "thresholds": {"tau_B": TAU_B, "tau_C": TAU_C, "min_keys": MINKEYS, "tol_D": TOL_D},
    "seconds_total_tests": float(sum(v for k, v in timing.items() if k != "load_and_hash_keys")),
    "seconds_load": float(t_load),
}
json.dump(out, open("results/dup.json", "w"), indent=2)
print(json.dumps(out, indent=2))
print(inj.to_string())
print(tim.to_string())
