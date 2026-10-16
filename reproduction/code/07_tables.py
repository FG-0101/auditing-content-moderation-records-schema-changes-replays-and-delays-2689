"""Write LaTeX tables and macros straight from the result files, so text and numbers cannot drift.
Run from the package root:  python3 code/07_tables.py"""
import json
import numpy as np
import pandas as pd

inv = json.load(open("data/derived/inventory.json"))
assert inv["partition_mismatch"] == 0 and inv["nonpositive_counts"] == 0, "Archive structural check failed: partition assignment or positive-count requirement"
q = json.load(open("results/quality.json"))
sc = json.load(open("results/schema.json"))
cp = json.load(open("results/capacity.json"))
reg = pd.read_csv("results/capacity_regressions.csv").set_index("model")
ev = pd.read_csv("results/capacity_event.csv").set_index("k")
rules = pd.read_csv("results/quality_rules.csv")
dst = pd.read_csv("results/schema_sexualized_destinations.csv")

def num(v, d):
    t = f"{abs(v):.{d}f}"
    return f"$-${t}" if v < 0 and float(t) != 0 else t

def big(x): return f"{int(x):,}".replace(",", "{,}")
def pct(x, d=1): return f"{100 * x:.{d}f}\\%"
def r(m, k, d=2): return f"{reg.loc[m, k]:.{d}f}"
M3, M4, M7 = "M3 log non automated on log inflow", "M4 log fully automated on log inflow", "M7 log not fully automated on log inflow within category"
M1, M2, M5, M6 = "M1 share fully automated on surge", "M2 share fully automated on log inflow", "M5 share fully automated on surge within category", "M6 placebo lead surge with own surge"
mac = {
 "nSoR": big(inv["sors"]), "nAggRows": big(inv["agg_rows"]), "nParts": big(inv["parquet_parts"]), "nPlat": inv["platforms"],
 "nDays": big(inv["days"]), "firstDay": pd.Timestamp(inv["first_day"]).strftime("%-d %B %Y"), "lastDay": pd.Timestamp(inv["last_day"]).strftime("%-d %B %Y"),
 "nSoRCut": big(q["total_records_to_2026_09_30"]), "topShare": pct(q["top1_share"]), "topFive": pct(q["top5_share"]),
 "hhi": f"{q['hhi']:.3f}", "nBig": q["platforms_ge_100k"], "nSingleAD": q["platforms_ge_100k_single_ad_value"],
 "nSingleADet": q["platforms_ge_100k_single_adet_value"], "nAnomDays": q["anomaly_platform_days"], "nAnomPlat": q["anomaly_platforms"],
 "anomShare": pct(q["anomaly_record_share"]), "nGapPlat": q["platforms_ge_100k_with_missing_days"],
 "nGapTen": q["platforms_ge_100k_missing_share_over_10pct"], "medGap": pct(q["median_missing_share_ge_100k"], 2),
 "oldOnCut": big(sc["old_category_records_on_cut_day"]), "nSexPlat": sc["sexualized_platforms_ge_10k_pre"],
 "nSexRec": big(sc["sexualized_records_pre_window"]), "nSexToCatch": sc["sexualized_top_gain_to_other_violation_tc"],
 "catchPre": pct(sc["catch_all_share_pre_excl_dominant"]), "catchPost": pct(sc["catch_all_share_post_excl_dominant"]),
 "capPlat": cp["platforms_in_sample"], "capDays": big(cp["platform_days"]), "capRec": big(cp["notice_decisions_in_sample"]),
 "nEvents": cp["surge_events"], "nEventPlat": cp["surge_platforms"], "nPos": cp["platforms_positive_slope_sig"],
 "nNeg": cp["platforms_negative_slope_sig"], "nSlopes": cp["platform_slopes"],
 "eHum": r(M3, "estimate"), "eHumLo": r(M3, "ci_low"), "eHumHi": r(M3, "ci_high"),
 "eAut": r(M4, "estimate"), "eAutLo": r(M4, "ci_low"), "eAutHi": r(M4, "ci_high"),
 "eHumC": r(M7, "estimate"), "eHumCLo": r(M7, "ci_low"), "eHumCHi": r(M7, "ci_high"),
 "evZero": f"{100 * ev.loc[0, 'dev_mean']:.1f}", "evZeroLo": num(100 * ev.loc[0, 'boot_lo'], 1), "evZeroHi": f"{100 * ev.loc[0, 'boot_hi']:.1f}",
 "pinPre": pct(dst.set_index("p").loc["Pinterest", "sex_share_pre"]), "pinGain": f"{100 * dst.set_index('p').loc['Pinterest', 'gain_to_catch_all']:.1f}",
 "rOneRec": big(rules.records[0]), "rTwoRec": big(rules.records[1]), "rThreeRec": big(rules.records[2]),
}
with open("tex/macros.tex", "w") as f:
    for k, v in mac.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# Table: consistency rules
lab = ["Fully automated decision, manual detection, own initiative", "Notice sourced, automated detection",
       "Illegal ground with a terms of service catch all category", "No restriction recorded", "Catch all category"]
with open("tex/tab_rules.tex", "w") as f:
    f.write("\\begin{tabular}{p{3.3cm}rrr}\\toprule\nRule & Records & Share & Platforms\\\\\\midrule\n")
    for l, (_, x) in zip(lab, rules.iterrows()):
        f.write(f"{l} & {big(x.records)} & {pct(x.share_of_all, 3)} & {int(x.platforms_with_any)}\\\\\n")
    f.write("\\bottomrule\\end{tabular}\n")

# Table: capacity regressions
rows = [(M1, "Fully automated share on surge", "Platform by weekday, month"),
        (M2, "Fully automated share on log reported decisions", "Platform by weekday, month"),
        (M3, "Log not automated on log reported decisions", "Platform by weekday, month"),
        (M4, "Log fully automated on log reported decisions", "Platform by weekday, month"),
        (M5, "Fully automated share on surge, within category", "Platform by category, platform by weekday, month"),
        (M7, "Log not fully automated on log reported decisions, within category", "Platform by category, platform by weekday, month"),
        (M6, "Placebo, next day surge", "Platform by weekday, month")]
with open("tex/tab_capacity.tex", "w") as f:
    f.write("\\begin{tabular}{p{3.0cm}rrrr}\\toprule\nSpecification & Estimate & SE & 95\\% CI & N\\\\\\midrule\n")
    for m, l, fe in rows:
        x = reg.loc[m]
        f.write(f"{l} & {num(x.estimate,3)} & {num(x.se,3)} & [{num(x.ci_low,2)}, {num(x.ci_high,2)}] & {big(x.nobs)}\\\\\n")
    f.write("\\bottomrule\\end{tabular}\n")
print("tables and macros written", len(mac))

# Additional macros from the complete aggregates
tl = json.load(open("results/timeliness.json"))
kw = json.load(open("results/keywords.json"))
rc = json.load(open("results/reconcile.json"))
dc = pd.read_csv("results/delay_capacity.csv")
kr = pd.read_csv("results/keywords_platform_recovery.csv").set_index("p")
def dcv(o, t, k, d=3): return num(float(dc[(dc.outcome == o) & (dc.term == t)][k].iloc[0]), d)
more = {
 "nSoRCorr": big(inv["sors"] - rc["diff_total"]), "nSoRCompl": big(rc["complete_total"]), "rcDiff": big(rc["diff_total"]),
 "rcDays": rc["days_disagree"], "rcMaxRel": pct(rc["max_abs_month_rel"]),
 "sameDay": pct(tl["share_same_day"]), "lateSeven": pct(tl["share_late7"]), "lateThirty": pct(tl["share_late30"]),
 "lateYear": big(tl["records_late365"]), "appAfter": big(tl["app_after_submit"]), "appFuture": big(tl["app_far_future"]),
 "contAfter": big(tl["content_after_app"]), "contSentinel": big(tl["content_2000_01_01"]),
 "nBigM": tl["platforms_ge_1m"], "nLateTen": tl["platforms_ge_1m_late7_over_10pct"], "medSame": pct(tl["platforms_ge_1m_median_sameday"]),
 "nContAfterPlat": tl["platforms_ge_100k_any_content_after_app"],
 "kwOne": pct(kw["kw_share_v1"]), "kwTwo": pct(kw["kw_share_v2"]), "kwRecHalf": kw["platforms_recovery_ge_half"],
 "kwRecTenth": kw["platforms_recovery_lt_tenth"], "kwPlat": kw["platforms"], "kwTopShare": pct(kw["top_sex_kw_category_share"]),
 "kwPreWithin": pct(kw["pre_kw_within_cat_weighted"]),
 "kwUnionPre": big(round(kw["union_daily_pre"])), "kwUnionPost": big(round(kw["union_daily_post"])),
 "pinRec": f"{kr.loc['Pinterest', 'recovery']:.2f}",
 "dSub": dcv("log_sub", "surge", "estimate"), "dSubLo": dcv("log_sub", "surge", "ci_low", 2), "dSubHi": dcv("log_sub", "surge", "ci_high", 2),
 "dMod": dcv("log_mod", "surge", "estimate"), "dModLo": dcv("log_mod", "surge", "ci_low", 2), "dModHi": dcv("log_mod", "surge", "ci_high", 2),
 "dSubLag": dcv("log_sub", "surge_lag_mean", "estimate"), "dModLag": dcv("log_mod", "surge_lag_mean", "estimate"),
 "dPlat": int(dc.platforms.iloc[0]), "dObs": big(dc.nobs.iloc[0]),
}
with open("tex/macros.tex", "a") as f:
    for k, v in more.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
with open("tex/tab_dates.tex", "w") as f:
    f.write("\\begin{tabular}{p{4.6cm}r}\\toprule\nDate rule & Records\\\\\\midrule\n")
    for l, v in [("Application date after submission date", tl["app_after_submit"]),
                 ("Application date in 2030 or later", tl["app_far_future"]),
                 ("Content date after application date", tl["content_after_app"]),
                 ("Content date equal to 1 January 2000", tl["content_2000_01_01"]),
                 ("Submitted more than 365 days after application", tl["records_late365"])]:
        f.write(f"{l} & {big(v)}\\\\\n")
    f.write("\\bottomrule\\end{tabular}\n")
print("extra macros", len(more))

# Enforcement design
ea = pd.read_csv("results/enforcement_att.csv").set_index("outcome")
er = pd.read_csv("results/enforcement_robust.csv")
ej = json.load(open("results/enforcement.json"))
with open("tex/tab_enforcement.tex", "w") as f:
    f.write("\\begin{tabular}{p{2.6cm}rrrrr}\\toprule\nOutcome & TWFE & did2s & RI $p$ & Control mean\\\\\\midrule\n".replace("rrrrr", "rrrr"))
    for y in ["y_catchall", "y_human", "y_same", "y_late7", "y_notice", "y_baddate", "y_logn"]:
        r = ea.loc[y]
        f.write(f"{r.label} & {num(r.twfe,3)} ({num(r.twfe_se,3)}) & {num(r.did2s,3)} ({num(r.did2s_se,3)}) & {r.ri_p:.3f} & {num(r.control_mean,3)}\\\\\n")
    f.write("\\bottomrule\\end{tabular}\n")
def rb(spec, y, k): return float(er[(er.spec == spec) & (er.outcome == y)][k].iloc[0])
loo = er[(er.outcome == "y_catchall") & er.spec.str.startswith("Without") & (er.spec != "Without Google Shopping")]
emac = {"enT": len(ej["treated"]), "enC": len(ej["controls"]), "enW": ej["weeks"], "enObs": big(ej["obs"]),
        "caTw": num(ea.loc["y_catchall", "twfe"], 3), "caTwP": f"{ea.loc['y_catchall','twfe_p']:.3f}",
        "caDs": num(ea.loc["y_catchall", "did2s"], 3), "caDsP": f"{ea.loc['y_catchall','did2s_p']:.3f}",
        "caRi": f"{ea.loc['y_catchall','ri_p']:.3f}", "caRiN": int(ea.loc["y_catchall", "ri_draws"]),
        "caLooMin": num(loo.est.min(), 3), "caLooMax": num(loo.est.max(), 3), "caLooPmax": f"{loo.p.max():.3f}",
        "caPlac": num(rb("Placebo, dates 26 weeks earlier, pre period only", "y_catchall", "est"), 3),
        "caPlacP": f"{rb('Placebo, dates 26 weeks earlier, pre period only', 'y_catchall', 'p'):.3f}",
        "caW": num(rb("Weighted by records", "y_catchall", "est"), 3), "caWP": f"{rb('Weighted by records', 'y_catchall', 'p'):.3f}",
        "huDs": num(ea.loc["y_human", "did2s"], 3), "huDsP": f"{ea.loc['y_human','did2s_p']:.3f}", "huRi": f"{ea.loc['y_human','ri_p']:.3f}",
        "huW": num(rb("Weighted by records", "y_human", "est"), 3), "huWP": f"{rb('Weighted by records', 'y_human', 'p'):.3f}",
        "huPlac": num(rb("Placebo, dates 26 weeks earlier, pre period only", "y_human", "est"), 3)}
ev = pd.read_csv("results/enforcement_event.csv"); c = ev[ev.outcome == "y_catchall"]
emac.update({"caPreMean": num(c[c.k < 0].est.mean(), 3), "caWkTwfour": num(float(c[c.k == 24].est.iloc[0]), 3)})
with open("tex/macros.tex", "a") as f:
    for k_, v in emac.items():
        f.write(f"\\newcommand{{\\{k_}}}{{{v}}}\n")
print("enforcement macros", len(emac))

# Replay detection (code/14_duplicates.py)
dj = json.load(open("results/dup.json"))
inj = pd.read_csv("results/dup_injection.csv").set_index("scenario")
tim = pd.read_csv("results/dup_timing.csv").set_index("step")
nk = len(dj["known_days"])
dmac = {"dupRows": big(dj["rows"]), "dupJobs": dj["jobs"], "dupUnits": big(dj["units_job_day"]),
        "dupUnitsP": big(dj["units_job_day_platform"]), "dupMultiDays": dj["multi_job_days"], "dupDays": big(dj["days"]),
        "dupExact": len(dj["exact_replay_days"]), "dupKnown": nk,
        "dupMuMin": f"{dj['near_replay_mu_known_min']:.3f}", "dupRhoMin": f"{dj['rho_known_min']:.3f}",
        "dupRhoOtherMax": f"{dj['rho_other_max']:.4f}", "dupMuOtherMax": f"{dj['mu_other_max']:.2f}",
        "dupNullRhoMax": f"{dj['null_rho_max']:.3f}", "dupNullRhoMean": f"{dj['null_rho_mean']:.3f}",
        "dupNullBoundMean": f"{dj['null_bound_mean']:.3f}", "dupNullBoundMax": f"{dj['null_bound_max']:.3f}",
        "dupNullDraws": big(dj["null_draws"]), "dupNullPlatUnits": big(dj["null_plat_units"]),
        "dupNullPlatExceed": big(dj["null_plat_exceed"]), "dupNullPlatRate": f"{dj['null_plat_exceed'] / dj['null_plat_units']:.5f}",
        "dupNullPlatMax": f"{dj['null_rho_plat_max']:.2f}",
        "dupDPlatDays": big(dj["d_platform_days"]), "dupDReplay": big(dj["d_replay_platform_days"]),
        "dupDRatioMax": f"{dj['d_ratio_known_max']:.6f}", "dupInjDays": dj["injection_days"], "dupCleanDays": big(dj["clean_days"]),
        "dupSecTests": f"{dj['seconds_total_tests']:.1f}", "dupSecLoad": f"{dj['seconds_load']:.1f}",
        "dupRedl": big(dj["redownload_cells"]), "dupRedlDiff": dj["redownload_cells_differ"],
        "tauB": f"{dj['thresholds']['tau_B']:.1f}", "tauC": f"{dj['thresholds']['tau_C']:.1f}", "minKeys": dj["thresholds"]["min_keys"],
        "tolD": f"{dj['thresholds']['tol_D']:.2f}"}
with open("tex/macros.tex", "a") as f:
    for k_, v in dmac.items():
        f.write(f"\\newcommand{{\\{k_}}}{{{v}}}\n")
lab = [("A_day", "A, exact, day", "Provenance"), ("A_plat", "A, exact, platform day", "Provenance"),
       ("B_day", "B, near, day", "Provenance"), ("B_plat", "B, near, platform day", "Provenance"),
       ("C_day", "C, equal counts, day", "None"), ("C_plat", "C, equal counts, platform day", "None"),
       ("D", "D, ratio, platform day", "Second product")]
scen = ["replay_k2", "partial_replay_20pct_platforms", "replay_with_late_records", "control_split_no_replay", "control_late_records"]
with open("tex/tab_dup.tex", "w") as f:
    f.write("\\begin{tabular}{p{3.2cm}lrrrrrrrr}\\toprule\n"
            " & & \\multicolumn{2}{c}{Observed days} & \\multicolumn{5}{c}{Injected into %d clean days} & \\\\\n"
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-9}\n"
            "Test & Needs & Doubled & Other & Full & Partial & With late & Split & Late only & Seconds\\\\\\midrule\n" % dj["injection_days"])
    for k_, l, need in lab:
        found = len(set(dj["flagged"][k_]) & set(dj["known_days"]))
        f.write(f"{l} & {need} & {found} of {nk} & {dj['false_flags_other_days'][k_]} & "
                + " & ".join(f"{inj.loc[s, k_]:.3f}" for s in scen) + f" & {tim.loc[k_, 'seconds']:.2f}\\\\\n")
    f.write("\\bottomrule\\end{tabular}\n")
print("replay macros", len(dmac))

# Summary of all checks, generated from the result files
summ = [
    ("Reconciliation of the two products", "Both", f"{rc['days_disagree']} days doubled, {big(rc['diff_total'])} records"),
    ("Replay of a daily file, no reference", "Simple", f"{len(dj['flagged']['C_day'])} days flagged, {dj['false_flags_other_days']['C_day']} false"),
    ("Fully automated decision, manual detection", "Simple", f"{big(rules.records[0])} records"),
    ("Notice sourced, automated detection", "Simple", f"{big(rules.records[1])} records"),
    ("Illegal ground, terms of service category", "Simple", f"{big(rules.records[2])} records"),
    ("Degenerate automation field", "Simple", f"{q['platforms_ge_100k_single_ad_value']} of {q['platforms_ge_100k']} platforms"),
    ("Volume burst above ten times median", "Simple", f"{big(q['anomaly_platform_days'])} platform days"),
    ("Application date after submission", "Complete", f"{big(tl['app_after_submit'])} records"),
    ("Content date after application date", "Complete", f"{big(tl['content_after_app'])} records"),
    ("Placeholder content date", "Complete", f"{big(tl['content_2000_01_01'])} records"),
    ("Submitted more than seven days late", "Complete", f"{pct(tl['share_late7'])} of records"),
    ("Retired category after migration", "Simple", f"{big(json.load(open('results/schema.json'))['old_category_records_after_cut'])} records"),
    ("Sexual keyword/category share ratio below one tenth", "Complete", f"{kw['platforms_recovery_lt_tenth']} of {kw['platforms']} platforms"),
]
with open("tex/tab_summary.tex", "w") as f:
    f.write("\\begin{tabular}{p{3.6cm}lp{2.6cm}}\\toprule\nCheck & Product & Result\\\\\\midrule\n")
    for a_, b_, c_ in summ:
        f.write(f"{a_} & {b_} & {c_}\\\\\n")
    f.write("\\bottomrule\\end{tabular}\n")
print("summary rows", len(summ))

# Date lists for the replay paragraph, generated so the text cannot drift from results/dup.json
def dlist(xs):
    t = [pd.Timestamp(x).strftime("%-d %B %Y") for x in sorted(xs)]
    return t[0] if len(t) == 1 else ", ".join(t[:-1]) + " and " + t[-1]
near = sorted(set(dj["known_days"]) - set(dj["exact_replay_days"]))
with open("tex/macros.tex", "a") as f:
    f.write(f"\\newcommand{{\\dupExactList}}{{{dlist(dj['exact_replay_days'])}}}\n")
    f.write(f"\\newcommand{{\\dupNearList}}{{{dlist(near)}}}\n")

# Territorial scope, automation and timeliness, account sanctions, weekend supply (code/16_extensions.py)
ex = json.load(open("results/ext.json"))
s_, a_, c_, w_ = ex["scope"], ex["autotime"], ex["accounts"], ex["weekend"]
DAYN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
dsh = w_["dow_share"]
def rr(blk, term, k, d=3): return num(blk[term][k], d)
xmac = {"scPartial": pct(s_["partial_share"]), "scSingle": pct(s_["single_share"]),
        "scCa": s_["top_countries"][0][0], "scCaS": pct(s_["top_countries"][0][1]),
        "scCb": s_["top_countries"][1][0], "scCbS": pct(s_["top_countries"][1][1]),
        "scCc": s_["top_countries"][2][0], "scCcS": pct(s_["top_countries"][2][1]),
        "scBig": s_["big_platforms"], "scZero": s_["big_partial_below_tenth_pct"], "scHalf": s_["big_partial_over_half"],
        "atCells": big(a_["cells"]), "atPlat": a_["platforms"],
        "atSame": rr(a_["reg"]["same"], "full", "est"), "atSameSE": rr(a_["reg"]["same"], "full", "se"),
        "atMod": num(abs(a_["reg"]["modsame"]["full"]["est"]), 3), "atModSE": rr(a_["reg"]["modsame"], "full", "se"),
        "atPlatN": a_["plat_n"], "atFaster": a_["plat_full_faster"], "atSlower": a_["plat_full_slower"],
        "acRec": big(c_["acct_records"]), "acPlat": c_["platforms"], "acMore": c_["more_human_for_accounts"],
        "acLess": c_["less_human_for_accounts"],
        "wkPlat": w_["platforms"], "wkDays": big(w_["platform_days"]), "wkDlDays": big(w_["delay_platform_days"]),
        "wkShare": rr(w_["reg"]["share"], "weekend", "est"), "wkShareSE": rr(w_["reg"]["share"], "weekend", "se"),
        "wkShareP": f"{w_['reg']['share']['weekend']['p']:.2f}",
        "wkShareCtrl": rr(w_["reg"]["share_ctrl"], "weekend", "est"), "wkShareCtrlP": f"{w_['reg']['share_ctrl']['weekend']['p']:.2f}",
        "wkHol": rr(w_["reg"]["share"], "holiday", "est"), "wkHolP": f"{w_['reg']['share']['holiday']['p']:.2f}",
        "wkInflow": num(abs(w_["reg"]["inflow"]["weekend"]["est"]), 2), "wkInflowP": f"{w_['reg']['inflow']['weekend']['p']:.2f}",
        "wkHolInflow": num(abs(w_["reg"]["inflow"]["holiday"]["est"]), 2), "wkHolInflowP": f"{w_['reg']['inflow']['holiday']['p']:.2f}",
        "wkMod": rr(w_["reg"]["same_day_moderation"], "weekend", "est"), "wkModP": f"{w_['reg']['same_day_moderation']['weekend']['p']:.2f}",
        "wkDowMinDay": DAYN[int(np.argmin(dsh))], "wkDowMin": pct(min(dsh)),
        "wkDowMaxDay": DAYN[int(np.argmax(dsh))], "wkDowMax": pct(max(dsh))}
import numpy as np
with open("tex/macros.tex", "a") as f:
    for k_, v in xmac.items():
        f.write(f"\\newcommand{{\\{k_}}}{{{v}}}\n")
print("extension macros", len(xmac))

# Ingestion facts and the title figure
ck = pd.read_csv("data/derived/complete/checksums.csv", header=None)
with open("tex/macros.tex", "a") as f:
    f.write(f"\\newcommand{{\\nArch}}{{{len(ck)}}}\n")
    f.write(f"\\newcommand{{\\nArchOK}}{{{int(ck[3].astype(str).eq('True').sum())}}}\n")
    f.write(f"\\newcommand{{\\nSoRBn}}{{{(inv['sors'] - rc['diff_total']) / 1e9:.1f}}}\n")
    f.write(f"\\newcommand{{\\rcDayList}}{{{dlist(rc['days_disagree_list'])}}}\n")

# p value written for running text
_p = rb("Weighted by records", "y_human", "p")
with open("tex/macros.tex", "a") as f:
    f.write("\\newcommand{\\huWPs}{" + ("p<0.001" if _p < 0.001 else f"p={_p:.3f}") + "}\n")

# Magnitudes for sentences that already state the direction ("falls by")
with open("tex/macros.tex", "a") as f:
    f.write(f"\\newcommand{{\\huWabs}}{{{abs(rb('Weighted by records', 'y_human', 'est')):.3f}}}\n")
    f.write(f"\\newcommand{{\\huDsAbs}}{{{abs(ea.loc['y_human', 'did2s']):.3f}}}\n")

# Render the platform timeliness table from the recomputed weighted counts.
platform_timeliness = pd.read_csv("results/timeliness_platforms.csv").sort_values("n", ascending=False).head(12)
with open("tex/tab_timeliness.tex", "w") as f:
    f.write("\\begin{tabular}{lrrrr}\\toprule\n")
    f.write("Platform & Records & Same day & Over 7 days & Content after decision\\\\\\midrule\n")
    for _, row in platform_timeliness.iterrows():
        f.write(f"{row.p} & {big(row.n)} & {pct(row.sameday, 1)} & {pct(row.late7, 1)} & {pct(row.content_after_app / row.n, 2)}\\\\\n")
    f.write("\\bottomrule\\end{tabular}\n")

from layout_tables import format_generated_tables
format_generated_tables('P01')
