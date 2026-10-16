"""Shared settings. Days on which the simple aggregate is exactly double the complete aggregate
(found by code/11_reconcile.py) are excluded from every day level analysis of the simple product."""
import os
import pandas as pd

def excluded_days():
    f = "results/reconcile_days.csv"
    if not os.path.exists(f):
        return []
    d = pd.read_csv(f)
    d = d[(d.n_complete > 0) & ((d.n_simple / d.n_complete - 2).abs() < 1e-4)]
    return [str(x)[:10] for x in d.d]

def sql_excl(col="d"):
    days = excluded_days()
    return "" if not days else f" AND {col}::DATE NOT IN (" + ", ".join(f"DATE '{x}'" for x in days) + ")"
