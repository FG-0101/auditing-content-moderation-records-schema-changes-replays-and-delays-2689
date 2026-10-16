#!/usr/bin/env python3
"""Run one analysis program under the numeric-thread environment run.py sets.

The shared anon5 driver passes on only PYTHONDONTWRITEBYTECODE.  The paper's own
run.py also pins the BLAS/OpenMP thread counts, keeps bytecode out of the tree
and gives matplotlib a private config directory, so that the numeric results are
reproducible.  This wrapper restores that environment and then replaces itself
with the requested program, so the CPU limit and niceness set by the driver are
inherited by the program rather than by this short-lived wrapper.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
env = dict(os.environ)
for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    env[name] = "2"
env["PYTHONUNBUFFERED"] = "1"
env["PYTHONDONTWRITEBYTECODE"] = "1"
env["MPLCONFIGDIR"] = str(ROOT / ".runtime" / "matplotlib")
env["ARCHIVE_LOCK"] = str(ROOT / "data" / "archive_snapshot_lock.json")
env["ARCHIVE_CACHE"] = str(ROOT / "data" / "raw" / "archive_cache")
env["PATH"] = (str(Path(sys.executable).parent) + os.pathsep
               + str(ROOT / "bin") + os.pathsep + env.get("PATH", ""))
os.execve(sys.executable, [sys.executable, "-u", "code/" + sys.argv[1]], env)
