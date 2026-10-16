Auditing 42.5 Billion Content Moderation Records
reproduction

Requirements
  CPython 3.14 on a POSIX system with the versions pinned in ENVIRONMENT.json
  (duckdb 1.5.5, matplotlib 3.11.2, numpy 2.5.3, pandas 3.0.5, pyarrow 26.0.0,
  pyfixest 0.60.0, scipy 1.18.1 and the rest of the pinned set).  The package is
  self-contained: every input is included under data/ (120 MB) and no network
  access is used.  The two acquisition steps that would rebuild data/ from the
  publisher archives (code/01_fetch.sh and code/08_complete.py) are not part of
  this package.

Command
  python3.14 -B reproduce.py

  The command copies code/, data/, bin/, results/, figures/ and tex/ into
  runs/<UTC timestamp>/ and runs the fifteen analysis programs there, in the
  dependency order the paper's run.py uses: the aggregate build, the two-product
  reconciliation, replay detection and its figure, the consistency checks, the
  capacity analysis, the schema migration analysis, timeliness, keywords,
  enforcement and its robustness figure, the extensions, the routing baseline,
  the figures and the tables.  Each program runs under the numeric-thread policy
  of the release, with the OMP, OpenBLAS, MKL, VECLIB and NUMEXPR thread counts
  all set to 2 and a private MPLCONFIGDIR, so that floating-point results are
  reproducible.  The shipped tree is never modified.  Each stage has a
  300-second CPU limit; use --cpu-limit 1800 if the machine is slow.

Comparison
  Every result file under runs/.../results/ is compared with expected.json,
  which holds the corresponding reference result.  This follows the policy the
  paper states: query timings and low-order floating solver differences can
  vary, while integer counts, scientific values at printed precision and query
  memberships are the comparison targets.  Execution records, environment
  fields, absolute paths and wall-clock timings are dropped before comparison;
  integers and text must then agree exactly and JSON floats agree to a relative
  1e-9.  A CSV is compared as a value-sorted row multiset, because row order is
  not part of the contract, and its numeric cells are compared at three decimal
  places, the precision at which the paper prints them.  Three decimals is still
  exact for every integer count; it is what keeps a regression standard error
  that differs only in its eighth significant digit from reading as a
  discrepancy, while a change the paper could show is still caught.  PASS and
  CHECK.json indicate a full match.

Excluded from the comparison
  results/dup_timing.csv is a per-step wall-clock timing table (the analysis
  steps and the seconds each took); it is a timing artifact, not a scientific
  result, and is excluded by name.  The two wall-clock fields of results/dup.json
  (seconds_load, seconds_total_tests) are dropped by the field filter, so
  results/dup.json is still compared on its content.  The release's execution
  records under reproduction_logs/ (the per-step .log files, environment.json
  and execution.json) are environment and timing records; they are neither
  shipped nor compared.  The generated figures/*.pdf and tex/*.tex are renderings
  rather than result files, and a PDF's vector coordinates vary in their sixth
  decimal; the numbers they render are compared through results/, so a coordinate
  or a table layout cannot mask or fake a numerical change.

Expected counts
  42,633,487,544 simple-product records against 42,535,834,864 complete-product
  records, a difference of 97,652,680 over six doubled days; 274 platforms; 15
  analysis programs; 47 compared result files under results/ (10 JSON and 37
  CSV).

Data
  data/agg/ holds the official simple-product aggregate, hive partitioned by
  month; data/raw/ holds the daily simple-product archives and the pinned archive
  snapshot lock; data/e derived/ holds the reductions built from them, including
  the complete-product tables under derived/complete/.  The paper tree's
  docs/input_manifest.csv lists their checksums.  Sources and terms are in
  SOURCE.txt and LICENSE.txt.

Files
  reproduce.py      the driver
  expected.json     expected result content, keyed by run-relative path
  code/             the analysis programs and the run_step.py environment shim
  data/             the reduced inputs
  bin/              the sha1sum shim the release PATH expects
  RESULTS.tsv       paper item to program, line, output and input
  ENVIRONMENT.json  the recorded release environment
