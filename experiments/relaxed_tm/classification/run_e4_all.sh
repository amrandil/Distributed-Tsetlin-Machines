#!/usr/bin/env bash
# E4 (Binary Iris): every run, one after the other, meant to run overnight.
#
#   bash experiments/relaxed_tm/classification/run_e4_all.sh
#
# From the repository root. Uses ./venv if it exists, otherwise the active Python.
# 1. A quick check (about a minute): every script runs a few short runs, nothing
#    is saved. If anything fails here, it stops before the long runs.
# 2. The real runs, in order of importance:
#      1  Granmo's Binary Iris demo, his way (1,000 ensembles)       part 1
#      2  relaxed TM, binary encoding: tuning pass                   part 2
#      3  relaxed TM, binary encoding: 1,000 final runs              part 2
#      4  standard TM (his code, our harness), binary, 1,000 runs    part 2, paired comparison
#      5  relaxed TM, thermometer encoding: tuning pass              part 3
#      6  relaxed TM, thermometer encoding: 1,000 final runs         part 3
#      7  standard TM (his code, our harness), thermometer, 1,000    part 3
#    A step whose results file already exists is skipped, so after an
#    interruption the same command continues where it stopped. FORCE=1 reruns
#    everything. If a step fails, the others still run (except a final run
#    whose tuning pass failed).
#
# Options (environment variables): WORKERS=8 (parallel runs, default 4), FORCE=1.
# Output: printed here and saved to experiments/relaxed_tm/classification/logs/.
# On macOS the Mac is kept awake (caffeinate) while it runs; keep it plugged in
# and the lid open.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$ROOT"

# keep the Mac awake for the whole run
if [[ "$(uname)" == "Darwin" && -z "${E4_CAFFEINATED:-}" ]] && command -v caffeinate >/dev/null; then
    export E4_CAFFEINATED=1
    exec caffeinate -i bash "$0" "$@"
fi

if [[ -f venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
fi
WORKERS="${WORKERS:-4}"
FORCE="${FORCE:-0}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
PKG="experiments.relaxed_tm.classification"
RES="$HERE/results"
mkdir -p "$HERE/logs" "$RES"
LOG="$HERE/logs/e4_$(date +%Y%m%d_%H%M%S).log"

say() { echo "$@" | tee -a "$LOG"; }
run() { "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }
hms() { printf '%dh%02dm' $(($1 / 3600)) $(($1 % 3600 / 60)); }

say "E4 Binary Iris, started $(date '+%Y-%m-%d %H:%M'); python: $(command -v python); workers: $WORKERS"
say "Log: $LOG"

# ---------------------------------------------------------------- 1. quick check
say ""
say "=== Quick check (short runs, nothing saved) ==="
check_ok=1
if ! run python -c "import torch, Cython, sklearn, numpy, setuptools, distutils.extension; print('packages ok: torch', torch.__version__, '| Cython', Cython.__version__)"; then
    say ""
    say "QUICK CHECK FAILED: a package is missing (see above). Install with: pip install -r requirements.txt"
    exit 1
fi
run python -m $PKG.granmo2018.april2018.repeat_iris_demo --ensembles 1 || check_ok=0
for enc in binary thermometer; do
    run python -m $PKG.run_e4_baseline --encoding "$enc" --runs 2 --epochs 5 --workers 2 --no-save || check_ok=0
    run python -m $PKG.run_e4_relaxed --encoding "$enc" --mode final --tau 10 --init-mean -1.5 --lr 0.05 \
        --runs 2 --epochs 5 --workers 2 --no-save || check_ok=0
done
if [[ $check_ok -ne 1 ]]; then
    say ""
    say "QUICK CHECK FAILED: see the error above. Nothing long was started."
    exit 1
fi
say "Quick check passed. Starting the real runs; you can leave it now."

# ---------------------------------------------------------------- 2. real runs
declare -a SUMMARY=()
total_start=$SECONDS

step() {  # step <number> <title> <results file> <command...>
    local n="$1" title="$2" file="$3"; shift 3
    say ""
    if [[ "$FORCE" != "1" && -f "$RES/$file" ]]; then
        say "=== Step $n/7: $title: already done ($file exists), skipped ==="
        SUMMARY+=("step $n  skipped (already done)  $title")
        return
    fi
    say "=== Step $n/7: $title  [$(date '+%H:%M')] ==="
    local t0=$SECONDS
    if run "$@"; then
        SUMMARY+=("step $n  done in $(hms $((SECONDS - t0)))     $title")
    else
        say "!!! Step $n FAILED (see above); continuing with the next step"
        SUMMARY+=("step $n  FAILED                $title")
    fi
}

step 1 "Granmo's demo, his way (part 1)" e4_granmo2018_demo_binary_iris.json \
    python -m $PKG.granmo2018.april2018.repeat_iris_demo
step 2 "relaxed TM, binary: tuning pass" e4_relaxed_tune_binary.json \
    python -m $PKG.run_e4_relaxed --encoding binary --mode tune --workers "$WORKERS"
step 3 "relaxed TM, binary: final runs (part 2)" e4_relaxed_final_binary.json \
    python -m $PKG.run_e4_relaxed --encoding binary --mode final --workers "$WORKERS"
step 4 "standard TM, binary, our harness" e4_standard_tm_binary.json \
    python -m $PKG.run_e4_baseline --encoding binary --workers "$WORKERS"
step 5 "relaxed TM, thermometer: tuning pass" e4_relaxed_tune_thermometer.json \
    python -m $PKG.run_e4_relaxed --encoding thermometer --mode tune --workers "$WORKERS"
step 6 "relaxed TM, thermometer: final runs (part 3)" e4_relaxed_final_thermometer.json \
    python -m $PKG.run_e4_relaxed --encoding thermometer --mode final --workers "$WORKERS"
step 7 "standard TM, thermometer, our harness" e4_standard_tm_thermometer.json \
    python -m $PKG.run_e4_baseline --encoding thermometer --workers "$WORKERS"

say ""
say "=== E4 finished $(date '+%Y-%m-%d %H:%M'), total $(hms $((SECONDS - total_start))) ==="
for line in "${SUMMARY[@]}"; do say "  $line"; done
say "Results: $RES/e4_*.json   Log: $LOG"
