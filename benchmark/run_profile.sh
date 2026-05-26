#!/usr/bin/env bash
# Profile the smoke planner run under cProfile, scalene, and py-spy.
#
# Each profiler runs independently. If one fails (e.g. py-spy needs sudo on
# macOS), we log the failure and continue with the next profiler so a partial
# result is still useful. Outputs land in benchmark/results/.

set -euo pipefail

REPO_ROOT="/Users/aangeles/robopiano"
BENCH_DIR="${REPO_ROOT}/benchmark"
RESULTS_DIR="${BENCH_DIR}/results"
LOG_FILE="${RESULTS_DIR}/profile_log.txt"

mkdir -p "${RESULTS_DIR}"

# shellcheck disable=SC1091
source "${BENCH_DIR}/activate_env.sh"

python "${BENCH_DIR}/gen_test_midi.py"

: > "${LOG_FILE}"

log() {
    # Single-arg logger so we don't have to fight bash 3.2 quoting.
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $1" | tee -a "${LOG_FILE}"
}

PLANNER_SCRIPT="${REPO_ROOT}/Impromptu/scripts/plan_trajectory.py"
OUTPUT_ROOT="/tmp/maestroso_smoke"
RUN_NAME="twinkle_smoke"

# Common planner arguments (kept as positional args so we can reuse them).
PLANNER_ARGS=(
    "--midi-path" "/tmp/twinkle.mid"
    "--output-root" "${OUTPUT_ROOT}"
    "--run-name" "${RUN_NAME}"
    "--environment-name" "RoboPianist-debug-TwinkleTwinkleLittleStar-v0"
    "--trajectory-mode" "joint_space_straighten"
    "--disable-adaptive-complex-song-defaults"
    "--max-duration-s" "8.0"
    "--active-window-last-s" "5.0"
    "--key-press-depth" "0.006"
    "--wrong-hand-penalty" "4.0"
    "--wrong-hand-split-key" "48"
    "--assignment-dynamic-hand-split"
    "--assignment-strategy" "legacy_previous_pose"
    "--assignment-fail-if-unassigned"
    "--anchor-stride" "2"
    "--ik-max-nfev" "80"
    "--residual-success-threshold" "0.02"
    "--ik-static-contact-validation"
    "--ik-static-contact-settle-steps" "1"
    "--disable-ik-multistart-on-failure"
)

# ---------- cProfile ----------
log "cProfile: start"
CPROF_FILE="${RESULTS_DIR}/profile.cprof"
CPROF_TOP="${RESULTS_DIR}/profile_top40.txt"
CPROF_STDOUT="${RESULTS_DIR}/cprofile_stdout.log"
CPROF_STDERR="${RESULTS_DIR}/cprofile_stderr.log"
: > "${CPROF_STDOUT}"
: > "${CPROF_STDERR}"

if ! python -m cProfile -o "${CPROF_FILE}" "${PLANNER_SCRIPT}" "${PLANNER_ARGS[@]}" \
        >>"${CPROF_STDOUT}" 2>>"${CPROF_STDERR}"; then
    log "cProfile: planner run failed: $?"
else
    if ! python -c "import pstats; p=pstats.Stats('${CPROF_FILE}'); p.sort_stats('cumulative').print_stats(40)" \
            > "${CPROF_TOP}" 2>>"${CPROF_STDERR}"; then
        log "cProfile: pstats dump failed: $?"
    fi
fi
log "cProfile: end"

# ---------- scalene ----------
log "scalene: start"
SCALENE_OUT="${RESULTS_DIR}/scalene_profile.html"
SCALENE_STDOUT="${RESULTS_DIR}/scalene_stdout.log"
SCALENE_STDERR="${RESULTS_DIR}/scalene_stderr.log"
: > "${SCALENE_STDOUT}"
: > "${SCALENE_STDERR}"

if ! scalene --outfile "${SCALENE_OUT}" --html --reduced-profile --no-browser \
        "${PLANNER_SCRIPT}" "${PLANNER_ARGS[@]}" \
        >>"${SCALENE_STDOUT}" 2>>"${SCALENE_STDERR}"; then
    log "scalene: failed: $?"
fi
log "scalene: end"

# ---------- py-spy ----------
log "py-spy: start"
PYSPY_OUT="${RESULTS_DIR}/py_spy_flame.svg"
PYSPY_STDOUT="${RESULTS_DIR}/py_spy_stdout.log"
PYSPY_STDERR="${RESULTS_DIR}/py_spy_stderr.log"
: > "${PYSPY_STDOUT}"
: > "${PYSPY_STDERR}"

if ! py-spy record -o "${PYSPY_OUT}" -d 120 --rate 100 -- \
        python "${PLANNER_SCRIPT}" "${PLANNER_ARGS[@]}" \
        >>"${PYSPY_STDOUT}" 2>>"${PYSPY_STDERR}"; then
    PYSPY_RC=$?
    log "py-spy: failed: ${PYSPY_RC} (likely needs codesign/sudo on macOS, skipping)"
fi
log "py-spy: end"

log "all profilers finished"
