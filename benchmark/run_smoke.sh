#!/usr/bin/env bash
# Smoke test for the Impromptu planner on a tiny Twinkle Twinkle MIDI.
#
# - Activates the conda env via benchmark/activate_env.sh (set up separately).
# - Ensures /tmp/twinkle.mid exists by calling gen_test_midi.py.
# - Runs plan_trajectory.py with the canonical short-song settings.
# - Captures wall-clock time, stdout, and stderr into benchmark/results/.
#
# Exits non-zero (preserving the planner's exit code) if the planner fails.

set -euo pipefail

REPO_ROOT="/Users/aangeles/robopiano"
BENCH_DIR="${REPO_ROOT}/benchmark"
RESULTS_DIR="${BENCH_DIR}/results"

mkdir -p "${RESULTS_DIR}"

# shellcheck disable=SC1091
source "${BENCH_DIR}/activate_env.sh"

python "${BENCH_DIR}/gen_test_midi.py"

OUTPUT_ROOT="/tmp/maestroso_smoke"
RUN_NAME="twinkle_smoke"

STDOUT_LOG="${RESULTS_DIR}/smoke_stdout.log"
STDERR_LOG="${RESULTS_DIR}/smoke_stderr.log"
TIME_FILE="${RESULTS_DIR}/smoke_wall_time.txt"

# Clear stale logs so partial outputs from prior runs cannot mislead.
: > "${STDOUT_LOG}"
: > "${STDERR_LOG}"
: > "${TIME_FILE}"

echo "[run_smoke] starting smoke planner run at $(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    | tee -a "${STDOUT_LOG}"

# /usr/bin/time -p emits real/user/sys on stderr in a portable POSIX format.
# We capture the planner's stdout/stderr separately and append the timing
# block to TIME_FILE.
set +e
/usr/bin/time -p \
    python "${REPO_ROOT}/Impromptu/scripts/plan_trajectory.py" \
        --midi-path /tmp/twinkle.mid \
        --output-root "${OUTPUT_ROOT}" --run-name "${RUN_NAME}" \
        --environment-name RoboPianist-debug-TwinkleTwinkleLittleStar-v0 \
        --trajectory-mode joint_space_straighten \
        --disable-adaptive-complex-song-defaults \
        --max-duration-s 8.0 --active-window-last-s 5.0 \
        --key-press-depth 0.006 --wrong-hand-penalty 4.0 --wrong-hand-split-key 48 \
        --assignment-dynamic-hand-split \
        --assignment-strategy legacy_previous_pose --assignment-fail-if-unassigned \
        --anchor-stride 2 \
        --ik-max-nfev 80 --residual-success-threshold 0.02 \
        --ik-static-contact-validation --ik-static-contact-settle-steps 1 \
        --disable-ik-multistart-on-failure \
        >>"${STDOUT_LOG}" 2>>"${STDERR_LOG}"
PLANNER_RC=$?
set -e

# /usr/bin/time writes its three-line summary to stderr. Pull the trailing
# real/user/sys block out of the planner's stderr log and into TIME_FILE.
# (We grep at the end because /usr/bin/time appends after the planner exits.)
{
    echo "planner_exit_code=${PLANNER_RC}"
    echo "wall_time_section:"
    grep -E '^(real|user|sys)[[:space:]]' "${STDERR_LOG}" | tail -n 3 || true
} > "${TIME_FILE}"

echo "[run_smoke] finished at $(date -u +%Y-%m-%dT%H:%M:%SZ) rc=${PLANNER_RC}" \
    | tee -a "${STDOUT_LOG}"

exit "${PLANNER_RC}"
