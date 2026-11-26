#!/usr/bin/env bash
# verify.sh -- ONE command: runs the whole artifact and prints PASS/FAIL lines.
#
#   ./verify.sh            # run everything
#   ./verify.sh -q         # same, but only the summary lines
#
# Exit code 0 iff every component passed.  Per-program logs land in ./last_run/.
#
# Note: tensor_verify.py and strassen.py deliberately print "FAIL: ..." lines as
# part of the corrupted-decomposition rejection tests.  Those are expected, so
# pass/fail is decided by (a) the exit code and (b) the program's FINAL
# "PASS:"/"FAIL:" line -- not by grepping the whole log for the word FAIL.

set -uo pipefail
cd "$(dirname "$0")" || exit 2

QUIET=0
[[ "${1:-}" == "-q" || "${1:-}" == "--quiet" ]] && QUIET=1

LOGDIR=last_run
mkdir -p "$LOGDIR"

PY=python3
command -v "$PY" >/dev/null 2>&1 || { echo "FATAL: python3 not found"; exit 2; }

echo "=============================================================="
echo " 100foom Q-7 -- exact tensor-decomposition verifier + Strassen"
echo " host: $(uname -srm)   cores: $(nproc)   python: $($PY -V 2>&1)"
echo "=============================================================="

overall=0
summary=""
total=0

run_one() {
    local name="$1"; shift
    local log="$LOGDIR/$name.log"
    local t0 t1 rc line status secs

    t0=$(date +%s.%N)
    "$@" >"$log" 2>&1
    rc=$?
    t1=$(date +%s.%N)
    secs=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')

    # the program's own final verdict line
    line=$(grep -E '^(PASS|FAIL):' "$log" | tail -n 1)
    [[ -z "$line" ]] && line="(no PASS/FAIL line emitted)"

    if [[ $rc -eq 0 && "$line" == PASS:* ]]; then
        status="PASS"
    else
        status="FAIL"
        overall=1
    fi

    total=$(awk -v a="$total" -v b="$secs" 'BEGIN{printf "%.2f", a+b}')
    printf '%-16s %-4s %6ss  %s\n' "$name" "$status" "$secs" "$line"

    if [[ $QUIET -eq 0 ]]; then
        # show the substantive output, minus the deliberately-failing reject lines
        grep -vE '^(PASS|FAIL):' "$log" | grep -vE '^\s*$' | sed 's/^/    /' || true
        echo
    fi
}

# Same idea, but the command is EXPECTED to fail (exit != 0).  Used for the
# negative control: a deliberately corrupted decomposition file must be rejected.
run_expect_fail() {
    local name="$1"; shift
    local log="$LOGDIR/$name.log"
    local t0 t1 rc status secs

    t0=$(date +%s.%N)
    "$@" >"$log" 2>&1
    rc=$?
    t1=$(date +%s.%N)
    secs=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')

    if [[ $rc -ne 0 ]] && grep -q '^\[file\] FAIL:' "$log"; then
        status="PASS"
        summary="rejected as required (exit $rc)"
    else
        status="FAIL"
        summary="corrupted input was NOT rejected (exit $rc)"
        overall=1
    fi
    total=$(awk -v a="$total" -v b="$secs" 'BEGIN{printf "%.2f", a+b}')
    printf '%-16s %-4s %6ss  %s\n' "$name" "$status" "$secs" "$summary"
    if [[ $QUIET -eq 0 ]]; then
        grep -vE '^\s*$' "$log" | sed 's/^/    /' || true
        echo
    fi
}

run_one "tensor_verify" "$PY" tensor_verify.py --selftest
run_one "strassen"      "$PY" strassen.py
run_one "omega"         "$PY" omega.py
run_one "json-accept"   "$PY" tensor_verify.py --check-file strassen_2x2x2.json
run_expect_fail "json-reject" "$PY" tensor_verify.py --check-file strassen_2x2x2_CORRUPT.json

echo "--------------------------------------------------------------"
echo "total wall time: ${total}s   (per-program logs: ./$LOGDIR/)"
echo "--------------------------------------------------------------"

if [[ $overall -eq 0 ]]; then
    echo "PASS: all components -- exact verifier, rank-7 Strassen, omega anchor (exit 0)"
else
    echo "FAIL: at least one component failed -- see the table and ./$LOGDIR/ (exit 1)"
fi
exit $overall
