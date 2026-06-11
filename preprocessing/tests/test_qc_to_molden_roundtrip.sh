#!/bin/bash
# test_qc_to_molden_roundtrip.sh
# ==============================
# End-to-end validation of the multi-format QC input support (future_plan
# item 2e).  Proves that converting a QC file through qc_to_molden.py
# (iodata) and feeding the canonical molden to preprocess_molden produces
# the SAME SCE physics as preprocessing the original molden directly.
#
# Covered:
#   1. molden -> qc_to_molden.py (iodata 'molden') -> canonical molden
#      -> preprocess.  Orbital norms + N_e must match direct-preprocess.
#   2. molden -> iodata fchk -> qc_to_molden.py (iodata 'fchk') -> molden
#      -> preprocess.  Exercises the Gaussian .fchk read path.
#   3. The C++ input-validation gate ABORTS (exit 3) on a normalisation-
#      corrupted molden.
#
# Requires:  qc-iodata  (pip install qc-iodata).  SKIPS cleanly (exit 0,
# prints SKIP) if iodata is not importable -- so it's safe to run anywhere.
#
# Usage:  bash preprocessing/tests/test_qc_to_molden_roundtrip.sh \
#               [path/to/preprocess_molden] [path/to/python]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PRE="${1:-$REPO/preprocessing/build/preprocess_molden}"
PY="${2:-python3}"
REF="$REPO/preprocessing/reference_data/h2o_ccpvdz_sph.molden"
CONV="$REPO/preprocessing/qc_to_molden.py"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fail() { echo "FAIL: $*"; exit 1; }

# ---- preflight ----
[ -x "$PRE" ] || fail "preprocess_molden not found/executable at '$PRE'"
[ -f "$REF" ] || fail "reference molden not found at '$REF'"
[ -f "$CONV" ] || fail "qc_to_molden.py not found at '$CONV'"

if ! "$PY" -c "import iodata" 2>/dev/null; then
    echo "SKIP: iodata not importable ('$PY -c import iodata' failed)."
    echo "      Install with: pip install qc-iodata   then re-run."
    exit 0
fi
echo "=== test_qc_to_molden_roundtrip ==="
echo "  preprocess: $PRE"
echo "  python    : $PY  (iodata $($PY -c 'import iodata; print(iodata.__version__)'))"

PP_FLAGS="--lmax 16 --dr 0.02 --rmax 12"

# Extract the two scalar invariants we compare: sum||psi||^2 and N_e(SCE rho).
# Both are printed on the [sanity] lines.
extract_metrics() {  # $1 = preprocess log
    local ne sum
    ne=$(grep "N_e (from SCE rho)" "$1"  | sed -E 's/.*= *([0-9.eE+-]+).*vs.*/\1/')
    sum=$(grep "sum ||psi_i||\^2"  "$1"  | sed -E 's/.*= *([0-9.eE+-]+).*vs.*/\1/')
    echo "$ne $sum"
}

# Numeric "within rel tol" comparison.
close() {  # $1 a  $2 b  $3 reltol
    "$PY" - "$1" "$2" "$3" <<'PYEOF'
import sys
a, b, tol = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
rel = abs(a - b) / max(abs(a), 1e-300)
sys.exit(0 if rel <= tol else 1)
PYEOF
}

# ---- baseline: preprocess the original molden directly ----
"$PRE" "$REF" "$TMP/orig.h5" $PP_FLAGS > "$TMP/orig.log" 2>&1 \
    || fail "baseline preprocess failed (see $TMP/orig.log)"
read -r NE0 SUM0 < <(extract_metrics "$TMP/orig.log")
echo "  baseline:           N_e=$NE0  sum||psi||^2=$SUM0"

# ---- (1) molden -> qc_to_molden (iodata molden) -> preprocess ----
"$PY" "$CONV" "$REF" "$TMP/canon.molden" --format molden > "$TMP/conv1.log" 2>&1 \
    || fail "qc_to_molden (molden) failed (see $TMP/conv1.log)"
"$PRE" "$TMP/canon.molden" "$TMP/canon.h5" $PP_FLAGS > "$TMP/canon.log" 2>&1 \
    || fail "preprocess of canonical molden failed (see $TMP/canon.log)"
read -r NE1 SUM1 < <(extract_metrics "$TMP/canon.log")
echo "  via iodata molden:  N_e=$NE1  sum||psi||^2=$SUM1"
close "$NE0"  "$NE1"  1e-6 || fail "(1) N_e mismatch: $NE0 vs $NE1"
close "$SUM0" "$SUM1" 1e-6 || fail "(1) sum||psi||^2 mismatch: $SUM0 vs $SUM1"
echo "  [ok] (1) molden->iodata->molden path matches baseline (<1e-6)"

# ---- (2) molden -> fchk -> qc_to_molden (iodata fchk) -> preprocess ----
"$PY" - "$REF" "$TMP/h2o.fchk" <<'PYEOF' || fail "iodata molden->fchk dump failed"
import sys, iodata
iodata.dump_one(iodata.load_one(sys.argv[1]), sys.argv[2], fmt="fchk")
PYEOF
"$PY" "$CONV" "$TMP/h2o.fchk" "$TMP/from_fchk.molden" --format fchk > "$TMP/conv2.log" 2>&1 \
    || fail "qc_to_molden (fchk) failed (see $TMP/conv2.log)"
"$PRE" "$TMP/from_fchk.molden" "$TMP/from_fchk.h5" $PP_FLAGS > "$TMP/fchk.log" 2>&1 \
    || fail "preprocess of fchk-derived molden failed (see $TMP/fchk.log)"
read -r NE2 SUM2 < <(extract_metrics "$TMP/fchk.log")
echo "  via iodata fchk:    N_e=$NE2  sum||psi||^2=$SUM2"
# fchk stores fewer significant digits -> looser tolerance (1e-5).
close "$NE0"  "$NE2"  1e-5 || fail "(2) N_e mismatch: $NE0 vs $NE2"
close "$SUM0" "$SUM2" 1e-5 || fail "(2) sum||psi||^2 mismatch: $SUM0 vs $SUM2"
echo "  [ok] (2) Gaussian fchk path matches baseline (<1e-5)"

# ---- (3) validation gate must ABORT on a normalisation-corrupted molden ----
"$PY" - "$REF" "$TMP/normbug.molden" <<'PYEOF' || fail "could not build corrupted molden"
import sys, iodata
mol = iodata.load_one(sys.argv[1])
c = mol.mo.coeffs.copy(); c[:, 1] *= 1.4   # scale occupied orbital 2 -> norm ~1.96
mol.mo.coeffs = c
iodata.dump_one(mol, sys.argv[2], fmt="molden")
PYEOF
# Capture the exit code directly (do NOT wrap in `if`, whose own status
# would mask the command's exit code).
"$PRE" "$TMP/normbug.molden" "$TMP/normbug.h5" $PP_FLAGS > "$TMP/normbug.log" 2>&1
rc=$?
[ "$rc" -ne 0 ] || fail "(3) preprocess of CORRUPTED molden returned 0 -- gate did not fire!"
[ "$rc" -eq 3 ] || fail "(3) expected exit 3 from validation gate, got $rc"
grep -q "INPUT VALIDATION FAILED" "$TMP/normbug.log" \
    || fail "(3) gate aborted but without the expected message"
echo "  [ok] (3) validation gate aborts (exit 3) on a normalisation bug"

echo "PASS  test_qc_to_molden_roundtrip"
exit 0
