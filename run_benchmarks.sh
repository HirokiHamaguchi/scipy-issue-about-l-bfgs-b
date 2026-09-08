#!/usr/bin/env bash
set -euo pipefail

# Check the conda environment.
CURRENT_ENV=$(basename "$CONDA_DEFAULT_ENV" 2>/dev/null)
if [ "$CURRENT_ENV" != "base" ]; then
    echo "Error: run from conda 'base'."
    exit 1
fi

# Run from either this directory or the sibling SciPy checkout.
here=$(cd -- "$(dirname -- "$0")" && pwd)
scipy=${SCIPY_DIR:-"$here/../scipy"}
mkdir -p "$here/results"
cd "$scipy"

run_one() {
    local label=$1 env=$2 build=$3

    # The two SciPy builds must already exist.
    [[ -d "$build" ]] || {
        echo "Build directory not found: $scipy/$build" >&2
        exit 1
    }

    # Confirm that this build loaded only the intended BLAS backend.
    conda run --no-capture-output -n "$env" \
        spin python --build-dir="$build" --no-build -- \
        "$here/check_blas_backends.py" "$label"

    # One BLAS thread is the primary comparison; the objective uses no BLAS.
    conda run --no-capture-output -n "$env" \
        spin python --build-dir="$build" --no-build -- \
        "$here/benchmark_lbfgsb.py" \
        >"$here/results/$label.json"
}

run_one openblas scipy-dev-openblas build-openblas
run_one mkl scipy-dev-mkl build-mkl

echo "Results written to $here/results"
