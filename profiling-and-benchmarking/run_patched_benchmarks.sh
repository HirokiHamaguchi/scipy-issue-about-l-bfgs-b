#!/usr/bin/env bash
set -euo pipefail

if [[ ${CONDA_DEFAULT_ENV:-} != "base" ]]; then
    echo "Error: run from the Conda base environment." >&2
    exit 1
fi

here=$(cd -- "$(dirname -- "$0")" && pwd)
baseline="$here/../../scipy"
patched="$here/../../scipy-patched-ver1"

run_one() {
    local source=$1 env=$2 build=$3 output=$4

    [[ -d "$source/$build" ]] || {
        echo "Build directory not found: $source/$build" >&2
        exit 1
    }

    cd "$source"
    conda run --no-capture-output -n "$env" \
        spin python --build-dir="$build" --no-build -- \
        "$here/check_blas_backends.py" openblas

    LBFGSB_RESULTS_DIR="$output" \
        conda run --no-capture-output -n "$env" \
        spin python --build-dir="$build" --no-build -- \
        "$here/benchmark_lbfgsb.py"
}

run_one "$baseline" scipy-dev-openblas build-openblas \
    "$here/results/patched-comparison/baseline"
run_one "$patched" scipy-dev-patched-ver1 build-patched-ver1 \
    "$here/results/patched-comparison/patched-ver1"

uv run --project "$here/.." python "$here/plot_patched_benchmarks.py"

echo "Results written to $here/results/patched-comparison"
